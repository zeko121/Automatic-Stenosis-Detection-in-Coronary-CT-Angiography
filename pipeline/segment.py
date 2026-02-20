"""
Sliding window segmentation on preprocessed zarr data.
Supports MONAI UNet and custom AttentionUNet3D. Falls back to CPU on OOM.
"""

import json
import time
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr

from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

_cached_model = None
_cached_config = None
_cached_model_dir = None
_cached_device = None


class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced = max(in_channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, D, H, W = x.shape
        avg_pool = F.adaptive_avg_pool3d(x, 1).view(B, C)
        max_pool = F.adaptive_max_pool3d(x, 1).view(B, C)
        attention = self.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        return attention.view(B, C, 1, 1, 1)

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        return self.sigmoid(self.conv(pooled))


class ParallelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_att = ChannelAttention3D(in_channels, reduction)
        self.spatial_att = SpatialAttention3D(spatial_kernel)
        self.norm = nn.GroupNorm(min(32, in_channels), in_channels)

    def forward(self, x):
        channel_out = x * self.channel_att(x)
        spatial_out = x * self.spatial_att(x)
        return self.norm(channel_out + spatial_out)


class AttentionGate3D(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch=None):
        super().__init__()
        inter_ch = inter_ch or max(skip_ch // 2, 32)
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_ch, inter_ch, 1, bias=False),
            nn.BatchNorm3d(inter_ch)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(skip_ch, inter_ch, 1, bias=False),
            nn.BatchNorm3d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_ch, 1, 1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g_proj = self.W_g(g)
        x_proj = self.W_x(x)
        if g_proj.shape[2:] != x_proj.shape[2:]:
            g_proj = F.interpolate(g_proj, size=x_proj.shape[2:], mode='trilinear', align_corners=False)
        alpha = self.psi(self.relu(g_proj + x_proj))
        return x * alpha

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.residual = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.residual(x)


class EncoderBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, use_attention=True, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock3D(in_ch, out_ch, dropout)
        self.attention = ParallelAttention3D(out_ch) if use_attention else None
        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x):
        features = self.conv(x)
        if self.attention is not None:
            features = features + self.attention(features)
        pooled = self.pool(features)
        return pooled, features


class DecoderBlock3D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_att_gate=True, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, 2, stride=2)
        self.att_gate = AttentionGate3D(in_ch, skip_ch) if use_att_gate else None
        self.conv = ConvBlock3D(in_ch + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        if self.att_gate is not None:
            skip = self.att_gate(g=x, x=skip)
        return self.conv(torch.cat([x, skip], dim=1))


class AttentionUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels=(32, 64, 128, 256, 512),
        use_attention=True,
        use_attention_gates=True,
        dropout=0.1
    ):
        super().__init__()
        self.use_attention = use_attention
        self.use_attention_gates = use_attention_gates
        self.n_levels = len(channels)

        self.encoders = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(zip([in_channels] + list(channels[:-1]), channels)):
            use_att = use_attention and (i > 0)
            drop = dropout if i > 1 else 0
            self.encoders.append(EncoderBlock3D(in_ch, out_ch, use_att, drop))

        self.bottleneck = nn.Sequential(
            ConvBlock3D(channels[-1], channels[-1], dropout),
            ParallelAttention3D(channels[-1]) if use_attention else nn.Identity()
        )

        self.decoders = nn.ModuleList()
        dec_channels = list(reversed(channels[:-1]))
        for i, (dec_ch, skip_ch) in enumerate(zip(dec_channels, dec_channels)):
            in_ch = channels[-1] if i == 0 else dec_channels[i-1]
            drop = dropout if i < 2 else 0
            self.decoders.append(DecoderBlock3D(in_ch, skip_ch, dec_ch, use_attention_gates, drop))

        self.output = nn.Conv3d(channels[0], out_channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skips = []
        for encoder in self.encoders[:-1]:
            x, skip = encoder(x)
            skips.append(skip)
        x, _ = self.encoders[-1](x)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        return self.output(x)


@dataclass
class ModelConfig:
    model_type: str = "UNet"
    in_channels: int = 1
    out_channels: int = 1
    channels: Tuple[int, ...] = (32, 64, 128, 256, 512)
    strides: Tuple[int, ...] = (2, 2, 2, 2)
    num_res_units: int = 2
    dropout: float = 0.1
    norm: str = "batch"
    use_attention: bool = True
    use_attention_gates: bool = True

    @classmethod
    def from_config_file(cls, config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        training_config = config_data.get("training_config", {})

        return cls(
            model_type=training_config.get("model_type", "UNet"),
            in_channels=training_config.get("in_channels", 1),
            out_channels=training_config.get("out_channels", 1),
            channels=tuple(training_config.get("channels", [32, 64, 128, 256, 512])),
            strides=tuple(training_config.get("strides", [2, 2, 2, 2])),
            num_res_units=training_config.get("num_res_units", 2),
            dropout=training_config.get("dropout", 0.1),
            norm=training_config.get("norm", "batch"),
            use_attention=training_config.get("use_attention", True),
            use_attention_gates=training_config.get("use_attention_gates", True),
        )


def create_model(config):
    if config.model_type == "UNet":
        return UNet(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=config.channels,
            strides=config.strides,
            num_res_units=config.num_res_units,
            dropout=config.dropout,
            norm=config.norm
        )
    elif config.model_type == "AttentionUNet3D":
        return AttentionUNet3D(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            channels=config.channels,
            use_attention=config.use_attention,
            use_attention_gates=config.use_attention_gates,
            dropout=config.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

def load_model(checkpoint_path, config_path, device):
    config = ModelConfig.from_config_file(config_path)
    model = create_model(config)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    return model, config


def get_or_load_model(model_dir, device=None):
    """Load model, reusing cache if same dir."""
    global _cached_model, _cached_config, _cached_model_dir, _cached_device

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_str = str(device)

    if (_cached_model is not None
            and _cached_model_dir == str(model_dir)
            and _cached_device == device_str):
        return _cached_model, _cached_config

    model_dir_path = Path(model_dir)
    checkpoint_path = model_dir_path / "checkpoints" / "best_model.pth"
    config_path = model_dir_path / "run_config.json"

    model, config = load_model(str(checkpoint_path), str(config_path), device)

    _cached_model = model
    _cached_config = config
    _cached_model_dir = str(model_dir)
    _cached_device = device_str

    return model, config


def run_inference(model, image, device, patch_size=(96, 96, 96), overlap=0.5,
                  use_amp=True, verbose=True):
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)

    if verbose:
        print(f"  tensor shape: {image_tensor.shape}")
        print(f"  patch: {patch_size}, overlap: {overlap}")

    with torch.no_grad():
        if use_amp and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                output = sliding_window_inference(
                    image_tensor,
                    roi_size=patch_size,
                    sw_batch_size=4,
                    predictor=model,
                    overlap=overlap,
                    mode='gaussian'
                )
        else:
            output = sliding_window_inference(
                image_tensor,
                roi_size=patch_size,
                sw_batch_size=4,
                predictor=model,
                overlap=overlap,
                mode='gaussian'
            )

    output = torch.sigmoid(output)
    # print("output range:", output.min().item(), output.max().item())
    probs = output.squeeze().cpu().numpy().astype(np.float16)

    return probs


def process(input_path, output_path, model_dir, patch_size=(160, 160, 160),
            overlap=0.5, threshold=0.5, verbose=True, checkpoint_name="best_model.pth"):
    t0 = time.time()
    input_path = Path(input_path)
    output_path = Path(output_path)
    model_dir = Path(model_dir)

    result = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "model_dir": str(model_dir),
        "status": "pending",
    }

    if not model_dir.exists():
        result["status"] = "error"
        result["error"] = f"Model directory not found: {model_dir}"
        return result

    if not input_path.exists():
        result["status"] = "error"
        result["error"] = f"Input path does not exist: {input_path}"
        return result

    checkpoint_path = model_dir / "checkpoints" / checkpoint_name
    config_path = model_dir / "run_config.json"

    if not checkpoint_path.exists():
        result["status"] = "error"
        result["error"] = f"Checkpoint not found: {checkpoint_path}"
        return result

    if not config_path.exists():
        result["status"] = "error"
        result["error"] = f"Config not found: {config_path}"
        return result

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if verbose:
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        gpu_name = None
        gpu_mem = None
        if verbose:
            print("no GPU, using CPU")

    result["device"] = str(device)
    result["gpu_name"] = gpu_name

    try:
        if verbose:
            print(f"loading model from {model_dir.name}...")

        model, model_config = get_or_load_model(str(model_dir), device)
        # print(f"model device: {next(model.parameters()).device}")

        if verbose:
            print(f"  type: {model_config.model_type}, channels: {model_config.channels}")
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  params: {n_params:,}")

        result["model_type"] = model_config.model_type
        result["model_params"] = sum(p.numel() for p in model.parameters())

        if verbose:
            print(f"reading {input_path.name}...")

        store = zarr.open_group(str(input_path), mode='r')
        image = store['image'][:]

        if verbose:
            print(f"  shape: {image.shape}, range: [{image.min():.2f}, {image.max():.2f}]")

        result["input_shape"] = list(image.shape)  # store for the final report

        if verbose:
            print(f"running inference (patch={patch_size})...")

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        try:
            probs = run_inference(
                model, image, device,
                patch_size=patch_size,
                overlap=overlap,
                use_amp=True,
                verbose=verbose
            )
        except torch.cuda.OutOfMemoryError:
            if verbose:
                print("  OOM on GPU, falling back to CPU...")
            torch.cuda.empty_cache()
            gc.collect()

            device = torch.device("cpu")
            model = model.to(device)
            result["device"] = "cpu (fallback)"

            probs = run_inference(
                model, image, device,
                patch_size=patch_size,
                overlap=overlap,
                use_amp=False,
                verbose=verbose
            )

        mask = (probs > threshold).astype(np.uint8)

        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6
            result["peak_gpu_memory_mb"] = round(peak_memory_mb, 1)
            if verbose:
                print(f"  peak GPU mem: {peak_memory_mb:.0f} MB")

        vessel_voxels = int(mask.sum())
        if verbose:
            print(f"  vessel voxels: {vessel_voxels:,}")

        result["vessel_voxels"] = vessel_voxels

        if verbose:
            print(f"saving to {output_path.name}...")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        import shutil
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(input_path, output_path)

        out_store = zarr.open_group(str(output_path), mode='r+')

        ZARR_V3 = zarr.__version__.startswith("3")
        if ZARR_V3:
            from zarr.codecs import Blosc
            compressor = [Blosc(cname="zstd", clevel=3, shuffle=0)]
            out_store.create_array(
                "mask",
                shape=mask.shape,
                dtype="uint8",
                chunks=(160, 160, 160),
                compressors=compressor
            )[:] = mask
            out_store.create_array(
                "probs",
                shape=probs.shape,
                dtype="float16",
                chunks=(160, 160, 160),
                compressors=compressor
            )[:] = probs
        else:
            from numcodecs import Blosc
            compressor = Blosc(cname="zstd", clevel=3, shuffle=0)
            out_store.create_dataset(
                "mask",
                data=mask,
                dtype="uint8",
                chunks=(160, 160, 160),
                compressor=compressor
            )
            out_store.create_dataset(
                "probs",
                data=probs,
                dtype="float16",
                chunks=(160, 160, 160),
                compressor=compressor
            )

        out_store.attrs.update({
            "segmentation_model": model_config.model_type,
            "segmentation_model_dir": str(model_dir.name),
            "segmentation_patch_size": list(patch_size),
            "segmentation_overlap": overlap,
            "segmentation_threshold": threshold,
            "vessel_voxels": vessel_voxels,
        })

        size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024 * 1024)

        if verbose:
            print(f"  saved ({size_mb:.1f} MB)")

        elapsed = time.time() - t0

        result.update({
            "status": "success",
            "output_shape": list(mask.shape),
            "file_size_mb": round(size_mb, 2),
            "runtime_sec": round(elapsed, 2),
        })

        if verbose:
            print(f"done in {elapsed:.1f}s")

        del image, mask, probs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        if verbose:
            print(f"error: {result['error']}")
            import traceback
            traceback.print_exc()

    return result


if __name__ == "__main__":
    import sys

    test_input = "temp/case-30.zarr"
    test_output = "temp/case-30_segmented.zarr"
    model_dir = "models/2025-12-31_02-53-53"

    if len(sys.argv) >= 4:
        test_input = sys.argv[1]
        test_output = sys.argv[2]
        model_dir = sys.argv[3]

    print("--- segmentation test ---")
    result = process(
        test_input,
        test_output,
        model_dir,
        patch_size=(160, 160, 160),
        overlap=0.5,
        verbose=True
    )
    print()
    for key, value in result.items():
        print(f"  {key}: {value}")
