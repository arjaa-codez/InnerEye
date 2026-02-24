from pathlib import Path
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
from .unet3d import UNet3D


def load_nifti(path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # z, y, x
    spacing = img.GetSpacing()[::-1]  # to match array order
    return arr.astype("float32"), spacing


def sliding_window_inference(model: torch.nn.Module, volume: np.ndarray, patch=96, overlap=16, device="cpu"):
    model.eval()
    with torch.no_grad():
        z, y, x = volume.shape
        stride = patch - overlap
        out_accum = torch.zeros((1, model.out.out_channels, z, y, x), device=device)
        weight_accum = torch.zeros_like(out_accum)
        vol_t = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(device)
        for zz in range(0, max(z - patch + 1, 1), stride):
            for yy in range(0, max(y - patch + 1, 1), stride):
                for xx in range(0, max(x - patch + 1, 1), stride):
                    patch_t = vol_t[:, :, zz:zz+patch, yy:yy+patch, xx:xx+patch]
                    logits = model(patch_t)
                    out_accum[:, :, zz:zz+patch, yy:yy+patch, xx:xx+patch] += logits
                    weight_accum[:, :, zz:zz+patch, yy:yy+patch, xx:xx+patch] += 1
        out_accum = out_accum / torch.clamp_min(weight_accum, 1e-3)
        return out_accum


def run_inference(nifti_path: str, checkpoint: str | None = None, device: str = "cpu") -> np.ndarray:
    volume, _ = load_nifti(nifti_path)
    model = UNet3D(in_ch=1, n_classes=4)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    logits = sliding_window_inference(model, volume, patch=96, overlap=24, device=device)
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype("uint8")
    return pred


def save_mask(mask: np.ndarray, ref_nifti: str, out_path: str):
    ref = sitk.ReadImage(ref_nifti)
    out_img = sitk.GetImageFromArray(mask.astype("uint8"))
    out_img.SetSpacing(ref.GetSpacing())
    out_img.SetDirection(ref.GetDirection())
    out_img.SetOrigin(ref.GetOrigin())
    sitk.WriteImage(out_img, out_path, useCompression=True)


if __name__ == "__main__":
    pred = run_inference("data/processed/scan.nii.gz")
    save_mask(pred, "data/processed/scan.nii.gz", "data/processed/mask.nii.gz")
