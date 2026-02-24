import os
import SimpleITK as sitk


def load_dicom_series(dicom_dir: str) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(files)
    return reader.Execute()


def hu_window(image: sitk.Image, wl: int = 40, ww: int = 400) -> sitk.Image:
    min_hu = wl - ww // 2
    max_hu = wl + ww // 2
    clamped = sitk.Clamp(image, lowerBound=min_hu, upperBound=max_hu)
    return sitk.RescaleIntensity(clamped, 0.0, 1.0)


def resample_isotropic(image: sitk.Image, spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    return resampler.Execute(image)


def save_nifti(image: sitk.Image, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sitk.WriteImage(image, out_path, useCompression=True)


def preprocess(dicom_dir: str, out_path: str, wl=40, ww=400, iso=1.0):
    img = load_dicom_series(dicom_dir)
    img = hu_window(img, wl, ww)
    img = resample_isotropic(img, (iso, iso, iso))
    save_nifti(img, out_path)


if __name__ == "__main__":
    preprocess("data/dicom", "data/processed/scan.nii.gz")
