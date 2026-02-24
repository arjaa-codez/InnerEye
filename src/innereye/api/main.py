from pathlib import Path
from uuid import uuid4
from enum import Enum
import numpy as np
import SimpleITK as sitk
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from innereye.preprocess.pipeline import preprocess
from innereye.model.inference import run_inference, save_mask
from innereye.mesh.mesh import mask_to_mesh, save_mesh

app = FastAPI(title="InnerEye Localized", version="0.1.0")


DATA_ROOT = Path("data")
PROCESSED_ROOT = DATA_ROOT / "processed"


class ScanStatus(str, Enum):
    uploaded = "uploaded"
    preprocessed = "preprocessed"
    inference = "inference"
    meshing = "meshing"
    ready = "ready"
    needs_review = "needs_review"
    failed = "failed"


class ScanCreateRequest(BaseModel):
    dicom_dir: str
    window_center: int = 40
    window_width: int = 400
    iso_spacing: float = 1.0
    checkpoint: str | None = None
    device: str = "cpu"


class ScanStatusResponse(BaseModel):
    scan_id: str
    status: ScanStatus
    dice_score: float | None = None
    mask_path: str | None = None
    mesh_path: str | None = None


class ScanRecord(BaseModel):
    scan_id: str
    status: ScanStatus
    dice_score: float | None = None
    nifti_path: str | None = None
    mask_path: str | None = None
    mesh_path: str | None = None


# In-memory store for demo purposes only; replace with DB.
SCAN_DB: dict[str, ScanRecord] = {}


def _ensure_dirs():
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)


def _to_response(rec: ScanRecord) -> ScanStatusResponse:
    return ScanStatusResponse(
        scan_id=rec.scan_id,
        status=rec.status,
        dice_score=rec.dice_score,
        mask_path=rec.mask_path,
        mesh_path=rec.mesh_path,
    )


@app.post("/scans", response_model=ScanStatusResponse)
async def create_scan(payload: ScanCreateRequest):
    _ensure_dirs()
    scan_id = str(uuid4())
    record = ScanRecord(scan_id=scan_id, status=ScanStatus.uploaded)
    SCAN_DB[scan_id] = record
    out_dir = PROCESSED_ROOT / scan_id
    out_dir.mkdir(parents=True, exist_ok=True)
    nifti_path = out_dir / "scan.nii.gz"
    mask_path = out_dir / "mask.nii.gz"
    mesh_path = out_dir / "mesh.stl"
    try:
        preprocess(
            payload.dicom_dir,
            str(nifti_path),
            wl=payload.window_center,
            ww=payload.window_width,
            iso=payload.iso_spacing,
        )
        record.status = ScanStatus.preprocessed
        record.nifti_path = str(nifti_path)

        record.status = ScanStatus.inference
        pred = run_inference(str(nifti_path), checkpoint=payload.checkpoint, device=payload.device)
        save_mask(pred, str(nifti_path), str(mask_path))
        record.mask_path = str(mask_path)

        record.status = ScanStatus.meshing
        spacing = sitk.ReadImage(str(nifti_path)).GetSpacing()[::-1]
        mesh_obj = mask_to_mesh(pred.astype(np.uint8), spacing=spacing)
        save_mesh(mesh_obj, str(mesh_path))
        record.mesh_path = str(mesh_path)

        record.status = ScanStatus.ready
    except Exception as exc:  # noqa: BLE001
        record.status = ScanStatus.failed
        raise HTTPException(status_code=500, detail=f"pipeline failed: {exc}") from exc

    return _to_response(record)


@app.get("/scans/{scan_id}/status", response_model=ScanStatusResponse)
async def get_status(scan_id: str):
    if scan_id not in SCAN_DB:
        raise HTTPException(status_code=404, detail="scan not found")
    return _to_response(SCAN_DB[scan_id])


@app.get("/scans/{scan_id}/mask")
async def get_mask(scan_id: str):
    if scan_id not in SCAN_DB:
        raise HTTPException(status_code=404, detail="scan not found")
    rec = SCAN_DB[scan_id]
    if not rec.mask_path:
        raise HTTPException(status_code=400, detail="mask not ready")
    return {"path": rec.mask_path}


@app.get("/scans/{scan_id}/mesh")
async def get_mesh(scan_id: str):
    if scan_id not in SCAN_DB:
        raise HTTPException(status_code=404, detail="scan not found")
    rec = SCAN_DB[scan_id]
    if not rec.mesh_path:
        raise HTTPException(status_code=400, detail="mesh not ready")
    return {"path": rec.mesh_path}


@app.post("/scans/{scan_id}/review")
async def review(scan_id: str, accepted: bool, notes: str | None = None):
    if scan_id not in SCAN_DB:
        raise HTTPException(status_code=404, detail="scan not found")
    rec = SCAN_DB[scan_id]
    rec.status = ScanStatus.ready if accepted else ScanStatus.needs_review
    SCAN_DB[scan_id] = rec
    return {"ok": True, "notes": notes}


@app.get("/health")
async def health():
    return {"status": "ok"}
