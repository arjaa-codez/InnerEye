from pathlib import Path
from uuid import uuid4
from enum import Enum
import os
import numpy as np
import SimpleITK as sitk
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from innereye.preprocess.pipeline import preprocess
from innereye.model.inference import run_inference, save_mask
from innereye.mesh.mesh import mask_to_mesh, save_mesh

app = FastAPI(title="InnerEye Localized", version="0.1.0")

# CORS middleware - allow frontend on different ports (e.g., 3000, 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    # Validate DICOM directory before processing
    dicom_path = Path(payload.dicom_dir)
    if not dicom_path.exists():
        record.status = ScanStatus.failed
        del SCAN_DB[scan_id]  # Remove failed record
        return JSONResponse(
            status_code=400,
            content={"error": "Directory not found", "path": payload.dicom_dir, "resolved": str(dicom_path.absolute())}
        )
    if not dicom_path.is_dir():
        record.status = ScanStatus.failed
        del SCAN_DB[scan_id]  # Remove failed record
        return JSONResponse(
            status_code=400,
            content={"error": "Path is not a directory", "path": payload.dicom_dir}
        )

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
        return JSONResponse(
            status_code=404,
            content={"error": f"Scan ID {scan_id} does not exist"}
        )
    return _to_response(SCAN_DB[scan_id])


@app.get("/scans/{scan_id}/mask")
async def get_mask(scan_id: str):
    if scan_id not in SCAN_DB:
        return JSONResponse(
            status_code=404,
            content={"error": f"Scan ID {scan_id} does not exist"}
        )
    rec = SCAN_DB[scan_id]
    if not rec.mask_path:
        return JSONResponse(
            status_code=400,
            content={"error": f"Mask for Scan ID {scan_id} is not ready yet"}
        )
    return {"path": rec.mask_path}


@app.get("/scans/{scan_id}/mesh")
async def get_mesh(scan_id: str):
    if scan_id not in SCAN_DB:
        return JSONResponse(
            status_code=404,
            content={"error": f"Scan ID {scan_id} does not exist"}
        )
    rec = SCAN_DB[scan_id]
    if not rec.mesh_path:
        return JSONResponse(
            status_code=400,
            content={"error": f"Mesh for Scan ID {scan_id} is not ready yet"}
        )
    return {"path": rec.mesh_path}


@app.post("/scans/{scan_id}/review")
async def review(scan_id: str, accepted: bool, notes: str | None = None):
    if scan_id not in SCAN_DB:
        return JSONResponse(
            status_code=404,
            content={"error": f"Scan ID {scan_id} does not exist"}
        )
    rec = SCAN_DB[scan_id]
    rec.status = ScanStatus.ready if accepted else ScanStatus.needs_review
    SCAN_DB[scan_id] = rec
    return {"ok": True, "notes": notes}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
# Root & Discovery Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint with API overview and available routes."""
    return {
        "service": "InnerEye Localized",
        "version": "0.1.0",
        "endpoints": {
            "GET /": "This overview",
            "GET /health": "Health check",
            "GET /scans": "List all scans",
            "POST /scans": "Create new scan (body: {dicom_dir, window_center?, window_width?, iso_spacing?, checkpoint?, device?})",
            "GET /scans/{scan_id}/status": "Get scan status",
            "GET /scans/{scan_id}/mask": "Get mask file path",
            "GET /scans/{scan_id}/mesh": "Get mesh file path",
            "POST /scans/{scan_id}/review": "Submit review (query: accepted, notes?)",
            "GET /analyze/{scan_id}": "Alias for /scans/{scan_id}/status",
            "POST /diagnostics/check-path": "Check if a file path exists",
        },
    }


@app.get("/scans")
async def list_scans():
    """List all scans in the in-memory database."""
    return {
        "count": len(SCAN_DB),
        "scans": [_to_response(rec) for rec in SCAN_DB.values()],
    }


# Alias for /scans/{scan_id}/status in case frontend uses /analyze/{id}
@app.get("/analyze/{scan_id}", response_model=ScanStatusResponse)
async def analyze_scan(scan_id: str):
    """Alias for GET /scans/{scan_id}/status."""
    if scan_id not in SCAN_DB:
        return JSONResponse(
            status_code=404,
            content={"error": f"Scan ID {scan_id} does not exist"}
        )
    return _to_response(SCAN_DB[scan_id])


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

class PathCheckRequest(BaseModel):
    path: str


@app.post("/diagnostics/check-path")
async def check_path(payload: PathCheckRequest):
    """Check if a file or directory exists and return diagnostic info."""
    p = Path(payload.path)
    exists = p.exists()
    is_file = p.is_file() if exists else False
    is_dir = p.is_dir() if exists else False
    parent_exists = p.parent.exists()
    return {
        "path": payload.path,
        "exists": exists,
        "is_file": is_file,
        "is_dir": is_dir,
        "parent_exists": parent_exists,
        "absolute_path": str(p.resolve()) if exists else str(p.absolute()),
    }


def diagnose_dicom_path(dicom_dir: str) -> dict:
    """Utility to diagnose DICOM directory before processing."""
    p = Path(dicom_dir)
    result = {
        "path": dicom_dir,
        "exists": p.exists(),
        "is_dir": p.is_dir() if p.exists() else False,
        "files": [],
    }
    if p.exists() and p.is_dir():
        files = list(p.glob("*"))
        result["file_count"] = len(files)
        result["files"] = [f.name for f in files[:10]]  # First 10 files
        result["has_dcm"] = any(f.suffix.lower() == ".dcm" for f in files)
    return result
