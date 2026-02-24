from enum import Enum
from dataclasses import dataclass


class Status(str, Enum):
    uploaded = "uploaded"
    preprocessed = "preprocessed"
    inference = "inference"
    meshing = "meshing"
    ready = "ready"
    needs_review = "needs_review"
    failed = "failed"


@dataclass
class ScanState:
    scan_id: str
    status: Status
    dice: float | None = None
    mask_uri: str | None = None
    mesh_uri: str | None = None


def on_upload(scan_id: str):
    return ScanState(scan_id=scan_id, status=Status.uploaded)


def after_preprocess(state: ScanState):
    state.status = Status.preprocessed
    return state


def after_infer(state: ScanState, dice: float | None, mask_uri: str):
    state.dice = dice
    state.mask_uri = mask_uri
    if dice is None or dice < 0.8:
        state.status = Status.needs_review
    else:
        state.status = Status.meshing
    return state


def after_mesh(state: ScanState, mesh_uri: str):
    state.mesh_uri = mesh_uri
    state.status = Status.ready
    return state
