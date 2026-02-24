import numpy as np
from skimage import measure
import trimesh


def mask_to_mesh(mask: np.ndarray, spacing=(1.0, 1.0, 1.0), level=0.5, smooth_iters=5):
    verts, faces, normals, _ = measure.marching_cubes(mask, level=level, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
    if smooth_iters > 0:
        trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iters, lamb=0.5)
    return mesh


def save_mesh(mesh: trimesh.Trimesh, path: str):
    mesh.export(path)


if __name__ == "__main__":
    arr = np.load("data/processed/mask.npy")
    mesh = mask_to_mesh(arr, spacing=(1.0, 1.0, 1.0))
    save_mesh(mesh, "data/processed/organ_mesh.stl")
