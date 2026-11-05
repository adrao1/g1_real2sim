"""Export FoundationPose OBJ meshes from the simulator's YCB visual USDs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdShade

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "banana": "banana/banana.usd",
    "cracker_box": "cracker_box/Axis_Aligned_Physics/003_cracker_box.usd",
    "mustard_bottle": "mustard_bottle/Axis_Aligned_Physics/006_mustard_bottle.usd",
}


def _visual_mesh_and_texture(stage: Usd.Stage) -> tuple[UsdGeom.Mesh, Path]:
    candidates = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        uv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        if not uv or uv.GetInterpolation() != UsdGeom.Tokens.vertex:
            continue
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if not material:
            continue
        for child in Usd.PrimRange(material.GetPrim()):
            if child.GetTypeName() != "Shader":
                continue
            texture = child.GetAttribute("inputs:diffuse_texture")
            if texture and texture.Get() and texture.Get().resolvedPath:
                candidates.append((UsdGeom.Mesh(prim), Path(texture.Get().resolvedPath)))
    if len(candidates) != 1:
        raise ValueError(f"Expected one textured visual mesh, found {len(candidates)}")
    return candidates[0]


def export(name: str) -> None:
    source = ROOT / "assets/ycb" / ASSETS[name]
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError(f"Could not open USD: {source}")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError(f"Expected a Z-up stage in meters: {source}")

    mesh, texture = _visual_mesh_and_texture(stage)
    prim = mesh.GetPrim()
    root = stage.GetDefaultPrim()
    xforms = UsdGeom.XformCache()
    mesh_to_root = xforms.GetLocalToWorldTransform(prim) * xforms.GetLocalToWorldTransform(root).GetInverse()
    points = [mesh_to_root.Transform(Gf.Vec3d(*point)) for point in mesh.GetPointsAttr().Get()]
    uv_primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    uvs = uv_primvar.ComputeFlattened()
    if len(uvs) != len(points):
        raise ValueError(f"Vertex UV count does not match the visual mesh: {source}")

    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if sum(counts) != len(indices):
        raise ValueError(f"Face indices do not match the visual mesh: {source}")

    output = ROOT / "assets/ycb" / name / "tracking"
    output.mkdir(parents=True, exist_ok=True)
    obj = output / "textured.obj"
    with obj.open("w") as file:
        file.write("mtllib textured.mtl\no tracked_item\nusemtl textured\n")
        for point in points:
            file.write(f"v {point[0]:.10g} {point[1]:.10g} {point[2]:.10g}\n")
        for uv in uvs:
            file.write(f"vt {uv[0]:.10g} {uv[1]:.10g}\n")
        offset = 0
        for count in counts:
            face = indices[offset : offset + count]
            for i in range(1, count - 1):
                triangle = (face[0], face[i], face[i + 1])
                file.write("f " + " ".join(f"{vertex + 1}/{vertex + 1}" for vertex in triangle) + "\n")
            offset += count
    (output / "textured.mtl").write_text("newmtl textured\nKd 1 1 1\nmap_Kd textured.png\n")
    shutil.copyfile(texture, output / "textured.png")

    loaded = trimesh.load(obj, force="mesh")
    expected_points = np.asarray(points, dtype=float)
    expected_bounds = np.stack((expected_points.min(axis=0), expected_points.max(axis=0)))
    if loaded.visual.kind != "texture" or loaded.visual.material.image is None:
        raise ValueError(f"OBJ texture did not load: {obj}")
    if loaded.visual.material.image.size != Image.open(texture).size:
        raise ValueError(f"OBJ loaded the wrong texture: {obj}")
    if not np.allclose(loaded.bounds, expected_bounds, atol=1e-6):
        raise ValueError(f"OBJ bounds differ from the USD mesh: {obj}")
    print(f"{name}: {len(loaded.vertices)} vertices, {len(loaded.faces)} triangles -> {obj}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("items", nargs="*", choices=tuple(ASSETS))
    args = parser.parse_args()
    for name in args.items or ASSETS:
        export(name)


if __name__ == "__main__":
    main()
