#!/usr/bin/env python3
"""
Edict Veritas — 3D Reconstruction Pipeline
==========================================
Inputs (all via CLI args parsed from JSON):
  --zone-id       UUID of the active zone
  --work-dir      Scratch directory for this job (e.g. /tmp/ev-zone-<uuid>)
  --images-json   Path to a JSON file describing each image:
                    [{"path": "/abs/path/img.jpg",
                      "scan_point_id": "uuid",
                      "capture_method": "iphone_lidar",
                      "sequence_order": 1,
                      "position_x": 0.0, "position_y": 0.0}]
  --output-glb    Output path for the final GLB file

Pipeline:
  1. Organise images into pycolmap-expected layout
  2. Run feature extraction (SIFT via pycolmap)
  3. Run sequential matcher (exploits ordered scan sequence)
  4. Sparse reconstruction (incremental mapper)
  5. Dense reconstruction via COLMAP's patch-match stereo + Poisson fusion
  6. Export the dense point cloud / mesh to PLY
  7. Load PLY in trimesh → clean → export GLB
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--zone-id", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--images-json", required=True)
    p.add_argument("--output-glb", required=True)
    return p.parse_args()


MAX_DIMENSION = 1500  # px — keeps phone images manageable for CPU SIFT


def resize_image(src: str, dst: str, max_dim: int = MAX_DIMENSION) -> bool:
    """Downscale image so its longest side ≤ max_dim. Saves to dst as JPEG.
    Returns True on success, False if the source file can't be read."""
    try:
        from PIL import Image, ExifTags
        img = Image.open(src)

        # Auto-rotate based on EXIF orientation
        try:
            exif = img._getexif()  # type: ignore[attr-defined]
            if exif:
                orient_key = next(
                    (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
                )
                if orient_key and orient_key in exif:
                    orientation = exif[orient_key]
                    rotations = {3: 180, 6: 270, 8: 90}
                    if orientation in rotations:
                        img = img.rotate(rotations[orientation], expand=True)
        except Exception:
            pass

        # Convert to RGB (handles HEIC/PNG with alpha)
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        img.save(dst, "JPEG", quality=92)
        return True
    except Exception as exc:
        log(f"Warning: could not resize {src} ({exc}), copying as-is")
        shutil.copy2(src, dst)
        return True


def setup_workspace(work_dir: str, images: list[dict]) -> str:
    """Resize and copy images into <work_dir>/images/. Returns that dir."""
    img_dir = os.path.join(work_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    for entry in images:
        src = entry["path"]
        sp_id = entry["scan_point_id"][:8]
        seq = entry["sequence_order"]
        dst = os.path.join(img_dir, f"sp{sp_id}_seq{seq:04d}.jpg")
        if os.path.exists(src):
            resize_image(src, dst)
        else:
            log(f"Warning: image not found at {src}, skipping")

    return img_dir


def run_colmap_pipeline(work_dir: str, img_dir: str) -> str:
    """Run COLMAP via pycolmap Python API. Returns path to dense PLY."""
    import pycolmap

    db_path = os.path.join(work_dir, "colmap.db")
    sparse_dir = os.path.join(work_dir, "sparse")
    dense_dir = os.path.join(work_dir, "dense")
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(dense_dir, exist_ok=True)

    log("Extracting features...")
    _reader_options = pycolmap.ImageReaderOptions()
    _reader_options.camera_model = "SIMPLE_RADIAL"
    _extraction_options = pycolmap.FeatureExtractionOptions()
    # Limit threads so SIFT does not exhaust RAM on large batches.
    # Images are already pre-resized to MAX_DIMENSION by setup_workspace.
    use_gpu = os.environ.get("COLMAP_USE_GPU", "0") == "1"
    _extraction_options.use_gpu = use_gpu
    if not use_gpu:
        # CPU mode: cap threads so SIFT does not exhaust RAM on large batches
        _extraction_options.num_threads = 2
    _extraction_options.max_image_size = MAX_DIMENSION
    try:
        # pycolmap >= 3.10: nested sift options
        _extraction_options.sift.max_num_features = 4096
        if not use_gpu:
            _extraction_options.sift.first_octave = 0  # no 2× upsample → saves 4× CPU RAM
    except AttributeError:
        pass  # older build: defaults are fine
    pycolmap.extract_features(
        database_path=db_path,
        image_path=img_dir,
        reader_options=_reader_options,
        extraction_options=_extraction_options,
    )

    log("Matching features (exhaustive)...")
    # Exhaustive matching is more reliable than sequential for small image sets.
    # pycolmap's matching wheel has no CUDA support compiled in, so force CPU.
    pycolmap.match_exhaustive(
        database_path=db_path,
        device=pycolmap.Device.cpu,
    )
    log("Running sparse reconstruction...")
    _mapping_opts = pycolmap.IncrementalPipelineOptions()
    # Default min_model_size=10 rejects any reconstruction with fewer images.
    # Default min_num_matches=15 is too strict for close-range scans.
    # Lower both so small scans (≥3 images) can still produce a model.
    _mapping_opts.min_model_size = 3
    _mapping_opts.min_num_matches = 10
    # Relax mapper thresholds for small scenes (defaults designed for 100+ images).
    _mapping_opts.mapper.init_min_num_inliers = 20   # default 100
    _mapping_opts.mapper.abs_pose_min_num_inliers = 10  # default 30
    maps = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=img_dir,
        output_path=sparse_dir,
        options=_mapping_opts,
    )
    if not maps:
        raise RuntimeError(
            "COLMAP sparse reconstruction produced no models. "
            "Ensure images have significant overlap (>50%) and cover the same scene."
        )

    log(f"Sparse reconstruction done ({len(maps)} model(s))")

    # Select the largest model (by image count)
    best_model_idx = max(maps.keys(), key=lambda k: len(maps[k].images))
    best_model_dir = os.path.join(sparse_dir, str(best_model_idx))
    maps[best_model_idx].write(best_model_dir)

   # Dense reconstruction is slow (5-30 min). Disabled by default.
    # Set COLMAP_DENSE=1 as an env var on the RunPod endpoint to enable it.
    enable_dense = os.environ.get("COLMAP_DENSE", "0") == "1"
    if enable_dense:
        ply_path = _try_dense_reconstruction(
            work_dir=work_dir,
            sparse_model_dir=best_model_dir,
            img_dir=img_dir,
            dense_dir=dense_dir,
        )
        if ply_path:
            return ply_path
        log("Dense reconstruction failed, falling back to sparse point cloud")
    else:
        log("Dense reconstruction disabled (set COLMAP_DENSE=1 to enable)")
    # Export sparse point cloud as PLY
    sparse_ply = os.path.join(work_dir, "sparse_points.ply")
    _export_sparse_points(maps[best_model_idx], sparse_ply)
    return sparse_ply


def _try_dense_reconstruction(
    work_dir: str,
    sparse_model_dir: str,
    img_dir: str,
    dense_dir: str,
) -> str | None:
    """
    Attempt dense reconstruction via colmap CLI (patch-match stereo + fusion).
    Returns path to fused.ply on success, None if colmap binary unavailable.
    """
    colmap_bin = shutil.which("colmap")
    if not colmap_bin:
        log("colmap binary not on PATH, skipping dense reconstruction")
        return None

    try:
        log("Undistorting images for dense reconstruction...")
        subprocess.run(
            [colmap_bin, "image_undistorter",
             "--image_path", img_dir,
             "--input_path", sparse_model_dir,
             "--output_path", dense_dir,
             "--output_type", "COLMAP"],
            check=True, capture_output=True, timeout=600,
        )

        log("Running patch-match stereo...")
        subprocess.run(
            [colmap_bin, "patch_match_stereo",
             "--workspace_path", dense_dir,
             "--workspace_format", "COLMAP",
             "--PatchMatchStereo.geom_consistency", "true"],
            check=True, capture_output=True, timeout=1800,
        )

        log("Fusing depth maps...")
        fused_ply = os.path.join(dense_dir, "fused.ply")
        subprocess.run(
            [colmap_bin, "stereo_fusion",
             "--workspace_path", dense_dir,
             "--workspace_format", "COLMAP",
             "--input_type", "geometric",
             "--output_path", fused_ply],
            check=True, capture_output=True, timeout=900,
        )

        if os.path.exists(fused_ply):
            log("Dense fusion complete")
            return fused_ply
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        log(f"Dense reconstruction failed: {e}")

    return None


def _export_sparse_points(reconstruction, ply_path: str) -> None:
    """Write sparse points3D from a pycolmap Reconstruction to a PLY file."""
    points = reconstruction.points3D
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt in points.values():
            r, g, b = (int(c) for c in pt.color[:3])
            f.write(f"{pt.xyz[0]:.6f} {pt.xyz[1]:.6f} {pt.xyz[2]:.6f} {r} {g} {b}\n")


def ply_to_glb(ply_path: str, output_glb: str) -> None:
    """Convert PLY (point cloud or mesh) to GLB via trimesh."""
    import trimesh
    import numpy as np

    log(f"Loading PLY: {ply_path}")
    scene_or_mesh = trimesh.load(ply_path)

    if isinstance(scene_or_mesh, trimesh.PointCloud):
        pc = scene_or_mesh
        log(f"Point cloud: {len(pc.vertices)} points")
        if len(pc.vertices) < 1:
            raise RuntimeError("Empty point cloud — reconstruction produced no 3D points")
        # Export as GLTF POINTS primitive — Three.js renders this as colored dots
        pc.export(output_glb, file_type="glb")
    elif isinstance(scene_or_mesh, trimesh.Scene):
        log("Loaded scene — merging geometries")
        mesh = trimesh.util.concatenate(list(scene_or_mesh.geometry.values()))
        _export_mesh(mesh, output_glb)
    else:
        _export_mesh(scene_or_mesh, output_glb)

    size_mb = os.path.getsize(output_glb) / (1024 * 1024)
    log(f"GLB exported ({size_mb:.1f} MB)")


def _export_mesh(mesh, output_glb: str) -> None:
    """Clean up and export a trimesh Mesh to GLB."""
    import trimesh
    log(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    if hasattr(mesh, 'is_watertight') and not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
    MAX_FACES = 500_000
    if len(mesh.faces) > MAX_FACES:
        target = MAX_FACES / len(mesh.faces)
        try:
            mesh = mesh.simplify_quadric_decimation(percent=target)
            log(f"After decimation: {len(mesh.faces)} faces")
        except Exception as e:
            log(f"Decimation warning: {e} — continuing with full mesh")
    log(f"Exporting mesh GLB to: {output_glb}")
    mesh.export(output_glb, file_type="glb")


def main() -> None:
    args = parse_args()

    with open(args.images_json) as f:
        images: list[dict] = json.load(f)

    log(f"Zone: {args.zone_id}, images: {len(images)}")

    if not images:
        print("ERROR: no images provided", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.work_dir, exist_ok=True)

    try:
        img_dir = setup_workspace(args.work_dir, images)
        ply_path = run_colmap_pipeline(args.work_dir, img_dir)
        ply_to_glb(ply_path, args.output_glb)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    log("Pipeline complete")


if __name__ == "__main__":
    main()
