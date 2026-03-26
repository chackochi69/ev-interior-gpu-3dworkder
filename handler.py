#!/usr/bin/env python3
"""
RunPod Serverless handler for the Edict Veritas 3D reconstruction pipeline.

Input schema:
  {
    "zone_id": "uuid",
    "images": [
      {
        "signed_url": "https://...",        # signed GCS GET URL (2-hour TTL)
        "scan_point_id": "uuid",
        "sequence_order": 0,
        "capture_method": "phone_camera"
      }
    ],
    "glb_upload_url": "https://..."         # signed GCS PUT URL (2-hour TTL)
  }

Output on success:  {"success": true}
Output on failure:  {"error": "...message...", "traceback": "..."}

GPU is enabled automatically when CUDA is available (COLMAP_USE_GPU=1 is set
inside the handler before calling the pipeline).
"""

import os
import sys
import shutil
import tempfile
import traceback

print("[startup] handler.py loading...", flush=True)

try:
    import requests
    print(f"[startup] requests OK", flush=True)
except Exception as _e:
    print(f"[startup] requests import FAILED: {_e}", flush=True)
    sys.exit(1)

# Both handler.py and process_zone.py are copied to /app in the Docker image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from process_zone import run_colmap_pipeline, ply_to_glb, resize_image, MAX_DIMENSION, log  # noqa: E402
    print("[startup] process_zone imported OK", flush=True)
except Exception as _e:
    print(f"[startup] process_zone import FAILED: {_e}", flush=True)
    sys.exit(1)

try:
    import runpod  # noqa: E402  (installed via pip)
    print(f"[startup] runpod {runpod.__version__} imported OK", flush=True)
except Exception as _e:
    print(f"[startup] runpod import FAILED: {_e}", flush=True)
    sys.exit(1)


def _download_images(images: list[dict], img_dir: str) -> int:
    """Download + resize each image from its signed URL into img_dir.
    Returns the number of images successfully written."""
    count = 0
    for entry in images:
        signed_url = entry.get("signed_url", "")
        sp_id = str(entry.get("scan_point_id", "unknown"))[:8]
        seq = int(entry.get("sequence_order", 0))

        raw_path = os.path.join(img_dir, f"raw_{sp_id}_{seq}.jpg")
        dst_path = os.path.join(img_dir, f"sp{sp_id}_seq{seq:04d}.jpg")

        log(f"Downloading image sp={sp_id} seq={seq}")
        try:
            resp = requests.get(signed_url, timeout=120, stream=True)
            resp.raise_for_status()
            with open(raw_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)

            resize_image(raw_path, dst_path)
            os.remove(raw_path)
            count += 1
        except Exception as exc:
            log(f"Warning: failed to download/resize sp={sp_id} seq={seq}: {exc}")

    return count


def _upload_glb(glb_path: str, upload_url: str) -> None:
    """PUT the GLB file to the pre-signed storage URL."""
    size = os.path.getsize(glb_path)
    log(f"Uploading GLB ({size / 1024 / 1024:.1f} MB)...")
    with open(glb_path, "rb") as fh:
        resp = requests.put(
            upload_url,
            data=fh,
            headers={
                "Content-Type": "model/gltf-binary",
                "Content-Length": str(size),
            },
            timeout=300,
        )
    resp.raise_for_status()
    log("GLB upload complete")


def handler(job: dict) -> dict:
    """RunPod serverless entry point."""
    job_input = job.get("input", {})
    zone_id = job_input.get("zone_id", "unknown")
    images = job_input.get("images", [])
    glb_upload_url = job_input.get("glb_upload_url", "")

    if not images:
        return {"error": "No images provided in job input"}
    if not glb_upload_url:
        return {"error": "No glb_upload_url provided in job input"}

    # Enable GPU SIFT + dense stereo on RunPod pods
    os.environ["COLMAP_USE_GPU"] = "1"

    work_dir = tempfile.mkdtemp(prefix=f"ev-zone-{zone_id}-")
    try:
        img_dir = os.path.join(work_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        count = _download_images(images, img_dir)
        log(f"Zone {zone_id}: {count}/{len(images)} images ready")

        if count == 0:
            return {"error": "All image downloads failed — nothing to reconstruct"}

        output_glb = os.path.join(work_dir, "model.glb")

        ply_path = run_colmap_pipeline(work_dir, img_dir)
        ply_to_glb(ply_path, output_glb)

        if not os.path.exists(output_glb):
            return {"error": "Pipeline finished but no GLB was produced"}

        _upload_glb(output_glb, glb_upload_url)
        return {"success": True}

    except Exception as exc:
        tb = traceback.format_exc()
        log(f"Handler error: {exc}\n{tb}")
        return {"error": str(exc), "traceback": tb}

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
