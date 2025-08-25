import csv
from pathlib import Path
from typing import Optional, List

import numpy as np
from mongoengine import DoesNotExist

from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.data.model import Reconstruction
from wormlab3d.data.model.midline3d import Midline3D, M3D_SOURCE_WT3D

def export_wt3d_endpoints_csv_from_reconstruction(
    rec_id: str,
    out_csv: Optional[Path] = None,
    source_file: Optional[str] = None,
    regenerate_2d: bool = False,
):
    """
    From a Reconstruction id:
      - read trial id, start_frame, end_frame
      - for each frame in [start_frame, end_frame], fetch WT3D Midline3D
      - project to 2D and output head/tail endpoints for ALL cameras (0,1,2)
    """
    rec: Reconstruction = Reconstruction.objects.get(id=rec_id)
    trial = rec.trial

    start = rec.start_frame
    end = rec.end_frame
    if start is None or end is None:
        raise ValueError("Reconstruction is missing start_frame or end_frame")

    if out_csv is None:
        out_csv = LOGS_PATH / f"{START_TIMESTAMP}_wt3d_endpoints_rec={rec.id}_trial={trial.id}.csv"
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    written, missing = 0, 0

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "trial_num", "frame_position", "frame_id", "x_head", "y_head", "x_tail", "y_tail"])

        for index, frame_num in enumerate(range(start, end + 1)):
            frame = trial.get_frame(frame_num)

            qs = Midline3D.objects(frame=frame.id, source=M3D_SOURCE_WT3D)
            if source_file:
                qs = qs.filter(source_file=source_file)
            m = qs.order_by("source_file", "id").first()

            if not m:
                missing += 1
                logger.info(f"[WT3D] No midline at frame={frame_num} (source_file={source_file!r}). Skipping frame.")
                continue

            # Get 2D projections for all 3 cameras
            triplet_2d: List[np.ndarray] = m.get_prepared_2d_coordinates(regenerate=regenerate_2d)

            for cam_idx, xy in enumerate(triplet_2d):
                if xy.size == 0:
                    logger.warning(f"[WT3D] Empty 2D coords for midline={m.id} cam={cam_idx} frame={frame_num}. Skipping cam.")
                    continue

                head = xy[0]
                tail = xy[-1]

                writer.writerow([
                    int((index*3) + cam_idx + 1),
                    int(trial.id),
                    int(frame_num),
                    int(cam_idx),
                    float(head[0]),
                    float(head[1]),
                    float(tail[0]),
                    float(tail[1]),
                ])
                written += 1

    logger.info(f"Wrote {written} rows to {out_csv} (skipped {missing} frames without WT3D midlines).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export WT3D 2D endpoints (head/tail) to CSV for a Reconstruction id.")
    parser.add_argument("--reconstruction", required=True, default="68ab201e69cf388a16fb5379")
    parser.add_argument("--out", type=str, default="/tmp/wt3d_endpoints.csv")
    parser.add_argument("--source-file", type=str, default=None)
    parser.add_argument("--regenerate-2d", action="store_true")

    args = parser.parse_args()
    export_wt3d_endpoints_csv_from_reconstruction(
        rec_id=args.reconstruction,
        out_csv=Path(args.out) if args.out else None,
        source_file=args.source_file,
        regenerate_2d=args.regenerate_2d,
    )
