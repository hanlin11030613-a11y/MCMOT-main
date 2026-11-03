import os
import sys
import cv2
import numpy as np
from glob import glob

# --- Visualization Functions ---
def convert_boxes(box_data, to_format="tlbr"):
    """
    Converts a list of tuples (box, track_id) from 'tlwh' to 'tlbr' or ensures 'tlwh'.
    Input boxes are assumed to be in tlwh format: [x, y, w, h]
    """
    if not len(box_data):
        return []

    converted = []
    for box, track_id in box_data:
        x, y, w, h = box
        if to_format == "tlbr":
            converted.append(([x, y, x + w, y + h], track_id))
        elif to_format == "tlwh":
            converted.append(([x, y, w, h], track_id))
    return converted

def draw_boxes_on_bg(bg_image, box_data, current_format="tlwh"):
    box_color  = (255, 255, 255)
    text_color = (255, 255, 255)
    thickness  = 2

    for box, track_id in box_data:
        if current_format == "tlwh":
            x, y, w, h = map(int, box)
            pt1 = (x, y)
            pt2 = (x + w, y + h)
        elif current_format == "tlbr":
            x1, y1, x2, y2 = map(int, box)
            pt1 = (x1, y1)
            pt2 = (x2, y2)

        cv2.rectangle(bg_image, pt1, pt2, box_color, thickness)

        text   = f"GID: {track_id}"
        text_y = pt1[1] - 7 if pt1[1] - 7 > 15 else pt1[1] + 15
        cv2.putText(bg_image, text, (pt1[0], text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thickness)

    return bg_image

# --- Path setup ---
path_to_botsort_parent = './'
if path_to_botsort_parent not in sys.path:
    sys.path.append(path_to_botsort_parent)

ROOT_FRAME_DIR = "/home/lakshh/workspace/reid/ds_backend_reid/MCDPT/deepstream_npy_output2"
# ROOT_FRAME_DIR = "/home/lakshh/workspace/reid/ds_backend_reid/MCDPT/deepstream_npy_output"

from botsort.bot_sort import BoTSORT
from botsort.global_registry import GlobalRegistry
from botsort.multi_camera_tracker import MultiCameraTracker

# --- Shared registry ---
registry = GlobalRegistry(
    match_threshold=0.25,
    min_frames=5,
    emb_dim=256,
)

# --- Static 2-camera tracker setup ---
_tracker_kwargs = dict(
    track_high_thresh=0.6,
    track_low_thresh=0.1,
    new_track_thresh=0.7,
    track_buffer=600,
    match_thresh=0.8,
    with_reid=True,
    proximity_thresh=0.7,
    appearance_thresh=0.25,
    euc_thresh=0.1,
    fuse_score=True,
    frame_rate=30,
    max_batch_size=8,
    map_len=None,
    real_data=True,
    registry=registry,
)

# cam_source must differ per camera - BoTSORT seeds its track_id space from
# it (cam_source * 1000), so this is what keeps tracker1/tracker2's
# track_ids disjoint in the shared registry. Previously both defaulted to
# the same cam_source, a real (if latent, since registry.step() was never
# actually called correctly here) track_id collision.
tracker1 = BoTSORT(cam_source=1, **_tracker_kwargs)  # camera 0
tracker2 = BoTSORT(cam_source=2, **_tracker_kwargs)  # camera 1

mct = MultiCameraTracker(registry=registry)
mct.add_camera(0, tracker1)
mct.add_camera(1, tracker2)

# --- Main Processing Loop ---
# --- Main Processing Loop ---
cur_frame     = 0
ACTIVE_FORMAT = "tlwh"

for i in range(3000):
    cur_frame += 1

    npy_path = f"{ROOT_FRAME_DIR}/batch_frame_{i}.npy"
    if not os.path.exists(npy_path):
        print(f"File not found: {npy_path}")
        continue

    frame_content = np.load(npy_path, allow_pickle=True)

    if len(frame_content) < 2:
        print(f"Frame {i}: only {len(frame_content)} camera(s), skipping")
        continue

    # ── Camera 0 detections ───────────────────────────────────────────────────
    detections1 = frame_content[0]['objects']
    for d in detections1:
        d['obj_meta'] = None

    # ── Camera 1 detections ───────────────────────────────────────────────────
    detections2 = frame_content[1]['objects']
    for d in detections2:
        d['obj_meta'] = None

    # ── 1+2. Per-camera tracking + global registry step (assigns/reuses
    # t_global_id on each track) — both cameras share one frame_id since
    # detections for both arrive together each tick in this offline replay.
    mct.update_batch({0: detections1, 1: detections2}, frame_id=cur_frame)

    # ── 3. Collect tracks from both cameras for display ───────────────────────
    extracted_data1 = []
    extracted_data2 = []
    for t in tracker1.tracked_stracks:
        if t.t_global_id != 0 and hasattr(t, 'tlwh'):
            extracted_data1.append((t.tlwh, t.t_global_id))
    for t in tracker2.tracked_stracks:
        if t.t_global_id != 0 and hasattr(t, 'tlwh'):
            extracted_data2.append((t.tlwh, t.t_global_id))

    # ── 4. Print ──────────────────────────────────────────────────────────────
    cam1_gids = [t.t_global_id for t in tracker1.tracked_stracks]
    cam2_gids = [t.t_global_id for t in tracker2.tracked_stracks]
    print(f"frame {i:04d}  cam1_gids={cam1_gids}  cam2_gids={cam2_gids}  registry={registry}")

    # ── 5. Visualise ──────────────────────────────────────────────────────────
    bg_frame1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bg_frame2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    if extracted_data1:
        formatted = convert_boxes(extracted_data1, to_format=ACTIVE_FORMAT)
        bg_frame1  = draw_boxes_on_bg(bg_frame1, formatted, current_format=ACTIVE_FORMAT)

    if extracted_data2:
        formatted = convert_boxes(extracted_data2, to_format=ACTIVE_FORMAT)
        bg_frame2  = draw_boxes_on_bg(bg_frame2, formatted, current_format=ACTIVE_FORMAT)

    
    bg_frame1 = cv2.resize(bg_frame1, (0, 0), fx=1/2, fy=1/2)
    bg_frame2 = cv2.resize(bg_frame2, (0, 0), fx=1/2, fy=1/2)


    combined = np.hstack((bg_frame1, bg_frame2))
    cv2.imshow("Detections - 1", combined)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# ── Final registry dump ───────────────────────────────────────────────────────
print("\n=== Final Gallery ===")
for e in registry.get_all_entries():
    print(e)
