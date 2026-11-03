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




import av
import numpy as np
import time
from IPython.display import display, Image as IPImage
import io
from PIL import Image


def get_frame(video_path: str, target_frame_num: int) -> np.ndarray:
    """
    Fetch a specific frame from a video file with low latency.

    Strategy (PyAV / FFmpeg):
      1. Open the container once.
      2. Compute the target PTS in stream time-base units.
      3. Seek to the nearest *preceding keyframe* — FFmpeg jumps there
         directly, skipping the entire prefix of the file.
      4. Decode only the handful of frames between that keyframe and the
         target — typically << 30 frames for modern encodings.

    Returns
    -------
    np.ndarray  shape (H, W, 3), dtype uint8, BGR colour order (OpenCV-compatible).
    """
    container = av.open(video_path)
    stream = container.streams.video[0]

    # Enable multi-threaded decoding for speed
    stream.thread_type = "AUTO"

    fps       = float(stream.average_rate)          # e.g. 30.0
    time_base = float(stream.time_base)             # e.g. 1/90000

    # Target presentation timestamp in stream time-base units
    target_sec       = target_frame_num / fps
    target_timestamp = int(target_sec / time_base)

    # Jump to nearest preceding keyframe (no 'whence' kwarg in this PyAV version)
    container.seek(target_timestamp, backward=True, stream=stream)

    img = None
    for frame in container.decode(video=0):
        # Compute frame index from PTS
        pts_sec       = frame.pts * time_base
        current_frame = int(round(pts_sec * fps))

        if current_frame >= target_frame_num:
            # Convert directly to a numpy BGR array (no PIL round-trip)
            img = frame.to_ndarray(format="bgr24")
            break

    container.close()

    if img is None:
        raise ValueError(
            f"Frame {target_frame_num} not found in '{video_path}'. "
            f"Total frames ≈ {int(fps * float(stream.duration) * time_base) if stream.duration else '?'}"
        )
    return img


path_to_botsort_parent = './'
if path_to_botsort_parent not in sys.path:
    sys.path.append(path_to_botsort_parent)

ROOT_FRAME_DIR = "/home/lakshh/workspace/reid/ds_backend_reid/MCDPT/deepstream_npy_output_videocutreid"
# ROOT_FRAME_DIR = "/home/lakshh/workspace/reid/ds_backend_reid/MCDPT/deepstream_npy_output"

from botsort.bot_sort import BoTSORT
from botsort.global_registry import GlobalRegistry

registry = GlobalRegistry(
    match_threshold=0.25,
    min_frames=5,
    emb_dim=256,
)

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
)

tracker1 = BoTSORT(**_tracker_kwargs)  # camera 0

CAM1_TID_OFFSET = 0

cur_frame     = 0
ACTIVE_FORMAT = "tlwh"
VIDEO_PATH = "/home/lakshh/Videos/video_cut_reid.mp4"


for i in range(3000):
    cur_frame += 1

    npy_path = f"{ROOT_FRAME_DIR}/batch_frame_{i}.npy"
    if not os.path.exists(npy_path):
        print(f"File not found: {npy_path}")
        continue

    frame_content = np.load(npy_path, allow_pickle=True)


    detections1 = frame_content[0]['objects']
    for d in detections1:
        d['obj_meta'] = None

    tracker1.update(detections1)

    # registry.step(tracker1, frame_id=cur_frame)

    extracted_data1 = []
    for t in tracker1.tracked_stracks:
        if t.t_global_id != 0 and hasattr(t, 'tlwh'):
            extracted_data1.append((t.tlwh, t.track_id))

    # cam1_gids = [t.t_global_id for t in tracker1.tracked_stracks]
    # print(f"frame {i:04d}  cam1_gids={cam1_gids}   registry={registry}")

    # bg_frame1 = get_frame(VIDEO_PATH, frame_content[0]['frame_id'])
    bg_frame1 = np.zeros((1080, 1920, 3), dtype=np.uint8)

    if extracted_data1:
        formatted = convert_boxes(extracted_data1, to_format=ACTIVE_FORMAT)
        bg_frame1  = draw_boxes_on_bg(bg_frame1, formatted, current_format=ACTIVE_FORMAT)

    
    bg_frame1 = cv2.resize(bg_frame1, (0, 0), fx=1/2, fy=1/2)
    cv2.imshow("Detections - 1", bg_frame1)
    cv2.waitKey(0)

cv2.destroyAllWindows()

print("\n=== Final Gallery ===")
for e in registry.get_all_entries():
    print(e)
