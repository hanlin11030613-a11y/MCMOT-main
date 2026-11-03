import os
import sys
import cv2
import numpy as np

# --- 引入路徑與模組 ---
path_to_botsort_parent = './'
if path_to_botsort_parent not in sys.path:
    sys.path.append(path_to_botsort_parent)

from evacuation_detector import EvacuationDetector
from botsort.bot_sort import BoTSORT
from botsort.global_registry import GlobalRegistry
from botsort.multi_camera_tracker import MultiCameraTracker

ROOT_FRAME_DIR = "/home/lakshh/workspace/reid/ds_backend_reid/MCDPT/deepstream_npy_output2"

# ==========================================
# 1. 建立各相機專屬之單應性矩陣 (Homography)
# ==========================================
# 梯形投影標定點: [遠端左, 遠端右, 近端右, 近端左] -> 映射至地平面矩形區域
src_cam0 = np.float32([[650, 200], [1270, 200], [1750, 1000], [170, 1000]])
src_cam1 = np.float32([[650, 200], [1270, 200], [1750, 1000], [170, 1000]])
dst_ground = np.float32([[100, 100], [500, 100], [500, 700], [100, 700]])

H_CAM0 = cv2.getPerspectiveTransform(src_cam0, dst_ground)
H_CAM1 = cv2.getPerspectiveTransform(src_cam1, dst_ground)

# ==========================================
# 2. 視覺化輔助函式
# ==========================================
def convert_boxes(box_data, to_format="tlwh"):
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
    box_color = (255, 255, 255)
    text_color = (255, 255, 255)
    thickness = 2

    for box, track_id in box_data:
        if current_format == "tlwh":
            x, y, w, h = map(int, box)
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            foot_pt = (int(x + w / 2.0), int(y + h))
        elif current_format == "tlbr":
            x1, y1, x2, y2 = map(int, box)
            pt1 = (x1, y1)
            pt2 = (x2, y2)
            foot_pt = (int((x1 + x2) / 2.0), int(y2))

        cv2.rectangle(bg_image, pt1, pt2, box_color, thickness)
        # 標出腳底接地點 (綠色圓點)
        cv2.circle(bg_image, foot_pt, 4, (0, 255, 0), -1)

        text = f"GID: {track_id}"
        text_y = pt1[1] - 7 if pt1[1] - 7 > 15 else pt1[1] + 15
        cv2.putText(bg_image, text, (pt1[0], text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thickness)

    return bg_image


def draw_alert_overlay(frame, cam_name, status):
    """
    依據逃散狀態繪製雙模態警報 HUD (同向箭頭或放射狀事發中心)
    """
    if status["alert"]:
        # 繪製醒目紅色警告橫幅
        cv2.rectangle(frame, (20, 20), (950, 125), (0, 0, 200), -1)
        cv2.rectangle(frame, (20, 20), (950, 125), (0, 0, 255), 3)

        evac_label = "PARALLEL FLEEING" if status["evac_type"] == "PARALLEL" else "RADIAL DISPERSION"
        alert_text = f"[ALERT] {cam_name} EVACUATION ({evac_label})!"
        info_text = (f"Runners: {status['active_runners']} | "
                     f"Phi_par: {status['phi_parallel']} | "
                     f"Phi_rad: {status['phi_radial']}")

        cv2.putText(frame, alert_text, (35, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2)
        cv2.putText(frame, info_text, (35, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 255), 2)

        # 模式 A: 繪製群體平均移動方向指示箭頭
        if status["evac_type"] == "PARALLEL":
            rad = np.radians(status["mean_direction_deg"])
            cx, cy = 880, 70
            arrow_len = 35
            end_x = int(cx + arrow_len * np.cos(rad))
            end_y = int(cy + arrow_len * np.sin(rad))
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (0, 255, 255), 3, tipLength=0.3)

        # 模式 B: 在畫面上標示推估的事發中心點
        elif status["evac_type"] == "RADIAL" and status["img_origin"] is not None:
            ox, oy = int(status["img_origin"][0]), int(status["img_origin"][1])
            cv2.circle(frame, (ox, oy), 35, (0, 255, 255), 3)
            cv2.circle(frame, (ox, oy), 12, (0, 0, 255), -1)
            cv2.putText(frame, "INCIDENT ORIGIN", (ox - 80, oy - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        # 常態綠色資訊列
        cv2.rectangle(frame, (20, 20), (700, 75), (0, 100, 0), -1)
        info_text = f"{cam_name} | Runners: {status['active_runners']} | Par: {status['phi_parallel']} | Rad: {status['phi_radial']}"
        cv2.putText(frame, info_text, (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


# ==========================================
# 3. 系統初始化
# ==========================================
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
    registry=registry,
)

tracker1 = BoTSORT(cam_source=1, **_tracker_kwargs)  # Camera 0
tracker2 = BoTSORT(cam_source=2, **_tracker_kwargs)  # Camera 1

mct = MultiCameraTracker(registry=registry)
mct.add_camera(0, tracker1)
mct.add_camera(1, tracker2)

# 為 Camera 0 與 Camera 1 分別注入單應性矩陣
detector_cam0 = EvacuationDetector(
    homography_matrix=H_CAM0,
    min_crowd_size=3,
    speed_thresh=3.5,
    parallel_thresh=0.75,
    radial_thresh=0.65,
    temporal_window=12
)
detector_cam1 = EvacuationDetector(
    homography_matrix=H_CAM1,
    min_crowd_size=3,
    speed_thresh=3.5,
    parallel_thresh=0.75,
    radial_thresh=0.65,
    temporal_window=12
)

# ==========================================
# 4. 主處理迴圈
# ==========================================
cur_frame = 0
ACTIVE_FORMAT = "tlwh"

for i in range(3000):
    cur_frame += 1

    npy_path = f"{ROOT_FRAME_DIR}/batch_frame_{i}.npy"
    if not os.path.exists(npy_path):
        continue

    frame_content = np.load(npy_path, allow_pickle=True)
    if len(frame_content) < 2:
        continue

    detections1 = frame_content[0]['objects']
    for d in detections1:
        d['obj_meta'] = None

    detections2 = frame_content[1]['objects']
    for d in detections2:
        d['obj_meta'] = None

    # 1. 執行多鏡頭追蹤關聯
    mct.update_batch({0: detections1, 1: detections2}, frame_id=cur_frame)

    # 2. 執行雙模態 BEV 逃散偵測
    status_cam0 = detector_cam0.update(tracker1.tracked_stracks, cur_frame)
    status_cam1 = detector_cam1.update(tracker2.tracked_stracks, cur_frame)

    if status_cam0["alert"]:
        print(f"[ALARM Frame {i:04d} Cam0] 逃散觸發 ({status_cam0['evac_type']})! "
              f"Runners: {status_cam0['active_runners']}, "
              f"Phi_par: {status_cam0['phi_parallel']}, Phi_rad: {status_cam0['phi_radial']}")
    if status_cam1["alert"]:
        print(f"[ALARM Frame {i:04d} Cam1] 逃散觸發 ({status_cam1['evac_type']})! "
              f"Runners: {status_cam1['active_runners']}, "
              f"Phi_par: {status_cam1['phi_parallel']}, Phi_rad: {status_cam1['phi_radial']}")

    # 3. 收集軌跡框
    extracted_data1 = []
    extracted_data2 = []
    for t in tracker1.tracked_stracks:
        if t.t_global_id != 0 and hasattr(t, 'tlwh'):
            extracted_data1.append((t.tlwh, t.t_global_id))
    for t in tracker2.tracked_stracks:
        if t.t_global_id != 0 and hasattr(t, 'tlwh'):
            extracted_data2.append((t.tlwh, t.t_global_id))

    # 4. 繪製可視化輸出
    bg_frame1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bg_frame2 = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # 繪製地平面透視梯形校正參考線
    cv2.polylines(bg_frame1, [src_cam0.astype(np.int32)], True, (50, 50, 50), 2)
    cv2.polylines(bg_frame2, [src_cam1.astype(np.int32)], True, (50, 50, 50), 2)

    if extracted_data1:
        formatted = convert_boxes(extracted_data1, to_format=ACTIVE_FORMAT)
        bg_frame1 = draw_boxes_on_bg(bg_frame1, formatted, current_format=ACTIVE_FORMAT)

    if extracted_data2:
        formatted = convert_boxes(extracted_data2, to_format=ACTIVE_FORMAT)
        bg_frame2 = draw_boxes_on_bg(bg_frame2, formatted, current_format=ACTIVE_FORMAT)

    # 繪製 HUD
    draw_alert_overlay(bg_frame1, "Camera 0", status_cam0)
    draw_alert_overlay(bg_frame2, "Camera 1", status_cam1)

    # 水平拼接並顯示 (約 30 FPS 連續播放)
    bg1_res = cv2.resize(bg_frame1, (0, 0), fx=0.5, fy=0.5)
    bg2_res = cv2.resize(bg_frame2, (0, 0), fx=0.5, fy=0.5)
    combined = np.hstack((bg1_res, bg2_res))

    cv2.imshow("MCMOT BEV Dual-Mode Evacuation Test", combined)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()