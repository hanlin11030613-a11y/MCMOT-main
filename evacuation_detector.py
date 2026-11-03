import numpy as np
from collections import deque


class EvacuationDetector:
    """
    人群集體逃散 / 恐慌偵測模組 (支援單應性矩陣 BEV 投影與雙模態判定)

    雙模態機制:
      1. 平行同向逃散 (Parallel Evacuation): 計算地平面運動方向極化率 Phi_parallel
      2. 放射狀四散逃離 (Radial Dispersion): 計算地平面徑向背離度 Phi_radial，並估算事發幾何中心
    """
    def __init__(
        self,
        homography_matrix: np.ndarray = None,  # 3x3 影像到地面的透視變換矩陣 (若為 None 則回退為像素座標)
        min_crowd_size: int = 3,              # 判定群體奔跑的最少人數門檻
        speed_thresh: float = 3.5,            # 速度門檻 (若有 H 則為 BEV 地面單位/幀；若無則為像素/幀)
        parallel_thresh: float = 0.75,        # 同向逃散極化率閾值 (0.0 ~ 1.0)
        radial_thresh: float = 0.65,          # 放射發散背離度閾值 (0.0 ~ 1.0)
        temporal_window: int = 12,            # 時序滑動視窗幀數
        trigger_ratio: float = 0.6,           # 視窗內觸發比例門檻
    ):
        self.H = homography_matrix
        self.H_inv = np.linalg.inv(homography_matrix) if homography_matrix is not None else None
        self.min_crowd_size = min_crowd_size
        self.speed_thresh = speed_thresh
        self.parallel_thresh = parallel_thresh
        self.radial_thresh = radial_thresh
        self.temporal_window = temporal_window
        self.trigger_ratio = trigger_ratio

        # 歷史軌跡紀錄: {track_id: deque([(frame_id, gx, gy)], maxlen=5)}
        self._history = {}
        self._alert_window = deque(maxlen=temporal_window)

    def _get_ground_point(self, track):
        """
        取得地平面接觸點。
        優先取雙腳中心點 (x + w/2, y + h)，消除人體俯視視差。
        """
        if hasattr(track, 'foot_point'):
            foot = track.foot_point
        elif hasattr(track, 'tlwh'):
            x, y, w, h = track.tlwh
            foot = np.array([x + w / 2.0, y + h], dtype=np.float32)
        elif hasattr(track, 'centroid'):
            foot = np.asarray(track.centroid, dtype=np.float32)
        else:
            return np.array([0.0, 0.0], dtype=np.float32)

        # 若有提供單應性矩陣，將影像座標投影至 BEV 平面
        if self.H is not None:
            pt = np.array([foot[0], foot[1], 1.0], dtype=np.float32)
            world_pt = np.dot(self.H, pt)
            w = world_pt[2] if abs(world_pt[2]) > 1e-6 else 1e-6
            return np.array([world_pt[0] / w, world_pt[1] / w], dtype=np.float32)

        return foot

    def _get_velocity(self, track, frame_id: int, ground_pos: np.ndarray):
        """
        計算該目標在地平面的速度向量 (vx, vy)。
        若使用 BEV 投影，則基於地表座標差分；
        若無投影且 track 內部有 Kalman 濾波速度，則優先讀取 mean[4:6]。
        """
        tid = track.track_id
        if tid not in self._history:
            self._history[tid] = deque(maxlen=5)

        self._history[tid].append((frame_id, ground_pos[0], ground_pos[1]))

        # 若歷史累積達 2 幀以上，採差分計算速度 (最符合 BEV 空間物理位移)
        if len(self._history[tid]) >= 2:
            prev_f, px, py = self._history[tid][0]
            dt = max(1, frame_id - prev_f)
            return (ground_pos[0] - px) / dt, (ground_pos[1] - py) / dt

        # 若無歷史且無 H 矩陣，嘗試讀取 Kalman Filter 的內部速度
        if self.H is None and track.mean is not None and len(track.mean) >= 6:
            vx, vy = track.mean[4], track.mean[5]
            if not (np.isnan(vx) or np.isnan(vy)):
                return float(vx), float(vy)

        return 0.0, 0.0

    def clean_inactive_tracks(self, active_tids: set):
        """清理已離開鏡頭之目標的歷史資料"""
        disappeared = set(self._history.keys()) - active_tids
        for tid in disappeared:
            del self._history[tid]

    def update(self, tracks, frame_id: int):
        """
        傳入當前鏡頭活躍的 tracks 清單 (如 BoTSORT 的 tracked_stracks)。
        """
        active_tids = {t.track_id for t in tracks}
        self.clean_inactive_tracks(active_tids)

        fast_tracks = []
        fast_positions = []
        fast_vectors = []
        speeds = []

        # 1. 篩選高速移動者
        for t in tracks:
            g_pos = self._get_ground_point(t)
            vx, vy = self._get_velocity(t, frame_id, g_pos)
            speed = np.hypot(vx, vy)

            if speed >= self.speed_thresh and not getattr(t, 'is_touching_edge', False):
                fast_tracks.append(t)
                fast_positions.append(g_pos)
                fast_vectors.append(np.array([vx, vy]))
                speeds.append(speed)

        num_runners = len(fast_tracks)
        instant_trigger = False
        evac_type = "NONE"
        phi_parallel = 0.0
        phi_radial = 0.0
        mean_dir_deg = 0.0
        bev_origin_center = None
        img_origin_center = None

        # 2. 雙模態逃散檢定
        if num_runners >= self.min_crowd_size:
            unit_vecs = [v / (s + 1e-6) for v, s in zip(fast_vectors, speeds)]

            # --- 模式 A: 平行同向逃散 ---
            sum_vec = np.sum(unit_vecs, axis=0)
            phi_parallel = float(np.linalg.norm(sum_vec) / num_runners)
            mean_angle = np.arctan2(sum_vec[1], sum_vec[0])
            mean_dir_deg = float(np.degrees(mean_angle) % 360)

            # --- 模式 B: 放射狀四散 ---
            pos_arr = np.array(fast_positions)
            bev_origin_center = np.mean(pos_arr, axis=0)

            # 逆映射至影像畫面座標，以利畫面繪製標記
            if self.H_inv is not None:
                ipt = np.dot(self.H_inv, np.array([bev_origin_center[0], bev_origin_center[1], 1.0]))
                iw = ipt[2] if abs(ipt[2]) > 1e-6 else 1e-6
                img_origin_center = (float(ipt[0] / iw), float(ipt[1] / iw))
            else:
                img_origin_center = (float(bev_origin_center[0]), float(bev_origin_center[1]))

            radial_projections = []
            for pos, v_unit in zip(fast_positions, unit_vecs):
                to_runner = pos - bev_origin_center
                dist = np.linalg.norm(to_runner)
                min_dist_filter = 1.5 if self.H is not None else 15.0
                if dist > min_dist_filter:
                    u_rad = to_runner / dist
                    radial_projections.append(np.dot(v_unit, u_rad))

            if len(radial_projections) >= self.min_crowd_size:
                phi_radial = float(np.mean(radial_projections))

            # 綜合判定 (滿足其一即觸發)
            if phi_parallel >= self.parallel_thresh:
                instant_trigger = True
                evac_type = "PARALLEL"
            elif phi_radial >= self.radial_thresh:
                instant_trigger = True
                evac_type = "RADIAL"

        # 3. 時序滑動視窗平滑
        self._alert_window.append(instant_trigger)
        trigger_count = sum(self._alert_window)
        is_alert = (trigger_count / max(1, len(self._alert_window))) >= self.trigger_ratio

        return {
            "alert": is_alert,
            "instant_trigger": instant_trigger,
            "evac_type": evac_type,                      # "NONE", "PARALLEL", "RADIAL"
            "active_runners": num_runners,
            "phi_parallel": round(phi_parallel, 3),      # 同向指標 (0~1)
            "phi_radial": round(phi_radial, 3),          # 放射發散指標 (-1~1)
            "mean_direction_deg": round(mean_dir_deg, 1),
            "bev_origin": bev_origin_center,             # BEV 空間的事發中心 (wx, wy)
            "img_origin": img_origin_center              # 監視器畫面上的事發中心 (cx, cy)
        }