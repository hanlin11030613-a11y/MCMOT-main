"""
Single-camera ReID gallery for BoT-SORT.

BoT-SORT already resolves frame-to-frame association on its own (it owns
track_id continuity). This module's only job is: once a tracklet has been
alive long enough to be trusted (`min_frames`), decide whether it belongs to
a person we've already met before (hand back their old global_id) or a
genuinely new person (mint a new one). That decision is made purely from
appearance, against a FAISS gallery of mean ReID embeddings - no IOU, no
motion. There's nothing camera-specific in this file at all: `query()`/
`step()` operate on whichever BoT-SORT instance is handed to them, so
multi-camera Global Tracklet Association (see botsort/multi_camera_tracker.py)
is just calling `step()` once per camera against one shared `GlobalRegistry` -
no separate multi-camera code path lives here.
"""

from __future__ import annotations

from typing import Optional

import json
import os
import queue
import threading

import numpy as np
import faiss
from sklearn.cluster import DBSCAN

from . import matching


def cosine_similarity(unit_a: np.ndarray, unit_b: np.ndarray, transform: Optional[np.ndarray] = None) -> float:
    """Similarity between two already-unit-normalized embeddings.

    Assumes both inputs are unit vectors - this is a drop-in for a plain dot
    product, not a general-purpose similarity (no internal renormalization).

    `transform`, if given, is a placeholder for a future learned
    cross-camera correction matrix M (`unit_a @ M @ unit_b`) - e.g. to
    correct for systematic appearance drift between two cameras' lighting/
    lens/color calibration. Not used anywhere yet (transform=None everywhere
    today), but kept here so that work has a single, obvious integration
    point. Note this would need to be applied to the query vector before a
    FAISS search (`query() @ M`), not inside FAISS itself - FAISS only
    computes plain inner products against the indexed (untransformed)
    gallery rows.
    """
    if transform is None:
        return float(np.dot(unit_a, unit_b))
    return float(unit_a @ transform @ unit_b)


class GalleryEntry:
    """One real-world identity living in the gallery.

    Appearance is summarized as a single EMA-updated centroid rather than a
    windowed mean over raw embeddings - O(1) per update instead of O(window),
    and paired with an outlier-rejection gate (see
    GlobalRegistry.outlier_reject_thresh) so a confidently-wrong observation
    (occlusion, motion blur, a brief mis-association) is dropped before it
    ever touches the centroid, instead of every historical embedding - good
    or corrupted - permanently sharing equal weight in a mean.
    """

    def __init__(self, global_id: int, feat: np.ndarray, frame_id: int, bbox: np.ndarray, ema_alpha: float = 0.9, cam_source=None):
        self.global_id = global_id
        self.active_tid: Optional[int] = None
        self.last_frame = frame_id
        self.last_bbox = bbox.copy()
        self.ema_alpha = ema_alpha

        self.centroid = feat.copy()
        self.update_count = 1
        # Logging-only: which camera (BoTSORT.cam_source) last claimed this
        # identity. Never read by query()/matching - GlobalRegistry stays
        # fully camera-agnostic, this exists purely so step() can report
        # whether a re-id match crossed cameras or not.
        self.last_cam_source = cam_source
        # Consecutive rejected (confidently-inconsistent) observations for
        # whoever currently holds active_tid. Used by GlobalRegistry.step()
        # to detect "this track_id's underlying identity got silently
        # swapped by BoT-SORT's own IOU-based association" (e.g. an
        # occlusion/crossing-paths mis-continuation) and revoke the
        # global_id rather than trust it forever. Resets whenever a
        # consistent observation comes back in, or active_tid changes.
        self.mismatch_streak = 0

    def add_embedding(self, feat: np.ndarray, frame_id: int, bbox: np.ndarray,
                      outlier_reject_thresh: float = 1.0, occlusion_score: float = 0.0) -> bool:
        """Returns True if the observation was consistent enough to blend
        into the centroid, False if it was rejected as an outlier.

        occlusion_score (0.0–1.0): how much the track overlaps with a peer
        track this frame. When > 0, the new observation's contribution to
        the centroid is scaled down proportionally — effective_alpha → 1.0
        (freeze) as occlusion_score → 1.0 — so overlap-contaminated
        embeddings drift the centroid less.
        """
        # Always record that the identity was seen this frame, even if the
        # observation itself gets rejected below - it shouldn't look stale
        # just because of a few corrupted frames in a row.
        self.last_frame = frame_id
        self.last_bbox = bbox.copy()

        n = np.linalg.norm(feat)
        feat_n = (feat / n).astype(np.float32) if n > 1e-9 else feat.astype(np.float32)

        cos_dist = 1.0 - cosine_similarity(self.centroid, feat_n)
        if cos_dist >= outlier_reject_thresh:
            # Confidently inconsistent with this identity's established
            # appearance - don't let it perturb the centroid.
            self.mismatch_streak += 1
            return False

        effective_alpha = self.ema_alpha + (1.0 - self.ema_alpha) * float(occlusion_score)
        c = effective_alpha * self.centroid + (1.0 - effective_alpha) * feat_n
        cn = np.linalg.norm(c)
        self.centroid = (c / cn).astype(np.float32) if cn > 1e-9 else c.astype(np.float32)
        self.update_count += 1
        self.mismatch_streak = 0
        return True

    def __repr__(self):
        return (f"GalleryEntry(gid={self.global_id}, tid={self.active_tid}, "
                f"n_updates={self.update_count}, last_frame={self.last_frame})")


class GlobalRegistry:
    """
    FAISS-backed appearance gallery, single camera only.

    match_threshold : cosine-distance ceiling to accept a re-ID match.
                       Lower = stricter (fewer false re-identifications, but
                       more new IDs minted for people seen before).
    min_frames      : a tracklet must survive this many frames before its
                       embedding is trusted enough to query/register with.
    min_margin      : required cosine-distance gap between the best and
                       second-best gallery hit. Guards against an ambiguous
                       gallery (two similar-looking identities) - if the
                       runner-up is nearly as close as the winner, treat the
                       query as unresolved rather than risk merging two
                       different people.
    ema_alpha       : weight given to an identity's existing centroid (vs.
                       the new observation) on each accepted update. Higher
                       = slower-drifting, more stable identity.
    outlier_reject_thresh : cosine-distance floor above which a new
                       observation is treated as confidently inconsistent
                       with an identity's established appearance and
                       dropped instead of blended into the centroid. Must
                       stay looser than match_threshold - same "confidently
                       right" vs "confidently wrong" split used for the
                       appearance veto in bot_sort.py. Defaults to
                       match_threshold + 0.3 if not given.
    identity_revoke_streak : once a track already carries a global_id,
                       step() trusts it and just refreshes the centroid -
                       it never re-verifies the identity. If BoT-SORT's own
                       IOU-based association silently hands that track_id
                       to a *different* person underneath it (e.g. an
                       occlusion/crossing-paths mis-continuation - a person
                       with very different appearance ends up carrying
                       someone else's track_id and therefore their
                       global_id), outlier_reject_thresh only protects the
                       centroid from corruption, it does nothing to correct
                       the now-wrong identity label. After this many
                       *consecutive* rejected observations for the same
                       track, conclude the underlying identity has likely
                       been swapped: revoke the global_id so the track
                       falls back through eligibility and gets freshly
                       re-queried/re-registered against its real, current
                       appearance.
    """

    def __init__(
        self,
        match_threshold: float = 0.35,
        min_frames: int = 5,
        emb_dim: int = 256,
        use_gpu: bool = True,
        min_margin: float = 0.05,
        ema_alpha: float = 0.9,
        outlier_reject_thresh: Optional[float] = None,
        identity_revoke_streak: int = 5,
        overlap_freeze_thresh: float = 0.3,
        overlap_cooldown_frames: int = 10,
        confusion_eps: float = 0.15,
        confusion_refresh_interval: int = 100,
        merge_threshold: float = 0.15,
        reid_log_path: Optional[str] = None,
        log_flush_interval: int = 500,
        log_format: str = "json",
    ):
        self.match_threshold = match_threshold
        self.min_frames = min_frames
        self.emb_dim = emb_dim
        self.min_margin = min_margin
        self.ema_alpha = ema_alpha
        self.identity_revoke_streak = identity_revoke_streak
        self.outlier_reject_thresh = (
            outlier_reject_thresh if outlier_reject_thresh is not None
            else min(1.0, match_threshold + 0.3)
        )

        self.overlap_freeze_thresh = overlap_freeze_thresh
        self.overlap_cooldown_frames = overlap_cooldown_frames
        # track_id → remaining cooldown frames after an overlap clears.
        # Prevents querying immediately after separation while smooth_feat
        # is still contaminated by the occluder.
        self._tid_cooldown: dict[int, int] = {}

        # Cosine-distance eps for confusion-zone DBSCAN and how often to
        # refresh. Gallery entries that cluster together at this distance
        # (i.e. two distinct identities whose centroids are suspiciously
        # similar — hard samples) get tighter match/margin thresholds in
        # query() to avoid merging two similar-clothing individuals.
        self.confusion_eps = confusion_eps
        self.confusion_refresh_interval = confusion_refresh_interval
        # Cosine distance below which a "claimed + new track" pair is treated
        # as a track split (same person) rather than a different person.
        # Must stay well below match_threshold so only near-certain splits
        # are merged — bypasses confusion-zone margin gating via _query_raw().
        self.merge_threshold = merge_threshold
        self._confusion_gids: set[int] = set()
        self._last_confusion_refresh: int = 0

        # Per-frame feature log.
        # smooth_feat is the unit-normalised EMA embedding that query() uses.
        #
        # log_format="jsonl" (default) — one JSON line per frame, append-only.
        #   No load-merge-save; background thread just appends. Readable with
        #   jq / pandas. Each line: {"frame": N, "gid": [[gid, [f0..f255]], ...]}
        # log_format="npz"   — numpy compressed archive.
        #   frame_{id}_gids (int32) + frame_{id}_feats (float32 N×D) per frame.
        #   ~10x smaller than JSON, fastest for numpy analysis. Requires load-
        #   merge-save on flush — use a larger log_flush_interval (e.g. 1000).
        #
        # The pipeline thread never does file I/O. It puts (frame_id, dict) into
        # a queue.Queue and returns immediately. A daemon background thread drains
        # the queue and writes to disk so the GStreamer/DeepStream thread is never
        # blocked by disk latency.
        self.reid_log_path = reid_log_path
        self.log_flush_interval = log_flush_interval
        self.log_format = log_format
        self._log_queue: queue.Queue = queue.Queue()
        self._log_stop = threading.Event()
        self._log_thread: Optional[threading.Thread] = None
        if reid_log_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(reid_log_path)), exist_ok=True)
            self._log_thread = threading.Thread(
                target=self._log_worker, name="reid-log-writer", daemon=True
            )
            self._log_thread.start()

        self._next_gid = 1
        self._gid_to_entry: dict[int, GalleryEntry] = {}
        self._tid_to_gid: dict[int, int] = {}

        cpu_index = faiss.IndexFlatIP(emb_dim)
        self._index = cpu_index
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            except Exception as exc:
                print(f"[REGISTRY] GPU FAISS unavailable ({exc}); using CPU index")

        self._index_gids: list[int] = []  # row i in self._index -> global_id

    # --- gallery search ------------------------------------------------

    def query(self, feat: np.ndarray) -> tuple[Optional[GalleryEntry], float]:
        """Nearest gallery entry to `feat`, gated by match_threshold and,
        when a second candidate exists, by min_margin (rejects ambiguous
        ties between two different gallery identities)."""
        if feat is None or self._index.ntotal == 0:
            return None, 1.0

        n = np.linalg.norm(feat)
        if n < 1e-9:
            return None, 1.0

        vec = np.ascontiguousarray((feat / n).astype(np.float32).reshape(1, -1))
        k = min(2, self._index.ntotal)
        sims, idxs = self._index.search(vec, k=k)
        pos = int(idxs[0, 0])
        if pos < 0:
            return None, 1.0

        cos_dist = 1.0 - float(sims[0, 0])

        gid = self._index_gids[pos]

        # Tighten thresholds when this identity is in a confusion zone
        # (its centroid is suspiciously close to another gallery entry —
        # a known hard-sample pair). Require stronger evidence before
        # committing to a match rather than risk merging two similar-
        # clothing individuals.
        if gid in self._confusion_gids:
            effective_match_thresh = self.match_threshold * 0.75
            effective_min_margin   = self.min_margin * 2.0
        else:
            effective_match_thresh = self.match_threshold
            effective_min_margin   = self.min_margin

        if cos_dist >= effective_match_thresh:
            return None, cos_dist

        if idxs.shape[1] > 1 and int(idxs[0, 1]) >= 0:
            cos_dist_2 = 1.0 - float(sims[0, 1])
            if (cos_dist_2 - cos_dist) < effective_min_margin:
                # Runner-up is nearly as close as the winner - the gallery
                # is ambiguous here, so don't risk merging two identities.
                return None, cos_dist

        return self._gid_to_entry.get(gid), cos_dist

    def _query_raw(self, feat: np.ndarray) -> tuple[Optional[GalleryEntry], float]:
        """Nearest gallery entry with NO threshold, margin, or confusion-zone
        gating. Used only by the track-split merge path in step() so that a
        confusion-zone margin rejection doesn't block a near-certain same-
        person merge (the feedback loop where splits keep minting new IDs
        that grow the confusion zone, making the next split's margin even
        tighter)."""
        if feat is None or self._index.ntotal == 0:
            return None, 1.0
        n = np.linalg.norm(feat)
        if n < 1e-9:
            return None, 1.0
        vec = np.ascontiguousarray((feat / n).astype(np.float32).reshape(1, -1))
        sims, idxs = self._index.search(vec, k=1)
        pos = int(idxs[0, 0])
        if pos < 0:
            return None, 1.0
        cos_dist = 1.0 - float(sims[0, 0])
        gid = self._index_gids[pos]
        return self._gid_to_entry.get(gid), cos_dist

    # --- gallery mutation ------------------------------------------------

    def _register(self, feat: np.ndarray, frame_id: int, bbox: np.ndarray, cam_source=None) -> int:
        gid = self._next_gid
        self._next_gid += 1
        self._gid_to_entry[gid] = GalleryEntry(gid, feat, frame_id, bbox, ema_alpha=self.ema_alpha, cam_source=cam_source)
        return gid

    def _rebuild_index(self):
        """FAISS IndexFlat has no cheap in-place update, so a centroid change
        means re-adding everything. Galleries here are small (one camera's
        worth of people), so this stays cheap."""
        self._index.reset()
        self._index_gids = []
        if not self._gid_to_entry:
            return
        gids = list(self._gid_to_entry.keys())
        centroids = np.ascontiguousarray(
            np.stack([self._gid_to_entry[g].centroid for g in gids], axis=0).astype(np.float32)
        )
        self._index.add(centroids)
        self._index_gids = gids

    def _refresh_confusion_zones(self, frame_id: int):
        """Periodically run DBSCAN on gallery centroids to detect identity
        pairs that are suspiciously close in embedding space (hard samples —
        typically similar-clothing individuals). Entries that cluster together
        are added to _confusion_gids so query() can apply tighter thresholds
        for them specifically."""
        if frame_id - self._last_confusion_refresh < self.confusion_refresh_interval:
            return
        self._last_confusion_refresh = frame_id
        if len(self._gid_to_entry) < 2:
            self._confusion_gids = set()
            return
        gids = list(self._gid_to_entry.keys())
        vecs = np.stack([self._gid_to_entry[g].centroid for g in gids])  # (N, D) unit vecs
        # Cosine distance matrix: 1 - dot(unit_vecs, unit_vecs.T)
        cos_dist = (1.0 - (vecs @ vecs.T)).clip(0.0, 2.0).astype(np.float64)
        labels = DBSCAN(eps=self.confusion_eps, min_samples=2, metric='precomputed').fit_predict(cos_dist)
        prev = self._confusion_gids
        self._confusion_gids = {gids[i] for i, lbl in enumerate(labels) if lbl >= 0}
        if self._confusion_gids != prev:
            print(f"[REGISTRY] confusion zones updated: {len(self._confusion_gids)} entries "
                  f"gids={sorted(self._confusion_gids)}")

    def deactivate_track(self, track_id: int):
        """Call when a track_id is permanently gone (removed, not merely
        lost) so its identity becomes available for re-ID again."""
        self._tid_cooldown.pop(track_id, None)
        gid = self._tid_to_gid.pop(track_id, None)
        if gid is None:
            return
        entry = self._gid_to_entry.get(gid)
        # Only clear active_tid when this was the primary holder. After a
        # track-split merge, two track_ids share one global_id — deactivating
        # the old one must not evict the new one's primary claim.
        if entry is not None and entry.active_tid == track_id:
            entry.active_tid = None

    # --- per-frame driver ------------------------------------------------

    def step(self, tracker, frame_id: int) -> list:
        """Run once per frame against a single BoTSORT instance.

        Returns the tracks that were just tied to a global_id during this
        call - either a fresh registration or a re-id match. Tracks that
        already had a global_id coming in (the common case, every frame)
        are not included. Useful for callers that want to react exactly
        once per identity-assignment event, e.g. saving a gallery snapshot.
        """
        index_dirty = False
        assigned_this_step: list = []
        cam_source = getattr(tracker, "cam_source", None)

        self._refresh_confusion_zones(frame_id)

        # Ground truth for "is this identity currently claimed by someone
        # else" is whoever BoT-SORT actually reports as tracked *this
        # frame* - not entry.active_tid by itself. BoT-SORT only calls
        # deactivate_track() once a lost track is permanently removed
        # (after the full max_time_lost buffer, tens of seconds), so a
        # quick exit-and-reentry would otherwise still see the old identity
        # as "claimed" and get minted a new global_id instead of being
        # re-identified.
        live_tids = {t.track_id for t in tracker.tracked_stracks}

        # --- Pairwise inter-track overlap (occlusion proxy) ----------------
        # Compute max IoU of each live track against every other live track.
        # Used to (a) scale down centroid updates for overlapping tracks and
        # (b) freeze global_id assignment while two people are overlapping.
        live = tracker.tracked_stracks
        if len(live) >= 2:
            iou_cost = matching.iou_distance(live, live)          # 1 - IoU
            iou_mat = 1.0 - iou_cost
            np.fill_diagonal(iou_mat, 0.0)
            peer_iou_vals = iou_mat.max(axis=1)
        else:
            peer_iou_vals = np.zeros(len(live))
        track_overlap = {t.track_id: float(peer_iou_vals[i]) for i, t in enumerate(live)}

        for track in tracker.tracked_stracks:
            feat = track.smooth_feat
            bbox = track.tlwh
            tid = track.track_id
            occlusion_score = track_overlap.get(tid, 0.0)

            if track.t_global_id != 0:
                entry = self._gid_to_entry.get(track.t_global_id)
                if entry is not None:
                    entry.active_tid = tid
                    entry.last_cam_source = cam_source
                    self._tid_to_gid[tid] = track.t_global_id
                    if feat is not None and not track.is_touching_edge:
                        index_dirty = True
                        if not entry.add_embedding(feat, frame_id, bbox,
                                                   outlier_reject_thresh=self.outlier_reject_thresh,
                                                   occlusion_score=occlusion_score):
                            if entry.mismatch_streak >= self.identity_revoke_streak:
                                print(f"[REGISTRY] identity revoked: track={tid} global_id={entry.global_id} "
                                      f"(appearance mismatched {entry.mismatch_streak} frames straight)")
                                entry.active_tid = None
                                entry.mismatch_streak = 0
                                self._tid_to_gid.pop(tid, None)
                                track.t_global_id = 0
                continue

            if track.is_touching_edge or track.tracklet_len < self.min_frames or feat is None:
                continue

            # --- Overlap freeze / cooldown for unidentified tracks ---------
            # While two people overlap, don't query: the embedding is
            # contaminated by the occluder. After they separate, wait
            # overlap_cooldown_frames for smooth_feat to recover before
            # committing to a gallery query.
            if occlusion_score > self.overlap_freeze_thresh:
                self._tid_cooldown[tid] = self.overlap_cooldown_frames
                continue
            if self._tid_cooldown.get(tid, 0) > 0:
                self._tid_cooldown[tid] -= 1
                continue
            # ---------------------------------------------------------------

            entry, dist = self.query(feat)

            if entry is not None and entry.active_tid not in live_tids:
                prev_cam = entry.last_cam_source
                cross_camera = prev_cam is not None and cam_source is not None and prev_cam != cam_source
                entry.active_tid = tid
                entry.last_cam_source = cam_source
                entry.add_embedding(feat, frame_id, bbox,
                                    outlier_reject_thresh=self.outlier_reject_thresh,
                                    occlusion_score=occlusion_score)
                self._tid_to_gid[tid] = entry.global_id
                track.t_global_id = entry.global_id
                index_dirty = True
                assigned_this_step.append(track)
                print(f"[REGISTRY] re-id match: track={tid} (cam={cam_source}) -> global_id={entry.global_id} "
                      f"(dist={dist:.3f}, prev_cam={prev_cam}, cross_camera={cross_camera})")
            else:
                # Check for track split before minting a new global_id.
                # query() may have returned (None, dist) because the closest
                # entry's active_tid is live (claimed) OR because confusion-
                # zone margin gating rejected it. In both cases dist is the
                # raw cosine distance to the closest entry. If that distance
                # is below merge_threshold, use _query_raw() (bypasses all
                # threshold/margin/confusion policy) to get the actual entry
                # and treat this as the same person with a split BoT-SORT track.
                raw_entry, raw_dist = self._query_raw(feat) if dist < self.merge_threshold else (None, 1.0)
                if raw_entry is not None and raw_dist < self.merge_threshold:
                    old_tid = raw_entry.active_tid
                    raw_entry.active_tid = tid
                    raw_entry.last_cam_source = cam_source
                    raw_entry.add_embedding(feat, frame_id, bbox,
                                            outlier_reject_thresh=self.outlier_reject_thresh,
                                            occlusion_score=occlusion_score)
                    self._tid_to_gid[tid] = raw_entry.global_id
                    track.t_global_id = raw_entry.global_id
                    # Keep old track's global_id mapping alive while BoT-SORT
                    # still reports it as tracked.
                    if old_tid is not None and old_tid in live_tids:
                        self._tid_to_gid[old_tid] = raw_entry.global_id
                        for t in tracker.tracked_stracks:
                            if t.track_id == old_tid:
                                t.t_global_id = raw_entry.global_id
                                break
                    index_dirty = True
                    assigned_this_step.append(track)
                    print(f"[REGISTRY] track split resolved: track={tid} -> global_id={raw_entry.global_id} "
                          f"(dist={raw_dist:.3f}, old_track={old_tid}, cam={cam_source})")
                else:
                    new_gid = self._register(feat, frame_id, bbox, cam_source=cam_source)
                    self._gid_to_entry[new_gid].active_tid = tid
                    self._tid_to_gid[tid] = new_gid
                    track.t_global_id = new_gid
                    index_dirty = True
                    assigned_this_step.append(track)
                    claimed_by = entry.active_tid if entry is not None else None
                    print(f"[REGISTRY] new identity: track={tid} (cam={cam_source}) -> global_id={new_gid} "
                          f"(closest_dist={dist:.3f}, closest_gid_claimed_by={claimed_by})")

        if index_dirty:
            self._rebuild_index()

        # --- Per-frame feature vector log (non-blocking) -------------------
        # Copy smooth_feat for every identified track and hand it off to the
        # background writer thread via a queue. The pipeline thread does zero
        # file I/O here — just builds a small dict and puts it in the queue.
        if self.reid_log_path is not None:
            frame_entry = {}
            for t in tracker.tracked_stracks:
                if t.t_global_id != 0 and t.smooth_feat is not None:
                    frame_entry[t.t_global_id] = t.smooth_feat.astype(np.float32).copy()
            if frame_entry:
                self._log_queue.put((frame_id, frame_entry))

        return assigned_this_step

    def _log_worker(self):
        """Background daemon thread: drains the log queue and writes to disk.
        Accumulates log_flush_interval frames in memory before each write so
        disk I/O happens in batches rather than per-frame."""
        buffer: list = []   # [(frame_id, {gid: np.ndarray})]
        while not self._log_stop.is_set():
            try:
                item = self._log_queue.get(timeout=0.5)
                buffer.append(item)
                if len(buffer) >= self.log_flush_interval:
                    self._write_buffer(buffer)
                    buffer.clear()
            except queue.Empty:
                # Timeout — flush whatever we have so data isn't held too long
                if buffer:
                    self._write_buffer(buffer)
                    buffer.clear()
        # Drain any items queued after stop was signalled
        while True:
            try:
                buffer.append(self._log_queue.get_nowait())
            except queue.Empty:
                break
        if buffer:
            self._write_buffer(buffer)

    def _write_buffer(self, buffer: list):
        """Write a batch of (frame_id, {gid: feat}) pairs to disk."""
        if not buffer:
            return
        if self.log_format == "npz":
            flat: dict = {}
            if os.path.exists(self.reid_log_path):
                try:
                    existing_npz = np.load(self.reid_log_path)
                    flat = {k: existing_npz[k] for k in existing_npz.files}
                    existing_npz.close()
                except Exception:
                    pass
            for frame_id, gid_feat_map in buffer:
                gids  = np.array(list(gid_feat_map.keys()), dtype=np.int32)
                feats = np.stack(list(gid_feat_map.values()), axis=0)
                flat[f"frame_{frame_id}_gids"]  = gids
                flat[f"frame_{frame_id}_feats"] = feats
            np.savez_compressed(self.reid_log_path, **flat)
        else:
            # jsonl — append one line per frame, no load needed
            with open(self.reid_log_path, "a") as f:
                for frame_id, gid_feat_map in buffer:
                    line = {"frame": frame_id,
                            "gid": [[int(g), v.tolist()] for g, v in gid_feat_map.items()]}
                    f.write(json.dumps(line) + "\n")

    def stop_logging(self):
        """Signal the background writer to finish and wait for it to exit.
        Call at pipeline shutdown to guarantee all queued frames are written."""
        if self._log_thread is None:
            return
        self._log_stop.set()
        self._log_thread.join(timeout=15)
        self._log_thread = None

    def get_all_entries(self) -> list[GalleryEntry]:
        return list(self._gid_to_entry.values())

    def size(self) -> int:
        return len(self._gid_to_entry)

    def __repr__(self):
        active = sum(1 for e in self._gid_to_entry.values() if e.active_tid is not None)
        return f"GlobalRegistry(total={len(self._gid_to_entry)}, active={active}, threshold={self.match_threshold})"
