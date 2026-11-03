"""
global_registry.py
──────────────────
A persistent identity gallery that sits above BoTSORT.

Responsibilities:
  1. Maintain a list of GalleryEntry objects (one per real-world person).
  2. When a new tracklet is about to be confirmed, query the gallery first.
     - Match found (cosine dist < threshold) → reuse old global_id.
     - No match                               → mint a new global_id.
  3. Update a gallery entry's centroid every time its track is seen.
  4. Archive entries when tracks are permanently removed (for future FAISS swap-in).

Data structure (simple Python list now, FAISS-ready later):
  self._entries : List[GalleryEntry]

Each GalleryEntry stores:
  global_id     : int       – stable identity across re-entries
  centroid      : np.array  – L2-normalised mean of all collected embeddings
  embeddings    : deque     – rolling buffer of raw embeddings (max_size)
  last_frame    : int       – frame when this entry was last updated
  last_bbox     : np.array  – tlwh bbox at last sighting (for ghost init)
  cam_id        : int       – camera id where entry was last observed
  active_tid    : int | None– the current BoTSORT track_id linked to this entry
                              None when person is out of frame
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import Optional
from scipy.optimize import linear_sum_assignment
import faiss



class GalleryEntry:
    def __init__(
        self,
        global_id:  int,
        feat:       np.ndarray,
        track_id:   int,
        frame_id:   int,
        bbox:       np.ndarray,
        cam_id:     int = 0,
        max_emb:    int = 50,
    ):
        self.global_id  = global_id
        self.active_tid = track_id
        self.last_frame = frame_id
        self.last_bbox  = bbox.copy()
        self.cam_id     = cam_id

        self.embeddings: deque = deque(maxlen=max_emb)
        self.embeddings.append(feat)
        self.centroid = feat.copy()

    def _recompute_centroid(self):
        c = np.mean(self.embeddings, axis=0).astype(np.float32)
        n = np.linalg.norm(c)
        self.centroid = c / n if n > 1e-9 else c

    def add_embedding(self, feat: np.ndarray, frame_id: int, bbox: np.ndarray, cam_id: int = 0):
        self.embeddings.append(feat)
        self._recompute_centroid()
        self.last_frame = frame_id
        self.last_bbox  = bbox.copy()
        self.cam_id     = cam_id

    def similarity(self, feat: np.ndarray) -> float:
        n = np.linalg.norm(feat)
        if n < 1e-9 or np.linalg.norm(self.centroid) < 1e-9:
            return 0.0
        return float(np.dot(self.centroid, feat / n))

    def cosine_distance(self, feat: np.ndarray) -> float:
        return 1.0 - self.similarity(feat)

    def __repr__(self):
        return (f"GalleryEntry(gid={self.global_id}, "
                f"tid={self.active_tid}, "
                f"cam={self.cam_id}, "
                f"n_emb={len(self.embeddings)}, "
                f"last_frame={self.last_frame})")

class GlobalRegistry:
    def __init__(
        self,
        match_threshold:  float = 0.30,
        min_frames:       int   = 5,
        max_emb:          int   = 50,
        emb_dim:          int   = 256,
        min_gap_frames:   int   = 10,
        intra_cam_gap:    int   = 30,
    ):
        self.match_threshold = match_threshold
        self.min_frames      = min_frames
        self.max_emb         = max_emb
        self.min_gap_frames  = min_gap_frames
        self.intra_cam_gap   = intra_cam_gap

        self._entries:       list[GalleryEntry] = []
        self._tid_to_gid:    dict[int, int]     = {}
        self._global_id_ctr: int                = 0

        self._emb_dim   = emb_dim
        self._index_cpu = faiss.IndexFlatIP(emb_dim)

        res             = faiss.StandardGpuResources()
        self._index     = faiss.index_cpu_to_gpu(res, 0, self._index_cpu)

        self._faiss_pos_to_gid: list[int] = []
        self._gid_to_entry: dict[int, GalleryEntry] = {}
        # maps gid → position in self._entries for O(1) cost-matrix lookup
        self._gid_to_pos:   dict[int, int] = {}

    def _new_global_id(self) -> int:
        self._global_id_ctr += 1
        return self._global_id_ctr

    def _can_link(self, entry: GalleryEntry, cam_id: int, frame_id: int) -> bool:
        """Return True only when it is safe to re-link a new tracklet to entry.

        Guards:
          1. Hard block if entry is still active (has a live track_id).
          2. Temporal gate: entry must have been inactive for min_gap_frames.
          3. Intra-camera suppression (ported from AIC21-MTMC filter.intracam_ignore):
             if the query track and the gallery entry share the same camera, require
             a longer gap before re-ID is allowed, preventing same-camera confusion.
        """
        gap = frame_id - entry.last_frame
        if entry.cam_id == cam_id and gap < self.intra_cam_gap:
            print("e2")
            return False
        return True

    def query(self, feat: np.ndarray) -> tuple[Optional[GalleryEntry], float]:
        if self._index.ntotal == 0 or feat is None:
            return None, 1.0

        vec = feat.astype(np.float32).reshape(1, -1)
        vec = np.ascontiguousarray(vec)

        sims, idxs = self._index.search(vec, k=1)

        sim       = float(sims[0, 0])
        faiss_pos = int(idxs[0, 0])

        if faiss_pos < 0:
            return None, 1.0

        cos_dist = 1.0 - sim

        if cos_dist < self.match_threshold:
            gid   = self._faiss_pos_to_gid[faiss_pos]
            entry = self._get_entry_by_gid(gid)
            return entry, cos_dist

        return None, cos_dist

    def query_batch(
        self,
        feats: list[np.ndarray],
    ) -> list[tuple[Optional[GalleryEntry], float]]:
        results: list[tuple[Optional[GalleryEntry], float]] = [
            (None, 1.0)
        ] * len(feats)

        if self._index.ntotal == 0 or not feats:
            return results

        valid_feats = []
        valid_idx   = []
        for i, f in enumerate(feats):
            if f is not None and np.linalg.norm(f) > 1e-9:
                valid_feats.append(f.astype(np.float32) / np.linalg.norm(f))
                valid_idx.append(i)

        if not valid_feats:
            return results

        query_mat = np.ascontiguousarray(
            np.stack(valid_feats, axis=0).astype(np.float32)
        )

        sims, idxs = self._index.search(query_mat, k=1)

        for qi, orig_i in enumerate(valid_idx):
            sim       = float(sims[qi, 0])
            faiss_pos = int(idxs[qi, 0])

            if faiss_pos < 0:
                continue

            cos_dist = 1.0 - sim

            if cos_dist < self.match_threshold:
                gid   = self._faiss_pos_to_gid[faiss_pos]
                entry = self._get_entry_by_gid(gid)
                results[orig_i] = (entry, cos_dist)
            else:
                results[orig_i] = (None, cos_dist)

        return results

    def _register_new(self, track_id, feat, frame_id, bbox, cam_id: int = 0):
        gid = self._new_global_id()
        entry = GalleryEntry(
            global_id=gid, feat=feat, track_id=track_id,
            frame_id=frame_id, bbox=bbox, cam_id=cam_id, max_emb=self.max_emb,
        )
        pos = len(self._entries)
        self._entries.append(entry)
        self._gid_to_entry[gid] = entry
        self._gid_to_pos[gid]   = pos
        self._tid_to_gid[track_id] = gid

        vec = entry.centroid.astype(np.float32).reshape(1, -1)
        vec = np.ascontiguousarray(vec)
        self._index.add(vec)
        self._faiss_pos_to_gid.append(gid)

        return gid


    def _link_existing(self, entry: GalleryEntry, track_id: int) -> bool:
        """Link track_id to an existing gallery entry.

        Returns False (no-op) if the entry is already active with a different
        track — the caller should treat this track as a new entry instead.
        """
        if entry.active_tid is not None and entry.active_tid != track_id:
            return False
        old_tid = entry.active_tid
        if old_tid is not None and old_tid in self._tid_to_gid:
            del self._tid_to_gid[old_tid]
        entry.active_tid = track_id
        self._tid_to_gid[track_id] = entry.global_id
        return True

    def deactivate_track(self, track_id: int):
        gid = self._tid_to_gid.pop(track_id, None)
        if gid is None:
            return
        entry = self._gid_to_entry.get(gid)  # O(1)
        if entry is not None:
            entry.active_tid = None


    def step(self, tracker, frame_id: int):
        # 1. Deactivate gone tracks
        current_tids = {t.track_id for t in tracker.tracked_stracks}
        linked_tids  = set(self._tid_to_gid.keys())
        for gone_tid in linked_tids - current_tids:
            self.deactivate_track(gone_tid)

        # 2. Rebuild FAISS with fresh centroids BEFORE any queries this frame
        self._rebuild_faiss_index()

        # 3. Update embeddings for already-linked tracks
        for track in tracker.tracked_stracks:
            if track.t_global_id == 0:
                continue
            entry = self._get_entry_by_gid(track.t_global_id)
            if entry is not None and track.smooth_feat is not None:
                cam_id = getattr(track, 'cam_id', 0)
                entry.add_embedding(track.smooth_feat, frame_id, track.tlwh, cam_id)

        # 4. Collect tracks that need a global_id assigned
        unlinked = [
            t for t in tracker.tracked_stracks
            if t.t_global_id == 0 and t.tracklet_len >= self.min_frames
        ]
        if not unlinked:
            return

        # 5. Batch query for all unlinked tracks
        feats        = [t.smooth_feat for t in unlinked]
        batch_result = self.query_batch(feats)

        # 6. Build cost matrix [unlinked tracks × all gallery entries]
        #    Only fill in cells that pass _can_link; rest remain INF (= 1.0)
        all_entries = self._entries          # stable order within this step
        n_tracks    = len(unlinked)
        n_entries   = len(all_entries)
        INF         = 1.0

        cost = np.full((n_tracks, max(n_entries, 1)), INF, dtype=np.float32)

        for ti, (track, (best_entry, best_dist)) in enumerate(
            zip(unlinked, batch_result)
        ):
            if best_entry is None:
                continue
            ei = self._gid_to_pos.get(best_entry.global_id)
            if ei is None:
                continue
            cam_id = getattr(track, 'cam_id', 0)
            if self._can_link(best_entry, cam_id, frame_id):
                cost[ti, ei] = best_dist

        # 7. Hungarian (linear sum) assignment — one-to-one, no entry stolen twice
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_track_indices = set()
        for ri, ci in zip(row_ind, col_ind):
            if ci >= n_entries or cost[ri, ci] >= self.match_threshold:
                continue
            track = unlinked[ri]
            entry = all_entries[ci]
            if not self._link_existing(entry, track.track_id):
                continue                         # entry became active mid-loop (safety)
            cam_id = getattr(track, 'cam_id', 0)
            feat   = track.smooth_feat
            if feat is not None:
                entry.add_embedding(feat, frame_id, track.tlwh, cam_id)
            track.t_global_id = entry.global_id
            matched_track_indices.add(ri)
            print(
                f"[REGISTRY] Re-entry: track_id={track.track_id} → "
                f"global_id={entry.global_id} (cos_dist={cost[ri, ci]:.3f})"
            )

        # 8. Register unmatched tracks as new gallery entries
        for ti, track in enumerate(unlinked):
            if ti in matched_track_indices:
                continue
            feat   = track.smooth_feat
            cam_id = getattr(track, 'cam_id', 0)
            if feat is None:
                feat = np.zeros(self._emb_dim, dtype=np.float32)
            gid = self._register_new(track.track_id, feat, frame_id, track.tlwh, cam_id)
            track.t_global_id = gid
            print(
                f"[REGISTRY] New entry: track_id={track.track_id} → global_id={gid}"
            )

    def _rebuild_faiss_index(self):
        if not self._entries:
            self._index.reset()
            self._faiss_pos_to_gid = []
            self._gid_to_pos       = {}
            return

        centroids = np.stack(
            [e.centroid.astype(np.float32) for e in self._entries], axis=0
        )
        centroids = np.ascontiguousarray(centroids)

        self._index.reset()
        self._index.add(centroids)
        self._faiss_pos_to_gid = [e.global_id for e in self._entries]
        self._gid_to_pos       = {e.global_id: i for i, e in enumerate(self._entries)}

    def _get_entry_by_gid(self, gid: int) -> Optional[GalleryEntry]:
        return self._gid_to_entry.get(gid)

    def get_all_entries(self) -> list[GalleryEntry]:
        return list(self._entries)

    def size(self) -> int:
        return len(self._entries)

    def __repr__(self):
        active = sum(1 for e in self._entries if e.active_tid is not None)
        return (f"GlobalRegistry("
                f"total={len(self._entries)}, "
                f"active={active}, "
                f"threshold={self.match_threshold})")