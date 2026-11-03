"""
Multi-camera tracking via Global Tracklet Association (GTA).

Pure orchestration - no new matching logic lives here. Cross-camera identity
matching is already provided by global_registry.py's GlobalRegistry: it has
no camera concept anywhere in it, so calling step() once per camera against
ONE shared GlobalRegistry instance makes a camera-B tracklet's appearance
query transparently match a camera-A-registered identity, through the exact
same code path already used for same-camera re-entry.

This depends on each camera's BoTSORT being constructed with a distinct
cam_source (see bot_sort.py - this seeds a per-instance ID_Assigner so
track_ids never collide across cameras) and the same shared `registry=`.

Assumption this design relies on: camera views are non-overlapping - a
person is never live-tracked in two cameras at the same instant. If that
assumption is ever relaxed, the first thing to revisit is
GlobalRegistry.step()'s concurrent-claim guard, which only sees one
tracker's live track_ids per call and can't detect two different cameras
claiming the same identity in the same tick.
"""

from __future__ import annotations

from typing import Optional

from .bot_sort import BoTSORT
from .global_registry import GlobalRegistry


class MultiCameraTracker:
    """Thin orchestrator over N per-camera BoTSORT instances sharing one
    GlobalRegistry. Does not implement any matching itself - all of it is
    delegated to BoTSORT.update() and GlobalRegistry.step()."""

    def __init__(self, registry: GlobalRegistry, trackers: Optional[dict] = None):
        self.registry = registry
        self.trackers: dict = dict(trackers) if trackers else {}
        # cam_id -> tracks that were just tied to a global_id on the most
        # recent update_camera() call for that camera (registry.step()'s
        # return value, see global_registry.py) - a fresh registration or a
        # re-id match, not the common "already identified" case. Side
        # channel rather than a return-value change so update_camera()'s
        # existing list[STrack] return stays backward compatible (e.g.
        # ds_multi_cam_tracking_rtsp.py uses it directly for display).
        self.last_assigned: dict = {}

    def add_camera(self, cam_id, tracker: BoTSORT):
        """Register a per-camera BoTSORT instance. Caller is responsible for
        constructing it with a distinct cam_source and registry=self.registry."""
        self.trackers[cam_id] = tracker

    def update_camera(self, cam_id, detections, frame_id: int) -> list:
        """Update a single camera and run the registry step for it.

        Use this for asynchronous/per-source pipelines where not every
        camera has a new frame on every tick (e.g. multiple RTSP sources
        batched together) - call once per camera that has a frame this
        tick, with that camera's own frame_id (e.g. frame_meta.frame_num).
        """
        tracker = self.trackers[cam_id]
        tracks = tracker.update(detections)
        self.last_assigned[cam_id] = self.registry.step(tracker, frame_id)
        return tracks

    def update_batch(self, detections_by_cam: dict, frame_id: int) -> dict:
        """Update all registered cameras for one synchronous tick, where
        every camera's detections arrive together and share one frame_id."""
        return {
            cam_id: self.update_camera(cam_id, detections_by_cam.get(cam_id, []), frame_id)
            for cam_id in self.trackers
        }
