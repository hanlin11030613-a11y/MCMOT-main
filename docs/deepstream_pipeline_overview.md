# DeepStream Multi-Camera Tracking Pipeline — Walkthrough

**Language:** English | [繁體中文](deepstream_pipeline_overview.zh-Hant.md)

> 📎 Presentation: [link-to-presentation](https://docs.google.com/presentation/d/1hlOluXIu6J6kOGV_ikMTZ4aSQCSX9h43_XvlXoMiOCo/edit?usp=sharing)

This doc explains how `ds_multi_cam_tracking.py` (and its siblings in this
directory) are put together, with a focus on the DeepStream/GStreamer side of
things — what PGIE and SGIE are doing, how the pipeline is wired, and how to
actually run it.

## 1. The big picture

The tracking/ReID logic itself (BoT-SORT + `GlobalRegistry` +
`MultiCameraTracker` in `botsort/`) is plain Python/NumPy and doesn't know or
care where its inputs come from. Its job, per frame, is: given a list of
person detections (bbox + confidence + a 256-d appearance embedding), update
tracks and decide who's who — including across cameras.

DeepStream's entire job in this repo is to be the **front end** that produces
those inputs at video rate: decode N camera streams, batch them, run a person
detector, run an embedding (ReID) model on every detected person crop, and
hand the result to a GStreamer pad probe where a Python callback pulls it out
of DeepStream's metadata structures and feeds it to `MultiCameraTracker`.

```
camera 1 ─┐
          ├─► nvstreammux (batches N streams) ─► PGIE (detector) ─► SGIE (ReID embedder) ─► [probe: pull detections + embeddings] ─► BoTSORT / GlobalRegistry ─► OSD draw ─► tiler ─► sink
camera 2 ─┘
```

Everything left of "[probe]" is DeepStream/GStreamer/TensorRT. Everything at
and right of "[probe]" is the Python tracking stack. `reid_pad_buffer_probe`
(in `ds_multi_cam_tracking.py`) is the seam between the two worlds.

## 2. DeepStream/GStreamer basics used here

DeepStream apps are GStreamer pipelines: a chain of `Gst.Element`s linked
src-pad→sink-pad, each doing one job on the buffers flowing through
(decode, batch, infer, convert, render...). NVIDIA's DeepStream plugins
(`nvstreammux`, `nvinfer`, `nvdsosd`, ...) attach structured **metadata**
(`NvDsBatchMeta` → `NvDsFrameMeta` → `NvDsObjectMeta` → ...) to each buffer as
it flows downstream, which is what `pyds` (the Python bindings) lets you read
in a pad probe.

Key elements in this pipeline, in order:

| Element | Role |
|---|---|
| `uridecodebin` / `nvurisrcbin` (per source, inside `create_source_bin`) | Decodes each camera's file/RTSP URI on its own GPU-accelerated decoder |
| `nvstreammux` | Batches the N per-camera streams into one batch (`batch-size`, `width`/`height` from `app_config.yml`) so downstream inference runs once per batch instead of once per stream |
| `nvvideoconvert` + `capsfilter` (RGBA) | Converts the muxer's output to an RGBA format that's CPU-mappable, so the gallery-snapshot code can pull real pixels back out with `pyds.get_nvds_buf_surface()` |
| **`nvinfer` (PGIE)** | Primary detector — see §3 |
| **`nvinfer` (SGIE)** | Secondary, per-object ReID embedder — see §4 |
| `nvdslogger` | No-op passthrough element; its src pad is where `reid_pad_buffer_probe` is attached (just a convenient, stable attachment point after both inference stages have run) |
| `nvmultistreamtiler` | Composites all camera views into one tiled frame for display (rows × columns computed from `num_sources`) |
| `nvdsosd` | Draws the display meta (boxes/labels/lines) that the probe built, onto the frame |
| sink (`nveglglessink` / `nv3dsink` / `fakesink`) | Renders to screen (or discards, in `PERF_MODE`) |

## 3. PGIE — Primary GIE (the detector)

"GIE" = **G**PU **I**nference **E**ngine. The **primary** GIE runs on the
*full frame* and produces the initial set of object detections — this is
what everything downstream (tracking, ReID) is built on.

Config: `ds_include/dstest3_pgie_config.yml`, model: PeopleNet
(`resnet34_peoplenet_int8.onnx`, DeepStream's stock person detector).

Relevant properties:
- `gie-unique-id: 1` — PGIE's ID; SGIE configs reference this to know whose
  detections to run on.
- `num-detected-classes: 3`, `filter-out-class-ids: 1;2` — PeopleNet detects
  person/bag/face; classes 1 and 2 are filtered out so only **person** (class
  0) boxes survive.
- `infer-dims: 3;544;960` — the fixed input resolution PeopleNet expects; the
  frame is resized/letterboxed to this before inference.
- `batch-size` — must equal the number of camera sources; `main()` overrides
  whatever's in the config file with `num_sources` at startup and warns if it
  had to.
- `model-engine-file` — the built TensorRT engine. If it's missing/stale for
  the given batch size, `nvinfer` rebuilds it from the `onnx-file` on first
  run (slow, one-time cost per batch-size/precision combination).

Output: each detected person becomes an `NvDsObjectMeta` (bbox +
`confidence`) hanging off that frame's `NvDsFrameMeta`.

## 4. SGIE — Secondary GIE (the ReID embedder)

A **secondary** GIE runs *on crops of objects the primary GIE already
found*, not on the full frame — `operate-on-class-ids: 0` in
`sgie_config.yml` means "run on every PGIE object of class 0 (person)".
Its purpose here isn't classification, it's feature extraction: for every
person crop, produce a 256-dimensional appearance embedding to be used for
ReID matching.

Config: `ds_include/sgie_config.yml`, model: a Market1501-trained ReID
network (`resnet50_market1501_aicity156.onnx` by default; an OSNet variant is
also checked into `ds_include/onnx_model/` and swappable in the commented-out
block).

Relevant properties:
- `gie-unique-id: 2`, `process-mode: 2` — marks this `nvinfer` instance as
  secondary, chained off PGIE's (`gie-unique-id: 1`) detections.
- `network-type: 100` — "other"/custom network, since this isn't a
  detector or classifier, just a raw feature vector.
- `output-tensor-meta: 1` — **the important one.** Instead of nvinfer trying
  to interpret the output itself (as boxes or class scores), it attaches the
  raw output tensor to the object's metadata (`NVDSINFER_TENSOR_OUTPUT_META`)
  for something downstream to read. That "something" is the pad probe.
- `net-scale-factor` / `offsets` / `model-color-format: 0` (RGB) — pixel
  normalization applied before the crop is fed to the network; must match
  whatever the model was trained with (see the comments in
  `ds_include/onnx_model/deepstream_reid_config.yml` for the exact
  mean/std this corresponds to).
- `infer-dims: 3;256;128` — every person crop is resized to 256×128 before
  embedding, regardless of aspect ratio (`maintain-aspect-ratio: 0`).

`ds_include/onnx_model/deepstream_reid_config.yml` is a reference/notes file
for wiring the *same* ReID model into DeepStream's native NvDCF/DeepSORT
tracker config format (`ReID:` block) instead of a standalone SGIE — not
used by the pipeline as it stands today, but useful if the built-in tracker
element is ever brought back (there's a commented-out `tracker:` block in
`app_config.yml` from when that was in use).

## 5. Pulling the embedding out: the pad probe

`reid_pad_buffer_probe`, attached to `nvdslogger`'s src pad, runs once per
batch and does two passes:

1. **Collect** — walk `batch_meta.frame_meta_list` → per frame's
   `obj_meta_list`. For each object, walk `obj_meta.obj_user_meta_list`
   looking for a `NvDsUserMeta` of type `NVDSINFER_TENSOR_OUTPUT_META` (this
   is exactly the tensor SGIE attached because of `output-tensor-meta: 1`).
   The raw buffer is a C float pointer (`pyds.get_ptr`); it's cast via
   `ctypes` and copied into a NumPy array — that's the 256-d `reid_vector`.
   Detections are grouped by `frame_meta.source_id` (which camera).
2. **Update** — `mct.update_batch(detections_by_source, frame_id=...)` runs
   BoT-SORT + `GlobalRegistry.step()` once per camera for this tick, then
   display meta (boxes, `g<global_id> | t<track_id>`, ROI lines) is built
   back onto each source's own `frame_meta` for `nvdsosd` to draw.

A second probe, `save_dets_pad_buffer_probe`, is a simpler alternative that
just dumps each batch's detections + embeddings to a `.npy` file per batch
instead of running the tracker inline — useful for capturing a run's raw
detector/embedder output to replay later.

## 6. Running it

1. Edit `ds_include/app_config.yml`:
   - `source-list.list` — one or more `file://`, `rtsp://`, or `v4l2://` URIs,
     `;`-separated for multiple cameras.
   - `streammux.batch-size` should match the number of sources (PGIE/SGIE
     batch-size get overridden to match automatically at startup either way).
   - `primary-gie.config-file-path` / `secondary-gie-1.config-file-path` —
     point these at `ds_include/dstest3_pgie_config.yml` and
     `ds_include/sgie_config.yml` (the checked-in values point at another
     machine's path and will need fixing).
2. Update `path_to_botsort_parent` at the top of `ds_multi_cam_tracking.py`
   to wherever the `botsort` package actually lives if it's not already on
   `PYTHONPATH`.
3. Run:
   ```bash
   python3 ds_multi_cam_tracking.py
   ```
   - `GALLERY_DIR` env var — where per-identity crop snapshots get written
     (default `./gallery_snapshots`).
   - `NVDS_TEST3_PERF_MODE=1` — swaps to `nvurisrcbin`/`fakesink` for
     throughput testing without a display.
4. First run against a given PGIE/SGIE batch-size will build TensorRT
   engines from the ONNX files if the matching `.engine` file isn't already
   sitting next to it — expect a one-time delay.

### Other entry points in this directory

- `ds_single_cam_tracking.py` — same pipeline collapsed to one camera / one
  `BoTSORT` instance, no `MultiCameraTracker`/cross-camera matching.
- `ds_multi_cam_tracking_rtsp.py` — a variant using `nvmultiurisrcbin`
  (DeepStream's combined uri-source + streammux element) instead of manually
  building per-source bins and a separate `nvstreammux`; also updates each
  camera independently as its frames arrive rather than assuming both
  cameras land in lockstep every batch. Has some hard-coded/experimental
  bits (paths, `if False:` blocks) — treat as a work-in-progress reference,
  not the canonical pipeline.
- `single_cam_test.py` / `multi_cam_test.py` — offline, DeepStream-free
  visualizers. They read back `.npy` batches written by
  `save_dets_pad_buffer_probe` and draw the resulting boxes/track IDs on a
  blank canvas with OpenCV, so a captured run can be reviewed without
  re-running the GStreamer pipeline.
- `visualize_reid_latent.py` — offline embedding-space visualization
  (reads the `reid_log_path` JSONL/NPZ log written by `GlobalRegistry`).

## 7. Gotchas worth knowing (While working in Deepstream)

- **Memory type**: gallery snapshots read pixels via
  `pyds.get_nvds_buf_surface()`, which needs host-mappable memory. On dGPU
  (x86) the default NvBuf memory type is GPU-only device memory, so
  `streammux` and the RGBA-converter element explicitly set
  `nvbuf-memory-type` to `NVBUF_MEM_CUDA_UNIFIED` (Jetson/`aarch64` uses
  `NVBUF_MEM_DEFAULT`, which is already unified). Skipping this segfaults.
- **RGBA conversion placement**: the `nvvideoconvert` → RGBA `capsfilter`
  pair sits right after the muxer, before PGIE/SGIE touch the buffer, purely
  so the snapshot code has a CPU-mappable frame available later — decoder
  output is otherwise NV12.
- **`gie-unique-id` chaining**: PGIE and SGIE are just two `nvinfer`
  instances; what makes the second one "secondary" and tied to the first is
  config (`gie-unique-id`/`operate-on-class-ids`), not a different element
  type.

## 8. PoC Model Reference Guide

Where to get the actual model weights referenced above, for anyone running
these networks outside DeepStream (plain Python).

### Download links

- NVIDIA PeopleNet (NGC): <!-- TODO: link -->
- NVIDIA ReIdentificationNet (NGC): <!-- TODO: link -->
- Occlusion-robust person-detector YOLO weights (see note below): <!-- TODO: link -->

### wget

```bash
# PeopleNet
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_decrypted_v2.3.4/files/resnet34_peoplenet_int8.onnx --output-document resnet34_peoplenet_int8.onnx

# ReIdentificationNet
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.2/files/resnet50_market1501_aicity156.onnx --output-document resnet50_market1501_aicity156.onnx

# YOLO person-detector (crowded/occluded scenes)
https://drive.google.com/file/d/190rZXaBEBOcMo1gjNG32aerveAbaX2Zm/view?usp=sharing
```

### Notes

- **PeopleNet doesn't translate well to a plain local inference script.** The
  NGC artifact is built around DeepStream's `nvinfer` pipeline — bbox
  decoding, anchor handling, and clustering (`cluster-mode`,
  `nms-iou-threshold`, `pre-cluster-threshold` in
  `dstest3_pgie_config.yml`) all happen inside the plugin, not in the graph
  itself, so running the raw ONNX through plain `onnxruntime` means
  reimplementing that decode/NMS step by hand. It's also just not that
  strong a detector in dense/occluded crowds — it's tuned for fairly clean
  CCTV framing. For local Python inference, use a pretrained YOLO model
  trained on crowded/occluded scenes instead (e.g. a CrowdHuman-trained
  YOLOv8/YOLOv7 checkpoint) — standard YOLO postprocessing, and noticeably
  more accurate on overlapping people.
- **ReIdentificationNet is the easy one.** It's a plain embedding-head CNN
  (no detection decode involved) — the same `.onnx` NGC ships loads directly
  with `onnxruntime.InferenceSession`. Feed it 256×128 RGB crops normalized
  the same way as `sgie_config.yml` (`net-scale-factor` /
  `offsets` / `model-color-format: 0`), and `session.run()` returns the same
  embedding vector the SGIE tensor meta gives you today.
