import sys
import os
import math
import platform
import yaml
import ctypes
import tty
import termios
import select

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import pyds

import numpy as np
import time
import threading
from collections import defaultdict
import nvtx

from glob import glob

path_to_botsort_parent = '/home/lakshh/workspace/reid/botsort-tracker'

if path_to_botsort_parent not in sys.path:
    sys.path.append(path_to_botsort_parent)
 
from botsort.bot_sort import BoTSORT
from botsort.global_registry import GlobalRegistry
from botsort.multi_camera_tracker import MultiCameraTracker


registry = GlobalRegistry(
    match_threshold=0.3,
    min_frames=5,
    emb_dim=256,
)

_tracker_kwargs = dict(
    track_high_thresh=0.6,
    track_low_thresh=0.1,
    new_track_thresh=0.3,
    track_buffer=600,
    match_thresh=0.8,
    with_reid=True,
    proximity_thresh=0.5,
    appearance_thresh=0.2,
    euc_thresh=0.1,
    fuse_score=True,
    frame_rate=30,
    max_batch_size=8,
    map_len=None,
    real_data=True,
    registry=registry,
)

tracker0 = BoTSORT(
    cam_source=1,
    track_high_thresh=0.6,
    track_low_thresh=0.1,
    new_track_thresh=0.3,
    track_buffer=600,
    match_thresh=0.8,
    with_reid=True,
    proximity_thresh=0.5,
    appearance_thresh=0.2,
    euc_thresh=0.1,
    fuse_score=True,
    frame_rate=30,
    max_batch_size=8,
    map_len=None,
    real_data=True,
    registry=registry,)  # camera source_id 0
tracker1 = BoTSORT(
    cam_source=2,
    track_high_thresh=0.6,
    track_low_thresh=0.1,
    new_track_thresh=0.3,
    track_buffer=600,
    match_thresh=0.8,
    with_reid=True,
    proximity_thresh=0.5,
    appearance_thresh=0.2,
    euc_thresh=0.1,
    fuse_score=True,
    frame_rate=30,
    max_batch_size=8,
    map_len=None,
    real_data=True,
    registry=registry)  # camera source_id 1

mct = MultiCameraTracker(registry=registry)
mct.add_camera(0, tracker0)
mct.add_camera(1, tracker1)

trackers = [tracker0, tracker1]

PERF_MODE = os.environ.get("NVDS_TEST3_PERF_MODE") == "1"

cur_frame  = 0
ACTIVE_FORMAT = "tlwh"

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End of stream\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write(f"WARNING from element {message.src.get_name()}: {err.message}\n")
        sys.stderr.write(f"Warning: {err.message}\n")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write(f"ERROR from element {message.src.get_name()}: {err.message}\n")
        if debug:
            sys.stderr.write(f"Error details: {debug}\n")
        loop.quit()
    elif t == Gst.MessageType.ELEMENT:
        struct = message.get_structure()
        if struct and struct.get_name() == "nvmsg-stream-eos":
            stream_id = struct.get_value("stream-id")
            sys.stdout.write(f"Got EOS from stream {stream_id}\n")
    return True

def cb_newpad(decodebin, decoder_src_pad, data):
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps(None)
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    multi_src_bin = data
    features = caps.get_features(0)

    if gstname.find("video") != -1:
        if features.contains("memory:NVMM"):
            bin_ghost_pad = multi_src_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write("Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy, Object, name, user_data):
    sys.stdout.write(f"Decodebin child added: {name}\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)
    if "source" in name:
        Object.set_property("drop-on-latency", True)

def create_source_bin(index, uri):
    sys.stdout.write(f"{uri}\n")
    bin_name = f"source-bin-{index:02d}"
    nbin = Gst.Bin.new(bin_name)


    uri_decode_bin = Gst.ElementFactory.make("nvmultiurisrcbin", "uri-decode-bin")

    uri_decode_bin.set_property("uri-list", uri)
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write("Failed to add ghost pad in source bin\n")
        return None

    return nbin

@nvtx.annotate("reid_probe", color="blue")
def reid_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    # ── Pass 1: collect detections and frame_meta refs for every source ──
    detections_by_source: dict[int, list] = defaultdict(list)
    frame_meta_by_source: dict[int, object] = {}

    nvtx.push_range("build_detections", color="green")
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        source_id = frame_meta.source_id
        frame_meta_by_source[source_id] = frame_meta

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 1.0)
            obj_meta.rect_params.border_width = 1
            obj_meta.text_params.display_text = ""
            reid_vector = None

            nvtx.push_range("reid_extract", color="yellow")
            l_user = obj_meta.obj_user_meta_list
            while l_user is not None:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                except StopIteration:
                    break

                if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                    embed_len = 1
                    for i in range(layer.inferDims.numDims):
                        embed_len *= layer.inferDims.d[i]
                    reid_vector = np.copy(np.ctypeslib.as_array(ptr, shape=(embed_len,)))

                l_user = l_user.next
            nvtx.pop_range()  # reid_extract

            is_touching_edge = (
                obj_meta.rect_params.left <= 0
                or obj_meta.rect_params.top <= 0
                or obj_meta.rect_params.left + obj_meta.rect_params.width >= 1900
                or obj_meta.rect_params.top + obj_meta.rect_params.height >= 1060
            )
            det = {
                "obj_meta": l_obj.data,
                "bbox": np.array([
                    obj_meta.rect_params.left,
                    obj_meta.rect_params.top,
                    obj_meta.rect_params.width,
                    obj_meta.rect_params.height,
                ], dtype=np.float32),
                "det_confidence": 0.0 if is_touching_edge else obj_meta.confidence,
                "reid_vector": reid_vector,
            }
            detections_by_source[source_id].append(det)
            l_obj = l_obj.next

        l_frame = l_frame.next
    nvtx.pop_range() 

    all_tracks_by_source: dict[int, list] = {}
    with nvtx.annotate("tracker_update", color="red"):
        # Iterate frame_meta_by_source so sources with zero detections in this
        # batch still advance their tracker (marks tracks lost over time),
        # and sources absent this tick are skipped entirely. update_camera()
        # runs BoTSORT.update() then registry.step() for that one camera,
        # using its own frame_meta.frame_num - cross-camera re-identification
        # (GTA) happens automatically via the shared GlobalRegistry, see
        # botsort/multi_camera_tracker.py.
        for source_id, frame_meta in frame_meta_by_source.items():
            if source_id < len(trackers):
                dets = detections_by_source.get(source_id, [])
                tracks = mct.update_camera(source_id, dets, frame_id=frame_meta.frame_num)
                all_tracks_by_source[source_id] = tracks

    nvtx.push_range("build_display_meta", color="cyan")
    MAX_DISPLAY_SLOTS = 16
    for source_id, all_tracks in all_tracks_by_source.items():
        frame_meta = frame_meta_by_source.get(source_id)
        if frame_meta is None:
            continue

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        slot = 0

        for t in all_tracks:
            if slot >= MAX_DISPLAY_SLOTS:
                display_meta.num_rects = MAX_DISPLAY_SLOTS
                display_meta.num_labels = MAX_DISPLAY_SLOTS
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
                display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                slot = 0

            rect_params = display_meta.rect_params[slot]
            rect_params.left = t.tlwh[0]
            rect_params.top = t.tlwh[1]
            rect_params.width = t.tlwh[2]
            rect_params.height = t.tlwh[3]
            rect_params.border_width = 1
            rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)
            rect_params.has_bg_color = 0

            text_params = display_meta.text_params[slot]
            text_params.display_text = f"G{t.t_global_id} L{t.track_id}"
            text_params.x_offset = max(0, int(t.tlwh[0]))
            text_params.y_offset = max(0, int(t.tlwh[1]))
            text_params.font_params.font_name = "Serif"
            text_params.font_params.font_size = 7
            text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            text_params.set_bg_clr = 1
            text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)
            slot += 1

        if slot >= MAX_DISPLAY_SLOTS:
            display_meta.num_rects = MAX_DISPLAY_SLOTS
            display_meta.num_labels = MAX_DISPLAY_SLOTS
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            slot = 0

        display_meta.num_rects = slot
        display_meta.num_labels = slot + 1
        summary = display_meta.text_params[slot]
        summary.display_text = f"Cam{source_id} active={len(all_tracks)}"
        summary.x_offset = 10
        summary.y_offset = 12
        summary.font_params.font_name = "Serif"
        summary.font_params.font_size = 10
        summary.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        summary.set_bg_clr = 1
        summary.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    nvtx.pop_range()  # build_display_meta


    if False:
        starting_frame = array_of_frames[0]["frame_id"]
        save_dir = "deepstream_npy_output"
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"batch_frame_{startig_frame}.npy")
        np_data = np.array(array_of_frames, dtype=object)
        np.save(filename, np_data)

    return Gst.PadProbeReturn.OK

def save_dets_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    
    array_of_frames = []

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
            
        frame_dict = {
            "frame_id": frame_meta.frame_num,
            "sensor_id": f"platform_{frame_meta.source_id}_camera_{chr(65 + (frame_meta.pad_index % 26))}",
            "objects": []
        }

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            obj_dict = {
                "obj_meta": None,
                "local_track_id": obj_meta.object_id,
                "bbox": np.array([
                    obj_meta.rect_params.left,
                    obj_meta.rect_params.top,
                    obj_meta.rect_params.width,
                    obj_meta.rect_params.height
                ], dtype=np.float32),
                "det_confidence": obj_meta.confidence,
                "reid_vector": None
            }

            l_user = obj_meta.obj_user_meta_list
            while l_user is not None:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                except StopIteration:
                    break

                if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                    
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                    
                    embed_len = 1
                    for i in range(layer.inferDims.numDims):
                        embed_len *= layer.inferDims.d[i]
                        
                    reid_array = np.ctypeslib.as_array(ptr, shape=(embed_len,))
                    obj_dict["reid_vector"] = np.copy(reid_array)

                l_user = l_user.next

            frame_dict["objects"].append(obj_dict)
            l_obj = l_obj.next
            
        array_of_frames.append(frame_dict)
        l_frame = l_frame.next

    # --- NEW SAVING LOGIC HERE ---
    if array_of_frames:
        # 1. Get the first frame number in this batch to use in the filename
        starting_frame = array_of_frames[0]["frame_id"]
        
        # 2. Define your output directory and ensure it exists
        save_dir = "/home/lakshh/workspace/reid/ds_backend_reid/MCDPT/deepstream_npy_output2"
        os.makedirs(save_dir, exist_ok=True)
        
        # 3. Create a unique filename for this batch
        filename = os.path.join(save_dir, f"batch_frame_{starting_frame}.npy")
        
        # 4. Cast the list to a NumPy object array and save
        # dtype=object is required because the list contains dictionaries
        np_data = np.array(array_of_frames, dtype=object)
        np.save(filename, np_data)

    return Gst.PadProbeReturn.OK

def main():
    global _pipeline_ref
    Gst.init(None)

    yaml_file = "ds_include/app_config.yml"
    with open(yaml_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            sys.stderr.write(f"Error in parsing configuration file: {exc}\n")
            return -1

    loop = GLib.MainLoop()
    pipeline = Gst.Pipeline.new("dstest3-pipeline")


    # streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    # pipeline.add(streammux)

    # Parse Source List
    # source_list_config = config.get('source-list', {})
    # sources = []
    # for key, value in source_list_config.items():
    #     if key.startswith('list'):
    #         if isinstance(value, str):
    #             sources.extend(value.split(';'))
    #         elif isinstance(value, list):
    #             sources.extend(value)
    # sources = [s for s in sources if s]

    # for i, uri in enumerate(sources):
    #     sys.stdout.write(f"Now playing : {uri}\n")
    #     multi_src_bin = create_source_bin(i, uri)
    #     if not multi_src_bin:
    #         sys.stderr.write("Failed to create source bin. Exiting.\n")
    #         return -1

    #     pipeline.add(multi_src_bin)
    #     pad_name = f"sink_{i}"
    #     sinkpad = streammux.request_pad_simple(pad_name)
    #     if not sinkpad:
    #         sys.stderr.write("Streammux request sink pad failed. Exiting.\n")
    #         return -1

    #     srcpad = multi_src_bin.get_static_pad("src")
    #     if not srcpad:
    #         sys.stderr.write("Failed to get src pad of source bin. Exiting.\n")
    #         return -1

    #     if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
    #         sys.stderr.write("Failed to link source bin to stream muxer. Exiting.\n")
    #         return -1

    # multi_src_bin = Gst.ElementFactory.make("nvmultiurisrcbin", "src-bin")


    multi_src_bin = Gst.ElementFactory.make("nvmultiurisrcbin", "multi-uri-source")

    if not multi_src_bin:
        print('cannot create')
        return


    multi_src_bin.set_property("uri-list", "file:///home/lakshh/Desktop/camera2_20260525_154131.mp4,file:///home/lakshh/Desktop/camera1_trim_20260525_154131.mp4")
    # multi_src_bin.set_property("uri-list", "rtsp://root:root@192.168.6.91/cam1/h264,rtsp://root:root@192.168.6.90/cam1/h264")
    num_sources = 2
    multi_src_bin.set_property("max-batch-size", 10)
    # multi_src_bin.set_property("batch-size", 1)
    # multi_src_bin.set_property("batched-push-timeout", 66666)

    multi_src_bin.set_property("ip-address", "localhost")
    multi_src_bin.set_property("port", 9000)

    # ADD THESE LINES: Define the uniform output resolution for the muxer
    multi_src_bin.set_property("width", 1920)
    multi_src_bin.set_property("height", 1080)

    pipeline.add(multi_src_bin)


    pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
    sgie1 = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine-1")

    queue1 = Gst.ElementFactory.make("queue", "queue1")
    queue2 = Gst.ElementFactory.make("queue", "queue2")
    queue3 = Gst.ElementFactory.make("queue", "queue3")
    queue4 = Gst.ElementFactory.make("queue", "queue4")
    queue5 = Gst.ElementFactory.make("queue", "queue5")
    queue6 = Gst.ElementFactory.make("queue", "queue6")

    nvdslogger = Gst.ElementFactory.make("nvdslogger", "nvdslogger")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "nvvideo-converter-1")
    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "nvvideo-converter-2")
    nvosd = Gst.ElementFactory.make("nvdsosd", "nv-onscreendisplay")

    is_aarch64 = platform.uname().machine == 'aarch64'
    
    if PERF_MODE:
        sink = Gst.ElementFactory.make("fakesink", "nvvideo-renderer")
    else:
        if is_aarch64:
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        else:
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")

    if not (pgie and sgie1 and nvdslogger and tiler and nvvidconv1 and nvvidconv2 and nvosd and sink):
        sys.stderr.write("One element could not be created. Exiting.\n")
        return -1

    # streammux_config = config.get('streammux', {})
    # if 'width' in streammux_config: streammux.set_property('width', streammux_config['width'])
    # if 'height' in streammux_config: streammux.set_property('height', streammux_config['height'])
    # if 'batch-size' in streammux_config: streammux.set_property('batch-size', streammux_config['batch-size'])


    pgie_config = config.get('primary-gie', {})
    pgie_config_path = pgie_config.get('config-file') or pgie_config.get('config-file-path')
    if pgie_config_path:
        pgie.set_property('config-file-path', pgie_config_path)

    sgie1_config = config.get('secondary-gie-1', {})
    sgie1_config_path = sgie1_config.get('config-file') or sgie1_config.get('config-file-path')
    if sgie1_config_path:
        sgie1.set_property('config-file-path', sgie1_config_path)

    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != num_sources:
        sys.stderr.write(f"WARNING: Overriding infer-config batch-size ({pgie_batch_size}) with number of sources ({num_sources})\n")
        pgie.set_property("batch-size", num_sources)
        sgie1.set_property("batch-size", num_sources)

    # tracker_config = config.get('tracker', {})
    # if 'll-config-file' in tracker_config: nvtracker.set_property('ll-config-file', tracker_config['ll-config-file'])
    # if 'll-lib-file' in tracker_config: nvtracker.set_property('ll-lib-file', tracker_config['ll-lib-file'])

    nvosd.set_property("display-text", 1)
    nvosd.set_property("process-mode", 1)

    tiler_rows = int(math.sqrt(num_sources))
    tiler_columns = int(math.ceil(1.0 * num_sources / tiler_rows))
    tiler_rows = 1
    tiler_columns = 2
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    
    tiler_config = config.get('tiler', {})
    if 'width' in tiler_config: tiler.set_property('width', tiler_config['width'])
    if 'height' in tiler_config: tiler.set_property('height', tiler_config['height'])

    # if PERF_MODE:
    #     if is_aarch64:
    #         streammux.set_property("nvbuf-memory-type", 4)
    #     else:
    #         streammux.set_property("nvbuf-memory-type", 2)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pipeline_flow = [nvvidconv1, queue1, pgie, queue2, queue3, sgie1, nvdslogger, tiler, queue4, nvvidconv2, queue5, nvosd, queue6, sink]

    for x in pipeline_flow: pipeline.add(x)

    multi_src_bin.link(pipeline_flow[0])
    # streammux.link(pipeline_flow[0])
    for i, ds_element in enumerate(pipeline_flow):
        if i == len(pipeline_flow) - 1: break
        ds_element.link(pipeline_flow[i+1])


    if True:
        reid_sgie_pad = nvdslogger.get_static_pad("src")
        if not reid_sgie_pad:
            sys.stderr.write("Could not get nvdslogger src pad. Exiting.\n")
            return -1
        reid_sgie_pad.add_probe(Gst.PadProbeType.BUFFER, reid_pad_buffer_probe, 0)

        

    pipeline.set_state(Gst.State.PLAYING)

    _pipeline_ref = pipeline
    try:
        loop.run()
    except BaseException:
        pass

    sys.stdout.write("Returned, stopping playback\n")
    pipeline.set_state(Gst.State.NULL)
    sys.stdout.write("Deleting pipeline\n")

    return 0

if __name__ == '__main__':
    sys.exit(main())