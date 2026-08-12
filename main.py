import cv2
import os
import queue

os.environ["GIO_USE_PROXY_RESOLVER"] = "dummy"
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp, GLib

import numpy as np
import openvino as ov

frame_queue = queue.Queue(maxsize=30)

Gst.init(None)

# --- OpenVINO setup ---
core = ov.Core()
model = core.read_model("./models/yolo26n_int8_openvino_model/yolo26n.xml")  # or .onnx
compiled_model = core.compile_model(model, "CPU")  # or "GPU", "AUTO"
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

N, C, H, W = input_layer.shape  # assume NCHW


def preprocess(frame, width, height):
    resized = cv2.resize(frame, (W, H))
    print(f"resized.shape = {resized.shape}, {H}, {W}")  # should print (H, W, 3)
    blob = resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
    return blob


def run_inference(frame, width, height):
    blob = preprocess(frame, width, height)
    result = compiled_model([blob])[output_layer]
    return result


# --- GStreamer pipeline ---
# Uses avdec_h264 as a safe default; swap for nvh264dec / vaapih264dec for HW decode
pipeline_str = (
    "rtspsrc location=rtsp://rtsp-mock:8554/live/stream latency=200 ! "
    "rtph264depay ! h264parse ! avdec_h264 ! "
    "videoconvert ! "
    "gvadetect model=./models/yolo26n_int8_openvino_model/yolo26n.xml device=CPU ! "
    "gvametaconvert format=json ! "
    "gvametapublish method=file file-path=/tmp/detections.json ! "
    "gvawatermark ! "
    "videoconvert ! autovideosink sync=false"
)

try:
    pipeline = Gst.parse_launch(pipeline_str)
except GLib.Error as e:
    print(f"Pipeline parse error: {e}")
    raise
appsink = pipeline.get_by_name("sink")


def parse_detections(result, conf_threshold=0.5, frame_width=640, frame_height=640):
    detections = result[0]  # shape (300, 6)
    boxes = []
    for det in detections:
        label, conf, x_min, y_min, x_max, y_max = det
        if conf < conf_threshold:
            continue
        # coords are normalized 0-1, scale to actual frame size
        boxes.append(
            {
                "label": int(label),
                "confidence": float(conf),
                "box": (
                    int(x_min * frame_width),
                    int(y_min * frame_height),
                    int(x_max * frame_width),
                    int(y_max * frame_height),
                ),
            }
        )
    return boxes


def on_new_sample(sink):
    sample = sink.emit("pull-sample")
    buf = sample.get_buffer()
    caps = sample.get_caps()
    width = caps.get_structure(0).get_value("width")
    height = caps.get_structure(0).get_value("height")

    success, mapinfo = buf.map(Gst.MapFlags.READ)
    if not success:
        return Gst.FlowReturn.ERROR

    frame = np.ndarray((height, width, 3), dtype=np.uint8, buffer=mapinfo.data).copy()
    buf.unmap(mapinfo)

    result = run_inference(frame, width, height)

    # non-blocking push, drop old frame if display thread is behind
    print([r for r in parse_detections(result) if r.get("confidence", 0.0) > 0.5])

    return Gst.FlowReturn.OK


try:
    appsink.connect("new-sample", on_new_sample)
except Exception as e:
    print(e)

pipeline.set_state(Gst.State.PLAYING)

loop = GLib.MainLoop()
bus = pipeline.get_bus()
bus.add_signal_watch()


def on_message(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, {debug}")
        loop.quit()


bus.connect("message", on_message, loop)

try:
    loop.run()
except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
    pipeline.set_state(Gst.State.NULL)
