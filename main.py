import time
import json
import cv2
import os
import queue

os.environ["GIO_USE_PROXY_RESOLVER"] = "dummy"
import gi
import gstgva as gva

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp, GLib

import numpy as np

frame_queue = queue.Queue(maxsize=30)

file = open("/tmp/python-detections.json", "w")

Gst.init(None)

CLIP_DURATION_SEC = 45
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 10  # match your negotiated caps framerate

output_writer = None
json_file = None
clip_start_time = None
clip_index = 0
FRAME_COUNT = 0


def get_writer():
    global output_writer
    if output_writer is None:
        print("CREATING WRITEER")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_writer = cv2.VideoWriter(
            "/workspace/output_annotated.mp4", fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT)
        )
        if not output_writer.isOpened():
            raise RuntimeError("Video writer failed")
    return output_writer


# --- GStreamer pipeline ---
# Uses avdec_h264 as a safe default; swap for nvh264dec / vaapih264dec for HW decode
pipeline_str = (
    "rtspsrc location=rtsp://rtsp-mock:8554/live/webcam latency=200 ! "
    "rtph264depay ! h264parse ! avdec_h264 ! "
    "videoconvert ! "
    "gvadetect model=/workspace/models/yolo26n_int8_openvino_model/yolo26n.xml device=CPU ! "
    "gvametaconvert format=json ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    # "gvametapublish method=file file-path=/tmp/detections.json ! "
    "queue max-size-buffers=5 leaky=downstream ! "
    "appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
    # "fakesink"
)

try:
    pipeline = Gst.parse_launch(pipeline_str)
except GLib.Error as e:
    print(f"Pipeline parse error: {e}")
    raise
appsink = pipeline.get_by_name("sink")


def start_new_clip():
    global output_writer, json_file, clip_start_time, clip_index

    if output_writer is not None:
        output_writer.release()
    if json_file is not None:
        json_file.close()

    os.makedirs("/workspace/output", exist_ok=True)

    video_path = f"/workspace/clip_{clip_index:04d}.mp4"
    json_path = f"/workspace/clip_{clip_index:04d}.jsonl"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_writer = cv2.VideoWriter(
        video_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT)
    )
    if not output_writer.isOpened():
        raise RuntimeError(f"Failed to open writer for {video_path}")

    json_file = open(json_path, "w")

    clip_start_time = time.time()
    clip_index += 1
    print(f"Started new clip: {video_path}")


def on_new_sample(sink):
    global FRAME_COUNT, clip_start_time

    sample = sink.emit("pull-sample")
    buf = sample.get_buffer()
    caps = sample.get_caps()

    # rotate to a new clip if we've hit the duration limit
    if clip_start_time is None or (time.time() - clip_start_time) >= CLIP_DURATION_SEC:
        start_new_clip()

    frame = gva.VideoFrame(buf, caps=caps)

    with frame.data() as image:
        image = image.copy()

        detections = []
        for roi in frame.regions():
            label = roi.label()
            confidence = roi.confidence()
            x, y, w, h = roi.rect()

            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "box": {"x": x, "y": y, "w": w, "h": h},
                }
            )

        record = {
            "frame": FRAME_COUNT,
            "timestamp": buf.pts / Gst.SECOND
            if buf.pts != Gst.CLOCK_TIME_NONE
            else None,
            "detections": detections,
        }
        json_file.write(json.dumps(record) + "\n")
        json_file.flush()

        output_writer.write(image)
        FRAME_COUNT += 1

    return Gst.FlowReturn.OK


appsink.connect("new-sample", on_new_sample)

pipeline.set_state(Gst.State.PLAYING)

loop = GLib.MainLoop()
bus = pipeline.get_bus()
bus.add_signal_watch()


def on_message(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End of stream!")
        loop.quit()
    if t == Gst.MessageType.TAG:
        tag_list = message.parse_tag()
        print(f"--- Metadata tags found from source: {message.src.get_name()} ---")
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, {debug}")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        warn, debug = message.parse_warning()
        print(f"GStreamer WARNING: {warn}\nDebug: {debug}")


bus.connect("message", on_message, loop)

try:
    loop.run()
except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
    file.close()

    if output_writer is not None:
        print("release")
        output_writer.release()
    if json_file is not None:
        json_file.close()
    pipeline.set_state(Gst.State.NULL)
