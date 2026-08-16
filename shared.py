import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import gstgva as gva
import cv2
import json
import time
import os

Gst.init(None)

CAMERAS = [
    {"id": "cam0", "location": "rtsp://rtsp-mock:8554/live/webcam"},
    {"id": "cam1", "location": "rtsp://rtsp-mock:8554/live/cam1"},
]

MODEL_PATH = "/workspace/models/yolo26n_int8_openvino_model/yolo26n.xml"
CLIP_DURATION_SEC = 45
FPS = 3


class CameraStream:
    def __init__(self, camera_id, output_dir="/workspace"):
        self.camera_id = camera_id
        self.output_dir = output_dir
        self.output_writer = None
        self.json_file = None
        self.clip_start_time = None
        self.clip_index = 0
        self.frame_count = 0
        os.makedirs(output_dir, exist_ok=True)

    def start_new_clip(self, width, height):
        if self.output_writer is not None:
            self.output_writer.release()
        if self.json_file is not None:
            self.json_file.close()

        video_path = (
            f"{self.output_dir}/{self.camera_id}_clip_{self.clip_index:04d}.mp4"
        )
        json_path = (
            f"{self.output_dir}/{self.camera_id}_clip_{self.clip_index:04d}.jsonl"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.output_writer = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))
        self.json_file = open(json_path, "w")
        self.clip_start_time = time.time()
        self.clip_index += 1
        print(f"[{self.camera_id}] Started new clip: {video_path}")

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        width = caps.get_structure(0).get_value("width")
        height = caps.get_structure(0).get_value("height")

        if (
            self.clip_start_time is None
            or (time.time() - self.clip_start_time) >= CLIP_DURATION_SEC
        ):
            self.start_new_clip(width, height)

        frame = gva.VideoFrame(buf, caps=caps)
        with frame.data() as image:
            image = image.copy()

            if image.std() < 5.0:  # skip flat/invalid warmup frames
                return Gst.FlowReturn.OK

            detections = []
            for roi in frame.regions():
                label, confidence = roi.label(), roi.confidence()
                x, y, w, h = roi.rect()
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    f"{label} {confidence:.2f}",
                    (x, max(y - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                detections.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "box": {"x": x, "y": y, "w": w, "h": h},
                    }
                )

            record = {
                "camera_id": self.camera_id,
                "frame": self.frame_count,
                "timestamp": buf.pts / Gst.SECOND
                if buf.pts != Gst.CLOCK_TIME_NONE
                else None,
                "detections": detections,
            }
            self.json_file.write(json.dumps(record) + "\n")
            self.json_file.flush()

            self.output_writer.write(image)
            self.frame_count += 1

        return Gst.FlowReturn.OK

    def close(self):
        if self.output_writer is not None:
            self.output_writer.release()
        if self.json_file is not None:
            self.json_file.close()


# Build one shared pipeline with N branches, all gvadetect sharing model-instance-id
branch_strs = []
streams = {}

for cam in CAMERAS:
    cam_id = cam["id"]
    streams[cam_id] = CameraStream(cam_id)
    branch = (
        f"rtspsrc location={cam['location']} latency=200 protocols=tcp ! "
        f"rtph264depay ! h264parse ! queue ! avdec_h264 ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"gvadetect model={MODEL_PATH} model-instance-id=shared0 device=CPU ! "
        f"gvametaconvert format=json ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink name=sink_{cam_id} emit-signals=true max-buffers=1 drop=true sync=false"
    )

    branch_strs.append(branch)

pipeline_str = " ".join(
    branch_strs
)  # each branch is independently valid GStreamer syntax, concatenated

pipeline = Gst.parse_launch(pipeline_str)

for cam in CAMERAS:
    cam_id = cam["id"]
    appsink = pipeline.get_by_name(f"sink_{cam_id}")
    if appsink is None:
        raise RuntimeError(f"Could not find appsink for {cam_id}")
    appsink.connect("new-sample", streams[cam_id].on_new_sample)

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
        print(f"GStreamer ERROR: {err}\nDebug: {debug}")
        loop.quit()


bus.connect("message", on_message, loop)

try:
    loop.run()
except KeyboardInterrupt:
    pass
finally:
    for stream in streams.values():
        stream.close()
    pipeline.set_state(Gst.State.NULL)
