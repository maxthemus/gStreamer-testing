import cv2
import json
import os
import time
import threading
from deep_sort_realtime.deepsort_tracker import DeepSort


import numpy as np


import openvino as ov


CAMERAS = [
    {"id": "cam0", "location": "rtsp://localhost:8554/live/webcam"},
    {"id": "cam1", "location": "rtsp://localhost:8554/live/cam1"},
]

MODEL_PATH = "models/yolo26n_int8_openvino_model/yolo26n.xml"

CLIP_DURATION_SEC = 45
OUTPUT_FPS = 3


class CameraStream:
    def __init__(self, camera_id, rtsp_url, infer_request, model_input_size):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.infer_request = infer_request
        self.input_width, self.input_height = model_input_size
        self.tracker = DeepSort(
            max_age=15,
            n_init=3,
            max_iou_distance=0.7,
            embedder="mobilenet",
            embedder_gpu=False,
        )

        self.capture = None
        self.writer = None
        self.json_file = None

        self.clip_start_time = None
        self.clip_index = 0
        self.frame_count = 0

        self.last_output_time = 0.0

        os.makedirs("output/", exist_ok=True)

    def start(self):
        self.capture = cv2.VideoCapture(self.rtsp_url)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"[{self.camera_id}] Failed to open RTSP stream: {self.rtsp_url}"
            )

        print(f"[{self.camera_id}] Connected")

    def start_new_clip(self, width, height):
        self.close_clip()

        video_path = f"./{self.camera_id}_clip_{self.clip_index:04d}.mp4"

        json_path = f"./{self.camera_id}_clip_{self.clip_index:04d}.jsonl"

        # mp4v is widely available through OpenCV.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.writer = cv2.VideoWriter(
            video_path,
            fourcc,
            OUTPUT_FPS,
            (width, height),
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"[{self.camera_id}] Failed to open video writer")

        self.json_file = open(json_path, "w")

        self.clip_start_time = time.time()
        self.clip_index += 1

        print(f"[{self.camera_id}] Started clip: {video_path}")

    def infer(self, frame):
        """
        Run OpenVINO inference.

        This assumes the model expects:
            NCHW
            RGB
            float32

        The exact preprocessing/output parsing depends on
        the exported YOLO OpenVINO model.
        """

        resized = cv2.resize(
            frame,
            (self.input_width, self.input_height),
        )

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # HWC -> CHW
        input_tensor = rgb.transpose(2, 0, 1)

        # Add batch dimension
        input_tensor = input_tensor[None]

        # YOLO models exported by Ultralytics commonly expect float32.
        input_tensor = input_tensor.astype("float32") / 255.0

        self.infer_request.infer({0: input_tensor})

        outputs = self.infer_request.results

        return outputs

    def track(self, original_bgr_frame, outputs):
        # 2. Extract raw array from OpenVINO output structure
        # OpenVINO outputs usually map the output node to a NumPy array
        output_node = list(outputs.keys())[0]
        raw_output = outputs[output_node]  # Shape is typically (1, 84, 8400)

        # 3. Transpose output to make processing easier: shape becomes (8400, 84)
        # Rows = 8400 bounding box candidates; Columns = [x_center, y_center, width, height, class0_conf, class1_conf, ...]
        predictions = np.squeeze(raw_output).T

        # Track configurations
        CONF_THRESHOLD = 0.4
        deepsort_detections = []

        # Assuming original image dimensions for scaling back bounding boxes
        img_h, img_w = (
            1080,
            1920,
        )  # Replace with your actual camera frame/video frame dimensions
        input_w, input_h = 640, 640  # Your YOLO model input shape

        # 4. Parse Raw YOLO Detections
        for pred in predictions:
            # Elements 0 to 3 are bounding box geometry
            # Elements 4 onwards are individual class confidence scores
            cx, cy, w, h = pred[0:4]
            class_scores = pred[4:]
            #
            # Get highest confidence class
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence >= CONF_THRESHOLD:
                # Scale bounding box coordinates back to original frame dimensions
                x1 = int((cx - w / 2) * (img_w / input_w))
                y1 = int((cy - h / 2) * (img_h / input_h))
                x2 = int((cx + w / 2) * (img_w / input_w))
                y2 = int((cy + h / 2) * (img_h / input_h))
                #
                # DeepSORT expects [left, top, width, height] format
                box_w = x2 - x1
                box_h = y2 - y1
                #
                # Format detection details for DeepSORT
                # Format: ([left, top, w, h], confidence, class_name_or_id)
                deepsort_detections.append(
                    ([x1, y1, box_w, box_h], confidence, str(class_id))
                )

        # 5. Push formatted detections into DeepSORT
        # Always pass the original raw BGR image frame so DeepSORT can extract appearance descriptors
        tracks = self.tracker.update_tracks(
            deepsort_detections, frame=original_bgr_frame
        )

        # 6. Use the Tracking Outputs
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = (
                track.to_ltrb()
            )  # [left, top, right, bottom] bounding box coordinates
            print(f"Object ID {track_id} is located at {ltrb}")

    def process_detections(self, frame, outputs):
        """
        Convert OpenVINO output into detections.

        IMPORTANT:
        The exact output format depends on the YOLO model.
        This function needs to match your yolo26n export.
        """

        detections = []

        # Placeholder for model-specific decoding.
        #
        # For example, once the output tensor is inspected:
        #
        # output = outputs[0]
        #
        # for detection in output:
        #     ...
        #
        # detections.append({
        #     "label": ...,
        #     "confidence": ...,
        #     "box": ...
        # })

        return detections

    def draw_detections(self, frame, detections):
        for detection in detections:
            x = detection["box"]["x"]
            y = detection["box"]["y"]
            w = detection["box"]["w"]
            h = detection["box"]["h"]

            label = detection["label"]
            confidence = detection["confidence"]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x, max(y - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    def process(self):
        self.start()

        while True:
            success, frame = self.capture.read()

            if not success:
                print(f"[{self.camera_id}] Failed to read frame")
                time.sleep(0.1)
                continue

            # OpenCV will potentially receive frames at 25/30 FPS.
            # We only process/write 3 FPS.
            now = time.time()

            if now - self.last_output_time < 1.0 / OUTPUT_FPS:
                continue

            self.last_output_time = now

            # Skip flat/invalid warmup frames.
            if frame.std() < 5.0:
                continue

            height, width = frame.shape[:2]

            if (
                self.clip_start_time is None
                or now - self.clip_start_time >= CLIP_DURATION_SEC
            ):
                self.start_new_clip(width, height)

            outputs = self.infer(frame)

            self.track(frame, outputs)

            detections = self.process_detections(
                frame,
                outputs,
            )

            record = {
                "camera_id": self.camera_id,
                "frame": self.frame_count,
                "timestamp": now,
                "detections": detections,
            }

            self.json_file.write(json.dumps(record) + "\n")
            self.json_file.flush()

            self.writer.write(frame)

            self.frame_count += 1

    def close_clip(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        if self.json_file is not None:
            self.json_file.close()
            self.json_file = None

    def close(self):
        self.close_clip()

        if self.capture is not None:
            self.capture.release()
            self.capture = None


def main():
    core = ov.Core()

    model = core.read_model(MODEL_PATH)

    compiled_model = core.compile_model(
        model,
        "CPU",
    )

    for output in compiled_model.outputs:
        # print("name:", output.get_any_name())
        print("shape:", output.get_partial_shape())
        print("type:", output.get_element_type())

    # One InferRequest per camera.
    infer_requests = [compiled_model.create_infer_request() for _ in CAMERAS]

    input_port = compiled_model.input()

    shape = input_port.get_partial_shape()

    print("Model input shape:", shape)

    input_height = shape[2].get_length()
    input_width = shape[3].get_length()

    streams = []

    for camera, infer_request in zip(
        CAMERAS,
        infer_requests,
    ):
        stream = CameraStream(
            camera_id=camera["id"],
            rtsp_url=camera["location"],
            infer_request=infer_request,
            model_input_size=(
                input_width,
                input_height,
            ),
        )

        streams.append(stream)

    threads = []

    for stream in streams:
        thread = threading.Thread(
            target=stream.process,
            daemon=True,
        )

        thread.start()
        threads.append(thread)

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
