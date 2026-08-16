import cv2
import json
import argparse
import os


def load_detections(json_path):
    """Load newline-delimited JSON detections, indexed by frame number."""
    detections_by_frame = {}
    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            detections_by_frame[record["frame"]] = record["detections"]
    return detections_by_frame


def draw_detections(image, detections, conf_threshold=0.0):
    for det in detections:
        if det["confidence"] < conf_threshold:
            continue
        x, y, w, h = det["box"]["x"], det["box"]["y"], det["box"]["w"], det["box"]["h"]
        label = det["label"]
        confidence = det["confidence"]

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{label} {confidence:.2f}"
        cv2.putText(
            image,
            text,
            (x, max(y - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return image


def main():
    parser = argparse.ArgumentParser(
        description="Replay a clip with detection boxes overlaid"
    )
    parser.add_argument("video", help="Path to the video clip (.mp4)")
    parser.add_argument("json", help="Path to the matching detections file (.jsonl)")
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Confidence threshold to display"
    )
    parser.add_argument(
        "--save",
        help="Optional: save annotated output to this path instead of just displaying",
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.exists(args.json):
        raise FileNotFoundError(f"Detections file not found: {args.json}")

    detections_by_frame = load_detections(args.json)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 5
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay_ms = max(int(1000 / fps), 1)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open output writer: {args.save}")

    frame_idx = 0
    paused = False

    print("Controls: [space] pause/resume, [q] quit")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break

            detections = detections_by_frame.get(frame_idx, [])
            annotated = draw_detections(frame, detections, conf_threshold=args.conf)

            if writer is not None:
                writer.write(annotated)
            else:
                cv2.imshow("Replay", annotated)

            frame_idx += 1

        if writer is None:
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated output to {args.save}")
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
