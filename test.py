import cv2
import numpy as np

width, height = 640, 480
fps = 25
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter("/workspace/test_fake.mp4", fourcc, fps, (width, height))
print("Writer opened:", writer.isOpened())

for i in range(50):
    frame = np.full((height, width, 3), (i * 5) % 255, dtype=np.uint8)
    cv2.putText(
        frame, f"Frame {i}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3
    )
    writer.write(frame)

writer.release()
print("Done writing test_fake.mp4")
