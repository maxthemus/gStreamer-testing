```bash
sudo apt update
sudo apt install -y \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav \
  python3-gi \
  gir1.2-gst-plugins-base-1.0
```


Streaming webcam
```bash
```bash
ffmpeg -f v4l2 -framerate 3 -video_size 1280x720 -i /dev/video0 -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp -rtsp_transport tcp rtsp://localhost:8554/live/webcam
```
```



```bash
ffmpeg -f lavfi -i testsrc2=size=1280x720:rate=10 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -g 10 -keyint_min 10 -sc_threshold 0 \
  -f rtsp -rtsp_transport tcp rtsp://localhost:8554/live/cam1
```
  ```
```
