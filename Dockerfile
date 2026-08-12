FROM intel/dlstreamer:2026.1.0-ubuntu22

# Confirm gvadetect and friends are present at build time
RUN gst-inspect-1.0 gvadetect

WORKDIR /workspace

# Models and model-proc configs are mounted at runtime via docker-compose,
# so you can swap them without rebuilding the image.

CMD ["bash"]
