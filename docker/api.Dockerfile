FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow, OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/labels.txt ./models/labels.txt

ENV TRITON_HOST=triton
ENV TRITON_GRPC_PORT=8001
ENV MODEL_NAME=resnet50_onnx

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
