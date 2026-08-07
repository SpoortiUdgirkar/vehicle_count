FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-create output/upload directories
RUN mkdir -p uploads outputs

# HuggingFace Spaces uses port 7860 by default
# Our app reads PORT from env so this works automatically
EXPOSE 7860

ENV KMP_DUPLICATE_LIB_OK=True
ENV PORT=7860

CMD ["python", "vehicle_app.py"]
