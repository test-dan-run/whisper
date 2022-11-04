docker run --rm -it -p 8000:8000 --gpus all \
    whisper:1.10.0-cuda11.3-tiny uvicorn app:app --host 0.0.0.0 --port 8000