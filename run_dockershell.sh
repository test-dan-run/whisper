docker run --rm -it -p 8000:8000 --gpus all \
    -v $PWD/dataset:/dataset \
    -v $PWD/output:/output \
    -v $PWD:/whisper \
    whisper:1.10.0-cuda11.3-tiny \
    python asr_inference.py --dataset_dir /dataset --task transcribe --language Indonesian --output_path /output/output.json