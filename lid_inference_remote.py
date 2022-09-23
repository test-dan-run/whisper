from clearml import Task, Dataset
task = Task.init(project_name='LID/Whisper', task_name='MMS')
task.set_base_docker('dleongsh/whisper:1.10.0-cuda11.3-large')
task.execute_remotely()

import os
import json
import tqdm
import whisper
import argparse
from typing import Any, List, Dict

def detect_language_from_path(path: str, model: Any) -> str:

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    lang = max(probs, key=probs.get)

    print(f"[{path}] Detected language: {lang}")

    return lang

def detect_language_from_manifest(manifest_path: str, model: Any) -> List[Dict[str, str]]:

    with open(manifest_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    items = [json.loads(line.strip('\r\n')) for line in lines]
    base_dir = os.path.dirname(manifest_path)

    for item in tqdm.tqdm(items):
        audio_filepath = os.path.join(base_dir, item['audio_filepath'])
        lang = detect_language_from_path(audio_filepath, model)
        item['language_pred'] = lang

    return items
    
def save_manifest(items: List[Dict[str, str]], save_path: str = 'pred_manifest.json') -> None:
    with open(save_path, 'w', encoding='utf-8') as fw:
        for item in items:
            fw.write(json.dumps(item)+'\n')
    return save_path

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run filewise language detection using Whisper')
    parser.add_argument('--dataset_id', type=str, required=True, help='ClearML Dataset ID')
    parser.add_argument('--manifest_name', type=str, required=True, help='Name of manifest file')
    args = parser.parse_args()

    dataset = Dataset.get(dataset_id=args.dataset_id)
    dataset_path = dataset.get_local_copy()
    manifest_path = os.path.join(dataset_path, args.manifest_name)

    model = whisper.load_model('large')
    pred_items = detect_language_from_manifest(manifest_path, model)
    save_path = save_manifest(pred_items)

    task.upload_artifact(save_path, save_path)
