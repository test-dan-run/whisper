import os
import json
import tqdm
import argparse
from typing import Any, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader



import whisper

class LIDInferenceDataset(Dataset):
    def __init__(self, manifest_path: str, device: str = 'cuda'):
        super(LIDInferenceDataset, self).__init__()

        self.device = device

        with open(manifest_path, mode='r', encoding='utf-8') as fr:
            lines = fr.readlines()
        base_dir = os.path.dirname(manifest_path)

        self.items = [json.loads(line.strip('\r\n')) for line in lines]
        self.paths = [os.path.join(base_dir, item['audio_filepath']) for item in self.items]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        path = self.paths[idx]
        audio = whisper.load_audio(path)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        return mel

def detect_language_from_manifest(manifest_path: str, model: Any, batch_size: int = 1, num_workers: int = 0) -> List[Dict[str, str]]:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = LIDInferenceDataset(manifest_path, device)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    langs = []
    for idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        with torch.no_grad():
            _, probs = model.detect_language(batch)
        batch_langs = [max(prob, key=prob.get) for prob in probs]
        langs.extend(batch_langs)

    for item, pred in zip(dataset.items, langs):
        item['language_pred'] = pred
    
    return dataset.items
    
def save_manifest(items: List[Dict[str, str]], save_path: str = 'pred_manifest.json') -> None:
    with open(save_path, 'w', encoding='utf-8') as fw:
        for item in items:
            fw.write(json.dumps(item)+'\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run filewise language detection using Whisper.')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--manifest', type=str, required=True, help='Path to manifest file')
    args = parser.parse_args()

    model = whisper.load_model(args.model, device='cuda')
    pred_items = detect_language_from_manifest(args.manifest, model)
    save_manifest(pred_items)
