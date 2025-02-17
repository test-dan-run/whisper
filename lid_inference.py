import os
import json
import tqdm
import argparse
from typing import Any, List, Dict

import torch
from torch.utils.data import DataLoader

import whisper

from .dataset import WhisperInferenceDataset

def detect_language_from_manifest(manifest_path: str, model: Any, batch_size: int = 1, num_workers: int = 0) -> List[Dict[str, str]]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WhisperInferenceDataset(manifest_path)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    langs = []
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        with torch.no_grad():
            batch = batch.to(device)
            _, probs = model.detect_language(batch)

        batch_langs = [max(prob, key=prob.get) for prob in probs]

        # del probs to save MEMORY!!
        del probs
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
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn', force=True)

    model = whisper.load_model(args.model, device='cuda')
    pred_items = detect_language_from_manifest(args.manifest, model, batch_size=args.batch_size, num_workers=args.num_workers)
    save_manifest(pred_items)
