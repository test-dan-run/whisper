import os
import json
import tqdm
import argparse
from typing import Any, List, Dict

import torch
from torch.utils.data import DataLoader

import whisper

from dataset import WhisperInferenceDataset

def transcribe_from_manifest(manifest_path: str, model: Any, options: Any, batch_size: int = 1, num_workers: int = 0) -> List[Dict[str, str]]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WhisperInferenceDataset(manifest_path)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    preds = []
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        with torch.no_grad():
            batch = batch.to(device)
            decoding_results = whisper.decode(model, batch, options)
        texts = [res.text for res in decoding_results]

        # tensors are in decoding_results, DELETE THEM TO SAVE MEMORY!!!
        del decoding_results
        preds.extend(texts)

    for item, pred in zip(dataset.items, preds):
        item['pred_text'] = pred

    return dataset.items

def save_manifest(items: List[Dict[str, str]], save_path: str = 'pred_manifest.json') -> None:
    with open(save_path, 'w', encoding='utf-8') as fw:
        for item in items:
            fw.write(json.dumps(item)+'\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run ASR using Whisper.')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--language', type=str, default=None, help='Language to transcribe to')
    parser.add_argument('--manifest', type=str, required=True, help='Path to manifest file')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn', force=True)

    model = whisper.load_model(args.model, device='cuda')
    options = whisper.DecodingOptions(language = args.language)

    pred_items = transcribe_from_manifest(args.manifest, model, options, batch_size=args.batch_size, num_workers=args.num_workers)
    save_manifest(pred_items)
