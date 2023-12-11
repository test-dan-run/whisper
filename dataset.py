import os
import json
import torch
from torch.utils.data import Dataset

import whisper

class WhisperInferenceDataset(Dataset):
    def __init__(self, manifest_path: str):
        super().__init__()

        base_dir = os.path.dirname(manifest_path)
        with open(manifest_path, mode='r', encoding='utf-8') as fr:
            self.items = [json.loads(line.strip('\r\n')) for line in fr.readlines()]

        self.paths = [os.path.join(base_dir, item['audio_filepath']) for item in self.items]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.paths[idx]
        audio = whisper.load_audio(path)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio)

        return mel
