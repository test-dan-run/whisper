'''
Main file for api service
uvicorn app:app --host 0.0.0.0 --port 8000
'''

import os
from typing import Optional, Dict
from fastapi import FastAPI, File, Depends
from pydantic import BaseModel
from asr_inference import load_model, execute_task

# load the model
model = load_model()
app = FastAPI()

class TranscribeMetadata(BaseModel):
    filename: str
    language: Optional[str] = 'English'

class TranslateMetadata(BaseModel):
    filename: str 
    language: Optional[str] = 'English'

def post_to_task(metadata: Dict, file: bytes = File(), task: str = 'transcribe'):

    ext = os.path.splitext(metadata['filename'])[-1]
    tmp_audio_filepath = f'./tmp/audio{ext}'

    with open(tmp_audio_filepath, 'wb') as f:
        f.write(file)    

    result_text = execute_task(model, tmp_audio_filepath, task=task, language=metadata['language'])
    os.remove(tmp_audio_filepath)

    return result_text    

@app.post('/translate')
def translate(metadata: TranslateMetadata = Depends(), file: bytes = File()):
    
    metadata = metadata.dict()
    result_text = post_to_task(metadata, file, task='translate')

    return {'filename': metadata['filename'], 'text': result_text}

@app.post('/transcribe')
def transcribe(metadata: TranscribeMetadata = Depends(), file: bytes = File()):

    metadata = metadata.dict()
    result_text = post_to_task(metadata, file, task='transcribe')

    return {'filename': metadata['filename'], 'text': result_text}

if __name__ == '__main__':
    pass
