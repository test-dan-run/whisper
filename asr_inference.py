import os
import json
import whisper
import argparse
from typing import Optional

def load_model(model_type: Optional[str] = None) -> whisper.Whisper:
    '''
    Takes in the model type and initialises the model object with the pretrained weights
    '''

    # Use the first model found in the cached directory as default
    if model_type is None:
        MODEL_CACHE_DIR = '/root/.cache/whisper/'
        if not os.path.exists(MODEL_CACHE_DIR):
            raise Exception(f'Model cache directory [{MODEL_CACHE_DIR}] does not exist. Are you sure you added a model?')
        else: 
            model_filenames = [fn for fn in os.listdir(MODEL_CACHE_DIR) if '.pt' in fn]
            
            if len(model_filenames) == 0:
                raise Exception(f'Model cache directory is empty. Are you sure you added a model?')
            else:
                model_type = model_filenames[0].replace('.pt', '')
                print(f'No model name specified. Defaulting to model type: {model_type}')

    print(f'Loading model: {model_type}')
    return whisper.load_model(model_type)

def execute_task(model: whisper.Whisper, audio_path: str, task: str = 'transcribe', language: Optional[str] = None) -> str:
    '''
    Executes task (transcribe | translate) on audio file via the initialized model
    Outputs results as string
    '''

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # decode the audio
    options = whisper.DecodingOptions(task=task, language=language)
    decoded_result = whisper.decode(model, mel, options)
    return decoded_result.text

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Run inference with whisper model')
    parser.add_argument('--dataset_dir', type=str, default='/dataset')
    parser.add_argument('--task', type=str, default='transcribe')
    parser.add_argument('--language', type=str, default=None)
    parser.add_argument('--output_path', type=str, default='output/output.json')
    args = parser.parse_args()

    model = load_model()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, mode='w') as fw:

        for filename in os.listdir(args.dataset_dir):
            print(f'Running inference on {filename}')
            file_extension = os.path.splitext(filename)[-1]
            # skip if file not an audio file
            if file_extension not in ['.wav', '.mp3']:
                continue
            
            result_text = execute_task(model, os.path.join(args.dataset_dir, filename), language=args.language)
            fw.write(json.dumps({'filename': filename, 'text': result_text}))