import os
import argparse
from whisper import available_models, _download, _MODELS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download specified model type')
    parser.add_argument('model_type', type=str, help='Model type')
    args = parser.parse_args()

    models = available_models()
    if args.model_type in models:

        download_root = os.getenv(
            "XDG_CACHE_HOME", 
            os.path.join(os.path.expanduser("~"), ".cache", "whisper")
            )

        _download(url=_MODELS[args.model_type], root=download_root, in_memory=False)

    else:
        raise Exception('Model type does not exist. Please refer to the README for available model types.')