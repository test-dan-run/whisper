import argparse
from whisper import _download, _MODELS

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download model.')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')

    args = parser.parse_args()

    _download(_MODELS[args.model], args.output_dir, in_memory=False)
