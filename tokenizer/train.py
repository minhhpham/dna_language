# modify PYTHONPATH to execute this script in a subdir
import sys
import os
import argparse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from models import BertWordPieceTokenizer  # noqa


def train(data_file: str, vocab_size: int, outdir: str):
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train(
        files=[data_file],
        vocab_size=vocab_size,
        wordpieces_prefix='##',
    )
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    tokenizer.save_model(outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train tokenizer from a text corpus"
    )
    parser.add_argument(
        "-d", "--data_file",
        type=str,
        help="path to corpus text file",
        required=True
    )
    parser.add_argument(
        "-s", "--vocab_size",
        type=int,
        help="vocabulary size",
        default=10000
    )
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        help="output directory",
        default="saved_models/dna_tokenizer"
    )
    args = parser.parse_args()
    train(args.data_file, args.vocab_size, args.outdir)
