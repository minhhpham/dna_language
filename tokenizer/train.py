# modify PYTHONPATH to execute this script in a subdir
import sys
import os
import argparse
from tqdm.auto import tqdm
from collections import defaultdict
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from models import BertWordPieceTokenizer  # noqa


def train_bert_tokenizer(data_file: str, vocab_size: int, outdir: str):
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train(
        files=[data_file],
        vocab_size=vocab_size,
        wordpieces_prefix='##',
    )
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    tokenizer.save_model(outdir)


def train_word_tokenizer(data_file: str, vocab_size: int, outdir: str):
    file = open(data_file, "r")
    lines = [line for line in
             tqdm(file, desc="scanning corpus file")]
    file.close()
    counts = defaultdict(int)
    for line in tqdm(lines, desc="counting words"):
        words = line.strip().split(" ")
        for word in words:
            counts[word] += 1
    print("sorting by frequency")
    counts = [(k, v) for k, v in counts.items()]
    counts.sort(key=lambda x: x[1], reverse=True)
    counts = counts[:vocab_size]
    print("writing output")
    outfile = open(f"{outdir}/vocab.txt", "w")
    for i, (word, counts) in tqdm(enumerate(counts)):
        print(word, file=outfile,
              end="" if i == len(counts)-1 else "\n")
    outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train a wordpiece/word tokenizer from a text corpus"
    )
    subparsers = parser.add_argument(
        "-t", "--type",
        choices=["word", "wordpiece"],
        help="tokenizer type"
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
        default=None
    )
    args = parser.parse_args()
    # set outdir default
    if args.outdir is None:
        args.outdir = f"saved_model/dna_tokenizer_{args.type}"
    if args.type == "word":
        train_word_tokenizer(args.data_file, args.vocab_size, args.outdir)
    else:
        train_bert_tokenizer(args.data_file, args.vocab_size, args.outdir)
