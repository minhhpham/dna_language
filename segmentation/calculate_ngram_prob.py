import argparse
from typing import Iterator
from tqdm import tqdm
from collections import defaultdict
import subprocess
import pickle
from os import listdir
from os.path import isfile, join
from itertools import islice
from typing import Dict


def count_chars(filepath: str) -> int:
    """count number of chars in a file"""
    out = subprocess.check_output(["wc", "-c", filepath])
    return int(out.split()[0])


def read_fasta(filepath: str) -> Iterator[str]:
    """only reading Primary assembly
    """
    def is_primary(seq_desc: str):
        return True if "primary assembly" in seq_desc.lower() else False
    current_seq = None  # not counting chars if None
    file = open(filepath, "r")
    progress = tqdm(desc="reading fasta file",
                    total=count_chars(filepath), position=0)
    for line in file:
        if line.startswith(">"):
            if current_seq is not None:
                yield current_seq
            if is_primary(line):
                current_seq = ""  # start counting seq chars
            else:
                current_seq = None  # ignore this sequence
        elif current_seq is not None:
            current_seq += line.strip().upper()
        progress.update(len(line))
    file.close()


def split_chunks(data: Dict, size=1000000):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}


def main(infile: str, outfile: str, max_n: int):
    counts = defaultdict(int)

    def process_seq(seq: str, i: int) -> None:
        nonlocal counts, max_n
        for r in tqdm(range(len(seq)),
                      position=1, desc=f"processing seq {i}", leave=False):
            if seq[r] == 'N':
                continue
            for l in range(r, max(r-max_n+1, 0)-1, -1):  # noqa
                if seq[l] == 'N':
                    break
                counts[seq[l:r+1]] += 1

    for i, seq in enumerate(read_fasta(infile)):
        process_seq(seq, i)

    for i, sdict in tqdm(enumerate(split_chunks(counts)), desc="write files"):
        with open(f"counts_pkl/{i}.pkl", "wb") as file:
            pickle.dump(sdict, file)


def load_ngram_counts():
    pkl_files = [
        join("segmentation/counts_pkl", f)
        for f in listdir("segmentation/counts_pkl")
        if isfile(join("segmentation/counts_pkl", f))
    ]
    subdicts = [
        pickle.load(open(filepath, "rb"))
        for filepath in tqdm(pkl_files, desc="reading pkl count files")
    ]
    return {
        k: v for d in tqdm(subdicts, desc="processing") for k, v in d.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input fasta file")
    parser.add_argument("-o", "--output", type=str,
                        help="Output dict count (pickled file)",
                        default="./ngram_count.pkl")
    parser.add_argument("-m", "--max-n", type=int, help="max N in Ngram",
                        default=14)
    args = parser.parse_args()
    main(args.input, args.output, args.max_n)
