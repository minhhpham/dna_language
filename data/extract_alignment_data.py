import re
import pysam
import subprocess
from skbio.alignment import StripedSmithWaterman
from tqdm.auto import tqdm
from typing import Iterator, Tuple, List
import multiprocessing as mp
from pandas import DataFrame
import sqlite3
from sqlite3 import Connection
import argparse
from segmentation.seq_segment import ProbabilitySegmentor


DB_NAME = "database.sqlite"


def get_reference(ref_path: str, contig: str, pos: int, length: int) -> str:
    """return the reference sequence from REF_FILE
    Args:
        contig (str): contig name
        pos (int): position on contig
        length (int): length of output sequence
    """
    output = subprocess.check_output([
        "samtools", "faidx", ref_path,
        f"{contig}:{pos}-{pos+length-1}"
    ])
    seqs = output.decode().strip().split('\n')
    out = ''.join(seqs[1:]).upper()
    assert len(out) == length, \
        f"\nreference sequence error at {contig}:{pos}-{pos+length-1}"
    return out


def batchify_sam(sam_file, batch_size=10000) -> Iterator[List[any]]:
    """yield batches of alignments
    only yield mapped alignments
    """
    cache = []
    for alignment in sam_file.fetch():
        if alignment.is_unmapped:
            continue
        cache.append(alignment)
        if len(cache) == batch_size:
            yield cache
            cache = []
    if len(cache) > 0:
        yield cache


def alignment_kernel(args) -> Tuple[str, str]:
    """input: alignment info and reference path
    output: aligned reference and read sequences
    return None if cannot extract information (ignore errors)
    """
    ref_path, ref_contig, ref_pos, ref_length, query = args
    try:
        ref_seq = get_reference(
            ref_path=ref_path,
            contig=ref_contig,
            pos=ref_pos,
            length=ref_length
        )
    except Exception as e:
        print(e)
        return None
    query_seq = query
    # get their exact alignment again
    SW_ref = StripedSmithWaterman(
        query_sequence=ref_seq,
        match_score=1,
        mismatch_score=-4,
        gap_open_penalty=6,
        gap_extend_penalty=1,
    )
    sw_result = SW_ref(query_seq)
    return (
        sw_result.aligned_target_sequence,
        sw_result.aligned_query_sequence
    )


def get_aligned_seqs(
        ref_path: str, sam_path: str) -> Iterator[Tuple[str, str]]:
    """get aligned pairs of reference and read sequences

    Args:
        sam_path (str): path to sam file
        ref_path (str): path to reference file
    """
    sam_file = pysam.AlignmentFile(sam_path, "r")
    pool = mp.get_context('spawn').Pool()
    for batch in batchify_sam(sam_file):
        args = [(ref_path, alignment.reference_name, alignment.reference_start,
                 alignment.reference_length, alignment.query)
                for alignment in batch]
        batch_out = list(tqdm(
            pool.map(alignment_kernel, args),
            desc="extract reference",
            total=len(args),
            position=1,
            leave=True,
        ))
        batch_out = [aln for aln in batch_out if aln is not None]
        for out in batch_out:
            yield out
    pool.close()
    sam_file.close()


def segment_kernel(seq: str, segmentor: ProbabilitySegmentor) -> str:
    """split a sequence and return split indices as comma-seperated strings"""
    # find positions of insert (- character)
    # and shifts all split index after them by 1
    try:
        specials = ['-', 'n', 'N']
        ins_idx = [i for i, char in enumerate(seq) if char in specials]
        seq_clean = re.sub('|'.join(specials), '', seq)
        split_idx = segmentor.segment(seq_clean).split_idx
        # now shift index according to ins_idx
        for pos in ins_idx:
            split_idx = [i+1 if i >= pos else i
                         for i in split_idx]
        assert split_idx[-1] < len(seq)
    except Exception as e:
        print(e)
        print(seq_clean)
        exit(1)
    return ','.join([str(i) for i in split_idx])


def save_to_db(
        alignments: List[Tuple[str, str, str]],
        conn: Connection,
        segmentor: ProbabilitySegmentor):
    """alignments: tuples of SRA, reference, read
    split references
    and save records of (SRA, reference, read, split_indices)
    """
    refs = [a[1] for a in alignments]
    # with mp.Pool() as pool:
    #     split_idx = list(tqdm(
    #         pool.imap(segment_kernel, refs),
    #         desc="Splitting",
    #         total=len(refs),
    #         position=1,
    #         leave=True,
    #     ))
    split_idx = [segment_kernel(ref, segmentor)
                 for ref in tqdm(
                     refs, desc="splitting", position=1, leave=True)
                 ]
    data = [(*seqs, split) for seqs, split in zip(alignments, split_idx)]
    data = DataFrame(data, columns=["SRA", "reference", "read", "split_idx"])
    data.to_sql("alignments", conn, if_exists="append", index=False)
    print(f"\nsaved {len(alignments)} alignments to database")


def main(ref_path: str, sam_path: str):
    """from reference path and sam path, extract aligned sequences
    then save them to a sqlite database
    database table has 3 fields: dataset's SRA, reference, read
    """
    cache = []
    cache_size = 10000
    conn = sqlite3.connect(DB_NAME)
    print("intializing Probability Segmentor ...")
    segmentor = ProbabilitySegmentor()

    # get SRA
    aln0 = next(iter(pysam.AlignmentFile(sam_path, "r")))
    SRA = aln0.query_name.split(".")[0]
    progress = tqdm(desc="extracting alignments", position=0)
    for ref, read in get_aligned_seqs(ref_path, sam_path):
        cache.append((SRA, ref, read))
        progress.update(1)
        if len(cache) >= cache_size:
            save_to_db(cache, conn, segmentor)
            cache = []

    if len(cache) > 0:
        save_to_db(cache, conn, segmentor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reference", required=True,
                        help="path to reference file")
    parser.add_argument("-s", "--sam", required=True, help="path to SAM file")
    args = parser.parse_args()
    main(args.reference, args.sam)
