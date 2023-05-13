from typing import List, Tuple
import argparse
import sqlite3
from tqdm.auto import tqdm


def query_db(limit=1e7, SRA: str = None) -> List[Tuple[str, str]]:
    """
    return list of tuples (reference string, split_idx string)
    """
    conn = sqlite3.connect("./database.sqlite")
    cursor = conn.cursor()
    query = f"""
        SELECT reference, split_idx
        FROM alignments
        {f"WHERE SRA = '{SRA}'" if SRA else ""}
        LIMIT {limit}
        """
    cursor.execute(query)
    return cursor.fetchall()


def main(outfilepath: str, limit=1e7, SRA: str = None):
    """get reference data from database and sasve into a corpus text file

    Args:
        outfilepath: output file path
        SRA (str, optional): filter by SRA, if none, take all SRA
        limit: number of samples to draw from database
    """
    data = query_db(limit)
    outfile = open(outfilepath, "w")
    for seq, split_idx in tqdm(data):
        split_idx = [int(i) for i in split_idx.split(',')]
        segments = [seq[i:j]
                    for i, j in zip((0, *split_idx), (*split_idx, None))]
        print(' '.join(segments), file=outfile)

    outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str,
                        required=True, help="output file path")
    parser.add_argument("-s", "--sra", type=str,
                        default=None,
                        help="select an SRA only")
    parser.add_argument("-m", "--max", type=int, default=1e7,
                        help="maximum number of samples in the corpus")
    args = parser.parse_args()
    main(outfilepath=args.output, limit=args.max)
