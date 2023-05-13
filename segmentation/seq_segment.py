from .calculate_ngram_prob import load_ngram_counts
from typing import List
from tqdm import tqdm
from dataclasses import dataclass
MAXGRAM = 14


@dataclass
class Segmented:
    score: float
    split_idx: List[int]
    segments: List[str]


class ProbabilitySegmentor:

    def __init__(self) -> None:
        counts = load_ngram_counts()
        # normalize to probability
        total = sum(counts.values()) / 10
        self.prob = {
            k: v/total
            for k, v in tqdm(counts.items(), desc="normalize probabilities")
        }

    @staticmethod
    def split_seq(seq: str, split_idx: List[int]) -> List[str]:
        return [seq[i:j] for i, j in zip((0, *split_idx), (*split_idx, None))]

    def segment(self, s: str):
        """return segmented strings and score
        """
        prob = self.prob
        result = {}  # (i, j) -> (best score, best segmentation)
        # tri loop:
        # sl: segment length
        # i, j: start and end of segment
        # k: split position
        for sl in range(1, len(s)+1):
            for i in range(0, len(s)-sl+1):
                # try all ways to segment s[i:j]
                j = i + sl
                best_scr = 0
                best_seg = None
                for k in range(i+1, j):
                    scr_k = result[(i, k)][0] * result[(k, j)][0]
                    if scr_k > best_scr:
                        best_scr = scr_k
                        seg_left = result[(i, k)][1]
                        seg_right = result[(k, j)][1]
                        best_seg = (*seg_left, k, *seg_right)
                # compare with whole string (no segment)
                if sl <= MAXGRAM and s[i:j] in prob:
                    scr_noseg = prob[s[i:j]]
                    if scr_noseg >= best_scr:
                        best_scr = scr_noseg
                        best_seg = ()  # () mean no segmentation
                result[(i, j)] = (best_scr, best_seg)
        score, splits = result[(i, len(s))]
        segments = self.split_seq(s, splits)
        return Segmented(
            score=score,
            split_idx=list(splits),
            segments=segments
        )
