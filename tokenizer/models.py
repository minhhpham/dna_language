import itertools
from typing import List, Tuple

import numpy as np
from tokenizers import BertWordPieceTokenizer as BertWordPieceTokenizer_
from transformers import BertTokenizerFast


class BertWordPieceTokenizer(BertWordPieceTokenizer_):
    def __init__(self) -> None:
        super().__init__(
            clean_text=False,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=False
        )


class BertWordPieceTokenizerFast(BertTokenizerFast):
    def __init__(self, vocab_file=None, tokenizer_file=None,
                 do_lower_case=True, unk_token="[UNK]",
                 sep_token="[SEP]", pad_token="[PAD]",
                 cls_token="[CLS]", mask_token="[MASK]",
                 tokenize_chinese_chars=True, strip_accents=None, **kwargs):
        super().__init__(vocab_file, tokenizer_file, do_lower_case, unk_token,
                         sep_token, pad_token, cls_token, mask_token,
                         tokenize_chinese_chars, strip_accents, **kwargs)

    def batch_tokenize_and_split(
        self, sentences: List[str], pad2len: int = None
    ) -> Tuple[List[str], List[int], List[int]]:
        """tokenize a string with bert wordpiece tokenizer
                and split sub-tokens into group of original words
                pad2len: max length to pad with [PAD] tokens
            Return: idxs, splits
                idxs: token indices
                attention_mask
                splits: lengths of each group of tokens that came from one word
        """
        batch_encodings = self(sentences, max_length=pad2len,
                               padding="max_length",
                               truncation=True,
                               add_special_tokens=False)
        tokens = [
            list(itertools.takewhile(lambda t: t != "[PAD]", e.tokens))
            for e in batch_encodings.encodings
        ]
        word_pos = [
            np.array(
                [idx for idx, tk in enumerate(_tokens)
                 if not tk.startswith("##")] +
                [len(_tokens)]
            )
            for _tokens in tokens
        ]
        split_sizes = [(pos[1:] - pos[0:-1]).tolist() for pos in word_pos]
        return (
            batch_encodings["input_ids"],
            batch_encodings["attention_mask"],
            split_sizes
        )
