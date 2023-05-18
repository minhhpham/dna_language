from transformers import BertConfig, BertForMaskedLM
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
import transformers
from typing import List
import numpy as np


class BertMLM(BertForMaskedLM):
    def __init__(
        self,
        vocab_size,
        max_position_embeddings=120,
        hidden_size=768,
        num_attention_heads=8,
        num_hidden_layers=6,
        type_vocab_size=1
    ) -> None:
        super().__init__(BertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=type_vocab_size
        ))
        self.name = f"vocab{vocab_size}_emb{max_position_embeddings}_attn{num_attention_heads}_layers{num_hidden_layers}"  # noqa


class BertCorrector(nn.Module):

    def __init__(
        self,
        out_vocab_size: int,
        bert_encoder_path: str,
        device: str = None
    ) -> None:
        super().__init__()
        self.bertEncoder = transformers.AutoModel.from_pretrained(
            bert_encoder_path)
        self.hidden_size = self.bertEncoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, out_vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(
            reduction='mean', ignore_index=0)  # ignore padding index 0
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def merge_encodings(self, bert_encodings: Tensor, splits: List[int]):
        """take average of subword embedding per input word
        """
        # trim to the original number of words
        bert_encodings = bert_encodings[:sum(splits), :]
        # split to a tuple of tensors
        split_encodings = torch.split(bert_encodings, splits, dim=0)
        batched_encodings = pad_sequence(
            split_encodings, batch_first=True, padding_value=0)
        return torch.div(
            torch.sum(batched_encodings, dim=1),
            torch.tensor(splits).reshape(-1, 1).to(self.device)
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        batch_splits: List[List[int]],
        targets: Tensor = None
    ) -> Tensor:
        """assume all tensors are already on self.device
        if training, return loss
        if eval, return logits of each word in out_vocab
        """
        bert_encodings = self.bertEncoder(
            input_ids, attention_mask, return_dict=False)[0]
        # we need to pad all sequence encodings in batch to equal length
        #   up to length of the longest one
        max_len = max(len(splits) for splits in batch_splits)
        pad_sample = torch.empty(max_len, self.hidden_size, device=self.device)
        # merge encodings by their original word prior to subword tokenizer
        bert_merged_encodings = pad_sequence(
            [pad_sample] +
            [self.merge_encodings(bert_seq_encodings, seq_splits)
             for bert_seq_encodings, seq_splits in zip(bert_encodings, batch_splits)],  # noqa
            batch_first=True,
            padding_value=0
        )
        bert_merged_encodings = bert_merged_encodings[1:]  # remove pad_sample
        logits = self.dense(bert_merged_encodings)
        if not self.training:
            return logits
        else:
            assert targets.shape[0] == logits.shape[0]  # batch size
            assert targets.shape[1] == logits.shape[1]  # max word count
            # logits are now in shape [BS, batch_maxlen, out_vocab_size]
            # transform logits to shape [BS, out_vocab_size, batch_maxlen]
            logits_permuted = logits.permute(0, 2, 1)  # [BS, batch_maxlen, out_vocab_size] # noqa
            return self.loss_fn(logits_permuted, targets)

    def predict_batch(
        self,
        input_ids: Tensor, attention_masks: Tensor,
        batch_splits: List[List[int]]
    ) -> np.ndarray:
        """predict a batch of input
        return corrected sentences (in form of index)
        """
        _ = self.eval()
        assert len(input_ids.shape) == 2 and len(attention_masks.shape) == 2
        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_masks,
            batch_splits=batch_splits
        )
        best_idx = torch.argmax(logits, dim=-1)  # [BS, max_nwords]

        return best_idx.cpu().detach().numpy()

    def predict(
        self,
        input_ids: List[int], attention_mask: List[int],
        batch_splits: List[int]
    ) -> np.ndarray:
        """
        predict a single input
        """
        return self.predict_batch(
            input_ids=torch.tensor([input_ids]),
            attention_masks=torch.tensor([attention_mask]),
            batch_splits=[batch_splits]
        )[0]
