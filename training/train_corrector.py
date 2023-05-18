import argparse
import os
import sqlite3
import sys
from math import ceil
from typing import List, Tuple

import torch
from pytorch_pretrained_bert import BertAdam
from tensorflow_text import BertTokenizer
from torch import nn, tensor
from torch.utils.data import DataLoader, Dataset
import tensorflow as tf
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from corrector.models import BertCorrector  # noqa
from tokenizer.models import WordVocab  # noqa
from training.metrics import DNACorrectorMetrics  # noqa
from training.monitor import TensorboardMonitor  # noqa

# from training.monitor import TensorboardMonitor  # noqa
tf.config.set_visible_devices([], 'GPU')
device = "cuda" if torch.cuda.is_available() else "cpu"
global TESTING_CODE, NSAMPLES, PAD_TOKEN, SAMPLE_MAXLEN
PAD_TOKEN = 0


def get_data() -> List[Tuple[str, str]]:
    """sampples of query sequence, corrected (aligned) sequence"""
    n_samples = 10000 if TESTING_CODE else NSAMPLES
    connection = sqlite3.Connection("database.sqlite")
    cursor = connection.cursor()
    print("querying ...")
    cursor.execute(f"""
        SELECT 
            REPLACE(read, '-', ''),
            REPLACE(reference, '-', ''),
            split_idx
        FROM alignments
        WHERE UPPER(read) NOT LIKE '%N%' AND UPPER(reference) NOT LIKE '%N%'
        LIMIT {n_samples}
    """)
    samples = cursor.fetchall()
    print(f"fetched {len(samples)} samples")
    split_idx = [[int(i) for i in s[2].split(',')] for s in samples]
    reads = [' '.join([s[0][i:j]
                       for i, j in zip((0, *idx), (*idx, None))])
             for s, idx in tqdm(zip(samples, split_idx),
                                desc="splitting read",
                                total=len(samples))
             ]
    refs = [' '.join([s[1][i:j]
                      for i, j in zip((0, *idx), (*idx, None))])
            for s, idx in tqdm(zip(samples, split_idx),
                               desc="splitting ref",
                               total=len(samples))
            ]
    return reads, refs


class CorrectorDataset(Dataset):
    def __init__(
            self,
            reads: List[str],
            refs: List[str],
            wordpiece_tokenizer: BertTokenizer,
            word_tokenizer: WordVocab,
            sample_maxlen=64) -> None:
        global SAMPLE_MAXLEN
        SAMPLE_MAXLEN = sample_maxlen
        # tokenize corpus and pad to sample_maxlen
        _read_tokens = wordpiece_tokenizer.tokenize(reads)
        _sample_sizes, _token_clusters = _read_tokens.nested_row_lengths()
        _sample_splits = tf.RaggedTensor.from_row_lengths(
            values=_token_clusters, row_lengths=_sample_sizes)
        self.read_token_splits = _sample_splits.numpy()
        # convert to torch tensor and pad to sample_maxlen
        _read_tokens = _read_tokens.merge_dims(1, 2).to_tensor(PAD_TOKEN)
        _read_tokens = torch.tensor(_read_tokens.numpy())[:, :sample_maxlen]
        self.read_tokens = torch.nn.functional.pad(
            _read_tokens,
            [0, sample_maxlen - _read_tokens.shape[1]],
            mode="constant"
        )
        # generate full attention
        self.read_attn = (self.read_tokens != 0) * 1

        print("tokenizing sample references ...")
        self.ref_tokens = word_tokenizer.batch_tokenize(refs)

    def __len__(self):
        return len(self.ref_tokens)

    def __getitem__(self, index) -> Tuple[tensor, tensor, tensor]:
        """
        Returns:
            inputs, attention mask, split_sizes, target
        """
        inputs = self.read_tokens[index]
        attn = self.read_attn[index]
        split_sizes = self.read_token_splits[index].tolist()
        target = self.ref_tokens[index]
        return inputs, attn, split_sizes, target

    @staticmethod
    def collate_fn(data: List[Tuple]):
        global SAMPLE_MAXLEN
        inputss, attn, split_sizes, target = list(zip(*data))
        # pad target sequences to batch's max length
        target = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(tokens[:SAMPLE_MAXLEN]) for tokens in target],
            batch_first=True,
            padding_value=PAD_TOKEN
        )
        return [
            torch.stack(inputss),
            torch.stack(attn),
            list(split_sizes),
            target
        ]


def train_corrector(
        wordpiece_vocab_file: str,
        word_vocab_file: str,
        bert_encoder_path: str,
        batch_size: int,
        ):
    samples_read, samples_ref = get_data()
    # load tokenizer
    word_tokenizer = WordVocab(word_vocab_file)
    wordpiece_tokenizer = BertTokenizer(wordpiece_vocab_file)
    # split train and validation
    cutoff = round(len(samples_ref) * 0.9)
    train_samples = CorrectorDataset(
        reads=samples_read[:cutoff],
        refs=samples_ref[:cutoff],
        wordpiece_tokenizer=wordpiece_tokenizer,
        word_tokenizer=word_tokenizer
    )
    valid_samples = CorrectorDataset(
        reads=samples_read[cutoff:],
        refs=samples_ref[cutoff:],
        wordpiece_tokenizer=wordpiece_tokenizer,
        word_tokenizer=word_tokenizer
    )
    trainloader = DataLoader(train_samples, batch_size=batch_size,
                             shuffle=True, num_workers=0,
                             pin_memory=True, drop_last=True,
                             collate_fn=CorrectorDataset.collate_fn)
    valloader = DataLoader(valid_samples, batch_size=batch_size,
                           shuffle=True, num_workers=0,
                           pin_memory=True, drop_last=True,
                           collate_fn=CorrectorDataset.collate_fn)
    # init model
    model = nn.DataParallel(
        BertCorrector(
            out_vocab_size=len(word_tokenizer),
            bert_encoder_path=bert_encoder_path
        )
    )
    print("number of parameters: ", sum(p.numel() for p in model.parameters()))
    model = model.to(device)
    n_batches = ceil(len(samples_ref) / batch_size)
    t_total = int(n_batches / 4 * 10)
    optimizer = BertAdam(model.parameters(), lr=5e-5,
                         warmup=0.1, t_total=t_total)

    # train
    n_batch_2val = round(5e3 / batch_size) if TESTING_CODE \
        else round(4e6 / batch_size)
    monitor = TensorboardMonitor(exp_name="DNACorrector",
        run_name="default_params", monitoring=not TESTING_CODE)
    class_metrics = DNACorrectorMetrics(out_vocab_size=len(word_tokenizer))
    for epoch in range(100):
        for bid, (inputs, attn, split_sizes, target) in enumerate(
                tqdm(trainloader, desc=f"Epoch {epoch} training")):
            model.train()
            try:
                inputs, attn, target = \
                    inputs.to(device), attn.to(device), target.to(device)
                optimizer.zero_grad()
                loss = model(
                    input_ids=inputs,
                    attention_mask=attn,
                    batch_splits=split_sizes,
                    targets=target
                )
                mean_loss = loss.mean()
                mean_loss.backward()
                optimizer.step()
                monitor.writer.add_scalar(
                    tag="Loss/train/batch",
                    scalar_value=mean_loss.item(),
                    global_step=epoch * len(trainloader) + bid
                )
            except Exception as e:
                print(e)

            # run validation every n_batch_2val batches
            if bid % n_batch_2val == 0:
                model.eval()
                val_loss = []
                class_metrics.start_batch_recording()

                for inputs, attn, split_sizes, target in tqdm(
                        valloader,
                        desc=f"Epoch {epoch} validation"):
                    try:
                        inputs, attn, target = \
                            inputs.to(device), attn.to(device), target.to(device)
                        logits = model(
                            input_ids=inputs,
                            attention_mask=attn,
                            batch_splits=split_sizes
                        )
                        logits_permuted = logits.permute(0, 2, 1)  # [BS, batch_maxlen, out_vocab_size] # noqa
                        loss = model.module.loss_fn(logits_permuted, target)
                        val_loss.append(loss.mean().item())  # average across GPUs
                        # merge batch dim and sentence dim
                        logits = torch.flatten(logits, end_dim=1)
                        target = torch.flatten(target)
                        class_metrics.record_batch(logits, target)
                    except Exception as e:
                        print(e)

                val_loss = sum(val_loss) / len(val_loss)
                monitor.writer.add_scalar(
                    tag="Loss/Eval",
                    scalar_value=val_loss,
                    global_step=epoch * len(trainloader) + bid
                )
                class_metrics.stop_batch_recording(
                    writer=monitor.writer,
                    step=epoch * len(trainloader) + bid)

        # save model after epoch
        torch.save(model.module.state_dict(),
            "./saved_models/bert_corrector/weights_{epoch}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wordpiece_vocab", type=str,
                        help="path to vocab file for wordpiece tokenizer",
                        required=True)
    parser.add_argument("-v", "--word_vocab", type=str,
                        help="path to vocab file for word tokenizer",
                        required=True)
    parser.add_argument("-e", "--encoder", type=str,
                        help="path to pretrained BERT encoder",
                        required=True)
    parser.add_argument("-b", "--batch_size", type=int,
                        default=64,
                        help="batch size")
    parser.add_argument("-n", "--nsamples", type=int,
                        default=int(35e7),
                        help="sample size")
    parser.add_argument('--test', action=argparse.BooleanOptionalAction,
                        default=False)
    args = parser.parse_args()
    TESTING_CODE = args.test
    NSAMPLES = args.nsamples

    train_corrector(
        wordpiece_vocab_file=args.wordpiece_vocab,
        word_vocab_file=args.word_vocab,
        bert_encoder_path=args.encoder,
        batch_size=args.batch_size)
