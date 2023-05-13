import argparse
from typing import List, Tuple

import torch
import tensorflow as tf
from tensorflow_text import BertTokenizer
from torch import nn, tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from corrector.models import BertMLM
from training.monitor import TensorboardMonitor

tf.config.set_visible_devices([], 'GPU')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_data(filepath: str) -> List[str]:
    lines = []
    with open(filepath, 'r') as file:
        for line in tqdm(file, desc="reading data file"):
            lines.append(line.strip())
    return lines


class MLMDataset(Dataset):
    def __init__(
            self,
            corpus: List[str],
            tokenizer: BertTokenizer,
            sample_maxlen=64) -> None:
        # tokenize corpus and pad to sample_maxlen
        print("tokenizing corpus ...")
        tokens = tokenizer.tokenize(corpus)
        tokens = tokens.merge_dims(1, 2)
        tokens = tokens.to_tensor(0)
        tokens = torch.tensor(tokens.numpy())[:, :sample_maxlen]
        tokens = torch.nn.functional.pad(
            tokens,
            [0, sample_maxlen - tokens.shape[1]],
            mode="constant"
        )
        self.tokens = tokens
        self.sample_maxlen = sample_maxlen

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, index) -> Tuple[tensor, tensor, tensor]:
        """
        Returns:
            input, attention mask, label
            input = random replace 15% of label with [MASK]
        """
        label = self.tokens[index]
        attention = torch.ones(label.shape)
        # make copy of labels tensor, this will be input_ids
        input = label.detach().clone()
        # mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
        rand = torch.rand(input.shape)
        mask = (rand < 0.15) * (input != 0) * \
            (input != 1) * (input != 2)
        MASK_TOKEN = 4
        input[mask] = MASK_TOKEN
        return input, attention, label


def main(corpus_filepath: str, wordpiece_vocab_path: str, batch_size: int):
    monitor = TensorboardMonitor(
        exp_name="BertMLM",
        run_name="64maxlen_768hidden_8attn_6layers",
        monitoring=True
    )
    tokenizer = BertTokenizer(vocab_lookup_table=wordpiece_vocab_path)
    # find vocab size
    with open(wordpiece_vocab_path, "r") as file:
        vocab_size = sum(1 for _ in file)
    lines = get_data(corpus_filepath)
    train_lines = round(len(lines) * 0.8)
    trainset = MLMDataset(corpus=lines[:train_lines], tokenizer=tokenizer)
    valset = MLMDataset(corpus=lines[train_lines:], tokenizer=tokenizer)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=0,
                             pin_memory=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=batch_size*10,
                           shuffle=True, num_workers=0,
                           pin_memory=True, drop_last=True)
    model = nn.DataParallel(BertMLM(
        vocab_size=vocab_size,
        max_position_embeddings=64,
        hidden_size=768,
        num_attention_heads=8,
        num_hidden_layers=6,
        type_vocab_size=1
    ))
    model = model.to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.set_printoptions(edgeitems=10)

    n_batch_2val = round(4e6 / batch_size)
    for epoch in range(100):
        # train
        for bid, batch in tqdm(
                enumerate(trainloader),
                desc=f"train epoch {epoch}",
                total=len(trainloader)):
            model.train()
            inputs, attentions, labels = batch
            optim.zero_grad()
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            attention_mask = attentions.to(DEVICE)
            outputs = model(input_ids=inputs,
                            attention_mask=attention_mask,
                            labels=labels)
            mean_loss = outputs.loss.mean()
            monitor.writer.add_scalar(
                tag="Loss/train/batch",
                scalar_value=mean_loss.item(),
                global_step=epoch * len(trainloader) + bid
            )
            mean_loss.mean().backward()
            optim.step()

            # run validation every n_batch_2val batches
            if bid % n_batch_2val == 0:
                model.eval()
                val_loss = []
                for val_batch in tqdm(valloader,
                                      desc=f"eval epoch {epoch}",
                                      total=len(valloader)):
                    inputs, attentions, labels = val_batch
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    attention_mask = attentions.to(DEVICE)
                    outputs = model(input_ids=inputs,
                                    attention_mask=attention_mask,
                                    labels=labels)
                    val_loss.append(outputs.loss.mean().item())
                val_loss = sum(val_loss) / len(val_loss)
                monitor.writer.add_scalar(
                    tag="Loss/Eval",
                    scalar_value=val_loss,
                    global_step=epoch * len(trainloader) + bid
                )
        # save model after epoch
        model.module.save_pretrained(f"./saved_models/bert_encoder_{epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus", type=str,
                        help="path to corpus file", required=True)
    parser.add_argument("-v", "--vocab", type=str,
                        help="path to vocab file for wordpiece tokenizer",
                        required=True)
    parser.add_argument("-b", "--batch_size", type=int,
                        default=64,
                        help="batch size")
    args = parser.parse_args()
    main(corpus_filepath=args.corpus,
         wordpiece_vocab_path=args.vocab,
         batch_size=args.batch_size)
