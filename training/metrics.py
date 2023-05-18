from torchmetrics import Accuracy, Precision, Recall, AUROC
import torch
from torch import tensor
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"


class DNACorrectorMetrics:
    def __init__(self, out_vocab_size: int) -> None:
        self.accuracy = Accuracy(
            task="multiclass", num_classes=out_vocab_size).to(device)
        self.precision = Precision(
            task="multiclass", num_classes=out_vocab_size).to(device)
        self.recall = Recall(
            task="multiclass",num_classes=out_vocab_size).to(device)
        self.auroc = AUROC(
            task="multiclass", num_classes=out_vocab_size).to(device)
    
    def calculate_metrics(self, logits: tensor, targets: tensor):
        """return all metrics in order
        """
        assert logits.shape[0] == targets.shape[0]  # batch dim must match
        return (
            self.accuracy(logits, targets).item(),
            self.precision(logits, targets).item(),
            self.recall(logits, targets).item(),
            # self.auroc(logits, targets)
        )

    def start_batch_recording(self):
        self._accu = []
        self._prec = []
        self._recall = []
        self._auc = []
    
    def record_batch(self, logits: tensor, targets: tensor):
        accu, prec, recall = self.calculate_metrics(logits, targets)
        self._accu.append(accu)
        self._prec.append(prec)
        self._recall.append(recall)
        # self._auc.append(auroc)

    def stop_batch_recording(
            self, writer: SummaryWriter, step: int, tag_prefix="") -> None:
        def _mean(_arr):
            return sum(_arr) / len(_arr)
        writer.add_scalar(tag=f"{tag_prefix}Accuracy",
            scalar_value=_mean(self._accu), global_step=step)
        writer.add_scalar(tag=f"{tag_prefix}Precision",
            scalar_value=_mean(self._prec), global_step=step)
        writer.add_scalar(tag=f"{tag_prefix}Recall",
            scalar_value=_mean(self._recall), global_step=step)
        # writer.add_scalar(tag=f"{tag_prefix}AUROC",
        #     scalar_value=_mean(self._auc), global_step=step)
