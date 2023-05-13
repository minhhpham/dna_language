import argparse
import os

from torch.utils.tensorboard import SummaryWriter


class DummySummaryWriter:
    """A dummy summary writer that does nothing
    """

    def add_scalar(self, *args, **kwargs):
        pass

    def add_graph(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass


class TensorboardMonitor:
    """
    provide a summary writer and a Tensorboard.dev uploader
    the uploader runs in the background
    logdir = tensorboard/exp_name/run_name
    """

    def __init__(self, exp_name: str, run_name: str, monitoring=True):
        if monitoring:
            log_dir = f"tensorboard/{exp_name}/{run_name}"
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = DummySummaryWriter()


def main():
    parser = argparse.ArgumentParser("Upload data to Tensorboard")
    subparsers = parser.add_subparsers(dest="command")
    upload_parser = subparsers.add_parser("upload",
                                          help="Upload data to Tensorboard")
    upload_parser.add_argument(
        "-e", "--exp_name",
        type=str,
        help="experiment name for upload to Tensorboard at exp_name/run_name",
        default="Test"
    )
    args = parser.parse_args()
    exp_name = args.exp_name
    print(f"uploading experiment {exp_name}")
    os.execlp(
        "tensorboard", "tensorboard", "dev", "upload",
        "--logdir", f"tensorboard/{exp_name}",
        "--name", exp_name,
    )


if __name__ == "__main__":
    main()
