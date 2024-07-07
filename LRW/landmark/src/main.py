from __future__ import annotations

# isort: off
if True:
    import jax

    jax.distributed.initialize()
# isort: on

import argparse
import random
import warnings
from collections import defaultdict
from functools import partial

import flax
import numpy as np
import tqdm
import wandb
from flax.jax_utils import unreplicate
from flax.training.common_utils import shard

from dataset import create_dataloaders
from training import TrainState, create_train_state, training_step, validation_step

warnings.filterwarnings("ignore")


class AverageMeter:
    def __init__(self, use_latest: list[str] = []):
        self.buffer = defaultdict(list)
        self.use_latest = use_latest

    def update(self, **kwargs: float):
        for k, v in kwargs.items():
            self.buffer[k].append(v)

    def average(self, prefix: str = "") -> dict[str, float]:
        buffer = {k: np.array(v) for k, v in self.buffer.items()}
        self.buffer.clear()

        return {
            f"{prefix}{k}": v[-1] if k in self.use_latest else np.mean(v)
            for k, v in buffer.items()
        }


def save_checkpoint(filename: str, state: TrainState):
    with open(filename, "wb") as fp:
        fp.write(flax.serialization.msgpack_serialize(unreplicate(state.params)))


def main(args: argparse.Namespace):
    # jax.distributed.initialize()
    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(args)

    state = create_train_state(args, len(train_dataloader)).replicate()
    wandb.init(name=args.name, entity=args.entity, project=args.project, config=args)

    train_average_meter, current_step = AverageMeter(use_latest=["learning_rate"]), 0
    valid_average_meter, test_average_meter = AverageMeter(), AverageMeter()

    for epoch in range(args.training_epochs):
        for batch in tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False):
            batch = shard(jax.tree_map(np.asarray, batch))
            state, metrics = training_step(state, batch)

            train_average_meter.update(**unreplicate(metrics))
            current_step += 1

            if args.log_interval > 0 and current_step % args.log_interval == 0:
                metrics = train_average_meter.average(prefix="train/")
                wandb.log(metrics, current_step)

        for batch in tqdm.tqdm(valid_dataloader, desc="Validate", leave=False):
            metrics = validation_step(state, shard(jax.tree_map(np.asarray, batch)))
            valid_average_meter.update(**jax.device_get(unreplicate(metrics)))
        wandb.log({"epoch": epoch, **valid_average_meter.average("val/")}, current_step)

    for batch in tqdm.tqdm(test_dataloader, leave=False):
        metrics = validation_step(state, shard(jax.tree_map(np.asarray, batch)))
        test_average_meter.update(**jax.device_get(unreplicate(metrics)))
    wandb.log(test_average_meter.average("test/"))
    save_checkpoint(f"{args.name}.msgpack", state)


if __name__ == "__main__":
    GENERATE_RANDOM_SEED = partial(random.randint, 0, 1000000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="../resources/LRW")
    parser.add_argument("--duration-path", default="../lipread_durations.csv")
    parser.add_argument("--num-train-workers", type=int, default=60)
    parser.add_argument("--num-valid-workers", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--valid-batch-size", type=int, default=128)

    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1) # label smoothing should be tested
    parser.add_argument("--audio-alignment", type=int, default=4)
    parser.add_argument("--vq-groups", type=int, default=2)
    parser.add_argument("--audio-vocab-size", type=int, default=320)
    parser.add_argument("--cmts-lambda", type=float, default=1.0)

    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--labels", type=int, default=500)
    parser.add_argument("--emb-dropout", type=float, default=0.1)
    parser.add_argument("--msa-dropout", type=float, default=0.1)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--droppath", type=float, default=0.1)

    parser.add_argument("--use-word-boundary", action="store_true", default=False)
    parser.add_argument("--init-seed", type=int, default=GENERATE_RANDOM_SEED())
    parser.add_argument("--mixup-seed", type=int, default=GENERATE_RANDOM_SEED())
    parser.add_argument("--dropout-seed", type=int, default=GENERATE_RANDOM_SEED())
    parser.add_argument("--pretrained-ckpt")

    parser.add_argument("--input-features", type=int, default=1434)
    parser.add_argument("--input-length", type=int, default=29)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-b1", type=float, default=0.9)
    parser.add_argument("--adam-b2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--max-norm", type=float, default=1.0)

    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--training-epochs", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=50)

    parser.add_argument("--entity", default="quoqa-nlp")
    parser.add_argument("--project", default="MobileVSR")
    parser.add_argument("--name")
    parser.add_argument("--ipaddr")
    parser.add_argument("--hostname")
    main(parser.parse_args())
