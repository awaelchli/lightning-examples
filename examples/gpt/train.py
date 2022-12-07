"""
Trains a character-level language model.
"""
import functools
import time

import torch
from gpt.config import GPTConfig, TrainerConfig
from gpt.dataset import CharDataset
from gpt.model import Block, GPT
from lightning_lite import seed_everything
from lightning_lite.lite import LightningLite
from lightning_lite.strategies import STRATEGY_REGISTRY
from lightning_lite.strategies.fsdp import FSDPStrategy
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data.dataloader import DataLoader
from tools import FlopCounter

auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
STRATEGY_REGISTRY.register(
    name="fsdp-gpt",
    strategy=FSDPStrategy,
    description="FSDP strategy with memory optimizations enabled for GPT large scale pretraining.",
    auto_wrap_policy=auto_wrap_policy,
    activation_checkpointing=[Block],
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
)


def main():
    model_config = GPTConfig(
        model_type="gpt2-xl",
        vocab_size=None,
        block_size=128,
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    )
    trainer_config = TrainerConfig(
        num_workers=4,
        max_iters=100,
        block_size=128,
        batch_size=64,
        learning_rate=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,  # only applied on matmul weights
        grad_norm_clip=1.0,
    )

    # TODO: precision 16 and cpu offload hangs
    lite = LightningLite(
        accelerator="cuda",
        devices=-1,
        precision=16,
        strategy="fsdp-gpt",
    )
    lite.launch()
    train(lite, model_config, trainer_config)


def train(lite, model_config, trainer_config):
    seed_everything(trainer_config.seed)

    # construct the training dataset
    train_dataset = CharDataset(textfile="data/tinyshakespeare.txt", block_size=model_config.block_size)
    model_config.vocab_size = train_dataset.get_vocab_size()

    lite.print(model_config)
    lite.print(trainer_config)

    # setup the model and optimizer
    with lite.sharded_model():
        model = GPT(model_config)
    model = lite.setup_module(model)

    lite.print(f"Number of parameters per device: {model.num_parameters / 1e6:.1f} M")
    lite.print(f"Total number of parameters: ~ {lite.world_size * model.num_parameters / 1e6:.1f} M")

    # TODO: support multiple param groups for FSDP
    # optimizer = model.configure_optimizers(config.trainer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trainer_config.learning_rate, betas=trainer_config.betas)
    optimizer = lite.setup_optimizers(optimizer)

    train_loader = DataLoader(
        train_dataset,
        # TODO: fix this in Lite
        # sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=True,
        pin_memory=True,
        batch_size=trainer_config.batch_size,
        num_workers=trainer_config.num_workers,
    )
    train_loader = lite.setup_dataloaders(train_loader)

    model.train()
    iteration = 0
    data_iter = iter(train_loader)
    flops = 0
    total_iter_dt = 0

    while True:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x, y = batch

        with FlopCounter(model) as flop_counter:
            _, loss = model(x, y)
            model.zero_grad(set_to_none=True)
            lite.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer_config.grad_norm_clip)
            optimizer.step()

        flops += flop_counter.total()
        iter_dt = flop_counter.time()
        total_iter_dt += iter_dt

        iteration += 1

        if iteration % 10 == 0:
            avg_gflops = flops / 1e9 / total_iter_dt
            lite.print(f"iteration time {iter_dt * 1e3:.2f}ms; iteration {iteration}; train loss {loss.item():.5f}; GFLOP/s: {avg_gflops:.2f}")

        if trainer_config.max_iters != -1 and iteration >= trainer_config.max_iters:
            break

    # For optimal memory throughput, make sure the summary shows 0 cudaMalloc retries and otherwise try lowering the batch size.
    lite.print(torch.cuda.memory_summary())


if __name__ == "__main__":
    main()
