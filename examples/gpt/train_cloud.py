# ! pip install --upgrade --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cu117
# ! PACKAGE_NAME=lite pip install git+https://github.com/Lightning-AI/lightning
# ! cd data && bash download-data.sh

from gpt.config import GPTConfig, TrainerConfig
from lightning import CloudCompute, LightningApp, LightningWork
from lightning.app.components import LiteMultiNode
from lightning_lite.lite import LightningLite
from train import train


class Work(LightningWork):
    def run(self):
        model_config = GPTConfig(
            model_type="gpt2-4xl",
            vocab_size=None,
            block_size=128,
            embd_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
        trainer_config = TrainerConfig(
            num_workers=4,
            max_iters=30,
            block_size=128,
            batch_size=16,
            learning_rate=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            grad_norm_clip=1.0,
        )
        lite = LightningLite(
            accelerator="cuda",
            devices=-1,
            precision=16,
            strategy="fsdp-gpt",
            num_nodes=4,  # TODO: Let MultiNode component set this value automatically
        )
        train(lite, model_config, trainer_config)


app = LightningApp(
    LiteMultiNode(
        Work,
        num_nodes=4,
        cloud_compute=CloudCompute(name="gpu-fast-multi"),
    )
)
