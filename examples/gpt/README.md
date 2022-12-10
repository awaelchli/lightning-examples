# Efficient Training of Large Language Models

We benchmarked different settings for efficient training of a GPT model with high parameter counts using the Fully Sharded Data Parallel strategy (FDSP).
The model architecture is based on OpenAI's GPT-2. The code was modified from [Andrej Karpathy's minGPT repository](https://github.com/karpathy/minGPT) and accelerated using
Lightning Lite for distributed training.
The hyperparameters in these experiments are not optimized for best convergence. We are only intersted in benchmarking how to scale the model to use the available hardware as efficiently as possible.

## 1.4B Model in a Single Machine

- Machine: 4x v100 @ 16 GB (p3.8xlarge)
- Model type: gpt2-xl
- Model size: ~1.4 B parameters
- Mixed precision (AMP)
- Python code: [train.py](train.py)
- Numbers are averages over 100 iterations of training. Listed batch size is per GPU.
- Sequence length: 128

In all experiments below, we shard the transformer block evenly across all GPU using the `transformer_auto_wrap_policy` from PyTorch.
The GPU usage was measured and averaged across all GPUs.

### FSDP Baseline

<details>
<summary>Code</summary>

```py
from lightning.lite import LightningLite
from lightning.lite.strategies.fsdp import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
lite = LightningLite(
    ...
    strategy=FSDPStrategy(auto_wrap_policy=auto_wrap_policy)
)
```

</details>


| Batch Size    | Memory (MB)       | GPU Usage %       | TFLOP/s       | Iteration Time (ms)   |
| ------------- | ----------------- | ----------------- | ------------- | --------------------- |
| 1             | 9028	            | 71.9 +/- 22.0 	| 6.28	        | 823                   |
| 2             | 10510	            | 70.4 +/- 19.1     | 12.68	        | 754                   |
| 4             | 12488	            | 79.9 +/- 15.6     | 24.96	        | 744                   |
| 7 ★           | 14736	            | 86.2 +/- 9.7	    | 43.52	        | 748                   |
| 8	            | 14992	            | 88.1 +/- 6.8	    | 36.8	        | 1676                  |
| 9	            | OOM			    | 	                |               |                       |

We found that batch size 8 fits, but 7 (★) is optimal. Due to cudaMalloc retries, the throuhput with size 8 is worse.
The slowdown is reflected in the table by a big jump in iteration time while memory, GPU usage and TFLOP/s remain roughly the same.

### FSDP + Activation Checkpointing

<details>
<summary>Code</summary>

```py
lite = LightningLite(
    ...
    strategy=FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy,
        activation_checkpointing=[Block],
    )
)
```
</details>



| Batch Size    | Memory (MB)       | GPU Usage %       | TFLOP/s       | Iteration Time (ms)   |
| ------------- | ----------------- | ----------------- | ------------- | --------------------- |
| 8	            | 8458	            | 83.5 +/- 17.2     | 47.28	        | 1024	                |
| 16	        | 8604	            | 94.3 +/- 3.7	    | 95.88	        | 1097	                |
| 32	        | 10188             | 94.8 +/- 1.0	    | 127.68	    | 1575	                |
| 64	        | 11490	            | 96.4 +/- 0.6	    | 142.28	    | 2748                  |

Enabling activation checkpointing allows us to fit much larger batch sizes (or alternatively a larger model), and at the same time get better throughput.
The iteration time is higher compared to the FSDP baseline, but overall we get more FLOP/s. By maximizing FLOP/s, we are using the hardware efficiently, leading to reduced training time and cost.

### FSDP + Activation Checkpointing + Backward Prefetching

<details>
<summary>Code</summary>

```py
from torch.distributed.fsdp import BackwardPrefetch

lite = LightningLite(
    ...
    strategy=FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy,
        activation_checkpointing=[Block],
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )
)
```
</details>


| Batch Size    | Memory (MB)       | GPU Usage %       | TFLOP/s       | Iteration Time (ms)   |
| ------------- | ----------------- | ----------------- | ------------- | --------------------- |
| 64            | 11490	            | 97.3 +/- 0.9	    | 144.96	    | 2700                  |

For this setting, adding backward prefetching did not impact the results. There is no significant difference in memory usage or throughput.


## 3B Model in a Single Machine

### FSDP + Activation Checkpointing

- Machine: 4x v100 @ 16 GB (p3.8xlarge)
- Model type: gpt2-xxl
- Model size: ~3 B parameters
- Mixed precision (AMP)
- Python code: [train.py](train.py)
- Numbers are averages over 100 iterations of training. Listed batch size is per GPU.
- Sequence length: 128


| Batch Size    | Memory (MB)       | GPU Usage %       | TFLOP/s       | Iteration Time (ms)   |
| ------------- | ----------------- | ----------------- | ------------- | --------------------- |
| 8	            | 14340	            | 83.7	+/- 18.5	| 47.4	        | 2054                  | 
| 16 ★	        | 14984	            | 93.15	+/- 7.2	    | 95.32	        | 2098	                | 
| 32	        | 14992	            | 91.55	+/- 4.94	| 100.36	    | 3844	                | 

★ marks the optimal batch size before cudaMalloc retries occur.



## Multi-node

- Each machine: 4x v100 @ 16 GB (gpu-fast-multi)
- Mixed precision (AMP)
- Python code: [train.py](train_cloud.py)
- Numbers are averages over 30 iterations of training.
- Sequence length: 128
- Fixed batch size per GPU: 16


| # GPUS | # Parameters  | Memory (MB)   | GPU Usage %       | TFLOP/s       | Iteration Time (ms)   | Cost | 
| ------ | ------------- | --------------| ----------------- | --------------| ----------------------| ---- |
| 4      | 2.9 B         | 	                            | 7.67	     | 12772                 |      |
| 8		 | 4.4 B         | 13858	     | 97.1 +/- 3.3	     | 36.68	     | 15717	             | 15
| 16  	 | 8.4 B         | 14626	     | 94.3 +/- 3.6      | 55.71	     | 39228	             | 30.5

In all experiments, we fixed the batch size to 16 and maximized the memory usage by scaling the number of layers, number of attention heads, and embedding size in the transformer block.


## CPU Offload

Coming soon.