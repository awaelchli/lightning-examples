# Character-level Language Model

Code modified from [Andrej Karpathy's minGPT repository](https://github.com/karpathy/minGPT).


## Performance Benchmarks

We benchmarked different settings for efficient training of the GPT model with high parameter counts using the Fully Sharded Data Parallel strategy (FDSP).

### Single Machine

- Machine: 4x v100 @ 16 GB (p3.8xlarge)
- Model Type: gpt2-xl
- Model Size: ~1.4 B parameters
- Mixed Precision (AMP)
- Numbers are average over 100 iterations of training. Listed batch size is per GPU.

In all experiments below, we shard the transformer block evenly across all GPU using the `transformer_auto_wrap_policy` from PyTorch.
The GPU usage was measured and averaged across all GPUs.

#### FSDP Baseline

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
| 1             | 9028	            | 71.9 +/- 22.0 	| 1.57	        | 823.47                |
| 2             | 10510	            | 70.4 +/- 19.1     | 3.17	        | 754.22                |
| 4             | 12488	            | 79.9 +/- 15.6     | 6.24	        | 744.17                |
| 7	            | 14736	            | 86.2 +/- 9.7	    | 10.88	        | 748.78                |
| 8	            | 14992	            | 88.1 +/- 6.8	    | 9.21	        | 1676.02               |
| 9	            | OOM			    | 	                |               |                       |

We found that batch size 8 fits, but 7 is optimal. Due to cudaMalloc retries, the throuhput with size 8 is worse.
The slowdown is reflected in the table by a big jump in iteration time while memory, GPU usage and TFLOP/s remain roughly the same.

#### FSDP + Activation Checkpointing

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
| 8	            | 8458	            | 83.5 +/- 17.2     | 11.82	        | 1024.5	            |
| 16	        | 8604	            | 94.3 +/- 3.7	    | 23.97	        | 1097.38	            |
| 32	        | 10188             | 94.8 +/- 1.0	    | 31.92	        | 1575.16	            |
| 64	        | 11490	            | 96.4 +/- 0.6	    | 35.57	        | 2748.41               |

Enabling activation checkpointing allows us to fit much larger batch sizes (or alternatively a larger model), and at the same time get better throughput.
The iteration time is higher compared to the FSDP baseline, but overall we get more FLOP/s. By maximizing FLOP/s, we are using the hardware efficiently, leading to reduced training time and cost.

#### FSDP + Activation Checkpointing + Backward Prefetching

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
| 64            | 11490	            | 97.3 +/- 0.9	    | 36.24	        | 2700.26               |

For this setting, adding backward prefetching did not impact the results. There is no significant difference in memory usage or throughput.

