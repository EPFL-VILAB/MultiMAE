# Pre-training

We provide MultiMAE pre-training scripts on (multi-modal) ImageNet-1K.  
Please check [SETUP.md](SETUP.md) for set-up instructions first.

All our models are pre-trained on a single node with **8 A100 GPUs**. 

To pre-train MultiMAE on 8 GPUs using default settings, run:
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 run_pretraining_multimae.py \
--config cfgs/pretrain/multimae-b_98_rgb+-depth-semseg_1600e.yaml \
--data_path /path/to/imagenet/train
```

### Modifying configs
The training scripts support both YAML config files and command-line arguments. See [here](cfgs/pretrain) for pre-training config files.

To modify pre-training settings, either edit / add config files or provide additional command-line arguments.

For a list of possible arguments, see [`run_pretraining_multimae.py`](run_pretraining_multimae.py).

:information_source: Config files arguments override default arguments, and command-line arguments override both default arguments and config arguments.

:warning: When changing settings, make sure to modify the `output_dir` and `wandb_run_name` (if logging is activated) to reflect the changes.

### Experiment logging
To activate logging to [Weights & Biases](https://docs.wandb.ai/), either edit the config files or use the `--log_wandb` flag along with any other extra logging arguments.
