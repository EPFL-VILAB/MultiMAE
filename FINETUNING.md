# Fine-tuning

We provide fine-tuning scripts for classification, semantic segmentation, depth estimation and more.
Please check [SETUP.md](SETUP.md) for set-up instructions first.

- [General information](#general-information)
- [Classification](#classification)
- [Semantic segmentation](#semantic-segmentation)
- [Depth estimation](#depth-estimation)
- [Taskonomy tasks](#taskonomy-tasks)

## General information

### Loading pre-trained models

All our fine-tuning scripts support models in the MultiMAE / MultiViT format. Pre-trained models using the timm / ViT format can be converted to this format using the [`vit2multimae_converter.py`](tools/vit2multimae_converter.py)
 script. More information can be found [here](README.md#model-formats).

### Modifying configs
The training scripts support both YAML config files and command-line arguments. See [here](cfgs/finetune) for all fine-tuning config files.

To modify fine-training settings, either edit / add config files or provide additional command-line arguments.

:information_source: Config files arguments override default arguments, and command-line arguments override both default arguments and config arguments.

:warning: When changing settings (e.g., using a different pre-trained model), make sure to modify the `output_dir` and `wandb_run_name` (if logging is activated) to reflect the changes.


### Experiment logging
To activate logging to [Weights & Biases](https://docs.wandb.ai/), either edit the config files or use the `--log_wandb` flag along with any other extra logging arguments.


## Classification

We use 8 A100 GPUs for classification fine-tuning. Configs can be found [here](cfgs/finetune/cls).

To fine-tune MultiMAE on ImageNet-1K classification using default settings, run:
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 run_finetuning_cls.py \
--config cfgs/finetune/cls/ft_in1k_100e_multimae-b.yaml \
--finetune /path/to/multimae_weights \
--data_path /path/to/in1k/train/rgb \
--eval_data_path /path/to/in1k/val/rgb
```

- For a list of possible arguments, see [`run_finetuning_cls.py`](run_finetuning_cls.py).

## Semantic segmentation

We use 4 A100 GPUs for semantic segmentation fine-tuning. Configs can be found [here](cfgs/finetune/semseg).

### ADE20K
To fine-tune MultiMAE on ADE20K semantic segmentation with default settings and **RGB** as the input modality, run:
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 run_finetuning_semseg.py \
--config cfgs/finetune/semseg/ade/ft_ade_64e_multimae-b_rgb.yaml \
--finetune /path/to/multimae_weights \
--data_path /path/to/ade20k/train \
--eval_data_path /path/to/ade20k/val
```

- For a list of possible arguments, see [`run_finetuning_semseg.py`](run_finetuning_semseg.py).


### Hypersim
To fine-tune MultiMAE on Hypersim semantic segmentation with default settings and **RGB** as the input modality, run:
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 run_finetuning_semseg.py \
--config cfgs/finetune/semseg/hypersim/ft_hypersim_25e_multimae-b_rgb.yaml \
--finetune /path/to/multimae_weights \
--data_path /path/to/hypersim/train \
--eval_data_path /path/to/hypersim/val
```

- To fine-tune using **depth-only** and **RGB + depth** as the input modalities, simply swap the config file to the appropriate one.
- For a list of possible arguments, see [`run_finetuning_semseg.py`](run_finetuning_semseg.py).



### NYUv2
To fine-tune MultiMAE on NYUv2 semantic segmentation with default settings and **RGB** as the input modality, run:
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 run_finetuning_semseg.py \
--config cfgs/finetune/semseg/nyu/ft_nyu_200e_multimae-b_rgb.yaml \
--finetune /path/to/multimae_weights \
--data_path /path/to/nyu/train \
--eval_data_path /path/to/nyu/test_or_val
```

- To fine-tune using **depth-only** and **RGB + depth** as the input modalities, simply swap the config file to the appropriate one.
- For a list of possible arguments, see [`run_finetuning_semseg.py`](run_finetuning_semseg.py).


## Depth estimation

We use 2 A100 GPUs for depth estimation fine-tuning. Configs can be found [here](cfgs/finetune/depth).


To fine-tune MultiMAE on NYUv2 depth estimation with default settings, run:
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 run_finetuning_depth.py \
--config cfgs/finetune/depth/ft_nyu_2000e_multimae-b.yaml \
--finetune /path/to/multimae_weights \
--data_path /path/to/nyu/train \
--eval_data_path /path/to/nyu/test_or_val
```
- For a list of possible arguments, see [`run_finetuning_depth.py`](run_finetuning_depth.py).

## Taskonomy tasks

We use 1 A100 GPU to fine-tune on Taskonomy tasks. Configs can be found [here](cfgs/finetune/taskonomy).

The tasks we support are: Principal curvature, z-buffer depth, texture edges, occlusion edges, 2D keypoints,
3D keypoints, surface normals, and reshading. 


For example, to fine-tune MultiMAE on Taskonomy reshading with default settings, run:
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 run_finetuning_taskonomy.py \
--config cfgs/finetune/taskonomy/rgb2reshading-1k/ft_rgb2reshading_multimae-b.yaml \
--finetune /path/to/multimae_weights \
--data_path /path/to/taskonomy_tiny
```

- To fine-tune on a different task, simply swap the config file to the appropriate one.
- For a list of possible arguments, see [`run_finetuning_taskonomy.py`](run_finetuning_taskonomy.py).
