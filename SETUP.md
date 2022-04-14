# Set-up

## Dependencies

This codebase has been tested with the packages and versions specified in `requirements.txt` and Python 3.8.

We recommend creating a new [conda](https://docs.conda.io/en/latest/) virtual environment:
```bash
conda create -n multimae python=3.8 -y
conda activate multimae
```
Then, install [PyTorch](https://pytorch.org/) 1.10.0+ and [torchvision](https://pytorch.org/vision/stable/index.html) 0.11.1+. For example:
```bash
conda install pytorch=1.10.0 torchvision=0.11.1 -c pytorch -y
```

Finally, install all other required packages:
```bash
pip install timm==0.4.12 einops==0.3.2 pandas==1.3.4 albumentations==1.1.0 wandb==0.12.11
```
:information_source: If data loading and image transforms are the bottleneck, consider replacing Pillow with [Pillow-SIMD](https://github.com/uploadcare/pillow-simd) and compiling it with [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo). You can find a detailed guide on how to do this [here](https://fastai1.fast.ai/performance.html#installation) or use the provided script:
```bash
sh tools/install_pillow_simd.sh
```

## Dataset Preparation

### Dataset structure

For simplicity and uniformity, all our datasets are structured in the following way:
```
/path/to/data/
├── train/
│   ├── modality1/
│   │   └── subfolder1/
│   │       ├── img1.ext1
│   │       └── img2.ext1
│   └── modality2/
│       └── subfolder1/
│           ├── img1.ext2
│           └── img2.ext2
└── val/
    ├── modality1/
    │   └── subfolder2/
    │       ├── img3.ext1
    │       └── img4.ext1
    └── modality2/
        └── subfolder2/
            ├── img3.ext2
            └── img4.ext2
```
The folder structure and filenames should match across modalities.
If a dataset does not have specific subfolders, a generic subfolder name can be used instead (e.g., `all/`). 

For most experiments, we use RGB  (`rgb`), depth (`depth`), and semantic segmentation (`semseg`) as our modalities.

RGB images are stored as either PNG or JPEG images. 
Depth maps are stored as either single-channel JPX or single-channel PNG images. 
Semantic segmentation maps are stored as single-channel PNG images.

### Datasets

We use the following datasets in our experiments:
- [**ImageNet-1K**](https://www.image-net.org/)
- [**ADE20K**](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
- [**Hypersim**](https://github.com/apple/ml-hypersim)
- [**NYUv2**](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- [**Taskonomy**](https://github.com/StanfordVL/taskonomy/tree/master/data)

To download these datasets, please follow the instructions on their respective pages. 
To extract semantic classes from NYUv2, follow the [data preparations instructions from ShapeConv](https://github.com/hanchaoleng/ShapeConv/tree/master/data_preparation).

### Pseudo labeling networks

We use two off-the-shelf networks to pseudo label the ImageNet-1K dataset. 

- **Depth estimation**: We use a [DPT](https://arxiv.org/abs/2103.13413) with a ViT-B-Hybrid backbone pre-trained on the [Omnidata](https://omnidata.vision/) dataset. You can find installation instructions and pre-trained weights for this model [**here**](https://docs.omnidata.vision/pretrained.html).
- **Semantic segmentation**: We use a [Mask2Former](https://bowenc0221.github.io/mask2former/) with a Swin-S backbone pre-trained on the [COCO](https://cocodataset.org/) dataset. You can find installation instructions and pre-trained weights for this model [**here**](https://github.com/facebookresearch/Mask2Former).

For an example of how to use these networks for pseudo labeling, please take a look at our [**Colab notebook**](https://colab.research.google.com/github/EPFL-VILAB/MultiMAE/blob/main/MultiMAE_Demo.ipynb).

:information_source: The MultiMAE pre-training strategy is flexible and can benefit from higher quality pseudo labels and ground truth data. So feel free to use different pseudo labeling networks and datasets than the ones we used!
