# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on the rock-pytorch codebase
# https://github.com/vita-epfl/rock-pytorch
# --------------------------------------------------------

import argparse
import os
from typing import Tuple, Optional

import numpy
import numpy as np
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm
# The h5py package is optional everywhere else, but it's required here.
try:
    import h5py
except ImportError:
    h5py = None
    print("h5py is not installed. Please install it to prepare the NYUv2 dataset.")


def get_args():
    parser = argparse.ArgumentParser("Script to prepare NYUv2 dataset", add_help=True)
    parser.add_argument('--dataset_path', type=str,
                        help="Path to the folder containing the NYUv2 dataset."
                             "Can be downloaded from: http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat")
    parser.add_argument('--splits_path', type=str,
                        help="Path to the folder containing the splits. "
                             "Can be downloaded from: http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat")
    parser.add_argument('--class_mapping_path', type=str,
                        help="Path to the class mapping file. "
                             "Can be downloaded from: https://github.com/ankurhanda/nyuv2-meta-data/raw/master/classMapping40.mat ")
    parser.add_argument('--normals_path', type=str, default=None,
                        help="Path to the folder containing the normals")
    parser.add_argument('--save_path', type=str, help="Path to where the dataset should be saved")
    return parser.parse_args()

class NYUv2Preprocessing(object):
    """Pre-processes the NYUv2 dataset
    Parses .mat files from the NYUv2 dataset, extracts necessary info
    and finds bounding boxes
    """

    def __init__(self, dataset_path: str, splits_path: str, class_mapping_path: Optional[str],
                 normals_path: Optional[str]) -> None:
        self.in_f = h5py.File(dataset_path, 'r')
        self.nyuv2 = {}

        for name, data in self.in_f.items():
            self.nyuv2[name] = data

        self.label_map = self.__read_label_map(class_mapping_path)
        self.imgs, self.depths, self.labels = self.__get_arrs(label_map=self.label_map)

        self.len = self.imgs.shape[0]

        self.train_idx, self.test_idx = self._splits(splits_path)

        self.val = False
        self.val_idx = []

        if normals_path is not None:
            self.masks, self.normals = get_surface_normals(normals_path)
        else:
            self.masks, self.normals = None, None

    def save(self, path: str, subset: str = 'all') -> None:
        """Saves a specified subset of the data at a given folder path.

        Subset can be `train`, `test`, `val` or `all`.
        """
        self._create_subdirs(path)

        if subset == 'train':
            self._save_subset(path, self.train_idx)
        elif subset == 'test':
            self._save_subset(path, self.test_idx)
        else:
            print("Couldn't find specified subset")

    def _save_rgb(self, base_path, idx, digits=4):
        save_path = os.path.join(base_path, 'rgb', 'data', str(idx).rjust(digits, '0') + '.png')
        img = Image.fromarray(self.imgs[idx])
        img.save(fp=save_path, format='png')

    def _save_depth(self, base_path, idx, digits=4):
        save_path = os.path.join(base_path, 'depth', 'data', str(idx).rjust(digits, '0') + '.png')
        # Save as uint16, max depth in NYUv2 is 10 meters
        scale_factor = 2**16 / 10
        depth = Image.fromarray(np.uint16(self.depths[idx] * scale_factor))
        depth.save(fp=save_path, format='png')

    def _save_semseg(self, base_path, idx, digits=4):
        save_path = os.path.join(base_path, 'semseg', 'data', str(idx).rjust(digits, '0') + '.png')
        semseg = Image.fromarray(np.uint8(self.labels[idx]), mode='P')
        semseg.putpalette(self.semseg_palette())
        semseg.save(fp=save_path, format='png')

    def _save_normal(self, base_path, idx, digits=4):
        save_path = os.path.join(base_path, 'normal', 'data', str(idx).rjust(digits, '0') + '.png')
        normals = Image.fromarray(self.normals[idx])
        normals.save(fp=save_path, format='png')

    def _save_mask(self, base_path, idx, digits=4):
        save_path = os.path.join(base_path, 'mask', 'data', str(idx).rjust(digits, '0') + '.png')
        mask = Image.fromarray(self.masks[idx])
        mask.save(fp=save_path, format='png')

    def _save_subset(self, path, indices):
        """ Save a specified subset of the data at a given path
        """
        for idx in tqdm(indices):
            self._save_rgb(path, idx)
            self._save_depth(path, idx)
            self._save_semseg(path, idx)
            if self.normals is not None:
                self._save_normal(path, idx)
            if self.masks is not None:
                self._save_mask(path, idx)

    def _create_subdirs(self, path):
        os.makedirs(os.path.join(path, "rgb", "data"), exist_ok=True)
        os.makedirs(os.path.join(path, "depth", "data"), exist_ok=True)
        os.makedirs(os.path.join(path, "semseg", "data"), exist_ok=True)
        if self.normals is not None:
            os.makedirs(os.path.join(path, "normal", "data"), exist_ok=True)
        if self.masks is not None:
            os.makedirs(os.path.join(path, "mask", "data"), exist_ok=True)

    @staticmethod
    def _splits(splits_path):
        """ Splits the dataset into a test set and training set
        """
        splits = loadmat(splits_path)

        train_splits = splits['trainNdxs'] - 1
        test_splits = splits['testNdxs'] - 1

        train_idx = [elem.item() for elem in train_splits]
        test_idx = [elem.item() for elem in test_splits]

        return train_idx, test_idx

    @staticmethod
    def _transpose_3d_from_mat(data):
        """ Transposes for .mat array format to numpy array format
        """
        elem_list = [np.transpose(elem, (2, 1, 0)) for elem in data]
        elems = np.stack(elem_list, axis=0)
        return elems

    @staticmethod
    def _transpose_2d_from_mat(data):
        """ Transposes for .mat array format to numpy array format
        """
        elem_list = [np.transpose(elem, (1, 0)) for elem in data]
        elems = np.stack(elem_list, axis=0)
        return elems

    def __get_arrs(self, label_map=None):
        """ Gets the images, depths, labels and label_instances as numpy arrays
        """
        imgs = self._transpose_3d_from_mat(self.nyuv2['images'])
        depths = self._transpose_2d_from_mat(self.nyuv2['depths'])
        labels = self._transpose_2d_from_mat(self.nyuv2['labels'])

        if label_map is not None:
            labels = np.vectorize(label_map.get)(labels)

        return imgs, depths, labels

    def __read_label_map(self, path_map):
        f_map = loadmat(path_map)
        map_class = f_map['mapClass'][0]

        dict_map = {0: 0}
        for ori_id, mapped_id in enumerate(map_class, start=1):
            dict_map[ori_id] = mapped_id
        return dict_map

    @staticmethod
    def semseg_palette():
        """Obtained using Seaborn

        palette = sns.color_palette("hls", 40) with rounding, background is set to black
        """
        palette = [
            (0.0, 0.0, 0.0), (0.86, 0.449, 0.34), (0.86, 0.527, 0.34), (0.86, 0.605, 0.34),
            (0.86, 0.683, 0.34), (0.86, 0.761, 0.34), (0.86, 0.839, 0.34), (0.803, 0.86, 0.34),
            (0.725, 0.86, 0.34), (0.647, 0.86, 0.34), (0.569, 0.86, 0.34), (0.491, 0.86, 0.34),
            (0.413, 0.86, 0.34), (0.34, 0.86, 0.345), (0.34, 0.86, 0.423), (0.34, 0.86, 0.501),
            (0.34, 0.86, 0.579), (0.34, 0.86, 0.657), (0.34, 0.86, 0.735), (0.34, 0.86, 0.813),
            (0.34, 0.829, 0.86), (0.34, 0.751, 0.86), (0.34, 0.673, 0.86), (0.34, 0.595, 0.86),
            (0.34, 0.517, 0.86), (0.34, 0.439, 0.86), (0.34, 0.361, 0.86), (0.397, 0.34, 0.86),
            (0.475, 0.34, 0.86), (0.553, 0.34, 0.86), (0.631, 0.34, 0.86), (0.709, 0.34, 0.86),
            (0.787, 0.34, 0.86), (0.86, 0.34, 0.855), (0.86, 0.34, 0.777), (0.86, 0.34, 0.699),
            (0.86, 0.34, 0.621), (0.86, 0.34, 0.543), (0.86, 0.34, 0.465), (0.86, 0.34, 0.387),
        ]

        # Flatten the palette values and convert to int
        palette = [int(255 * val) for sublist in palette for val in sublist]

        return palette


def get_surface_normals(path: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Obtains arrays of surface normals and normals mask arrays from input image

    Args:
        path (str): path of the folder containing folders of normals and masks

    Returns:
        (tuple): tuple containing:
            masks (numpy.ndarray): array of image masks
            normals (numpy.ndarray): list of normals
    """

    masks_path = os.path.join(path, "masks")
    normals_path = os.path.join(path, "normals")

    masks_files = sorted([os.path.join(masks_path, file) for file in os.listdir(masks_path) if file.endswith(".png")])
    normals_files = sorted([os.path.join(normals_path, file) for file in os.listdir(normals_path) if file.endswith(".png")])

    masks = np.stack([np.array(Image.open(file)) for file in masks_files], axis=0)
    normals = np.stack([(np.array(Image.open(file))) for file in normals_files], axis=0)

    return masks, normals


if __name__ == "__main__":
    args = get_args()

    print("Preparing the dataset...")
    dataset = NYUv2Preprocessing(dataset_path=args.dataset_path, splits_path=args.splits_path,
                                 class_mapping_path=args.class_mapping_path, normals_path=args.normals_path)

    train_save_path = os.path.join(args.save_path, "train")
    os.makedirs(train_save_path, exist_ok=True)
    print(f"Saving training data to {train_save_path}")
    dataset.save(path=train_save_path, subset="train")

    test_save_path = os.path.join(args.save_path, "test")
    os.makedirs(test_save_path, exist_ok=True)
    print(f"Saving test data to {test_save_path}")
    dataset.save(path=test_save_path, subset="test")

    print("Done!")
