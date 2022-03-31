import os

import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

from .transforms import task_transform


class TaskonomyDataset(Dataset):
    def __init__(self, 
                 data_root,
                 tasks, 
                 split='train', 
                 variant='tiny',
                 image_size=256,
                 max_images=None):
        """
        Taskonomy dataloader.

        Args:
            data_root: Root of Taskonomy data directory
            tasks: List of tasks. Any of ['rgb', 'depth_euclidean', 'depth_zbuffer',
                'edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d', 'normal',
                'principal_curvature', 'reshading', 'mask_valid'].
            split: One of {'train', 'val', 'test'}
            variant: One of {'debug', 'tiny', 'medium', 'full', 'fullplus'}
            image_size: Target image size
            max_images: Optional subset selection
        """
        super(TaskonomyDataset, self).__init__()
        self.data_root = data_root
        self.tasks = tasks
        self.split = split
        self.variant = variant
        self.image_size=image_size
        self.max_images = max_images
        
        self.image_ids = pd.read_csv(
            os.path.join(os.path.dirname(__file__), 'splits', f'{self.variant}_{self.split}.csv')
        ).to_numpy()
        
        if isinstance(self.max_images, int):
            self.image_ids = self.image_ids[:self.max_images]
        
        print(f'Initialized TaskonomyDataset with {len(self.image_ids)} images from variant {self.variant} in split {self.split}.')
        
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        
        # building / point / view
        building, point, view = self.image_ids[index]
        
        result = {}
        for task in self.tasks:
            task_id = 'depth_zbuffer' if task == 'mask_valid' else task
            path = os.path.join(
                self.data_root, task, building, f'point_{point}_view_{view}_domain_{task_id}.png'
            )
            img = Image.open(path)
            # Perform transformations
            img = task_transform(img, task=task, image_size=self.image_size)
            result[task] = img

        return result
