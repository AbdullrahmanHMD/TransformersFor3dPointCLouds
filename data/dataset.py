from tkinter import E
import torch
import numpy as np
import point_cloud_utils as pcu

import os
from torch.utils import data
import torch

arr = []

class ModelNet40(data.Dataset):
    
    def __init__(self, dataset_path, test=False, sample_size=1024, sampling='fps'):
        """
        The constructor of the class. Once initialized, this constructor
        creates a list of the pathes of all the data points of ModelNet40.
        Args:
            dataset_path (str): The absolute path of the dataset
            test (bool, optional): A boolean to specify whether to load the
                                    test set or the train set.
            sample_size (int, optional): The size of the point cloud data
                                    will be reduced to.
            sampling (str, optional): A string that specifies the sampling method
                                    . fps: Fartherst point sampling.
                                    . uni: Uniform sampling.
                                    . uni-sph: Uniform sampling with scaling to
                                                unit sphere.
                                    . non: No sampling
        """
        self.SAMPLE_SIZE = sample_size
        self.sample = sampling
        self.data_points_paths = self.format_data(dataset_path=dataset_path, test=test)
            
    
    def __getitem__(self, index):
        """
        This method loads data points on demand from the files of the dataset.
        Args:
            index (int): The index of a data point to be loaded.

        Returns:
            vector (ndarray):   The data of the specified point cloud.       
            label (int):        The label mapped to an integer
            label_txt (str):    The label as a text.
        """
        item_path, label, label_txt = self.data_points_paths[index]
        vector, _ = pcu.load_mesh_vf(item_path)
        
        if vector.size == 0:
            return None
                
        if self.sample == 'fps':
            try:
                vector = self.fps(vector, self.SAMPLE_SIZE)
            except ValueError:
                # arr.append(self.data_points_paths[index])
                return None

            
        elif self.sample == 'uni':
            vector = self.uniform_sampling(vector, self.SAMPLE_SIZE)
        
        elif self.sample == 'non':
            return torch.tensor(vector), torch.tensor(label), label_txt
        
        elif self.sample == 'uni-sph':
            vector = self.uniform_sampling(vector, self.SAMPLE_SIZE)
            vector = self.rescale_to_unit_sphere(vector)
            
        return torch.tensor(vector), torch.tensor(label), label_txt
    
    def __len__(self):
        """
        Returns:
            int: The length of the whole dataset.
        """
        return len(self.data_points_paths)
    
    def format_data(self, dataset_path : str, test : bool):
        """
        Loads the data points' pathes along with their labels
        into an array and returns it.

        Args:
            dataset_path (str): The absolute path to the dataset.
            test (bool): A boolean that specifies whether to load
                        the test or the train set.

        Returns:
            ndarray: An array with all the data (train or test) along
                    with their lebels.
        """
        objects = os.listdir(dataset_path)
        data_points_list = []
        
        for i, obj in enumerate(objects):
            datapoint_path = os.path.join(dataset_path, obj)
            
            if test:
                datapoint_path = os.path.join(datapoint_path, 'test')
            else:
                datapoint_path = os.path.join(datapoint_path, 'train')
            
            data_points = os.listdir(datapoint_path)
            
            for datum in data_points:
                datum_path = os.path.join(datapoint_path, datum)
                data_points_list.append((datum_path, i, obj))
                
        return data_points_list


    def uniform_sampling(self, vec, sample_size=1024):
        """
        Uniformly samples an input point cloud.

        Args:
            vec (ndarray): The point cloud to be sampled
            sample_size (int, optional): The size of the point cloud
                                after sampling. Defaults to 1024.
        Returns:
            ndarray: The sampled point cloud
        """
        sampling_mask = np.random.randint(0, vec.shape[0], size=sample_size)
        vec = vec[sampling_mask]
        return vec
    
    
    def rescale_to_unit_sphere(self, vec):
        """
        Rescales a point cloud into unit sphere.
        Args:
            vec (ndarray): The point cloud to be rescaled

        Returns:
            ndarray: The rescaled point cloud.
        """
        R = 1
        norms = np.linalg.norm(vec, axis=1).reshape((vec.shape[0], 1))
        rescaled_vecs = vec * (R / norms)
        
        return rescaled_vecs
    
    # ===================================================
    # === Acknowledegments ==============================

    # NOTE: The code below (fps method) was taken from the
    # following resource:
    #
    # https://minibatchai.com/ai/2021/08/07/FPS.html
    
    def fps(self, points, n_samples):
        """
        points: [N, 3] array containing the whole point cloud
        n_samples: samples you want in the sampled point cloud typically << N 
        """
        points = np.array(points)
        
        if points.shape[0] <= n_samples:
            return points
        
        # Represent the points by their indices in points
        points_left = np.arange(len(points)) # [P]

        # Initialise an array for the sampled indices
        sample_inds = np.zeros(n_samples, dtype='int') # [S]

        # Initialise distances to inf
        dists = np.ones_like(points_left) * float('inf') # [P]

        # Select a point from points by its index, save it
        selected = 0
        sample_inds[0] = points_left[selected]

        # Delete selected 
        points_left = np.delete(points_left, selected) # [P - 1]

        # Iteratively select points for a maximum of n_samples
        for i in range(1, n_samples):
            # Find the distance to the last added point in selected
            # and all the others
            last_added = sample_inds[i-1]
            
            dist_to_last_added_point = (
                (points[last_added] - points[points_left])**2).sum(-1) # [P - i]

            # If closer, updated distances
            dists[points_left] = np.minimum(dist_to_last_added_point, 
                                            dists[points_left]) # [P - i]

            # We want to pick the one that has the largest nearest neighbour
            # distance to the sampled points
            selected = np.argmax(dists[points_left])
            sample_inds[i] = points_left[selected]

            # Update points_left
            points_left = np.delete(points_left, selected)

        return points[sample_inds]
    
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)