import glob
import os
from random import sample
import numpy as np
import h5py
from torch.utils.data import Dataset
import dataAugmenter


class NormalizedModelNet40(Dataset):
    
    CLASSES = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car'
               , 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot'
               , 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor'
               , 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa'
               , 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe'
               , 'xbox']
    
    CLASS_MAPPING = dict(zip(range(0, len(CLASSES)), CLASSES))
    
    def __init__(self, dataset_path, partition='train', sample_size=None, sampling_method='fps'):
        self.CLASSES = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car'
            , 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot'
            , 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor'
            , 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa'
            , 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe'
            , 'xbox']
        
        
        self.dataset_path = dataset_path
        self.aug_data_path = os.path.join(self.dataset_path, 'augmented_data.h5')

        self.partition = partition
        
        self.sample_size = sample_size
        self.sampling_method = sampling_method
        
        self.CLASS_MAPPING = dict(zip(range(0, len(self.CLASSES)), self.CLASSES))
        self.data, self.labels = self.load_data()
        
        if self.need_augment():
            self.generate_augmented_data()
    
    
    def __len__(self):
        return self.labels.shape[0]
    
    
    def __getitem__(self, index):
        data = self.data[index]
        
        if self.sampling_method == 'fps':
            if self.sample_size != None:
                data = self.fps(data, self.sample_size)    
            else:
                raise Exception(f"Invalid sampling size: Given {self.sample_size} expected a number greater than 0")        
            
        if self.sampling_method == 'uni':
            if self.sample_size != None:
                data = self.uniform_sampling(data, self.sample_size)    
            else:
                raise Exception(f"Invalid sampling size: Given {self.sample_size} expected a number greater than 0")        
        
        return data, self.labels[index], self.CLASS_MAPPING[self.labels[index]]
    
    
    def augmentation_count(self):
        # Returns the number of needed augmentation for each class.
        data_dist = np.bincount(self.labels)
        max_instance = np.max(data_dist)
        
        return max_instance - data_dist

    
    def class_indicies(self, class_num):
        # Returns the instances indicies of a given class.
        indicies = [i for i, x in enumerate(self.labels) if x == class_num]
        return indicies
    
    
    def generate_augmented_data(self):
        
        self._augment_data()
        self.save_to_h5()
    
    def need_augment(self):
        return  os.path.exists(self.aug_data_path) == False
        
    
    def _augment_data(self):
        
      
        augmentation_nums = self.augmentation_count()
        x, y = [], []

        for i, n in enumerate(augmentation_nums):
            class_indices = self.class_indicies(i)
            N = len(class_indices)
            for _ in range(n):
                random_idx = np.random.randint(0, N, 1)
                point_cloud = self.data[random_idx]
                rotated_data = dataAugmenter.rotate_point_cloud(point_cloud)
                rotated_data = dataAugmenter.rotate_perturbation_point_cloud(rotated_data)
                jittered_data = dataAugmenter.random_scale_point_cloud(rotated_data)
                jittered_data = dataAugmenter.shift_point_cloud(jittered_data)
                jittered_data = dataAugmenter.jitter_point_cloud(jittered_data)
                rotated_data = jittered_data
                x.append(rotated_data)
                y.append(i)
        self.data = np.append(self.data, x, axis=0)
        self.labels = np.append(self.labels, y)

    
    def save_to_h5(self):
        """Writing data to an h5 file"""
     
        hf = h5py.File(self.aug_data_path, 'w')
        hf.create_dataset('data', data=self.data)
        hf.create_dataset('labels', data=self.labels)
        hf.close()
        
        
    
    def class_distribution(self):
        distribution = np.bincount(self.labels)
        return distribution
    
    # NOTE: This method was taken from:
    def load_data(self):
        all_data = []
        all_label = []
        
        for h5_name in glob.glob(os.path.join(self.dataset_path, 'ply_data_%s*.h5'%self.partition)):
            f = h5py.File(h5_name, 'r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label , axis=0).reshape(-1)
            
        return all_data, all_label

    
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
    

    def calculate_class_weights(self):
        
        bin_count = np.bincount(self.labels)
        dataset_size = len(self.labels)
        
        class_weights = [1 - (class_/dataset_size) for class_ in bin_count]
        
        return class_weights
    
    
    def class_indicies_distribution(self):
        indicies_distribution = {}
        labels = self.labels.reshape(-1)
        num_classes = np.unique(labels)
        for class_ in num_classes:
            mask = class_ == labels
            indicies_distribution[class_] = [i for i, x in enumerate(mask) if x]
        return indicies_distribution
    
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