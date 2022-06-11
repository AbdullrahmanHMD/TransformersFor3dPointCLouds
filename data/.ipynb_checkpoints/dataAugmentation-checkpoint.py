import os
import sys
import numpy as np
# import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:,idx,:]


def rotate_point_cloud(point_cloud):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cos = np.cos(rotation_angle)
    sin = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos, 0, sin],
                                [0, 1, 0],
                                [-sin, 0, cos]])
    rotated_data = torch.matmul(point_cloud, torch.from_numpy(rotation_matrix)))
    return rotated_data


def rotate_point_cloud_z(point_cloud):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cos = np.cos(rotation_angle)
    sin = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    rotated_data = torch.matmul(point_cloud, torch.from_numpy(rotation_matrix)))
    return rotated_data


def rotate_point_cloud_with_normal(normal_point_cloud):
    ''' Randomly rotate XYZ, normal point cloud.
        Input:
            normal_point_cloud: N,6, first three channels are XYZ, last 3 all normal
        Output:
            N,6, rotated XYZ, normal point cloud
    '''
    rotated_data = torch.from_numpy(normal_point_cloud)
    shape_pc = normal_point_cloud[:,0:3]
    shape_normal = normal_point_cloud[:,3:6]
    rotated_data[:,0:3] = rotate_point_cloud(shape_pc.reshape((-1, 3)))
    rotated_data[:,3:6] = rotate_point_cloud(shape_normal.reshape((-1, 3)))
    return rotated_data


def rotate_perturbation_point_cloud_with_normal(normal_point_cloud, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx6 array, original batch of point clouds and point normals
        Return:
          Nx3 array, rotated batch of point clouds
    """
    rotated_data = torch.from_numpy(normal_point_cloud)
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    shape_pc = normal_point_cloud[:,0:3]
    shape_normal = normal_point_cloud[:,3:6]
    rotated_data[:,0:3] = torch.matmul(shape_pc.reshape((-1, 3)), torch.from_numpy(R))
    rotated_data[:,3:6] = torch.matmul(shape_normal.reshape((-1, 3)), torch.from_numpy(R))
    return rotated_data



def rotate_point_cloud_by_angle(point_cloud, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    rotated_data = point_cloud
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = torch.matmul(point_cloud, torch.from_numpy(rotation_matrix))
    return rotated_data



def rotate_point_cloud_by_angle_with_normal(normal_point_cloud, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          Nx6 array, original batch of point clouds with normal
          scalar, angle of rotation
        Return:
          Nx6 array, rotated batch of point clouds iwth normal
    """
    rotated_data = normal_point_cloud
 
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    
    shape_pc = normal_point_cloud[:,0:3]
    shape_normal = normal_point_cloud[:,3:6]
    
    rotated_data[k,:,0:3] = torch.matmul(shape_pc.reshape((-1, 3)), torch.from_numpy(rotation_matrix))
    rotated_data[k,:,3:6] = torch.matmul(shape_normal.reshape((-1,3)), torch.from_numpy(rotation_matrix))
    
    return rotated_data



def rotate_perturbation_point_cloud(point_cloud, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    
    rotated_data = torch.matmul(point_cloud.reshape((-1, 3)), torch.from_numpy(R))
    return rotated_data



def jitter_point_cloud(point_cloud, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """
    N, C = point_cloud.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_data = torch.from_numpy(jittered_data)
    jittered_data += point_cloud
    return jittered_data



def shift_point_cloud(point_cloud, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, shifted batch of point clouds
    """
    N, C = point_cloud.shape
    shift = np.random.uniform(-shift_range, shift_range, (1,3))
    point_cloud += torch.from_numpy(shift)
    return point_cloud 



def random_scale_point_cloud(point_cloud, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original batch of point clouds
        Return:
            Nx3 array, scaled batch of point clouds
    """
    N, C = batch_data.shape
    scale = np.random.uniform(scale_low, scale_high, 1)
    point_cloud *= scale
    return point_cloud



def random_point_dropout(point_cloud, max_dropout_ratio=0.875):
    ''' point_cloud: Nx3 '''
    dropout_ratio =  np.random.random() * max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((point_cloud.shape[0]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        point_cloud[drop_idx,:] = point_cloud[0,:] # set to the first point
    return point_cloud

