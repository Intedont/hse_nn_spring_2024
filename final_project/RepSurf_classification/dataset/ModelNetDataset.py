import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

#DATA_DIR = '/home/madusov/nn_project/RepSurf/classification/data/modelnet40_ply_hdf5_2048'
#CORRUPTED_DATA_DIR = '/home/madusov/nn_project/RepSurf/classification/data/modelnet_c'

def load_data(data_dir, partition):
    # partition = train или test - значит работаем с датасетом modelnet40
    # partition = clean, scale, jitter и тд, значит работаем с pointcloud_c
    if(partition == 'train' or partition == 'test'):
        all_data = []
        all_label = []
        for h5_name in glob.glob(os.path.join(data_dir, 'ply_data_%s*.h5'%partition)):
            # print(f"h5_name: {h5_name}")
            f = h5py.File(h5_name,'r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data, all_label
    else:
        h5_name = os.path.join(data_dir, partition + '.h5')
        f = h5py.File(h5_name, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        return data, label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


class ModelNet40(Dataset):
    def __init__(self, data_dir, num_points=2048, split='train'):
        self.data, self.label = load_data(data_dir, split)
        self.num_points = num_points
        self.partition = split        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud.transpose(1,0), label.item()

    def __len__(self):
        return self.data.shape[0]