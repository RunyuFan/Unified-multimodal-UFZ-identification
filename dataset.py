import os
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import shapefile as shp
import pandas as pd
from torchvision import transforms


def MyLoader(file_path):
    with rasterio.open(file_path) as src:
        image = src.read()
        if image.dtype == 'uint16':  # 发现是 uint16 数据
            image = image.astype('float32')  # 归一化到 [0, 1]
    return image


class MyDataset(Dataset):
    def __init__(self, path, shp_path, images_path, transform=None, target_transform=None, loader=None):
        file = {}
        read_shp = shp.Reader(shp_path)
        for feature in read_shp.iterRecords():
            file[int(feature['idx'])] = {}
            if int(feature['type_id']) in [4, 5, 6, 7]:
                file[int(feature['idx'])]['label'] = 3
            else:
                file[int(feature['idx'])]['label'] = int(feature['type_id']) - 1

        for f in os.listdir(path):
            name = f.split('.')[0]
            csv_path = os.path.join(path, f)
            df = pd.read_csv(csv_path, index_col=False)
            for idx in file:
                try:
                    row = df[df['idx'] == idx].iloc[0, 2:].values
                    file[idx][name] = row
                except IndexError:
                    zero_array = np.zeros(len(df.columns) - 2)
                    file[idx][name] = zero_array

        for idx in file:
            file[idx]['image_path'] = f'{images_path}\\{str(idx)}.tif'

        file = {new_key: file[old_key] for new_key, old_key in enumerate(file.keys())}

        self.file = file
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        doc_e = torch.from_numpy(self.file[index]['doc_embeddings']).float()
        graph_e = torch.from_numpy(self.file[index]['graph_embeddings']).float()
        poi_l = torch.from_numpy(self.file[index]['poi_local_density']).float()
        poi_g = torch.from_numpy(self.file[index]['poi_global_density']).float()
        sv_f = torch.from_numpy(self.file[index]['street_view']).float()
        label = self.file[index]['label']
        rs_f = self.transform(torch.from_numpy(MyLoader(self.file[index]['image_path'])).float())
        return doc_e, graph_e, poi_l, poi_g, rs_f, sv_f, label

    def __len__(self):
        return len(self.file)


class preDataset(Dataset):
    def __init__(self, path, shp_path, images_path, transform=None, target_transform=None, loader=None):
        file = {}
        read_shp = shp.Reader(shp_path)
        for feature in read_shp.iterRecords():
            file[int(feature['idx'])] = {}
            file[int(feature['idx'])]['key'] = int(feature['idx'])

        for f in os.listdir(path):
            name = f.split('.')[0]
            csv_path = os.path.join(path, f)
            df = pd.read_csv(csv_path, index_col=False)
            for idx in file:
                try:
                    row = df[df['idx'] == idx].iloc[0, 2:].values
                    file[idx][name] = row
                except IndexError:
                    zero_array = np.zeros(len(df.columns) - 2)
                    file[idx][name] = zero_array

        for idx in file:
            file[idx]['image_path'] = f'{images_path}\\{str(idx)}.tif'

        file = {new_key: file[old_key] for new_key, old_key in enumerate(file.keys())}

        self.file = file
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        doc_e = torch.from_numpy(self.file[index]['doc_embeddings']).float()
        graph_e = torch.from_numpy(self.file[index]['graph_embeddings']).float()
        poi_c = torch.from_numpy(self.file[index]['poi_counts_vector']).float()
        sv_f = torch.from_numpy(self.file[index]['street_view']).float()
        rs_f = self.transform(torch.from_numpy(MyLoader(self.file[index]['image_path'])).float())
        keys = self.file[index]['key']
        return doc_e, graph_e, poi_c, rs_f, sv_f, keys

    def __len__(self):
        return len(self.file)