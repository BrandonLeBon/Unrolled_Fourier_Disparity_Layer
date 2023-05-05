import os

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from einops import rearrange



'''
------------------------------------------------------------------------
CLASS LightFieldDataset
    A Pytorch dataset used to load and pre-process a light field dataset
------------------------------------------------------------------------
'''
class LightFieldDataset(Dataset):
    def __init__(self, directories, transform, angular_dims):
        self.light_field_directories_list = self.light_fields_from_directories(directories)
        self.number_of_light_fields = len(self.light_field_directories_list)
        self.transform = transform
        self.angular_dims = angular_dims
        
    ''' Return the list of leaf directories from a root directory, assuming these leaves correspond to light field directories. '''
    def light_fields_from_directories(self, directories):
        light_field_directories_list = []
        for directory in directories:
            for root, dirs, files in os.walk(directory):
                if (not dirs or all("." in s for s in dirs)) and "." not in root:
                    light_field_directories_list.append(root) 
                    
        return light_field_directories_list
    
    ''' Create a tensor of size (C,V,U,H,W) from image files in a light field directory '''
    def tensor_from_light_field_directory(self, light_field_directory):
        to_tensor = transforms.Compose([transforms.ToTensor()])
    
        view_name_list = [file for file in os.listdir(light_field_directory) if ".png" in file]
        sorted_view_name_list = sorted(view_name_list)
        last_view_name = sorted_view_name_list[-1]
        last_view_name_split = last_view_name.split(".")[0].split("_")
        
        max_u = int(last_view_name_split[-1])
        max_v = int(last_view_name_split[-2])
        
        center_u = int(max_u/2.0)
        center_v = int(max_v/2.0)
        views_v = []
        for u in range(center_u-int(self.angular_dims[1]/2.0),center_u-int(self.angular_dims[1]/2.0) + self.angular_dims[0]):
            views_u = []
            for v in range(center_v-int(self.angular_dims[0]/2.0),center_v-int(self.angular_dims[0]/2.0) + self.angular_dims[1]): 
                index = max_u * u + v
                file_name = os.path.join(light_field_directory, sorted_view_name_list[index])
                view = self.load_image(file_name)
                view = to_tensor(view).unsqueeze(1)
                views_u.append(view)
            views_u = torch.cat(views_u, dim=1).unsqueeze(1)
            views_v.append(views_u)
        light_field_tensor = torch.cat(views_v, dim=1)
        return light_field_tensor
    
    ''' Load an image file as a PIL RGB image '''
    def load_image(self, file_name):
        with open(file_name,'rb') as f:
            image = Image.open(f).convert("RGB")
        return image
        
    ''' Apply a pre-process as Pytorch transforms on a (C,V,U,H,W) light field tensor and return a (H,W,V,U,C) light field tensor '''
    def apply_transform(self, light_field, coordinates):
        if len(self.transform.transforms) > 0:
            reduced_light_field = light_field.view(-1, *light_field.shape[3:])
            reduced_coordinates = coordinates.view(-1, *coordinates.shape[3:])
            i, j, h, w = transforms.RandomCrop.get_params(reduced_light_field, output_size=(64, 64))
            reduced_light_field = TF.crop(reduced_light_field, i, j, h, w)
            reduced_coordinates = TF.crop(reduced_coordinates, i, j, h, w)
            light_field = reduced_light_field.view([light_field.size(0),light_field.size(1),light_field.size(2),reduced_light_field.size(1),reduced_light_field.size(2)]).permute([3,4,1,2,0])
            coordinates = reduced_coordinates.view([coordinates.size(0),coordinates.size(1),coordinates.size(2),reduced_coordinates.size(1),reduced_coordinates.size(2)]).permute([3,4,1,2,0])
        else:
            light_field = light_field.permute([3,4,1,2,0])
            coordinates = coordinates.permute([3,4,1,2,0])
        return light_field, coordinates
    
    ''' Return the number of light field in the dataset '''    
    def __len__(self):
        return self.number_of_light_fields

    ''' Load a light field from an index. Return the light field name and a (H,W,V,U,C) tensor '''
    def __getitem__(self, item):
        light_field_name = self.light_field_directories_list[item]
        light_field = self.tensor_from_light_field_directory(light_field_name)
        light_field = self.transform(light_field)
        return light_field, light_field_name