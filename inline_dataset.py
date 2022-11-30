import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import torch

# get group ids
def get_groups(image):
    id = image['id']
    group_set = set()
    try:
        for point in image.find_all('points'):
            group_set.add(point['group_id'])
        for box in image.find_all('box'):
            group_set.add(box['group_id'])
    except Exception as e:
        print( f'image {id} has no group ids' )
        return None

    return group_set

def create_pepper(group_id, image):

    pepper_bbox = None

    boxes = {
        'stem' : None,
        'body' : None,
    }

    kps = {
        'right_shoulder' : [0,0,0],
        'left_shoulder' : [0,0,0],
        'center_shoulder' : [0,0,0],
        'stem' : [0,0,0],
        'body' : [0,0,0],
    }

    # visibility:
    #   0: not labeled
    #   1: labeled, but not visible
    #   2: labeled, visible

    for point in image.find_all('points'):
        if point['group_id'] != group_id:
            continue
        label = point['label']
        points = point['points'] # should only be 1 in here
        xy_str = points.split(',')
        x = float(xy_str[0])
        y = float(xy_str[1])
        kps[label] = [x, y, 2]

    for box in image.find_all('box'):
        if box['group_id'] != group_id:
            continue
        label = box['label']
        boxes[label] = [
            float(box['xtl']),
            float(box['ytl']),
            float(box['xbr']),
            float(box['ybr']),
        ]    


    if boxes['stem'] is not None and boxes['pepper'] is not None:
        pepper_bbox = [
            min(boxes['stem'][0],boxes['pepper'][0]),
            min(boxes['stem'][1],boxes['pepper'][1]),
            max(boxes['stem'][2],boxes['pepper'][2]),
            max(boxes['stem'][3],boxes['pepper'][3]),
        ]

    elif boxes['pepper'] is not None:
        pepper_bbox = boxes['pepper']
    elif boxes['stem'] is not None:
        pepper_bbox = boxes['stem']
    

    area = (pepper_bbox[2] - pepper_bbox[0]) * (pepper_bbox[3] - pepper_bbox[1])

    
    kp_array = []
    kp_array.append(kps['right_shoulder'])
    kp_array.append(kps['left_shoulder'])
    kp_array.append(kps['center_shoulder'])
    kp_array.append(kps['stem'])
    kp_array.append(kps['body'])

    annotation = {
        "id" : group_id, # int,
        "image_id" : image['id'],# int,
        "category_id" : 1, # int,
        "segmentation" : None, # RLE or [polygon],
        "area" : area, # float,
        "bbox" : pepper_bbox, # [x,y,width,height],
        "iscrowd" : 0, #0 or 1,
        "keypoints": kp_array,# [x1,y1,v1,...],
        "num_keypoints" : 5, # : int,
    }

    return annotation


# pass beaustiful soup image object
def parse_image(image):

    image_data = {
        "id" : image['id'], # int,
        "width" : image['width'], # int,
        "height" : image['height'], # int,
        "file_name" : image['name'], #str,
        "license" : 0, # int,
        "flickr_url" : "", # str,
        "coco_url" : "", #str,
        "date_captured": None, #datetime,
    }

    groups = get_groups(image)
    if groups is None:
        return image_data, None

    annotations = []
    for group_id in groups:
        pepper = create_pepper(group_id, image)
        annotations.append(pepper)



    return image_data, annotations


def parse_xml(xml_path):
    with open(xml_path, 'r') as xml_doc:
        soup = BeautifulSoup(xml_doc, 'lxml')
        #print(soup.prettify())

        anno = soup.find('annotation')
        #print(anno.prettify())

        filename = anno.filename.contents[0]

        size = anno.size

        width = float(size.width.contents[0])
        height = float(size.height.contents[0])

        objs = anno.find_all('object')
        boxes= []
        areas = []
        for obj in objs:
            #print(obj.prettify())
            #print(obj.xmin.contents[0])
            box = [
                float(obj.xmin.contents[0]), 
                float(obj.ymin.contents[0]),
                float(obj.xmax.contents[0]),
                float(obj.ymax.contents[0]),
            ]

            area = (box[2] - box[0]) * (box[3] - box[1])
            
            boxes.append(box)
            areas.append(area)

        return boxes, filename, areas

# parse the annotations file and return a organized object
def parse_annotations(folder_path):

    annotation_folder = os.path.join(folder_path, 'annotations')
    image_folder = os.path.join(folder_path, 'images')
    xml_list = os.listdir(annotation_folder)

    data_list = []
    for xml_file in xml_list:
        xml_path = os.path.join(annotation_folder, xml_file)
        boxes, image_file, areas = parse_xml(xml_path)
        image_path = os.path.join(image_folder, image_file)
        
        data_list.append(
            {
                'image_path' : image_path, 
                'boxes' : boxes,
                'area' :  areas,
            }
        )

    return data_list

class PepperDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data = parse_annotations(data_folder)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image_path = self.data[idx]['image_path']
        boxes = self.data[idx]['boxes']
        area = self.data[idx]['area']

        image = read_image(image_path)
        image = image / 255.0

        num_objs = len(boxes)

        # convert to tensors
        labels = torch.ones((num_objs,), dtype=torch.int64)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        area = torch.as_tensor(area, dtype=torch.float32)
        #keypoints = torch.tensor(kps, dtype=torch.float32)
        image_id = torch.tensor([idx])



        # if self.target_transform:
        #     label = self.target_transform(label)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks_tensor
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        #target["keypoints"] = keypoints

        if self.transform:
            image, target = self.transform(image, target)
        
        return image, target


import random
import torch

from torchvision.transforms import functional as F
from torchvision import transforms  as T

# import albumentations as A


def _flip_pepper_keypoints(kps, width):
    flip_inds = [1, 0, 2, 3, 4]
    flipped_data = kps[:, flip_inds]
    #print(flipped_data)
    flipped_data[..., 0] = width - flipped_data[..., 0]
    #print(flipped_data)
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data

def _offset_keypoints(kps, off_y, off_x):
    # print('kps \n', kps[:, :, 1], '\n')
    kps[:, :, 1] = kps[:, :, 1] - off_y
    kps[:, :, 0] = kps[:, :, 0] - off_x
    
    #print('mask \n', kps[:, :, 2], '\n')
    #mask = (kps[:, :, 2] != 0).nonzero()
    #print(mask)
    
    #print('zeros \n', kps[mask], '\n')
    
    #for m in mask:
    #    kps[mask[0], mask[1], :] = 0.0
    
    return kps


def _normalize_keypoints(kps, height, width):
    # print('kps \n', kps[:, :, 1], '\n')
    kps[:, :, 1] = kps[:, :, 1] / height
    kps[:, :, 0] = kps[:, :, 0] / width
    
    return kps

def _normalize_bboxes(boxes, height, width):
    print(boxes)
    # print('kps \n', kps[:, :, 1], '\n')
    boxes[:,  [1,3]] = boxes[:, [1,3]] / height
    boxes[:, [0,2]] = boxes[:, [0,2]] / width
    
    return boxes

def _normalize_area(area, height, width):
    # print('kps \n', kps[:, :, 1], '\n'
    
    return area / height / width
 
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_pepper_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size
        self.crop = T.CenterCrop(size)
        # this is a tuple of height, width

    def __call__(self, image, target):
        height, width = image.shape[-2:]
        off_y = (height - self.size[0]) / 2
        off_x = (width - self.size[1]) / 2
        image = self.crop(image)
        bbox = target["boxes"]
        #print('before', bbox)
        bbox[:, [0,2]] = bbox[:, [0,2]] - off_x
        bbox[:, [1,3]] = bbox[:, [1,3]] - off_y
        #print('after', bbox)
        target["boxes"] = bbox
        if "masks" in target:
            target["masks"] = self.crop(target["masks"])
        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = _offset_keypoints(keypoints, off_y, off_x)
            target["keypoints"] = keypoints
        return image, target

class Resize(object):
    def __init__(self, size):
        self.size = size
        self.resize = T.Resize(size)
        # this is a tuple of height, width

    def __call__(self, image, target):

        colors, height, width = image.shape
        
        image = self.resize(image)
        bbox = target["boxes"]
        bbox = _normalize_bboxes(bbox, height, width)
        target["boxes"] = bbox


        area = target["area"]
        area = _normalize_area(area, height, width)
        target["area"] = area

        if "masks" in target:
            target["masks"] = self.resize(target["masks"])
        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = _normalize_keypoints(keypoints, height, width)
            target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target



def get_transform(train=True):
    transforms = []
    #transforms.append(CenterCrop((400,1920)))
        
    if train:

      transforms.append(RandomHorizontalFlip(0.5))

    
    return Compose(transforms)

if __name__ == "__main__":
    dataset = PepperDataset('data', transform=get_transform(train=False))
    img, target = dataset[25]

    print(img.shape)
    print(target)