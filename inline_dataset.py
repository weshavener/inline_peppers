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

        objs = anno.find_all('object')
        boxes= []
        areas = []
        for obj in objs:
            #print(obj.prettify())
            #print(obj.xmin.contents[0])
            box = [
                int(obj.xmin.contents[0]), 
                int(obj.ymin.contents[0]),
                int(obj.xmax.contents[0]),
                int(obj.ymax.contents[0])
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

if __name__ == "__main__":
    dataset = PepperDataset('data')
    img, target = dataset[25]

    print(target)