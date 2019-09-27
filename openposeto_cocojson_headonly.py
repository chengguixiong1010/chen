# import numpy as np
# import pandas as pd
# import  cv2
# from pycococreatortools import  *
import os, sys, shutil
# import time
import datetime
import json
import argparse
import cv2


def get_image_info(image_name):
    file_name = image_name
    id = image_name.split('.')[0]
    #id = int(id)
    return id, file_name


def get_segmentation_info(seg_path):
    with open(seg_path) as f:

        json_data = json.load(f)
        shapes = json_data['shapes']
        segmentations = []
        for i in range(len(shapes)):
            segmentation = []
            point = shapes[i]['points']
            for j in range(len(point)):
                segmentation.append(point[j][0])
                segmentation.append(point[j][1])
            segmentations.append(segmentation)

    return segmentations


def get_keypoints_info(json_path):
    with open(json_path) as f:

        json_data = json.load(f)
        pdata = json_data['people']
        pose_keypoints = pdata['pose_keypoints_2d']
        num_keypoints = 0
        keypoints = pose_keypoints
        # for i in range(3):
        #    keypoints.pop(3)
        for i in range(0, len(keypoints), 3):
            if keypoints[i] != 0 or keypoints[i + 1] != 0:
                keypoints[i + 2] = 2
                num_keypoints += 1
            else:
                keypoints[i] = 0
                keypoints[i + 1] = 0
                keypoints[i + 2] = 0
    print(num_keypoints)
    print(len(keypoints))
    return keypoints, num_keypoints


def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    image = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url
    }

    return image


def create_annotation_info(num_keypoints, keypoints, image_id, id):
    annotation = {
        "segmentation": [
            [125.12, 539.69, 140.94, 522.43, 100.67, 496.54, 84.85, 469.21,
             73.35, 450.52, 104.99, 342.65, 168.27, 290.88, 179.78, 288]],
        "num_keypoints": num_keypoints,
        "area": 47803.27955,
        "iscrowd": 0,
        "keypoints": keypoints,
        "image_id": image_id,
        "bbox": [73.35, 206.02, 300.58, 372.5],
        "category_id": 1,
        "id": id
    }

    return annotation


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir_name", type=str, default='/home/dabai/datasets/aichallenger/tmp',required=False,
                        help="image's file")
    parser.add_argument("--jsons_dir_name", type=str, default='/home/dabai/datasets/aichallenger/ai_challenger_json/',required=False,
                        help="jsons path")
    parser.add_argument("--out_json_path", type=str, default = './rrr.json',required=False,
                        help="openpose 2 coco path")
    args = parser.parse_args()
    coco_output = {}
    # size = [1920,1080]

    info = {
        "description": "",
        "url": "",
        "version": "",
        "year": 2019,
        "contributor": "",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }
    licenses = [
        {
            "id": 1,
            "name": "",
            "url": ""
        }
    ]
    categories = [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": [
                "nose1","right_shoulder2", "right_elbow3", "right_wrist4",
                "left_shoulder5", "left_elbow6", "left_wrist7", "right_hip9",
                "right_knee10", "right_ankle11", "left_hip12", "left_knee13", "left_ankle14",
                "right_eye15", "left_eye16", "right_ear17", "left_ear18",
                "hip8",
                "right_heel24", "right_bigtoe22", "right_smalltoe23",
                "left_heel21", "left_bigtoe19", "left_smalltoe20"],

            "skeleton": [
                # Nose, Neck, R hand, L hand, R leg, L leg, Eyes, Ears
                [1, 15], [15, 17], [1, 16], [16, 18],
                [2, 3], [3, 4], [2, 5], [5, 6], [6, 7],
                [8, 9], [9, 10], [10, 11], [11, 23], [11, 24], [24, 22],
                [8, 12], [12, 13], [13, 14], [14, 21], [14, 20], [19, 21],
                [2, 17], [5, 18]
                # [1, 25], [25, 8],
                # [2, 9], [5, 12]
            ]
        }
    ]
    images = []
    annotations = []

    # images

    images_dir = os.listdir(args.images_dir_name)
    images_id = []
    for image_name in images_dir:
        #print(os.path.join(args.images_dir_name, image_name))
        img = cv2.imread(os.path.join(args.images_dir_name, image_name))

        size = [img.shape[1], img.shape[0]]
        id, file_name = get_image_info(image_name=image_name)
        image = create_image_info(id, file_name, size)
        images.append(image)
        images_id.append(id)



    print("-----finished images-----")

    # annotations
    jsons_dir = os.listdir(args.jsons_dir_name)
    id = 0
    for j in range(len(jsons_dir)):
        json_file = args.jsons_dir_name + jsons_dir[j]
        image_id = jsons_dir[j].split('_')[0]
        #image_id = int(image_id)
        with open(json_file) as f:
            json_data = json.load(f)
            pdatas = json_data['people']
            for pdata in pdatas:
                num_keypoints = 0
                keypoints = pdata['pose_keypoints_2d']

                new_keypoints = keypoints

                for j in range(3):
                    new_keypoints.pop(3)

                for i in range(0, len(new_keypoints), 3):
                    # if new_keypoints[i+2] > 0.1:
                    if new_keypoints[i] != 0 or new_keypoints[i + 1] != 0:
                        new_keypoints[i + 2] = 2
                        num_keypoints += 1
                    else:
                        new_keypoints[i] = new_keypoints[i + 1] = new_keypoints[i + 2] = 0
                id += 1
                print(len(new_keypoints))

                new_keypoints0 = new_keypoints[0:3]#鼻子
                new_keypoints1 = new_keypoints[42:54]#眼睛，耳朵
                new_keypoints1.append(new_keypoints0[0])
                new_keypoints1.append(new_keypoints0[1])
                new_keypoints1.append(new_keypoints0[2])
                new_keypoints = new_keypoints1
                print(len(new_keypoints))

                annotation = create_annotation_info(num_keypoints, new_keypoints, image_id, id)
                annotations.append(annotation)
                #print(annotations)

    print("-----finished annotations-----")

    coco_output["info"] = info
    coco_output["licenses"] = licenses
    coco_output["categories"] = categories
    coco_output["images"] = images
    coco_output["annotations"] = annotations

    print("transform to coco")

    # coco = pd.DataFrame(coco_output)
    # coco_output.to_json('./openpose.json')
    with open(args.out_json_path, 'w') as f:
        json.dump(coco_output, f)
        print('write in json')

