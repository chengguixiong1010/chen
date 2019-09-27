import os
import os,sys, shutil
import cv2
import json
import numpy as np
import datetime
import argparse

'''r = [0,20,3,4,5,6,7,8,9,10]
for i in range(3):
    r.pop(3)'''
path_6print_json =   '/home/dabai/datasets/aichallenger/6points_annos_temp'#'./6points-annos_temp/'
path_ai_json = '../human_pose/ai_challenger_keypoint_train_annotations_20170909/keypoint_train_annotations_20170909.json'
#u = open(path_ai_json,'w+')
with open(path_ai_json) as f:
    kpt_ai_json = json.load(f)
    #for ss,val in enumerate(kpt_ai_json):
        #print(val['image_id'] == img_kpt6.split('.')[0] )

    jsons_list = os.listdir(path_6print_json)
    for dir_list6 in jsons_list:
        path6 = os.path.join(path_6print_json,dir_list6)
        with open(path6) as f:
            kpt_6_json = json.load(f)
            for img_kpt6 in kpt_6_json :
                print(img_kpt6)

                for ss,val in enumerate(kpt_ai_json):
                    #print(img_kpt6.split('.')[0])
                    if (val['image_id'] == img_kpt6.split('.')[0]):
                        print(1)


                #print(kpt_6_json[img_kpt6])
                for people in kpt_6_json[img_kpt6]['people']:
                    for i in range(0, len(people['pose_keypoints_2d']), 3):
                        print(2)

            
