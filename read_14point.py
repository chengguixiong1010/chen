import os,sys, shutil
import cv2
import json
import numpy as np
import datetime
import argparse
from openposeto_cocojson_headonly import get_image_info,create_image_info,create_annotation_info

def get_keypoints(ai_json_dir):
    with open(ai_json_dir) as f:
        json_f = json.load(f)
        return json_f
def if_in_bbox(bbox,keypoints):#bbox [x1,y1,x2,y2]为左上角和右下角坐标
    list_kpts = []
    for j in range(0, len(keypoints), 3):
        if keypoints[j]!=0 and keypoints[j+1]!=0:
            list_kpts.append(keypoints[j:j+3])
            p = np.vstack(list_kpts)
    #print(p)
    xmin = np.min(p[:, 0])
    ymin = np.min(p[:, 1])
    xmax = np.max(p[:, 0])
    ymax = np.max(p[:, 1])
    
    if xmin>bbox[0] and ymin>bbox[1] and xmax<bbox[2] and ymax <bbox[3]:
        return True
    else:

        return False
    
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir_name", type=str, default='/home/dabai/datasets/aichallenger/tmp/',required=False,
                        help="image's file")
    parser.add_argument("--openpose_jsons_dir_name", type=str, default='/home/dabai/datasets/aichallenger/ai_challenger_json/',required=False,
                        help="jsons path")
    parser.add_argument("--ai_json_dir_name", type=str, default='../human_pose/ai_challenger_keypoint_train_annotations_20170909/keypoint_train_annotations_20170909.json',required=False,
                        help="jsons path")#'../human_pose/keypoint_train_annotations_20170902.json'
    parser.add_argument("--out_json_path", type=str, default = './cocee.json',required=False,
                        help="openpose 2 coco path")
    args = parser.parse_args()
    coco_output = {}
    # size = [1920,1080]
    #------------------------coco_格式-------------------------------
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
                [1, 15], [15, 17], [1, 16], [16, 18],
                [2, 3], [3, 4], [2, 5], [5, 6], [6, 7],
                [8, 9], [9, 10], [10, 11], [11, 23], [11, 24], [24, 22],
                [8, 12], [12, 13], [13, 14], [14, 21], [14, 20], [19, 21],
                [2, 17], [5, 18]

            ]
        }
    ]
    images = []
    annotations = []
    images_id = []

    #------------------------coco_格式-------------------------------
    img_dir = args.images_dir_name
    op_json_dir = args.openpose_jsons_dir_name
    ai_json_dir = args.ai_json_dir_name
    #------------------------aichallenge数据读取-------------------------
    ai_json = get_keypoints(ai_json_dir)
    #print(ai_json[5]['keypoint_annotations'])
    idex = 0
    for i,val in enumerate(ai_json): #遍历aichallenger的json文件，以其为基准，因为它是完整的。
        print (i)
        #print(img_dir+r[i]['image_id']+'.jpg')
        if os.path.exists(img_dir+ai_json[i]['image_id']+'.jpg'):#os.path.isfile() 
            img = cv2.imread(img_dir+ai_json[i]['image_id']+'.jpg')
            #cv2.imwrite('./res_img/'+str(i)+'.jpg',img)
            imag_id = ai_json[i]['image_id']
            print(image_id,'image_id')
            #imag_id = str(i)
            size = [img.shape[1], img.shape[0]]
            id, file_name = get_image_info(image_name=ai_json[i]['image_id']+'.jpg')
            image = create_image_info(id, file_name, size)
            images.append(image)
            images_id.append(id)



        coco_kpts = []
        #print(op_json_dir+ai_json[i]['image_id']+'.json')
        if os.path.exists(op_json_dir+ai_json[i]['image_id']+'_keypoints.json'): #判断openpose 生成的json文件是否存在目录中。

            with open(op_json_dir+ai_json[i]['image_id']+'_keypoints.json') as f:
                openpose_json = json.load(f)
                pdatas = openpose_json['people']

                
                #print(ai_json[i]['keypoint_annotations'].keys())
                for key in ai_json[i]['keypoint_annotations'].keys():#得到ai_challenger数据的keypoints     
                    num_keypoints = 0
                    keypoint_ai = ai_json[i]['keypoint_annotations'][key]
                    coco_kpts = keypoint_ai
                    for u in range(0, len(coco_kpts), 3):#将ai标注格式改为coco格式
                        # if new_keypoints[i+2] > 0.1:
                        if coco_kpts[u] != 0 or coco_kpts[u + 1] != 0:
                            coco_kpts[u + 2] = 2
                            num_keypoints += 1
                        else:
                            coco_kpts[u] = coco_kpts[u + 1] = coco_kpts[u + 2] = 0
                    bbox_ai = ai_json[i]['human_annotations'][key]
                    idex += 1
                    for p_i,pdata in enumerate(pdatas):#得到openpose生成数据的keypoints  
                        keypoints_openpose = pdata['pose_keypoints_2d']
                        condition = if_in_bbox(bbox_ai,keypoints_openpose)#匹配条件 重点！！ 
                        #condition = 1 
                        if condition and len(coco_kpts)<57: #匹配条件至少两个--> 1：所有点必须在bbox里面，2：保证最终的len(coco_kpts) 必须 == 固定值。
                            #计算bbox条件时要去掉0的情况
                            #
                            new_keypoints = keypoints_openpose

                            for j in range(0, len(new_keypoints), 3):#将openpose格式转为coco格式
                                # if new_keypoints[j+2] > 0.1:
                                if new_keypoints[j] != 0 or new_keypoints[j + 1] != 0:
                                    new_keypoints[j + 2] = 2
                                    num_keypoints += 1
                                else:
                                    new_keypoints[j] = new_keypoints[j + 1] = new_keypoints[j + 2] = 0
                            
                            #print(len(new_keypoints))

                            new_keypoints0 = new_keypoints[0:3]#鼻子
                            #new_keypoints1 = new_keypoints[42:54]#眼睛，耳朵
                            new_keypoints1 = new_keypoints[45:57]#眼睛，耳朵
                            coco_kpts.append(new_keypoints0[0])
                            coco_kpts.append(new_keypoints0[1])
                            coco_kpts.append(new_keypoints0[2])
                            for v,val in enumerate(new_keypoints1):
                                coco_kpts.append(new_keypoints1[v])

                        '''for z in range(0, len(coco_kpts), 3):
                            cv2.circle(img,(int(coco_kpts[z]),int(coco_kpts[z+1])),5,(0,0,255),-1)
                            #cv2.imwrite('./res_img/'+ai_json[i]['image_id']+'.jpg',img)
                            cv2.imwrite('./res_img/'+str(i)+'.jpg',img)'''
                         

                    if len(coco_kpts)<57:#这个判断条件要保证上一步后coco_kpts 长度为57。
                        print('---------------------------------?????')
                        for o in range(15):
                            
                            coco_kpts.append(0)
                    #print(len(coco_kpts),'------len(coco_kpts)')
                    #for pdata in pdatas:#得到人工标注数据的keypoints
                        #if condition and len(coco_kpts)<=57+7*3: 同样需要的一个匹配条件
                        #需要把标注的关键点改为coco格式，最后添加到coco_kpts中 
                        #最后需要确保 len(coco_kpts) == 固定值，所以要加一个判断条件
                    for pp in range(18):
                        coco_kpts.append(0)
                    #print(len(coco_kpts),'------len(coco_kpts)')
                    annotation = create_annotation_info(num_keypoints, coco_kpts, imag_id, idex)
                    print(annotation,'------------')
                    annotations.append(annotation)
                    #print(annotations,'------------')
   

                     
    print("-----finished annotations-----")

    coco_output["info"] = info
    coco_output["licenses"] = licenses
    coco_output["categories"] = categories
    coco_output["images"] = images
    coco_output["annotations"] = annotations   
                
    with open(args.out_json_path, 'w') as f:
        json.dump(coco_output, f)
        print('write in json')






