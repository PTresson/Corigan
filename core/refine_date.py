import os
import pandas as pd
import numpy as np
from Calc_IoU import calc_IOU, calc_Inter
from utils import load_bbox
from plot_bbox import plot_bbx
from PIL import Image
import re

def get_date_taken(path):
    return Image.open(path)._getexif()[36867]

def read_yolo_detection(result_file):

    detections = []
    with open(result_file) as f:
        lines = f.readlines()

        for i in range(len(lines)-1):  # last line always "Enter Image Path: "

            if "left_x" in lines[i]:  # all coordinates lines contain "left_x: ... top_y: .... width: .... height: ...."

                image_path = lines[i-1].split(':')[-2]
                # get the split image information within the name.
                # if several detections on a single split, get the first image path before the detection
                k = i
                while not image_path.startswith(' /'):
                    image_path = lines[k].split(':')[-2]
                    k -= 1

                image_name = image_path.split('/')[-1]
                root_image = image_name.split('|')[0]
                coord_ext = image_name.split("|")[1]
                coord = coord_ext.split(".")[0]
                y_slice = int(coord.split("_")[0])
                x_slice = int(coord.split("_")[1])
                SliceWidth = int(coord.split("_")[2])
                SliceHeight = int(coord.split("_")[3])
                nW = int(coord.split("_")[5])
                nH = int(coord.split("_")[6])
                # print(x_slice,y_slice,SliceWidth, SliceHeight, nW, nH)

                obj_class = lines[i].split(':')[0]

                # get all integers (signed or not) within detection
                regex = re.compile(r"[+-]?(?<!\.)\b[0-9]+\b(?!\.[0-9])")
                obj_coord_str = regex.findall(lines[i])

                confidence = float(int(obj_coord_str[0]) / 100)

                left_x = int(obj_coord_str[1])
                top_y = int(obj_coord_str[2])
                w_obj = int(obj_coord_str[3])
                h_obj = int(obj_coord_str[4])

                left_x_abs = left_x + x_slice
                top_y_abs = top_y + y_slice
                x_abs = left_x_abs + int(w_obj/2)
                y_abs = top_y_abs + int(h_obj/2)

                x_rel = float(x_abs/nW)
                y_rel = float(y_abs/nH)
                w_rel = float(w_obj/nW)
                h_rel = float(h_obj/nH)

                coordinates = [root_image, obj_class, confidence, x_rel, y_rel, w_rel, h_rel]
                detections.append(coordinates)

    return detections




def refine_detections(result_file, overlap_threshold=0.5, detection_threshold=0.2, outname='test_temp/refined_detections.csv', test_im_dir='test_images'):

    list_detection = read_yolo_detection(result_file)  # get yolo detections into a list of all detections...

    df = pd.DataFrame(list_detection, columns=('root_image', 'obj_class', 'confidence', 'x', 'y', 'w', 'h'))
    # and into a dataframe

    ####################### comparison of detected objects #################################

    # list of detections to keep after non max suppression
    list_nms = []

    # for each image
    for img_root in df.root_image.unique():

        df_img = df[df.root_image == img_root]

        # for each class
        for dift_class in df_img.obj_class.unique():

            df_class = df_img[df_img.obj_class == dift_class]  # keep only detections of this class

            for rowA in df_class.iterrows():  # every object of this class ...

                df_comp = df_class.drop([rowA[0]])  # ...will be compared with every other objects of this class

                if len(df_comp) == 0:  # if only one object of this class, we can keep it without further comparison

                    index, data = rowA
                    list_nms.append(data.tolist())

                else:

                    xA = rowA[1].x
                    yA = rowA[1].y
                    wA = rowA[1].w
                    hA = rowA[1].h
                    areaA = wA * hA
                    areamax = areaA  # keep track of the biggest object to keep it at the end
                    row_to_keep = rowA

                    for rowB in df_comp.iterrows():

                        xB = rowB[1].x
                        yB = rowB[1].y
                        wB = rowB[1].w
                        hB = rowB[1].h
                        areaB = wB * hB

                        inter = calc_Inter(xA, xB, yA, yB, wA, wB, hA, hB)  # intersection area between two detections

                        interA = float(inter / areaA)  # how much objects are covered by the intersection
                        interB = float(inter / areaB)

                        if interA > overlap_threshold or interB > overlap_threshold:
                            # if one or the other object is largely covered by the intersection
                            # i.e. if this object is 'within' the other

                            if areaB > areamax:
                                areamax = areaB
                                row_to_keep = rowB  # if the compared object is bigger, we keep it

                    index_keep, data_keep = row_to_keep  # all the biggest objects that intersect with other are kept
                    list_nms.append(data_keep.tolist())

    df_tokeep = pd.DataFrame(list_nms, columns=('root_image', 'obj_class', 'confidence', 'x', 'y', 'w', 'h'))

    df_tokeep = df_tokeep.drop_duplicates()  # all kept objects have been counted several times


    ############################## filter detections over a given confidence threshold #################################

    df_refined = df_tokeep[df_tokeep['confidence'] > detection_threshold]

    df_refined['date'] = 'NaN'
    core_directory = os.path.dirname(__file__)
    pipeline_directory = os.path.split(core_directory)[0]
    test_im_directory = os.path.join(pipeline_directory, 'test_images')


    for row in df_refined.iterrows():
        index, data = row
        root_image_name = str(data['root_image'] + '.JPG')
        root_image_path = os.path.join(test_im_directory,root_image_name)
        date_image = get_date_taken(root_image_path)
        df_refined.at[index, 'date'] = date_image

    # export CSV

    df_refined.to_csv(outname, index=False)


