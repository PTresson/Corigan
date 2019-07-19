import os
from os import listdir
from os.path import isfile, join
import shutil
from PIL import Image

########################### load bounding boxes coordinates into a list from a list of txt files #################################

def load_bbox(files_bbox, files_dir):
    data = [None] * len(files_bbox)

    i = 0
    while i < len(files_bbox):

        file_tot = os.path.join(files_dir, files_bbox[i])

        with open(file_tot) as f:
            list_of_lists = []
            for line in f:
                inner_list = [float(elt.strip()) for elt in line.split(' ')]
                # in alternative, if you need to use the file content as numbers
                # inner_list = [int(elt.strip()) for elt in line.split(',')]
                list_of_lists.append(inner_list)
        data[i] = list_of_lists
        i += 1
    return data


########################### get global coordinates of an object detected on a slice #################################

def get_coord(name, detection):

    # get split information out of image name
    root_image = name.split("|")[0]
    coord_ext = name.split("|")[1]
    coord = coord_ext.split(".")[0]
    x_slice = int(coord.split("_")[0])
    y_slice = int(coord.split("_")[1])
    SliceWidth = int(coord.split("_")[2])
    SliceHeight = int(coord.split("_")[3])
    nW = int(coord.split("_")[5])
    nH = int(coord.split("_")[6])

    # get detection information out of detection file
    obj_class = int(detection[0])
    confidence = detection[1]
    x = detection[2]
    y = detection[3]
    w = detection[4]
    h = detection[5]

    x_glob_abs = x_slice + x * SliceWidth
    y_glob_abs = y_slice + y * SliceHeight
    w_abs = w * SliceWidth
    h_abs = h * SliceHeight

    x_rel = x_glob_abs / nW
    y_rel = y_glob_abs / nH
    w_rel = w_abs / nW
    h_rel = h_abs / nH

    # to save as a dictionary rather than a list

    # coordinates = {}
    # coordinates["root_image"] = root_image
    # coordinates["obj_class"] = obj_class
    # coordinates["confidence"] = confidence
    # coordinates["x"] = x_rel
    # coordinates["y"] = y_rel
    # coordinates["w"] = w_rel
    # coordinates["h"] = h_rel
    coordinates = [root_image, obj_class, confidence, x_rel, y_rel, w_rel, h_rel]

    return coordinates


#################################### list files with a given extension in a dir ########################################


def list_files (dir_to_list, extension, out_dir, outname='list'):

    files_to_list = [f for f in os.listdir(dir_to_list) if f.endswith(extension)]

    outname_tot = os.path.join(out_dir, outname)
    outname_tot_ext = outname_tot + '.txt'
    list_file = open(outname_tot_ext,"w")

    n = len(files_to_list)

    i=0
    while i<n :
        list_file.write("%s/%s\n" %(dir_to_list,files_to_list[i]))
        i += 1
    list_file.close()

################################################### clean dir ##########################################################

def clean_dir (dir_to_clean) :
    for f in os.listdir(dir_to_clean):
        f_path = os.path.join(dir_to_clean, f)
        try :
            if os.path.isfile(f_path):
                os.unlink(f_path)
            elif os.path.isdir(f_path) : shutil.rmtree(f_path)

        except Exception as e :
            print(e)

################################################## get date taken from jpg image #######################################

def get_date_taken(path):
    return Image.open(path)._getexif()[36867]