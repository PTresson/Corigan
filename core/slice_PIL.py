# -*- coding: utf-8 -*-
"""
cuts original images into slices and recalculates labels of images containing enough of an object
Version of the slicing functions using PIL rather than openCV. PIL is slower but easier to use and to install.
"""

from __future__ import print_function
import os
from PIL import Image
import time
import Calc_IoU
from utils import load_bbox


# slice train slices images and bboxes according to certain decision rules
def slice_train(orig_dir, slice_dir, sliceHeight=256, sliceWidth=256, zero_frac_thresh=0.2, overlap=0.2, Pobj = 0.4, Pimage = 0.5 ,slice_sep='|', out_ext='.jpg', verbose=False) :

    def slice_tot(image_path, bbox_path, out_name, outdir):

        image_path_tot = os.path.join(orig_dir, image_path)

        image0 = Image.open(image_path_tot)  # cv2.imread(image_path_tot, 1)  # color

        if len(out_ext) == 0:
            ext = '.' + image_path.split('.')[-1]
        else:
            ext = out_ext
            #
        width, height = image0.size

        print(width, height)

        dx = int((1. - overlap) * sliceWidth)
        dy = int((1. - overlap) * sliceHeight)

        # bbox
        m = len(bbox_path)
        data_txt_abs = bbox_path

        for j in range(m):
            data_txt_abs[j][0] = int(bbox_path[j][0])  # classe
            data_txt_abs[j][1] = int(bbox_path[j][1] * nW)  # x
            data_txt_abs[j][2] = int(bbox_path[j][2] * nH)  # y
            data_txt_abs[j][3] = int(bbox_path[j][3] * nW)  # w
            data_txt_abs[j][4] = int(bbox_path[j][4] * nH)  # h


        for y0 in range(0, height, dy):#sliceHeight):
            for x0 in range(0, width, dx):#sliceWidth):

                # make sure we don't have a tiny image on the edge
                if y0 + sliceHeight > height:
                    y = height - sliceHeight
                else:
                    y = y0
                if x0 + sliceWidth > width:
                    x = width - sliceWidth
                else:
                    x = x0

                # extract image
                window_c = image0.crop((x, y, x + sliceWidth, y + sliceHeight))

                outpath = os.path.join(outdir, out_name + \
                                       slice_sep + str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(
                    sliceWidth) + ext)

                window_c.save(outpath)

                ##################### slicing bboxes #############################################

                x_min_slice = x
                x_max_slice = x + sliceWidth
                y_min_slice = y
                y_max_slice = y + sliceHeight
                x_slice_abs = x + int(0.5 * sliceWidth)
                y_slice_abs = y + int(0.5 * sliceHeight)

                txt_file_name = ("%s.txt" % (outpath[0:-4]))

                text_file = open(txt_file_name, "w")


                m = len(data_txt[i])
                for j in range(m):

                    X_obj_abs = data_txt_abs[j][1]
                    Y_obj_abs = data_txt_abs[j][2]
                    W_obj_abs = data_txt_abs[j][3]
                    H_obj_abs = data_txt_abs[j][4]
                    object_area_abs = W_obj_abs * H_obj_abs

                    X_min_obj_abs = int(X_obj_abs - 0.5 * W_obj_abs)
                    X_max_obj_abs = int(X_obj_abs + 0.5 * W_obj_abs)
                    Y_min_obj_abs = int(Y_obj_abs - 0.5 * H_obj_abs)
                    Y_max_obj_abs = int(Y_obj_abs + 0.5 * H_obj_abs)


                    Inter = int(Calc_IoU.calc_Inter(X_obj_abs, x_slice_abs, Y_obj_abs, y_slice_abs, W_obj_abs, sliceWidth, H_obj_abs, sliceHeight))
                    object_covering = float(Inter/object_area_abs)

                    window_covering = float(Inter/win_size)

                    centroid_inside_window = ((x_min_slice <= X_obj_abs <= x_max_slice) and (y_min_slice <= Y_obj_abs <= y_max_slice))

                    if (object_covering > Pobj or window_covering > Pimage or centroid_inside_window):
                        x_obj_slice_temp = float((X_obj_abs - x_min_slice) / sliceWidth)
                        y_obj_slice_temp = float((Y_obj_abs - y_min_slice) / sliceHeight)
                        w_obj_slice_temp = float(W_obj_abs / sliceWidth)
                        h_obj_slice_temp = float(H_obj_abs / sliceHeight)
                        # print(w_obj_slice_temp, h_obj_slice_temp)

                        x_min_obj_slice = float(x_obj_slice_temp - 0.5 * w_obj_slice_temp)
                        x_max_obj_slice = float(x_obj_slice_temp + 0.5 * w_obj_slice_temp)
                        y_min_obj_slice = float(y_obj_slice_temp - 0.5 * h_obj_slice_temp)
                        y_max_obj_slice = float(y_obj_slice_temp + 0.5 * h_obj_slice_temp)
                        # print("xmin xmax ymin ymax")
                        # print(x_min_obj_slice, x_max_obj_slice, y_min_obj_slice, y_max_obj_slice)

                        if (x_min_obj_slice < 0):
                            x_min_obj_slice = 0
                        if (y_min_obj_slice < 0):
                            y_min_obj_slice = 0
                        if (x_max_obj_slice > 1):
                            x_max_obj_slice = 1
                        if (y_max_obj_slice > 1):
                            y_max_obj_slice = 1

                        w_obj_slice = float(x_max_obj_slice - x_min_obj_slice)
                        h_obj_slice = float(y_max_obj_slice - y_min_obj_slice)
                        x_obj_slice = float(x_min_obj_slice + 0.5 * w_obj_slice)
                        y_obj_slice = float(y_min_obj_slice + 0.5 * h_obj_slice)
                        #print(x_obj_slice, y_obj_slice, w_obj_slice, h_obj_slice)

                        text_file.write("%d %4f %4f %4f %4f\n" % (
                            data_txt_abs[j][0], x_obj_slice, y_obj_slice, w_obj_slice,
                            h_obj_slice))

                text_file.close()

                # remove file if empty
                # print(os.path.getsize(txt_file_name))

                if os.stat(txt_file_name).st_size == 0:
                    os.remove(txt_file_name)


        return


    data_image = sorted([f for f in os.listdir(orig_dir) if (f.endswith('.JPG') or f.endswith('.jpg'))])
    n = len(data_image)

    # print(origine_slice)
    files_bbox = sorted([f for f in os.listdir(orig_dir) if f.endswith('.txt')])
    # print(files_bbox)
    data_txt = load_bbox(files_bbox, orig_dir)

    for i in range(n):
        slice_tot(data_image[i],data_txt[i] ,data_image[i][0:-4], slice_dir)

###########################   slice for test and regular detection  ###########################################

# slice test only slices images
def slice_test(orig_dir, slice_dir, sliceHeight=256, sliceWidth=256, zero_frac_thresh=0.2, overlap=0.2, slice_sep='|', out_ext='.jpg', verbose=False) :

    def slice_im(image_path, out_name, outdir):

        image_path_tot = os.path.join(orig_dir, image_path)

        image0 = Image.open(image_path_tot)  # cv2.imread(image_path_tot, 1)  # color

        if len(out_ext) == 0:
            ext = '.' + image_path.split('.')[-1]
        else:
            ext = out_ext
            #
        width, height = image0.size

        print(width, height)

        dx = int((1. - overlap) * sliceWidth)
        dy = int((1. - overlap) * sliceHeight)

        for y0 in range(0, height, dy):  # sliceHeight):
            for x0 in range(0, width, dx):  # sliceWidth):

                # make sure we don't have a tiny image on the edge
                if y0 + sliceHeight > height:
                    y = height - sliceHeight
                else:
                    y = y0
                if x0 + sliceWidth > width:
                    x = width - sliceWidth
                else:
                    x = x0

                # extract image
                window_c = image0.crop((x, y, x + sliceWidth, y + sliceHeight))

                outpath = os.path.join(outdir, out_name + \
                                       slice_sep + str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(
                    sliceWidth) + ext)

                window_c.save(outpath)


        return


    data_image = sorted([f for f in os.listdir(orig_dir) if (f.endswith('.JPG') or f.endswith('.jpg'))])
    n = len(data_image)

    # print(origine_slice)
    files_bbox = sorted([f for f in os.listdir(orig_dir) if f.endswith('.txt')])
    # print(files_bbox)
    data_txt = load_bbox(files_bbox, orig_dir)

    for i in range(n):
        slice_im(data_image[i], data_image[i][0:-4], slice_dir)


# def slice_im(image_path, out_name, outdir):
#
#     image_path_tot = os.path.join(orig_dir,image_path)
#
#     image0 = Image.open(image_path_tot) #cv2.imread(image_path_tot, 1)  # color
#
#     if len(out_ext) == 0:
#         ext = '.' + image_path.split('.')[-1]
#     else:
#         ext = out_ext
#         #
#     width, height = image0.size
#
#     print(width, height)
#
#     dx = int((1. - overlap) * sliceWidth)
#     dy = int((1. - overlap) * sliceHeight)
#
#     for y0 in range(0, height, dy):#sliceHeight):
#         for x0 in range(0, width, dx):#sliceWidth):
#
#             # make sure we don't have a tiny image on the edge
#             if y0+sliceHeight > height:
#                 y = height - sliceHeight
#             else:
#                 y = y0
#             if x0+sliceWidth > width:
#                 x = width - sliceWidth
#             else:
#                 x = x0
#
#             # extract image
#             window_c = image0.crop((x,y,x+sliceWidth,y+sliceHeight))
#
#
#             outpath = os.path.join(outdir, out_name + \
#             slice_sep + str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(sliceWidth) + ext)
#
#             window_c.save(outpath)
#
#
#     print(n_ims)
#
#
#     return
