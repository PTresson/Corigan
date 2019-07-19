import os
import utils
import time
import pandas as pd
import argparse
from refine_date import read_yolo_detection
from refine_date import refine_detections
from utils import load_bbox
from Calc_IoU import calc_IOU
from plot_bbox import plot_bbx
from statistics import mean

pd.set_option('display.expand_frame_repr', False)

###################################### get precision, recall, mAP ######################################################

def get_metrics(test_im_dir,detection_csv_file, names_file, examples_file_path, print_res=True, edit_res=False):


    ################### function to get average precision ################################

    def voc_ap(rec, prec):
        # from https://github.com/Cartucho/mAP/blob/master/main.py
        """
        --- Official matlab code VOC2012---
        mrec=[0 ; rec ; 1];
        mpre=[0 ; prec ; 0];
        for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
        end
        i=find(mrec(2:end)~=mrec(1:end-1))+1;
        ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        rec.insert(0, 0.0)  # insert 0.0 at begining of list
        rec.append(1.0)  # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0)  # insert 0.0 at begining of list
        prec.append(0.0)  # insert 0.0 at end of list
        mpre = prec[:]
        """
         This part makes the precision monotonically decreasing
          (goes from the end to the beginning)
          matlab:  for i=numel(mpre)-1:-1:1
                      mpre(i)=max(mpre(i),mpre(i+1));
        """
        # matlab indexes start in 1 but python in 0, so I have to do:
        #   range(start=(len(mpre) - 2), end=0, step=-1)
        # also the python function range excludes the end, resulting in:
        #   range(start=(len(mpre) - 2), end=-1, step=-1)
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        """
         This part creates a list of indexes where the recall changes
          matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
        """
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                i_list.append(i)  # if it was matlab would be i + 1
        """
         The Average Precision (AP) is the area under the curve
          (numerical integration)
          matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
        return ap, mrec, mpre


    ###########################################################################################################

    ######################## Formating both detection and GT data into same template ##########################

    file_bbox = sorted([f for f in os.listdir(test_im_dir) if f.endswith('.txt')])
    images = sorted([f for f in os.listdir(test_im_dir) if f.endswith('.JPG')])

    # sort only images with labels and labels attached to images

    image_labelled = []

    for i in range(len(file_bbox)):
        root_name_bbx = file_bbox[i].split('.')[0]
        for image in images:
            if image.split('.')[0] == root_name_bbx:
                image_labelled.append(image)

    image_labelled = sorted(image_labelled)

    labels_with_images = []

    for i in range(len(image_labelled)):
        root_name_image = image_labelled[i].split('.')[0]
        for label in file_bbox:
            if label.split('.')[0] == root_name_image:
                labels_with_images.append(label)

    labels_with_images = sorted(labels_with_images)

    bbox = load_bbox(labels_with_images, test_im_dir)

    merged_bbox_list = []

    for i in range(len(bbox)):
        for j in range(len(bbox[i])):
            bbox[i][j].insert(0, images[i][0:-4])
        merged_bbox_list = merged_bbox_list + bbox[i]

    ground_truth = merged_bbox_list

    detection_csv = detection_csv_file


    # gt_df stores GT information
    # dt_df stores detection information
    gt_df = pd.DataFrame(ground_truth, columns=('root_image','obj_class', 'x', 'y', 'w', 'h'))
    dt_df = pd.read_csv(detection_csv)

    ######################### convert classes according .names file ###################

    gt_df['obj_class'] = gt_df.obj_class.astype(str)

    with open(names_file) as f:
        labels = [line.rstrip('\n') for line in f]

    for row in gt_df.iterrows():
        index, data = row
        for i in range(len(labels)):
            if (row[1].obj_class == str(float(i))):
                gt_df.at[index, 'obj_class'] = labels[i]


    ################## compare only images that have associated GT ####################

    dt_df = dt_df[dt_df.root_image.isin(gt_df.root_image.unique())]

    ################## get the list of all classes (detected of not) ####################

    class_list = []
    max_name_class_length = 0
    for dift_class in gt_df.obj_class.unique():
        class_list.append(dift_class)
        # included by Dom
        max_name_class_length = max(max_name_class_length, len(dift_class))
        # end of included by Dom

    for dift_class in dt_df.obj_class.unique():
        if dift_class not in class_list:
            class_list.append(dift_class)

    ########################### generating template dict for per classes metrics ###############

    mAP_class_dict = {}
    prec_class_dict = {}
    for dift_class in class_list :
        mAP_class_dict[dift_class] = 0
        prec_class_dict[dift_class] = 0

    ####################### initialize variables #############################

    dt_df = dt_df.sort_values(['obj_class'])
    dt_df['correct'] = 0
    dt_df['precision'] = 0.0
    dt_df['recall'] = 0.0

    gt_df = gt_df.sort_values(['obj_class'])
    gt_df['used'] = 0

    # overlap threshold for acceptance
    overlap_threshold = 0.5


    ####################### comparison of detected objects  to GT #################################

    for image in dt_df.root_image.unique():

        print(image)

        dt_df_img = dt_df[dt_df['root_image'] == image]
        gt_df_img = gt_df[gt_df['root_image'] == image]

        # list different classes present in GT

        gt_classes = gt_df_img.obj_class.unique()

        # list out wrong predicted classes

        dt_df_correct_classes = dt_df_img[dt_df_img['obj_class'].isin(gt_classes)]

       # for each  correct class

        for dift_class in gt_classes:

           gt_df_class = gt_df_img[gt_df_img.obj_class == dift_class]

           dt_df_class = dt_df_correct_classes[dt_df_correct_classes.obj_class == dift_class]

           for rowA in gt_df_class.iterrows():  # compare each GT object of this class ...

                xA = rowA[1].x
                yA = rowA[1].y
                wA = rowA[1].w
                hA = rowA[1].h
                used = rowA[1].used

                for rowB in dt_df_class.iterrows():  # ... with every detected object of this class

                    xB = rowB[1].x
                    yB = rowB[1].y
                    wB = rowB[1].w
                    hB = rowB[1].h

                    IOU = calc_IOU(xA, xB, yA, yB, wA, wB, hA, hB)

                    if IOU > overlap_threshold :  # i.e. if detection and GT overlap

                        if used == 0:  # gt not found yet
                            indexB, dataB = rowB
                            dt_df.at[indexB, 'correct'] = 1

                            indexA, dataA = rowA
                            gt_df.at[indexA, 'used'] = 1

    df_TP = dt_df[dt_df.correct == 1]
    df_FP = dt_df[dt_df.correct == 0]
    df_FN = gt_df[gt_df.used == 0]

    TP = len(df_TP.correct)
    FP = len(df_FP.correct)
    FN = len(df_FN.used)


    precision_general = float(TP / (TP + FP))
    recall_general = float(TP / (TP + FN))
    F1_general = 2 * float((precision_general * recall_general) / (precision_general + recall_general))


    ########################### mAP #############################

    # once we have for each detection its status (TP or FP), we can compute average precision for each class
    # for this we will need to compute recall and precision for each class

    # included by Dom
    gt_classes = gt_df.obj_class.unique()
    nb_classes = len(gt_classes)
    df_metrics_classes = pd.DataFrame(pd.np.empty((nb_classes, 10)), columns=['classes', 'precision', 'recall', 'F1',
                                                                             'mAP', 'GT', 'TP',
                                                                             'FP', 'FN', 'training_examples'])
    df_metrics_classes['classes'] = gt_classes
    df_metrics_classes['precision'] = 0
    df_metrics_classes['recall'] = 0
    df_metrics_classes['F1'] = 0
    df_metrics_classes['mAP'] = 0
    df_metrics_classes['GT'] = 0
    df_metrics_classes['TP'] = 0
    df_metrics_classes['FP'] = 0
    df_metrics_classes['FN'] = 0
    # end of included by Dom

    df_examples = pd.read_csv(examples_file_path, index_col='name')
    print('*********')
    print(df_examples)

    for dift_class in gt_df.obj_class.unique():
        gt_df_class = gt_df[gt_df.obj_class == dift_class]
        items_to_detect = len(gt_df_class)
        df_metrics_classes.loc[df_metrics_classes.classes == dift_class, 'GT'] = items_to_detect
        df_metrics_classes.loc[df_metrics_classes.classes == dift_class, 'training_examples'] = df_examples.at[dift_class,'count']
        FN_count = 0
        for row in gt_df_class.iterrows():

            index, data = row

            if (data['used'] == 0):
                 FN_count +=1
            df_metrics_classes.loc[df_metrics_classes.classes == dift_class, 'FN'] = FN_count

    for dift_class in dt_df.obj_class.unique():

        gt_df_class = gt_df[gt_df.obj_class == dift_class]
        dt_df_class = dt_df[dt_df.obj_class == dift_class]

        # GT number of objects to find (potential positives = PP)

        PP = len(gt_df_class)

        if PP == 0: # if false positive
            mAP_class_dict[dift_class] = 0

        if PP != 0:
            TP_count = 0
            FP_count = 0
            row_count = 0  # (for recall)

            for row in dt_df_class.iterrows():

                row_count += 1
                index, data = row

                if (data['correct'] == 1):
                    TP_count += 1
                else:
                    FP_count += 1

                # if (data['used'] == 0):
                #     FN_count +=1

                recall = float(TP_count / PP)
                #print('recall = ', recall)
                precision = float(TP_count / row_count)
                #print('precision = ', precision)

                dt_df_class.at[index, 'recall'] = recall
                dt_df_class.at[index, 'precision'] = precision

            prec = dt_df_class['precision'].tolist()
            rec = dt_df_class['recall'].tolist()

            # included by Dom
            avg_prec_class = mean(prec)

            if avg_prec_class != 0 and recall != 0:
                F1_score_class = 2 * float((avg_prec_class * recall) / (avg_prec_class + recall))
            else:
                F1_score_class = 0

            df_metrics_classes.loc[df_metrics_classes.classes == dift_class, 'precision'] = avg_prec_class
            df_metrics_classes.loc[df_metrics_classes.classes == dift_class, 'recall'] = recall
            df_metrics_classes.loc[df_metrics_classes.classes == dift_class, 'F1'] = F1_score_class
            df_metrics_classes.loc[df_metrics_classes.classes == dift_class, 'TP'] = TP_count
            df_metrics_classes.loc[df_metrics_classes.classes == dift_class, 'FP'] = FP_count



            # end of included by Dom

            ap, mrec, mprec = voc_ap(rec, prec)  # according previously defined function

            mAP_class_dict[dift_class] += ap

    # this is just a fix for some issues pandas can encounter with .from_dict() function
    # but sometimes, it does work perfectly fine.
    new_dict = {k: [v] for k, v in mAP_class_dict.items()}
    temp_df = pd.DataFrame(new_dict, index = ['mAP'])
    mAP_class_df = temp_df.T

    df_metrics_classes.training_examples=df_metrics_classes.training_examples.astype(int)
    for dift_class in dt_df.obj_class.unique():
        df_metrics_classes.loc[df_metrics_classes.classes == dift_class, 'mAP'] = mAP_class_df.at[dift_class,'mAP']

    # mAP_class_df = pd.DataFrame.from_dict(mAP_class_dict, orient = 'index', columns = ['mAP'])


    mAP = float(float(sum(mAP_class_dict.values()))/float(len(class_list)))

    if print_res:
        print('\n metrics per class_______________________\n')
        print(df_metrics_classes)
        print('\n general metrics _________________\n')
        print('mAP: \t\t%.4f' % mAP)
        print('precision: \t%.4f' % precision_general)
        print('recall: \t%.4f' % recall_general)
        print('F1: \t\t%.4f' % F1_general)
        df_metrics_classes.to_csv('test_temp/metrics_per_class.ods', index=False)
        df_metrics_classes.to_csv('test_temp/metrics_per_class.csv', index=False)
        df_general_metrics = pd.DataFrame(pd.np.empty((1, 4)), columns=['mAP', 'precision', 'recall', 'F1'])

        df_general_metrics['mAP'] = mAP
        df_general_metrics['precision'] = precision_general
        df_general_metrics['recall'] = recall_general
        df_general_metrics['F1'] = F1_general
        df_general_metrics.to_csv('test_temp/general_metrics.csv', index=False)
        df_general_metrics.to_csv('test_temp/general_metrics.ods', index=False)

        #################### edit results to csv file

    #modified by Dom; variables were not defined; maybe to delete?
    if edit_res:
        res_list = [['mAP', mAP], ['precision', precision_general], ['recall', recall_general], ['F1', F1_general]]
        res_df = pd.DataFrame(res_list, columns=['metric', 'value'])
        # print(res_df.T)
        # res_df.to_csv('test_temp/general_metrics.csv', index=False)
        # res_df.to_csv('test_temp/general_metrics.ods', index=False)

    return TP, FP, FN, precision_general, recall_general, F1_general, mAP, mAP_class_dict



###################################### confusion matrix ###############################################################


def get_confusion_matrix(test_im_dir,detection_csv_file, names_file, outname = 'test_temp/confusion_matrix.csv'):


    file_bbox = sorted([f for f in os.listdir(test_im_dir) if f.endswith('.txt')])
    images = sorted([f for f in os.listdir(test_im_dir) if f.endswith('.JPG')])

    bbox = load_bbox(file_bbox,test_im_dir)

    merged_bbox_list = []

    for i in range(len(bbox)):
        for j in range(len(bbox[i])):
            bbox[i][j].insert(0, images[i][0:-4])
        merged_bbox_list = merged_bbox_list + bbox[i]

    ground_truth = merged_bbox_list

    detection_csv = detection_csv_file

    gt_df = pd.DataFrame(ground_truth, columns=('root_image','obj_class', 'x', 'y', 'w', 'h'))
    gt_df['obj_class'] = gt_df.obj_class.astype(str)

    dt_df = pd.read_csv(detection_csv)

    ######################### convert classes according .names file ###################

    with open(names_file) as f:
        labels = [line.rstrip('\n') for line in f]

    for row in gt_df.iterrows():
        index, data = row
        for i in range(len(labels)):
            if (row[1].obj_class == str(float(i))):
                gt_df.at[index, 'obj_class'] = labels[i]

    ##########################
    dt_df = dt_df[dt_df.root_image.isin(gt_df.root_image.unique())]

    dt_df = dt_df.sort_values(['obj_class'])
    dt_df['correct'] = 0

    gt_df = gt_df.sort_values(['obj_class'])
    gt_df['used'] = 0

    ########################### generating template dict for per classes metrics ###############

    # get all classes (detected and GT)

    all_classes = []

    for dift_class in dt_df.obj_class.unique():
        all_classes.append(dift_class)

    for dift_class in gt_df.obj_class.unique():
        if dift_class not in all_classes:
            all_classes.append(dift_class)

    all_classes = sorted(all_classes)
    template_dic = {}

    for dift_class_x in all_classes :
        template_dic[dift_class_x] = {}
        for dift_class_y in all_classes :
            template_dic[dift_class_x][dift_class_y] = 0

    ####################### comparison of detected objects  to GT #################################

    # overlap threshold for acceptance
    overlap_threshold = 0.5
    for image in dt_df.root_image.unique():

        dt_df_img = dt_df[dt_df['root_image'] == image]
        gt_df_img = gt_df[gt_df['root_image'] == image]

        for rowA in gt_df_img.iterrows():

            xA = rowA[1].x
            yA = rowA[1].y
            wA = rowA[1].w
            hA = rowA[1].h
            classA = rowA[1].obj_class

            for rowB in dt_df_img.iterrows():  # ... with every detected object of this class

                xB = rowB[1].x
                yB = rowB[1].y
                wB = rowB[1].w
                hB = rowB[1].h
                classB = rowB[1].obj_class

                IOU = calc_IOU(xA, xB, yA, yB, wA, wB, hA, hB)

                if IOU > overlap_threshold :  # i.e. if detection and GT overlap

                    template_dic[classA][classB] += 1
                    #template_dic[classB][classA] += 1


    matrix_df = pd.DataFrame(template_dic)
    #print(matrix_df)
    matrix_df.to_csv(outname)