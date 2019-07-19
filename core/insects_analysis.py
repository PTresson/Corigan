import os
import time
import pandas as pd
import argparse
from itertools import combinations
from Calc_IoU import calc_IOU
from PIL import Image
from utils import load_bbox, get_date_taken

pd.set_option('display.expand_frame_repr', False)

############################### species count #########################################

def count_species_detection (detections_csv_file, print_res=False, edit_res=False, outname='results/object_count.csv'):

    df = pd.read_csv(detections_csv_file)
    list_of_list = []

    for image in df.root_image.unique():
        df_image = df[df.root_image == image]

        for classe in df_image.obj_class.unique():

            count = 0
            df_class = df_image[df_image.obj_class == classe]

            for row in df_class.iterrows():

                date = row[1].date
                count += 1

            list = [date, classe, count]
            list_of_list.append(list)

    df_count = pd.DataFrame(list_of_list, columns = ('date','obj_class', 'count'))
    if print_res:

        print(df_count)

    if edit_res:
        df_count.to_csv(outname, index=False)

############################### interactions count ####################################

def count_interactions_detection (detections_csv_file, print_res=False, edit_res=False, outname='results/object_inter.csv'):

    df = pd.read_csv(detections_csv_file)
    # generating template for result df

    list_of_list = []

    for image in df.root_image.unique() :

        df_image = df[df.root_image == image]
        date = df_image['date'].iloc[0]

        for combo in combinations(df_image.obj_class.unique(),2):
            #print(combo)

            list = [image, date, combo[0], combo[1],0]
            list_of_list.append(list)

    df_inter = pd.DataFrame(list_of_list, columns = ('root_image','date','classA', 'classB', 'count'))

    #
    for image in df.root_image.unique() :

        df_image = df[df.root_image == image]
        df_inter_image = df_inter[df_inter.root_image == image]

        for classe in df_image.obj_class.unique():

            df_class = df_image[df_image.obj_class == classe]
            df_not_class = df_image[df_image.obj_class != classe]
            #print(classe)

            for rowA in df_class.iterrows():
                xA = rowA[1].x
                yA = rowA[1].y
                wA = rowA[1].w
                hA = rowA[1].h
                classA = rowA[1].obj_class

                for rowB in df_not_class.iterrows():
                    xB = rowB[1].x
                    yB = rowB[1].y
                    wB = rowB[1].w
                    hB = rowB[1].h
                    classB = rowB[1].obj_class

                    IOU = calc_IOU(xA, xB, yA, yB, wA, wB, hA, hB)

                    if IOU > 0:

                        df_inter_bis = df_inter_image[df_inter_image.classA == classA]
                        df_inter_ter = df_inter_bis[df_inter_bis.classB == classB]

                        if len(df_inter_ter.classA) == 0 :

                            df_inter_bis = df_inter_image[df_inter_image.classA == classB]
                            df_inter_ter = df_inter_bis[df_inter_bis.classB == classA]


                        index_to_change = df_inter_ter.index[0]

                        df_inter.at[index_to_change,'count']+=1

    df_inter.loc[:,'count'] /= 2
    #pd.to_numeric(df_inter.loc[:,'count'], downcast='integer')
    df_inter = df_inter.sort_values(['root_image'])

    if print_res:
        print(df_inter)

    if edit_res:
        df_inter.to_csv(outname, index=False)


############################## interaction within a specie ################################

def count_intra_detection(detections_csv_file, print_res=False, edit_res=False, outname='results/object_intra.csv'):

    df = pd.read_csv(detections_csv_file)

    # generating template for result df

    list_of_list = []

    for image in df.root_image.unique() :

        df_image = df[df.root_image == image]
        date = df_image['date'].iloc[0]

        for obj_class in df_image.obj_class.unique():

            list = [image, date, obj_class,0]
            list_of_list.append(list)


    df_inter = pd.DataFrame(list_of_list, columns = ('root_image','date','obj_class', 'count'))

    #

    for image in df.root_image.unique() :

        df_image = df[df.root_image == image]
        df_inter_image = df_inter[df_inter.root_image == image]

        for classe in df_image.obj_class.unique():

            df_class = df_image[df_image.obj_class == classe]

            for rowA in df_class.iterrows():
                xA = rowA[1].x
                yA = rowA[1].y
                wA = rowA[1].w
                hA = rowA[1].h
                classA = rowA[1].obj_class

                df_comp = df_class.drop([rowA[0]])

                for rowB in df_comp.iterrows():
                    xB = rowB[1].x
                    yB = rowB[1].y
                    wB = rowB[1].w
                    hB = rowB[1].h
                    classB = rowB[1].obj_class

                    IOU = calc_IOU(xA, xB, yA, yB, wA, wB, hA, hB)

                    if IOU > 0:

                        df_inter_bis = df_inter_image[df_inter_image.obj_class == classA]

                        index_to_change = df_inter_bis.index[0]

                        df_inter.at[index_to_change,'count']+=1

    df_inter.loc[:,'count'] /= 2
    #pd.to_numeric(df_inter.loc[:,'count'], downcast='integer')
    df_inter = df_inter.sort_values(['root_image'])

    if print_res:
        print(df_inter)

    if edit_res:
        df_inter.to_csv(outname, index=False)


#######################################################################################################################
#                                                                                                                     #
#                                     exact same functions but for GT                                                 #
#                                                                                                                     #
#######################################################################################################################





def count_species_gt(test_im_dir, names_file, print_res=False, edit_res=False, outname='results/object_count_gt.csv'):

    # clean data preparation : remove images without labels, txt files that are no label file ...

    file_bbox = sorted([f for f in os.listdir(test_im_dir) if f.endswith('.txt')])
    images = sorted([f for f in os.listdir(test_im_dir) if f.endswith('.JPG')])

    image_labelled = []

    for i in range(len(file_bbox)):
        root_name_bbx = file_bbox[i].split('.')[0]
        for image in images :
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
            bbox[i][j].insert(0, image_labelled[i][0:-4])
        merged_bbox_list = merged_bbox_list + bbox[i]

    ground_truth = merged_bbox_list

    gt_df = pd.DataFrame(ground_truth, columns=('root_image', 'obj_class', 'x', 'y', 'w', 'h'))

    gt_df['root_image_path'] = 'test_images/'+gt_df['root_image']+'.JPG'
    gt_df['obj_class'] = gt_df.obj_class.astype(str)


    with open(names_file) as f:
        labels = [line.rstrip('\n') for line in f]

    for row in gt_df.iterrows():
        index, data = row
        for i in range(len(labels)):
            if (row[1].obj_class == str(float(i))):
                gt_df.at[index, 'obj_class'] = labels[i]

    list_of_list = []

    for image in gt_df.root_image_path.unique() :
        date = get_date_taken(image)
        gt_df_image = gt_df[gt_df.root_image_path == image]

        for classe in gt_df_image.obj_class.unique():

            count = 0
            gt_df_class = gt_df_image[gt_df_image.obj_class == classe]

            for row in gt_df_class.iterrows():
                count += 1

            list = [image, date, classe, count]
            list_of_list.append(list)

    gt_df_count = pd.DataFrame(list_of_list, columns = ('root_image_path', 'date','obj_class', 'count'))

    if print_res:
        print(gt_df_count)

    if edit_res:
        gt_df_count.to_csv(outname, index=False)



def count_inter_gt(test_im_dir,names_file, print_res=False, edit_res=False, outname='results/object_inter_gt.csv'):
    # clean data preparation : remove images without labels, txt files that are no label file ...

    file_bbox = sorted([f for f in os.listdir(test_im_dir) if f.endswith('.txt')])
    images = sorted([f for f in os.listdir(test_im_dir) if f.endswith('.JPG')])

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
            bbox[i][j].insert(0, image_labelled[i][0:-4])
        merged_bbox_list = merged_bbox_list + bbox[i]

    ground_truth = merged_bbox_list

    gt_df = pd.DataFrame(ground_truth, columns=('root_image', 'obj_class', 'x', 'y', 'w', 'h'))
    gt_df['root_image_path'] = 'test_images/'+gt_df['root_image']+'.JPG'
    gt_df['obj_class'] = gt_df.obj_class.astype(str)

    with open(names_file) as f:
        labels = [line.rstrip('\n') for line in f]

    for row in gt_df.iterrows():
        index, data = row
        for i in range(len(labels)):
            if (row[1].obj_class == str(float(i))):
                gt_df.at[index, 'obj_class'] = labels[i]

    df = gt_df  # I have only copied the previous function, now that df are of the same format

    # generating template for result df

    list_of_list = []

    for image in df.root_image_path.unique() :

        date = get_date_taken(image)
        df_image = df[df.root_image_path == image]

        for combo in combinations(df_image.obj_class.unique(),2):
            #print(combo)

            list = [image, date, combo[0], combo[1],0]
            list_of_list.append(list)

    df_inter = pd.DataFrame(list_of_list, columns = ('root_image_path','date','classA', 'classB', 'count'))

    #
    for image in df.root_image_path.unique() :
        date = get_date_taken(image)
        df_image = df[df.root_image_path == image]
        df_inter_image = df_inter[df_inter.root_image_path == image]

        for classe in df_image.obj_class.unique():

            df_class = df_image[df_image.obj_class == classe]
            df_not_class = df_image[df_image.obj_class != classe]

            for rowA in df_class.iterrows():
                xA = rowA[1].x
                yA = rowA[1].y
                wA = rowA[1].w
                hA = rowA[1].h
                classA = rowA[1].obj_class

                for rowB in df_not_class.iterrows():
                    xB = rowB[1].x
                    yB = rowB[1].y
                    wB = rowB[1].w
                    hB = rowB[1].h
                    classB = rowB[1].obj_class

                    IOU = calc_IOU(xA, xB, yA, yB, wA, wB, hA, hB)

                    if IOU > 0:

                        df_inter_bis = df_inter_image[df_inter_image.classA == classA]
                        df_inter_ter = df_inter_bis[df_inter_bis.classB == classB]

                        if len(df_inter_ter.classA) == 0 :

                            df_inter_bis = df_inter_image[df_inter_image.classA == classB]
                            df_inter_ter = df_inter_bis[df_inter_bis.classB == classA]


                        index_to_change = df_inter_ter.index[0]

                        df_inter.at[index_to_change,'count']+=1

    df_inter.loc[:,'count'] /= 2
    #pd.to_numeric(df_inter.loc[:,'count'], downcast='integer')
    df_inter = df_inter.sort_values(['root_image_path'])

    if print_res:
        print(df_inter)

    if edit_res:
        df_inter.to_csv(outname, index=False)


def count_intra_gt(test_im_dir,names_file, print_res=False, edit_res=False, outname='results/object_intra_gt.csv'):

    # clean data preparation : remove images without labels, txt files that are no label file ...

    file_bbox = sorted([f for f in os.listdir(test_im_dir) if f.endswith('.txt')])
    images = sorted([f for f in os.listdir(test_im_dir) if f.endswith('.JPG')])

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
            bbox[i][j].insert(0, image_labelled[i][0:-4])
        merged_bbox_list = merged_bbox_list + bbox[i]

    ground_truth = merged_bbox_list

    gt_df = pd.DataFrame(ground_truth, columns=('root_image', 'obj_class', 'x', 'y', 'w', 'h'))
    gt_df['root_image_path'] = 'test_images/'+gt_df['root_image']+'.JPG'
    gt_df['obj_class'] = gt_df.obj_class.astype(str)

    with open(names_file) as f:
        labels = [line.rstrip('\n') for line in f]

    for row in gt_df.iterrows():
        index, data = row
        for i in range(len(labels)):
            if (row[1].obj_class == str(float(i))):
                gt_df.at[index, 'obj_class'] = labels[i]

    df = gt_df  # I have only copied the previous function, now that df are of the same format


    # generating template for result df

    list_of_list = []

    for image in df.root_image_path.unique() :

        date = get_date_taken(image)
        df_image = df[df.root_image_path == image]

        for obj_class in df_image.obj_class.unique():

            list = [image, date, obj_class,0]
            list_of_list.append(list)


    df_inter = pd.DataFrame(list_of_list, columns = ('root_image_path','date','obj_class', 'count'))

    #

    for image in df.root_image_path.unique() :
        date = get_date_taken(image)
        df_image = df[df.root_image_path == image]
        df_inter_image = df_inter[df_inter.root_image_path == image]

        for classe in df_image.obj_class.unique():

            df_class = df_image[df_image.obj_class == classe]

            for rowA in df_class.iterrows():
                xA = rowA[1].x
                yA = rowA[1].y
                wA = rowA[1].w
                hA = rowA[1].h
                classA = rowA[1].obj_class

                df_comp = df_class.drop([rowA[0]])

                for rowB in df_comp.iterrows():
                    xB = rowB[1].x
                    yB = rowB[1].y
                    wB = rowB[1].w
                    hB = rowB[1].h
                    classB = rowB[1].obj_class

                    IOU = calc_IOU(xA, xB, yA, yB, wA, wB, hA, hB)

                    if IOU > 0:

                        df_inter_bis = df_inter_image[df_inter_image.obj_class == classA]

                        index_to_change = df_inter_bis.index[0]

                        df_inter.at[index_to_change,'count']+=1

    df_inter.loc[:,'count'] /= 2
    #pd.to_numeric(df_inter.loc[:,'count'], downcast='integer')
    df_inter = df_inter.sort_values(['root_image_path'])

    if print_res:
        print(df_inter)

    if edit_res:
        df_inter.to_csv(outname, index=False)



############################################## interactions analysis ###################################################


def interaction_analysis(outname_interaction_matrix = 'results/interaction_matrix.csv',
                         outname_interaction_table = 'results/interaction_table.csv',
                         outname_prey_predator_matrix = 'results/prey_pred_matrix.csv'):

    ############################################# interactions whatsowever ###########################"

    df = pd.read_csv('results/object_inter.csv')
    df_intra = pd.read_csv('results/object_intra.csv')

    class_list = []

    for dift_class in df.classA.unique():
        class_list.append(dift_class)

    for dift_class in df.classB.unique():
        if dift_class not in class_list:
            class_list.append(dift_class)

    class_list = sorted(class_list)

    ######################################## as a matrix ########################################

    list_of_dic = []

    for i in range(len(class_list)):
        class_dic = {}
        for dift_class in class_list:
            class_dic[dift_class] = 0
        list_of_dic.append(class_dic)

    int_df = pd.DataFrame(list_of_dic)

    class_index_dic = {}

    for i in range(len(class_list)):
        class_index_dic[i] = class_list[i]

    int_df.rename(index=class_index_dic, inplace=True)

    for row in df.iterrows():
        index, data = row
        int_df.at[data.classA, data.classB] += data[4]
        int_df.at[data.classB, data.classA] += data[4]

    for row in df_intra.iterrows():
        index, data = row
        int_df.at[data.obj_class, data.obj_class] += data[3]

    #print(int_df)

    int_df.to_csv(outname_interaction_matrix)

    ################################################# as a table (R friendly structure) #################################

    # n = len(class_list)
    #
    # list_of_dic = []
    #
    # for classA in class_list:
    #     for classB in class_list:
    #         dic = {}
    #         dic['classA'] = classA
    #         dic['classB'] = classB
    #         dic['count'] = 0
    #         list_of_dic.append(dic)
    #
    # int_df = pd.DataFrame(list_of_dic)
    #
    # for row in df.iterrows():
    #     index, data = row
    #     # print(data[4])
    #     int_df_temp = int_df[int_df.classA]
    #     int_df.at[data.classA, data.classB] += data[4]
    #     int_df.at[data.classB, data.classA] += data[4]
    #
    # for row in df_intra.iterrows():
    #     index, data = row
    #     # print(data)
    #     print(data[3])
    #     int_df.at[data.obj_class, data.obj_class] += data[3]
    #
    # int_df.to_csv(outname_interaction_table)

    ################################################# predator -> prey oriented #######################################

    with open('cfg/insects_detailed_predation.names') as f:
        labels = [line.rstrip('\n') for line in f]

    all_preys = []
    all_predators = []

    for label in labels:
        if label.split(' ')[1] == 'prey':
            all_preys.append(label.split(' ')[0])
        if label.split(' ')[1] == 'predator':
            all_predators.append(label.split(' ')[0])

    preys = []
    predators = []

    for dift_class in class_list:
        if dift_class in all_preys:
            preys.append(dift_class)
        if dift_class in all_predators:
            predators.append(dift_class)

    list_of_dic = []

    for i in range(len(preys)):
        class_dic = {}
        for dift_class in predators:
            class_dic[dift_class] = 0
        list_of_dic.append(class_dic)

    int_df = pd.DataFrame(list_of_dic)

    class_index_dic = {}

    for i in range(len(preys)):
        class_index_dic[i] = preys[i]

    int_df.rename(index=class_index_dic, inplace=True)

    for row in df.iterrows():
        index, data = row

        if (data.classA in preys) and (data.classB in predators):
            int_df.at[data.classA, data.classB] += data[4]

        if (data.classB in preys) and (data.classA in predators):
            int_df.at[data.classB, data.classA] += data[4]

    int_df.to_csv(outname_prey_predator_matrix)

#################################################### predation statistics #############################################


def predation_statistics(detections_csv_file):
    df = pd.read_csv(detections_csv_file)

    # generating template for result df

    with open('cfg/insects_detailed_predation.names') as f:
        labels = [line.rstrip('\n') for line in f]

    all_preys = []
    all_predators = []

    for label in labels:
        if label.split(' ')[1] == 'prey':
            all_preys.append(label.split(' ')[0])
        if label.split(' ')[1] == 'predator':
            all_predators.append(label.split(' ')[0])

    preys = []
    predators = []

    for dift_class in df.obj_class.unique():
        if dift_class in all_preys:
            preys.append(dift_class)
        if dift_class in all_predators:
            predators.append(dift_class)

    dic_of_dic = {}

    for prey in preys:
        class_dic = {}
        for dift_class in predators:
            class_dic[dift_class] = []
        dic_of_dic[prey] = class_dic

    #

    for image in df.root_image.unique():
        df_image = df[df.root_image == image]

        preys_img = []
        pred_img = []

        for classe in df_image.obj_class.unique():
            if classe in preys:
                preys_img.append(classe)

        for classe in df_image.obj_class.unique():
            if classe in predators:
                pred_img.append(classe)

        for classe_prey in preys_img:

            df_prey = df_image[df_image.obj_class == classe_prey]

            for rowA in df_prey.iterrows():
                xA = rowA[1].x
                yA = rowA[1].y
                wA = rowA[1].w
                hA = rowA[1].h

                for classe_pred in pred_img:

                    used = False
                    count = 0

                    df_pred = df_image[df_image.obj_class == classe_pred]

                    for rowB in df_pred.iterrows():
                        xB = rowB[1].x
                        yB = rowB[1].y
                        wB = rowB[1].w
                        hB = rowB[1].h

                        IOU = calc_IOU(xA, xB, yA, yB, wA, wB, hA, hB)

                        if IOU > 0:
                            used = True
                            count += 1

                    if used:

                        dic_of_dic[classe_prey][classe_pred].append(count)

    # get number of interactions between each prey and pred

    int_num_dic = {}

    for prey in preys:
        class_dic = {}
        for dift_class in predators:
            class_dic[dift_class] = 0
        int_num_dic[prey] = class_dic

    for prey in preys:
        for pred in predators :
            int_num_dic[prey][pred] = len(dic_of_dic[prey][pred])

    number_df = pd.DataFrame(int_num_dic)
    #print(number_df)
    number_df.to_csv('results/predation_event_numbers.csv')

    # get mean predator number per predation

    int_num_dic = {}

    for prey in preys:
        class_dic = {}
        for dift_class in predators:
            class_dic[dift_class] = 0
        int_num_dic[prey] = class_dic

    for prey in preys:
        for pred in predators :
            if len(dic_of_dic[prey][pred]) != 0 :
                mean = float(float(sum(dic_of_dic[prey][pred]))/float(len(dic_of_dic[prey][pred])))
                int_num_dic[prey][pred] = mean
            else:
                int_num_dic[prey][pred] = 'Na'

    mean_df = pd.DataFrame(int_num_dic)
    #print(mean_df)
    mean_df.to_csv('results/mean_pred_per_prey.csv')

    # get max predator number per predation

    int_num_dic = {}

    for prey in preys:
        class_dic = {}
        for dift_class in predators:
            class_dic[dift_class] = 0
        int_num_dic[prey] = class_dic

    for prey in preys:
        for pred in predators :
            if len(dic_of_dic[prey][pred]) != 0 :
                max_pred = max(dic_of_dic[prey][pred])
                int_num_dic[prey][pred] = max_pred
            else:
                int_num_dic[prey][pred] = 'Na'

    max_df = pd.DataFrame(int_num_dic)
    #print(max_df)
    max_df.to_csv('results/max_pred_per_prey.csv')

    # get mode predator number per predation

    int_num_dic = {}

    for prey in preys:
        class_dic = {}
        for dift_class in predators:
            class_dic[dift_class] = 0
        int_num_dic[prey] = class_dic

    for prey in preys:
        for pred in predators :
            if len(dic_of_dic[prey][pred]) != 0 :
                mode_pred = max(set(dic_of_dic[prey][pred]), key = dic_of_dic[prey][pred].count)
                int_num_dic[prey][pred] = mode_pred
            else:
                int_num_dic[prey][pred] = 'Na'

    mode_df = pd.DataFrame(int_num_dic)
    #print(mode_df)
    mode_df.to_csv('results/mode_pred_per_prey.csv')


############################ interactions for visualizing as a network #############################################


def reformat_interaction_file(interaction_csv='results/object_inter.csv', relation_csv='cfg/insects_relation.csv'):

    if not os.path.exists(relation_csv):

        print('csv file defining relations not found')

        return

    ################################# reformating ########################################

    df = pd.read_csv(interaction_csv)
    #print(df)

    class_list = []

    for dift_class in df.classA.unique():
        class_list.append(dift_class)

    for dift_class in df.classB.unique():
        if dift_class not in class_list:
            class_list.append(dift_class)

    class_list = sorted(class_list)

    #print(class_list)

    list_of_list = []

    for combo in combinations(class_list, 2):
        list = []
        list = [combo[0], combo[1], 0.0]
        list_of_list.append(list)

    df_inter = pd.DataFrame(list_of_list, columns=('from', 'to', 'count'))

    #print(df_inter)

    for row_int in df_inter.iterrows():

        index_int, data_int = row_int

        for row in df.iterrows():

            index, data = row

            if data_int['from'] == data['classA'] and data_int['to'] == data['classB']:
                #print(data['count'])
                #print(df_inter.at[index_int, 'count'])

                df_inter.at[index_int, 'count'] += data['count']

            if data_int['from'] == data['classB'] and data_int['to'] == data['classA']:
                df_inter.at[index_int, 'count'] += data['count']

    #df_inter.to_csv('inter_r_friendly.csv', index=False)

    ############################# add qualification #################################

    # add 'relation' column

    df_inter['relation'] = 'undefined'

    # read relation relation_csv

    relation_df = pd.read_csv(relation_csv)

    # replace relation type with the one defined in the csv

    for row_inter in df_inter.iterrows():

        index_inter, data_inter = row_inter

        for row_rel in relation_df.iterrows():

            index_rel, data_rel = row_rel

            if (data_rel['from'] == data_inter['from'] and data_rel['to'] == data_inter['to']) or  (data_rel['from'] == data_inter['to'] and data_rel['to'] == data_inter['from']):

                df_inter.at[index_inter, 'relation'] = data_rel['relation']

    df_inter.to_csv('results/inter_r_friendly.csv', index=False)
