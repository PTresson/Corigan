from collections import Counter
import pandas as pd
import numpy as np
import os

def create_df(MyNameClassFile):
    nb_classes = 0
    with open(MyNameClassFile) as f:
            for i, l in enumerate(f):
                            pass
    nb_classes = i+1
    #print('nb_classes = ', nb_classes)
    df = pd.DataFrame(pd.np.empty((nb_classes, 3)), columns=['class', 'name', 'count'])
    names = [line.rstrip('\n') for line in open(MyNameClassFile)]
    df['class'] = range(0, nb_classes)
    df['name'] = names
    df['count'] = 0

    return df

def file_list_object(MyLabelFile, df):
    with open(MyLabelFile) as f:
            list_of_lists = []
            for line in f:
                inner_list = line.split(' ', 1)[0]
                list_of_lists.append(inner_list)

    count = dict(Counter(list_of_lists))
    #print('count = ', count)
    for key in count:
        df.at[int(key), 'count'] += count[key]
    #print(df)

def find_all_files(labels_dir, names_data_file_path, saved_data_file_path):

    files = []
    for f in os.listdir(labels_dir):
        if f.endswith(".txt"):
            #print(f)
            files.append(os.path.join(labels_dir, f))

    df = create_df(names_data_file_path)
    for f in files:
        #print(f)
        file_list_object(f, df)
    #print(df)
    df.to_csv(saved_data_file_path, index=False)