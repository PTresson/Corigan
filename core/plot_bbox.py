import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from PIL import Image
import numpy as np
import pandas as pd

def plot_bbx(source_img, coordinates_df, show=True, save = False, outname='plot.jpg', res=400) :

    #df = pd.DataFrame(coordinates_list, columns=('root_image', 'obj_class', 'confidence', 'x', 'y', 'w', 'h'))

    df = coordinates_df
    root_name_path = source_img.split(".")[0]
    root_name = root_name_path.split('/')[-1]
    df = df[df.root_image == root_name]

    im = np.array(Image.open(source_img), dtype=np.uint8)
    nH = im.shape[0]
    nW = im.shape[1]

    classes = df.obj_class.unique()

    colors = cm.rainbow(np.linspace(0, 1, len(classes)))

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    for row in df.iterrows():

        # left_x = row[1].x
        # top_y = row[1].y
        # w_obj = row[1].w
        # h_obj = row[1].h

        left_x = int((row[1].x - 0.5 * row[1].w) * nW)
        top_y = int((row[1].y - 0.5 * row[1].h) * nH)

        w_obj = row[1].w * nW
        h_obj = row[1].h * nH

        # Create a Rectangle patch

        # different color for each class

        for i in range(len(classes)):

            if row[1].obj_class == classes[i]:
                color_obj = colors[i]

        rect = patches.Rectangle((left_x, top_y), w_obj, h_obj, linewidth=1, edgecolor=color_obj, facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    for i in range(len(classes)):

        ax.text(0.99, 1.01+float(i/45), classes[i],
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                color=colors[i], fontsize=5)

    if show:
        plt.show()

    if save:
        plt.savefig(outname, dpi=res)


