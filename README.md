# CORIGAN

Corigan is a pipeline for the detection of small objects on large input images and the analysis of the interctions between these objects.

## Requierements

Neural network related computations require CUDNN >= 7.0, CUDA >= 9.0 and OpenCV >= 2.4 (optional). Interactions require Python 3.6, numpy and pandas. Analysis scripts work best with python-opencv but are possible with PIL. Matplotlib is used for ploting the results.
A working Dockerfile will be provided anytime soon.

Neural network related computations are performed within the Darknet framework provided by J. Redmon and maintained by AlexeyAB https://github.com/AlexeyAB/darknet. We recommand that you have tested the Darknet framework first.
To work, Darknet needs a `.cfg` file specifying the configuration of the network, a `.names` file specifying the classes you are working with and a `.data` file linking all data together. 
The configuration changes with the number of classes you are working on, for detailled explaination, please refer to https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects. 

## Use

The Darknet executive should ultimatly be in the `Corigan` directory.You can compile Darknet inside the `Corigan` directory or elsewhere and later copy the darknet executive to the `Corigan` directory.
`.cfg`, `.names` and `.data` are in the `Corigan/cfg` directory.


pipeline is located in `Corigan/core` but is to be excecuted from the `Corigan` directory, not `/core` directory

```
~$ cd Corigan
~/Corigan$ python3 core/pipeline.py
```

### Train the model

Copy your train dataset (images and labels) into the `train_images` directory.
`.cfg` file and `.names` file should be in the `/cfg` directory. A train.data file will be generated automatically.
To start training, execute the following command line inside the `Corigan` directory.

```
~/Corigan$ python3 core/pipeline.py --mode='train'
 ```

### Test the model

Copy your test dataset (images and labels) into the `/test_images` directory.
`.cfg` file and `.names` file should be in the `/cfg` directory. The pipeline will use the previously generated `train.data` file but you can change it easily.
You can specify wich weight file you want to use for the network.
To start testing, execute the following command line inside the 'pipeline' directory.

```
~/Corigan$ python3 core/pipeline.py --mode='test' --weight_file='yolov3-detailled_final.weigths'
```

Note that 'test' mode is the default mode, so you can execute

```
~/Corigan$ python3 core/pipeline.py --weight_file='yolov3-detailled_final.weigths'
```

with the same result.

In the `/test_temp` directory will be a `result.txt` countaining raw YOLO detections and a `refined_detection.csv` countaining the refined detections of the network (the ones you will likely be working on).

### Classes mode

We have compared the performance between simple and detailled classes, so `.cfg` files and `.names` files exist is these two versions. You can easily get rid of this precision by commenting or deleting the lines 

```
#28 parser.add_argument('--classes', type=str, default='detailed', help="simple or detailed")
#46 classes_mode = args.classes
```
in the `core/pipeline.py' file
and changing

```
#72 cfg_file = 'yolov3-' + classes_mode + '.cfg'
#80 name_file = 'insects_' + classes_mode + '.names'
```

to your `.cfg` file and `.name` file name. 



### Plot the result

Plotting is disabled by default but you can enable it and show plots or save plots. If you have a large number of test images, show plot might not be the best option since it will pop figures up. As well, with a large number or plot to save, Matplotlib will throw a warning (but plotting is still done).

```
~/Corigan$ python3 core/pipeline.py --mode='test' --weight_file='yolov3-detailled_final.weigths' --save_plot=True
```

### Not using OpenCV

We use OpenCV since it is faster than PIL to precess images. However, all OpenCV version are not compatible with our setup (prefer version <4.0) and OpenCV can be complicated to install depending on your setup. For these reasons, we also provide a version of the slicing using PIL. In this case, you can specify

```
~/Corigan$ python3 core/pipeline.py --mode='train' --dont_use_openCV=True
~/Corigan$ python3 core/pipeline.py --mode='test' --dont_use_openCV=True
```

### Skip slicing

If you are not changing your dataset, you might want to skip slicing. In this case you can specifiy:

```
~/Corigan$ python3 core/pipeline.py --mode='train' --dont_slice=True
~/Corigan$ python3 core/pipeline.py --mode='test' --dont_slice=True
```

### Skip comparison with ground truth

If you only want the detections of the network without a performance analysis, you can skip this part by specifying:

```
~/Corigan$ python3 core/pipeline.py --mode='train' --dont_metrics=True
~/Corigan$ python3 core/pipeline.py --mode='test' --dont_metrics=True
```

### Ecological outputs

Outputs will be found in the `/results` directory.
R scripts for generating plots with the data are available in the `/result` directory as well.
