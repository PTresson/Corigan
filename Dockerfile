FROM nvidia/cuda:9.0-devel-ubuntu16.04
#Starting from a NVIDIA image with CUDA 9

# Get CUDNN
ENV CUDNN_VERSION 7.6.0.64
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"


# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]


RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# Install python 3, dependencies etc ...
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-4.8 \
        git \
        vim \
        wget \
	unzip \
        ca-certificates \
        python3 \
        python3-dev \
	libmysqlclient-dev \
	python3-setuptools \
	python3-pip  \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install requiered python modules with pip
RUN pip3 install --upgrade setuptools --upgrade pip
RUN pip3 install future typing \	
	numpy \
        pandas \
        matplotlib \
	Pillow


# Install OpenCV for darknet (here we go with version 3.2.0)

WORKDIR /
ENV OPENCV_VERSION="3.2.0"
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& unzip ${OPENCV_VERSION}.zip \
&& mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
&& cd /opencv-${OPENCV_VERSION}/cmake_binary \
&& cmake -DBUILD_TIFF=ON \
  -DBUILD_opencv_java=OFF \
  -DWITH_CUDA=OFF \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  .. \
&& make install \
&& rm /${OPENCV_VERSION}.zip \
&& rm -r /opencv-${OPENCV_VERSION}


RUN apt-get update
RUN apt-get dist-upgrade -y
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y libopencv-dev
RUN pkg-config --cflags opencv
RUN pkg-config --libs opencv

RUN bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
RUN ldconfig




