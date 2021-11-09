FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
# python install
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN update-alternatives --set python /usr/bin/python3.7
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py &&     python get-pip.py --force-reinstall &&     rm get-pip.py
RUN apt-get install unzip 
RUN apt-get update
RUN apt-get install graphviz

# pip install
RUN pip install -U pip
RUN pip install --upgrade pip
RUN pip install opencv-python==3.4.2.17
RUN pip install Pillow==8.3.1
RUN pip install tensorflow==2.1.0
RUN pip install pandas
RUN pip install h5py==2.10.0
RUN pip install turfpy
RUN pip install numpy==1.19.5
RUN pip install matplotlib==3.4.3
RUN pip install pandas==1.3.2
RUN pip install tqdm
RUN pip install flask
RUN pip install PyYAML
RUN pip install torchvision==0.8.2
RUN pip install ffmpeg-python>=0.2.0
RUN pip install matplotlib>=3.0.2
RUN pip install munkres>=1.1.2
RUN pip install numpy>=1.16
RUN pip install Pillow>=5.4
RUN pip install vidgear>=0.1.4
RUN pip install torch>=1.4.0
RUN pip install tqdm>=4.26
RUN pip install tensorboard==2.1.1
RUN pip install tensorboardX==1.6
RUN pip install pydot==1.4.2
RUN pip install terminaltables


RUN apt-get update && apt-get install -y libopencv-dev 

WORKDIR /home/duser/