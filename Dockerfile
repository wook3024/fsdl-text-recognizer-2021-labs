FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# ARG LAB_NUM=1
# ENV PYTHONPATH="$PYTHONPATH:/workspace/lab${LAB_NUM}"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata && \
    apt-get -y install texlive-xetex && \
    apt-get -y install python3-pip
    
RUN pip3 install --upgrade pip

COPY ./ /workspace

WORKDIR /workspace
RUN pip3 install pip-tools && \
    pip-compile requirements/prod.in && pip-compile requirements/dev.in

RUN rm -r /usr/lib/python3/dist-packages/pycrypto* && \
    rm -r /usr/lib/python3/dist-packages/pygobject* && \
    rm -r /usr/lib/python3/dist-packages/pyxdg*

RUN pip-sync requirements/prod.txt requirements/dev.txt

RUN pip3 install einops

# WORKDIR /workspace/lab${LAB_NUM}
# RUN pip3 install --upgrade pip && \
#     pip3 install torch==1.7.0 torchvision==0.8.1 -f https://download.pytorch.org/whl/cu101/torch_stable.html
# RUN python3 training/run_experiment.py --max_epochs=10 --gpus=-1 --accelerator=ddp --num_workers=20 --data_class=MNIST --model_class=mlp
