FROM ubuntu:latest

USER root

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Get updates and install necessary packages
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN apt-get install -y graphviz 


# Download and Install miniconda python 3.6
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.3.31-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH=/root/miniconda3/bin:$PATH

# Install necessary python packages for development
COPY requirements.txt /
RUN pip install --no-cache-dir -r requirements.txt


WORKDIR /home/code

# start jupyter notebook at exposed port
EXPOSE 8888
CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root

