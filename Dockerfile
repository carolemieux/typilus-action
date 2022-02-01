FROM python:3.8-buster

ENV PYTHONUNBUFFERED=1

RUN apt update && apt -y upgrade
RUN apt install -y python3-numpy python3-pip python3-requests
RUN pip3 install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch-scatter==2.0.4+cpu --use-deprecated=legacy-resolver -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN pip3 install dpu-utils typed-ast 
RUN apt-get install -y git
RUN cd /usr/src && git clone https://github.com/microsoft/ptgnn.git && cd /usr/src/ptgnn && git checkout tags/v0.7.8 && pip3 install -e . && python3 setup.py test

ENV PYTHONPATH=/usr/src/
ADD https://github.com/typilus/typilus-action/releases/download/v0.1/typilus20200507.pkl.gz /usr/src/model.pkl.gz
COPY src /usr/src
COPY entrypoint.py /usr/src/entrypoint.py

ENTRYPOINT ["python", "/usr/src/entrypoint.py"]
