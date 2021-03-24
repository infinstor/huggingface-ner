FROM pytorch/pytorch

RUN apt update
RUN apt install libgdcm-tools git emacs -y

RUN pip install 'git+https://github.com/huggingface/transformers' mlflow infinstor infinstor-mlflow-plugin jupyterlab-infinstor

