FROM ghcr.io/nvidia/jax:nightly-2023-10-06

RUN pip install sagemaker-training

COPY . /opt/ml/code

RUN pip install -e /opt/ml/code && pip install wandb && wandb login 4f5005827ec2a759dedaf23af4a049f9d6e6b568 
RUN pip install haliax==1.2 torch==2.2.2
RUN pip install -U "jax[cuda12_pip]"==0.4.18 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

ENV SAGEMAKER_PROGRAM train.py
