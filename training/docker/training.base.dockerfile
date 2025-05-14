FROM pytorch/pytorch:latest
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git python3.8 python3.8-dev python3.8-venv ffmpeg libsm6 libxext6
ENV VIRTUAL_ENV=/opt/venv
RUN python3.8 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY training/requirements.txt /tmp/training-requirements.txt
RUN mkdir /tmp/adex_gym
COPY adex_gym /tmp/adex_gym/adex_gym
COPY setup.py /tmp/adex_gym/setup.py
COPY README.md /tmp/adex_gym/README.md
COPY requirements.txt /tmp/adex_gym/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/training-requirements.txt \
    && pip install /tmp/adex_gym