FROM ubuntu:18.04

WORKDIR /usr/src/app
RUN apt update
RUN apt-get install -y \
    python3.6 \
    python3-pip
RUN pip3 install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
COPY utils utils
COPY dataset dataset
COPY trained_models trained_models
COPY evaluate.py evaluate.py
COPY train.py train.py

CMD [ "python3", "train.py"]
CMD [ "python3", "evaluate.py"]

