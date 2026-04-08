#FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
#
#WORKDIR /app
#
## Install Python & dependencies
#RUN apt-get update && apt-get install -y \
#    python3.10 \
#    python3-pip \
#    && rm -rf /var/lib/apt/lists/*
#
## Copy requirements
#COPY requirements.txt .
#
## Install Python packages
#RUN pip install --no-cache-dir -r requirements.txt
#
## Copy application
#COPY . .
#
## Expose port
#EXPOSE 5000
#
## Run app
#CMD ["python3", "app.py"]


FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir \
    --default-timeout=1000 \
    --retries 5 \
    -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python3", "app.py"]