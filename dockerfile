# Usamos la imagen oficial NVIDIA CUDA con cuDNN sobre Ubuntu 20.04
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

# Evitamos prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# 1) Actualizamos e instalamos Python y dependencias de sistema
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential \
    libjpeg-dev libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# 2) Copiamos el requirements y los instalamos
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# 3) Copiamos el código de la aplicación
COPY . /app

# 4) Creamos carpeta de uploads
RUN mkdir -p static/uploads

# 5) Exponemos el puerto de Flask
EXPOSE 5000

# 6) Comando por defecto
CMD ["python3", "app.py"]
