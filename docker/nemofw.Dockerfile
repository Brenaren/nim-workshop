FROM nvcr.io/nvidia/nemo:25.04.01
USER root
# Set root password
RUN echo 'root:nvidia123' | chpasswd

ARG BUILDVER=1.0

EXPOSE 8888
EXPOSE 8787
EXPOSE 8786

ENV DEBIAN_FRONTEND=noninteractive

RUN chmod 777 /tmp
COPY apt_packages.txt /workspace/apt_packages.txt
RUN apt-get update && \
    xargs -a /workspace/apt_packages.txt apt-get install -y --no-install-recommends  && \
    rm -rf /var/lib/apt/lists/*

# Allow sudo group to use sudo without password
RUN usermod -aG sudo ubuntu
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Switch to ubuntu user
USER ubuntu
WORKDIR /workspace

# copy api keys to container
COPY .env /workspace/

############ Install Python pkgs #########
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip && pip install --upgrade -r /workspace/requirements.txt

# Temporary - reintegrate above later
USER root
RUN mkdir -p /home/ubuntu/.config && chown -R ubuntu:ubuntu /home/ubuntu/.config 
# && chown ubuntu:ubuntu -R /opt/
ENV PATH="${PATH}:/home/ubuntu/.local/bin"

# mkdir /gitrepos
# git clone https://github.com/NVIDIA/NeMo-Skills.git
RUN mkdir -p /gitrepos
RUN git clone https://github.com/NVIDIA/NeMo-Skills.git /gitrepos/NeMo-Skills && \
    git clone https://github.com/NVIDIA/NeMo.git /gitrepos/NeMo && \
    git clone https://github.com/NVIDIA/NeMo-Curator.git /gitrepos/NeMo-Curator


############ Install Python pkgs #########
RUN echo 'ubuntu:nvidia123' | chpasswd
USER ubuntu
ENV PYTHONPATH="/opt/NeMo/:/workspace/scripts:${PYTHONPATH}"
