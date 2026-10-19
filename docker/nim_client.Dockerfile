# Assume you have a client running on a gpu enbaled machine.  Using pytorch
# container b/c has a lot of nice development tools already installed.
# This is a good starting point for a client container.
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Install any extra packages you might want.
USER root
RUN apt-get update && apt-get install -y iputils-ping wget tree

############ Add Custom USER - Many ways to do this  #########
# I create a dir at home which I'll use to persist after the container is closed (need to change it's ownership)
# Add my id to make this easier, and be rootless
ARG UID=1000
ARG DOCKER_ID=nim_user
# Set the PS1 environment variable to customize the prompt and Echo the desired prompt format to .bashrc
#add sudoers capability

RUN useradd -m -u ${UID} -s /bin/bash ${DOCKER_ID}  && \
   echo '${DOCKER_ID} ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
   echo 'PS1="\u@docker:\$PWD\$ "' >> /home/${DOCKER_ID}/.bashrc

USER $DOCKER_ID
WORKDIR /workspace
# copy api keys to container
COPY .env /workspace/

############ Install Python pkgs #########
    # Install Python packages
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip && pip install --upgrade -r /workspace/requirements.txt
