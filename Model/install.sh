#!/bin/bash

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then
  
  # put your install commands here:
  apt update
  apt clean
 # apt-get install python3-tk
  #apt-get install python3-tk
  #python -m requirements.txt
  pip3 install -r requirements.txt
  #pip3 install tensorboardX
  pip3 install timm 
  pip3 install torch_geometric
  pip3 install info-nce-pytorch
  pip3 install vec2text
#pip3 install gdown
  #gdown --folder https://drive.google.com/drive/folders/1_MiEfICh-T2cc3I8_LwWvY0xvb-C4MsJ
  nvidia-smi
  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi

# This runs your wrapped command
"$@"
