srun -K  --job-name=lali \
	 --partition=A100-80GB,H100,H200,RTXA6000\
	 --gpus=1 \
	 --cpus-per-task=8 \
	 --mem=200G \
	 --output output/console.%A_%a.out \
	 --error output/console.%A_%a.error  \
         --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.07-py3.sqsh \
         --container-workdir=`pwd` \
         --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
         --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
          install.sh python Cluster_Pretraining_JLR_11.py "$@" &
