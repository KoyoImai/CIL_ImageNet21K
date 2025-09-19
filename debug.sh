export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200


python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --config-path ./configs/default/ --config-name default-er