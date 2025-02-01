CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR100 --algorithm PreciseFCL --target_dir_name output --num_glob_iters 500
python main.py --dataset CIFAR100 --algorithm PreciseFCL --target_dir_name output --num_glob_iters 500 --wandb True
python main.py --dataset CIFAR100 --algorithm PreciseFCL --target_dir_name output --num_glob_iters 500 --wandb True --lr 1e-05
CUDA_VISIBLE_DEVICES=1 python main.py --dataset CIFAR100 --target_dir_name output --num_glob_iters 500 --wandb True --lr 1e-05