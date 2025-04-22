CUDA_VISIBLE_DEVICES=1 python main.py --dataset CIFAR100 --target_dir_name output --num_glob_iters 500 --wandb True
python main.py --dataset CIFAR100 --algorithm PreciseFCL --target_dir_name output --num_glob_iters 500 --wandb True
# python main.py --dataset CIFAR100 --algorithm PreciseFCL --target_dir_name output --num_glob_iters 500 --wandb True --lr 1e-05
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset CIFAR100 --target_dir_name output --num_glob_iters 500 --wandb True --lr 1e-05

python main.py --algorithm PreciseFCL --dataset CIFAR100 --target_dir_name output --num_glob_iters 50 --wandb True
python main.py --dataset CIFAR100 --target_dir_name output --num_glob_iters 50 --wandb True

python main.py --algorithm PreciseFCL --dataset CIFAR100 --target_dir_name output --num_glob_iters 200 --model Resnet18_plus --wandb True
python main.py  --dataset CIFAR100 --target_dir_name output --num_glob_iters 200 --model Resnet18_plus --wandb True
python main.py  --dataset CIFAR100 --target_dir_name output --num_glob_iters 200 --model S_ConvNet --wandb True
python main.py --algorithm PreciseFCL --dataset CIFAR100 --target_dir_name output --num_glob_iters 200 --model S_ConvNet --wandb True
python main.py  --dataset CIFAR100 --target_dir_name output --num_glob_iters 200 --model Resnet8_plus --wandb True
python main.py --algorithm PreciseFCL --dataset CIFAR100 --target_dir_name output --num_glob_iters 200 --model Resnet8_plus --wandb True
python main.py --algorithm PreciseFCL --dataset IMAGENET1k --target_dir_name output --num_glob_iters 200 --model S_ConvNet --wandb True
python main.py --algorithm PreciseFCL --dataset IMAGENET1k --target_dir_name output --num_glob_iters 200 --model S_ConvNet


CUDA_VISIBLE_DEVICES=1 python main.py --algorithm PreciseFCL --dataset CIFAR100 --target_dir_name output --num_glob_iters 500 --model Resnet12_plus --wandb True