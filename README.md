# FCL with Continual ImageNet
This directory contains the implementation of the continual binary ImageNet classification problem.

The first step to replicate the results is to download the data. The data can be downloaded [here](https://drive.google.com/file/d/1i0ok3LT5_mYmFWaN7wlkpHsitUngGJ8z/view?usp=sharing).
Then download a file [here](https://drive.google.com/file/d/1qt6ucxtgVKsRdGvw72Phm916mSNlTMZB/view?usp=sharing).
Create a directory named `data` and extract the downloaded data folder in `data`. Also move the downloaded file `class_order`  to `data` 
```sh
cd FCL/
mkdir data
```

Remember to change the name of the extracted folder as `imagenet1k-classes`. Do not change the name of the `class_order` file. Then run the following command to run the code

```sh
python system/main.py --dataset IMAGENET1k --num_classes 1000 --wandb True 
```


