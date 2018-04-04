# Adversarial Examples on Image Recognition with DNNs

```
Created by Ran Liu, Zheng Qu, Yong Wu
```

### Statement: we would like to restate that we utilized several utils, gradient computation, class architecture from [cleverhans](https://github.com/tensorflow/cleverhans) community, we really appreciate the great contribution of the community

## 1. Get Started

Our approach is implemented in *++src/cleverhans/attack/attach_gradient_ascent_noise.py++*, borrowed several code snippets like load_images, save_images, and the tensorflow graph related info from cleverhans. Files under directory src/cleverhans/lib and src/cleverhans/defense are also from cleverhans.

We recommend using the following version info to run our code:

```
python: >= 3.5
conda(if available): >= 4.3
tensorflow: >= 1.0.1
```



## 2. Dataset Preparation

```
sh $ROOT_DIR/etc/download_data.sh
       
```
(Note: ROOT_DIR is the root directory of your repository, for example, "/home/ywu118/adversarial/"), script sh download_data.sh is used to help download relevant dataset, including images, and tensorflow pretrained models like resnet, ineception-v3

## 3. Execute Gradient Ascent with Noise approach


- **1). Perform "Gradient Ascent with Noise" attack:** 

```
cd $ROOT_DIR/src/cleverhans/attack 
python3 attach_gradient_ascent_noise.py  \
        --master="" \
        --checkpoint_path="$Dir/inception_v3.ckpt" \
        --input_dir=$images_dir \
        --output_dir=$output_dir \
        --max_epsilon=100 \
        --image_width=299 \
        --image_height=299 \
        --batch_size=1
```
Note: currently we just support batch_size 1, will support batch operation latter.

- **2). Get perturbation rate:** 

```
cd $ROOT_DIR/src/cleverhans 
python3 perturb_ratio.py --input_dir=$images_dir --output_dir=$output_dir
Note: Perturbation here means ratio of the sum of perturbation to the sum of original image pixels
```





### 4. Appendix
- **1). Examples We generated** 

<img src="https://github.com/yongcale/adversarial/blob/master/etc/dataset/d6eac6858474111c(ini-323-).png" width="200">         **+**         <img src="https://github.com/yongcale/adversarial/blob/master/etc/dataset/perturb_d6eac6858474111c.png" width="200">         **=**         <img src="https://github.com/yongcale/adversarial/blob/master/etc/dataset/d6eac6858474111c(adv-327).png" width="200">




