# Adversarial Examples on Image Recognition with DNNs

```
JHU graduate capstone project in Fall 2017
 Created by Ran Liu, Zheng Qu, Yong Wu
```

### We would like to restate that we utilized several utils, gradient computation, class architecture from [cleverhans](https://github.com/tensorflow/cleverhans) community, we really appreciate the great contribution of the community

### Get Started

Our approach is implemented in *++src/cleverhans/attack/attach_gradient_ascent_noise.py++*, borrowed several code snippets like load_images, save_images, and the tensorflow graph related info from cleverhans. Files under directory src/cleverhans/lib and src/cleverhans/defense are also from cleverhans.

We recommend using the following version info to run our code:

```
python: >= 3.5
conda(if available): >= 4.3
tensorflow: >= 1.0.1
```



### Get Dataset

```
sh $ROOT_DIR/etc/download_data.sh  <br>
       
```
(Note: ROOT_DIR is the root directory of your repository, for example, "/home/ywu118/adversarial/"), script sh download_data.sh is used to help download relevant dataset, including images, and tensorflow pretrained models like resnet, ineception-v3

### Execute Gradient Ascent with Noise approach


- **perform gradient ascent with noise attack:** 

```
cd $ROOT_DIR/src/cleverhans/attack 
python3 attach_gradient_ascent_noise.py  
        --master=""
        --checkpoint_path="$Dir/inception_v3.ckpt"
        --input_dir=$images_dir
        --output_dir=$output_dir
        --max_epsilon=100 
        --image_width=299
        --image_height=299 
        --batch_size=1
```
Note: currently we just support batch_size 1, will support batch operation latter.

- **Get perturbation rate:** 

```
cd $ROOT_DIR/src/cleverhans 
python3 perturb_ratio.py
--input_dir=$images_dir --output_dir=$output_dir
```





##### the following info are several docs we generated, please skip them


*proposal - adversarial/etc/reports/[proposal.pdf](https://github.com/yongcale/adversarial/blob/master/etc/report/proposal.pdf)<br>
CPA - adversarial/etc/reports/[CPA.pdf](https://github.com/yongcale/adversarial/blob/master/etc/report/Capstone%20%20CPA.pdf) <br>
mid-project [update PPT](https://docs.google.com/document/d/1TQSCHRkNbKEyEXy47DUizdd5kNxNmqapIAxt5I4Ckf0/edit)<br>
progress presentation [PPT](https://docs.google.com/document/d/1TQSCHRkNbKEyEXy47DUizdd5kNxNmqapIAxt5I4Ckf0/edit?usp=sharing)<br>
Final Reports [report.pdf](https://github.com/yongcale/adversarial/blob/master/etc/report/capstone-adversarial.pdf) -- ([Latex version](https://www.overleaf.com/10853570cqtykqmmnmtd) )<br>
Final presentation [PPT](https://docs.google.com/presentation/d/10aPZfIskeboQ367dO0CCDr6wGIdnv3mY8C0ZBPir_Xs/edit)*


