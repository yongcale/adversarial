# Adversarial Examples on Image Recognition with DNNs

```
JHU graduate capstone project in Fall 2017
 Created by Ran Liu, Zheng Qu, Yong Wu
```

### We would like to state that we use several utils, gradient computation, class architecture from [cleverhans](https://github.com/tensorflow/cleverhans) community, we really appreciate the great contribution of the community

### Get Started

Our approach is implemented in *++src/cleverhans/attack/attach_gradient_ascent_noise.py++*, borrow several code snippet like load_images, save_images, and the tensorflow graph related info from cleverhans. Files under directory src/cleverhans/lib and src/cleverhans/defense are also from cleverhans.

We recommend to use he following version info to run our code:

```
python: >= 3.5
conda(if available): >= 4.3
tensorflow: >= 1.0.1
```



### Get Dataset

```
sh $ROOT_DIR/etc/download_data.sh  <br>
       
```
(Note: ROOT_DIR is the root directory of your github repository, for example, "/home/ywu118/adversarial/"), Perform "sh download_data.sh" to get related images dataset, and tensorflow pretrained models like resnet, ineception-v3

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
Note: currently we just support batch_size 1

- **Get perturbation rate:** 

```
cd $ROOT_DIR/src/cleverhans 
python3 perturb_ratio.py
--input_dir=$images_dir --output_dir=$output_dir
```





##### the following info is some docs we generate, please skip them
*Reports Latex template - https://www.overleaf.com/10853570cqtykqmmnmtd
proposal - adversarial/reports/[CPA&Proposal.pdf](https://github.com/yongcale/adversarial/blob/master/etc/report/CPA%26Proposal.pdf)
mid-project update[https://docs.google.com/document/d/1TQSCHRkNbKEyEXy47DUizdd5kNxNmqapIAxt5I4Ckf0/edit]
progress presentation PPT [https://docs.google.com/document/d/1TQSCHRkNbKEyEXy47DUizdd5kNxNmqapIAxt5I4Ckf0/edit?usp=sharing]
Final presentation PPT[https://docs.google.com/presentation/d/10aPZfIskeboQ367dO0CCDr6wGIdnv3mY8C0ZBPir_Xs/edit?ts=5a1b49f5#slide=id.p]*


