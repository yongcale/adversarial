
### Note: codes under adversarial/src/cleverhans are from the following sources,
###       more details please directly access them

- https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks.py
- https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py

To run the attack:

1) download related dataset:  <br>
       sh $ROOT_DIR/etc/download_data.sh  <br>
       (Note: ROOT_DIR is the root directory of your github repository, for example, "/home/ywu118/adversarial/")

2) then perform fgsm attack: <br>

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
