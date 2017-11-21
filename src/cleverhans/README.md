
## Fast Gradient Sign Method:

### Note: codes under adversarial/src/cleverhans are from the following sources,
###       more details please directly access them

- https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks.py
- https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py

To run the sample attack with Fast Gradient Sign Method:

1) download related dataset:  <br>
       sh $ROOT_DIR/etc/download_data.sh  <br>
       (Note: ROOT_DIR is the root directory of your github repository, for example, "/home/ywu118/adversarial/")

2) then perform fgsm attack: <br>
       cd $ROOT_DIR/src/cleverhans/fgsm  <br>
       python3 ./attack_fgsm.py  --master="" --checkpoint_path="./inception_v3.ckpt" --input_dir="input" --output_dir="output" --max_epsilon=10 --image_width=299 --image_height=299 --batch_size=16
