
## Fast Gradient Sign Method:

### Note: codes under adversarial/src/cleverhans are from the following sources,
###       more details please directly access them

- https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks.py
- https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py

To run the sample attack with Fast Gradient Sign Method:

1) download related dataset:
       sh /etc/download_data.sh

2) then perform fgsm attack
       python3 fgsm/attack_fgsm.py  --master="" --checkpoint_path="./inception_v3.ckpt"  \
                                    --input_dir="input" --output_dir="output" --max_epsilon=10 \
                                    --image_width=299 --image_height=299 --batch_size=16