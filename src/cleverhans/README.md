
To run the attack:

1) download related dataset:  <br>
       sh $ROOT_DIR/etc/download_data.sh  <br>
       (Note: ROOT_DIR is the root directory of your github repository, for example, "/home/ywu118/adversarial/")

2) then perform the attack: <br>

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

