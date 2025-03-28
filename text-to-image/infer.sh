gpu_num=1
python sample_t2i.py --prompt "渔舟唱晚"  --no-enhance --infer-steps 100 --image-size 1024 1024 
# if you need accelerate, add following command to the end of the script
# --cache --cache-step 2