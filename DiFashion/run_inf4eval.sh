nohup accelerate launch --config_file config.yaml inf4eval.py --eta 0.1 --task $1 --mode $2 --use_ema --use_ema_fashion --enable_xformers_memory_efficient_attention --mixed_precision fp16 >log_inf4eval_$1_$2.txt 2>&1 &

# sh run_inf4eval.sh FITB test
# sh run_inf4eval.sh GOR test
