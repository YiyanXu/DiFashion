nohup python -u evaluate_grounding_gor.py  --dataset $1 --eval_version $2 --ckpts $3 --task GOR --mode $4 --log_name $5 --gpu $6 >./eval_logs/log_eval_grounding_gor_$1_$2_$3_$4_$5.txt 2>&1 &

# sh run_eval_grounding_gor.sh ifashion DiFashion all test log 4


