export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

for model_type in ssast; do
    for gen_method in class_prompt llm; do
        CUDA_VISIBLE_DEVICES=0, taskset -c 1-30 python3 finetune_gen_esc50.py --pretrain_model $model_type --dataset $dataset --learning_rate 0.0001 --num_epochs 30 --gen-per-class 30 --gen-method $gen_method --gen-model audiogen
    done
done