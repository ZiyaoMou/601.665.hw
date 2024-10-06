#!/bin/bash

# 参数设置
cs=(1)    # C 的值
ds=(10 50 200)         # d 的值
train_data_paths=("../data/gen_spam/train/gen" "../data/gen_spam/train/spam")  # 训练数据路径

# 训练模型的循环
for c in "${cs[@]}"
do
    for d in "${ds[@]}"
    do
        for train_data in "${train_data_paths[@]}"
        do
            # 根据训练数据路径提取 'gen' 或 'spam'
            dataset=$(basename "$train_data")
            
            # 生成模型输出文件名
            model_output="${dataset}-log-linear-c=${c}-d=${d}.model"
            
            # 运行训练脚本
            echo "Running: ./train_lm.py vocab-genspam.txt log_linear --lexicon ../lexicons/words-gs-only-${d}.txt --epochs 10 --l2_regularization ${c} ${train_data} --output ${model_output}"
            
            ./train_lm.py vocab-genspam.txt log_linear --lexicon ../lexicons/words-gs-only-${d}.txt --epochs 10 --l2_regularization ${c} ${train_data} --output ${model_output}
        done
    done
done