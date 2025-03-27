# !/bin/bash
device=auto

category=synthetic
scenario=univariate
for model_name in gemini-1.5-flash gemini-1.5-pro gpt-4o-mini gpt-4o llama3-llava-next-8b llava-next-72b Qwen2-VL-7B-Instruct Qwen2-VL-72B-Instruct
do
    for data in global contextual seasonal trend shapelet
    do
        python main.py --category $category --scenario $scenario --model_name $model_name --data $data --device $device
    done
done

category=synthetic
scenario=multivariate
for dim in 4 9 16 25 36
do
    for model_name in gemini-1.5-flash gemini-1.5-pro gpt-4o-mini gpt-4o llama3-llava-next-8b llava-next-72b Qwen2-VL-7B-Instruct Qwen2-VL-72B-Instruct
    do
        for data in triangle square sawtooth random_walk
        do
            python main.py --category $category --scenario $scenario --model_name $model_name --data $data --dim $dim --device $device
        done
    done
done

category=synthetic
scenario=irr_univariate
for drop_ratio in 0.05 0.10 0.15 0.20 0.25
do
    for model_name in gemini-1.5-flash gemini-1.5-pro gpt-4o-mini gpt-4o llama3-llava-next-8b llava-next-72b Qwen2-VL-7B-Instruct Qwen2-VL-72B-Instruct
    do
        for data in global seasonal trend shapelet
        do
            python main.py --category $category --scenario $scenario --model_name $model_name --data $data --drop_ratio $drop_ratio --device $device
        done
    done
done

category=synthetic
scenario=irr_multivariate
for drop_ratio in 0.05 0.10 0.15 0.20 0.25
do
    for model_name in gemini-1.5-flash gemini-1.5-pro gpt-4o-mini gpt-4o llama3-llava-next-8b llava-next-72b Qwen2-VL-7B-Instruct Qwen2-VL-72B-Instruct
    do
        for data in triangle square sawtooth random_walk
        do
            python main.py --category $category --scenario $scenario --model_name $model_name --data $data --drop_ratio $drop_ratio --device $device
        done
    done
done



category=semi
scenario=univariate
tsname=Symbols
for model_name in gemini-1.5-flash gemini-1.5-pro gpt-4o-mini gpt-4o llama3-llava-next-8b llava-next-72b Qwen2-VL-7B-Instruct Qwen2-VL-72B-Instruct
do
    for data in global contextual trend shapelet
    do
        python main.py --category $category --scenario $scenario --tsname $tsname --model_name $model_name --data $data --device $device
    done
done

category=semi
scenario=irr_univariate
tsname=Symbols
for drop_ratio in 0.05 0.10 0.15 0.20 0.25
do
    for model_name in gemini-1.5-flash gemini-1.5-pro gpt-4o-mini gpt-4o llama3-llava-next-8b llava-next-72b Qwen2-VL-7B-Instruct Qwen2-VL-72B-Instruct
    do
        for data in global trend shapelet
        do
            python main.py --category $category --scenario $scenario --tsname $tsname --model_name $model_name --data $data --drop_ratio $drop_ratio --device $device
        done
    done
done

category=semi
scenario=multivariate
tsname=ArticularyWordRecognition
for dim in 4 9 16 25 36
do
    for model_name in gemini-1.5-flash gemini-1.5-pro gpt-4o-mini gpt-4o llama3-llava-next-8b llava-next-72b Qwen2-VL-7B-Instruct Qwen2-VL-72B-Instruct
    do
        for data in triangle square sawtooth random_walk
        do
            python main.py --category $category --scenario $scenario --tsname $tsname --model_name $model_name --data $data --dim $dim --device $device
        done
    done
done

category=semi
scenario=irr_multivariate
tsname=ArticularyWordRecognition
dim=9
for drop_ratio in 0.05 0.10 0.15 0.20 0.25
do
    for model_name in gemini-1.5-flash gemini-1.5-pro gpt-4o-mini gpt-4o llama3-llava-next-8b llava-next-72b Qwen2-VL-7B-Instruct Qwen2-VL-72B-Instruct
    do
        for data in triangle square sawtooth random_walk
        do
            python main.py --category $category --scenario $scenario --tsname $tsname --model_name $model_name --data $data --drop_ratio $drop_ratio --dim $dim --device $device
        done
    done
done
