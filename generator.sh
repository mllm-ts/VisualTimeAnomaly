# !/bin/bash
num_ts=100

category=synthetic
scenario=univariate
for anomaly_type in global contextual seasonal trend shapelet
do
    python generator.py --category $category --scenario $scenario --anomaly_type $anomaly_type --num_ts $num_ts
done

category=synthetic
scenario=irr_univariate
for drop_ratio in 0.05 0.10 0.15 0.20 0.25
do
    for anomaly_type in global seasonal trend shapelet
    do
        python generator.py --category $category --scenario $scenario --anomaly_type $anomaly_type --num_ts $num_ts --drop_ratio $drop_ratio
    done
done

category=synthetic
scenario=multivariate
for dim in 4 9 16 25 36
do
    for anomaly_type in triangle square sawtooth random_walk
    do
        python generator.py --category $category --scenario $scenario --anomaly_type $anomaly_type --num_ts $num_ts --dim $dim
    done
done

category=synthetic
scenario=irr_multivariate
for drop_ratio in 0.05 0.10 0.15 0.20 0.25
do
    for anomaly_type in triangle square sawtooth random_walk
    do
        python generator.py --category $category --scenario $scenario --anomaly_type $anomaly_type --num_ts $num_ts --drop_ratio $drop_ratio
    done
done



category=semi
scenario=univariate
tsname=Symbols
for anomaly_type in global contextual trend shapelet
do
    python generator.py --category $category --scenario $scenario --tsname $tsname --anomaly_type $anomaly_type --num_ts $num_ts
done

category=semi
scenario=irr_univariate
tsname=Symbols
for drop_ratio in 0.05 0.10 0.15 0.20 0.25
do
    for anomaly_type in global trend shapelet
    do
        python generator.py --category $category --scenario $scenario --tsname $tsname --anomaly_type $anomaly_type --num_ts $num_ts --drop_ratio $drop_ratio
    done
done

category=semi
scenario=multivariate
tsname=ArticularyWordRecognition
for dim in 4 9 16 25 36
do
    for anomaly_type in triangle square sawtooth random_walk
    do
        python generator.py --category $category --scenario $scenario --tsname $tsname --anomaly_type $anomaly_type --num_ts $num_ts --dim $dim
    done
done

category=semi
scenario=irr_multivariate
tsname=ArticularyWordRecognition
dim=9
for drop_ratio in 0.05 0.10 0.15 0.20 0.25
do
    for anomaly_type in triangle square sawtooth random_walk
    do
        python generator.py --category $category --scenario $scenario --tsname $tsname --anomaly_type $anomaly_type --num_ts $num_ts --drop_ratio $drop_ratio --dim $dim
    done
done



