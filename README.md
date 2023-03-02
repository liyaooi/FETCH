# FETCH

## Introduction
Code for [**"Learning a Data-Driven Policy Network for Pre-Training Automated Feature Engineering "**](https://openreview.net/forum?id=688hNNMigVX)

> Accepted in ICLR 2023 Conference

This is an automated feature engineering framework "FETCH", implemented in PyTorch.
Data can be accessed through URL in the paper.

## How to run it

`requirements.txt`: install needed packages under the environment settings by pip.

`python main_attention.py`: after specifying the dataset, cuda, and other parameters, you can run FETCH to automate feature engineering for Random Forest or other pre-defined model.

`. run_table_1.sh`: run Experiment 4.2.

`. run_table_3.sh`: run Experiment C.3 on large-scale dataset (> 50k rows).

`. run_eval_pretrain.sh`: run Experiment 4.3 on evaluating pre-trained models.

`. run_diff_models.sh`: run Experiment 4.4 and C.5 on different models.

`. run_table_5.sh`: run Experiment C.6.

## Reference Code

- NFS: https://github.com/TjuJianyu/NFS
- DIFER: https://github.com/PasaLab/DIFER