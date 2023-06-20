# FETCH

## Introduction
Code for [**"Learning a Data-Driven Policy Network for Pre-Training Automated Feature Engineering "**](https://openreview.net/forum?id=688hNNMigVX)

> Accepted in ICLR 2023 Conference

This is an automated feature engineering framework "FETCH", implemented in PyTorch.
Data can be accessed through URL in the paper.

## How to run it

`pip install -r requirements.txt`: install needed packages under the environment settings by pip.

`python main_attention.py`: after specifying the dataset, cuda, and other parameters, you can run FETCH to automate feature engineering for Random Forest or other pre-defined model.

`. run_table_1.sh`: run Experiment 4.2.

`. run_table_3.sh`: run Experiment C.3 on large-scale dataset (> 50k rows).

`. run_eval_pretrain.sh`: run Experiment 4.3 on evaluating pre-trained models.

`. run_diff_models.sh`: run Experiment 4.4 and C.5 on different models.

`. run_table_5.sh`: run Experiment C.6.

## Environment
The code in this repository is designed and highly recommended to be run on `Ubuntu 20.04` or other Linux systems.

> Note that running the code with multiple processes (`args.worker > 1`) on `Windows` may encounter issues with variable sharing. 
> If you are using `Windows`, consider setting up a Linux environment (e.g., using a virtual machine or WSL) to run the code.

## Citation
```
@inproceedings{li2023learning,
  title={Learning a Data-Driven Policy Network for Pre-Training Automated Feature Engineering},
  author={Li, Liyao and Wang, Haobo and Zha, Liangyu and Huang, Qingyi and Wu, Sai and Chen, Gang and Zhao, Junbo},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```

## Reference Code

- NFS: https://github.com/TjuJianyu/NFS
- DIFER: https://github.com/PasaLab/DIFER