import copy
import logging
import os
import pickle
import random
import time

import multiprocessing
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold, cross_val_score

from feature_engineer import PPO, Memory
from feature_engineer import get_ops
from feature_engineer.attention_searching.training_ops import sample, multiprocess_reward, apply_actions
from feature_engineer.attention_searching.worker import Worker
from feature_engineer.fe_parsers import parse_actions
from metrics import metric_fuctions
from metrics.metric_evaluate import rae_score
from models import *
from process_data import Feature_type_recognition, split_train_test, Pipeline, feature_pipeline
from process_data.feature_process import label_encode_to_onehot, features_process, remove_duplication
from utils import log_dir, get_key_from_dict, reduce_mem_usage


def get_test_score(df_train, df_test, label_train, label_test, args, mode, model, metric):
    if args.worker == 0 or args.worker == 1:
        n_jobs = -1
    else:
        n_jobs = 1
    model = model_fuctions[f"{model}_{mode}"](n_jobs)
    model.fit(df_train, label_train)
    # pred = model.predict(df_test)
    score = metric_fuctions[metric](model, df_test, label_test, label_train)
    return score


class AutoFE:
    """Main entry for class that implements automated feature engineering (AutoFE)"""

    def __init__(self, input_data: pd.DataFrame, args):
        # Create log directory
        times = time.strftime('%Y%m%d-%H%M')
        log_path = fr"./logs/train/{args.file_name}_{times}"
        if args.enc_c_pth != '':
            log_path = fr"./logs/pre/{args.file_name}_{args.enc_c_pth.split('_')[4].split('.')[0]}_{times}"
        log_dir(log_path)
        logging.info(args)
        logging.info(f'File name: {args.file_name}')
        logging.info(f'Data shape: {input_data.shape}')
        # Fixed random seed
        self.seed = args.seed
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        self.shuffle = args.shuffle
        # Deal with input parameters
        self.train_size = args.train_size
        self.split = args.split_train_test
        self.combine = args.combine
        self.info_ = {}
        self.info_['target'] = args.target
        self.info_['file_name'] = args.file_name
        self.info_['mode'] = args.mode
        self.info_['metric'] = args.metric
        self.info_['model'] = args.model
        if args.c_columns is None or args.d_columns is None:
            # Detect if a feature column is continuous or discrete
            feature_type_recognition = Feature_type_recognition()
            feature_type = feature_type_recognition.fit(input_data.drop(columns=self.info_['target']))
            args.d_columns = get_key_from_dict(feature_type, 'cat')
            args.c_columns = get_key_from_dict(feature_type, 'num')
        self.info_['c_columns'] = args.c_columns
        self.info_['d_columns'] = args.d_columns

        for col in input_data.columns:
            col_type = input_data[col].dtype
            if col_type != 'object':
                input_data[col].fillna(0, inplace=True)
            else:
                input_data[col].fillna('unknown', inplace=True)
        self.dfs_ = {}
        self.dfs_[self.info_['file_name']] = input_data

        # Split or shuffle training and test data if needed
        self.dfs_['FE_train'] = self.dfs_[self.info_['file_name']]
        self.dfs_['FE_test'] = pd.DataFrame()
        if self.split:
            self.dfs_['FE_train'], self.dfs_['FE_test'] = split_train_test(self.dfs_[self.info_['file_name']],
                                                                           self.info_['d_columns'],
                                                                           self.info_['target'],
                                                                           self.info_['mode'], self.train_size,
                                                                           self.seed, self.shuffle)
            self.dfs_['FE_train'].reset_index(inplace=True, drop=True)
            self.dfs_['FE_test'].reset_index(inplace=True, drop=True)
        elif self.shuffle:
            self.dfs_['FE_train'] = self.dfs_['FE_train'].sample(frac=1, random_state=self.seed).reset_index(drop=True)

        feature_pipeline.Candidate_features = self.dfs_['FE_train'].copy()
        self.is_cuda, self.device = None, None
        self.set_cuda(args.cuda)
        logging.info(f'Done AutoFE initialization.')

    def set_cuda(self, cuda):
        if cuda == 'False':
            self.device = 'cpu'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda
            self.is_cuda = torch.cuda.is_available()
            self.device = torch.device('cuda:0') if self.is_cuda else torch.device('cpu')
            if self.is_cuda:
                logging.info(f"Use device: {cuda}, {self.device}, {torch.cuda.get_device_name(self.device)}")
                return
        logging.info(f"Use device: {self.device}")

    def fit_attention(self, args):
        """Fit for searching the best autofe strategy of attention method"""
        df = self.dfs_['FE_train']
        c_columns, d_columns = self.info_['c_columns'], self.info_['d_columns']
        if len(self.info_['d_columns']) == 0:
            args.combine = False
        target, mode, model, metric = self.info_['target'], self.info_['mode'], self.info_['model'], self.info_[
            'metric']

        pool = multiprocessing.Pool(processes=args.worker)

        n_features_c, n_features_d = len(self.info_['c_columns']), len(self.info_['d_columns'])
        c_ops, d_ops = get_ops(n_features_c, n_features_d)

        # Get baseline score of 5-fold cross validation
        score_b, scores_b = self._get_cv_baseline(df, args, mode, model, metric)
        logging.info(f'score_b={score_b}, scores_b={scores_b}')

        if self.split:
            score_test_baseline = get_test_score(self.dfs_['FE_train'].drop(columns=[target]),
                                                 self.dfs_['FE_test'].drop(columns=[target]),
                                                 self.dfs_['FE_train'][target], self.dfs_['FE_test'][target],
                                                 args, mode, model, metric)
            logging.info(f'Baseline score on test={score_test_baseline}')

        # Get encoded (normalized) data as init state if needed
        if args.preprocess:
            df_d_labelencode, df_c_encode, df_d_encode, df_t, df_t_norm = features_process(df, mode, c_columns,
                                                                                           d_columns, target)
            x_d_onehot = label_encode_to_onehot(df_d_labelencode.values)
        else:
            df_c_encode, df_d_encode = df.loc[:, c_columns + [target]], df.loc[:, d_columns + [target]]
            x_d_onehot, df_d_labelencode = df.loc[:, d_columns], df.loc[:, d_columns]
            df_t, df_t_norm = df.loc[:, target], df.loc[:, target]

            # Searching autofe strategy
        data_nums = self.dfs_['FE_train'].shape[0]
        operations_c = len(c_ops)
        operations_d = len(d_ops)
        d_model = args.d_model
        d_k = args.d_k
        d_v = args.d_v
        d_ff = args.d_ff
        n_heads = args.n_heads
        self.ppo = PPO(args, data_nums, operations_c, operations_d, d_model, d_k, d_v, d_ff, n_heads, self.device)
        pipline_args_train = {'dataframe': self.dfs_['FE_train'],
                              'continuous_columns': self.info_['c_columns'],
                              'discrete_columns': self.info_['d_columns'],
                              'label_name': self.info_['target'],
                              'mode': self.info_['mode'],
                              'isvalid': False,
                              'memory': None}

        # Samples used to record the top5 reward of the search process
        self.workers_top5 = []
        if args.combine:
            ori_nums = df_c_encode.shape[1] - 1 + df_d_labelencode.shape[1] - 1
        else:
            ori_nums = df_c_encode.shape[1] - 1

        # Get the data with constructed features by action plan

        # Train a model to validate constructed features by 5-fold

        # Calculate reward

        init_workers_c = []
        init_workers_d = []
        worker_c = Worker(args)
        worker_d = Worker(args)
        init_state_c = torch.from_numpy(df_c_encode.values).float().transpose(0, 1)
        init_state_d = torch.from_numpy(df_d_encode.values).float().transpose(0, 1)
        worker_c.states = [init_state_c]
        worker_d.states = [init_state_d]
        worker_c.actions, worker_d.actions, worker_c.steps, worker_d.steps = [], [], [], []
        worker_c.log_probs, worker_d.log_probs, worker_c.dones, worker_d.dones = [], [], [], []
        worker_c.features, worker_d.features, worker_c.ff, worker_d.ff = [], [], [], []
        dones = [False for i in range(args.steps_num)]
        dones[-1] = True
        worker_c.dones, worker_d.dones = dones, dones
        init_pipline_list = []
        pipline_ff_c = Pipeline(pipline_args_train)
        for i in range(args.episodes):
            init_workers_c.append(copy.deepcopy(worker_c))
            init_workers_d.append(copy.deepcopy(worker_d))
            init_pipline_list.append(copy.deepcopy(pipline_ff_c))

        for epoch in range(args.epochs):
            workers_c = []
            workers_d = []

            logging.debug(f'Start Sampling......')
            # Parallel sampling or not
            if args.worker == 0 or args.worker == 1:
                for i in range(args.episodes):
                    # Get feature engineer action plans
                    w_c, w_d = sample(args, self.ppo, pipline_args_train, df_c_encode, df_d_encode, df_t_norm, c_ops,
                                      d_ops,
                                      epoch, i, self.device)
                    workers_c.append(w_c)
                    workers_d.append(w_d)
            else:

                workers_c = copy.deepcopy(init_workers_c)
                workers_d = copy.deepcopy(init_workers_d)
                pipline_list = copy.deepcopy(init_pipline_list)

                for step in range(args.steps_num):
                    logging.debug(f'Start step {step}..')
                    p_lst = []
                    for i in range(args.episodes):
                        if i < args.episodes // 2:
                            sample_rule = True
                        else:
                            sample_rule = False
                        if df_c_encode.shape[0] > 1:
                            actions, log_probs, m1_output, m2_output, m3_output, action_softmax = self.ppo.choose_action_c(
                                workers_c[i].states[-1].to(self.device),
                                step, epoch, c_ops, sample_rule)
                            workers_c[i].actions.append(actions)
                            workers_c[i].log_probs.append(log_probs)

                            workers_c[i].m1.append(m1_output.detach().cpu())
                            workers_c[i].m2.append(m2_output.detach().cpu())
                            workers_c[i].m3.append(m3_output.detach().cpu())
                            workers_c[i].action_softmax.append(action_softmax.detach().cpu())
                        if args.combine:
                            actions, log_probs, m1_output, m2_output, m3_output, action_softmax = self.ppo.choose_action_d(
                                workers_d[i].states[-1].to(self.device), step,
                                epoch, c_ops, sample_rule)
                            workers_d[i].actions.append(actions)
                            workers_d[i].log_probs.append(log_probs)

                            workers_c[i].m1.append(m1_output.detach().cpu())
                            workers_c[i].m2.append(m2_output.detach().cpu())
                            workers_c[i].m3.append(m3_output.detach().cpu())
                            workers_c[i].action_softmax.append(action_softmax.detach().cpu())

                    logging.debug(f'Start apply_actions..')
                    for i in range(args.episodes):
                        res = pool.apply_async(apply_actions,
                                               (
                                                   args, pipline_list[i], df_c_encode, df_d_encode, df_t_norm, c_ops,
                                                   d_ops,
                                                   epoch, i, self.device, step, workers_c[i], workers_d[i]))
                        p_lst.append(res)
                    for i, p in enumerate(p_lst):
                        # ret = p
                        ret = p.get()
                        workers_c[i] = ret[0]
                        workers_d[i] = ret[1]
                        pipline_list[i] = ret[2]
                        workers_c[i].steps.append(step)
                        workers_d[i].steps.append(step)
                for i in range(args.episodes):
                    if df_c_encode.shape[0] > 1:
                        workers_c[i].states = workers_c[i].states[0:-1]
                    if args.combine:
                        workers_d[i].states = workers_d[i].states[0:-1]
            logging.debug(f'End sample ')

            # Validate the performance of seached action plans
            if args.worker == 0 or args.worker == 1:
                for num, worker_c in enumerate(workers_c):
                    worker_d = workers_d[num]
                    w_c, w_d = multiprocess_reward(args, worker_c, worker_d, c_columns, d_columns, scores_b, mode,
                                                   model, metric, x_d_onehot, df_t.values, df_d_labelencode)
                    workers_c[num] = w_c
                    workers_d[num] = w_d
            else:
                p_lst = []
                for num, worker_c in enumerate(workers_c):
                    worker_d = workers_d[num]
                    workers_c[num] = None
                    workers_d[num] = None
                    res = pool.apply_async(multiprocess_reward, (
                        args, worker_c, worker_d, c_columns, d_columns, scores_b, mode, model, metric, x_d_onehot,
                        df_t.values, df_d_labelencode))
                    p_lst.append(res)
                workers_c = []
                workers_d = []
                for p in p_lst:
                    ret = p.get()
                    workers_c.append(ret[0])
                    workers_d.append(ret[1])

            for i, worker_c in enumerate(workers_c):
                worker_d = workers_d[i]
                new_nums = worker_c.fe_nums[-1]

                logging.info(
                    f"worker{i + 1} ,results:{worker_c.accs},cv:{worker_c.cvs[-1]},"
                    f"feature_nums:{new_nums / ori_nums, new_nums, ori_nums},repeat_nums:{worker_c.repeat_fe_nums},ff_c:{worker_c.ff},ff_d:{worker_d.ff}")
                for step in range(args.steps_num):
                    worker = Worker(args)
                    worker.accs = worker_c.accs[step]
                    worker.fe_nums = worker_c.fe_nums[step]
                    worker.scores = worker_c.scores[step]
                    worker.repeat_fe_nums = worker_c.repeat_fe_nums
                    worker.features = [worker_c.features[0:step + 1]] + [worker_d.features[0:step + 1]]
                    worker.ff = [worker_c.ff[0:step + 1]] + [worker_d.ff[0:step + 1]]
                    self.workers_top5.append(worker)

            baseline = np.mean([worker.accs for worker in workers_c], axis=0)
            logging.info(f"epoch:{epoch},baseline:{baseline},score_b:{score_b},scores_b:{scores_b}")

            self.workers_top5.sort(key=lambda worker: worker.scores.mean(), reverse=True)
            self.workers_top5 = self.workers_top5[0:5]
            for i in range(5):
                new_nums = self.workers_top5[i].fe_nums
                if self.split:
                    self.workers_top5[i] = test_one_worker(args, self.workers_top5[i], c_columns, d_columns, target,
                                                           mode,
                                                           model, metric, self.dfs_['FE_train'], self.dfs_['FE_test'])
                    logging.info(
                        f"top_{i + 1}:score:{self.workers_top5[i].scores.mean()},test_score:{self.workers_top5[i].scores_test[0]},feature_nums:{new_nums / ori_nums, new_nums, ori_nums}, repeat_nums:{self.workers_top5[i].repeat_fe_nums},{self.workers_top5[i].ff}")
                else:
                    logging.info(
                        f"top_{i + 1}:score:{self.workers_top5[i].scores.mean()},feature_nums:{new_nums / ori_nums, new_nums, ori_nums}, repeat_nums:{self.workers_top5[i].repeat_fe_nums},{self.workers_top5[i].ff}")

            if df_c_encode.shape[0]:
                self.ppo.update_c(workers_c)
            if args.combine:
                self.ppo.update_d(workers_d)

        # df_train, df_test = self.transform(self.dfs_['FE_train'].copy(), self.dfs_['FE_test'].copy(), args,
        #                                    self.workers_top5[0].ff[0], self.workers_top5[0].ff[1])
        # y_train, y_test = self.dfs_['FE_train'][self.info_["target"]], self.dfs_['FE_test'][self.info_["target"]]
        #
        # logging.info(f"{df_train.shape}, {df_test.shape}, {len(y_train)}, {len(y_test)}")
        # model_test = model_fuctions[f"{model}_{mode}"](-1)
        # model_test.fit(df_train, y_train)
        # if mode == 'classify':
        #     from metrics.metric_evaluate import f1_metric
        #     score = f1_metric(model_test, df_test, y_test, y_train)
        # else:
        #     from metrics.metric_evaluate import rae_score
        #     score = rae_score(model_test, df_test, y_test)
        # logging.info(f"The test score is {score}")

        pool.close()
        pool.join()

    def transform(self, df_train, df_test, args, actions_c, actions_d):
        """Apply the best autofe strategy to input data and return the transformed data"""
        c_columns = self.info_["c_columns"]
        d_columns = self.info_["d_columns"]
        target = self.info_["target"]

        memory = Memory()
        pipline_args_train = {'dataframe': df_train,
                              'continuous_columns': c_columns,
                              'discrete_columns': d_columns,
                              'label_name': target,
                              'mode': self.info_['mode'],
                              'isvalid': False,
                              'memory': memory}
        pipline_train = Pipeline(pipline_args_train)
        pipline_args_test = {'dataframe': df_test,
                             'continuous_columns': c_columns,
                             'discrete_columns': d_columns,
                             'label_name': target,
                             'mode': self.info_['mode'],
                             'isvalid': True,
                             'memory': memory}
        pipline_test = Pipeline(pipline_args_test)

        for step in range(len(actions_c)):
            action_c = actions_c[step]
            _, x_c = pipline_train.process_continuous(action_c)
            x_c = x_c.astype(np.float32).apply(np.nan_to_num)
            x_c, mask_c = remove_duplication(x_c)
            var_selector = VarianceThreshold()
            x_c = var_selector.fit_transform(x_c)
            _, x_test_c = pipline_test.process_continuous(action_c)
            x_test_c = x_test_c.astype(np.float32).apply(np.nan_to_num)
            x_test_c = x_test_c.values[:, mask_c]
            x_test_c = var_selector.transform(x_test_c)
            x = x_c
            x_test = x_test_c

            if args.combine:
                action_d = actions_d[step]
                _, x_d = pipline_train.process_discrete(action_d)
                x_d = x_d.astype(np.float32).apply(np.nan_to_num)
                x_d, mask_d = remove_duplication(x_d)
                var_selector = VarianceThreshold()
                x_d = var_selector.fit_transform(x_d)
                _, x_test_d = pipline_test.process_discrete(action_d)
                x_test_d = x_test_d.astype(np.float32).apply(np.nan_to_num)
                x_test_d = x_test_d.values[:, mask_d]
                x_test_d = var_selector.transform(x_test_d)

                x = np.concatenate((x_c, x_d), axis=1)
                x_test = np.concatenate((x_test_c, x_test_d), axis=1)

        return x, x_test

    def save(self, file):
        """Save AutoFE object"""
        pickle.dump(self, file)

    def _get_cv_baseline(self, df: pd.DataFrame, args, mode, model, metric):
        c_columns = self.info_["c_columns"]
        d_columns = self.info_["d_columns"]
        target = self.info_["target"]
        if args.worker == 0 or args.worker == 1:
            n_jobs = -1
        else:
            n_jobs = 1
        model = model_fuctions[f"{model}_{mode}"](n_jobs)

        encode = False
        logging.info(f'Start getting CV baseline...')

        if not args.shuffle: args.seed = None
        if args.cv == 1:
            if mode == "classify":
                my_cv = StratifiedShuffleSplit(n_splits=args.cv, train_size=args.cv_train_size,
                                               random_state=args.seed)
            else:
                my_cv = ShuffleSplit(n_splits=args.cv, train_size=args.cv_train_size, random_state=args.seed)
        else:
            if mode == "classify":
                my_cv = StratifiedKFold(n_splits=args.cv, shuffle=args.shuffle, random_state=args.seed)
            else:
                my_cv = KFold(n_splits=args.cv, shuffle=args.shuffle, random_state=args.seed)
        scores = []
        if args.preprocess:

            logging.info(f'CV split end. Start CV scoring...')
            for index_train, index_test in my_cv.split(df, df[target].values):
                df_train = df.iloc[index_train]
                df_test = df.iloc[index_test]
                memory = Memory()
                pipline_args_train = {'dataframe': df_train,
                                      'continuous_columns': c_columns,
                                      'discrete_columns': d_columns,
                                      'label_name': target,
                                      'mode': mode,
                                      'isvalid': False,
                                      'memory': memory}
                pipline_args_test = {'dataframe': df_test,
                                     'continuous_columns': c_columns,
                                     'discrete_columns': d_columns,
                                     'label_name': target,
                                     'mode': mode,
                                     'isvalid': True,
                                     'memory': memory}
                pipline_train = Pipeline(pipline_args_train)
                # print(memory.normalization_info)
                pipline_test = Pipeline(pipline_args_test)
                # print(pipline_test.memory.normalization_info)
                c_fes_train, d_fes_train, y_train = pipline_train.ori_c_columns_norm, pipline_train.discrete_reward, pipline_train.label
                c_fes_test, d_fes_test, y_test = pipline_test.ori_c_columns_norm, pipline_test.discrete_reward, pipline_test.label
                # onehot
                if encode:
                    logging.info(f'encoding')
                    d_fes_test = label_encode_to_onehot(d_fes_test, d_fes_train)
                    d_fes_train = label_encode_to_onehot(d_fes_train)
                else:
                    logging.info(f'no encoding')
                if isinstance(d_fes_train, np.ndarray):
                    x_train = np.hstack((c_fes_train, d_fes_train))
                    x_test = np.hstack((c_fes_test, d_fes_test))
                else:
                    x_train = c_fes_train
                    x_test = c_fes_test

                # df_d = pd.DataFrame(d_fes_train, columns=d_columns)
                # df[df_d.columns] = df_d
                # df.to_csv('df.csv', index=False)
                logging.info(f'Start training model...')
                model.fit(x_train, y_train)
                score = metric_fuctions[metric](model, x_test, y_test, y_train)
                scores.append(round(score, 4))
        else:
            # X = df[c_columns + d_columns]
            X = df.drop(columns=[target])
            y = df[target]
            scores = []

            if mode == "classify":
                if metric == 'f1':
                    scores = cross_val_score(model, X, y, scoring='f1_micro', cv=my_cv, error_score="raise")
                elif metric == 'auc':
                    auc_scorer = make_scorer(roc_auc_score, needs_proba=True, average="macro", multi_class="ovo")
                    scores = cross_val_score(model, X, y, scoring=auc_scorer, cv=my_cv, error_score="raise")
            else:
                if metric == 'mae':
                    scores = cross_val_score(model, X, y, cv=my_cv, scoring='neg_mean_absolute_error')
                elif metric == 'mse':
                    scores = cross_val_score(model, X, y, cv=my_cv, scoring='neg_mean_squared_error')
                elif metric == 'r2':
                    scores = cross_val_score(model, X, y, cv=my_cv, scoring='r2')
                elif metric == 'rae':
                    scores = cross_val_score(model, X, y, cv=my_cv, scoring=rae_score)
        return np.array(scores).mean(), scores


def test_one_worker(args, worker, c_columns, d_columns, target, mode, model, metric, df_train, df_test):
    if worker.scores_test is not None:
        return worker
    scores = []
    new_fe_nums = []

    memory = Memory()
    pipline_args_train = {'dataframe': df_train,
                          'continuous_columns': c_columns,
                          'discrete_columns': d_columns,
                          'label_name': target,
                          'mode': mode,
                          'isvalid': False,
                          'memory': memory}
    pipline_train = Pipeline(pipline_args_train)
    pipline_args_test = {'dataframe': df_test,
                         'continuous_columns': c_columns,
                         'discrete_columns': d_columns,
                         'label_name': target,
                         'mode': mode,
                         'isvalid': True,
                         'memory': memory}
    pipline_test = Pipeline(pipline_args_test)
    if args.combine:
        for step in range(len(worker.ff[0])):
            action_c = worker.ff[0][step]
            action_d = worker.ff[1][step]
            x_c = worker.features[0][step]
            x_d = worker.features[1][step]
            x_c, mask_c = remove_duplication(x_c)
            x_d, mask_d = remove_duplication(x_d)
            var_selector = VarianceThreshold()
            x_c = var_selector.fit_transform(x_c)
            _, x_test_c = pipline_test.process_continuous(action_c)
            x_test_c = x_test_c.astype(np.float32).apply(np.nan_to_num)
            x_test_c = x_test_c.values[:, mask_c]
            x_test_c = var_selector.transform(x_test_c)
            var_selector = VarianceThreshold()
            x_d = var_selector.fit_transform(x_d)
            _, x_test_d = pipline_test.process_discrete(action_d)
            x_test_d = x_test_d.astype(np.float32).apply(np.nan_to_num)
            x_test_d = x_test_d.values[:, mask_d]
            x_test_d = var_selector.transform(x_test_d)
            new_fe_num = x_c.shape[1] + x_d.shape[1]
            new_fe_nums.append(new_fe_num)
            if len(c_columns):
                x = np.concatenate((x_c, x_d), axis=1)
                x_test = np.concatenate((x_test_c, x_test_d), axis=1)
            else:
                x = worker.features[1]
                x_test = x_test_c

        score_test = get_test_score(x, x_test, df_train[target], df_test[target], args, mode, model, metric)
        scores.append(score_test)
    else:
        for step in range(len(worker.ff[0])):
            action_c = worker.ff[0][step]
            x_c = worker.features[0][step]
            x_c, mask = remove_duplication(x_c)
            var_selector = VarianceThreshold()
            x_c = var_selector.fit_transform(x_c)
            _, x_test = pipline_test.process_continuous(action_c)
            x_test = x_test.astype(np.float32).apply(np.nan_to_num)
            x_test = x_test.values[:, mask]
            x_test = var_selector.transform(x_test)
            new_fe_num = x_c.shape[1]
            new_fe_nums.append(new_fe_num)
            x = x_c
        score_test = get_test_score(x, x_test, df_train[target], df_test[target], args, mode, model, metric)
        scores.append(score_test)

    worker.scores_test = scores
    worker.features = None

    return worker
