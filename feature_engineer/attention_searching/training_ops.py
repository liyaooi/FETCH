import logging

import numpy as np
import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold

from feature_engineer.attention_searching.worker import Worker
from feature_engineer.fe_parsers import parse_actions
from metrics.metric_evaluate import rae_score
from models import model_fuctions
from process_data import Pipeline
from process_data.feature_process import remove_duplication, label_encode_to_onehot


def sample(args, ppo, pipline_args_train, df_c_encode, df_d_encode, df_t_norm, c_ops, d_ops, epoch, i, device):
    logging.debug(f'Get in pipline_ff_c for episode {i}...')
    pipline_ff_c = Pipeline(pipline_args_train)
    logging.debug(f'End pipline_ff_c')
    worker_c = Worker(args)
    states_c = []
    actions_c = []
    log_probs_c = []
    features_c = []
    ff_c = []
    steps = []

    worker_d = Worker(args)
    states_d = []
    actions_d = []
    log_probs_d = []
    features_d = []
    ff_d = []

    n_features_c = df_c_encode.shape[1] - 1
    n_features_d = df_d_encode.shape[1] - 1
    init_state_c = torch.from_numpy(df_c_encode.values).float().transpose(0, 1).to(device)
    init_state_d = torch.from_numpy(df_d_encode.values).float().transpose(0, 1).to(device)

    steps_num = args.steps_num
    if i < args.episodes // 2:
        sample_rule = True
    else:
        sample_rule = False
    logging.debug(f'Start sample episode {i}...')
    for step in range(steps_num):
        steps.append(step)

        if df_c_encode.shape[0] > 1:
            state_c = init_state_c

            logging.debug(f'Start choose_action_c for step {step}, state_c: {state_c.shape}')
            actions, log_probs, m1_output, m2_output, m3_output, action_softmax = ppo.choose_action_c(state_c, step, epoch, c_ops,
                                                                                      sample_rule)
            logging.debug(f'Start parse_actions...')
            fe_c = parse_actions(actions, c_ops, n_features_c, continuous=True)
            ff_c.append(fe_c)

            logging.debug(f'Start process_continuous...')
            # x_c_encode, x_c_reward = pipline_ff_c.process_continuous(fe_c)
            x_c_encode, x_c_combine = pipline_ff_c.process_continuous(fe_c)

            logging.debug(f'Start astype, x_c_encode: {x_c_encode.shape}')
            # Process np.nan and np.inf in np.float32
            x_c_encode = x_c_encode.astype(np.float32).apply(np.nan_to_num)
            x_c_combine = x_c_combine.astype(np.float32).apply(np.nan_to_num)
            features_c.append(x_c_combine)
            logging.debug(f'Start hstack...')
            if x_c_encode.shape[0]:
                x_encode_c = np.hstack((x_c_encode, df_t_norm.values.reshape(-1, 1)))
                # x_encode_c = np.hstack((x_c_combine, df_t_norm.values.reshape(-1, 1)))
                x_encode_c = torch.from_numpy(x_encode_c).float().transpose(0, 1).to(device)
                # x_encode_c = torch.from_numpy(x_c_encode.values).float().transpose(0, 1)
                init_state_c = x_encode_c
                states_c.append(state_c.cpu())
                # states_c.append(state_c)
                actions_c.append(actions)
                log_probs_c.append(log_probs)
            else:
                states_c.append(state_c.cpu())
                # states_c.append(state_c)
                actions_c.append(actions)
                log_probs_c.append(log_probs)
            logging.debug(f'End append, state_c: {state_c.shape}')
        if args.combine:
            state_d = init_state_d

            logging.debug(f'Start choose_action_c for step {step}, state_d: {state_d.shape}')
            actions, log_probs, m1_output, m2_output, m3_output, action_softmax = ppo.choose_action_d(state_d, step, epoch, c_ops, sample_rule)
            logging.debug(f'Start parse_actions...')
            fe_d = parse_actions(actions, d_ops, n_features_d, continuous=False)
            ff_d.append(fe_d)
            logging.debug(f'Start process_discrete...')
            x_d_norm, x_d = pipline_ff_c.process_discrete(fe_d)
            # for ff_action in fe_d:
            #     logging.debug(f'Start process_discrete 2...')
            #     x_d_norm, x_d = pipline_ff_c.process_discrete(ff_action)

            # Process np.nan and np.inf in np.float32
            logging.debug(f'Start astype, x_d_norm: {x_d_norm.shape}')
            x_d_norm = x_d_norm.astype(np.float32).apply(np.nan_to_num)
            x_d = x_d.astype(np.float32).apply(np.nan_to_num)
            # x_d_norm = np.nan_to_num(x_d_norm.astype(np.float32))
            # x_d = np.nan_to_num(x_d.astype(np.float32))
            features_d.append(x_d)
            logging.debug(f'Start hstack...')
            try:
                x_encode_d = np.hstack((x_d_norm, df_t_norm.values.reshape(-1, 1)))
            except:
                breakpoint()
            x_encode_d = torch.from_numpy(x_encode_d).float().transpose(0, 1).to(device)
            init_state_d = x_encode_d
            states_d.append(state_d.cpu())
            actions_d.append(actions)
            log_probs_d.append(log_probs)
            logging.debug(f'End append, state_d: {state_d.shape}')
    dones = [False for i in range(steps_num)]
    dones[-1] = True

    worker_c.steps = steps
    worker_c.states = states_c
    worker_c.actions = actions_c
    worker_c.log_probs = log_probs_c
    worker_c.dones = dones
    worker_c.features = features_c
    worker_c.ff = ff_c

    worker_d.steps = steps
    worker_d.states = states_d
    worker_d.actions = actions_d
    worker_d.log_probs = log_probs_d
    worker_d.dones = dones
    worker_d.features = features_d
    worker_d.ff = ff_d
    return worker_c, worker_d


def apply_actions(args, pipline_ff_c, df_c_encode, df_d_encode, df_t_norm, c_ops, d_ops, epoch, i, device, cur_step, worker_c, worker_d):
    n_features_c = df_c_encode.shape[1] - 1
    n_features_d = df_d_encode.shape[1] - 1

    if df_c_encode.shape[0] > 1:
        state_c = worker_c.states[-1]

        fe_c = parse_actions(worker_c.actions[-1], c_ops, n_features_c, continuous=True)
        worker_c.ff.append(fe_c)

        x_c_encode, x_c_combine = pipline_ff_c.process_continuous(fe_c)

        # Process np.nan and np.inf in np.float32
        x_c_encode = x_c_encode.astype(np.float32).apply(np.nan_to_num)
        x_c_combine = x_c_combine.astype(np.float32).apply(np.nan_to_num)

        worker_c.features.append(x_c_combine)
        if x_c_encode.shape[0]:
            x_encode_c = np.hstack((x_c_encode, df_t_norm.values.reshape(-1, 1)))
            x_encode_c = torch.from_numpy(x_encode_c).float().transpose(0, 1)
            state_c = x_encode_c
            worker_c.states.append(state_c)
        else:
            worker_c.states.append(state_c)
    if args.combine:
        state_d = worker_d.states[-1]
        fe_d = parse_actions(worker_d.actions[-1], d_ops, n_features_d, continuous=False)
        worker_d.ff.append(fe_d)
        x_d_norm, x_d = pipline_ff_c.process_discrete(fe_d)
        # for ff_action in fe_d:
        #     x_d_norm, x_d = pipline_ff_c.process_discrete(ff_action)

        # Process np.nan and np.inf in np.float32
        x_d_norm = x_d_norm.astype(np.float32).apply(np.nan_to_num)
        x_d = x_d.astype(np.float32).apply(np.nan_to_num)
        worker_d.features.append(x_d)

        x_encode_d = np.hstack((x_d_norm, df_t_norm.values.reshape(-1, 1)))
        x_encode_d = torch.from_numpy(x_encode_d).float().transpose(0, 1)
        state_d = x_encode_d
        worker_d.states.append(state_d)
    return worker_c, worker_d, pipline_ff_c


def multiprocess_reward(args, worker_c, worker_d, c_columns, d_columns, scores_b, mode, model, metric, x_d_onehot, y,
                        df_d_labelencode):
    accs = []
    cvs = []
    scores = []
    new_fe_nums = []
    repeat_fe_nums = []
    repeat_ratio = cal_repeat_actions(len(c_columns), worker_c.ff)
    repeat_fe_nums.append(repeat_ratio)
    if args.combine:
        for step in range(args.steps_num):
            x_c = worker_c.features[step]
            x_d = worker_d.features[step]
            x_c, _ = remove_duplication(x_c)
            x_d, _ = remove_duplication(x_d)
            var_selector = VarianceThreshold()
            x_c = var_selector.fit_transform(x_c)
            var_selector = VarianceThreshold()
            x_d = var_selector.fit_transform(x_d)
            new_fe_num = x_c.shape[1] + x_d.shape[1]
            new_fe_nums.append(new_fe_num)
            if len(c_columns):
                if model == "lr":
                    d_onehot = label_encode_to_onehot(x_d)
                    var_selector = VarianceThreshold()
                    d_onehot = var_selector.fit_transform(d_onehot)
                    d_onehot, _ = remove_duplication(d_onehot)
                    x = np.concatenate((x_c, d_onehot), axis=1)
                else:
                    x = np.concatenate((x_c, x_d), axis=1)
            else:
                x = worker_d.features[step]
            acc, cv, score = get_reward(x, y, args, scores_b, mode, model, metric, step, repeat_ratio)
            accs.append(acc)
            cvs.append(cv)
            scores.append(score)
    else:
        for step in range(args.steps_num):
            x_c = worker_c.features[step]
            x_c, _ = remove_duplication(x_c)
            var_selector = VarianceThreshold()
            x_c = var_selector.fit_transform(x_c)
            new_fe_num = x_c.shape[1]
            new_fe_nums.append(new_fe_num)
            if len(d_columns):
                x = np.concatenate((x_c, x_d_onehot), axis=1)
            else:
                x = x_c
            acc, cv, score = get_reward(x, y, args, scores_b, mode, model, metric, step, repeat_ratio)
            accs.append(acc)
            cvs.append(cv)
            scores.append(score)
    worker_c.fe_nums = new_fe_nums
    worker_c.accs = accs
    worker_c.cvs = cvs
    worker_c.scores = scores
    worker_c.features = worker_c.features
    worker_c.repeat_fe_nums = repeat_fe_nums

    worker_d.fe_nums = new_fe_nums
    worker_d.accs = accs
    worker_d.cvs = cvs
    worker_d.scores = scores
    worker_d.features = worker_d.features
    worker_d.repeat_fe_nums = repeat_fe_nums
    return worker_c, worker_d


def cal_repeat_actions(n_features_c, ff):
    add, subtract, multiply, divide, value_convert = [], [], [], [], []
    for i in range(len(ff)):
        for dic in ff[i]:
            if list(dic.keys())[0] == "value_convert":
                eval(list(dic.keys())[0]).extend(list(dic.values())[0].items())
            else:
                eval(list(dic.keys())[0]).extend(tuple(x) for x in list(dic.values())[0])
    total_actions = len(add) + len(subtract) + len(multiply) + len(divide) + len(value_convert)
    add = set(add)
    subtract = set(subtract)
    multiply = set(multiply)
    divide = set(divide)
    value_convert = set(value_convert)
    effective_actions = len(add) + len(subtract) + len(multiply) + len(divide) + len(value_convert)
    repeat_ratio = (total_actions - effective_actions) / (4 * n_features_c)
    return repeat_ratio


def get_reward(x, y, args, scores_b, mode, model, metric, step, repeat_ratio):
    if args.worker == 0 or args.worker == 1:
        n_jobs = -1
    else:
        n_jobs = 1
    model = model_fuctions[f"{model}_{mode}"](n_jobs)

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

    if mode == "classify":
        if metric == 'f1':
            scores = cross_val_score(model, x, y, scoring='f1_micro', cv=my_cv, error_score="raise")
        elif metric == 'auc':
            auc_scorer = make_scorer(roc_auc_score, needs_proba=True, average="macro", multi_class="ovo")
            scores = cross_val_score(model, x, y, scoring=auc_scorer, cv=my_cv, error_score="raise")
    else:
        if metric == 'mae':
            scores = cross_val_score(model, x, y, cv=my_cv, scoring='neg_mean_absolute_error')
        elif metric == 'mse':
            scores = cross_val_score(model, x, y, cv=my_cv, scoring='neg_mean_squared_error')
        elif metric == 'r2':
            scores = cross_val_score(model, x, y, cv=my_cv, scoring='r2')
        elif metric == 'rae':
            scores = cross_val_score(model, x, y, cv=my_cv, scoring=rae_score)

    values = np.array(scores) - np.array(scores_b)
    mask = values < 0
    negative = values[mask]
    negative_sum = negative.sum()
    reward = np.array(scores).mean() + negative_sum
    if step == (args.steps_num - 1):
        reward = reward - repeat_ratio
    return round(reward, 4), values, scores
