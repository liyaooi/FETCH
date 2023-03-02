import numpy as np
import torch


def parse_actions(actions, ops, n_features, continuous=True):
    if continuous:
        add = []
        subtract = []
        multiply = []
        divide = []
        value_convert = {}
        for index, action in enumerate(actions):
            if 0 <= action < n_features:
                # add
                add.append([index, action])
            elif n_features <= action < (2 * n_features):
                # subtract
                subtract.append([index, action - n_features])

            elif (2 * n_features) <= action < (3 * n_features):
                # multiply
                multiply.append([index, action - n_features * 2])
            elif (3 * n_features) <= action < (4 * n_features):
                # divide
                divide.append([index, action - n_features * 3])
            else:
                value_convert[index] = ops[action]
        """值转换先处理索引大，不然delete,terminate会影响后续的操作"""
        value_convert = dict(sorted(value_convert.items(), key=lambda e: e[0], reverse=True))
        action_all = [{"add": add}, {"subtract": subtract}, {"multiply": multiply}, {"divide": divide},
                      {"value_convert": value_convert}]
    else:
        combine = []
        delete = {}
        none = {}
        for index, action in enumerate(actions):
            if 0 <= action < n_features:
                if index < n_features:
                    if index != action:
                        combine.append([index, action])
                else:
                    combine.append([index, action])
            elif ops[action] == "delete":
                delete[index] = "delete"
            elif ops[action] == "None":
                none[index] = "None"
        action_all = [{"combine": combine}, {"delete": delete}, {"none": none}]

    return action_all

