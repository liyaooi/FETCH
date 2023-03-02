import numpy as np
from .memory_no_encode import Memory
from .utils_memory import ff, sort_count, calculate_chi2, categories_to_int
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

def binning_with_tree(ori_fe:np.array,
                      label:np.array,
                      col_index:int=None,
                      memory: Memory = None,
                      isvalid: bool = False):
    if isvalid:
        boundry = memory.binning_continuous_info[col_index]
        assert col_index in memory.binning_continuous_info, 'Col index not in boundry memory,check your code!'
    else:
        boundry = []
        clf = DecisionTreeClassifier(criterion='entropy',  # “信息熵”最小化准则划分
                                     max_leaf_nodes=6,  # 最大叶子节点数
                                     min_samples_leaf=0.05)  # 叶子节点样本数量最小占比
        fe = ori_fe.reshape(-1,1)
        clf.fit(fe, label.astype("int"))  # 训练决策树

        n_nodes = clf.tree_.node_count  # 决策树的节点数
        children_left = clf.tree_.children_left  # node_count大小的数组，children_left[i]表示第i个节点的左子节点
        children_right = clf.tree_.children_right  # node_count大小的数组，children_right[i]表示第i个节点的右子节点
        threshold = clf.tree_.threshold  # node_count大小的数组，threshold[i]表示第i个节点划分数据集的阈值
        for i in range(n_nodes):
            if children_left[i] != children_right[i]:
                boundry.append(threshold[i])
        boundry.sort()

        if memory is not None:
            try:
                if col_index not in memory.binning_continuous_info:
                    memory.binning_continuous_info[col_index] = boundry
            except:
                raise ValueError('Invalid col index occurs')

    if len(boundry):
        new_fe = np.array([ff(x, boundry) for x in ori_fe])
    else:
        new_fe = ori_fe
    return new_fe

# 单一特征操作,返回的特征全部转化为shape = (n,1)的数组，方便直接concat
def sqrt(col):
    '''
    :type col: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    # 需要保证输入的特征全部是数值型特征
    # 求三次方根
    try:
        sqrt_col = np.sqrt(np.abs(col))
        return sqrt_col.reshape(-1, 1)
    except:
        raise ValueError('Value type error,check feature type')


def power3(col):
    col = np.array(col)
    new_col = np.power(col, 3)
    return new_col.reshape(-1, 1)


def sigmoid(col):
    col = np.array(col)
    new_col = 1 / (1 + np.exp(-col))
    return new_col.reshape(-1, 1)


def tanh(col):
    col = np.array(col)
    new_col = (np.exp(col) - np.exp(-col)) / (np.exp(col) + np.exp(-col))
    print(new_col)
    exit()
    return new_col.reshape(-1, 1)


def inverse(col, memory=None):
    '''
    :type col: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    try:
        col = np.array(col)
        # if np.any(col == 0):
        #     return None
        new_col = np.array([1 / x if x != 0 else x for x in col])
        return new_col.reshape(-1, 1)
    except:
        raise ValueError('Value type error,check feature type')


def square(col):
    '''
    :type col: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    col = np.array(col)
    new_col = np.square(col).reshape(-1, 1)
    return new_col.reshape(-1, 1)


def abss(col):
    col = np.array(col)
    new_col = np.abs(col)
    return new_col.reshape(-1, 1)


def log(col):
    '''
    :type col: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    # 底数为自然底数e
    try:
        log_col = np.array([np.log(abs(x)) if abs(x) > 0 else np.log(1) for x in col])
        return log_col.reshape(-1, 1)
    except:
        raise ValueError('Value type error,check feature type')


def min_max(col: list or np.ndarray, col_index: int = None, memory: Memory = None, isvalid: bool = False):
    col = np.array(col)
    if isvalid:
        min, max = memory.min_max_info[col_index]
    else:
        min = np.min(col, axis=0)
        max = np.max(col, axis=0)
    if memory is not None:
        try:
            if col_index not in memory.min_max_info:
                memory.min_max_info[col_index] = (min, max)
        except:
            raise ValueError('Invalid col index occurs')
    if min == max:
        return col
    else:
        scaled = (col - min) / (max - min)
        return scaled.reshape(-1, 1)


def normalization(col: list or np.ndarray,
                  col_index: int = None,
                  memory: Memory = None,
                  isvalid: bool = False) -> np.array:
    '''
    Parameters
    ----------
    :param col: list or np.array
    Returns
    ----------
    return:
        - col:np.array
    '''
    # 特征z-core标准化
    col = np.array(col)
    if isvalid:
        mu, sigma = memory.normalization_info[col_index]
    else:
        mu = np.mean(col, axis=0)
        sigma = np.std(col, axis=0)
    if memory is not None:
        try:
            if col_index not in memory.normalization_info:
                memory.normalization_info[col_index] = (mu, sigma)
        except:
            raise ValueError('Invalid col index occurs')
    if sigma == 0:  # while sigma is 0,return ori_col
        return col.reshape(-1, 1)
    else:
        scaled = ((col - mu) / sigma)
    return scaled.reshape(-1, 1)


# 两个数值特征的四则运算操作
def add(col1, col2):
    '''
    :type col1,col2: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    # 数值特征加法
    try:
        col1 = np.array(col1)
        col2 = np.array(col2)
        return (col1 + col2).reshape(-1, 1)
    except:
        raise ValueError('Value type error,check feature type')


def multiply(col1, col2):
    '''
    :type col1,col2: list or np.array
    :rtype: np.array,shape = (len(array),1)
    '''
    # 数值特征乘法
    try:
        col1 = np.array(col1)
        col2 = np.array(col2)
        return (col1 * col2).reshape(-1, 1)
    except:
        raise ValueError('Value type error,check feature type')


def subtract(col1, col2):
    '''
    :type col1,col2: list or np.array
    :rtype: np.array,shape = (len(array),2)
    '''
    # 数值特征减法，不指定被减数的话，生成的应该是两列特征
    try:
        col1 = np.array(col1)
        col2 = np.array(col2)
        return np.abs(col1 - col2).reshape(-1, 1)
    except:
        raise ValueError('Value type error,check feature type')


def divide(col1, col2):
    '''
    :type col1,col2: list or np.array
    :rtype: np.array,shape = (len(array),2)
    '''
    try:
        col1 = np.array(col1)
        col2 = np.array(col2)
        # if np.any(col2 == 0):
        #     return None
        col_d1 = np.array([x1 / x2 if x2 != 0 else 1 for x1, x2 in zip(col1, col2)]).reshape(-1, 1)
        return col_d1
        # return np.concatenate((col_d1, col_d2), axis=1)
    except:
        raise ValueError('Value type error,check feature type')


def convert_2_onehot(ori_fe: np.array,
                     categories_map: dict) -> np.array:
    '''convert category value type to onehot'''
    unique_values = list(np.unique(list(categories_map.values())))
    unique_values_index = {}
    for ind, v in enumerate(unique_values):
        unique_values_index[v] = ind
    k, c = len(ori_fe), len(unique_values)
    one_hot_fe = np.zeros((k, c))
    for i, v in enumerate(ori_fe):
        if v in unique_values:
            # index = unique_values.index(v)
            index = unique_values_index[v]
            one_hot_fe[i, index] = 1
        else:
            one_hot_fe[i, -1] = 1
    return one_hot_fe


def one_hot_encode(ori_fe: np.array,
                   col_index: int,
                   memory: Memory = None,
                   isvalid: bool = False) -> np.array:
    '''one hot encoder of category feature'''
    if isvalid:
        if col_index in memory.category_to_int_info:
            # discrete feature have two step to convert value
            categories_map = memory.category_to_int_info[col_index]
            if col_index in memory.binning_discrete_info:
                categories_map = memory.binning_discrete_info[col_index]
        elif col_index in memory.binning_continuous_info:
            fre_list = memory.binning_continuous_info[col_index]
            categories_map = {}
            for i in range(len(fre_list) + 1):
                categories_map[i] = i
        else:
            raise ValueError('Col index {} not exist in memory,check code'.format(col_index))
    else:
        unique_type = list(np.unique(ori_fe))
        categories_map = {}
        for v in unique_type:
            categories_map[v] = v
    one_hot_fe = convert_2_onehot(ori_fe, categories_map)
    return one_hot_fe


def reset_value(ori_fe, c, merged_values, k):
    '''将原始分类变量值重置为其他'''
    for merged_value in merged_values:
        indexs = np.argwhere(ori_fe == merged_value).reshape(-1)
        # 将原始ori_fe的低频率值今进行修改查找
        new_value = k + c  # 这样基本能保证不会跟原始值重复
        ori_fe[indexs] = new_value


def recur_merge_regression(bins, frequency_list, value_types, residual_f, ori_fe):
    # 递归合并频概率变量，这里默认变量频率已经排序
    # 针对回归问题的版本
    k = len(ori_fe)
    if bins == 1:
        merged_values = value_types
        reset_value(ori_fe, len(value_types), merged_values, k)
        return
    target_frequency = residual_f / bins
    merged_f, merged_values, ptr = 0, [], 0
    for i, f in enumerate(frequency_list):
        residual_f -= f
        ptr = i + 1
        if f < target_frequency:
            merged_f += f
            merged_values.append(value_types[i])
            if merged_f >= target_frequency:
                bins -= 1
                break
        else:
            bins -= 1
            break
    reset_value(ori_fe, len(value_types), merged_values, k)
    frequency_list, value_types = frequency_list[ptr:], value_types[ptr:]
    recur_merge_regression(bins, frequency_list, value_types, residual_f, ori_fe)


def recur_merge_classify(chi2_dict, bins, ori_fe):
    '''卡方分箱的思想是，不断将chi2值最小的两个类别合并
    直至分箱数等于目标分箱为止'''

    def merge_value_type(chi2_value_tuple, chi2_dict, c):
        chi2_1, chi2_2 = chi2_value_tuple
        if chi2_1 == chi2_2:
            index1 = list(chi2_dict.values()).index(chi2_1)
            index2 = index1 + 1
        else:
            index1 = list(chi2_dict.values()).index(chi2_1)
            index2 = list(chi2_dict.values()).index(chi2_2)
        value_type_of_chi2_1 = list(chi2_dict.keys())[index1]
        value_type_of_chi2_2 = list(chi2_dict.keys())[index2]
        new_chi2_value = chi2_1 + chi2_2
        k = len(ori_fe) + value_type_of_chi2_2 + value_type_of_chi2_1
        merged_values = [value_type_of_chi2_1, value_type_of_chi2_2]
        reset_value(ori_fe, c, merged_values, k)
        new_value_type = k + c
        chi2_dict[new_value_type] = new_chi2_value
        del chi2_dict[value_type_of_chi2_1]
        del chi2_dict[value_type_of_chi2_2]
        chi2_dict = dict(sorted(chi2_dict.items(), key=lambda x: x[1], reverse=True))
        return chi2_dict

    c = len(np.unique(ori_fe))
    while c > bins:
        chi2_value_list = np.array(list(chi2_dict.values()))
        chi2_value_tuple = (chi2_value_list[-1], chi2_value_list[-2])
        chi2_dict = merge_value_type(chi2_value_tuple, chi2_dict, c)
        c = len(list(chi2_dict.values()))


# 分类变量归并操作,处理大型分类变量
def binning_for_discrete(ori_fe: np.array,
                         bins: int,
                         mode: str,
                         label: np.array,
                         col_index: int = None,
                         memory: Memory = None,
                         isvalid: bool = False):
    """Merge discrete feature to target bins.

    Input original col must be discrete feature.

    Parameters
    ----------
    :param ori_fe: np.array
    :param bins: int
    :param mode: str, value must be 'classify' or 'regression'
    :param col_index: int
    :param memory: class Memory
    :param isvalid: bool,judge whether to get information in memory
    Returns
    ----------
    returns: tuple (new_fe, fre_list, new_fe_encode)
        - new_fe : np.array, 1D
        - fre_list : list of floats
        - new_fe_encode : np.array, 2D
    """
    # 对于回归问题，默认采用类似等频分箱的思路，逐步归并分类值
    # 对于分类问题,采用卡方分箱，按照卡方相关性的重要程度排序，逐步归并分类值

    ori_fe_copy = ori_fe.copy()
    if isvalid:
        if col_index in memory.binning_discrete_info:
            value_mapping_dict = memory.binning_discrete_info[col_index]
            mapping_values = list(value_mapping_dict.values())
            sorted_keys = sort_count(mapping_values)  # 对mapping value 按出现的频次排序
            for i, value in enumerate(ori_fe):
                if value in value_mapping_dict:
                    ori_fe[i] = value_mapping_dict[value]
                else:
                    ori_fe[i] = sorted_keys[-1]  # 容错，当映射原值不存在时，归并到最低频次的类中
        return ori_fe

    unique_value = np.unique(ori_fe)
    k = len(unique_value)
    if k <= bins:
        return np.array(ori_fe).reshape(-1, 1)
    if mode == 'regression':
        n = len(ori_fe)
        # 1.先计算每个分类变量的frequency
        frequency = dict(Counter(ori_fe))
        sorted_frequency = dict(sorted(frequency.items(), key=lambda x: x[1], reverse=True))
        for key in sorted_frequency.keys():
            sorted_frequency[key] /= n
        frequency_list = list(sorted_frequency.values())
        value_types = list(sorted_frequency.keys())
        recur_merge_regression(bins, frequency_list, value_types, residual_f=1.0, ori_fe=ori_fe)

    else:
        # 先计算每个分类变量的卡方值
        sorted_chi2_dict = calculate_chi2(ori_fe, label)
        recur_merge_classify(sorted_chi2_dict, bins, ori_fe)
    ori_fe = categories_to_int(ori_fe)
    if memory is not None:
        value_mapping_dict = {}
        for ori_v, v in zip(ori_fe_copy, ori_fe):
            if ori_v not in value_mapping_dict:
                value_mapping_dict[ori_v] = v
        memory.binning_discrete_info[col_index] = value_mapping_dict
    return ori_fe


def binning(ori_fe: np.array,
            bins: int,
            col_index: int,
            method: str = 'frequency',
            memory: Memory = None,
            isvalid: bool = False):
    """Merge continous feature to target bins.

    Input original col must be continuous feature.

    Parameters
    ----------
    :param ori_fe: np.array
    :param bins: int
    :param method: str, value must be 'frequency' or 'distance'
    :param memory: class Memory
    :param isvalid: bool,judge whether to get frelist in memory
    Returns
    ----------
    returns: tuple (new_fe, fre_list, new_fe_encode)
        - new_fe : np.array, 1D
        - fre_list : list of floats
        - new_fe_encode : np.array, 2D
    """

    ori_fe = np.array(ori_fe)
    if isvalid:
        if col_index in memory.binning_continuous_info:
            fre_list = memory.binning_continuous_info[col_index]
            new_fe = np.array([ff(x, fre_list) for x in ori_fe])
            return new_fe.reshape(len(new_fe), 1), fre_list

    if method == 'frequency':
        fre_list = [np.percentile(ori_fe, 100 / bins * i) for i in range(1, bins)]
        fre_list = sorted(list(set(fre_list)))
    elif method == 'distance':
        umax = np.percentile(ori_fe, 99.99)
        umin = np.percentile(ori_fe, 0.01)
        step = (umax - umin) / bins
        fre_list = [umin + i * step for i in range(bins)]
    else:
        raise ValueError('Method value must be frequency or distance.')
    if memory is not None:
        memory.binning_continuous_info[col_index] = fre_list
    new_fe = np.array([ff(x, fre_list) for x in ori_fe])
    return new_fe.reshape(-1, 1), fre_list


def generate_combine_onehot(ori_fes: np.array,
                            feasible_values: list) -> np.array:
    '''convert combine category feature to onehot feature'''
    c, k = len(feasible_values), len(ori_fes)
    new_fes_encode = np.zeros((k, c))
    for i in range(k):
        combine_feature_value = ''.join(str(int(x)) for x in ori_fes[i])
        if combine_feature_value in feasible_values:
            ind = feasible_values.index(combine_feature_value)
            new_fes_encode[i, ind] = 1
        else:
            new_fes_encode[i, -1] = 1
    return new_fes_encode


def dfs(combine_categories_list: list,
        feasible_value: str,
        feasible_values: list) -> None:
    '''recursion to generate combine feasible_values list'''
    if not len(combine_categories_list):
        feasible_values.append(feasible_value)
        return
    for category_num in combine_categories_list[0]:
        feasible_value = feasible_value + str(int(category_num))
        dfs(combine_categories_list[1:], feasible_value, feasible_values)
        feasible_value = feasible_value[:-1]


def features_combine(ori_fes: np.array,
                     indexs: list,
                     memory: Memory = None,
                     isvalid: bool = False) -> np.array:
    '''function to category feature combine operation'''
    # 这几列的组合操作还没有记录
    if isvalid:
        feasible_values = memory.feature_combine_info[str(indexs)]
        new_fes_encode = generate_combine_onehot(ori_fes, feasible_values)
        return new_fes_encode

    combine_type = ori_fes.shape[1]
    c, combine_categories_list = 1, []
    for i in range(combine_type):
        fe_categories_unique = np.unique(ori_fes[:, i])
        combine_categories_list.append(fe_categories_unique)
        c *= len(fe_categories_unique)
    feasible_values = []

    dfs(combine_categories_list, '', feasible_values)
    if memory is not None:
        memory.feature_combine_info[str(indexs)] = feasible_values

    new_fes_encode = generate_combine_onehot(ori_fes, feasible_values)
    return new_fes_encode


# def generate_combine_fe(ori_fes: np.array,
#                         feasible_values: list) -> np.array:
def generate_combine_fe(ori_fes: np.array,
                        feasible_values: dict) -> np.array:
    '''convert combine category feature to onehot feature'''
    k = len(ori_fes)
    new_fe = np.zeros(k)
    for i in range(k):
        combine_feature_value = ''.join(str(int(x)) for x in ori_fes[i])
        ind = feasible_values[combine_feature_value]
        new_fe[i] = ind
    return new_fe.reshape(-1, 1)


def features_combine_ori(ori_fes: np.array,
                         indexs: list,
                         memory: Memory = None,
                         isvalid: bool = False) -> np.array:
    '''function to category feature combine operation'''
    # 这几列的组合操作还没有记录
    if isvalid:
        feasible_values = memory.feature_combine_info[str(indexs)]
        new_fe = generate_combine_fe(ori_fes, feasible_values)
        return new_fe

    combine_type = ori_fes.shape[1]
    c, combine_categories_list = 1, []
    for i in range(combine_type):
        fe_categories_unique = np.unique(ori_fes[:, i])
        combine_categories_list.append(fe_categories_unique)
        c *= len(fe_categories_unique)
    feasible_values = []

    dfs(combine_categories_list, '', feasible_values)
    if memory is not None:
        memory.feature_combine_info[str(indexs)] = feasible_values

    new_fe = generate_combine_fe(ori_fes, feasible_values)

    return new_fe
