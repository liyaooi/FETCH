import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2
from sklearn.utils import as_float_array, check_random_state
from sklearn.utils.validation import check_X_y, check_array
from scipy.sparse import issparse
from sklearn.utils.fixes import _astype_copy_false
from sklearn.preprocessing import scale
from sklearn.feature_selection._mutual_info import _compute_mi, _iterate_columns
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.utils.multiclass import check_classification_targets
from sklearn.feature_selection import SelectFromModel


def _clean_nans(scores):
    """
    Fixes Issue #1240: NaNs can't be properly compared, so change them to the
    smallest value of scores's dtype. -inf seems to be unreliable.
    """

    scores = as_float_array(scores, copy=True)
    scores[np.isnan(scores)] = np.finfo(scores.dtype).min
    return scores


class Selector(object):
    def __init__(self):
        pass

    def fit(self, x, y):
        raise NotImplementedError


# 1.方差选择，自动剔除方差小于阈值的
# 用于连续特征
class VarianceSelector(Selector):

    def __init__(self, threshold=1e-4):
        super(VarianceSelector, self).__init__()
        self.threshold = threshold

    def fit(self, x: np.array, y=None):
        '''
        return type: new_x: np.mat
                     valid_indexs_list:np.array
        '''
        vars = np.var(x, axis=0)
        valid_indexs_list = np.argwhere(vars > self.threshold).reshape(-1)
        new_x = x[:, valid_indexs_list]
        return new_x, valid_indexs_list

    def _check_params(self, X, y):
        if not 0 <= self.threshold <= 0:
            raise ValueError("threshold should be >=0, got %r"
                             % self.threshold)


# 2.采用chi2值进行分类单变量特征选择，根据输入的百分比来选择特征，默认为50，建议加入搜索机制
# 卡方选择器用于分类问题的分类
class SelectPercentileChi2(Selector):

    def __init__(self, percentile=50):
        super(SelectPercentileChi2, self).__init__()
        self.percentile = percentile

    def fit(self, x: np.array, y):
        self._scores, self.pvalues = chi2(x, y)
        scores = _clean_nans(self._scores)
        scores_percentile = np.percentile(scores, self.percentile)
        valid_index_list = np.argwhere(self._scores > scores_percentile and self.pvalues < 0.05).reshape(-1)
        return x[:, valid_index_list], valid_index_list

    def _check_params(self, X, y):
        if not 0 <= self.percentile <= 100:
            raise ValueError("percentile should be >=0, <=100; got %r"
                             % self.percentile)


# 3.根据pearson相关系数对数值型变量进行筛选
# Pearson选择器只能针对回归问题
class SelectPercentilePearson(Selector):
    def __init__(self, percentile=50):
        super(SelectPercentilePearson, self).__init__()
        self.percentile = percentile

    def fit(self, x: np.array, y: np.array):
        y = y.reshape(-1)
        scores, k = [], x.shape[1]
        for i in range(k):
            x_col = x[:, i]
            pr, pv = pearsonr(x_col, y)
            scores.append(pr)
        # scores,pv = pearsonr(x,y)
        scores = np.array(scores)
        score_percentile = np.percentile(np.abs(scores), self.percentile)
        valid_indexs_list = np.argwhere(abs(scores) > score_percentile).reshape(-1)
        return x[:, valid_indexs_list], valid_indexs_list

    def _check_params(self, X, y):
        if not 0 <= self.percentile <= 100:
            raise ValueError("percentile should be >=0, <=100; got %r"
                             % self.percentile)


# 4.互信息法特征筛选,可用于分类变量，也可用于连续变量；可用于分类问题，也可用于回归问题
# 筛选互信息大于0的特征列
class SelectMutualInfo(Selector):

    # 互信息法特征选择默认模式是分类
    def __init__(self, mode='classify', threshold=0):
        # mode:{'classify' or 'regression'}
        super(SelectMutualInfo, self).__init__()
        self.mode = mode
        self.threshold = threshold

    def estimate_mi(self, x, y, discrete_features='auto', discrete_target=False,
                    n_neighbors=3, copy=True, random_state=None):
        y = y.reshape(-1)
        x, y = check_X_y(x, y, accept_sparse='csc', y_numeric=not discrete_target)
        n_samples, n_features = x.shape

        if isinstance(discrete_features, (str, bool)):
            if isinstance(discrete_features, str):
                if discrete_features == 'auto':
                    discrete_features = issparse(x)
                else:
                    raise ValueError("Invalid string value for discrete_features.")
            discrete_mask = np.empty(n_features, dtype=bool)
            discrete_mask.fill(discrete_features)
        else:
            discrete_features = check_array(discrete_features, ensure_2d=False)
            if discrete_features.dtype != 'bool':
                discrete_mask = np.zeros(n_features, dtype=bool)
                discrete_mask[discrete_features] = True
            else:
                discrete_mask = discrete_features

        continuous_mask = ~discrete_mask
        if np.any(continuous_mask) and issparse(x):
            raise ValueError("Sparse matrix `X` can't have continuous features.")

        rng = check_random_state(random_state)
        if np.any(continuous_mask):
            if copy:
                x = x.copy()

            if not discrete_target:
                x[:, continuous_mask] = scale(x[:, continuous_mask],
                                              with_mean=False, copy=False)

        x = x.astype(float, **_astype_copy_false(x))
        means = np.maximum(1, np.mean(np.abs(x[:, continuous_mask]), axis=0))
        x[:, continuous_mask] += 1e-10 * means * rng.randn(
            n_samples, np.sum(continuous_mask))

        if not discrete_target:
            if not discrete_target:
                y = scale(y, with_mean=False)
                y += 1e-10 * np.maximum(1, np.mean(np.abs(y))) * rng.randn(n_samples)

        mi = [_compute_mi(x, y, discrete_feature, discrete_target, n_neighbors) for
              x, discrete_feature in zip(_iterate_columns(x), discrete_mask)]

        return np.array(mi)

    def calculate_mutual_info(self, x, y, *, discrete_features='auto', n_neighbors=3,
                              copy=True, random_state=None):
        if self.mode == 'classify':
            return self.estimate_mi(x, y, discrete_features, False,
                                    n_neighbors, copy, random_state)
        elif self.mode == 'regression':
            check_classification_targets(y)
            return self.estimate_mi(x, y, discrete_features, True, n_neighbors, copy, random_state)
        else:
            raise ValueError('Selector mode valueError,choose classify or regression')

    def fit(self, x: np.array, y):
        mutual_informations = self.calculate_mutual_info(x, y)
        # mutual_information_boundry = np.percentile(mutual_informations,self.percentile)
        valid_index = np.argwhere(mutual_informations > self.threshold).reshape(-1)
        return x[:, valid_index], valid_index

    def _check_params(self, X, y):
        if self.mode not in ['regression', 'classify']:
            raise ValueError("selector mode should be classify or regression; got %r"
                             % self.mode)


def mutual_info_filter(x, y, mode,threshold=0):
    if mode == "classify":
        mic = mutual_info_classif(x, y,random_state=1)
        mic_index = np.argwhere(mic > threshold).reshape(-1)
        return x[:, mic_index], mic_index
    elif mode == "regression":
        mir = mutual_info_regression(x, y,random_state=1)
        mir_index = np.argwhere(mir > threshold).reshape(-1)
        return x[:, mir_index], mir_index


def chi2_filter(x_train, y_train, p_threshold=0.05):
    '''
    Parameters
    ----------
    x_train : 2darray
    y_train : 1darray
        must discrete.
    p_threshold : float, 0.01/ 0.05
        The threshold of the chi-square test. The default is 0.05.
    Returns
    -------
    x_train_p : 2darray
    只针对discrete columns
    '''
    # 记录onehot columns and idx
    """ 
    continue_idx = [idx for idx in range(x_train.shape[1]) if len(np.unique(x_train[:, idx])) > 2]  # 连续特征index
    not_continue_idx = [idx for idx in range(x_train.shape[1]) if idx not in continue_idx]
    x_train_discrete = np.delete(x_train, continue_idx, axis=1)  # 离散特征

    x_train_continuous = np.delete(x_train, not_continue_idx, axis=1)  # 连续特征
    zerostd_idx = [idx for idx in range(x_train_discrete.shape[1]) if np.std(x_train_discrete[:, idx]) == 0]
    x_train_discrete_nozerostd = np.delete(x_train_discrete, zerostd_idx, axis=1)  # 标准差不为0

    chivalue, pvalues_chi = chi2(x_train_discrete_nozerostd, y_train)
    # k = chivalue.shape[0] - (pvalues_chi > 0.05).sum()
    # X_fschi = SelectKBest(chi2, k=k).fit_transform(x_train, y_train)
    delete_p_idx = [idx for idx in range(len(pvalues_chi)) if pvalues_chi[idx] > p_threshold]
    x_train_discrete_nozerostd_pfilter = np.delete(x_train_discrete_nozerostd, delete_p_idx, axis=1)  # 卡方检验筛选

    x_train_res = np.concatenate((x_train_continuous, x_train_discrete_nozerostd_pfilter), axis=1)
    """
    chi2_values, p_values = chi2(x_train, y_train)
    chi2_index = np.argwhere(p_values < p_threshold).reshape(-1)
    return x_train[:, chi2_index], chi2_index


class ModelSelector(Selector):
    def __init__(self, model, percentile=80):
        super(ModelSelector, self).__init__()
        self.model = model
        self.percentile = percentile

    def fit_old(self, X, y):
        self.model.fit_attention(X, y)
        feature_importances = self.model.feature_importances_
        theshold = np.percentile(feature_importances, 100. - self.percentile)
        valid_indexs_list = np.argwhere(feature_importances >= theshold).reshape(-1)
        selected_feature = X[:, valid_indexs_list]
        return selected_feature, feature_importances

    def _check_params(self, X, y):
        if not 0 <= self.percentile <= 100:
            raise ValueError("percentile should be >=0, <=100; got %r"
                             % self.percentile)

    def fit(self, X, y):
        # 按特征重要性百分比筛选，percentile ，要保留多少的信息量百分比
        self.model.fit_attention(X, y)
        feature_importances = self.model.feature_importances_
        feature_importances_dict = {idx: imp for idx, imp in enumerate(feature_importances)}
        sort_imp_dict = sorted(feature_importances_dict.items(), key=lambda x: x[1], reverse=True)
        sort_imp_idx = [item[0] for item in sort_imp_dict]
        sort_imp_value = [item[1] for item in sort_imp_dict]
        cumsum_imp = np.cumsum(sort_imp_value)
        percentile_len = sum(cumsum_imp <= self.percentile / 100) + 1

        delete_idx = sort_imp_idx[percentile_len:]
        res_idx = sort_imp_idx[:percentile_len]  # 保留的索引

        res_fes = np.delete(X, delete_idx, axis=1)
        return res_fes, res_idx

# if __name__ == "__main__":
#     # 测试几种Selector
#
#     # 1. test_variance_selector
#     # data = np.random.random((10,5))
#     # data2 = np.zeros((10,1))
#     # new_data = np.concatenate((data,data2),axis=1)
#     # var_selectors = VarianceSelector() # theshold默认为 0.0001
#     # af_data,index = var_selectors.fit(new_data)
#     # print(af_data)
#     # print(index)
#
#     # 2.test_chi2_selector
#     # data = np.random.randint(2,size = (10,10))
#     # label = np.random.randint(2,size = (10,1))
#     # print(data)
#     # print(label)
#     # chi2_selector = SelectPercentileChi2() # percentile 默认为50
#     # x,index = chi2_selector.fit(data,label)
#     # print(x)
#     # print(index)
#     # 3.test_pearson_selector
#     # data = np.random.random((10,10))
#     # label = np.random.random((10,1))
#     # # print(data)
#     # # print(label)
#     # pearson_selector = SelectPercentilePearson()
#     # new_x,index = pearson_selector.fit(data,label)
#     # print(new_x)
#     # print(index)
#     # 4 test_mutual_info_selector
#     # 1.classify mode
#     import time
#     start = time.time()
#     data = np.random.random(size = (10,1000))
#     label = np.random.random(size = (10,1))
#     # print(label)
#     mutual_info_selector = SelectMutualInfo()
#     new_x,index = mutual_info_selector.fit(data,label)
#     print(new_x.shape)
#     print(len(index))
#     end = time.time()
#     print(end - start)
