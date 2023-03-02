class Memory(object):
    '''get feature generate related informations,include max_min
    /binning for discrete/binning for continuous features/
    feature combine operations'''
    def __init__(self):
        self.normalization_info = {}
        self.binning_continuous_info = {}
        self.binning_discrete_info = {}  # 需要同时记录分几箱以及不同数值的映射
        self.feature_combine_info = {}
        self.category_to_int_info = {}
        self.min_max_info = {}

class MemorySelect(object):
    '''get feature selected index'''
    def __init__(self):
        self.chi2_selected =  None
        self.mutual_info_selected = None
        self.SVD_selected = None
        self.PCA_selected = None
