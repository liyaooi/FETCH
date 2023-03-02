class Worker(object):
    def __init__(self, args):
        self.args = args

        self.states = None
        self.actions = None
        self.log_probs = None
        self.values = None
        self.accs = None
        self.model = None
        self.orders = None
        self.dones = None
        self.cvs = None
        self.steps = None
        self.features = None
        self.ff = None
        self.steps = None
        self.fe_nums = None
        self.repeat_fe_nums = None
        self.scores = None
        self.scores_test = None
        self.m1 = []
        self.m2 = []
        self.m3 = []
        self.action_softmax = []
