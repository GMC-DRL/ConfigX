import pickle
from config import get_options
from utils.utils import *
from env import bbob
from torch.utils.data import Dataset
from env.protein_docking import Protein_Docking_Dataset
import copy
import numpy as np
from env.optimizer_env import Optimizer
from components.operators import *
import xgboost as xgb


# Control or not: {module name: {sub-module name: {parameters}}}
module_dict = {
    'Uncontrollable': {
        'Initialization': [
            {'class': 'Gaussian_Init', 'param': {}},
            {'class': 'Sobol_Init', 'param': {}},
            {'class': 'LHS_Init', 'param': {}},
            {'class': 'Halton_Init', 'param': {}},
            {'class': 'Uniform_Init', 'param': {}},
        ],
        'Niching': [
            {'class': 'Rand_Nich', 'param': {'Npop': 2}},
            {'class': 'Rank_Nich', 'param': {'Npop': 2}},
            {'class': 'Distance_Nich', 'param': {'Npop': 2}},
            {'class': 'Rand_Nich', 'param': {'Npop': 3}},
            {'class': 'Rank_Nich', 'param': {'Npop': 3}},
            {'class': 'Distance_Nich', 'param': {'Npop': 3}},
            {'class': 'Rand_Nich', 'param': {'Npop': 4}},
            {'class': 'Rank_Nich', 'param': {'Npop': 4}},
            {'class': 'Distance_Nich', 'param': {'Npop': 4}},
        ],
        'BC': [
            {'class': 'Clip_BC', 'param': {}},
            {'class': 'Rand_BC', 'param': {}},
            {'class': 'Periodic_BC', 'param': {}},
            {'class': 'Reflect_BC', 'param': {}},
            {'class': 'Halving_BC', 'param': {}},
        ],
        'Selection': [
            {'class': 'DE_like', 'param': {}},
            {'class': 'Crowding', 'param': {}},
            {'class': 'PSO_like', 'param': {}},
            {'class': 'Ranking', 'param': {}},
            {'class': 'Tournament', 'param': {}},
            {'class': 'Roulette', 'param': {}},
        ],
        'Restart': [
            {'class': 'Stagnation', 'param': {}},
            {'class': 'Conver_x', 'param': {}},
            {'class': 'Conver_y', 'param': {}},
            {'class': 'Conver_xy', 'param': {}},
        ],
        'Reduction': [
            {'class': 'Linear', 'param': {}},
            {'class': 'Non_Linear', 'param': {}},
        ],
        'Termination': [{'class': 'Termination', 'param': {}}]
    },
    'Controllable': {
        'Mutation': [
            {'class': 'rand1', 'param': {}},
            {'class': 'rand2', 'param': {}},
            {'class': 'best1', 'param': {}},
            {'class': 'best2', 'param': {}},
            {'class': 'current2best', 'param': {}},
            {'class': 'current2rand', 'param': {}},
            {'class': 'rand2best', 'param': {}},
            {'class': 'current2best', 'param': {'use_qbest': True, }},  # current-to-pbest
            {'class': 'current2best', 'param': {'use_qbest': True, 'use_archive': True, 'archive_id': [1]}},  # current-to-pbest/1+archive
            {'class': 'weighted_rand2best', 'param': {'use_qbest': True}},  # weighted-rand-to-pbest
            {'class': 'current2rand', 'param': {'use_archive': True, 'archive_id': [2]}},  # current-to-rand/1+archive
            {'class': 'rand2best', 'param': {'use_qbest': True}},
            {'class': 'best2', 'param': {'use_qbest': True,}},
            {'class': 'rand2', 'param': {'use_archive': True, 'archive_id': [4]}},
            {'class': 'rand2best', 'param': {'use_qbest': True,}},
            {'class': 'rand2best', 'param': {'use_qbest': True, 'use_archive': True, 'archive_id': [1]}},
            {'class': 'gaussian', 'param': {}},
            {'class': 'polynomial', 'param': {}},
            {'class': 'random_mutation', 'param': {}},
        ],
        'Crossover': [
            {'class': 'binomial', 'param': {}},
            {'class': 'exponential', 'param': {}},
            {'class': 'binomial', 'param': {'use_qbest': True, }},  # qbest-Binomial
            {'class': 'binomial', 'param': {'use_qbest': True, 'united_qbest': True, 'use_archive': True}},  # qbest-Binomial+archive
            {'class': 'sbx', 'param': {}},
            {'class': 'arithmetic', 'param': {}},
            {'class': 'mpx', 'param': {}},
        ],
        'PSO_update': [
            {'class': 'vanilla_PSO', 'param': {}},
            {'class': 'FDR_PSO', 'param': {}},
            {'class': 'CLPSO', 'param': {}},
        ],
        'Multi_strategy': [
            {'class': 'Multi_strategy', 'param': {'op_list':   # Multi_BC
                                           [{'class': 'Clip_BC', 'param': {}},
                                            {'class': 'Rand_BC', 'param': {}},
                                            {'class': 'Periodic_BC', 'param': {}},
                                            {'class': 'Reflect_BC', 'param': {}},
                                            {'class': 'Halving_BC', 'param': {}},],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_1
                                            [{'class': 'current2best', 'param': {'use_qbest': True, 'use_archive': True, 'archive_id': [1]}},  # current-to-pbest/1+archive
                                             {'class': 'current2rand', 'param': {'use_archive': True, 'archive_id': [2]}},  # current-to-rand/1+archive
                                             {'class': 'weighted_rand2best', 'param': {'use_qbest': True}}],}},  # weighted-rand-to-pbest
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_2
                                            [{'class': 'rand1', 'param': {}},
                                             {'class': 'rand2', 'param': {}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_3
                                            [{'class': 'rand1', 'param': {}},
                                             {'class': 'best2', 'param': {}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_4
                                            [{'class': 'rand1', 'param': {}},
                                             {'class': 'best2', 'param': {}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_5
                                            [{'class': 'rand1', 'param': {}},
                                             {'class': 'best1', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_6
                                            [{'class': 'current2best', 'param': {}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_7
                                            [{'class': 'rand2', 'param': {}},
                                             {'class': 'current2best', 'param': {}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_8
                                            [{'class': 'best2', 'param': {}},
                                             {'class': 'current2best', 'param': {}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_9
                                            [{'class': 'current2best', 'param': {}},
                                             {'class': 'rand2best', 'param': {}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_10
                                            [{'class': 'rand2', 'param': {}},
                                             {'class': 'best2', 'param': {}},
                                             {'class': 'current2best', 'param': {}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_11
                                            [{'class': 'rand2', 'param': {}},
                                             {'class': 'best2', 'param': {}},
                                             {'class': 'current2best', 'param': {'use_qbest': True,}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_12
                                            [{'class': 'current2best', 'param': {'use_qbest': True, }},  # current-to-pbest
                                             {'class': 'current2rand', 'param': {'use_archive': True, 'archive_id': [2]},}],}},  # current-to-rand/1+archive
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_13
                                            [{'class': 'current2best', 'param': {'use_qbest': True, 'use_archive': True, 'archive_id': [1]}},  # current-to-pbest/1+archive
                                             {'class': 'rand1', 'param': {}},
                                             {'class': 'best1', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_14
                                            [{'class': 'rand1', 'param': {}},
                                             {'class': 'rand2', 'param': {}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_15
                                            [{'class': 'best1', 'param': {}},
                                             {'class': 'best2', 'param': {}},
                                             {'class': 'current2best', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_16
                                            [{'class': 'rand2best', 'param': {'use_qbest': True}},
                                             {'class': 'current2best', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_17
                                            [{'class': 'rand2', 'param': {}},
                                             {'class': 'best2', 'param': {'use_qbest': True,}},
                                             {'class': 'current2best', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_18
                                            [{'class': 'rand2', 'param': {}},
                                             {'class': 'rand2best', 'param': {'use_qbest': True, 'use_archive': True, 'archive_id': [1]},}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_19
                                            [{'class': 'rand2', 'param': {}},
                                             {'class': 'best2', 'param': {'use_qbest': True,}},
                                             {'class': 'rand2best', 'param': {'use_qbest': True,},}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_20
                                            [{'class': 'rand2', 'param': {'use_archive': True, 'archive_id': [4]}},
                                             {'class': 'best2', 'param': {'use_qbest': True,},}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_21
                                            [{'class': 'rand2best', 'param': {}},
                                             {'class': 'rand2best', 'param': {'use_qbest': True, 'use_archive': True, 'archive_id': [1]},}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_22
                                            [{'class': 'rand2', 'param': {'use_archive': True, 'archive_id': [4]}},
                                             {'class': 'best2', 'param': {'use_qbest': True,}},
                                             {'class': 'current2best', 'param': {'use_qbest': True, 'use_archive': True, 'archive_id': [1]}},
                                             {'class': 'current2rand', 'param': {'use_archive': True, 'archive_id': [2]},}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_23
                                            [{'class': 'best1', 'param': {}},
                                             {'class': 'best2', 'param': {},}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_24
                                            [{'class': 'rand1', 'param': {}},
                                             {'class': 'rand2', 'param': {},}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_25
                                            [{'class': 'current2best', 'param': {'use_qbest': True, 'use_archive': True, 'archive_id': [1]}},
                                             {'class': 'current2best', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_26
                                            [{'class': 'current2rand', 'param': {'use_archive': True, 'archive_id': [2]}},
                                             {'class': 'current2rand', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_27 GA
                                            [{'class': 'polynomial', 'param': {}},
                                             {'class': 'gaussian', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_28 GA
                                            [{'class': 'polynomial', 'param': {}},
                                             {'class': 'random_mutation', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_29 GA
                                            [{'class': 'random_mutation', 'param': {}},
                                             {'class': 'gaussian', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Mutation_30 GA
                                            [{'class': 'polynomial', 'param': {}},
                                             {'class': 'random_mutation', 'param': {}},
                                             {'class': 'gaussian', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_1
                                            [{'class': 'binomial', 'param': {}},
                                             {'class': 'exponential', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_2
                                            [{'class': 'binomial', 'param': {}},
                                             {'class': 'binomial', 'param': {'use_qbest': True, 'united_qbest': True, 'use_archive': True}}],}},  # qbest-Binomial+archive
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_3
                                            [{'class': 'binomial', 'param': {}},
                                             {'class': 'binomial', 'param': {'use_qbest': True, },}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_4
                                            [{'class': 'binomial', 'param': {'use_qbest': True, }},
                                             {'class': 'exponential', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_5
                                            [{'class': 'binomial', 'param': {'use_qbest': True, 'united_qbest': True, 'use_archive': True}},
                                             {'class': 'exponential', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_6
                                            [{'class': 'binomial', 'param': {}},
                                             {'class': 'binomial', 'param': {'use_qbest': True}},
                                             {'class': 'exponential', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_7 GA
                                            [{'class': 'sbx', 'param': {}},
                                             {'class': 'arithmetic', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_8 GA
                                            [{'class': 'sbx', 'param': {}},
                                             {'class': 'mpx', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_9 GA
                                            [{'class': 'mpx', 'param': {}},
                                             {'class': 'arithmetic', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_Crossover_10 GA
                                            [{'class': 'sbx', 'param': {}},
                                             {'class': 'mpx', 'param': {}},
                                             {'class': 'arithmetic', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_PSO_1
                                            [{'class': 'FDR_PSO', 'param': {}},
                                             {'class': 'CLPSO', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_PSO_2
                                            [{'class': 'vanilla_PSO', 'param': {}},
                                             {'class': 'CLPSO', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_PSO_3
                                            [{'class': 'vanilla_PSO', 'param': {}},
                                             {'class': 'FDR_PSO', 'param': {}}],}},
            {'class': 'Multi_strategy', 'param': {'op_list':  # Multi_PSO_4
                                            [{'class': 'vanilla_PSO', 'param': {}},
                                             {'class': 'CLPSO', 'param': {}},
                                             {'class': 'FDR_PSO', 'param': {}}],}},
        ],
        'Sharing': [
            {'class': 'Comm', 'param': {}},
        ],
    }
}


class Module_pool():
    def __init__(self, module_dict=module_dict, ban_range=[], ) -> None:
        self.module_dict = module_dict
        self.ban_range = ban_range
        self.pool = {}
        self.id_count = {}
        self.N = 0
        if module_dict is not None:
            for cab in module_dict.keys():
                for mod in module_dict[cab].keys():
                    for submod_dict in module_dict[cab][mod]:
                        submod = submod_dict['class']
                        if mod == 'Multi_strategy':
                            op_list = []
                            for op_dict in submod_dict['param']['op_list']:
                                op = op_dict['class']
                                sub_item = eval(op)(**op_dict['param'])
                                op_list.append(sub_item)
                            item = eval(mod)(op_list)
                        else:
                            item = eval(submod)(**submod_dict['param'])
                        topo_type = item.topo_type
                        if item.mod_type not in self.id_count.keys():
                            self.id_count[item.mod_type] = 0
                        self.id_count[item.mod_type] += 1
                        if item.__class__.__name__ in ban_range or item.topo_type in ban_range:
                            continue
                        item.id.append(self.id_count[item.mod_type])
                        if topo_type not in self.pool.keys():
                            self.pool[topo_type] = []
                        self.pool[topo_type].append(item)
                        self.N += 1
            print(f'Load {self.N} sub-modules.')
        

    def register(self, dict):
        pass

    def get(self, topo_type, rng=None) -> Module:
        if rng is None:
            rng = np.random
        item = rng.choice(self.pool[topo_type])
        return item
    

def construct_algorithm(pool: Module_pool, ban_range=[], rng=None):  # DE range: ['Initialization', 'Niching', 'BC', 'DE_Selection', 'Restart', 'Reduction', 'DE_Mutation', 'DE_Crossover', 'Sharing', 'Termination']
    if rng is None:
        rng = np.random
    modules = []
    init = pool.get('Initialization', rng=rng)
    modules.append(init)
    topo_rule = []
    for rule in init.get_topo_rule():
        if rule not in ban_range:
            topo_rule.append(rule)
    nex = pool.get(rng.choice(topo_rule), rng=rng)
    npop = 1
    if nex.topo_type == 'Niching':
        modules.append(nex)
        npop = nex.Npop
    if npop < 2:
        ban_range.append('Sharing')
    for p in range(npop):
        modules.append([])
        if nex.topo_type != 'Niching':
            modules[-1].append(nex)
        former_type = init.topo_type
        topo_rule = []
        for rule in nex.get_topo_rule(former_type):
            if rule not in ban_range:
                topo_rule.append(rule)
        nnex = pool.get(rng.choice(topo_rule), rng=rng)
        former_type = nex.topo_type
        while nnex.topo_type != 'Termination':
            modules[-1].append(nnex)
            topo_rule = []
            for rule in modules[-1][-1].get_topo_rule(former_type):
                if rule not in ban_range:
                    topo_rule.append(rule)
            nnex = pool.get(rng.choice(topo_rule), rng=rng)
            former_type = modules[-1][-1].topo_type
    return modules

ban_list4GAPSO = ['GA_Crossover', 'PSO_update', 'GA_Mutation', 'GA_Selection', 'PSO_Selection', ]


class Taskset(Dataset):
    def __init__(self,
                 data,
                 batch_size=16):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)

    @staticmethod
    def get_datasets(train_set, test_set,
                     train_batch_size=1,
                     test_batch_size=1,
                     ):
        # with open(path, 'rb') as f:
        #     data = pickle.load(f)
        # train_set, test_set = data['train'], data['test']
        return Taskset(train_set, train_batch_size), Taskset(test_set, test_batch_size)

    def __getitem__(self, item):
        # if self.batch_size < 2:
        #     return self.data[self.index[item]]
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: 'Taskset'):
        return Taskset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)


class Problem_hpo(object):
    def __init__(self,bst_surrogate,dim,y_min,y_max,config) -> None:
        self.bst_surrogate=bst_surrogate
        self.y_min=y_min
        self.y_max=y_max
        self.config=config
        self.dim=dim

    def reset(self):
        pass

    def eval(self,position):
        x = (position + self.config.Xmax) / (2 * self.config.Xmax)
        x_q = xgb.DMatrix(x.reshape(-1,self.dim))
        new_y = self.bst_surrogate.predict(x_q)
        
        return 6-self.normalize(new_y)

    def normalize(self, y):
        # if y_min is None:
        #     return (y-np.min(y))/(np.max(y)-np.min(y))
        # else:
        return np.clip((y-self.y_min)/(self.y_max-self.y_min),0,5)


def get_usual_tasks(config):
    traintask, testtask = [], []
    ban_range = []
    if config.task_class == 'DE':
        ban_range=ban_list4GAPSO
    pool = Module_pool(ban_range=copy.deepcopy(ban_range))
    traindata_seed = config.traindata_seed
    cover_record = {}
    # while len(list(cover_record.keys())) < pool.N - 1:  # exclude the Termination
    set_seed(traindata_seed)
    rng = np.random.RandomState(traindata_seed)
    nmax = 0
    for i in range(config.trainsize):
        mods = construct_algorithm(pool, ban_range=copy.deepcopy(ban_range), rng=rng)
        traintask.append(mods)
        n = 0
        for m in mods:
            if isinstance(m, list):
                for mm in m:
                    if mm.get_id_hash() not in cover_record.keys():
                        cover_record[mm.get_id_hash()] = 0
                    cover_record[mm.get_id_hash()] += 1
                    n += 1
            else:
                if m.get_id_hash() not in cover_record.keys():
                    cover_record[m.get_id_hash()] = 0
                cover_record[m.get_id_hash()] += 1
                n += 1
        nmax = max(nmax, n)
        # if len(list(cover_record.keys())) < pool.N - 1:
        #     print(traindata_seed, len(list(cover_record.keys())))
        #     traindata_seed += 1
        #     cover_record = {}
        #     traintask = []
        #     nmax = 0
    print('traindata_seed:', traindata_seed)
    print('train_mod_num:', cover_record)
    print('train maximal mod num:', nmax)

    cover_record = {}
    set_seed(config.testdata_seed)
    rng = np.random.RandomState(config.testdata_seed)
    nmax = 0
    for i in range(config.testsize):
        mods = construct_algorithm(pool, ban_range=copy.deepcopy(ban_range), rng=rng)
        testtask.append(mods)
        n = 0
        for m in mods:
            if isinstance(m, list):
                for mm in m:
                    if mm.__class__.__name__ not in cover_record.keys():
                        cover_record[mm.__class__.__name__] = 0
                    cover_record[mm.__class__.__name__] += 1
                    n += 1
            else:
                if m.__class__.__name__ not in cover_record.keys():
                    cover_record[m.__class__.__name__] = 0
                cover_record[m.__class__.__name__] += 1
                n += 1
        nmax = max(nmax, n)
    print('test_mod_num:', cover_record)
    print('test maximal mod num:', nmax)
    if config.task_class == 'DE':
        np.save('DE_alg_dataset.npy', {'train': traintask, 'test': testtask}, allow_pickle=True)
    else:
        np.save('overall_alg_dataset.npy', {'train': traintask, 'test': testtask}, allow_pickle=True)
    return traintask, testtask


def get_real_tasks(config):
    de = {'sample': 'uniform',
            'Npop': 1,
            'NPmax': [100],
            'NPmin': [100],
            'Vmax': 0,
            'NA': 1.0,
            'grouping': None, 
            'regroup': 0, 
            'arch_replace': 'rand',
            'global_strategy': {
            # 'SHA_F': {'class':'SHA', 'args': (100, 'Cauchy', 0.5, -1, 0.1, [0, 1], 'regenerate', 'regenerate', 2, 1, None)},
            # 'SHA_Cr': {'class':'SHA', 'args': (100, 'Normal', 0.5, -1, 0.1, [0, 1], 'regenerate', 'regenerate', 2, 1, None)},
            }, 
            'subpops': [
                {
                'ops': 
                [
                {'class': 'rand2', 
                'args': (False, False, False, [], []), 
                'param_ada': {
                    'F1': 0.5,
                    'F2': 0.5, 
                    }
                },
                {'class': 'binomial', 
                    'args': (False, False, False, [], []),
                    'param_ada': {
                    'Cr': 0.9, 
                    }
                },
                ], 
                'bounding': 'clip',
                'selection': 'direct',
                'comm': None,
                'pop_reduce': None,  
                }, 
            ],
            'regrouping': None,
            }

    shade = {'sample': 'uniform',
            'Npop': 1,
            'NPmax': [230],
            'NPmin': [4],
            'Vmax': 0,
            'NA': 1.0,
            'grouping': None, 
            'regroup': 0, 
            'arch_replace': 'rand',
            'global_strategy': {
            'SHA_F': {'class':'SHA', 'args': (100, 'Cauchy', 0.5, -1, 0.1, [0, 1], 'regenerate', 'regenerate', 2, 1, None)},
            'SHA_Cr': {'class':'SHA', 'args': (100, 'Normal', 0.5, -1, 0.1, [0, 1], 'regenerate', 'regenerate', 2, 1, None)},
            }, 
            'subpops': [
                {
                'ops': 
                [
                {'class': 'current2best', 
                'args': (True, False, True, [1,], []), 
                'param_ada': {
                    'F1': 'SHA_F',
                    'F2': 'F1', 
                    'q': {'class': 'Bound_rand', 'args': (0, 0.2)}
                    }
                },
                {'class': 'binomial', 
                    'args': (False, False, False, [], []),
                    'param_ada': {
                    'Cr': 'SHA_Cr', 
                    }
                },
                ], 
                'bounding': 'halving',
                'selection': 'direct',
                'comm': None,
                'pop_reduce': None,  
                }, 
            ],
            'regrouping': None,
            }

    madde = {'sample': 'uniform',
                'Npop': 1,
                'NPmax': [200],
                'NPmin': [4],
                'Vmax': 0,
                'NA': 2.3,
                'grouping': None, 
                'regroup': 0, 
                'arch_replace': 'worst',
                'global_strategy': {
                'SHA_F': {'class':'SHA', 'args': (100, 'Cauchy', 0.2, 0.5, 0.1, [0, 1], 'regenerate', 'regenerate', 2, 1, None)},
                'SHA_Cr': {'class':'SHA', 'args': (100, 'Normal', 0.2, 0.5, 0.1, [0, 1], 'regenerate', 'regenerate', 2, 1, None)},
                },
                'subpops': [
                    {
                    'ops':
                    [
                        {'class': 'Multi_op', 
                        'op_select': {'class': 'Fitness_rand', 'args': (3, [0.1, 0.9])}, 
                        'op_list': [
                            {'class': 'current2best', 
                            'args': (True, False, True, [1,], []), 
                            'param_ada': {
                            'F1': 'SHA_F',
                            'F2': 'F1', 
                            'q': {'class': 'Linear', 'args': (0.36, 0.18)}
                            }},
                            {'class': 'current2rand', 'args': (False, False, True, [1,], []), 
                            'param_ada': {
                            'F1': 'SHA_F',
                            'F2': 'F1', 
                            }},
                            {'class': 'rand2best', 'args': (True, False, False, [], []), 'param_ada': {
                            'F1': 'SHA_F',
                            'F2': 'F1', 
                            'q': {'class': 'Linear', 'args': (0.36, 0.18)}
                            }},
                        ]},
                        {'class': 'Multi_op', 
                        'op_select': {'class': 'Probability_rand', 'args': (2, [0.99, 0.01])}, 
                        'op_list': [
                            {'class': 'binomial', 'args': (False, False, False, [], []),
                            'param_ada': {
                            'Cr': 'SHA_Cr',
                            }},
                            {'class': 'binomial', 'args': (True, True, True, [], []),
                            'param_ada': {
                            'Cr': 'SHA_Cr',
                            'q': {'class': 'Linear', 'args': (0.36, 0.18)},
                            }},
                        ]},
                    ], 
                    'bounding': 'halving',
                    'selection': 'direct',
                    'comm': None,
                    'pop_reduce': 'linear',  
                    }, 
                ],
                'regrouping': None,
                }

    dmspso = {'sample': 'uniform',
            'Npop': 3,
            'NPmax': 30,
            'NPmin': 30,
            'Vmax': 0.2*config.Xmax,
            'NA': 1.0,
            'grouping': 'rand', 
            'regroup': 5, 
            'arch_replace': 'rand',
            'global_strategy': {
            }, 
            'subpops': [
                [
                {'class': 'currentxbest3', 
                'args': (False, False, False, [], []), 
                'param_ada': {
                    'w': {'class': 'Linear', 'args': (0.9, 0.2)},
                    'c1': {'class': 'DMS', 'args': (0, 1.49445, 0.9)}, 
                    'c2': {'class': 'DMS', 'args': (1.49445, 0, 0.9)},
                    'c3': 1.49445,
                    }
                },
                ], 
                [
                {'class': 'currentxbest3', 
                'args': (False, False, False, [], []), 
                'param_ada': {
                    'w': {'class': 'Linear', 'args': (0.9, 0.2)},
                    'c1': {'class': 'DMS', 'args': (0, 1.49445, 0.9)}, 
                    'c2': {'class': 'DMS', 'args': (1.49445, 0, 0.9)},
                    'c3': 1.49445,
                    }
                },
                ], 
                [
                {'class': 'currentxbest3', 
                'args': (False, False, False, [], []), 
                'param_ada': {
                    'w': {'class': 'Linear', 'args': (0.9, 0.2)},
                    'c1': {'class': 'DMS', 'args': (0, 1.49445, 0.9)}, 
                    'c2': {'class': 'DMS', 'args': (1.49445, 0, 0.9)},
                    'c3': 1.49445,
                    }
                },
                ], 
            ], 
            'bounding': 'clip',
            'selection': 'inherit',
            'regrouping': 'rand',
            'comm': None,
            'pop_reduce': None,  
            }

    nl_shade_lbc = {'sample': 'uniform',
             'Npop': 1,
             'NPmax': [230],
             'NPmin': [4],
             'Vmax': 0,
             'NA': 1.0,
             'grouping': None, 
             'regroup': 0, 
             'arch_replace': 'rand',
             'global_strategy': {
                'SHA_F': {'class':'SHA', 'args': (200, 'Cauchy', 0.5, -1, 0.1, [0, 1], 'regenerate', 'regenerate', 1.5, 1.5, {'class': 'Linear', 'args': (1.5, 3.5)})},
                'SHA_Cr': {'class':'SHA', 'args': (200, 'Normal', 0.9, -1, 0.1, [0, 1], 'regenerate', 'regenerate', 1.5, 1.5, {'class': 'Linear', 'args': (1.5, 1.0)})},
             },
             'subpops': [
                 {
                 'ops':
                 [
                     {'class': 'current2best', 
                          'args': (True, False, True, [1,], []), 
                          'param_ada': {
                              'F1': 'SHA_F', 
                              'F2': 'F1', 
                              'q': {'class': 'Linear', 'args': (0.2, 0.3)},
                          }
                         },
                     {'class': 'binomial', 
                         'args': (False, False, False, [], []),
                         'param_ada': {
                            'Cr': 'SHA_Cr',
                            }},
                 ], 
                'bounding': 'halving',
                'selection': 'direct',
                'comm': None,
                'pop_reduce': 'linear',  
                },
              ],
             'regrouping': None,
             }

    ga = {'sample': 'uniform',
        'Npop': 1,
        'NPmax': 100,
        'NPmin': 100,
        'Vmax': 0,
        'NA': 1.0,
        'grouping': 'rand', 
        'regroup': 0, 
        'arch_replace': 'rand',
        'global_strategy': {
        }, 
        'subpops': [
            [
            {'class': 'sbx', 
            'args': (False, False, False, [], []), 
            'param_ada': {
                'n': 2,
                'select': 'rand', 
                }
            },
            {'class': 'gaussian', 
            'args': (False, False, False, [], []), 
            'param_ada': {
                'sigma': 0.1,
                }
            },
            ], 
        ], 
        'bounding': 'clip',
        'selection': 'tournament', 
        'regrouping': None,
        'comm': None,
        'pop_reduce': None,  
        }

    de_task = Optimizer(config, None, import_setting=de, strategy_mode='Given')
    shade_task = Optimizer(config, None, import_setting=shade, strategy_mode='Given')
    # dmspso_task = Optimizer(config, None, import_setting=dmspso, strategy_mode='Given')
    # ga_task = Optimizer(config, None, import_setting=ga, strategy_mode='Given')
    madde_task = Optimizer(config, None, import_setting=madde, strategy_mode='Given')
    nlshadelbc_task = Optimizer(config, None, import_setting=nl_shade_lbc, strategy_mode='Given')
    # return shade_task, dmspso_task, ga_task
    return de_task, shade_task, madde_task#, nlshadelbc_task
    # trainproblem, testproblem = make_dataset(config)

    # shade_tasks, dmspso_tasks, ga_tasks = [], [], []
    # for p in testproblem:
    #     task = copy.deepcopy(shade_task)
    #     task.problem = p
    #     shade_tasks.append(task)

    # for p in testproblem:
    #     task = copy.deepcopy(dmspso_task)
    #     task.problem = p
    #     dmspso_tasks.append(task)

    # for p in testproblem:
    #     task = copy.deepcopy(ga_task)
    #     task.problem = p
    #     ga_tasks.append(task)

    # return shade_tasks, dmspso_tasks, ga_tasks


def make_dataset(config):
    return bbob.BBOB_Dataset.get_datasets(upperbound=config.Xmax, Dim=config.dim)


def make_hpob_dataset(config):
    problem_set = []
    meta_train_data,meta_vali_data,meta_test_data,bo_initializations,surrogates_stats=get_data(root_dir="env/HPO-B-main/hpob-data/", mode="v3-test", surrogates_dir="env/HPO-B-main/saved-surrogates/")
    for search_space_id in meta_test_data.keys():
        for dataset_id in meta_test_data[search_space_id].keys():
            bst_model,y_min,y_max=get_bst(surrogates_dir='env/HPO-B-main/saved-surrogates/',search_space_id=search_space_id,dataset_id=dataset_id,surrogates_stats=surrogates_stats)
            X = np.array(meta_test_data[search_space_id][dataset_id]["X"])
            y = np.array(meta_test_data[search_space_id][dataset_id]["y"])
            dim = X.shape[1]
            p=Problem_hpo(bst_surrogate=bst_model,dim=dim,y_min=y_min,y_max=y_max,config=config)
            problem_set.append(p)
    return problem_set


def make_taskset(config):
    trainproblem, testproblem = make_dataset(config)

    # traintask, testtask = [], []
    traintask, testtask = get_usual_tasks(config)
    
    ret_train, ret_test = [], []

    set_seed(config.traindata_seed)
    for i in range(len(trainproblem)):
        for j in range(len(traintask)):
            task = Optimizer(copy.deepcopy(config), copy.deepcopy(trainproblem[i]), copy.deepcopy(traintask[j]))
            ret_train.append(task)

    set_seed(config.testdata_seed)
    for i in range(len(testproblem)):
        for j in range(len(testtask)):
            task = Optimizer(copy.deepcopy(config), copy.deepcopy(testproblem[i]), copy.deepcopy(testtask[j]))
            ret_test.append(task)

    return Taskset.get_datasets(ret_train, ret_test, config.batch_size, config.test_batch_size)
    

def make_taskset4test(config):
    trainproblem, testproblem = make_dataset(config)
    train, validate = make_taskset(config, True)
    test = []
    for p in testproblem:
        for i in range(len(train)):
            task = copy.deepcopy(train.data[i])
            task.problem = p
            test.append(task)
    return validate, Taskset(test, config.test_batch_size)


def make_all_taskset(config):
    trainproblem, testproblem = make_dataset(config)

    # with open('op_features.pkl', 'rb') as f:
    #     op_feature = pickle.load(f)

    set_seed(config.dataseed)
    traintask, testtask = get_usual_tasks(config)

    # train task + train prob
    trtr = []
    for i in range(len(trainproblem)):
        for j in range(len(traintask)):
            task = copy.deepcopy(traintask[j])
            task.problem = trainproblem[i]
            trtr.append(task)

    # train task + test prob
    trte = []
    for i in range(len(testproblem)):
        for j in range(len(traintask)):
            task = copy.deepcopy(traintask[j])
            task.problem = testproblem[i]
            trte.append(task)
    
    # test task + train prob
    tetr = []
    for i in range(len(trainproblem)):
        for j in range(len(testtask)):
            task = copy.deepcopy(testtask[j])
            task.problem = trainproblem[i]
            tetr.append(task)

    # test task + test prob
    tete = []
    for i in range(len(testproblem)):
        for j in range(len(testtask)):
            task = copy.deepcopy(testtask[j])
            task.problem = testproblem[i]
            tete.append(task)

    return Taskset(trtr, config.batch_size), Taskset(trte, config.batch_size), Taskset(tetr, config.batch_size), Taskset(tete, config.batch_size)


def make_protein_taskset(config):
    # trainproblem, testproblem = make_dataset(config)
    _, protein_problem = Protein_Docking_Dataset.get_datasets()
    # print(len(protein_problem))
    set_seed(config.dataseed)
    traintask, testtask = get_usual_tasks(config)

    # train task + protein prob
    trpr = []
    for i in range(len(protein_problem)):
        for j in range(len(traintask)):
            task = copy.deepcopy(traintask[j])
            task.problem = protein_problem[i]
            trpr.append(task)

    # test task + protein prob
    tepr = []
    for i in range(len(protein_problem)):
        for j in range(len(testtask)):
            task = copy.deepcopy(testtask[j])
            task.problem = protein_problem[i]
            tepr.append(task)
    
    return Taskset(trpr, config.batch_size), Taskset(tepr, config.batch_size)


def make_hpob_taskset(config):
    problem_set = make_hpob_dataset(config)
    set_seed(config.dataseed)
    traintask, testtask = get_usual_tasks(config)

    # train task + hpob prob
    trhp = []
    for i in range(len(problem_set)):
        for j in range(len(traintask)):
            task = copy.deepcopy(traintask[j])
            task.problem = problem_set[i]
            trhp.append(task)

    # test task + hpob prob
    tehp = []
    for i in range(len(problem_set)):
        for j in range(len(testtask)):
            task = copy.deepcopy(testtask[j])
            task.problem = problem_set[i]
            tehp.append(task)
    
    return Taskset(trhp, config.batch_size), Taskset(tehp, config.batch_size)


def make_real_taskset(config):
    tasks = [0, 0, 0]
    tasks[0], tasks[1], tasks[2] = get_real_tasks(config)  # shade, dmspso, ga
    bbob_train, bbob_test = make_dataset(config)
    _, protein_test = Protein_Docking_Dataset.get_datasets()
    # hpob_test = make_hpob_dataset(config)

    bbob_tr, bbob_te, protein_te, hpob_te = [], [], [], []
    for i in range(len(bbob_train)):
        for j in range(len(tasks)):
            task = copy.deepcopy(tasks[j])
            task.problem = bbob_train[i]
            bbob_tr.append(task)

    for i in range(len(bbob_test)):
        for j in range(len(tasks)):
            task = copy.deepcopy(tasks[j])
            task.problem = bbob_test[i]
            bbob_te.append(task)

    for i in range(len(protein_test)):
        for j in range(len(tasks)):
            task = copy.deepcopy(tasks[j])
            task.problem = protein_test[i]
            protein_te.append(task)

    # for i in range(len(hpob_test)):
    #     for j in range(len(tasks)):
    #         task = copy.deepcopy(tasks[j])
    #         task.problem = hpob_test[i]
    #         hpob_te.append(task)

    return Taskset(bbob_tr, config.test_batch_size), Taskset(bbob_te, config.test_batch_size), Taskset(protein_te, config.test_batch_size)#, Taskset(hpob_te, config.batch_size)

# use_qbest, united_qbest, use_archive, archive_id, united_id

def make_heat_taskset(config):
    # op, op+qb, op+arc,op+qb+arc
    # mu
    b2 = {'class': 'best2', 
           'args': (False, False, False, [], []), 
           'param_ada': {
               'F1': 0.5,
               'F2': 0.5, 
            }
          }
    r2 = {'class': 'rand2', 
           'args': (False, False, False, [], []), 
           'param_ada': {
               'F1': 0.5,
               'F2': 0.5, 
            }
          }
    c2r = {'class': 'current2rand', 
           'args': (False, False, False, [], []), 
           'param_ada': {
               'F1': 0.5,
               'F2': 0.5, 
            }
          }
    c2b = {'class': 'current2best', 
           'args': (False, False, False, [], []), 
           'param_ada': {
               'F1': 0.5,
               'F2': 0.5, 
            }
          }
    r2b = {'class': 'rand2best', 
           'args': (False, False, False, [], []), 
           'param_ada': {
               'F1': 0.5,
               'F2': 0.5, 
            }
          }
    
    bino = {'class': 'binomial', 
            'args': (False, False, False, [], []),
            'param_ada': {
                'Cr': 0.9, 
            }
            }
    expo = {'class': 'exponential', 
            'args': (False, False, False, [], []),
            'param_ada': {
                'Cr': 0.9, 
            }
            }
    
    mus = [b2, r2, c2b, c2r, r2b]
    crss = [bino, expo]

    de = {'sample': 'uniform',
        'Npop': 1,
        'NPmax': [100],
        'NPmin': [100],
        'Vmax': 0,
        'NA': 1.0,
        'grouping': None, 
        'regroup': 0, 
        'arch_replace': 'rand',
        'global_strategy': {
        # 'SHA_F': {'class':'SHA', 'args': (100, 'Cauchy', 0.5, -1, 0.1, [0, 1], 'regenerate', 'regenerate', 2, 1, None)},
        # 'SHA_Cr': {'class':'SHA', 'args': (100, 'Normal', 0.5, -1, 0.1, [0, 1], 'regenerate', 'regenerate', 2, 1, None)},
        }, 
        'subpops': [
            {
            'ops': 
            [
            # {'class': 'current2best', 
            # 'args': (True, False, True, [1,], []), 
            # 'param_ada': {
            #     'F1': 'SHA_F',
            #     'F2': 'F1', 
            #     'q': {'class': 'Bound_rand', 'args': (0, 0.2)}
            #     }
            # },
            # {'class': 'binomial', 
            #     'args': (False, False, False, [], []),
            #     'param_ada': {
            #     'Cr': 'SHA_Cr', 
            #     }
            # },
            ], 
            'bounding': 'clip',
            'selection': 'direct',
            'comm': None,
            'pop_reduce': None,  
            }, 
        ],
        'regrouping': None,
        }
    # use_qbest, united_qbest, use_archive, archive_id, united_id

    des = []
    for i in range(4):
        crs = copy.deepcopy(crss)
        if i == 1:
            crs[0]['args'] = (True, False, False, [], [])
            crs[0]['param_ada']['q'] = 0.2
            crs[1]['args'] = (True, False, False, [], [])
            crs[1]['param_ada']['q'] = 0.2
        elif i == 2:
            crs[0]['args'] = (False, False, True, [], [])
            crs[1]['args'] = (False, False, True, [], [])
        elif i == 3:
            crs[0]['args'] = (True, True, True, [], [])
            crs[0]['param_ada']['q'] = 0.2
            crs[1]['args'] = (True, True, True, [], [])
            crs[1]['param_ada']['q'] = 0.2

        # op
        for m in mus:
            for c in crs:
                dei = copy.deepcopy(de)
                dei['subpops'][0]['ops'].append(m)
                dei['subpops'][0]['ops'].append(c)
                des.append(dei)

        # op+qb
        for m in mus:
            mi = copy.deepcopy(m)
            mi['args'] = (True, False, False, [], [])
            mi['param_ada']['q'] = 0.2
            for c in crs:
                dei = copy.deepcopy(de)
                dei['subpops'][0]['ops'].append(mi)
                dei['subpops'][0]['ops'].append(c)
                des.append(dei)

        # op+arc
        for m in mus:
            mi = copy.deepcopy(m)
            mi['args'] = (False, False, True, [1,], [])
            for c in crs:
                dei = copy.deepcopy(de)
                dei['subpops'][0]['ops'].append(mi)
                dei['subpops'][0]['ops'].append(c)
                des.append(dei)

        # op+qb+arc
        for m in mus:
            mi = copy.deepcopy(m)
            mi['args'] = (True, True, True, [], [1,])
            mi['param_ada']['q'] = 0.2
            for c in crs:
                dei = copy.deepcopy(de)
                dei['subpops'][0]['ops'].append(mi)
                dei['subpops'][0]['ops'].append(c)
                des.append(dei)

    bbob_train, bbob_test = make_dataset(config)
    data = []
    for d in des:
        for p in bbob_train:
            data.append(Optimizer(config, p, import_setting=d, strategy_mode='Given'))
        for p in bbob_test:
            data.append(Optimizer(config, p, import_setting=d, strategy_mode='Given'))

    return Taskset(data, config.batch_size)


def make_op_pool(config):
    poolpath = config.pool_path
    if config.pool_path is None:
        run_name = config.run_name
        poolpath = 'pool_' + run_name + '.npy'
        
    else:
        pool = np.load(poolpath, allow_pickle=True).item()  # dict{'name': id}
        return pool


def make_algorithms_mamba(config, target_space=[2, 4, 6, 8, 10, 12, 13, 14, 15, 16]):
    set_seed(config.dataseed)
    def check_space(o):
        space = o.get_config_space()
        return len(space.keys())
    problems, test = make_dataset(config)
    tasks = []
    test_tasks = []
    algorithms = []
    existing = []
    size = len(target_space)
    for i in range(size):
        # print(i)
        p = None
        o = Optimizer(config, p, )
        omin = copy.deepcopy(o)
        c = 0
        while check_space(o) < target_space[i] or (i+1 < size and check_space(o) >= target_space[i+1]) or o.get_config_space() in existing or check_space(o)>16:
            o = Optimizer(config, p, )
            if check_space(o) < check_space(omin):
                omin = copy.deepcopy(o)
            c += 1
            if c > 10000:
                print('break')
                break
        if c > 10000:
            o = omin
        algorithms.append(o)
        existing.append(o.get_config_space())
        # print(o.get_config_space())
        # print(o.ops)
        # print(len(o.get_config_space()))
        for p in problems:
            oi = copy.deepcopy(o)
            oi.problem = copy.deepcopy(p)
            tasks.append(oi)
        for p in test:
            oi = copy.deepcopy(o)
            oi.problem = copy.deepcopy(p)
            test_tasks.append(oi)
    with open('algorithm_set_for_mamba.pkl', 'wb') as f:
        pickle.dump(algorithms, f)
    with open('algorithm_set_for_mamba_test.pkl', 'wb') as f:
        pickle.dump(test_tasks, f)
    with open('task_set_for_mamba.pkl', 'wb') as f:
        pickle.dump(tasks, f)
    return algorithms, tasks, test_tasks
        

def make_mamba_taskset(config):
    _, train, test = make_algorithms_mamba(config)
    return Taskset(train[:16]*4, 64), Taskset(test[:8], 8)