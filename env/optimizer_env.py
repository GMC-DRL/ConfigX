from sqlite3 import NotSupportedError
import time
import copy
import gym
from gym import spaces
import numpy as np
from scipy.spatial.distance import cdist
from components.operators import *
from components.Population import Population


class Optimizer(gym.Env):
    def __init__(self, config, problem, modules, strategy_mode='RL', seed=None) -> None:

        self.config = config
        self.problem = problem
        # self.MaxFEs = self.config.MaxFEs
        self.MaxGen = self.config.MaxGen
        self.skip_step = config.skip_step
        self.strategy_mode = strategy_mode

        # ---------------------------- init some variables may needed ---------------------------- #
        self.op_strategy = None
        self.global_strategy = {}
        self.bound_strategy = []
        self.select_strategies = []
        self.regroup_strategy = None
        self.restart_strategies = []

        self.rng_seed = seed
        self.rng = np.random
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.rng = np.random.RandomState(seed)

        self.trng = torch.random.get_rng_state()

        pe_id = 0
        self.init_mod = modules[0]
        self.init_mod.pe_id = pe_id
        pe_id += 1
        self.nich_mod = None
        self.modules = modules[1:]  # ignore init
        self.Npop = 1
        if isinstance(modules[1], Niching):
            self.Npop = modules[1].Npop
            self.nich_mod = modules[1]
            self.nich_mod.pe_id = pe_id
            pe_id += 1
            self.modules = modules[2:]  # ignore init and niching
        self.NPmax = [self.rng.choice(self.config.NPmax) for _ in range(self.Npop)]
        self.NPmin = [self.rng.choice(self.config.NPmin) for _ in range(self.Npop)]
        self.NA = self.rng.choice(self.config.NA)
        self.Vmax = self.rng.choice(self.config.Vmax)
        self.arch_replace = self.rng.choice(['oldest', 'worst', 'rand'])
        self.n_component = 0
        self.n_control = 0
        self.n_subs = np.zeros(self.Npop)
        for i, mod in enumerate(self.modules):
            if isinstance(mod, list):
                for j, submod in enumerate(mod):
                    self.modules[i][j].pe_id = pe_id
                    pe_id += 1
                    if isinstance(submod, Controllable):
                        self.modules[i][j].act_index = self.n_control
                        self.n_control += 1
                    self.n_component += 1
                    self.n_subs[i] += 1
            else:
                self.modules[i].pe_id = pe_id
                pe_id += 1
                if isinstance(mod, Controllable):
                    self.modules[i].act_index = self.n_control
                    self.n_control += 1
                self.n_component += 1
        # -------------------------------- ob space -------------------------------- #
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_control, self.config.maxAct, ),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(0, 100, shape=(self.n_component, self.config.maxAct * 2))

    def seed(self, seed=None):
        self.rng_seed = seed
        np.random.seed(seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        self.trng = torch.random.get_rng_state()

    def reset(self):
        self.problem.reset()
        self.population = Population(self.problem,
                                     NPmax=self.NPmax, 
                                     NPmin=self.NPmin, 
                                     NA=self.NA,
                                     Xmax=self.config.Xmax, 
                                     Vmax=self.Vmax, 
                                     multiPop=self.Npop, 
                                     arch_replace=self.arch_replace,
                                     MaxGen=self.config.MaxGen,
                                     rng=self.rng,
                                     )
        self.population = self.init_mod(self.population, self.rng, self.rng_seed)
        if self.nich_mod is not None:
            self.population = self.nich_mod(self.population, self.rng)
        self.gbest = self.pre_gb = self.init_gb = self.population.gbest  # For reward
        self.stag_count = 0
        self.FEs = self.population.NP
        self.step_count = 0
        return self.get_state()
            
    def cal_feature(self, group, cost, gbest, gbest_solution, cbest, cbest_solution):
        features = [] # 9

        gbest_ = np.log10(max(1e-8, gbest) + 0)
        cbest_ = np.log10(max(1e-8, cbest) + 0)
        cost[cost < 1e-8] = 1e-8
        cost_ = np.log10(cost + 0)
        init_max = np.log10(self.population.init_max + 0)
        features.append(gbest_ / init_max)
        features.append(cbest_ / init_max)
        features.append(np.mean(cost_ / init_max))
        features.append(np.std(cost_ / init_max))

        dist = np.sqrt(np.sum((group[None,:,:] - group[:,None,:]) ** 2, -1))
        features.append(np.max(dist) / (self.population.Xmax - self.population.Xmin) / np.sqrt(self.problem.dim))
        top10 = np.argsort(cost)[:int(max(1, 0.1*len(cost)))]
        dist10 = np.sqrt(np.sum((group[top10][None,:,:] - group[top10][:,None,:]) ** 2, -1))
        features.append((np.mean(dist10) - np.mean(dist)) / (self.population.Xmax - self.population.Xmin) / np.sqrt(self.problem.dim))

        # FDC
        d_lbest = np.sqrt(np.sum((group - gbest_solution) ** 2, -1))
        c_lbest = cost - gbest
        features.append(np.mean((c_lbest - np.mean(c_lbest)) * (d_lbest - np.mean(d_lbest))) / (np.std(c_lbest) * np.std(d_lbest) + 0.00001))
        d_cbest = np.sqrt(np.sum((group - cbest_solution) ** 2, -1))
        c_cbest = cost - cbest
        features.append(np.mean((c_cbest - np.mean(c_cbest)) * (d_cbest - np.mean(d_cbest))) / (np.std(c_cbest) * np.std(d_cbest)+ 0.00001))

        # features.append((self.MaxFEs - self.FEs) / self.MaxFEs)
        # features = []
        features.append((self.MaxGen - self.step_count) / self.MaxGen)
        
        features = torch.tensor(features)

        return features

    def get_state(self):
        states = []
        for i, ops in enumerate(self.modules):
            local_state = self.cal_feature(self.population.group[i], 
                                           self.population.cost[i], 
                                           self.population.lbest[i], 
                                           self.population.lbest_solution[i], 
                                           self.population.cbest, 
                                           self.population.cbest_solution)
            for io, op in enumerate(ops):
                if isinstance(op, Uncontrollable):
                    continue
                if self.config.morphological:
                    states.append(torch.concat((torch.tensor([op.pe_id]), op.get_id(), local_state), -1))
                else:
                    states.append(local_state)

        states = torch.stack(states)
        if states.shape[0] < self.config.maxCom:  # mask 
            mask = torch.zeros(self.config.maxCom - states.shape[0], states.shape[-1])
            states = torch.concat((states, mask), 0)
        return states

    def cal_symbol_feature(self, group, cost, gbest, gbest_solution, cbest, cbest_solution):
        def dist(x,y):
            return np.sqrt(np.sum((x-y)**2,axis=-1))
        fea_1=(cost-gbest)/(self.init_gb-gbest+1e-8)
        fea_1=np.mean(fea_1)
        
        distances = cdist(group, group, metric='euclidean')
        np.fill_diagonal(distances, 0)
        mean_distance = np.mean(distances)
        fea_2=mean_distance/np.sqrt(self.config.Xmax*2*self.problem.dim)

        fit=np.zeros_like(cost)
        fit[:group.shape[0]//2]=self.init_gb
        fit[group.shape[0]//2:]=self.gbest
        maxstd=np.std(fit)
        fea_3=np.std(cost)/(maxstd+1e-8)

        fea_4=(self.MaxGen-self.step_count)/self.MaxGen

        fea_5=self.stag_count/self.MaxGen
        
        fea_6=dist(group,cbest_solution[None,:])/np.sqrt(self.config.Xmax*2*self.problem.dim)
        fea_6=np.mean(fea_6)

        fea_7=(cost-cbest)/(self.init_gb-gbest+1e-8)
        fea_7=np.mean(fea_7)

        fea_8=dist(group,gbest_solution[None,:])/np.sqrt(self.config.Xmax*2*self.problem.dim)
        fea_8=np.mean(fea_8)

        fea_9=0
        if self.gbest<self.pre_gb:
            fea_9=1
        feature=np.array([fea_1,fea_2,fea_3,fea_4,fea_5,fea_6,fea_7,fea_8,fea_9])
        return feature

    def get_symbol_state(self):
        global_state = self.cal_symbol_feature(np.concatenate(self.population.group),
                                        np.concatenate(self.population.cost), 
                                        self.population.gbest, 
                                        self.population.gbest_solution, 
                                        self.population.cbest, 
                                        self.population.cbest_solution)

        return global_state

    def get_reward(self):
        return (self.pre_gb - self.gbest) / self.init_gb * 10

    def get_config_space(self, name_only=False):  # for SMAC3
        space = {}
        for ip in range(self.Npop):
            for io, op in enumerate(self.ops[ip]):
                name = f'{ip}_{io}_'
                if isinstance(op, Multi_strategies):
                    space[name + 'multi_op'] = np.arange(len(op.ops)).tolist()
                    for iso, subop in enumerate(op.ops): 
                        if name_only:
                            space[name+f'{iso}'] = subop
                            continue
                        for param in subop.control_param:
                            if param['type'] == 'float':
                                space[name+f'{iso}_'+param['name']] = (float(param['range'][0]), float(param['range'][1]))
                            else:
                                space[name+f'{iso}_'+param['name']] = np.arange(len(param['range'])).tolist()
                else:
                    if name_only:
                        space[name] = op
                        continue
                    for param in op.control_param:
                        if param['type'] == 'float':
                            space[name+param['name']] = (float(param['range'][0]), float(param['range'][1]))
                        else:
                            space[name+param['name']] = np.arange(len(param['range'])).tolist()
            if self.enable_bc:
                space[f'bound_{ip}'] = self.boundings[ip] if name_only else np.arange(len(self.boundings[ip].control_param[0]['range'])).tolist()
        # space['select'] = np.arange(len(self.selection.control_param[0]['range'])).tolist()
        if self.reg_id is None:
            # Communication
            if self.Npop > 1: 
                for ip in range(self.Npop):
                    space[f'comm_{ip}'] = self.Comms[ip] if name_only else np.arange(self.Npop).tolist()
        if self.reg_id is not None:
            space[f'regroup'] = self.regrouping if name_only else np.arange(self.regrouping.nop).tolist()   
        return space
    
    def set_config_space(self, config, op_index=False):  # for SMAC3
        for ip in range(self.Npop):
            for io, op in enumerate(self.ops[ip]):
                name = f'{ip}_{io}_'
                if isinstance(op, Multi_strategies):
                    op.default_op = config[name + 'multi_op']
                    for iso, subop in enumerate(op.ops):
                        if op_index:
                            op.ops[iso] = copy.deepcopy(config[op.ops[iso].op_id])
                            continue
                        for param in subop.control_param:
                            if param['type'] == 'float':
                                param['default'] = config[name+f'{iso}_'+param['name']]
                            else:
                                param['default'] = param['range'][config[name+f'{iso}_'+param['name']]]
                else:
                    if op_index:
                        self.ops[ip][io] = copy.deepcopy(config[self.ops[ip][io].op_id])
                        continue
                    for param in op.control_param:
                        if param['type'] == 'float':
                            param['default'] = config[name+param['name']]
                        else:
                            param['default'] = param['range'][config[name+param['name']]]
            if self.enable_bc:
                if op_index:
                    self.boundings[ip] = copy.deepcopy(config[self.boundings.op_id])
                else:
                    self.boundings[ip].control_param[0]['default'] = self.boundings[ip].control_param[0]['range'][config[f'bound_{ip}']]

            if self.reg_id is None and self.Npop > 1:
                if op_index:
                    self.Comms[ip] = copy.deepcopy(config[self.comm.op_id])
                else:
                    self.Comms[ip].default_action = config[f'comm_{ip}']

        # self.selection.control_param[0]['default'] = config['select']
        if self.reg_id is not None:
            if op_index:
                self.regrouping = copy.deepcopy(config[self.regrouping.op_id])
            else:
                self.regrouping.default_action = self.regrouping.ops[config['regroup']]

    def given_multi_op_action(self, multi_op, strategy, size, ratio, rng=None):
        if rng is None:
            rng = np.random

        def manage_adaptive(method, sha_id=None):
            if isinstance(method, SHA):  # use successful history adaption
                act = method.get(size, ids=sha_id, rng=rng)
            elif isinstance(method, jDE):  # use adaption proposed in jDE
                act = method.get(size)
            elif isinstance(method, Linear):  # use linearly changing value(s)
                act = method.get(ratio, size)
            elif isinstance(method, Bound_rand):  # use random value(s)
                act = method.get(size, rng=rng)
            else:
                raise NotSupportedError
            return act
        
        actions = {}
        op_selection = strategy['op_select'].get(size, rng=rng)
        actions['op_select'] = op_selection.copy()
        actions['op_list'] = []
        for io, op in enumerate(multi_op.ops):
            actions['op_list'].append({})
            for param in op.control_param:
                name = param['name']
                param_ada = strategy['op_list'][io][name]
                if isinstance(param_ada, Param_adaption):
                    act = manage_adaptive(param_ada)

                elif param_ada in actions['op_list'][-1].keys():  # use the same value as another param
                    act = actions['op_list'][-1][param_ada].copy()

                elif param_ada in self.global_strategy.keys():  # use a global shared self-adaptive method
                    if param_ada in self.global_strategy['message'].keys():  # use the shared values which have already been generated
                        act = self.global_strategy['message'][param_ada].copy()
                    else:
                        sha_id = None  # use the same successful history adaption memory indices, such as F anc Cr usually share the same indices
                        if isinstance(self.global_strategy[param_ada], SHA):
                            if 'SHA_id' not in self.global_strategy['message'].keys():
                                sha_id = self.global_strategy[param_ada].get_ids(size, rng=rng)
                                self.global_strategy['message']['SHA_id'] = sha_id
                            else:
                                sha_id = self.global_strategy['message']['SHA_id']
                        act = manage_adaptive(self.global_strategy[param_ada], sha_id)

                else:  # use the spercified value, could be float, int, or other user spercified types
                    act = param_ada
                if isinstance(act, np.ndarray):
                    act[op_selection != io] = 0
                if isinstance(param_ada, Param_adaption):
                    param_ada.history_value = act.copy()
                actions['op_list'][-1][name] = act
        return actions

    def given_action(self, op, strategy, size, ratio, rng=None):
        if rng is None:
            rng = np.random

        def manage_adaptive(method, sha_id=None):
            if isinstance(method, SHA):  # use successful history adaption
                act = method.get(size, ids=sha_id, rng=rng)
            elif isinstance(method, jDE):  # use adaption proposed in jDE
                act = method.get(size)
            elif isinstance(method, Linear):  # use linearly changing value(s)
                act = method.get(ratio, size)
            elif isinstance(method, Bound_rand):  # use random value(s)
                act = method.get(size, rng=rng)
            elif isinstance(method, DMS):  # use DMS-PSO
                act = method.get(ratio, size)
            else:
                raise NotSupportedError
            return act

        actions = {}
        for param in op.control_param:
            name = param['name']
            param_ada = strategy[name]
            if isinstance(param_ada, Param_adaption):
                act = manage_adaptive(param_ada)

            elif param_ada in strategy.keys():  # use the same value as another param
                act = actions[param_ada]

            elif param_ada in self.global_strategy.keys():  # use a global shared self-adaptive method
                if param_ada in self.global_strategy['message'].keys():  # use the shared values which have already been generated
                    act = self.global_strategy['message'][param_ada]
                else:
                    sha_id = None  # use the same successful history adaption memory indices, such as F anc Cr usually share the same indices
                    if isinstance(self.global_strategy[param_ada], SHA):
                        if 'SHA_id' not in self.global_strategy['message'].keys():
                            sha_id = self.global_strategy[param_ada].get_ids(size, rng=rng)
                            self.global_strategy['message']['SHA_id'] = sha_id
                        else:
                            sha_id = self.global_strategy['message']['SHA_id']
                    act = manage_adaptive(self.global_strategy[param_ada], sha_id)

            else:  # use the spercified value, could be float, int, or other user spercified types
                act = param_ada

            actions[name] = act
        return actions

    def self_strategy_reset(self):
        for ip in range(self.Npop):
            for io, op in enumerate(self.ops[ip]):
                if isinstance(self.ops[ip][io], Multi_strategies):
                    # self.ops[ip][io].reset()
                    if self.op_strategy is not None:
                        self.op_strategy[ip][io]['op_select'].reset()
                        for sop in self.op_strategy[ip][io]['op_list']:
                            for stra in sop.values():
                                if isinstance(stra, Param_adaption):
                                    stra.reset()
                elif self.op_strategy is not None:
                    for stra in self.op_strategy[ip][io].values():
                        if isinstance(stra, Param_adaption):
                            stra.reset()

    def self_strategy_update(self, old_population):
        # update local strategy for each operator
        subpops = self.population.get_subpops()
        old_subpops = old_population.get_subpops()
        for ip in range(self.Npop):
            for io, op in enumerate(self.ops[ip]):
                if isinstance(self.ops[ip][io], Multi_strategies):
                    # self.ops[ip][io].update(old_subpops[ip]['cost'], subpops[ip]['cost'], self.step_count/self.MaxGen)  # default strategy
                    if self.op_strategy is not None:  # given strategy
                        self.op_strategy[ip][io]['op_select'].update(old_subpops[ip]['cost'], subpops[ip]['cost'], self.step_count/self.MaxGen)
                        for sop in self.op_strategy[ip][io]['op_list']:
                            for stra in sop.values():
                                if isinstance(stra, Param_adaption):
                                    stra.update(old_subpops[ip]['cost'], subpops[ip]['cost'], self.step_count/self.MaxGen)
                elif self.op_strategy is not None:
                    for stra in self.op_strategy[ip][io].values():
                        if isinstance(stra, Param_adaption):
                            stra.update(old_subpops[ip]['cost'], subpops[ip]['cost'], self.step_count/self.MaxGen)

        # update global strategies
        for k in self.global_strategy.keys():
            if k == 'message':
                self.global_strategy[k] = {}
                continue
            self.global_strategy[k].update(old_population.cost, self.population.cost, self.step_count/self.MaxGen)

    def step(self, logits):
        rewards = 0
        state,reward,is_end,info = self.one_step(logits)
        rewards += reward
        if self.skip_step < 2:
            return state,rewards,is_end,info
        for t in range(1, self.skip_step):
            _,reward,is_end,_ = self.one_step([None]*logits.shape[0], had_action=info['had_action'])
            rewards += reward
        return self.get_state(),rewards,is_end,info

    def one_step(self, logits, had_action=None):
        torch.random.set_rng_state(self.trng)
        pre_state = self.get_symbol_state()
        old_population = copy.deepcopy(self.population)

        if had_action is None:
            had_action = [None for _ in range(self.n_control)]
        had_action_rec = [None for _ in range(self.n_control)]
        # reproduction
        action_values = [[] for _ in range(self.n_control)]
        logp_values = 0
        entropys = []
        syn_bar = np.zeros(self.Npop)
        while not (syn_bar >= self.n_subs).all():
            for ip in range(self.Npop):
                self.population.process_ip = ip
                # for io, op in enumerate(self.modules[ip]):
                st = int(syn_bar[ip])
                for io in range(st, int(self.n_subs[ip])):
                    op = self.modules[ip][io]
                    # subpops[ip]['problem'] = self.problem
                    if op.sym_bar_require and io > syn_bar[ip]:
                        syn_bar[ip] = io
                        break
                    syn_bar[ip] += 1
                    if isinstance(op, Controllable):
                        if self.strategy_mode == 'Given':
                            pass
                        elif self.strategy_mode == 'RL':
                            act_index = op.act_index
                            res = op(logits[act_index], self.population, softmax=self.config.softmax, rng=self.rng, had_action=had_action[act_index])
                            # print(ip, io, i_component, op)
                            if had_action[act_index] is None:
                                action_values[act_index] = res['actions']
                                logp_values += np.sum(res['logp'])
                                entropys += res['entropy']
                                had_action_rec[act_index] = res['had_action']
                        else:
                            res = op(None, self.population, softmax=self.config.softmax, rng=self.rng)
                        self.population = res['result']
                    else:
                        self.population = op(self.population)
        self.population.update_subpops()

        # update self-adaptive strategies
        if not self.strategy_mode == 'RL':
            self.self_strategy_update(old_population)

        self.step_count += 1

        self.pre_gb = self.gbest
        if self.gbest > self.population.gbest:
            self.gbest = min(self.gbest, self.population.gbest)
            self.stag_count = 0
        else:
            self.stag_count += 1

        # print(np.max(np.abs(self.population.group)))
        info = {}
        info['action_values'] = action_values
        info['logp'] = logp_values
        info['entropy'] = entropys
        info['gbest_val'] = self.population.gbest
        info['gbest_sol'] = self.population.gbest_solution
        info['init_gb'] = self.init_gb
        info['had_action'] = had_action_rec
        info['pre_state'] = pre_state
        info['nex_state'] = self.get_symbol_state()
        # is_done = self.FEs >= self.MaxFEs or self.population.gbest <= 1e-8
        is_done = self.step_count >= self.MaxGen or self.population.gbest <= 1e-8
        self.trng = torch.random.get_rng_state()

        return self.get_state(), self.get_reward(), is_done, info

    def action_interpret(self, logits, fixed_action):
        logp_values = 0
        entropys = []

        for ip in range(self.Npop):
            for op in self.modules[ip]:
                if isinstance(op, Controllable):
                    _, _, logp, entropy = action_interpret(op.config_space, logits[op.act_index], softmax=self.config.softmax, fixed_action=fixed_action[op.act_index])
                    logp_values += np.sum(logp)
                    entropys += entropy

        return logp_values, entropys
            
