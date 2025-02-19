import os
import json
import torch
import pprint
from tensorboardX import SummaryWriter
import warnings
from config import get_options
import numpy as np
from ppo import PPO
from utils.utils import set_seed
# DummyVectorEnv for Windows or Linux, SubprocVectorEnv for Linux
from env import DummyVectorEnv,SubprocVectorEnv
import platform


def load_agent(name):
    agent = {
        'ppo': PPO,
    }.get(name, None)
    assert agent is not None, "Currently unsupported agent: {}!".format(name)
    return agent

def run(opts):
    # only one mode can be specified in one time, test or train
    assert opts.train==None or opts.test==None, 'Between train&test, only one mode can be given in one time'
    
    sys=platform.system()
    opts.is_linux=True if sys == 'Linux' else False
    torch.multiprocessing.set_sharing_strategy('file_system')
    # figure out the max_fes(max function evaluation times), in our experiment, we use 20w for 10D problem and 100w for 30D problem

    # Pretty print the run args
    pprint.pprint(vars(opts))

    # Set the random seed to initialize the network
    set_seed(opts.seed)


    # Set the device, you can change it according to your actual situation
    # opts.device = torch.device("cuda:0")
    opts.device = torch.device("cpu")
           
    # Figure out the RL algorithm
    # if opts.is_linux:
    agent = PPO(opts,SubprocVectorEnv)
    # else:
    #     agent = PPO(opts,DummyVectorEnv)

    # Load data from load_path(if provided)
    if opts.load_name is not None:
        # opts.run_name = opts.load_name
        load_path = os.path.join(opts.load_path, opts.load_name)
        if opts.load_epoch is None:
            epoch_list = os.listdir(load_path)
            id_list = []
            for eid in epoch_list:
                id_list.append(int(eid[6:-3]))
            opts.load_epoch = np.max(id_list)
        
        load_path = os.path.join(load_path, f'epoch-{opts.load_epoch}.pt')
        agent.load(load_path)

    # Do validation only
    if opts.test:
        # Testing
        from utils.make_dataset import *
        from rollout import rollout
        if opts.load_name is not None:
            opts.run_name = opts.load_name
        opts.log_dir = os.path.join('rollout_outputs', opts.run_name)
        if not os.path.exists(opts.log_dir):
            os.makedirs(opts.log_dir)

        # Load the validation datasets
        training_dataloader, test_dataloader = make_taskset(opts)

        set_seed(opts.testseed)
        avg_best,sigma,rew=rollout(test_dataloader,opts, agent=agent, run_name=opts.run_name, epoch_id=opts.load_epoch)
        print(f'test: gbest_mean:{gbest_mean}, std:{std}, Rewards: {rew}')

    else:  
        # configure tensorboard
        path = os.path.join(opts.log_dir, opts.run_name)
        if not os.path.exists(path):
            os.makedirs(path)
        tb_logger = SummaryWriter(path)
  
        set_seed(opts.seed)
        # Start the actual training loop
        agent.start_training(tb_logger)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # main process
    run(get_options())
