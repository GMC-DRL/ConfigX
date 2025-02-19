import os
import time
import warnings
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from utils.utils import set_seed
from utils import clip_grad_norms
from nets.actor_network import Actor
from nets.critic_network import Critic
from utils import torch_load_cpu, get_inner_model
from utils.logger import log_to_tb_train, log_to_val
import copy
from utils.make_dataset import *
from rollout import rollout
from env.optimizer_env import Optimizer


# memory for recording transition during training process
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


def lr_sd(epoch, opts):
    return opts.lr_decay ** epoch


class PPO:
    def __init__(self, opts,vector_env):

        # figure out the options
        self.opts = opts
        # the parallel environment
        self.vector_env=vector_env
        # figure out the actor network
        self.actor = Actor(opts)
        
        if not opts.test:
            # for the sake of ablation study, figure out the input_dim for critic according to setting
            input_critic=opts.embedding_dim
            # figure out the critic network
            self.critic = Critic(
                input_dim = input_critic,
                hidden_dim1 = opts.hidden_dim1_critic,
                hidden_dim2 = opts.hidden_dim2_critic,
            )

            # figure out the optimizer
            self.optimizer = torch.optim.Adam(
                [{'params': self.actor.parameters(), 'lr': opts.lr_model}] +
                [{'params': self.critic.parameters(), 'lr': opts.lr_model}])
            # figure out the lr schedule
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, opts.lr_decay, last_epoch=-1,)

        # move to cuda
        self.actor.to(opts.device)
        if not opts.test:
            self.critic.to(opts.device)


    # load model from load_path
    def load(self, load_path):

        assert load_path is not None
        load_data = torch_load_cpu(load_path)

        # load data for actor
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('actor', {})})

        if not self.opts.test:
            # load data for critic
            model_critic = get_inner_model(self.critic)
            model_critic.load_state_dict({**model_critic.state_dict(), **load_data.get('critic', {})})
            # load data for optimizer
            self.optimizer.load_state_dict(load_data['optimizer'])
            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            # if self.opts.use_cuda:
            #     torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # done
        print(' [*] Loading data from {}'.format(load_path))

    # save trained model
    def save(self, epoch):
        print('Saving model and state...')
        run_name = self.opts.run_name
        if self.opts.single_task is not None:
            run_name += f"_single-{self.opts.single_task}"
        path = os.path.join(self.opts.save_dir, run_name)
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'critic': get_inner_model(self.critic).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(path, 'epoch-{}.pt'.format(epoch))
        )

    # change working mode to evaling
    def eval(self):
        torch.set_grad_enabled(False)  ##
        self.actor.eval()
        if not self.opts.test: self.critic.eval()

    # change working mode to training
    def train(self):
        torch.set_grad_enabled(True)  ##
        self.actor.train()
        if not self.opts.test: self.critic.train()


    def start_training(self, tb_logger):
        train(0, self, tb_logger)

    

# inference for training
def train(rank, agent, tb_logger):  
    print("begin training")
    opts = agent.opts
    warnings.filterwarnings("ignore")

    training_dataloader, test_dataloader = make_taskset(opts)
    # training_dataloader, test_dataloader = make_mamba_taskset(opts)
    # _, test_dataloader = make_taskset(opts)
    # training_dataloader = test_dataloader
    if opts.rew_train == 1:
        opts.trainsize = opts.trainsize//2
    if opts.rew_train == 2:
        opts.trainsize *= 2

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    # move optimizer's data onto chosen device
    for state in agent.optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(opts.device)

    # generatate the train_dataset and test_dataset
    
    best_epoch=None
    best_avg_best_cost=None
    best_avg_best_rew=None
    best_epoch_list=[]
    train_rew_list = []
    mean_per_list=[]
    sigma_per_list=[]
    rew_per_list=[]
    pre_step=0

    epoch = 0
    agent.save(epoch)
    avg_best,sigma,rew = 0, 0, 0
    # validate the new model
    if opts.problem == 'protein':
        training_dataloader, test_dataloader = make_protein_taskset(opts)
    if opts.single_task is None:
        avg_best,sigma,rew=rollout(test_dataloader,opts,agent,tb_logger,epoch_id=epoch)
    else:
        avg_best,sigma,rew=rollout(training_dataloader,opts,agent,tb_logger,epoch_id=epoch)


    mean_per_list.append(avg_best)
    sigma_per_list.append(sigma)
    rew_per_list.append(rew)
    if epoch==opts.epoch_start:
        best_avg_best_cost=avg_best
        best_avg_best_rew=rew
        best_epoch=epoch
    # elif avg_best<best_avg_best_cost:
    #     best_avg_best_cost=avg_best
    #     best_epoch=epoch
    elif rew>best_avg_best_rew:
        best_avg_best_rew=rew
        best_epoch=epoch
    best_epoch_list.append(best_epoch)
    print(f'mean_performance:{mean_per_list}')
    print(f'reward_performance:{rew_per_list}')
    print(f'sigma_performance:{sigma_per_list}')
    
    task_returns = np.zeros(opts.trainsize)
    task_weights = torch.ones(opts.trainsize)
    stop_training=False
    # Start the actual training loop
    for epoch in range(opts.epoch_start, opts.epoch_end):
        # Training mode
        # set_seed()
        agent.train()
        # agent.lr_scheduler_critic.step(epoch)
        # agent.lr_scheduler_actor.step(epoch)
        agent.lr_scheduler.step(epoch)
        task_return = np.zeros(opts.trainsize)
        if opts.shuffle:
            training_dataloader.shuffle()

        # logging
        if rank == 0:
            print('\n\n')
            print("|",format(f" Training epoch {epoch} ","*^60"),"|")
            print("Training with actor lr={:.3e} critic lr={:.3e} for run {}".format(agent.optimizer.param_groups[0]['lr'],
                                                                                     agent.optimizer.param_groups[1]['lr'], opts.run_name) , flush=True)

        # start training
        step = np.ceil(len(training_dataloader) / opts.batch_size)
        pbar = tqdm(total = step * opts.MaxGen // opts.skip_step * opts.K_epochs // opts.n_step,
                    desc = 'training',
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        for batch_id, batch in enumerate(training_dataloader):
            # if batch_id < 22:
            #     continue
            env_list=[lambda e=p: e for p in batch]
            envs=agent.vector_env(env_list)
            q_lengths = torch.zeros(len(batch))
            for i in range(len(batch)):
                q_lengths[i] = batch[i].n_component
            # train procedule for a batch
            tw_l = (batch_id*opts.batch_size)%opts.trainsize
            tw_r = ((batch_id+1)*opts.batch_size)%opts.trainsize
            if tw_r < 1:
                tw_r = opts.trainsize
            batch_step, batch_return=train_batch(rank,
                                   envs,
                                   agent,
                                   epoch,
                                   pre_step,
                                   task_weights[tw_l:tw_r],
                                   tb_logger,
                                   opts,
                                   q_lengths,
                                   batch_id, 
                                   pbar)
            envs.close()
            # pbar.update()
            pre_step += batch_step
            task_return[tw_l:tw_r] += batch_return
            # see if the learning step reach the max_learning_step, if so, stop training
            # if pre_step>=opts.max_learning_step:
            #     stop_training=True
            #     break
        pbar.close()
        task_return /= len(training_dataloader) // opts.trainsize

        if opts.weighted_start is not None and opts.single_task is None and (epoch-opts.epoch_start) >= opts.weighted_start - 1:
            if np.max(task_returns) < 1e-8:
                task_returns = task_return.copy()
            else:
                task_weights = torch.softmax(1 - torch.from_numpy(task_return - task_returns), 0) * opts.trainsize
                task_returns = task_return.copy()

        if (epoch-opts.epoch_start) %  opts.checkpoint_epochs == 0:
            agent.save(epoch+1)

        if (epoch-opts.epoch_start) % opts.update_best_model_epochs==0 or epoch == opts.epoch_end-1:
            # validate the new model
            if opts.single_task is None:
                avg_best,sigma,rew=rollout(test_dataloader,opts,agent,tb_logger,epoch_id=epoch+1)
            else:
                avg_best,sigma,rew=rollout(training_dataloader,opts,agent,tb_logger,epoch_id=epoch+1)
            mean_per_list.append(avg_best)
            sigma_per_list.append(sigma)
            rew_per_list.append(rew)
            if epoch+1==opts.epoch_start:
                best_avg_best_cost=avg_best
                best_avg_best_rew=rew
                best_epoch=epoch+1
            # elif avg_best<best_avg_best_cost:
            #     best_avg_best_cost=avg_best
            #     best_epoch=epoch+1
            elif rew>best_avg_best_rew:
                best_avg_best_rew=rew
                best_epoch=epoch+1
            best_epoch_list.append(best_epoch)

        # logging
        train_rew_list.append(np.mean(task_return))
        print('current_epoch:{}, best_epoch:{}'.format(epoch,best_epoch))
        print('best_epoch_list:{}'.format(best_epoch_list))
        print(f'train_reward:{train_rew_list}')
        print(f'mean_performance:{mean_per_list}')
        print(f'reward_performance:{rew_per_list}')
        print(f'sigma_performance:{sigma_per_list}')
        print(f'best_mean:{mean_per_list[(best_epoch-opts.epoch_start)  // opts.update_best_model_epochs]}')
        print(f'best_reward:{rew_per_list[(best_epoch-opts.epoch_start)  // opts.update_best_model_epochs]}')
        print(f'best_train:{np.argmax(train_rew_list)} {np.max(train_rew_list)}')
        print(f'best_std:{sigma_per_list[(best_epoch-opts.epoch_start)  // opts.update_best_model_epochs]}')
        
        
        if stop_training:
            print('Have reached the maximum learning steps')
            break
    print(best_epoch_list)


def train_batch(
        rank,
        problem,
        agent,
        epoch,
        pre_step,
        task_weights,
        tb_logger,
        opts,
        q_lengths,
        batch_id,
        pbar,
        ):
    
    # setup
    agent.train()
    memory = Memory()
    # initial instances and solutions
    batch_return = np.zeros(opts.batch_size)
    problem.seed(opts.seed + np.arange(opts.batch_size) + epoch*1024 + batch_id*opts.batch_size)

    trng = torch.random.get_rng_state()

    state=problem.reset()
    state=torch.FloatTensor(state).to(opts.device)
    
    torch.random.set_rng_state(trng)

    # params for training
    gamma = opts.gamma
    n_step = opts.n_step
    
    K_epochs = opts.K_epochs
    eps_clip = opts.eps_clip
    
    t = 0
    # initial_cost = obj
    done=False
    
    # sample trajectory
    while not done:
        t_s = t
        total_cost = 0
        entropy = []
        bl_val_detached = []
        bl_val = []

        # accumulate transition
        while t - t_s < n_step :  
            memory.states.append(state.clone())
            # memory.states.append(state)
            logits, _to_critic = agent.actor(state, 
                                             q_length=q_lengths,
                                             to_critic=True
                                             )
            if torch.sum(torch.isnan(logits)) > 0:
                exit()
            trng = torch.random.get_rng_state()

            next_state,rewards,is_end,info = problem.step(logits.detach().cpu())

            torch.random.set_rng_state(trng)
            if opts.weighted_start is not None and opts.single_task is None:
                batch_return += rewards
            action, entro_p = [], []
            log_lh = []
            for ifo in info:
                action.append(ifo['action_values'])
                log_lh.append(ifo['logp'])
                entro_p += ifo['entropy']
            # log_lh, _ = agent.actor.get_logp(logits, action)
            memory.actions.append(copy.deepcopy(action))
            memory.logprobs.append(torch.tensor(log_lh))
            # action=action.cpu().numpy()
            entropy += entro_p

            baseline_val_detached, baseline_val = agent.critic(_to_critic)
            bl_val_detached.append(baseline_val_detached)
            bl_val.append(baseline_val)

            # state transient
            memory.rewards.append(torch.FloatTensor(rewards).to(opts.device))
            # next
            t = t + 1
            state=torch.FloatTensor(next_state).to(opts.device)
            # state = copy.deepcopy(next_state)
            if is_end.all():
                done=True
                break
        # store info
        t_time = t - t_s
        total_cost = total_cost / t_time
        # begin update
        # old_actions = torch.stack(memory.actions)
        old_actions = memory.actions
        old_states = torch.stack(memory.states).detach() #.view(t_time, bs, ps, dim_f)
        # old_states = memory.states
        # old_actions = all_actions.view(t_time, bs, ps, -1)
        old_logprobs = torch.stack(memory.logprobs).detach().view(-1)
        
        # Optimize PPO policy for K mini-epochs:
        old_value = None
        for _k in range(K_epochs):
            # t1 = time.time()
            # t2 = t3 = t4 = 0
            if _k == 0:
                logprobs = memory.logprobs
                # ttt = time.time()
            else:
                # Evaluating old actions and values :
                logprobs = []
                entropy = []
                bl_val_detached = []
                bl_val = []
                for tt in range(t_time):
                    # ttt = time.time()
                    # get new action_prob
                    logits, _to_critic = agent.actor(old_states[tt],
                                                     q_length=q_lengths,
                                                     detach_state = True,
                                                     to_critic = True
                                                     )
                    # t2 += time.time() - ttt
                    # ttt = time.time()
                    log_p, entro_p = problem.action_interpret(logits.detach().cpu(), old_actions[tt])
                    # t3 += time.time() - ttt
                    # ttt = time.time()
                    # log_p, entro_p = agent.actor.get_logp(logits, old_actions[tt])
                    # t4 += time.time() - ttt
                    # ttt = time.time()
                    logprobs.append(torch.tensor(log_p))
                    entropy += entro_p

                    baseline_val_detached, baseline_val = agent.critic(_to_critic)

                    bl_val_detached.append(baseline_val_detached)
                    bl_val.append(baseline_val)
            logprobs = torch.stack(logprobs, 0).view(-1)
            entropy = torch.stack(entropy, 0).view(-1)
            # entropy = torch.tensor(entropy)
            bl_val_detached = torch.stack(bl_val_detached).view(-1)
            bl_val = torch.stack(bl_val).view(-1)


            # get traget value for critic
            Reward = []
            reward_reversed = memory.rewards[::-1]
            # get next value
            R = agent.critic(agent.actor(state,q_length=q_lengths,only_critic = True))[0]
            t5 = time.time()
            # R = agent.critic(x_in)[0]
            critic_output=R.clone()
            for r in range(len(reward_reversed)):
                R = R * gamma + reward_reversed[r]
                Reward.append(R)
            # clip the target:
            Reward = torch.stack(Reward[::-1], 0)
            Reward = Reward.view(-1)
            t6 = time.time()
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(torch.clamp(logprobs - old_logprobs.detach(), -torch.inf, 10)).to(Reward.device)
            # print('t2', t2)
            # print('t3', t3)
            # print('t4', t4)
            # print('t5', t5-ttt)
            # print('t6', t6-t5)
            # Finding Surrogate Loss:
            advantages = Reward - bl_val_detached

            # adaptive weights
            if opts.weighted_start is not None and opts.single_task is None:
                weights = task_weights.repeat(advantages.shape[0]//task_weights.shape[0])
            else:
                weights = 1.


            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
            # if torch.sum(torch.isinf(surr1)) > 0:
            #     print('='*25 + ' nan surr1 ' + '='*25)
            reinforce_loss = -torch.min(surr1, surr2) * weights
            reinforce_loss = reinforce_loss.mean()

            # define baseline loss
            if old_value is None:
                baseline_loss = ((bl_val - Reward) ** 2) * weights
                baseline_loss = baseline_loss.mean()
                old_value = bl_val.detach()
            else:
                vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
                v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2)) * weights
                baseline_loss = v_max.mean()

            # check K-L divergence (for logging only)
            approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
            approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0
            # calculate loss
            loss = baseline_loss + reinforce_loss
            # print(baseline_loss, reinforce_loss)
            # update gradient step
            agent.optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm and get (clipped) gradient norms for logging
            current_step = int(pre_step + t//n_step * K_epochs  + _k)
            grad_norms = clip_grad_norms(agent.optimizer.param_groups, opts.max_grad_norm)

            # perform gradient descent
            agent.optimizer.step()
            pbar.update()
            # Logging to tensorboard
            if current_step % int(opts.log_step) == 0:
                log_to_tb_train(tb_logger, agent, Reward,R,critic_output, ratios, bl_val_detached, total_cost, grad_norms, memory.rewards, entropy, approx_kl_divergence,
                                reinforce_loss, baseline_loss, logprobs, opts.show_figs, current_step)

            # if rank == 0: pbar.update(1)
            # end update
        

        memory.clear_memory()

    # return learning steps
    return ( t // n_step + 1) * K_epochs, batch_return

