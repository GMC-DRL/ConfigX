from utils.utils import set_seed
import numpy as np
import torch
from tqdm import tqdm
from utils.logger import log_to_val
import os, time
from env import DummyVectorEnv,SubprocVectorEnv, RayVectorEnv

# interface for rollout
def rollout(dataloader,opts,agent=None,tb_logger=None, run_name=None, epoch_id=0):
    if run_name is None:
        run_name = opts.run_name
    run_time = time.strftime("%Y%m%dT%H%M%S")
    rollout_name=f'MTRL_{run_name}_{epoch_id}'
    if agent:
        # rollout_name='MTRL-'+rollout_name
        agent.eval()
    T = opts.MaxGen // opts.skip_step
    batch_size = dataloader.batch_size
    # to store the whole rollout process
    cost_rollout=np.zeros((len(dataloader),T))
    
    time_eval=0
    collect_mean=[]
    collect_std=[]
    collect_R = np.zeros(len(dataloader))
    
    # set the same random seed before rollout for the sake of fairness
    set_seed(opts.testseed)
    pbar = tqdm(total=np.ceil(len(dataloader) / batch_size) * T, desc = rollout_name)
    for bat_id,batch in enumerate(dataloader):
        # figure out the backbone algorithm
        batch_size = len(batch)
        # see if there is agent to aid the backbone
        env_list=[lambda e=p: e for p in batch]
        # Parallel environmen SubprocVectorEnv can only be used in Linux
        vector_env=SubprocVectorEnv if opts.is_linux else DummyVectorEnv
        # vector_env=RayVectorEnv if opts.is_linux else DummyVectorEnv
        problem=vector_env(env_list)
        q_lengths = torch.zeros(batch_size)
        for i in range(batch_size):
            q_lengths[i] = batch[i].n_component
        # list to store the final optimization result
        collect_gbest=np.zeros((batch_size,opts.repeat))
        R = np.zeros(batch_size)
        for i in range(opts.repeat):
            # reset the backbone algorithm
            is_end=False

            trng = torch.random.get_rng_state()

            problem.seed(opts.testseed + i)
            state=problem.reset()
            state=torch.FloatTensor(state).to(opts.device)

            torch.random.set_rng_state(trng)

            time_eval+=1
            info = None
            # visualize the rollout process
            for t in range(T):
                
                logits = agent.actor(state, 
                                    q_length=q_lengths,
                                    to_critic=False,
                                    detach_state=True,
                                    )
                trng = torch.random.get_rng_state()

                next_state,rewards,is_end,info = problem.step(logits.detach().cpu())

                torch.random.set_rng_state(trng)

                R += np.array(rewards)
                # put action into environment(backbone algorithm to be specific)
                state=torch.FloatTensor(next_state).to(opts.device)
                pbar.update()
                # store the rollout cost history
                for tt in range(batch_size):
                    cost_rollout[tt+batch_size*bat_id,t]+=info[tt]['gbest_val']
                # if is_end.all():
                #     if t+1<T:
                #         for tt in range(batch_size):
                #             cost_rollout[tt+batch_size*bat_id,t+1:]+=info[tt]['gbest_val']
                    # store the final cost in the end of optimization process
            for tt in range(batch_size):
                collect_gbest[tt,i]=info[tt]['gbest_val']
                    # break
        # collect the mean and std of final cost
        collect_std.append(np.mean(np.std(collect_gbest,axis=-1)).item())
        collect_mean.append(np.mean(collect_gbest).item())
        collect_R[batch_size*bat_id:batch_size*(bat_id+1)] = R / opts.repeat
        # close the env
        problem.close()


    cost_rollout/=opts.repeat
    # cost_rollout=np.mean(cost_rollout,axis=0)
    
    pbar.close()
    # save rollout data to file
    saving_path=os.path.join(opts.log_dir, "rollout_{}_{}".format(epoch_id, run_time))
    # only save part of the optimization process
    # save_list=[cost_rollout[int((opts.dim**(k/5-3) * opts.max_fes )// opts.population_size -1 )].item() for k in range(15)]
    save_dict={'mean':np.mean(collect_mean).item(),'std':np.mean(collect_std).item(),'process':cost_rollout,'R_process': collect_R}
    np.save(saving_path,save_dict)

    # log to tensorboard if needed
    if tb_logger:
        log_to_val(tb_logger, np.mean(collect_gbest).item(), np.mean(collect_R), epoch_id)
    
    # calculate and return the mean and std of final cost
    return np.mean(collect_gbest).item(),np.mean(collect_std).item(), np.mean(collect_R).item()