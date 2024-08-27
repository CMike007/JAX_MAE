#%% Library Import
# Import Libraries

import os
import sys
import time

import chex
import flax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import Any, Dict, NamedTuple, Sequence
import distrax
from gymnax.environments import spaces

sys.path.append('/Users/millionaire/Desktop/UCL/Thesis/purejaxrl-main')
sys.path.append('/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3')
#Code snippet to disable all jitting.
from jax import config

from gymnax_exchange.jaxen.exec_env_relP_stableVersion0720 import ExecutionEnv

config.update("jax_disable_jit", False) 
# config.update("jax_disable_jit", True)
config.update("jax_check_tracer_leaks",False) #finds a whole assortment of leaks if true... bizarre.

import datetime
#wandbOn = True
wandbOn = False
if wandbOn:
    import wandb
    

from purejaxrl.experimental.s5.s5 import StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO
from purejaxrl.experimental.s5.wrappers import FlattenObservationWrapper, LogWrapper

#%% Definition ActorCritic NN

d_model = 256
ssm_size = 256
C_init = "lecun_normal"
discretization="zoh"
dt_min=0.001
dt_max=0.1
n_layers = 4
conj_sym=True
clip_eigs=False
bidirectional=False

blocks = 1
block_size = int(ssm_size / blocks)

Lambda, _, B, V, B_orig = make_DPLR_HiPPO(ssm_size)

block_size = block_size // 2
ssm_size = ssm_size // 2

Lambda = Lambda[:block_size]
V = V[:, :block_size]

Vinv = V.conj().T


ssm_init_fn = init_S5SSM(H=d_model,
                            P=ssm_size,
                            Lambda_re_init=Lambda.real,
                            Lambda_im_init=Lambda.imag,
                            V=V,
                            Vinv=Vinv,
                            C_init=C_init,
                            discretization=discretization,
                            dt_min=dt_min,
                            dt_max=dt_max,
                            conj_sym=conj_sym,
                            clip_eigs=clip_eigs,
                            bidirectional=bidirectional)

class ActorCriticS5(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    def setup(self):
        self.encoder_0 = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.encoder_1 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
    
        self.action_body_0 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.action_body_1 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.action_decoder = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.5))

        self.value_body_0 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.value_body_1 = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.value_decoder = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

        self.s5 = StackedEncoderModel(
            ssm=ssm_init_fn,
            d_model=d_model,
            n_layers=n_layers,
            activation="half_glu1",
        )
        self.actor_logtstd = self.param("log_std", nn.initializers.constant(-0.7), (self.action_dim,))


    def __call__(self, hidden, x):
        obs, dones = x
        embedding = self.encoder_0(obs)
        embedding = nn.leaky_relu(embedding)
        embedding = self.encoder_1(embedding)
        embedding = nn.leaky_relu(embedding)

        hidden, embedding = self.s5(hidden, embedding, dones)

        actor_mean = self.action_body_0(embedding)
        actor_mean = nn.leaky_relu(actor_mean)
        actor_mean = self.action_body_1(actor_mean)
        actor_mean = nn.leaky_relu(actor_mean)
        actor_mean = self.action_decoder(actor_mean)


        #pi = distrax.Categorical(logits=actor_mean)
        #Old version ^^
        # actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(self.actor_logtstd))
        #New version ^^

        critic = self.value_body_0(embedding)
        critic = nn.leaky_relu(critic)
        critic = self.value_body_1(critic)
        critic = nn.leaky_relu(critic)
        critic = self.value_decoder(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

#%% Clickbait Config I

timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
config = {
    # "LR": 2.5e-3,
    # "LR": 2.5e-4,
    "LR": 2.5e-5,
    # "LR": 2.5e-6,
    "ENT_COEF": 0.1,
    # "ENT_COEF": 0.01,
    "NUM_ENVS": 2,     # reduced from 500
    "TOTAL_TIMESTEPS": 1e8,     # reduced from 1e8
    # "TOTAL_TIMESTEPS": 1e7,
    # "TOTAL_TIMESTEPS": 3.5e7,
    "NUM_MINIBATCHES": 2,
    # "NUM_MINIBATCHES": 4,
    "UPDATE_EPOCHS": 5,
    # "UPDATE_EPOCHS": 4,
    "NUM_STEPS": 455,
    # "NUM_STEPS": 10,
    "CLIP_EPS": 0.2,
    # "CLIP_EPS": 0.2,
    
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 2.0,
    "ANNEAL_LR": True,
    "NORMALIZE_ENV": False,
    
    "ACTOR_TYPE":"S5",
    
    "ENV_NAME": "alphatradeExec-v0",
    # "WINDOW_INDEX": 0,
    "WINDOW_INDEX": -1,
    "DEBUG": True,     # changed to False
    "ATFOLDER": '/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/',
    "TASKSIDE":'sell',
    "REWARD_LAMBDA":1,
    "ACTION_TYPE":"pure",
    # "ACTION_TYPE":"delta",
    "TASK_SIZE":500,
    "RESULTS_FILE":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/results_file_"+f"{timestamp}",
    "CHECKPOINT_DIR":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/checkpoints_"+f"{timestamp}",
}

#%% Clickbait Config II

config["NUM_UPDATES"] = (
    config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
)
config["MINIBATCH_SIZE"] = (
    config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
)
env = ExecutionEnv(config["ATFOLDER"],config["TASKSIDE"],config["WINDOW_INDEX"],config["ACTION_TYPE"],config["TASK_SIZE"],config["REWARD_LAMBDA"])
env_params = env.default_params
env = LogWrapper(env)

def linear_schedule(count):
    frac = (
        1.0
        - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
        / config["NUM_UPDATES"]
    )
    return config["LR"] * frac

#%% train intitialization
device = jax.devices()[0]
rng = jax.device_put(jax.random.PRNGKey(0), device)

# INIT NETWORK
network = ActorCriticS5(env.action_space(env_params).shape[0], config=config)
rng, _rng = jax.random.split(rng)
init_x = (
    jnp.zeros(
        (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
    ),
    jnp.zeros((1, config["NUM_ENVS"])),
)
init_hstate = StackedEncoderModel.initialize_carry(config["NUM_ENVS"], ssm_size, n_layers)
network_params = network.init(_rng, init_hstate, init_x)
if config["ANNEAL_LR"]:
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=linear_schedule,b1=0.9,b2=0.99, eps=1e-5),
    )
else:
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"],b1=0.9,b2=0.99, eps=1e-5),
    )
train_state = TrainState.create(
    apply_fn=network.apply,
    params=network_params,
    tx=tx,
)

# INIT ENV
rng, _rng = jax.random.split(rng)
reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
init_hstate = StackedEncoderModel.initialize_carry(config["NUM_ENVS"], ssm_size, n_layers)

# Def Runner State
rng, _rng = jax.random.split(rng)
runner_state = (
    train_state,
    env_state,
    obsv,
    jnp.zeros((config["NUM_ENVS"]), dtype=bool),
    init_hstate,
    _rng,
)

#%% Playground

# obsv is an array with one entry per "environment"

# each environment entry is an array with 610 entries
# those 610 entries are initiated through env_params depending on buy or sell objective
# if task == buy: --> env_params.obs_buy_list otherwise env_params.obs_sell_list
# obs_buy / obs_sell list are split in 13 cubes with 610 entries each
# each 

obsv
len(obsv)

env_params.obs_buy_list[0]

env_state

len(ac_in[0][0][0])

AttributeError: 'jaxlib.xla_extension.ArrayImpl' object has no attribute 'type'
obsv.shape
Out[38]: (2, 610)

rng_step.shape

action.shape


rng_step
env_state
action

#%% Train_Loop --> _update_step | _env_step


train_state, env_state, last_obs, last_done, hstate, rng = runner_state
rng, _rng = jax.random.split(rng)

# SELECT ACTION
ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
hstate, pi, value = network.apply(train_state.params, hstate, ac_in)

action = pi.sample(seed=_rng)

log_prob = pi.log_prob(action)

value, action, log_prob = (
    value.squeeze(0),
    action.squeeze(0),
    log_prob.squeeze(0),
)

# STEP ENV
rng, _rng = jax.random.split(rng)
rng_step = jax.random.split(_rng, config["NUM_ENVS"])


# --> I´m here and about to find out how the inputs and then outputs really look like...
obsv, env_state, reward, done, info = jax.vmap(
    env.step, in_axes=(0, 0, 0, None)
)(rng_step, env_state, action, env_params)

transition = Transition(
    last_done, action, value, reward, log_prob, last_obs, info
)
runner_state = (train_state, env_state, obsv, done, hstate, rng)

print(runner_state, transition)

#%% Playground

env_state

env_params

len(env_params.stateArray_list[0])

env_params.obs_sell_list[0][0]


len(env_params.message_data[0])

stateArray = env_params.message_data[0]

env_params.message_data.shape
env_params.message_data[0].shape

blob = stateArray[:,0:6]

blob
blob[:,:,3][180:185]

stateArray[:,0:6]
stateArray[:,6:12]
stateArray[:,12:18]
stateArray[:,18:20]
stateArray[:,20:22]
stateArray[0:2,22:23]
stateArray[2:4,22:23]
stateArray[4:5,22:23][0]

len(stateArray[:,0:6])
len(stateArray[:,6:12])
len(stateArray[:,12:18])
len(stateArray[:,18:20])
len(stateArray[:,20:22])
len(stateArray[0:2,22:23])
len(stateArray[2:4,22:23])
len(stateArray[4:5,22:23][0])
            
            
            

idx_window = 12
step_counter = 2

env_params.message_data[idx_window,step_counter,:,:]

env_state["step_counter"]

#%%
# reset_env() Simulation (exec_env)

idx_data_window = 0

def stateArray2state(stateArray):
    
    self_task_size = 500
    self_max_steps_in_episode_arr = [269, 183]
    
    state0 = stateArray[:,0:6];
    state1 = stateArray[:,6:12];
    state2 = stateArray[:,12:18];
    state3 = stateArray[:,18:20];
    state4 = stateArray[:,20:22];
    state5 = stateArray[0:2,22:23].squeeze(axis=-1);
    state6 = stateArray[2:4,22:23].squeeze(axis=-1);
    state9= stateArray[4:5,22:23][0].squeeze(axis=-1)

    return (state0,state1,state2,state3,state4,state5,state6,0,idx_data_window,state9,self_task_size,0,0,0,self_max_steps_in_episode_arr[idx_data_window],0,0,0,0,0)

params=env.default_params
stateArray = params.stateArray_list[idx_data_window]    

state_ = stateArray2state(stateArray)    

state_
print(*state_)

#%%

# get_data_messages() Simulation (JaxOrderBookArrays)

params.message_data[0, 150, :, :]
        
#%% Unused / OLD CODE

    initial_hstate = runner_state[-2]
    runner_state, traj_batch = jax.lax.scan(
        _env_step, runner_state, None, config["NUM_STEPS"]
    )

    # CALCULATE ADVANTAGE
    train_state, env_state, last_obs, last_done, hstate, rng = runner_state
    ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
    _, _, last_val = network.apply(train_state.params, hstate, ac_in)
    last_val = last_val.squeeze(0)
    def _calculate_gae(traj_batch, last_val, last_done):
        def _get_advantages(carry, transition):
            gae, next_value, next_done = carry
            done, value, reward = transition.done, transition.value, transition.reward 
            delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
            gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
            return (gae, value, done), gae
        _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val, last_done), traj_batch, reverse=True, unroll=16)
        return advantages, advantages + traj_batch.value
    advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

    # UPDATE NETWORK
    def _update_epoch(update_state, unused):
        def _update_minbatch(train_state, batch_info):
            init_hstate, traj_batch, advantages, targets = batch_info

            def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                # RERUN NETWORK
                _, pi, value = network.apply(
                    params, init_hstate, (traj_batch.obs, traj_batch.done)
                )
                log_prob = pi.log_prob(traj_batch.action)

                # CALCULATE VALUE LOSS
                value_pred_clipped = traj_batch.value + (
                    value - traj_batch.value
                ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                value_loss = (
                    0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                )

                # CALCULATE ACTOR LOSS
                ratio = jnp.exp(log_prob - traj_batch.log_prob)
                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor1 = ratio * gae
                loss_actor2 = (
                    jnp.clip(
                        ratio,
                        1.0 - config["CLIP_EPS"],
                        1.0 + config["CLIP_EPS"],
                    )
                    * gae
                )
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                loss_actor = loss_actor.mean()
                entropy = pi.entropy().mean()

                total_loss = (
                    loss_actor
                    + config["VF_COEF"] * value_loss
                    - config["ENT_COEF"] * entropy
                )
                return total_loss, (value_loss, loss_actor, entropy)

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params, init_hstate, traj_batch, advantages, targets
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, total_loss

        (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        ) = update_state

        rng, _rng = jax.random.split(rng)
        permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
        batch = (init_hstate, traj_batch, advantages, targets)

        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=1), batch
        )

        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(
                jnp.reshape(
                    x,
                    [x.shape[0], config["NUM_MINIBATCHES"], -1]
                    + list(x.shape[2:]),
                ),
                1,
                0,
            ),
            shuffled_batch,
        )

        train_state, total_loss = jax.lax.scan(
            _update_minbatch, train_state, minibatches
        )
        update_state = (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        )
        return update_state, total_loss

    init_hstate = initial_hstate # TBH
    update_state = (
        train_state,
        init_hstate,
        traj_batch,
        advantages,
        targets,
        rng,
    )
    update_state, loss_info = jax.lax.scan(
        _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
    )
    # grad_norm=jnp.mean(loss_info[1])
    train_state = update_state[0]
    # metric = (traj_batch.info,train_state.params,grad_norm)
    metric = (traj_batch.info,train_state.params)
    rng = update_state[-1]

    runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
    return runner_state, metric

rng, _rng = jax.random.split(rng)
runner_state = (
    train_state,
    env_state,
    obsv,
    jnp.zeros((config["NUM_ENVS"]), dtype=bool),
    init_hstate,
    _rng,
)
runner_state, metric = jax.lax.scan(
    _update_step, runner_state, None, config["NUM_UPDATES"]
)
return {"runner_state": runner_state, "metric": metric}



#%%
    
if __name__ == "__main__":
    # try:
    #     ATFolder = sys.argv[1] 
    # except:
    #     ATFolder = "/homes/80/kang/AlphaTrade/training_oneDay"
    #     # ATFolder = '/homes/80/kang/AlphaTrade'
    #     # ATFolder = '/home/duser/AlphaTrade'
    # print("AlphaTrade folder:",ATFolder)
    timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
    os.environ["NCCL_P2P_DISABLE"] = "1"

    ppo_config = {
        # "LR": 2.5e-3,
        # "LR": 2.5e-4,
        "LR": 2.5e-5,
        # "LR": 2.5e-6,
        "ENT_COEF": 0.1,
        # "ENT_COEF": 0.01,
        "NUM_ENVS": 500,     # reduced from 500
        "TOTAL_TIMESTEPS": 1e8,     # reduced from 1e8
        # "TOTAL_TIMESTEPS": 1e7,
        # "TOTAL_TIMESTEPS": 3.5e7,
        "NUM_MINIBATCHES": 2,
        # "NUM_MINIBATCHES": 4,
        "UPDATE_EPOCHS": 5,
        # "UPDATE_EPOCHS": 4,
        "NUM_STEPS": 455,
        # "NUM_STEPS": 10,
        "CLIP_EPS": 0.2,
        # "CLIP_EPS": 0.2,
        
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 2.0,
        "ANNEAL_LR": True,
        "NORMALIZE_ENV": False,
        
        "ACTOR_TYPE":"S5",
        
        "ENV_NAME": "alphatradeExec-v0",
        # "WINDOW_INDEX": 0,
        "WINDOW_INDEX": -1,
        "DEBUG": True,     # changed to False
        "ATFOLDER": '/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/',
        "TASKSIDE":'sell',
        "REWARD_LAMBDA":1,
        "ACTION_TYPE":"pure",
        # "ACTION_TYPE":"delta",
        "TASK_SIZE":500,
        "RESULTS_FILE":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/results_file_"+f"{timestamp}",
        "CHECKPOINT_DIR":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/checkpoints_"+f"{timestamp}",
    }

    if wandbOn:
        run = wandb.init(
            project="AlphaTradeJAX_Train",
            config=ppo_config,
            # sync_tensorboard=True,  # auto-upload  tensorboard metrics
            save_code=True,  # optional
        )
        import datetime;params_file_name = f'params_file_{wandb.run.name}_{timestamp}'
        print(f"Results would be saved to {params_file_name}")
    else:
        import datetime;params_file_name = f'params_file_{timestamp}'
        print(f"Results would be saved to {params_file_name}")


    device = jax.devices()[0]
    # device = jax.devices()[1]
    # device = jax.devices()[-1]
    rng = jax.device_put(jax.random.PRNGKey(0), device)
    train_jit = jax.jit(f (ppo_config), device=device)
    out = train_jit(rng)

    # if jax.device_count() == 1:
    #     # +++++ Single GPU +++++
    #     rng = jax.random.PRNGKey(0)
    #     # rng = jax.random.PRNGKey(30)
    #     train_jit = jax.jit(make_train(ppo_config))
    #     start=time.time()
    #     out = train_jit(rng)
    #     print("Time: ", time.time()-start)
    #     # +++++ Single GPU +++++
    # else:
    #     # +++++ Multiple GPUs +++++
    #     num_devices = int(jax.device_count())
    #     rng = jax.random.PRNGKey(30)
    #     rngs = jax.random.split(rng, num_devices)
    #     train_fn = lambda rng: make_train(ppo_config)(rng)
    #     start=time.time()
    #     out = jax.pmap(train_fn)(rngs)
    #     print("Time: ", time.time()-start)
    #     # +++++ Multiple GPUs +++++
    
    

    # '''
    # # ---------- Save Output ----------
    import flax

    train_state = out['runner_state'][0] # runner_state.train_state
    params = train_state.params
    


    import datetime;params_file_name = f'params_file_{wandb.run.name}_{timestamp}'
    # Save the params to a file using flax.serialization.to_bytes
    with open(params_file_name, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
        print(f"pramas saved")

    # Load the params from the file using flax.serialization.from_bytes
    with open(params_file_name, 'rb') as f:
        restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        print(f"pramas restored")
        
    # jax.debug.breakpoint()
    # assert jax.tree_util.tree_all(jax.tree_map(lambda x, y: (x == y).all(), params, restored_params))
    # print(">>>")
    # '''

    if wandbOn:
        run.finish()
        
      
