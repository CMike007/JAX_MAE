#%% Import Libraries

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

from gymnax_exchange.jaxen.exec_env_test_Sigmoid_bugFix import ExecutionEnv

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

#%% Define the Network (ActorCriticS5) & Transition Class

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

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(self.actor_logtstd))
        
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
    
#%% Define Config

timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
os.environ["NCCL_P2P_DISABLE"] = "1"

config = {
    "AGENTS": ["OOE"],
    "wandb_RUN_NAME":'atomic-dust-88_08-03_10-41',
    "NUM_ENVS": 10,            # reduced from 500
    #"TOTAL_TIMESTEPS": 1e2,     # reduced from 1e8
    #"NUM_MINIBATCHES": 1,       # (they also tried 4 instead)
    #"UPDATE_EPOCHS": 5,         # slightly confusing as it is of course used in the network update, not sure though if we would be forced to use the same for all...
    "NUM_STEPS": 1000,
    "ENV_NAME": "alphatradeExec-v0",
    "WINDOW_INDEX": -1,
    "DEBUG": False,     # changed to False
    "wandbOn": False,
    "PARAMSFOLDER": '/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/params',
    "ATFOLDER": '/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/',
    "RESULTS_FILE":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/results_file_"+f"{timestamp}",
    "CHECKPOINT_DIR":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/checkpoints_"+f"{timestamp}",
    "OOE": {
        "TASKSIDE":'sell',
        "TASK_SIZE":2000,
        "ACTION_TYPE":"pure",
        # "ACTION_TYPE":"delta",
        "LR": 2.5e-5,
        "CLIP_EPS": 0.2,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "REWARD_LAMBDA":1,
        "ENT_COEF": 0.1,    # Entropy Coefficient: ratio of the entropy to deduct from the total loss (_loss_fn)
        "VF_COEF": 0.5,         # Value Loss Coefficient: how much of the value loss should be added to the total loss
        "MAX_GRAD_NORM": 2.0,
        "ANNEAL_LR": True,
    },
    "TAgent": {
        "TASK_SIZE":300,
        "LR": 2.5e-5,
        "CLIP_EPS": 0.2,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "REWARD_LAMBDA":1,
        "ENT_COEF": 0.1,    # Entropy Coefficient: ratio of the entropy to deduct from the total loss (_loss_fn)
        "VF_COEF": 0.5,         # Value Loss Coefficient: how much of the value loss should be added to the total loss
        "MAX_GRAD_NORM": 2.0,
        "ANNEAL_LR": True,            
        "OMEGA": 0.75,        # Reward = omega * alpha * PnL(y) - (1 - omega) * Flow(q) <-- used to weight the tradeoff between flow and PnL
        "ALPHA": 1,
        "GAMMA": 0.05,
    }
}

config["OOE"]["NUM_STEPS"] = config["NUM_STEPS"]

#%% Import of the agent params

def load_model(config, agent):
    if config['wandb_RUN_NAME']:
        runname = config['wandb_RUN_NAME']
        folder = config['PARAMSFOLDER']
        params_file_name = f'{folder}/Aparam_{runname}_{agent}'
    
        with open(params_file_name, 'rb') as f:
            restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
            print(f"{agent} Params restored")
            return(restored_params)
    else:
        print(f"{agent} Params could not be restored")
           
params_OOE = load_model(config, "OOE")
params_TA = load_model(config, "TA")

#%% Declare Environment & Initialize RNG

env = ExecutionEnv(config["ATFOLDER"],config["WINDOW_INDEX"],config['OOE'],config['TAgent'])
env_params = env.default_params
env = LogWrapper(env)

device = jax.devices()[0]
rng = jax.device_put(jax.random.PRNGKey(0), device)

#%% OOE - INIT NETWORK WITH PRE-TRAINED PARAMETERS

#enable the linear schedule
def linear_schedule(count):
    frac = (
        1.0
        - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
        / config["NUM_UPDATES"]
    )
    return config["OOE"]["LR"] * frac


# OOE - INIT NETWORK
OOE_network = ActorCriticS5(env.action_space(env_params).shape[0], config=config)
rng, _rng = jax.random.split(rng)
init_OOE_x = (
    jnp.zeros(
        (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
    ),
    jnp.zeros((1, config["NUM_ENVS"])),
)
init_OOE_hstate = StackedEncoderModel.initialize_carry(config["NUM_ENVS"], ssm_size, n_layers)
OOE_network_params = OOE_network.init(_rng, init_OOE_hstate, init_OOE_x)
if config["OOE"]["ANNEAL_LR"]:
    OOE_tx = optax.chain(
        optax.clip_by_global_norm(config["OOE"]["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=linear_schedule,b1=0.9,b2=0.99, eps=1e-5),
    )
else:
    OOE_tx = optax.chain(
        optax.clip_by_global_norm(config["OOE"]["MAX_GRAD_NORM"]),
        optax.adam(config["OOE"]["LR"],b1=0.9,b2=0.99, eps=1e-5),
    )
train_state_OOE = TrainState.create(
    apply_fn=OOE_network.apply,
    params=OOE_network_params,
    tx=OOE_tx,
)

'''
# TAgent - INIT NETWORK
TAgent_network = ActorCriticS5(3, config=config)
rng, _rng = jax.random.split(rng)
init_TAgent_x = (
    jnp.zeros(
        (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
    ),
    jnp.zeros((1, config["NUM_ENVS"])),
)
init_TAgent_hstate = StackedEncoderModel.initialize_carry(config["NUM_ENVS"], ssm_size, n_layers)
TAgent_network_params = TAgent_network.init(_rng, init_TAgent_hstate, init_TAgent_x)
if config["TAgent"]["ANNEAL_LR"]:
    TAgent_tx = optax.chain(
        optax.clip_by_global_norm(config["TAgent"]["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=linear_schedule,b1=0.9,b2=0.99, eps=1e-5),
    )
else:
    TAgent_tx = optax.chain(
        optax.clip_by_global_norm(config["TAgent"]["MAX_GRAD_NORM"]),
        optax.adam(config["TAgent"]["LR"],b1=0.9,b2=0.99, eps=1e-5),
    )
train_state_TAgent = TrainState.create(
    apply_fn=TAgent_network.apply,
    params=TAgent_network_params,
    tx=TAgent_tx,
)
'''

# Assume NUM_AGENTS is the number of TAgents per environment
NUM_AGENTS = 5

# TAgent - INIT NETWORK
TAgent_network = ActorCriticS5(3, config=config)
rng, _rng = jax.random.split(rng)

# Adjust the shape to account for the number of agents per environment
init_TAgent_x = (
    jnp.zeros(
        (NUM_AGENTS, config["NUM_ENVS"], *env.observation_space(env_params).shape)
    ),
    jnp.zeros((NUM_AGENTS, config["NUM_ENVS"])),
)

# Adjust hidden state initialization to accommodate multiple agents
init_TAgent_hstate = StackedEncoderModel.initialize_carry(config["NUM_ENVS"] * NUM_AGENTS, ssm_size, n_layers)

# Initialize the network parameters
TAgent_network_params = TAgent_network.init(_rng, init_TAgent_hstate, init_TAgent_x)

# Initialize the training state
if config["TAgent"]["ANNEAL_LR"]:
    TAgent_tx = optax.chain(
        optax.clip_by_global_norm(config["TAgent"]["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=linear_schedule, b1=0.9, b2=0.99, eps=1e-5),
    )
else:
    TAgent_tx = optax.chain(
        optax.clip_by_global_norm(config["TAgent"]["MAX_GRAD_NORM"]),
        optax.adam(config["TAgent"]["LR"], b1=0.9, b2=0.99, eps=1e-5),
    )

train_state_TAgent = TrainState.create(
    apply_fn=TAgent_network.apply,
    params=TAgent_network_params,
    tx=TAgent_tx,
)

#%% ENV - INIT ENV

rng, _rng = jax.random.split(rng)
reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
obsv_OOE, obsv_TAgent = obsv[0], obsv[1]
rng, _rng = jax.random.split(rng)

runner_state = (
    env_state,
    jnp.zeros((config["NUM_ENVS"]), dtype=bool),
    _rng,
    [train_state_OOE, obsv_OOE, init_OOE_hstate],
    [train_state_TAgent, obsv_TAgent, init_TAgent_hstate]
)

#%%

# Test LOOP
def _env_step(runner_state, unused):
    
    # UNPACK Runner State Data
    env_state, last_done, rng, OOE_data, TAgent_data = runner_state
    test_state_OOE, last_obs_OOE, OOE_hstate = OOE_data[0], OOE_data[1], OOE_data[2]
    test_state_TAgent, last_obs_TAgent, TAgent_hstate = TAgent_data[0], TAgent_data[1], TAgent_data[2]
    

    # OOE SELECT ACTION
    rng, _rng = jax.random.split(rng)
    OOE_ac_in = (last_obs_OOE[np.newaxis, :], last_done[np.newaxis, :])
    OOE_hstate_, OOE_pi, OOE_value = OOE_network.apply(test_state_OOE.params, OOE_hstate, OOE_ac_in)
    OOE_action = OOE_pi.sample(seed=_rng)
    OOE_log_prob = OOE_pi.log_prob(OOE_action)
    OOE_value, OOE_action, OOE_log_prob = (
        OOE_value.squeeze(0),
        OOE_action.squeeze(0),
        OOE_log_prob.squeeze(0),
    )
    
    # TAgent SELECT ACTION
    rng, _rng = jax.random.split(rng)
    TAgent_ac_in = (last_obs_TAgent[np.newaxis, :], last_done[np.newaxis, :])
    TAgent_hstate_, TAgent_pi, TAgent_value = TAgent_network.apply(test_state_TAgent.params, TAgent_hstate, TAgent_ac_in)
    TAgent_action = TAgent_pi.sample(seed=_rng)
    TAgent_log_prob = TAgent_pi.log_prob(TAgent_action)
    TAgent_value, TAgent_action, TAgent_log_prob = (
        TAgent_value.squeeze(0),
        TAgent_action.squeeze(0),
        TAgent_log_prob.squeeze(0),
    )
    
#%% Spielplatz

NUM_AGENTS = 5

# Vectorize the TAgent network application over the number of agents
def apply_tagent_network(params, hstate, ac_in):
    return TAgent_network.apply(params, hstate, ac_in)

# Apply the TAgent network for all agents
rng, _rng = jax.random.split(rng)
TAgent_ac_in = (last_obs_TAgent[np.newaxis, :], last_done[np.newaxis, :])

# Vectorize across the number of agents
TAgent_hstate_, TAgent_pi, TAgent_value = jax.vmap(apply_tagent_network, in_axes=(None, 0, None))(
    test_state_TAgent.params, TAgent_hstate, TAgent_ac_in
)

# Sample actions for each agent
TAgent_action = jax.vmap(lambda pi, seed: pi.sample(seed=seed))(TAgent_pi, jax.random.split(_rng, NUM_AGENTS))
TAgent_log_prob = jax.vmap(lambda pi, action: pi.log_prob(action))(TAgent_pi, TAgent_action)

# Squeeze dimensions where necessary
TAgent_value, TAgent_action, TAgent_log_prob = (
    TAgent_value.squeeze(0),
    TAgent_action.squeeze(0),
    TAgent_log_prob.squeeze(0),
)


#%%
    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, config["NUM_ENVS"])
    
    obsv, env_state, reward, done, info = jax.vmap(
        env.step, in_axes=(0, 0, 0, 0, None)
    )(rng_step, env_state, OOE_action, TAgent_action, env_params)
    
    obsv_OOE, obsv_TAgent = obsv[0], obsv[1]
    reward_OOE, reward_TAgent = reward[0], reward[1]
    
    OOE_transition = Transition(
        last_done, OOE_action, OOE_value, reward_OOE, OOE_log_prob, last_obs_OOE, info
    )
    
    TAgent_transition = Transition(
        last_done, TAgent_action, TAgent_value, reward_TAgent, TAgent_log_prob, last_obs_TAgent, info
    )
    
    runner_state = (
        env_state,
        done,
        rng,
        [test_state_OOE, obsv_OOE, OOE_hstate],
        [test_state_TAgent, obsv_TAgent, TAgent_hstate]
    )

    return runner_state, [OOE_transition, TAgent_transition]

runner_state, traj_batch = jax.lax.scan(
    _env_step, runner_state, None, config["NUM_STEPS"]
)

#%%

OOE_transitions = traj_batch[0]
TA_transiitons = traj_batch[1]

OOE_transitions.info["Price_history"].shape

OOE_transitions.info["Price_history"][:,0]
OOE_transitions.info["Buy_history"][0][0][-1]

