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

import matplotlib.pyplot as plt

sys.path.append('/Users/millionaire/Desktop/UCL/Thesis/purejaxrl-main')
sys.path.append('/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3')
#Code snippet to disable all jitting.
from jax import config

from gymnax_exchange.jaxen.exec_env_relP import ExecutionEnv

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

'''
# General Adjustments:
    
    -   config (ppo_config):
        packaging agent specific variables in designated dictionaries and include one dictionary per NN
        
    -   runner_state:
        old composition:
            - train_state       -->     agent-specific
            - env_state         -->     general
            - obsv              -->     agent-specific
            - jnp.zeros         -->     general
            - hstate            -->     agent-specific (not sure if agent or network)
            - _rng              -->     general
            
        new composition:
            - env_state
            - jnp.zeros
            - _rng
            - OOE_data
            - TAgent_data
            
            where OOE_data and TAgent_data are lists with respectively:
            - train_state
            - obsv
            - hstate

# Additions:
    
    -   TAgent - INIT NETWORK
        basically initilizing a complete new network that the trading agent uses to update its policy
        very similar to the existing OOE framework until now
    
    
'''
    
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
os.environ["NCCL_P2P_DISABLE"] = "1"
steps = 455

config =  {
    "NUM_ENVS": 10,              # reduced from 500
    "TOTAL_TIMESTEPS": 3e6,     # reduced from 1e8 (last run #dandy-leaf-28 was done with 1e6 --> 2h runtime)
    "NUM_MINIBATCHES": 2,       # (they also tried 4 instead)
    "UPDATE_EPOCHS": 5,         # slightly confusing as it is of course used in the network update, not sure though if we would be forced to use the same for all...
    "NUM_STEPS": 455,
    "ENV_NAME": "alphatradeExec-v0",
    "WINDOW_INDEX": -1,
    "DEBUG": True,
    "wandbOn": True,
    "ATFOLDER": '/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/',
    "RESULTS_FILE":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/results_file_"+f"{timestamp}",
    "CHECKPOINT_DIR":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/checkpoints_"+f"{timestamp}",
    "PICTURE_DIR":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/plot"+f"{timestamp}price_history.png",
    "OOE": {
        "TASKSIDE":'sell',
        "TASK_SIZE":500,
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
        "ANNEAL_LR": True
    },
    "TAgent": {
        "TASK_SIZE":1000,
        "LR": 2.5e-5,
        "CLIP_EPS": 0.2,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "REWARD_LAMBDA":1,
        "ENT_COEF": 0.1,    # Entropy Coefficient: ratio of the entropy to deduct from the total loss (_loss_fn)
        "VF_COEF": 0.5,         # Value Loss Coefficient: how much of the value loss should be added to the total loss
        "MAX_GRAD_NORM": 2.0,
        "ANNEAL_LR": True,
        "OMEGA": 0.4,        # Reward = omega * alpha * PnL(y) - (1 - omega) * Flow(q) <-- used to weight the tradeoff between flow and PnL
        "ALPHA": 1,
        "GAMMA": 0.05
    }
}

#%% Clickbait Config II

# ENV - BASE PARAMETERS ENVIRONMENT 
config["NUM_UPDATES"] = (
    config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
)
config["MINIBATCH_SIZE"] = (
    config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
)

env = ExecutionEnv(config["ATFOLDER"],config["WINDOW_INDEX"],config['OOE'],config['TAgent'])
env_params = env.default_params
env = LogWrapper(env)

# ENV - SETTING LEARNING SCHEDULE 
# even though this is for the AGENT, this code is completely re-useable
def linear_schedule(count):
    frac = (
        1.0
        - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
        / config["NUM_UPDATES"]
    )
    return config["OOE"]["LR"] * frac

#%% train intitialization
device = jax.devices()[0]
rng = jax.device_put(jax.random.PRNGKey(0), device)

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
    

# ENV - INIT ENV
rng, _rng = jax.random.split(rng)
reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
obsv_OOE, obsv_TAGent = obsv[0], obsv[1]
rng, _rng = jax.random.split(rng)

runner_state = (
    env_state,
    jnp.zeros((config["NUM_ENVS"]), dtype=bool),
    _rng,
    [train_state_OOE, obsv_OOE, init_OOE_hstate],
    [train_state_TAgent, obsv_TAGent, init_TAgent_hstate]
)

#%%

#_env_step Function
for i in range(0, 100):
    
    env_state, last_done, rng, OOE_data, TAgent_data = runner_state
    train_state_OOE, last_obs_OOE, OOE_hstate = OOE_data[0], OOE_data[1], OOE_data[2]
    train_state_TAgent, last_obs_TAgent, TAgent_hstate = TAgent_data[0], TAgent_data[1], TAgent_data[2]
    
    
    # OOE SELECT ACTION
    rng, _rng = jax.random.split(rng)
    OOE_ac_in = (last_obs_OOE[np.newaxis, :], last_done[np.newaxis, :])
    OOE_hstate_, OOE_pi, OOE_value = OOE_network.apply(train_state_OOE.params, OOE_hstate, OOE_ac_in)
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
    TAgent_hstate_, TAgent_pi, TAgent_value = TAgent_network.apply(train_state_TAgent.params, TAgent_hstate, TAgent_ac_in)
    TAgent_action = TAgent_pi.sample(seed=_rng)
    TAgent_log_prob = TAgent_pi.log_prob(TAgent_action)
    TAgent_value, TAgent_action, TAgent_log_prob = (
        TAgent_value.squeeze(0),
        TAgent_action.squeeze(0),
        TAgent_log_prob.squeeze(0),
    )
    
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
        [train_state_OOE, obsv[0], OOE_hstate],
        [train_state_TAgent, obsv[1], TAgent_hstate]
    )
    
    traj_batch = [OOE_transition, TAgent_transition]

#%%

initial_hstate_OOE = runner_state[-2][-1]
initial_hstate_TAgent = runner_state[-1][-1]
          

# CALCULATE ADVANTAGE Function
env_state, last_done, rng, OOE_data, TAgent_data = runner_state
train_state_OOE, last_obs_OOE, init_OOE_hstate = OOE_data[0], OOE_data[1], OOE_data[2]
train_state_TAgent, last_obs_TAgent, init_TAgent_hstate = TAgent_data[0], TAgent_data[1], TAgent_data[2]

def _calculate_gae(traj_batch, last_val, last_done):
    def _get_advantages(carry, transition):
        gae, next_value, next_done = carry
        done, value, reward = transition.done, transition.value, transition.reward 
        delta = reward + config["OOE"]["GAMMA"] * next_value * (1 - next_done) - value
        gae = delta + config["OOE"]["GAMMA"] * config["OOE"]["GAE_LAMBDA"] * (1 - next_done) * gae
        return (gae, value, done), gae
    _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val, last_done), traj_batch, reverse=True, unroll=16)
    return advantages, advantages + traj_batch.value

# CALCULATE ADVANTAGE OOE
ac_in_OOE = (last_obs_OOE[np.newaxis, :], last_done[np.newaxis, :])
_, _, last_val_OOE = OOE_network.apply(train_state_OOE.params, init_OOE_hstate, ac_in_OOE)
last_val_OOE = last_val_OOE.squeeze(0)

advantages_OOE, targets_OOE = _calculate_gae(traj_batch[0], last_val_OOE, last_done)

'''

# CACLCULATE ADVANTAGE TAgent
ac_in_TAgent = (last_obs_TAgent[np.newaxis, :], last_done[np.newaxis, :])
_, _, last_val_TAgent = TAgent_network.apply(train_state_TAgent.params, init_TAgent_hstate, ac_in_TAgent)
last_val_TAgent = last_val_TAgent.squeeze(0)

advantages_TAgent, targets_TAgent = _calculate_gae(traj_batch[1], last_val_TAgent, last_done)

'''

#%%

# UPDATE NETWORK FUNCTION OOE
def _update_epoch_OOE(update_state, unused):
    def _update_minbatch(train_state, batch_info):
        init_hstate, traj_batch, advantages, targets = batch_info

        def _loss_fn(params, init_hstate, traj_batch, gae, targets):
            # RERUN NETWORK
            _, pi, value = OOE_network.apply(
                params, init_hstate, (traj_batch.obs, traj_batch.done)
            )
            log_prob = pi.log_prob(traj_batch.action)

            # CALCULATE VALUE LOSS
            value_pred_clipped = traj_batch.value + (
                value - traj_batch.value
            ).clip(-config["OOE"]["CLIP_EPS"], config["OOE"]["CLIP_EPS"])
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
                    1.0 - config["OOE"]["CLIP_EPS"],
                    1.0 + config["OOE"]["CLIP_EPS"],
                )
                * gae
            )
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
            loss_actor = loss_actor.mean()
            entropy = pi.entropy().mean()

            total_loss = (
                loss_actor
                + config["OOE"]["VF_COEF"] * value_loss
                - config["OOE"]["ENT_COEF"] * entropy
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

#%%
           
# UPDATE OOE - NETWORK
update_state_OOE = (
    train_state_OOE,
    initial_hstate_OOE,
    traj_batch[0],
    advantages_OOE,
    targets_OOE,
    rng,
)

update_state_OOE, loss_info_OOE = jax.lax.scan(
    _update_epoch_OOE, update_state_OOE, None, config["UPDATE_EPOCHS"]
)

# grad_norm=jnp.mean(loss_info[1])
train_state_OOE = update_state_OOE[0]
# metric = (traj_batch.info,train_state.params,grad_norm)
metric_OOE = (traj_batch[0].info,train_state_OOE.params)
rng = update_state_OOE[-1]


#%%


price_hist = info["Price_history"][0]
index = int(jnp.where(price_hist == 0)[0][0])
price_hist = price_hist[0][:index]
buy_hist = info["Buy_history"][0][0][:index]
sell_hist = info["Sell_history"][0][0][:index]
OOE_history = info["OOE_history"][0][0][:index]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(price_hist, label='Price History')

# Adding green dots for buys and red dots for sells
for i in range(index):
    if buy_hist[i] != 0:
        plt.scatter(i, price_hist[i]+30, color='green', label='Buy' if i == 0 else "")
        plt.text(i, price_hist[i]+35, f'{buy_hist[i]}', fontsize=9, ha='right')
    if sell_hist[i] != 0:
        plt.scatter(i, price_hist[i]-30, color='red', label='Sell' if i == 0 else "")
        plt.text(i, price_hist[i]-35, f'{sell_hist[i]}', fontsize=9, ha='left')
    if OOE_history[i] != 0:
        plt.scatter(i, price_hist[i]-60, color='blue', label='OOE Execution' if i == 0 else "")
        plt.text(i, price_hist[i]-65, f'{OOE_history[i]}', fontsize=9, ha='left')

# Adding labels and legend
plt.xlabel('Timestep')
plt.ylabel('Price')
plt.title('Price History with Buy/Sell Points')
plt.legend()
plt.grid(True)
plt.show()

# Save the plot
rounded_timestep = 100
plot_filename = os.path.join(config['PICTURE_DIR'], f"checkpoint_{rounded_timestep}.png")
os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
plt.savefig(plot_filename)
plt.show()


#%%

info = traj_batch[0].info

price_hist = info["Price_history"][0]
index = int(jnp.where(price_hist == 0)[0][0])
price_hist = price_hist[:index]
buy_hist = info["Buy_history"][0][:index]
sell_hist = info["Sell_history"][0][:index]
OOE_history = info["OOE_history"][0][:index]

#%%

price_hist.shape

price_hist = info["Price_history"][0]
index = int(jnp.where(price_hist == 0)[0][0])
price_hist = price_hist[:index]
buy_hist = info["Buy_history"][0][:index]
sell_hist = info["Sell_history"][0][:index]
OOE_history = info["OOE_history"][0][:index]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(price_hist, label='Price History')

# Adding green dots for buys and red dots for sells
for i in range(index):
    if buy_hist[i] != 0:
        plt.scatter(i, price_hist[i]+30, color='green', label='Buy' if i == 0 else "")
        plt.text(i, price_hist[i]+35, f'{buy_hist[i]}', fontsize=9, ha='right')
    if sell_hist[i] != 0:
        plt.scatter(i, price_hist[i]-30, color='red', label='Sell' if i == 0 else "")
        plt.text(i, price_hist[i]-35, f'{sell_hist[i]}', fontsize=9, ha='left')
    if OOE_history[i] != 0:
        plt.scatter(i, price_hist[i]-60, color='blue', label='OOE Execution' if i == 0 else "")
        plt.text(i, price_hist[i]-65, f'{OOE_history[i]}', fontsize=9, ha='left')

# Adding labels and legend
plt.xlabel('Timestep')
plt.ylabel('Price')
plt.title('Price History with Buy/Sell Points')
plt.legend()
plt.grid(True)

# Save the plot
#rounded_timestep = (timesteps[0] // 1000) * 1000
rounded_timestep = 8
plot_filename = os.path.join(config['PICTURE_DIR'], f"checkpoint_{rounded_timestep}.png")
os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
plt.savefig(plot_filename)
plt.show()




#%%

liste = list(info["Price_history"][0])
index = liste.index(0)


#%%

