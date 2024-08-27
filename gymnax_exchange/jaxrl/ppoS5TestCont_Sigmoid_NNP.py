#%% Import Libraries

import os
import sys
import time
import matplotlib.pyplot as plt

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

#from gymnax_exchange.jaxen.exec_env_test_TWAP import ExecutionEnv
from gymnax_exchange.jaxen.exec_env_test_Sigmoid import ExecutionEnv

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

#% Define the Network (ActorCriticS5) & Transition Class

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
    "AGENTS": ["OOE", "TA"],
    "NrTAgents": 2,
    "wandb_RUN_NAME": 'scarlet-frog-168', #'rare-glitter-166',
    "NUM_ENVS": 1,            # reduced from 500
    #"TOTAL_TIMESTEPS": 1e2,     # reduced from 1e8
    #"NUM_MINIBATCHES": 1,       # (they also tried 4 instead)
    #"UPDATE_EPOCHS": 5,         # slightly confusing as it is of course used in the network update, not sure though if we would be forced to use the same for all...
    "NUM_STEPS": 200,
    "ENV_NAME": "alphatradeExec-v0",
    "WINDOW_INDEX": -1,
    "DEBUG": False,     # changed to False
    "wandbOn": False,
    'seed': 0,
    "MAX_Duration_in_Min": 5,
    "PARAMSFOLDER": '/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/params',
    "ATFOLDER": '/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/',
    "RESULTS_FILE":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/results_file_"+f"{timestamp}",
    "CHECKPOINT_DIR":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/checkpoints_"+f"{timestamp}",
    "OOE": {
        "TASKSIDE":'sell',
        "TASK_SIZE":200,
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
        "Episode_Time": 60*20, #60seconds times 20 minutes = 1200seconds * 10(scaling factor) = 12000
        "ANNEAL_LR": True,
    },
    "TAgent": {
        "Dimensions": 1,
        "TASK_SIZE": 200,
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

device = jax.devices()[0]
rng = jax.device_put(jax.random.PRNGKey(config['seed']), device)

config["OOE"]["NUM_STEPS"] = config["NUM_STEPS"]
config["TAgent"]["NrTAgents"] = config["NrTAgents"]
config["TAgent"]["TASK_SIZE"] = jnp.array([config["TAgent"]["TASK_SIZE"]*2] * config["NrTAgents"])
config["TAgent"]["gamma"] = jax.random.uniform(rng, (config["NrTAgents"],), minval=0.90, maxval=0.99)
config["TAgent"]["omega"] = jax.random.uniform(rng, (config["NrTAgents"],), minval=0.50, maxval=0.90)
config['OOE']['TASKSIDE_INT'] = -1 if config['OOE']['TASKSIDE'] else 1
config['OOE']['NUM_STEPS'] = config['NUM_STEPS']

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

#% Declare Environment & Initialize RNG

env = ExecutionEnv(config["ATFOLDER"], config["WINDOW_INDEX"], config['OOE'], config['TAgent'])
env_params = env.default_params
env = LogWrapper(env)

device = jax.devices()[0]
rng = jax.device_put(jax.random.PRNGKey(0), device)

#% OOE - INIT NETWORK WITH PRE-TRAINED PARAMETERS

OOE_network = ActorCriticS5(env.action_space(params_OOE).shape[0], config=config)
init_OOE_x = (
    jnp.zeros(
        (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
    ),
    jnp.zeros((1, config["NUM_ENVS"])),
)
init_OOE_hstate = StackedEncoderModel.initialize_carry(config["NUM_ENVS"], ssm_size, n_layers)
OOE_network_params = params_OOE  # Directly use loaded params
OOE_tx = optax.set_to_zero()    # Just to provide a learning schedule - not used
test_state_OOE = TrainState.create(
    apply_fn=OOE_network.apply,
    params=OOE_network_params,
    tx = OOE_tx,
)

#% TAgent - INIT NETWORK WITH PRE-TRAINED PARAMETERS

TAgent_network = ActorCriticS5(config["TAgent"]["Dimensions"], config=config)
shapeAdj = env.observation_space(env_params).shape
init_TAgent_x = (
    jnp.zeros(
        (1, config["NrTAgents"], *env.observation_space(env_params).shape)
    ),
    jnp.zeros((1, config["NrTAgents"])),
)
init_TAgent_hstate = StackedEncoderModel.initialize_carry(config["NrTAgents"], ssm_size, n_layers)
TAgent_network_params = params_TA  # Directly use loaded params
TA_tx = optax.set_to_zero()    # Just to provide a learning schedule - not used
test_state_TAgent = TrainState.create(
    apply_fn=TAgent_network.apply,
    params=TAgent_network_params,
    tx = TA_tx,
)

#% ENV - INIT ENV

rng, _rng = jax.random.split(rng)
reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
# env_state now contains lists with integers instead of pure integers for the variables of the TAgent
obsv_OOE, obsv_TAgent = obsv[0], obsv[1] # HERE 1
rng, _rng = jax.random.split(rng)

runner_state = (
    env_state,
    jnp.zeros((config["NUM_ENVS"]), dtype=bool),
    _rng,
    [test_state_OOE, obsv_OOE, init_OOE_hstate],
    [test_state_TAgent, obsv_TAgent, init_TAgent_hstate]
)

#%% Test LOOP


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
    OOE_action = OOE_action.squeeze(0)
    
    # TAgent SELECT ACTION
    rng, _rng = jax.random.split(rng)
    #ac_in_I = duplicated_array = jnp.repeat(last_obs_OOE[np.newaxis, :], config["NrTAgents"], axis=1) #This here needs to be adj and replaced with real obs (to do so adjust reset_env and get_obs) HERE 1.1
    ac_in_I = last_obs_TAgent[np.newaxis, :].squeeze(0)
    ac_in_II = jnp.tile(last_done[np.newaxis, :], (1, config["NrTAgents"]))
    TAgent_hstate_, TAgent_pi, TAgent_value = TAgent_network.apply(test_state_TAgent.params, TAgent_hstate, (ac_in_I, ac_in_II))
    TAgent_action = TAgent_pi.sample(seed=_rng)
    TAgent_action = TAgent_action.squeeze(0) 
    TAgent_action = jnp.transpose(TAgent_action)
    

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, config["NUM_ENVS"])
    
    obsv, env_state, reward, done, info = jax.vmap(
        env.step, in_axes=(0, 0, 0, 0, None)
    )(rng_step, env_state, OOE_action, TAgent_action, env_params)
    
    obsv_OOE, obsv_TAgent = obsv[0], obsv[1]
    reward_OOE, reward_TAgent = reward[0], reward[1]
    
    #deleted params:
    TAgent_log_prob, OOE_log_prob = 0.1, 0.1
    
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

#%% Run Step

info = traj_batch[0][-1]

priceReal = info['Price'].flatten()
priceNoIm = info['Price_NoImpact'].flatten()
BestAsk = info['BestAsk'].flatten()
BestBid = info['BestBid'].flatten()
OOE_quant = info['OOE_quant'].flatten()
OOE_price = info['OOE_price'].flatten()
OOE_task = info['OOE_task'].flatten()
TA_quantBuy = info['TA_quantBuy'].squeeze()
TA_quantSell = info['TA_quantSell'].squeeze()
TA_price = info['TA_price'].squeeze()
TA_inventory = info['TA_inventory'].squeeze()
RelPriceDelta = info['RelPriceDelta'].flatten()

OOE_quant.sum()
OOE_quant

#%% Chart Price History + Origianl Price + Bid & Ask + OOE executions

# Assuming info dictionary has already been loaded with the data
marg = 100

# Prepare the figure and axis
plt.figure(figsize=(14, 7))
plt.plot(priceReal, label='Real Price', linewidth=2)
plt.plot(priceNoIm, label='Price without Impact', linewidth=2)
plt.plot(BestAsk, label='Ask Price', linestyle='--', linewidth=1, color='b')
plt.plot(BestBid, label='Bid Price', linestyle='--', linewidth=1, color='b')
plt.ylim(min(priceReal) - marg, max(priceReal) + marg)

# Identifying points where OOE trades occur
ooe_trade_indices = np.where(OOE_quant != 0)

# Plotting the OOE trades on the graph
plt.scatter(ooe_trade_indices, OOE_price[ooe_trade_indices], color='red', label='OOE Trades', marker='x', s=100)

# Adding labels and legend
plt.title('Real vs. No-Impact Price with OOE Trades')
plt.xlabel('Time (Steps)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

# Calculate the volume-weighted average price (VWAP)
non_zero_indices = OOE_quant != 0
vwap = np.sum(OOE_price[non_zero_indices] * OOE_quant[non_zero_indices]) / np.sum(OOE_quant[non_zero_indices])
print(f"The volume-weighted average price of OOE trades is: {vwap:.2f}")

#%% Print Executed Volumes and price impact

# Calculate the net volume for OOE Agent
OOE_net_volume = OOE_quant * OOE_task

TA_total_buy = np.sum(TA_quantBuy, axis=1)  # Sum across the second dimension (agents)
TA_total_sell = np.sum(TA_quantSell, axis=1)  # Sum across the second dimension (agents)

# First chart: Volumes
plt.figure(figsize=(14, 10))

plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
plt.bar(np.arange(len(OOE_net_volume)), OOE_net_volume, color='blue', label='OOE Agent Volume')
plt.bar(np.arange(len(TA_total_buy)), TA_total_buy, color='green', alpha=0.5, label='TAgent Total Buy Volume')
plt.bar(np.arange(len(TA_total_sell)), -TA_total_sell, color='red', alpha=0.5, label='TAgent Total Sell Volume')
plt.title('Executed Volume by Agents')
plt.xlabel('Time (Steps)')
plt.ylabel('Volume')
plt.legend()
plt.grid(True)

# Second chart: Price Delta
plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
plt.plot(RelPriceDelta, color='green', label='Relative Price Delta')
plt.title('Relative Price Delta Over Time')
plt.xlabel('Time (Steps)')
plt.ylabel('Price Delta')
plt.legend()
plt.grid(True)

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

#%%

import pandas as pd
import numpy as np

# Initialize the results DataFrame
columns = ['Agent', 'PnL', 'Total Shares Bought', 'Total Shares Sold']
results = pd.DataFrame(columns=columns)

# Iterate over each TAgent (agents are columns in TA_quantBuy and TA_quantSell)
for i in range(TA_quantBuy.shape[1]):  # Iterate across the second dimension (agent axis)
    # Extracting data for each agent
    agent_quant_buy = TA_quantBuy[:, i]
    agent_quant_sell = TA_quantSell[:, i]
    agent_price = TA_price[:, i]

    # Calculating total shares bought and sold
    agent_shares_bought = np.sum(agent_quant_buy)
    agent_shares_sold = np.sum(agent_quant_sell)
    agent_net_shares = agent_shares_bought - agent_shares_sold

    # Initial PnL calculation based on transactions made
    agent_PnL = np.sum(agent_price * agent_quant_buy) - np.sum(agent_price * agent_quant_sell)

    # Determine the price for final adjustments:
    final_price = BestBid[-1] if agent_net_shares > 0 else BestAsk[-1]

    # Adjusting the PnL for the remaining shares at the final price
    # Subtract if net shares are positive, add if negative.
    agent_PnL += abs(agent_net_shares) * final_price * (-1 if agent_net_shares > 0 else 1)

    # Create a row for this agent and append it to the results DataFrame
    new_row = pd.DataFrame({
        'Agent': [f'TAgent {i+1}'],
        'PnL': [agent_PnL],
        'Total Shares Bought': [agent_shares_bought],
        'Total Shares Sold': [agent_shares_sold]
    })
    results = pd.concat([results, new_row], ignore_index=True)

# Print the DataFrame
print(results)

#%%



