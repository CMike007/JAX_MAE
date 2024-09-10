#%% Import Libraries
import pandas as pd
import numpy as np

import os
import sys
import time
import matplotlib.pyplot as plt

import chex
import flax
from flax.core import frozen_dict
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

#from gymnax_exchange.jaxen.exec_env_test_TWAP import ExecutionEnv
from gymnax_exchange.jaxen.exec_env_test_Sigmoid import ExecutionEnv

timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
os.environ["NCCL_P2P_DISABLE"] = "1"

config = {
    "AGENTS": ["OOE", "TA"],
    "NrTAgents": 30,
    "wandb_RUN_NAME": 'floral-durian-202', #'zany-shape-198',
    "NUM_ENVS": 1,            # reduced from 500
    #"TOTAL_TIMESTEPS": 1e2,     # reduced from 1e8
    #"NUM_MINIBATCHES": 1,       # (they also tried 4 instead)
    #"UPDATE_EPOCHS": 5,         # slightly confusing as it is of course used in the network update, not sure though if we would be forced to use the same for all...
    "NUM_STEPS": 180,
    "ENV_NAME": "alphatradeExec-v0",
    "WINDOW_INDEX": -1,
    "DEBUG": False,     # changed to False
    "wandbOn": False,
    'seed': 2,
    "MAX_Duration_in_Min": 5,
    "PARAMSFOLDER": '/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/params',
    "ATFOLDER": '/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/',
    "RESULTS_FILE":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/results_file_"+f"{timestamp}",
    "CHECKPOINT_DIR":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/checkpoints_"+f"{timestamp}",
    "OOE": {
        "TASKSIDE":'sell',
        "TASK_SIZE":40000,
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
        "Episode_Time": 60*30, #60seconds times 20 minutes = 1200seconds * 10(scaling factor) = 12000
        "ANNEAL_LR": True,
    },
    "TAgent": {
        "Dimensions": 1,
        "TASK_SIZE": 100,
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

#%% Define load_model function % load the trained agent parameters

def load_model(config, agent):
    runname = config['wandb_RUN_NAME']
    folder = config['PARAMSFOLDER']
    params_file_name = f'{folder}/Aparam_{runname}_{agent}'

    with open(params_file_name, 'rb') as f:
        restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        print(f"{agent} Params restored")
        
        # Reshape 'log_std' based on its shape
        if 'log_std' in restored_params['params']:
            log_std_shape = restored_params['params']['log_std'].shape
            def adjust_4Core(dic):
                dic['bias'] = dic['bias'][0]
                dic['kernel'] = dic['kernel'][0]
                
            def adjust_4CoreNN(dic):
                dic['B'] = dic['B'][0]
                dic['C'] = dic['C'][0]
                dic['D'] = dic['D'][0]
                dic['Lambda_im'] = dic['Lambda_im'][0]
                dic['Lambda_re'] = dic['Lambda_re'][0]
                dic['log_step'] = dic['log_step'][0]

            # If 'log_std' shape is (4, 1) for TA, reshape to (1,) using mean
            if log_std_shape == (4, 1):
                reshaped_log_std = restored_params['params']['log_std'].mean(axis=0).reshape((1,))
                restored_params['params']['log_std'] = reshaped_log_std
                for key in restored_params['params']:
                    if key != 'log_std' and key != 's5':
                        adjust_4Core(restored_params['params'][key])
                    elif key == 's5':
                        for layer in restored_params['params']['s5']:
                            adjust_4Core(restored_params['params']['s5'][layer]['out2'])
                            adjust_4CoreNN(restored_params['params']['s5'][layer]['seq'])
                print("TA adjusted")
            
            # If 'log_std' shape is (4, 4) for OOE, reshape to (4,) using mean
            elif log_std_shape == (4, 4):
                reshaped_log_std = restored_params['params']['log_std'].mean(axis=1).reshape((4,))
                restored_params['params']['log_std'] =  reshaped_log_std
                for key in restored_params['params']:
                    if key != 'log_std' and key != 's5':
                        adjust_4Core(restored_params['params'][key])
                    elif key == 's5':
                        for layer in restored_params['params']['s5']:
                            adjust_4Core(restored_params['params']['s5'][layer]['out2'])
                            adjust_4CoreNN(restored_params['params']['s5'][layer]['seq'])           
                print("OOOE adjusted")
                
        return restored_params
    
params_OOE = load_model(config, "OOE")
params_TA = load_model(config, "TA")
         

#%% Iniitate Environment and define helpfunctions

#% Declare Environment & Initialize RNG
env = ExecutionEnv(config["ATFOLDER"], config["WINDOW_INDEX"], config['OOE'], config['TAgent'])
env_params = env.default_params
env = LogWrapper(env)

device = jax.devices()[0]
rng = jax.device_put(jax.random.PRNGKey(0), device)

#% OOE - INIT NETWORK WITH PRE-TRAINED PARAMETERS

def init_network(rng):
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
    
    return runner_state, OOE_network, TAgent_network

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


#% Definition of some magic Graphs... 

def HarryPlotter(priceReal, priceNoIm, OOE_price, OOE_quant, BestBid, BestAsk):
    asks = [int(max(ask, price+10)) for ask, price in zip(BestAsk, priceReal)]
    bids = [int(min(bid, price-1)) for bid, price in zip(BestBid, priceReal)]

    # Assuming info dictionary has already been loaded with the data
    marg = 0
    
    # Prepare the figure and axis
    plt.figure(figsize=(14, 7))
    plt.plot(priceReal, label='Real Price', linewidth=2)
    plt.plot(priceNoIm, label='Price without Impact', linewidth=2)
    plt.plot(asks, label='Ask Price', linestyle='--', linewidth=1, color='b')
    plt.plot(bids, label='Bid Price', linestyle='--', linewidth=1, color='b')
    plt.ylim(min(priceReal) - marg, max(priceReal) + marg)
    
    # Identifying points where OOE trades occur
    ooe_trade_indices = np.where(OOE_quant != 0)
    
    # Plotting the OOE trades on the graph
    plt.scatter(ooe_trade_indices, OOE_price[ooe_trade_indices], color='red', label='OOE Trades', marker='x', s=100)
    
    # Adding labels and legend
    plt.title('Real vs. No-Impact Price with OOE Trades')
    plt.xlabel('Time (Steps)')
    plt.ylabel('Price time 100.000')
    plt.legend()
    
    # Make the whole thing look a little bit better
    plt.grid(True, which='major', linestyle='-', linewidth='0.4', color='1')
    plt.grid(True, which='minor', linestyle=':', linewidth='0.4', color='0.8')
    plt.minorticks_on()
    
    # Improving aesthetics with spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    
    
    # Display the plot
    plt.show()
    
    # Calculate the volume-weighted average price (VWAP)
    non_zero_indices = OOE_quant != 0
    vwap = np.sum(OOE_price[non_zero_indices] * OOE_quant[non_zero_indices]) / np.sum(OOE_quant[non_zero_indices])
    print(f"Shares Executed: {sum(OOE_quant[non_zero_indices]):.2f}")
    print(f"The volume-weighted average price of OOE trades is: {vwap:.2f}")
    print(f"Analysis for the Window {config['WINDOW_INDEX']}")


def Graphindor(OOE_quant, OOE_task, TA_quantBuy, TA_quantSell, RelPriceDelta):
    # Calculate the net volume for OOE Agent
    OOE_net_volume = OOE_quant * OOE_task
    
    TA_total_buy = np.sum(TA_quantBuy, axis=1)  # Sum across the second dimension (agents)
    TA_total_sell = np.sum(TA_quantSell, axis=1)  # Sum across the second dimension (agents)
    
    # First chart: Volumes
    plt.figure(figsize=(14, 7))
    
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
    plt.bar(np.arange(len(OOE_net_volume)), OOE_net_volume, color='blue', label='OOE Agent Volume')
    plt.bar(np.arange(len(TA_total_buy)), TA_total_buy, color='green', alpha=0.5, label='TAgent Total Buy Volume')
    plt.bar(np.arange(len(TA_total_sell)), -TA_total_sell, color='red', alpha=0.5, label='TAgent Total Sell Volume')
    plt.title('Executed Volume by Agents')
    plt.xlabel('Time (Steps)')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)
    
    deltaInPips = [delta / 10 for delta in RelPriceDelta]
    # Second chart: Price Delta
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
    plt.plot(deltaInPips, color='green', label='Relative Price Delta')
    plt.title('Relative Price Delta Over Time')
    plt.xlabel('Time (Steps)')
    plt.ylabel('Price Delta in pips')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    
    # Make the whole thing look a little bit better
    plt.grid(True, which='major', linestyle='-', linewidth='0.4', color='1')
    plt.grid(True, which='minor', linestyle=':', linewidth='0.4', color='0.8')
    plt.minorticks_on()
    
    # Improving aesthetics with spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    
    
    plt.show()


def ExPnLiarmus(TA_quantBuy, TA_quantSell, TA_price, BestAsk, BestBid):
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
        
    print(results)
    return results


def HarryPlotterII(priceReal, priceNoIm, BestBid, BestAsk, TA_quantBuy, TA_quantSell, TA_price):
    # Prepare ask and bid prices to plot
    asks = [int(max(ask, price + 10)) for ask, price in zip(BestAsk, priceReal)]
    bids = [int(min(bid, price - 1)) for bid, price in zip(BestBid, priceReal)]

    # Prepare the figure and axis
    plt.figure(figsize=(14, 7))
    plt.plot(priceReal, label='Real Price', linewidth=2)
    plt.plot(priceNoIm, label='Price without Impact', linewidth=2)
    plt.plot(asks, label='Ask Price', linestyle='--', linewidth=1, color='b')
    plt.plot(bids, label='Bid Price', linestyle='--', linewidth=1, color='b')

    # Setting y-limits with some margin
    marg = 10
    plt.ylim(min(priceReal) - marg, max(priceReal) + marg)

    # Plot TA buy trades
    for t in range(TA_quantBuy.shape[0]):  # Iterate over each timestep
        for a in range(TA_quantBuy.shape[1]):  # Iterate over each agent
            if TA_quantBuy[t, a] != 0:  # Check if there is a buy trade
                plt.scatter(t, TA_price[t, a], color='green', label='TA Buy Trades' if t == 0 and a == 0 else "", marker='o', s=80)

    # Plot TA sell trades
    for t in range(TA_quantSell.shape[0]):  # Iterate over each timestep
        for a in range(TA_quantSell.shape[1]):  # Iterate over each agent
            if TA_quantSell[t, a] != 0:  # Check if there is a sell trade
                plt.scatter(t, TA_price[t, a], color='purple', label='TA Sell Trades' if t == 0 and a == 0 else "", marker='^', s=80)

    # Adding labels and legend
    plt.title('Real vs. No-Impact Price with TAgent Buy and Sell Trades')
    plt.xlabel('Time (Steps)')
    plt.ylabel('Price times 100,000')
    plt.legend()

    # Improve grid aesthetics
    plt.grid(True, which='major', linestyle='-', linewidth='0.4', color='1')
    plt.grid(True, which='minor', linestyle=':', linewidth='0.4', color='0.8')
    plt.minorticks_on()

    # Improving aesthetics with spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')

    # Display the plot
    plt.show()


def HistoGramPnL(PnLArray):
    totalPnLTAs_array = np.array(PnLArray)
    
    # Calculate the mean of the data
    mean_pnl = np.mean(totalPnLTAs_array)
    
    # Plotting the histogram
    plt.figure(figsize=(12, 6))
    plt.hist(totalPnLTAs_array, bins=30, color='skyblue', edgecolor='black')
    
    # Add a vertical line for the mean
    plt.axvline(mean_pnl, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {mean_pnl:.2f}')
    
    # Add titles and labels
    plt.title('Distribution of Total Profit and Loss (PnL)', fontsize=16)
    plt.xlabel('PnL (in currency units)', fontsize=14)
    plt.ylabel('Frequency (Counts)', fontsize=14)
    
    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend()
    
    # Show the plot
    plt.show()
    
    # Conduct a one-sample t-test to check if the mean is significantly different from 0
    t_statistic, p_value = stats.ttest_1samp(totalPnLTAs_array, 0)
    
    print(f"T-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    # Interpretation of the result
    if p_value < 0.05:
        print("The mean deviation from 0 is statistically significant (p < 0.05).")
    else:
        print("The mean deviation from 0 is not statistically significant (p >= 0.05).")
    

#%% Test LOOP - run one testrond


rng, _rng = jax.random.split(rng)
runner_state, OOE_network, TAgent_network = init_network(rng)

runner_state, traj_batch = jax.lax.scan(
    _env_step, runner_state, None, config["NUM_STEPS"]
)

info = traj_batch[0][-1]

dones = info['done'].flatten()
step = info['current_step'].flatten()
priceReal = info['Price'].flatten()
quant_executed = info['quant_executed'].flatten()
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
VWAPPrice = info['vwap_rm'][-1].flatten()
Slippage = info['slippage_rm']

HarryPlotter(priceReal, priceNoIm, OOE_price, OOE_quant, BestBid, BestAsk)
Graphindor(OOE_quant, OOE_task, TA_quantBuy, TA_quantSell, RelPriceDelta)
TAgent_Data = ExPnLiarmus(TA_quantBuy, TA_quantSell, TA_price, BestAsk, BestBid)
HarryPlotterII(priceReal, priceNoIm, BestBid, BestAsk, TA_quantBuy, TA_quantSell, TA_price)

#%% Track multiple rounds - RL OOE

Init_Price, price_init_0 = [], []
OOE_AvgPrices, OOE_Quants, OOE_VWAPPrice, OOE_Shortfall, OOE_ExecTime, SlippageL = [], [], [], [], [], []
TAgent_Actions, totoalPnLTAs = [], []
'''
df = pd.read_csv('/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/data/performanceTrackNew.csv')
Init_Price = df['Init_Price'].tolist()
OOE_AvgPrices = df['OOE_AvgPrices'].tolist()
OOE_Quants = df['OOE_Quants'].tolist()
OOE_VWAPPrice = df['OOE_VWAPPrice'].tolist()
OOE_Shortfall = df['OOE_Shortfall'].tolist()
OOE_ExecTime = df['OOE_ExecTime'].tolist()
SlippageL

'''

for i in range(1, 265): #264
    print(f"Round {i}")
    
    rng, _rng = jax.random.split(rng)
    runner_state, OOE_network, TAgent_network = init_network(rng)
    
    runner_state, traj_batch = jax.lax.scan(
        _env_step, runner_state, None, config["NUM_STEPS"]
    )

    info = traj_batch[0][-1]
    OOE_quant = info['OOE_quant'].flatten()
    OOE_price = info['OOE_price'].flatten()
    OOE_task = info['OOE_task'].flatten()
    price_init = info['Price'][0].flatten()
    VWAP = info['vwap_rm'][-1].flatten()

    priceReal = info['Price'].flatten()
    priceNoIm = info['Price_NoImpact'].flatten()
    BestAsk = info['BestAsk'].flatten()
    BestBid = info['BestBid'].flatten()
    Slippage = info['slippage_rm'][-1].flatten()
    RelPriceDelta = info['RelPriceDelta'].flatten()
    
    TA_quantBuy = info['TA_quantBuy'].squeeze()
    TA_quantSell = info['TA_quantSell'].squeeze()
    TA_price = info['TA_price'].squeeze()
    
    timeweightedQuant = sum((x * (i + 1) for i, x in enumerate(OOE_quant)))
    ExecTime = round(float(timeweightedQuant / (len(OOE_quant) * sum(OOE_quant))), 3)

    # Calculate the volume-weighted average price (VWAP)
    non_zero_indices = OOE_quant != 0
    vwap = np.sum(OOE_price[non_zero_indices] * OOE_quant[non_zero_indices]) / np.sum(OOE_quant[non_zero_indices])
    quant_exec = sum(OOE_quant[non_zero_indices])
    
    if int(vwap) > 35000:
        OOE_AvgPrices.append(int(vwap))
        OOE_Quants.append(int(quant_exec))
        Init_Price.append(int(price_init[0]))
        OOE_VWAPPrice.append(int(VWAP[0]) * 10)
        OOE_Shortfall.append(int(vwap) - int(price_init[0]))
        OOE_ExecTime.append(ExecTime)
        SlippageL.append(int(Slippage[0]))
        
        HarryPlotter(priceReal, priceNoIm, OOE_price, OOE_quant, BestBid, BestAsk)
        Graphindor(OOE_quant, OOE_task, TA_quantBuy, TA_quantSell, RelPriceDelta)
        TAgent_Data = ExPnLiarmus(TA_quantBuy, TA_quantSell, TA_price, BestAsk, BestBid)
        
        TAgent_Actions.append(TAgent_Data)
        totoalPnLTAs.append(sum(TAgent_Data['PnL']))

    else:
        print(int(vwap))
        
        
    data = {
        'Init_Price': Init_Price,
        'OOE_AvgPrices': OOE_AvgPrices,
        'OOE_Quants': OOE_Quants,
        'OOE_VWAPPrice': OOE_VWAPPrice,
        'OOE_Shortfall': OOE_Shortfall,
        'OOE_ExecTime': OOE_ExecTime,
        'SlippageL': SlippageL,
        'TAgent_Actions': TAgent_Actions,
        'totoalPnLTAs': totoalPnLTAs
        
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.to_csv('/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/data/performanceTrackNew.csv', index=False)       


'''
df.to_csv('/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/data/performanceTrackNew.csv', index=False)       

print(f'StartPices: {Init_Price}')
print(f'OOE_AvgPrices: {OOE_AvgPrices}')
print(f'OOE_Quants: {OOE_Quants}')
print(f'OOE_VWAPPrice: {OOE_VWAPPrice}')
print(f'OOE_Shortfall: {OOE_Shortfall}')
print(f'ExecTime: {OOE_ExecTime}')
'''

HistoGramPnL(totoalPnLTAs)

print(f'Length StartPices: {len(Init_Price)}')
print(f'Length OOE_AvgPrices: {len(OOE_AvgPrices)}')
print(f'Length OOE_Quants: {len(OOE_Quants)}')
print(f'Length OOE_VWAPPrice: {len(OOE_VWAPPrice)}')
print(f'Length OOE_Shortfall: {len(OOE_Shortfall)}')
print(f'Length ExecTime: {len(OOE_ExecTime)}')

#%% running TWAP model

from gymnax_exchange.jaxen.exec_env_test_TWAP import ExecutionEnv

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

#% Running the model

runner_state, traj_batch = jax.lax.scan(
    _env_step, runner_state, None, config["NUM_STEPS"]
)


infoTWAP = traj_batch[0][-1]

priceReal = infoTWAP['Price'].flatten()
priceNoIm = infoTWAP['Price_NoImpact'].flatten()
BestAsk = infoTWAP['BestAsk'].flatten()
BestBid = infoTWAP['BestBid'].flatten()
OOE_quant = infoTWAP['OOE_quant'].flatten()
OOE_price = infoTWAP['OOE_price'].flatten()
OOE_task = infoTWAP['OOE_task'].flatten()
TA_quantBuy = infoTWAP['TA_quantBuy'].squeeze()
TA_quantSell = infoTWAP['TA_quantSell'].squeeze()
TA_price = infoTWAP['TA_price'].squeeze()
TA_inventory = infoTWAP['TA_inventory'].squeeze()
RelPriceDelta = infoTWAP['RelPriceDelta'].flatten()

#%% Chart Price History + Origianl Price + Bid & Ask + OOE executions in TWAP try

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
print(f"Shares Executed: {sum(OOE_quant[non_zero_indices]):.2f}")
print(f"The volume-weighted average price of OOE trades is: {vwap:.2f}")

#%% Print Executed Volumes and price impact in TWAP try

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

#%%T TAgent Actions in TWAP try

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


#%% Running a big loop to compare execution prices of RL and TWAP
import time
#from gymnax_exchange.jaxen.exec_env_test_TWAP import ExecutionEnv

timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
os.environ["NCCL_P2P_DISABLE"] = "1"

config = {
    "AGENTS": ["OOE", "TA"],
    "NrTAgents": 5,
    "wandb_RUN_NAME": 'jumping-grass-177', #'zany-shape-198',
    "NUM_ENVS": 1,            # reduced from 500
    #"TOTAL_TIMESTEPS": 1e2,     # reduced from 1e8
    #"NUM_MINIBATCHES": 1,       # (they also tried 4 instead)
    #"UPDATE_EPOCHS": 5,         # slightly confusing as it is of course used in the network update, not sure though if we would be forced to use the same for all...
    "NUM_STEPS": 180,
    "ENV_NAME": "alphatradeExec-v0",
    "WINDOW_INDEX": 1,
    "DEBUG": False,     # changed to False
    "wandbOn": False,
    'seed': 2,
    "MAX_Duration_in_Min": 5,
    "PARAMSFOLDER": '/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/params',
    "ATFOLDER": '/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/',
    "RESULTS_FILE":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/results_file_"+f"{timestamp}",
    "CHECKPOINT_DIR":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/checkpoints_"+f"{timestamp}",
    "OOE": {
        "TASKSIDE":'sell',
        "TASK_SIZE":20000,
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
        "Episode_Time": 60*30, #60seconds times 20 minutes = 1200seconds * 10(scaling factor) = 12000
        "ANNEAL_LR": True,
    },
    "TAgent": {
        "Dimensions": 1,
        "TASK_SIZE": 100,
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


#% Definition Funcitons:
    
    
def load_model(config, agent):
    runname = config['wandb_RUN_NAME']
    folder = config['PARAMSFOLDER']
    params_file_name = f'{folder}/Aparam_{runname}_{agent}'

    with open(params_file_name, 'rb') as f:
        restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        print(f"{agent} Params restored")
        
        # Reshape 'log_std' based on its shape
        if 'log_std' in restored_params['params']:
            log_std_shape = restored_params['params']['log_std'].shape
            def adjust_4Core(dic):
                dic['bias'] = dic['bias'][0]
                dic['kernel'] = dic['kernel'][0]
                
            def adjust_4CoreNN(dic):
                dic['B'] = dic['B'][0]
                dic['C'] = dic['C'][0]
                dic['D'] = dic['D'][0]
                dic['Lambda_im'] = dic['Lambda_im'][0]
                dic['Lambda_re'] = dic['Lambda_re'][0]
                dic['log_step'] = dic['log_step'][0]

            # If 'log_std' shape is (4, 1) for TA, reshape to (1,) using mean
            if log_std_shape == (4, 1):
                reshaped_log_std = restored_params['params']['log_std'].mean(axis=0).reshape((1,))
                restored_params['params']['log_std'] = reshaped_log_std
                for key in restored_params['params']:
                    if key != 'log_std' and key != 's5':
                        adjust_4Core(restored_params['params'][key])
                    elif key == 's5':
                        for layer in restored_params['params']['s5']:
                            adjust_4Core(restored_params['params']['s5'][layer]['out2'])
                            adjust_4CoreNN(restored_params['params']['s5'][layer]['seq'])
                print("TA adjusted")
            
            # If 'log_std' shape is (4, 4) for OOE, reshape to (4,) using mean
            elif log_std_shape == (4, 4):
                reshaped_log_std = restored_params['params']['log_std'].mean(axis=1).reshape((4,))
                restored_params['params']['log_std'] =  reshaped_log_std
                for key in restored_params['params']:
                    if key != 'log_std' and key != 's5':
                        adjust_4Core(restored_params['params'][key])
                    elif key == 's5':
                        for layer in restored_params['params']['s5']:
                            adjust_4Core(restored_params['params']['s5'][layer]['out2'])
                            adjust_4CoreNN(restored_params['params']['s5'][layer]['seq'])           
                print("OOOE adjusted")
                
        return restored_params
    
    
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

Init_Price, price_init_0 = [], []
OOE_AvgPrices, OOE_Quants, OOE_VWAPPrice, OOE_Shortfall, OOE_ExecTime = [], [], [], [], []
TWAP_AvgPrices, TWAP_Quants, TWAP_VWAPPrice, TWAP_Shortfall, TWAP_ExecTime = [], [], [], [], []

df = pd.read_csv('/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/data/performanceTrack.csv')
Init_Price = df['Init_Price'].tolist()
OOE_AvgPrices = df['OOE_AvgPrices'].tolist()
OOE_Quants = df['OOE_Quants'].tolist()
OOE_VWAPPrice = df['OOE_VWAPPrice'].tolist()
OOE_Shortfall = df['OOE_Shortfall'].tolist()
OOE_ExecTime = df['OOE_ExecTime'].tolist()


#%%
from gymnax_exchange.jaxen.exec_env_test_Sigmoid import ExecutionEnv
for i in range(246, 264):
   
    print(f'Round {i}')
    config["WINDOW_INDEX"] = i

    # RL execution
    params_OOE = load_model(config, "OOE")
    params_TA = load_model(config, "TA")

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
    
    
    runner_state, traj_batch = jax.lax.scan(
        _env_step, runner_state, None, config["NUM_STEPS"]
    )
        
    info = traj_batch[0][-1]
    OOE_quant = info['OOE_quant'].flatten()
    OOE_price = info['OOE_price'].flatten()
    price_init = info['Price'][0].flatten()
    VWAP = info['vwap_rm'][-1].flatten()
    
    timeweightedQuant = sum((x * (i + 1) for i, x in enumerate(OOE_quant)))
    ExecTime = round(float(timeweightedQuant / (len(OOE_quant) * sum(OOE_quant))), 3)

    # Calculate the volume-weighted average price (VWAP)
    non_zero_indices = OOE_quant != 0
    vwap = np.sum(OOE_price[non_zero_indices] * OOE_quant[non_zero_indices]) / np.sum(OOE_quant[non_zero_indices])
    quant_exec = sum(OOE_quant[non_zero_indices])
    
    if int(vwap) > 350000:
        OOE_AvgPrices.append(int(vwap))
        OOE_Quants.append(int(quant_exec))
        Init_Price.append(int(price_init[0]))
        OOE_VWAPPrice.append(int(VWAP[0]) * 10)
        OOE_Shortfall.append(int(vwap) - int(price_init[0]))
        OOE_ExecTime.append(ExecTime)
        
    data = {
        'Init_Price': Init_Price,
        'OOE_AvgPrices': OOE_AvgPrices,
        'OOE_Quants': OOE_Quants,
        'OOE_VWAPPrice': OOE_VWAPPrice,
        'OOE_Shortfall': OOE_Shortfall,
        'OOE_ExecTime': OOE_ExecTime
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    df.to_csv('/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/data/performanceTrack.csv', index=False)       

print(f'StartPices: {Init_Price}')
print(f'OOE_AvgPrices: {OOE_AvgPrices}')
print(f'OOE_Quants: {OOE_Quants}')
print(f'OOE_VWAPPrice: {OOE_VWAPPrice}')
print(f'OOE_Shortfall: {OOE_Shortfall}')
print(f'ExecTime: {OOE_ExecTime}')

print(f'Length StartPices: {len(Init_Price)}')
print(f'Length OOE_AvgPrices: {len(OOE_AvgPrices)}')
print(f'Length OOE_Quants: {len(OOE_Quants)}')
print(f'Length OOE_VWAPPrice: {len(OOE_VWAPPrice)}')
print(f'Length OOE_Shortfall: {len(OOE_Shortfall)}')
print(f'Length ExecTime: {len(OOE_ExecTime)}')

from gymnax_exchange.jaxen.exec_env_test_TWAP import ExecutionEnv
price_init_0 = []

    
for i in range(1, 45):
   
    print(f'Round {i}')
    config["WINDOW_INDEX"] = i
    # running TWAP model
               
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
    
    #% Running the model
    
    runner_state, traj_batch = jax.lax.scan(
        _env_step, runner_state, None, config["NUM_STEPS"]
    )
      
    infoTWAP = traj_batch[0][-1]
    TWAP_quant = infoTWAP['OOE_quant'].flatten()
    TWAP_price = infoTWAP['OOE_price'].flatten()
    price_init = infoTWAP['Price'][0].flatten()
    VWAP = infoTWAP['vwap_rm'][-1].flatten()
    
    # Calculate the volume-weighted average price (VWAP)
    non_zero_indices = TWAP_quant != 0
    vwap = np.sum(TWAP_price[non_zero_indices] * TWAP_quant[non_zero_indices]) / np.sum(TWAP_quant[non_zero_indices])
    quant_exec = sum(TWAP_quant[non_zero_indices])
        
    if int(vwap) > 350000:
        TWAP_AvgPrices.append(int(vwap))
        TWAP_Quants.append(int(quant_exec))
        price_init_0.append(int(price_init[0]))
        TWAP_VWAP_Price.append(int(VWAP[0]) * 10)
        TWAP_Shortfall.append(int(vwap) - int(price_init[0]))
        

#%%

try:
    df = pd.read_csv('/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/data/performanceTrackBase.csv')
    
    price_init_0 = df['price_init_0'].tolist()
    TWAP_AvgPrices = df['TWAP_AvgPrices'].tolist()
    TWAP_Quants = df['TWAP_Quants'].tolist()
    TWAP_VWAPPrice = df['TWAP_VWAPPrice'].tolist()
    TWAP_Shortfall = df['TWAP_Shortfall'].tolist()
    TWAP_ExecTime = df['TWAP_ExecTime'].tolist()
except:
    price_init_0 = []
    TWAP_AvgPrices, TWAP_Quants, TWAP_VWAPPrice, TWAP_Shortfall, TWAP_ExecTime = [], [], [], [], []

print(len(TWAP_AvgPrices))

from gymnax_exchange.jaxen.exec_env_test_TWAP import ExecutionEnv
for i in range(196, 264):
   
    print(f'Round {i}')
    config["WINDOW_INDEX"] = i

    # RL execution
    params_OOE = load_model(config, "OOE")
    params_TA = load_model(config, "TA")

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
    
    
    runner_state, traj_batch = jax.lax.scan(
        _env_step, runner_state, None, config["NUM_STEPS"]
    )
        
    info = traj_batch[0][-1]
    TWAP_quant = info['OOE_quant'].flatten()
    TWAP_price = info['OOE_price'].flatten()
    price_init = info['Price'][0].flatten()
    VWAP = info['vwap_rm'][-1].flatten()
    
    timeweightedQuant = sum((x * (i + 1) for i, x in enumerate(TWAP_quant)))
    ExecTime = round(float(timeweightedQuant / (len(TWAP_quant) * sum(TWAP_quant))), 3)

    # Calculate the volume-weighted average price (VWAP)
    non_zero_indices = TWAP_quant != 0
    vwap = np.sum(TWAP_price[non_zero_indices] * TWAP_quant[non_zero_indices]) / np.sum(TWAP_quant[non_zero_indices])
    quant_exec = sum(TWAP_quant[non_zero_indices])
    
    if int(vwap) > 350000:
        TWAP_AvgPrices.append(int(vwap))
        TWAP_Quants.append(int(quant_exec))
        price_init_0.append(int(price_init[0]))
        TWAP_VWAPPrice.append(int(VWAP[0]) * 10)
        TWAP_Shortfall.append(int(vwap) - int(price_init[0]))
        TWAP_ExecTime.append(ExecTime)
        
    data = {
        'price_init_0': price_init_0,
        'TWAP_AvgPrices': TWAP_AvgPrices,
        'TWAP_Quants': TWAP_Quants,
        'TWAP_VWAPPrice': TWAP_VWAPPrice,
        'TWAP_Shortfall': TWAP_Shortfall,
        'TWAP_ExecTime': TWAP_ExecTime
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    df.to_csv('/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/data/performanceTrackBase.csv', index=False)       

print(f'StartPices: {price_init_0}')
print(f'OOE_AvgPrices: {TWAP_AvgPrices}')
print(f'OOE_Quants: {TWAP_Quants}')
print(f'OOE_VWAPPrice: {TWAP_VWAPPrice}')
print(f'OOE_Shortfall: {TWAP_Shortfall}')
print(f'ExecTime: {TWAP_ExecTime}')

#%% Analysis Average Execution Price

import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
#TWAP_AvgPrices = [price + random.choice([-310, -490, -300, - 300, -400, 200, 100]) for price in OOE_AvgPrices]

'''
Files Available:
   Init_Price
   OOE_AvgPrices
   OOE_Quants
   OOE_VWAPPrice
   OOE_Shortfall
   OOE_ExecTime    
'''

# Adjustment to the actual price
TWAPPrices = TWAP_AvgPrices / 100000
OOEPrices = OOE_AvgPrices / 100000

# Combine both price lists and find the range for the histogram
all_prices = TWAPPrices + OOEPrices
min_price, max_price = min(all_prices), max(all_prices)

# Create bins for the histogram
bins = np.linspace(min_price, max_price, 20)  # Adjust the number of bins as needed

# Calculate the histogram for each list
TWAP_hist, _ = np.histogram(TWAPPrices, bins=bins)
OOE_hist, _ = np.histogram(OOEPrices, bins=bins)

# Plot the bar chart
bin_centers = (bins[:-1] + bins[1:]) / 2  # Center of each bin
bar_width = (bins[1] - bins[0]) * 0.4  # Adjust bar width for spacing

plt.bar(bin_centers - bar_width / 2, TWAP_hist, width=bar_width, color='red', alpha=0.6, label='TWAP')
plt.bar(bin_centers + bar_width / 2, OOE_hist, width=bar_width, color='blue', alpha=0.6, label='OOE')

# Calculate and plot vertical lines for the mean of each list
mean_TWAP = np.mean(TWAPPrices)
mean_OOE = np.mean(OOEPrices)
plt.axvline(x=mean_TWAP, color='red', linestyle='--', linewidth=2, label='Mean TWAP')
plt.axvline(x=mean_OOE, color='blue', linestyle='--', linewidth=2, label='Mean OOE')

# Add labels and legend
plt.xlabel('Prices')
plt.ylabel('Quantities (Occurrences)')
plt.title('Distribution of the Average Execution Price')
plt.legend()

# Make the whole thing look a little bit better
plt.grid(True, which='major', linestyle='-', linewidth='0.4', color='1')
plt.grid(True, which='minor', linestyle=':', linewidth='0.4', color='0.8')
plt.minorticks_on()

# Improving aesthetics with spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['bottom'].set_color('black')

# Show the plot
plt.show()

u_stat, p_value = stats.mannwhitneyu(TWAPPrices, OOEPrices, alternative='two-sided')

print("U-statistic:", u_stat)
print("P-value:", p_value)

# Interpret the result
if p_value < 0.05:
    print("There is a statistically significant difference between the two samples.")
else:
    print("There is no statistically significant difference between the two samples.")

#%% Analysis Median Execution Time Ratio

#TWAP_ExecTime = [time + random.choice([0.08, 0.07, 0.2, 0.3, 0.35, 0.12, 0.09]) for time in OOE_ExecTime]

'''
Files Available:
   Init_Price
   OOE_AvgPrices
   OOE_Quants
   OOE_VWAPPrice
   OOE_Shortfall
   OOE_ExecTime    
'''

# Combine both price lists and find the range for the histogram
all_prices = TWAP_ExecTime + OOE_ExecTime
min_price, max_price = min(all_prices), max(all_prices)

# Create bins for the histogram
bins = np.linspace(min_price, max_price, 20)  # Adjust the number of bins as needed

# Calculate the histogram for each list
TWAP_hist, _ = np.histogram(TWAP_ExecTime, bins=bins)
OOE_hist, _ = np.histogram(OOE_ExecTime, bins=bins)

# Plot the bar chart
bin_centers = (bins[:-1] + bins[1:]) / 2  # Center of each bin
bar_width = (bins[1] - bins[0]) * 0.4  # Adjust bar width for spacing

plt.bar(bin_centers - bar_width / 2, TWAP_hist, width=bar_width, color='red', alpha=0.6, label='TWAP')
plt.bar(bin_centers + bar_width / 2, OOE_hist, width=bar_width, color='blue', alpha=0.6, label='OOE')

# Calculate and plot vertical lines for the mean of each list
mean_TWAP = np.mean(TWAP_ExecTime)
mean_OOE = np.mean(OOE_ExecTime)
plt.axvline(x=mean_TWAP, color='red', linestyle='--', linewidth=2, label='Mean TWAP')
plt.axvline(x=mean_OOE, color='blue', linestyle='--', linewidth=2, label='Mean OOE')

# Add labels and legend
plt.xlabel('Proportion of Time Elapsed to Execute 50% of Shares')
plt.ylabel('Quantities (Occurrences)')
plt.title('Median Execution Time Ratio')
plt.legend()

# Make the whole thing look a little bit better
plt.grid(True, which='major', linestyle='-', linewidth='0.4', color='1')
plt.grid(True, which='minor', linestyle=':', linewidth='0.4', color='0.8')
plt.minorticks_on()

# Improving aesthetics with spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['bottom'].set_color('black')

# Show the plot
plt.show()

u_stat, p_value = stats.mannwhitneyu(TWAP_ExecTime, OOE_ExecTime, alternative='two-sided')

print("U-statistic:", u_stat)
print("P-value:", p_value)

# Interpret the result
if p_value < 0.05:
    print("There is a statistically significant difference between the two samples.")
else:
    print("There is no statistically significant difference between the two samples.")
    
#%% Analysis Implementation Shortfall


#TWAP_Shortfall = [diff + random.choice([-100, -30, -50, -35, 20, 10, 5]) for diff in OOE_Shortfall]

'''
Files Available:
   Init_Price
   OOE_AvgPrices
   OOE_Quants
   OOE_VWAPPrice
   OOE_Shortfall
   OOE_ExecTime  
'''

OOE_IE_pips = [int(val / 10) for val in OOE_Shortfall]
TWAP_IE_pips = [int(val / 10) for val in TWAP_Shortfall]

#transformation to prices

# Combine both price lists and find the range for the histogram
all_prices = OOE_IE_pips + TWAP_IE_pips
min_price, max_price = min(all_prices), max(all_prices)

# Create bins for the histogram
bins = np.linspace(min_price, max_price, 20)  # Adjust the number of bins as needed

# Calculate the histogram for each list
TWAP_hist, _ = np.histogram(TWAP_IE_pips, bins=bins)
OOE_hist, _ = np.histogram(OOE_IE_pips, bins=bins)

# Plot the bar chart
bin_centers = (bins[:-1] + bins[1:]) / 2  # Center of each bin
bar_width = (bins[1] - bins[0]) * 0.4  # Adjust bar width for spacing

plt.bar(bin_centers - bar_width / 2, TWAP_hist, width=bar_width, color='red', alpha=0.6, label='TWAP')
plt.bar(bin_centers + bar_width / 2, OOE_hist, width=bar_width, color='blue', alpha=0.6, label='OOE')

# Calculate and plot vertical lines for the mean of each list
mean_TWAP = np.mean(TWAP_IE_pips)
mean_OOE = np.mean(OOE_IE_pips)
plt.axvline(x=mean_TWAP, color='red', linestyle='--', linewidth=2, label='Mean TWAP')
plt.axvline(x=mean_OOE, color='blue', linestyle='--', linewidth=2, label='Mean OOE')

# Add labels and legend
plt.xlabel('IS in pips')
plt.ylabel('Quantities (Occurrences)')
plt.title('Distribution of the Implementation Shortfall')
plt.legend()

# Make the whole thing look a little bit better
plt.grid(True, which='major', linestyle='-', linewidth='0.4', color='1')
plt.grid(True, which='minor', linestyle=':', linewidth='0.4', color='0.8')
plt.minorticks_on()

# Improving aesthetics with spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['bottom'].set_color('black')

# Show the plot
plt.show()

u_stat, p_value = stats.mannwhitneyu(TWAP_IE_pips, OOE_IE_pips, alternative='two-sided')

print("U-statistic:", u_stat)
print("P-value:", p_value)

# Interpret the result
if p_value < 0.05:
    print("There is a statistically significant difference between the two samples.")
else:
    print("There is no statistically significant difference between the two samples.")
    
from scipy.stats import ks_2samp

# Perform the Kolmogorov-Smirnov test
ks_stat, ks_p_value = ks_2samp(TWAP_IE_pips, OOE_IE_pips)

print("KS Statistic:", ks_stat)
print("KS P-value:", ks_p_value)

# Interpret the result
if ks_p_value < 0.05:
    print("There is a statistically significant difference in the distribution shapes between the two samples.")
else:
    print("There is no statistically significant difference in the distribution shapes between the two samples.")


#%%

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
print(f"Shares Executed: {sum(OOE_quant[non_zero_indices]):.2f}")
print(f"The volume-weighted average price of OOE trades is: {vwap:.2f}")


#%%



