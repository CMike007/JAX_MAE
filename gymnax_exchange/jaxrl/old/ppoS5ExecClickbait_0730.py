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

from gymnax_exchange.jaxen.exec_env_relP_update import ExecutionEnv

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

#%%

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
    
# #%% Definition ActorCritic NN

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

# #%% Clickbait Config I

timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
config = {
    "NUM_ENVS": 1,            # reduced from 500
    "TOTAL_TIMESTEPS": 1e2,     # reduced from 1e8
    "NUM_MINIBATCHES": 2,       # (they also tried 4 instead)
    "UPDATE_EPOCHS": 5,         # slightly confusing as it is of course used in the network update, not sure though if we would be forced to use the same for all...
    "NUM_STEPS": 455,
    "ENV_NAME": "alphatradeExec-v0",
    "WINDOW_INDEX": -1,
    "DEBUG": False,     # changed to False
    "wandbOn": False,
    "ATFOLDER": '/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/',
    "RESULTS_FILE":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/results_file_"+f"{timestamp}",
    "CHECKPOINT_DIR":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/checkpoints_"+f"{timestamp}",
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
        "TASK_SIZE":50,
        "LR": 2.5e-5,
        "CLIP_EPS": 0.2,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "REWARD_LAMBDA":1,
        "ENT_COEF": 0.1,    # Entropy Coefficient: ratio of the entropy to deduct from the total loss (_loss_fn)
        "VF_COEF": 0.5,         # Value Loss Coefficient: how much of the value loss should be added to the total loss
        "MAX_GRAD_NORM": 2.0,
        "ANNEAL_LR": True,            
        "OMEGA": 0.1,        # Reward = omega * alpha * PnL(y) - (1 - omega) * Flow(q) <-- used to weight the tradeoff between flow and PnL
        "ALPHA": 1,
        "GAMMA": 0.05
    }
}

# #%% Clickbait Config II

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

# #%% train intitialization
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
    [train_state_OOE, obsv[0], init_OOE_hstate],
    [train_state_TAgent, obsv[1], init_TAgent_hstate]
)

#%%

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

#%%

gamma = 0.99
mid_price = 111100
state_last_mid = 111000
state_inventory_TA = -15

delta_PnL_Inventory = (mid_price - state_last_mid) * state_inventory_TA
delta_PnL =  


def PnL_TA(gamma):
    delta_PnL_Inventory = (mid_price - state_last_mid) * state_inventory_TA
    delta_PnL = ((-15 * 1118875) / -15 - mid_price) * -15 + delta_PnL_Inventory
    return (delta_PnL - gamma * abs(delta_PnL_Inventory)).astype(jnp.int32)


def PnL_TA(gamma):
    delta_PnL_Inventory = (mid_price - state.last_mid) * state.inventory_TA
    delta_PnL = ((OOETrades[:, 0] * OOETrades[:, 1]).sum() / OOETrades[:, 1].sum() - mid_price) * OOETrades[:, 1].sum() + delta_PnL_Inventory
    return (delta_PnL - gamma * abs(delta_PnL_Inventory)).astype(jnp.int32)
          

#%% artificial exec_env functions

executed = jnp.array([[  1118900,       100 ,    -9001 ,    -8999 ,    43200  , 5779999],
 [  1118750  ,      27  ,   -8003, 135316772,     43201,  44370000],
 [        0 ,        0 ,        0  ,       0   ,      0  ,       0],
 [        0  ,       0 ,        0 ,        0  ,       0  ,       0]], dtype=jnp.int32)


mask_TAgent = (-9000 > executed[:, 2]) | (-9000 > executed[:, 3])

TAgentTrades = jnp.where(mask_TAgent[:, jnp.newaxis], executed, 0)

TAgentTrades

shares_sold = jax.lax.cond(newActions[0] != 0, lambda x: TAgentTrades[:, 1].sum(), lambda x: 0, operand=None)
#%%

#def wrappers_step( I )
key = rng_step
state = env_state
actionOOE = OOE_action
actionTA = TAgent_action
params = env_params

#obs, env_state, reward, done, info = self._env.step(key, state.env_state, actionOOE, actionTA, params)

#%%

action_space_clipping = lambda action,task_size: jnp.round((action-0.5)*task_size).astype(jnp.int32) 
input = action_space_clipping(OOE_action[0], 30)
input.astype(jnp.int32)



#INPUT
OOE_action[0]
Out[19]: Array([0.14342394, 0.5068016 , 1.200112  , 0.19170448], dtype=float32)
#OUTPUT
action_space_clipping(OOE_action[0], 30)
Out[20]: Array([-11,   0,  21,  -9], dtype=int32)

#%%

def truncate_action(action, remainQuant, agentType):
    action = jnp.round(action).astype(jnp.int32).clip(0, 500)
    scaledAction = jnp.where(action.sum() > remainQuant, (action * remainQuant / action.sum()).astype(jnp.int32), action)
    return scaledAction


action_OOE = truncate_action(input, 30, 'OOE')
action_OOE
#def Environment_step(

jnp.ones((4,),jnp.int32)

key = key
state = state
actionOOE = actionOOE
actionTA = actionTA
params = params


# def step_env(
action_OOE = actionOOE
action_TAgent = actionTA

action_type = 'delta'
self_task_size = 500
self_task_size_TA = 100
def reshape_action(actions, state, params):
    action_space_clipping = lambda action,task_size: jnp.round((action-0.5)*task_size).astype(jnp.int32) if action_type=='delta' else jnp.round(action*task_size).astype(jnp.int32).clip(0,task_size) # clippedAction 
    def twapV3(state, env_params, agentType):
        # ---------- ifMarketOrder ----------
        remainingTime = 100 #env_params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
        marketOrderTime = 120 #jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
        ifMarketOrder = (remainingTime <= marketOrderTime)
        if agentType == 'OOE':
            remainedQuant = 430
        else:
            remainedQuant = 30
        remainedStep = 560
        stepQuant = jnp.ceil(remainedQuant/remainedStep).astype(jnp.int32) # for limit orders
        limit_quants = jax.random.permutation(key, jnp.array([stepQuant-stepQuant//2,stepQuant//2]), independent=True)
        market_quants = jnp.array([stepQuant,stepQuant])
        quants = jnp.where(ifMarketOrder,market_quants,limit_quants)
        # ---------- quants ----------
        return jnp.array(quants) 
    get_base_action = lambda state, agentType, params:twapV3(state, params, agentType)
    def truncate_action(action, remainQuant, agentType):
        action = jnp.round(action).astype(jnp.int32).clip(0,self_task_size) if agentType == 'OOE' else jnp.round(action).astype(jnp.int32).clip(0,self_task_size_TA)
        scaledAction = jnp.where(action.sum() > remainQuant, (action * remainQuant / action.sum()).astype(jnp.int32), action)
        return scaledAction

    action_OOE_ = get_base_action(state, 'OOE', params)  + action_space_clipping(actions[0],state.task_to_execute) if action_type=='delta' else action_space_clipping(actions[0], state.task_to_execute)
    action_TAgent_ = get_base_action(state, params, 'TA')  + action_space_clipping(actions[1],state.task_to_execute_TA)  if action_type=='delta' else action_space_clipping(actions[1], state.task_to_execute_TA)
    action_OOE = truncate_action(action_OOE_, state.task_to_execute-state.quant_executed, 'OOE')
    action_TAgent = truncate_action(action_TAgent_, state.task_to_execute_TA-state.quant_executed_TA, 'TA')
    return [action_OOE, action_TAgent]

actions = reshape_action([action_OOE, action_TAgent], state, params)

#Get the next batch of historical message orders
data_messages=job.get_data_messages(params.message_data,state.window_index,state.step_counter)

#Adjust the historical order by the current delta (adjustment to a relative price process)
data_messages = data_messages.at[:, 3].add(state.price_delta)
jax.debug.print('price delta: {}', state.price_delta)

#Assumes that all actions are limit orders for the moment - get all 8 fields for each action message
action_msgs_OOE = self.getActionMsgs(actions[0], state, params, 'OOE')
action_msgs_TAgent = self.getActionMsgs(actions[0], state, params, 'TAgent')

cnl_msgs=job.getCancelMsgs(state.ask_raw_orders if self.task=='sell' else state.bid_raw_orders,-8999,self.n_actions,-1 if self.task=='sell' else 1)

#Add to the top of the data messages
total_messages=jnp.concatenate([cnl_msgs,action_msgs_OOE, action_msgs_TAgent, data_messages],axis=0) # TODO DO NOT FORGET TO ENABLE CANCEL MSG

#Add RPP - historical messages 
historical_messages = jnp.concatenate([data_messages], axis=0)

#Save time of final message to add to state
time=total_messages[-1:][0][-2:]
#To only ever consider the trades from the last step simply replace state.trades with an array of -1s of the same size. 
trades_reinit=(jnp.ones((self.nTradesLogged,6))*-1).astype(jnp.int32)
#Process messages of step (action+data) through the orderbook
# jax.debug.breakpoint()
asks, bids, trades, bestasks,bestbids = job.scan_through_entire_array_save_bidask(total_messages,(state.ask_raw_orders,state.bid_raw_orders,trades_reinit), self.stepLines) 

#Add RPP - Computation bestasks and bestbids only using historical data
_, _, _, bestasks_hist, bestbids_hist = job.scan_through_entire_array_save_bidask(historical_messages,(state.ask_raw_orders,state.bid_raw_orders,trades_reinit), self.stepLines) 
#Add RPP - Computation 2 * Midprices
mid_price = (bestasks[-1, 0] + bestbids[-1, 0]) // 2
mid_price_hist = (bestasks_hist[-1, 0] + bestbids_hist[-1, 0]) // 2
price_delta = mid_price - mid_price_hist
# --> we now only need to save the pricedelta and then adjust the new orders in the next step by it

#jax.debug.print("price_delta: {}", price_delta)
#jax.debug.print("state.price_delta: {}", state.price_delta)
# jax.debug.print("bestasks {}", bestbids)
# jax.debug.breakpoint()

# ========== get reward and revenue ==========
def truncate_agent_trades(agentTrades, remainQuant):
    quantities = agentTrades[:, 1]
    cumsum_quantities = jnp.cumsum(quantities)
    cut_idx = jnp.argmax(cumsum_quantities >= remainQuant)
    truncated_agentTrades = jnp.where(jnp.arange(len(quantities))[:, jnp.newaxis] > cut_idx, jnp.zeros_like(agentTrades[0]), agentTrades.at[:, 1].set(jnp.where(jnp.arange(len(quantities)) < cut_idx, quantities, jnp.where(jnp.arange(len(quantities)) == cut_idx, remainQuant - cumsum_quantities[cut_idx - 1], 0))))
    return jnp.where(remainQuant >= jnp.sum(quantities), agentTrades, jnp.where(remainQuant <= quantities[0], jnp.zeros_like(agentTrades).at[0, :].set(agentTrades[0]).at[0, 1].set(remainQuant), truncated_agentTrades))

#Gather the 'trades' that are nonempty, make the rest 0
executed = jnp.where((trades[:, 0] >= 0)[:, jnp.newaxis], trades, 0)

# OOE
#Mask to keep only the trades where the RL OOE agent is involved, apply mask.
mask_OOE = ((-9000 < executed[:, 2]) & (executed[:, 2] < 0)) | ((-9000 < executed[:, 3]) & (executed[:, 3] < 0))
OOETrades = jnp.where(mask_OOE[:, jnp.newaxis], executed, 0)
#TODO: Is this truncation still needed given the action quantities will always < remaining exec quant?
OOETrades = truncate_agent_trades(OOETrades, state.task_to_execute-state.quant_executed)
new_executionOOE = OOETrades[:,1].sum()
revenue = (OOETrades[:,0]//self.tick_size * OOETrades[:,1]).sum()
agentQuant = OOETrades[:,1].sum()
vwapFunc = lambda executed: (executed[:,0]//self.tick_size* executed[:,1]).sum()//(executed[:,1]).sum()
vwap = vwapFunc(executed) # average_price of all the tradings, from the varaible executed
rollingMeanValueFunc_INT = lambda average_price,new_price:((average_price*state.step_counter+new_price)/(state.step_counter+1)).astype(jnp.int32)
vwap_rm = rollingMeanValueFunc_INT(state.vwap_rm,vwap) # (state.market_rap*state.step_counter+executedAveragePrice)/(state.step_counter+1)

'''
# TAgent
maskTAgent = (-9000 > executed[:, 2]) | (-9000 > executed[:, 3])
TAgenttrades = jnp.where(maskTAgent[:, jnp.newaxis], executed, 0)
TAgentTrades = truncate_agent_trades(TAgentTrades, state.task_to_execute_TA-state.quant_executed_TA)
new_executionTAgent = TAgentTrades[:,1].sum()

PnL_TAgent_t = 
revenue = (TAgentTrades[:,0]//self.tick_size * TAgentTrades[:,1]).sum()
agentQuant = TAgentTrades[:,1].sum()


vwapFunc = lambda executed: (executed[:,0]//self.tick_size* executed[:,1]).sum()//(executed[:,1]).sum()
vwap = vwapFunc(executed) # average_price of all the tradings, from the varaible executed
rollingMeanValueFunc_FLOAT = lambda average_price,new_price:(average_price*state.step_counter+new_price)/(state.step_counter+1)
rollingMeanValueFunc_INT = lambda average_price,new_price:((average_price*state.step_counter+new_price)/(state.step_counter+1)).astype(jnp.int32)
vwap_rm = rollingMeanValueFunc_INT(state.vwap_rm,vwap) # (state.market_rap*state.step_counter+executedAveragePrice)/(state.step_counter+1)
'''


#TODO VWAP price (vwap) is only over all trades in between steps. 
advantage = revenue - vwap_rm * agentQuant ### (weightedavgtradeprice-vwap)*agentQuant ### revenue = weightedavgtradeprice*agentQuant
rewardLambda = self.rewardLambda
drift = agentQuant * (vwap_rm - state.init_price//self.tick_size)
# ---------- used for slippage, price_drift, and  RM(rolling mean) ----------
price_adv_rm = rollingMeanValueFunc_INT(state.price_adv_rm,revenue//agentQuant - vwap) # slippage=revenue/agentQuant-vwap, where revenue/agentQuant means agentPrice 
slippage_rm = rollingMeanValueFunc_INT(state.slippage_rm,revenue - state.init_price//self.tick_size*agentQuant)
price_drift_rm = rollingMeanValueFunc_INT(state.price_drift_rm,(vwap - state.init_price//self.tick_size)) #price_drift = (vwap - state.init_price//self.tick_size)
# ---------- compute the final reward ----------
rewardValueOOE = revenue - vwap_rm * agentQuant # advantage_vwap_rm
# rewardValue = vwap_rm
# rewardValue = advantage + rewardLambda * drift
rewardOOE = jnp.sign(agentQuant) * rewardValueOOE # if no value agentTrades then the reward is set to be zero
# ---------- noramlize the reward ----------
rewardOOE /= 10000

# reward /= params.avg_twap_list[state.window_index]
# ========== get reward and revenue END ==========

# just for now:
rewardTAgent = rewardOOE
task_to_execute_TA = self.task_size_TA
quant_executed_TA = 5
PnL_TA = 100
omega_TA = self.omega_TA

#Update state (ask,bid,trades,init_time,current_time,OrderID counter,window index for ep, step counter,init_price,trades to exec, trades executed)
def bestPircesImpute(bestprices,lastBestPrice):
    def replace_values(prev, curr):
        last_non_999999999_values = jnp.where(curr != 999999999, curr, prev) #non_999999999_mask
        replaced_curr = jnp.where(curr == 999999999, last_non_999999999_values, curr)
        return last_non_999999999_values, replaced_curr
    def forward_fill_999999999_int(arr):
        last_non_999999999_values, replaced = jax.lax.scan(replace_values, arr[0], arr[1:])
        return jnp.concatenate([arr[:1], replaced])
    def forward_fill(arr):
        index = jnp.argmax(arr[:, 0] != 999999999)
        return forward_fill_999999999_int(arr.at[0, 0].set(jnp.where(index == 0, arr[0, 0], arr[index][0])))
    back_fill = lambda arr: jnp.flip(forward_fill(jnp.flip(arr, axis=0)), axis=0)
    mean_forward_back_fill = lambda arr: (forward_fill(arr)+back_fill(arr))//2
    return jnp.where((bestprices[:,0] == 999999999).all(),jnp.tile(jnp.array([lastBestPrice, 0]), (bestprices.shape[0],1)),mean_forward_back_fill(bestprices))
bestasks, bestbids = bestPircesImpute(bestasks[-self.stepLines:],state.best_asks[-1,0]),bestPircesImpute(bestbids[-self.stepLines:],state.best_bids[-1,0])
state = EnvState(asks,bids,trades,bestasks,bestbids,state.init_time,time,state.customIDcounter+self.n_actions,state.window_index,\
    state.init_price,state.task_to_execute,state.quant_executed+new_executionOOE,state.total_revenue+revenue,state.step_counter+1,\
    state.max_steps_in_episode,price_delta+state.price_delta,task_to_execute_TA, quant_executed_TA, PnL_TA, omega_TA, slippage_rm,price_adv_rm,price_drift_rm,vwap_rm)
    # state.max_steps_in_episode,state.twap_total_revenue+twapRevenue,state.twap_quant_arr)
# jax.debug.breakpoint()
done = self.is_terminal(state,params)
return self.get_obs(state,params), state, [rewardOOE, rewardTAgent], done,\
    {"window_index":state.window_index,"total_revenue":state.total_revenue,\
    "quant_executed":state.quant_executed,"task_to_execute":state.task_to_execute,\
    "average_price":state.total_revenue/state.quant_executed,\
    "current_step":state.step_counter,\
    'done':done,
    'slippage_rm':state.slippage_rm,"price_adv_rm":state.price_adv_rm,
    "price_drift_rm":state.price_drift_rm,"vwap_rm":state.vwap_rm,\
    "advantage_reward":advantage,\
    }


                
#%% Train_Loop --> _update_step | _env_step

env_state, last_done, rng, OOE_data, TAgent_data = runner_state
train_state_OOE, last_obs_OOE, OOE_hstate = OOE_data[0], OOE_data[1], OOE_data[2]
train_state_TAgent, last_obs_Tagent, TAgent_hstate = TAgent_data[0], TAgent_data[1], TAgent_data[2]


runner_state = (
    env_state,
    jnp.zeros((config["NUM_ENVS"]), dtype=bool),
    _rng,
    [train_state_OOE, obsv[0], init_OOE_hstate],
    [train_state_TAgent, obsv[1], init_TAgent_hstate]
)



# OOE SELECT ACTION
rng, _rng = jax.random.split(rng)
OOE_ac_in = (last_obs_OOE[np.newaxis, :], last_done[np.newaxis, :])
OOE_hstate_, OOE_pi, OOE_value = OOE_network.apply(train_state_OOE.params, OOE_hstate, OOE_ac_in)
# sampled = pi.sample(seed=_rng)
# action = jnp.clip(sampled, -5, 5)
OOE_action = OOE_pi.sample(seed=_rng)
# jax.debug.print("sampled \n {}", sampled)
OOE_log_prob = OOE_pi.log_prob(OOE_action)
# jax.debug.print("log_prob \n {}", log_prob)
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

# --> I´m here and about to find out how the inputs and then outputs really look like...
obsv, env_state, reward, done, info = jax.vmap(
    env.step, in_axes=(0, 0, 0, None)
)(rng_step, env_state, action, env_params)

obsv_OOE, obsv_TAGent = obsv[0], obsv[1]

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
        
      
