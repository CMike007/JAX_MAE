# from jax import config
# config.update("jax_enable_x64",True)

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

from gymnax_exchange.jaxen.exec_env_relP_bugSearch import ExecutionEnv

config.update("jax_disable_jit", False) 
# config.update("jax_disable_jit", True)
config.update("jax_check_tracer_leaks",False) #finds a whole assortment of leaks if true... bizarre.

import datetime
import wandb

from purejaxrl.experimental.s5.s5 import StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO
from purejaxrl.experimental.s5.wrappers import FlattenObservationWrapper, LogWrapper

def save_checkpoint(params, filename):
    with open(filename, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
        print(f"Checkpoint saved to {filename}")


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

def make_train(config):
    
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
    
    # STARTING TRAINING CYCLE
    def train(rng):

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                
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

                return runner_state, [OOE_transition, TAgent_transition]

            initial_hstate_OOE = runner_state[-2][-1]
            initial_hstate_TAgent = runner_state[-1][-1]
            
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
                         
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
            

            # CACLCULATE ADVANTAGE TAgent
            ac_in_TAgent = (last_obs_TAgent[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val_TAgent = TAgent_network.apply(train_state_TAgent.params, init_TAgent_hstate, ac_in_TAgent)
            last_val_TAgent = last_val_TAgent.squeeze(0)
            
            advantages_TAgent, targets_TAgent = _calculate_gae(traj_batch[1], last_val_TAgent, last_done)


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

            # UPDATE NETWORK FUNCTION OOE
            def _update_epoch_TAgent(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = TAgent_network.apply(
                            params, init_hstate, (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["TAgent"]["CLIP_EPS"], config["TAgent"]["CLIP_EPS"])
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
                                1.0 - config["TAgent"]["CLIP_EPS"],
                                1.0 + config["TAgent"]["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["TAgent"]["VF_COEF"] * value_loss
                            - config["TAgent"]["ENT_COEF"] * entropy
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
            
    
            # UPDATE TAgent - NETWORK
            update_state_TAgent = (
                train_state_TAgent,
                initial_hstate_TAgent,
                traj_batch[1],
                advantages_TAgent,
                targets_TAgent,
                rng,
            )
            
            update_state_TAgent, loss_info_TAgent = jax.lax.scan(
                _update_epoch_TAgent, update_state_TAgent, None, config["UPDATE_EPOCHS"]
            )
            train_state_TAgent = update_state_TAgent[0]
            metric_TAgent = (traj_batch[1].info, train_state_TAgent.params)
            rng = update_state_TAgent[-1]
            

            if config.get("DEBUG"):
                
                def callback(metric):
                    
                    info,trainstate_params=metric
                    # info,trainstate,grad_norm=metric
                    
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    
                    revenues = info["total_revenue"][info["returned_episode"]]
                    quant_executed = info["quant_executed"][info["returned_episode"]]
                    average_price = info["average_price"][info["returned_episode"]]
                    
                    slippage_rm = info["slippage_rm"][info["returned_episode"]]
                    price_drift_rm = info["price_drift_rm"][info["returned_episode"]]
                    price_adv_rm = info["price_adv_rm"][info["returned_episode"]]
                    vwap_rm = info["vwap_rm"][info["returned_episode"]]
                    
                    current_step = info["current_step"][info["returned_episode"]]
                    advantage_reward = info["advantage_reward"][info["returned_episode"]]
                    
                    reward_TA = info["reward_TradingAgent"][info["returned_episode"]]
                    PnL_TA = info["PnL_TradingAgent"][info["returned_episode"]]
                    Flow_TA = info["Flow_TradingAgent"][info["returned_episode"]]
    
                    Quant_Buy = info["shares_bought"][info["returned_episode"]]
                    Quant_Sell = info["shares_sold"][info["returned_episode"]]
                    
                    Buy_obj = info["Buy_OBJ"][info["returned_episode"]]
                    Sell_obj = info["Sell_OBJ"][info["returned_episode"]]
                    Ratio_obj = info["Ratio_OBJ"][info["returned_episode"]]

                    for t in range(len(timesteps)):  
                        if ppo_config['wandbOn']:
                            wandb.log(
                                {
                                    "global_step": timesteps[t],
                                    "episodic_return": return_values[t],
                                    "episodic_revenue": revenues[t],
                                    "quant_executed":quant_executed[t],
                                    "average_price":average_price[t],
                                    "slippage_rm":slippage_rm[t],
                                    "price_adv_rm":price_adv_rm[t],
                                    "price_drift_rm":price_drift_rm[t],
                                    "vwap_rm":vwap_rm[t],
                                    "current_step":current_step[t],
                                    "advantage_reward":advantage_reward[t],
                                    "reward_TA":reward_TA[t],
                                    "PnL_TA":PnL_TA[t],
                                    "Flow_TA":Flow_TA[t],
                                    "Quantity_Buy":Quant_Buy[t],
                                    "Quantity_Sold":Quant_Sell[t],
                                    "Buy_Objective":Buy_obj[t],
                                    "Sell_Objective":Sell_obj[t],
                                    "Ratio_Objective":Ratio_obj[t]
                                    # "grad_norm":grad_norm,
                                }
                            ) 
                            print(
                                #f"global step={timesteps[t]:<8} | episodic return={return_values[t]:<15} | episodic revenue={revenues[t]:<15} | average_price={average_price[t]:<20}",\
                                #file=open(config['RESULTS_FILE'],'a')
                            )   
                        else:
                            print(
                                #f"global step={timesteps[t]:<8} | episodic return={return_values[t]:<15} | episodic revenue={revenues[t]:<15} | average_price={average_price[t]:<20}"
                            )
                     
                    '''
                    price_hist = info["Price_history"][0][0]
                    index = int(jnp.where(price_hist == 0)[0][0])
                    print(index)
                    
                    print(price_hist)
                    '''
                            
                    if config["Plot"]:
                        index = info["current_step"][0][0]
                        price_hist = info["Price_history"][0][0][:index]
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

                jax.debug.callback(callback, metric_OOE)
            
            runner_state = (
                env_state, 
                last_done, 
                rng, 
                [train_state_OOE, last_obs_OOE, init_OOE_hstate], 
                [train_state_TAgent, last_obs_TAgent, init_TAgent_hstate]
            ) # adjust to include actual values
            #runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return runner_state, [metric_OOE, metric_TAgent]

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
 
        # CALLING THE TRAIN LOOP
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        
        return {"runner_state": runner_state, "metric": metric}
    
    return train


#%%
    
if __name__ == "__main__":

    timestamp=datetime.datetime.now().strftime("%m-%d_%H-%M")
    os.environ["NCCL_P2P_DISABLE"] = "1"

    ppo_config = {
        "NUM_ENVS": 10,              # reduced from 500
        "TOTAL_TIMESTEPS": 6e6,     # reduced from 1e8 (last run #dandy-leaf-28 was done with 1e6 --> 2h runtime)
        "NUM_MINIBATCHES": 2,       # (they also tried 4 instead)
        "UPDATE_EPOCHS": 5,         # slightly confusing as it is of course used in the network update, not sure though if we would be forced to use the same for all...
        "NUM_STEPS": 200,           # brought down from 455
        "ENV_NAME": "alphatradeExec-v0",
        "WINDOW_INDEX": -1,
        "DEBUG": True,
        "wandbOn": True,
        "ATFOLDER": '/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/',
        "PARAMSFOLDER": '/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/params',
        "RESULTS_FILE":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/results_file_"+f"{timestamp}",
        "CHECKPOINT_DIR":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/checkpoints_"+f"{timestamp}",
        "PICTURE_DIR":"/Users/millionaire/Desktop/UCL/Thesis/AlphaTrade-jaxV3/plot"+f"{timestamp}",
        "Plot": True,
        "OOE": {
            "TASKSIDE":'sell',
            "TASK_SIZE":2500,
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
            "TASK_SIZE":300,
            "LR": 2.5e-5,
            "CLIP_EPS": 0.2,
            "GAE_LAMBDA": 0.95,
            "REWARD_LAMBDA":1,
            "ENT_COEF": 0.1,    # Entropy Coefficient: ratio of the entropy to deduct from the total loss (_loss_fn)
            "VF_COEF": 0.5,         # Value Loss Coefficient: how much of the value loss should be added to the total loss
            "MAX_GRAD_NORM": 2.0,
            "ANNEAL_LR": True,
            "OMEGA": 0.75,        # Reward = omega * alpha * PnL(y) - (1 - omega) * Flow(q) <-- used to weight the tradeoff between flow and PnL
            "ALPHA": 1,
            "GAMMA": 0.1
        }
    }

    if ppo_config['wandbOn']:
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
    # !!! activate the jit again
    train_jit = jax.jit(make_train(ppo_config), device=device)
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

    train_state_OOE = out['runner_state'][3][0] # runner_state.train_state
    train_state_TA = out['runner_state'][4][0]
    params_OOE = train_state_OOE.params
    params_TA = train_state_TA.params

    import datetime;
    if ppo_config['wandbOn']:
        params_file_name = f'Aparam_{wandb.run.name}'
    else:
        params_file_name = f'{timestamp}'
    # Save the params to a file using flax.serialization.to_bytes
    with open(params["PARAMSFOLDER"] + params_file_name + '_OOE', 'wb') as f:
        f.write(flax.serialization.to_bytes(params_OOE))
        print(f"pramas of the Optimal Order Execution Agent saved")
    
    # Save the params to a file using flax.serialization.to_bytes
    with open(params["PARAMSFOLDER"] + params_file_name + '_TA', 'wb') as f:
        f.write(flax.serialization.to_bytes(params_TA))
        print(f"pramas of the Trading Agent saved")

    # Load the params from the file using flax.serialization.from_bytes
    '''
    with open(params_file_name, 'rb') as f:
        restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        print(f"pramas restored")
    '''
        
    # jax.debug.breakpoint()
    # assert jax.tree_util.tree_all(jax.tree_map(lambda x, y: (x == y).all(), params, restored_params))
    # print(">>>")
    # '''

    if ppo_config['wandbOn']:
        run.finish()
