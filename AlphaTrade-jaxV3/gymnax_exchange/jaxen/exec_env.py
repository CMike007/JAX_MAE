# from jax import config
# config.update("jax_enable_x64",True)

# ============== testing scripts ===============
import jax
import jax.numpy as jnp
import gymnax
import sys

sys.path.append('/cs/student/projects3/cf/2023/mbirkenz/JAX_MAE/AlphaTrade-jaxV3')

# from gymnax_exchange.jaxen.exec_env import ExecutionEnv
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
import chex
import timeit

import faulthandler
faulthandler.enable()

print("Num Jax Devices:",jax.device_count(),"Device List:",jax.devices())
chex.assert_gpu_available(backend=None)

'''
Change Mike 10.07
Adjustment to change the price process from an absolut replay to an relativ replay
!!! OPEN: Please check if the cnl_msgs within the step_env function should be included for the historical_messages
'''

# #Code snippet to disable all jitting.
from jax import config
#config.update("jax_disable_jit", False)
config.update("jax_disable_jit", True)

import random
# ============== testing scripts ===============



from ast import Dict
from contextlib import nullcontext
from email import message
from random import sample
from re import L
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxen.base_env import BaseLOBEnv
# from gymnax_exchange.test_scripts.comparison import twapV3
import time 

@struct.dataclass
class EnvState:
    ask_raw_orders: chex.Array
    bid_raw_orders: chex.Array
    trades: chex.Array
    best_asks: chex.Array
    best_bids: chex.Array
    init_time: chex.Array
    time: chex.Array

    # OOE Agent variables
    customIDcounter: int
    window_index:int
    init_price:int
    task_to_execute:int
    quant_executed:int
    total_revenue:int
    step_counter: int
    max_steps_in_episode: int

    # new param for RPP
    price_delta_ST: int
    price_delta_LT: int

    # 3 new params for TAgent, till now int and float, will be exchanged with arrays to supply multiple
    task_to_execute_TA: int
    quant_BUY_TA: int # changed
    quant_SELL_TA: int # changed
    inventory_TA: int # changed
    last_mid: int

    # extra params just to get a better understanding of what that RL fucker actually does
    PnL_TA_cummulative: int
    price_hist: int
    buy_hist: int
    sell_hist: int
    OOE_hist: int

    # more params for the reward function
    slippage_rm: int
    price_adv_rm: int
    price_drift_rm: int
    vwap_rm: int

    # 2 Parameters for agent heterogeniety
    gamma_PnL_Function: int         # degree of risk aversion
    omega_Reward_Function: int      # PnL / Flow reward ratio


@struct.dataclass
class EnvParams:
    message_data: chex.Array
    book_data: chex.Array
    stateArray_list: chex.Array
    obs_sell_list: chex.Array
    obs_buy_list: chex.Array
    episode_time: int =  60*30 #60seconds times 30 minutes = 1800seconds
    # max_steps_in_episode: int = 100 # TODO should be a variable, decied by the data_window
    # messages_per_step: int=1 # TODO never used, should be removed?
    time_per_step: int= 0 ##Going forward, assume that 0 implies not to use time step?    

    time_delay_obs_act: chex.Array = struct.field(default_factory=lambda: jnp.array([0, 0]))  # 0ns time delay
    avg_twap_list=jnp.array([312747.47,
                            312674.06,
                            313180.38,
                            312813.25,
                            312763.78,
                            313094.1,
                            313663.97,
                            313376.72,
                            313533.25,
                            313578.9,
                            314559.1,
                            315201.1,
                            315190.2])
    
class ExecutionEnv(BaseLOBEnv):
    def __init__(self, alphatradePath, window_index, OOE_data, TAgent_data = None, RLAgent3 = None):
        super().__init__(alphatradePath)
        #self.n_actions = 2 # [A, MASKED, P, MASKED] Agressive, MidPrice, Passive, Second Passive
        # self.n_actions = 2 # [MASKED, MASKED, P, PP] Agressive, MidPrice, Passive, Second Passive
        # [FT, M, NT, PP] Agressive, MidPrice, Passive, Second Passive
        self.n_actions = 4
        self.n_actions_TA = 3
        self.task = OOE_data['TASKSIDE']
        self.window_index = window_index
        self.action_type = OOE_data['ACTION_TYPE']
        self.rewardLambda = OOE_data['REWARD_LAMBDA']
        self.Gamma = OOE_data['GAMMA']
        self.task_size = OOE_data['TASK_SIZE']
        self.n_fragment_max=2
        self.n_ticks_in_book=2 #TODO: Used to be 20, too large for stocks with dense LOBs

        self.task_size_TA = TAgent_data['TASK_SIZE'] # task size per side (100 --> 100 buy and 100 sell)
        self.trader_unique_id_TA = -500232
        self.episodeTime = OOE_data['Episode_Time']

        self.alpha_TA = TAgent_data['ALPHA'] # not yet sure what alpha_TA really stands for or if its actually neccessary so far

        # we´re still missing all info from the TAgent... (e.g. shares executed, PnL) however
        # I´m not even shure if here´s the right place for it...

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        # return EnvParams(self.messages,self.books)
        return EnvParams(self.messages,self.books,self.stateArray_list,self.obs_sell_list,self.obs_buy_list)
    

    def step_env(
        self, 
        key: chex.PRNGKey,
        state: EnvState,
        action_OOE: dict,
        action_TAgent: dict,
        params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        #Obtain the messages for the step from the message data

        self.gamma = (state.gamma_PnL_Function)
        self.omega = (state.omega_Reward_Function)

        def reshape_action(action : Dict, state: EnvState, params : EnvParams):
            action_space_clipping = lambda action,task_size: jnp.round((action-0.5)*task_size).astype(jnp.int32) if self.action_type=='delta' else jnp.round(action*task_size).astype(jnp.int32).clip(0,task_size)# clippedAction 
            def twapV3(state, env_params):
                # ---------- ifMarketOrder ----------
                remainingTime = env_params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
                marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
                ifMarketOrder = (remainingTime <= marketOrderTime)
                # print(f"{i} remainingTime{remainingTime} marketOrderTime{marketOrderTime}")
                # ---------- ifMarketOrder ----------
                # ---------- quants ----------
                remainedQuant = state.task_to_execute - state.quant_executed
                remainedStep = state.max_steps_in_episode - state.step_counter
                stepQuant = jnp.ceil(remainedQuant/remainedStep).astype(jnp.int32) # for limit orders
                limit_quants = jax.random.permutation(key, jnp.array([stepQuant-stepQuant//2,stepQuant//2]), independent=True)
                market_quants = jnp.array([stepQuant,stepQuant])
                quants = jnp.where(ifMarketOrder,market_quants,limit_quants)
                # ---------- quants ----------
                return jnp.array(quants) 
            get_base_action = lambda state, params:twapV3(state, params)
            def truncate_action(action, remainQuant):
                action = jnp.round(action).astype(jnp.int32).clip(0,self.task_size)
                scaledAction = jnp.where(action.sum() > remainQuant, (action * remainQuant / action.sum()).astype(jnp.int32), action)
                return scaledAction
            
            action_ = get_base_action(state, params) + action_space_clipping(action,state.task_to_execute)  if self.action_type=='delta' else action_space_clipping(action,state.task_to_execute)
            action = truncate_action(action_, state.task_to_execute-state.quant_executed)
            # jax.debug.print("base_ {}, delta_ {}, action_ {}; action {}",base_, delta_,action_,action)
            return action


        action_OOE = reshape_action(action_OOE, state, params)

        #jax.debug.print("Initial signal: {}", action_TAgent)
        #jax.debug.print("Quantity Sold: {}", state.quant_SELL_TA)

        def process_action(input_value):
            scaling_factor = self.task_size_TA * 0.1
            cutoff = 0.6

            def buy_action(val):
                return jnp.array([(val * scaling_factor).astype(jnp.int32), 0, 0])

            def sell_action(val):
                return jnp.array([0, -(val * scaling_factor).astype(jnp.int32), 0])

            def hold_action(val):
                return jnp.array([0, 0, 0])

            input_value_scalar = input_value[-1]  # Convert the array to a scalar

            return lax.cond(input_value_scalar > cutoff,
                            buy_action,
                            lambda val: lax.cond(input_value_scalar < -cutoff*0.5,
                                                 sell_action,
                                                 hold_action,
                                                 val),
                            input_value_scalar)

        def getActionMsgsTA(action: Dict, state: EnvState, params: EnvParams) -> chex.Array:
            self_n_actions_TA = 1
            best_ask, best_bid = state.best_asks[-1,0], state.best_bids[-1,0]
            trader_ids=jnp.ones((self_n_actions_TA,),jnp.int32)*self.trader_unique_id_TA #This agent will always have the same (unique) trader ID (-9001)
            order_ids=jnp.ones((self_n_actions_TA,),jnp.int32)*self.trader_unique_id_TA #Each message has a unique ID
            sides = jax.lax.cond(action[1] != 0,lambda _: -1 * jnp.ones((self_n_actions_TA,), jnp.int32),lambda _: jnp.ones((self_n_actions_TA), jnp.int32),operand=None)
            types=jnp.ones((self_n_actions_TA,),jnp.int32)
            times=jnp.resize(state.time+params.time_delay_obs_act,(self_n_actions_TA,2)) #time from last (data) message of prev. step + some delay
            prices = jax.lax.cond(action[1] != 0,lambda _: (best_ask) * jnp.ones((self_n_actions_TA,), jnp.int32),lambda _: (best_bid) * jnp.ones((self_n_actions_TA), jnp.int32),operand=None)
            quants = jnp.array([jnp.max(action)])
            #quants = jnp.array([jnp.clip(jnp.max(action), a_min=None, a_max=15)]) <- if i want to introduce a certain max trade per episode

            action_msgs=jnp.stack([types,sides,quants,prices,trader_ids,-(trader_ids * sides)],axis=1)
            action_msgs=jnp.concatenate([action_msgs,times],axis=1)
            return action_msgs

        action_TAgent = process_action(action_TAgent)

        #Get the next batch of historical message orders
        data_messages=job.get_data_messages(params.message_data,state.window_index,state.step_counter)
        #jax.debug.print("Data Messages in main Loop: {}", data_messages[:5])


        #Adjust the historical order by the current delta (adjustment to a relative price process)
        data_messages = data_messages.at[:, 3].add(state.price_delta_LT + state.price_delta_ST)

        cnl_msgs=job.getCancelMsgs(state.ask_raw_orders if self.task=='sell' else state.bid_raw_orders,-8999,self.n_actions,-1 if self.task=='sell' else 1)

        #Assumes that all actions are limit orders for the moment - get all 8 fields for each action message
        #action_msgs_OOE = self.getActionMsgs(actions[0], state, params)
        action_msgs_OOE = self.getActionMsgs(action_OOE, state, params)

        def conditional_concatenate(action_TAgent, cnl_msgs, action_msgs_OOE, data_messages, state, params):
            def true_fun(_):
                return jnp.concatenate([cnl_msgs, action_msgs_OOE, data_messages], axis=0)

            def false_fun(_):
                return jnp.concatenate([cnl_msgs, getActionMsgsTA(action_TAgent, state, params), action_msgs_OOE, data_messages[:-1]], axis=0)

            # OLD: 
            # total_messages = jax.lax.cond((jnp.sum(action_TAgent) == 0) | (state.quant_BUY_TA + state.quant_SELL_TA > state.task_to_execute_TA), true_fun, false_fun, operand=None)
            # NEW:
            total_messages = jax.lax.cond(
                (jnp.sum(action_TAgent) == 0) | (
                (action_TAgent[1] != 0) & (state.quant_SELL_TA > state.task_to_execute_TA)) | (
                (action_TAgent[0] != 0) & (state.quant_BUY_TA > state.task_to_execute_TA)), 
                true_fun, false_fun, operand=None
                ) 
            return total_messages
        
        total_messages = conditional_concatenate(action_TAgent, cnl_msgs, action_msgs_OOE, data_messages, state, params)
        #jax.debug.print("Total Messages {}", total_messages)
        
        #Add RPP - historical messages 
        historical_messages = jnp.concatenate([data_messages], axis=0)
        
        #Save time of final message to add to state
        time=total_messages[-1:][0][-2:]
        #To only ever consider the trades from the last step simply replace state.trades with an array of -1s of the same size. 
        trades_reinit=(jnp.ones((self.nTradesLogged,6))*-1).astype(jnp.int32)
        #Process messages of step (action+data) through the orderbook
        asks, bids, trades, bestasks,bestbids = job.scan_through_entire_array_save_bidask(total_messages,(state.ask_raw_orders,state.bid_raw_orders,trades_reinit), self.stepLines) 


        #Add RPP - Computation bestasks and bestbids only using historical data
        _, _, _, bestasks_hist, bestbids_hist = job.scan_through_entire_array_save_bidask(historical_messages,(state.ask_raw_orders,state.bid_raw_orders,trades_reinit), self.stepLines) 
        #Add RPP - Computation 2 * Midprices
        mid_price = ((bestasks[-1, 0] + bestbids[-1, 0]) // 2).astype(jnp.int32)
        mid_price_hist = (bestasks_hist[-1, 0] + bestbids_hist[-1, 0]) // 2
        price_delta = mid_price - mid_price_hist
        # --> we now only need to save the pricedelta and then adjust the new orders in the next step by it
        

        # ========== get reward and revenue ==========
        #Gather the 'trades' that are nonempty, make the rest 0
        executed = jnp.where((trades[:, 0] >= 0)[:, jnp.newaxis], trades, 0)


        # OOE_____________________
        #Mask to keep only the trades where the RL OOE agent is involved, apply mask.
        mask_OOE = ((-9000 < executed[:, 2]) & (executed[:, 2] < 0)) | ((-9000 < executed[:, 3]) & (executed[:, 3] < 0))
        OOETrades = jnp.where(mask_OOE[:, jnp.newaxis], executed, 0)
        new_execution_OOE = OOETrades[:,1].sum()
        revenue = (OOETrades[:,0]//self.tick_size * OOETrades[:,1]).sum()
        agentQuant = OOETrades[:,1].sum()
        vwapFunc = lambda executed: (executed[:,0]//self.tick_size* executed[:,1]).sum()//(executed[:,1]).sum()
        vwap = vwapFunc(executed) # average_price of all the tradings, from the varaible executed
        rollingMeanValueFunc_INT = lambda average_price,new_price:((average_price*state.step_counter+new_price)/(state.step_counter+1)).astype(jnp.int32)
        vwap_rm = rollingMeanValueFunc_INT(state.vwap_rm,vwap) # (state.market_rap*state.step_counter+executedAveragePrice)/(state.step_counter+1)
        # OOE_____________________

        '''
        jax.debug.print('Example of trades array: {}', OOETrades)
        OOETrades look like this:
        [[  1125550       100     -9001     -8999     39600 133490000],
        [  1125550       100     -8999  93418893     39600 681259999]]
        [price, quantity, trader_ids, order_ids, time, time_in_ns]
        '''

        # TAgent__________________
        mask_TAgent = (self.trader_unique_id_TA == executed[:, 2]) | (self.trader_unique_id_TA == executed[:, 3]) | (-self.trader_unique_id_TA == executed[:, 3])
        TAgentTrades = jnp.where(mask_TAgent[:, jnp.newaxis], executed, 0)
        shares_sold = jax.lax.cond(TAgentTrades[0][3] < 0, lambda x: TAgentTrades[:, 1].sum(), lambda x: 0, operand=None)
        shares_bought = jax.lax.cond(TAgentTrades[0][3] > 0, lambda x: TAgentTrades[:, 1].sum(), lambda x: 0, operand=None)
        direction = jax.lax.cond(shares_sold > 0, lambda _: -1, lambda _: 1, operand=None)

        def PnL_TA(gamma, direction):
            delta_PnL_Inventory = (mid_price - state.last_mid) * state.inventory_TA
            delta_PnL = ((TAgentTrades[:, 0] * TAgentTrades[:, 1]).sum() / jnp.maximum(TAgentTrades[:, 1].sum(), 1) - mid_price) * TAgentTrades[:, 1].sum() * direction + delta_PnL_Inventory
            return (delta_PnL + delta_PnL_Inventory - gamma * abs(delta_PnL_Inventory)).astype(jnp.int32)

        buy_obj = jnp.minimum((3*(self.task_size_TA * state.step_counter / state.max_steps_in_episode) - ((abs(state.quant_BUY_TA - (state.step_counter / state.max_steps_in_episode) * self.task_size_TA))**1.3)).astype(jnp.int32), 30)
        sell_obj = jnp.minimum((3*(self.task_size_TA * state.step_counter / state.max_steps_in_episode) - ((abs(state.quant_SELL_TA - (state.step_counter / state.max_steps_in_episode) * self.task_size_TA))**1.3)).astype(jnp.int32), 30)

        Flow_TA = (buy_obj + sell_obj) * 0.00002 * mid_price

        #Just for tracking purposes:
        PnL = PnL_TA(self.gamma, direction)

        #Reward function outlined at page 4 (Towards a fully RL-based Market Simulator, Ardon et al.)
        rewardTAgent = (self.omega * self.alpha_TA * PnL + (1-self.omega) * Flow_TA).astype(jnp.int32)
        # TAgent__________________


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
        RelPriceDelta_ST = ((price_delta * 0.95 + state.price_delta_ST) * 0.5).astype(jnp.int32) # The 0.5 will lead to a exponential decay in the delta over time
        RelPriceDelta_LT = (price_delta * 0.05 + state.price_delta_LT).astype(jnp.int32)
        #jax.debug.print('RelPriceDelta_LT: {}', RelPriceDelta_LT)
        #jax.debug.print('RelPriceDelta_ST: {}', RelPriceDelta_ST)
        price_delta_rm = rollingMeanValueFunc_INT((state.price_delta_LT + state.price_delta_ST), (RelPriceDelta_LT + RelPriceDelta_ST))
       
        
        state = EnvState(asks,bids,trades,bestasks,bestbids,state.init_time,time,state.customIDcounter+self.n_actions,state.window_index,\
            state.init_price,state.task_to_execute,state.quant_executed+new_execution_OOE,state.total_revenue+revenue,state.step_counter+1,\
            state.max_steps_in_episode, RelPriceDelta_ST, RelPriceDelta_LT, self.task_size_TA, 
            state.quant_BUY_TA + shares_bought, state.quant_SELL_TA+shares_sold, state.inventory_TA+shares_bought-shares_sold, mid_price,
            state.PnL_TA_cummulative + PnL, mid_price, shares_bought, shares_sold, new_execution_OOE,
            slippage_rm,price_adv_rm,price_drift_rm,vwap_rm,
            state.gamma_PnL_Function, state.omega_Reward_Function)

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
            "reward_TradingAgent":rewardTAgent,\
            "PnL_TradingAgent": PnL,\
            "Flow_TradingAgent":Flow_TA,\
            "shares_sold": state.quant_SELL_TA,\
            "shares_bought": state.quant_BUY_TA,\
            "Price_history": mid_price,\
            "Buy_history": shares_bought,\
            "Sell_history": shares_sold,\
            "OOE_history": new_execution_OOE,\
            "Buy_OBJ": buy_obj,\
            "Sell_OBJ": sell_obj,\
            "BuyExecuted_Step": state.quant_BUY_TA * (state.max_steps_in_episode / state.step_counter),\
            "SellExecuted_Step": state.quant_SELL_TA * (state.max_steps_in_episode / state.step_counter),\
            "price_delta_rm": (price_delta_rm),\
            }

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position in OB."""
        # all windows can be reached
        
        idx_data_window = jax.random.randint(key, minval=0, maxval=self.n_windows, shape=()) if self.window_index == -1 else jnp.array(self.window_index,dtype=jnp.int32)
        # idx_data_window = jnp.array(self.window_index,dtype=jnp.int32)
        # one window can be reached
        
        # jax.debug.print("window_size {}",self.max_steps_in_episode_arr[0])
        
        # task_size,content_size,array_size = self.task_size,self.max_steps_in_episode_arr[idx_data_window],self.max_steps_in_episode_arr.max().astype(jnp.int32) 
        # task_size,content_size,array_size = self.task_size,self.max_steps_in_episode_arr[idx_data_window],1000
        # base_allocation = task_size // content_size
        # remaining_tasks = task_size % content_size
        # array = jnp.full(array_size, 0, dtype=jnp.int32)
        # array = array.at[:remaining_tasks].set(base_allocation+1)
        # twap_quant_arr = array.at[remaining_tasks:content_size].set(base_allocation)
        
        def stateArray2state(stateArray):
            state0 = stateArray[:,0:6];
            state1 = stateArray[:,6:12];
            state2 = stateArray[:,12:18];
            state3 = stateArray[:,18:20];
            state4 = stateArray[:,20:22];
            state5 = stateArray[0:2,22:23].squeeze(axis=-1);
            state6 = stateArray[2:4,22:23].squeeze(axis=-1);
            state9 = stateArray[4:5,22:23][0].squeeze(axis=-1);
            gamma = jax.random.uniform(key, (1,), minval=0.01, maxval=0.1)[0];
            omega = jax.random.uniform(key, (1,), minval=0.50, maxval=0.80)[0];
            
            return (
                state0,state1,state2,state3,state4,state5,state6,0,idx_data_window,state9,
                self.task_size,0,0,0,self.max_steps_in_episode_arr[idx_data_window],0, 0,
                self.task_size_TA, 0, 0, 0, state9, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
                gamma, omega
            )

        stateArray = params.stateArray_list[idx_data_window]
        state_ = stateArray2state(stateArray)
        # print(self.max_steps_in_episode_arr[idx_data_window])
        # jax.debug.breakpoint()
        obs_sell = params.obs_sell_list[idx_data_window]
        obs_buy = params.obs_buy_list[idx_data_window]
        state = EnvState(*state_)
        obs_OOE = obs_sell if self.task == "sell" else obs_buy
        obs_TAgent = jnp.concatenate([obs_sell[:305],obs_buy[:305]],axis=0)
        
        return [obs_OOE, obs_TAgent], state
        #return obs_OOE, state

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return ((state.time-state.init_time)[0]>self.episodeTime) | (state.task_to_execute-state.quant_executed<=0)


    def getActionMsgs(self, action: Dict, state: EnvState, params: EnvParams):
        '''
        funciton not actually fit to produce orders for the Trading Agent, due to problems like:
        - sides only determined by self.task (does not apply to TAgent)
        - trader_ids(if possible I have a destinctive set of order IDs for each agent (type) - just easier to track)
        '''
        # ============================== Get Action_msgs ==============================
        # --------------- 01 rest info for deciding action_msgs --------------- 

        trader_ids=jnp.ones((self.n_actions,),jnp.int32)*self.trader_unique_id #This agent will always have the same (unique) trader ID (-8999)
        order_ids=jnp.ones((self.n_actions,),jnp.int32)*(self.trader_unique_id+state.customIDcounter)+jnp.arange(0,self.n_actions)
        sides=-1*jnp.ones((self.n_actions,),jnp.int32) if self.task=='sell' else jnp.ones((self.n_actions),jnp.int32) #if self.task=='buy'
        types=jnp.ones((self.n_actions,),jnp.int32)
        times=jnp.resize(state.time+params.time_delay_obs_act,(self.n_actions,2))

        # --------------- 01 rest info for deciding action_msgs ---------------
        
        best_ask, best_bid = state.best_asks[-1,0], state.best_bids[-1,0]
        # --------------- 02 info for deciding prices OOE ---------------
        # Can only use these if statements because self is a static arg.
        # Done: We said we would do ticks, not levels, so really only the best bid/ask is required -- Write a function to only get those rather than sort the whole array (get_L2) 
        FT = best_bid if self.task=='sell' else best_ask # aggressive: far touch
        M = (best_bid + best_ask)//2//self.tick_size*self.tick_size # Mid price
        NT = best_ask if self.task=='sell' else best_bid #Near touch: passive
        PP= best_ask+self.tick_size*self.n_ticks_in_book if self.task=='sell' else best_bid-self.tick_size*self.n_ticks_in_book #Passive, N ticks deep in book
        MKT = 0 if self.task=='sell' else job.MAX_INT
        # --------------- 02 info for deciding prices OOE ---------------

        # --------------- 03 Limit/Market Order (prices/qtys) ---------------
        remainingTime = params.episode_time - jnp.array((state.time-state.init_time)[0], dtype=jnp.int32)
        marketOrderTime = jnp.array(60, dtype=jnp.int32) # in seconds, means the last minute was left for market order
        ifMarketOrder = (remainingTime <= marketOrderTime)

        def normal_order_logic_OOE(state: EnvState, action: jnp.ndarray):
            quants = action.astype(jnp.int32) # from action space
            prices = jnp.asarray((FT,M,NT,PP), jnp.int32)
            return quants, prices
        def market_order_logicOOE(state: EnvState):
            quant = state.task_to_execute - state.quant_executed
            quants = jnp.asarray((quant,0,0,0),jnp.int32) 
            prices = jnp.asarray((MKT, M,M,M),jnp.int32)
            return quants, prices

        market_quants, market_prices = market_order_logicOOE(state)
        normal_quants, normal_prices = normal_order_logic_OOE(state, action)
        quants = jnp.where(ifMarketOrder, market_quants, normal_quants)
        prices = jnp.where(ifMarketOrder, market_prices, normal_prices)

        # --------------- 03 Limit/Market Order (prices/qtys) ---------------

        action_msgs=jnp.stack([types,sides,quants,prices,trader_ids,order_ids],axis=1)
        action_msgs=jnp.concatenate([action_msgs,times],axis=1)
        return action_msgs
        # ============================== Get Action_msgs ==============================

    def get_obs(self, state: EnvState, params:EnvParams) -> chex.Array:
        """Return observation from raw state trafo."""
        # ========= self.get_obs(state,params) =============
        # --------- shared parameter -----------------------
        best_asks=state.best_asks[:,0]
        best_bids =state.best_bids[:,0]
        mid_prices=(best_asks+best_bids)//2//self.tick_size*self.tick_size 
        second_passives = best_asks+self.tick_size*self.n_ticks_in_book if self.task=='sell' else best_bids-self.tick_size*self.n_ticks_in_book
        spreads = best_asks - best_bids
        timeOfDay = state.time
        deltaT = state.time - state.init_time
        shallowImbalance = state.best_asks[:,1]- state.best_bids[:,1]

        # --------- params OOE -----------------------------
        initPrice = state.init_price
        taskSize = state.task_to_execute
        priceDrift = mid_prices[-1] - state.init_price
        executed_quant=state.quant_executed

        # --------- params TAgent --------------------------
        sharesBUY = state.quant_BUY_TA
        sharesSELL = state.quant_SELL_TA
        taskSize_TA = state.task_to_execute_TA
        inventory_TA = state.inventory_TA
        executed_quant_TA = state.quant_BUY_TA + state.quant_SELL_TA
        gamma = state.gamma_PnL_Function
        omega = state.omega_Reward_Function

        # I could also get rid of
        #--> the step step_counter
        #--> the max steps in the episode


        # ========= self.get_obs(state,params) =============
        obsOOE = jnp.concatenate((best_bids,best_asks,mid_prices,second_passives,spreads,timeOfDay,deltaT,jnp.array([initPrice]),jnp.array([priceDrift]),\
            jnp.array([taskSize]),jnp.array([executed_quant]),shallowImbalance,jnp.array([state.step_counter]),jnp.array([state.max_steps_in_episode])))

        ''' OLD VERSION w. Look ahead
        # Look ahead data for Trading Agent
        data_messages_1=job.get_data_messages(params.message_data,state.window_index,state.step_counter)
        data_messages_1 = data_messages_1.at[:, 3].add(state.price_delta)
        data_messages_2=job.get_data_messages(params.message_data,state.window_index,state.step_counter+1)
        data_messages_2 = data_messages_2.at[:, 3].add(state.price_delta)
        #jax.debug.print("Data Messages in get_obs: {}", data_messages[:5, 3])
        obsTA = jnp.concatenate((best_bids,best_asks,mid_prices,second_passives,timeOfDay,deltaT,jnp.array([sharesBUY]),jnp.array([sharesSELL]),\
            jnp.array([inventory_TA]), jnp.array([taskSize_TA]),jnp.array([executed_quant_TA]),data_messages_1[:, 3],data_messages_2[:, 3],jnp.array([state.step_counter])))
        '''

        data_messages_1 = jnp.zeros(100)
        data_messages_2= jnp.zeros(98)

        obsTA = jnp.concatenate((best_bids,best_asks,mid_prices,second_passives,timeOfDay,deltaT,jnp.array([sharesBUY]),jnp.array([sharesSELL]),\
            jnp.array([inventory_TA]), jnp.array([taskSize_TA]),jnp.array([executed_quant_TA]),data_messages_1,data_messages_2,
            jnp.array([gamma]), jnp.array([omega]), jnp.array([state.step_counter])))

        # jax.debug.breakpoint()
        def obsNorm_OOE(obs):
            return jnp.concatenate((
                obs[:400]/(initPrice*2), # best_bids,best_asks,mid_prices,second_passives 
                obs[400:500]/100000, # spreads
                obs[500:501]/100000, # timeOfDay
                obs[501:502]/1000000000, # timeOfDay
                obs[502:503]/params.episode_time, # deltaT
                obs[503:504]/1000000000,# deltaT
                obs[504:505]/(initPrice*2),# initPrice
                obs[505:506]/100000,# priceDrift
                obs[506:507]/taskSize, # taskSize
                obs[507:508]/taskSize, # executed_quant
                obs[508:608]/100, # shallowImbalance 
                obs[608:609]/state.max_steps_in_episode, # step_counter 
                obs[609:610]/state.max_steps_in_episode, # max_steps_in_episode
            ))

        def obsNorm_TAgent(obs):
            return jnp.concatenate((
                obs[:400]/(initPrice*2), # best_bids,best_asks,mid_prices,second_passives
                obs[400:401]/100000, # timeOfDay
                obs[401:402]/1000000000, # timeOfDay
                obs[402:403]/params.episode_time, #10, # deltaT
                jnp.array([0]), #obs[403:404]/1000000000, # deltaT
                obs[404:405]/taskSize_TA, # executed Sell
                obs[405:406]/taskSize_TA, # executed Buy
                obs[406:407]/taskSize_TA, # inventory_TA
                obs[407:408]/(taskSize_TA*2), # taskSize_TA
                obs[408:409]/(taskSize_TA*2), # executed_quant
                obs[409:509]/(initPrice*2), # Empty Frame
                obs[509:607]/(initPrice*2), # Empty Frame
                obs[607:609], # omega and gamma
                obs[609:610]/state.max_steps_in_episode, # step_counter
            ))

        obsNorm_OOE=obsNorm_OOE(obsOOE)
        obsNorm_TAgent=obsNorm_TAgent(obsTA)
        # jax.debug.breakpoint()
        return [obsNorm_OOE, obsNorm_TAgent]


    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(-5,5,(self.n_actions,),dtype=jnp.int32) if self.action_type=='delta' else spaces.Box(0,100,(self.n_actions,),dtype=jnp.int32)
    
    #FIXME: Obsevation space is a single array with hard-coded shape (based on get_obs function): make this better.
    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        space = spaces.Box(-10,10,(610,),dtype=jnp.float32) 
        return space

    #FIXME:Currently this will sample absolute gibberish. Might need to subdivide the 6 (resp 5) 
    #           fields in the bid/ask arrays to return something of value. Not sure if actually needed.   
    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "bids": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nOrdersPerSide),dtype=jnp.int32),
                "asks": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nOrdersPerSide),dtype=jnp.int32),
                "trades": spaces.Box(-1,job.MAXPRICE,shape=(6,self.nTradesLogged),dtype=jnp.int32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "alphatradeExec-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_actions
    
# ============================================================================= #
# ============================================================================= #
# ================================== MAIN ===================================== #
# ============================================================================= #
# ============================================================================= #
