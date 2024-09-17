# JAX_LOB_Simulation

This project presents an extension to [JAX-LOB](https://arxiv.org/pdf/2308.13289), a framework introduced by Frey et al., and shows how extending it allows a more realistic creation of market dynamics by individually modelling major market participants and their strategies through agent-based models.

The original repository  can be found here: [AlphaTrade](https://github.com/KangOxford/AlphaTrade/tree/jaxV3)

The original setup got extended by:
- The introduction of a heterogeneous set of RL Trading Agents using a hybrid actor-critic model that incorporates a Recurrent Neural Network (RNN) with ProximalPolicy Optimization (PPO).
- The introduction of a relative price process, allowing for a dynamic market response and market impact.
- The introduction of a synthetic data creation algorithm using a poisson process in order to create an alternative to [LOBSTER](https://lobsterdata.com) data.

The model is currently trained using the exec_env script (AlphaTrade-jaxV3/gymnax_exchange/jaxen/exec_env).
- Currently implemented to train one Optimal Order Execution Agent as well as one Trading Agent in around 300-500 parallel environments.
- Heterogeniety in the set of Trading Agents is introduced through a variation in the risk aversion parameter $\gamma$ and the parameter $\omega$ that balances the payoff between Flow and PnL in the reward function (see below):

  $$R = \omega * \alpha_{TA} * PnL + (1 - \omega) * Flow$$

  where:
  $$PnL = PnL_{Inventory} + PnL_{Trading} - \gamma * |PnL_{Inventory}|$$
  $$Flow = (obj_{buy} + obj_{sell}) * P_{mid}(t)$$
  
Testing is done using the exec_env_test_Sigmoid script (AlphaTrade-jaxV3/gymnax_exchange/jaxen/exec_env). 
- Currently implemented using one Optimal Order Execution Agent as well as 30-50 Trading Agents within one environment.
