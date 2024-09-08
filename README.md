# JAX_LOB_Simulation

This project is based on the AlphaTrade code that can be found here:
https://github.com/KangOxford/AlphaTrade/tree/jaxV3



This project addresses the challenge of realistically modelling the market microstructure of the foreign exchange (FX) markets, a task with high utility due to its potential for optimising order execution, enhancing high-frequency trading strategies, and improving risk management approaches. It presents an extension to JAX-LOB, a framework introduced by \cite{frey2023jaxlob}, and shows how extending it allows a more realistic creation of market dynamics by individually modelling major market participants and their strategies through agent-based models. The study improves the base model by providing an algorithm to synthetically provide level-10 Limit Order Book (LOB) data, the introduction of a new price formation process, as well as a set of heterogeneous set of Reinforcement Learning (RL) trading agents. The study concludes by demonstrating how a RL agent can be trained to execute larger orders, analyzing the performance of the execution, the impact on the market, as well as the performance and actions of the trading agents involved. The proposed solution provides a gateway and accelerator to study market impact and, more broadly, to offer an optimised and adjustable environment to train Reinforcement Learning agents.
