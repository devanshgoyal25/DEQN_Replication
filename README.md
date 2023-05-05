# DeepEquilibriumNets

We estimate the solutions to the Overlapping Generations Model with Stochastic Production developed by [Kreuger and Kubler (2004)](https://doi.org/10.1016/S0165-1889(03)00111-8)
numerically approximated using a neural network as computed by Azinovic et al. in ['Deep Equilibrium Nets'](https://onlinelibrary.wiley.com/doi/epdf/10.1111/iere.12575). 

The algorithm generates data by choosing a reasonable starting point and predicting the values of different variables of interest for several periods. The neural network then
updates network parameters to minimize the loss function, which can be derived from the equilibrium condition of the economic model. Repeating this process several times yields
strikingly close predictions. 

## Folder Management

## Instructions for Running the Project