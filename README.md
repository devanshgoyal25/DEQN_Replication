# DeepEquilibriumNets

We estimate the solutions to the Overlapping Generations Model with Stochastic Production developed by [Kreuger and Kubler (2004)](https://doi.org/10.1016/S0165-1889(03)00111-8)
numerically approximated using a neural network as computed by Azinovic et al. in ['Deep Equilibrium Nets'](https://onlinelibrary.wiley.com/doi/epdf/10.1111/iere.12575). 

The algorithm generates data by choosing a reasonable starting point and predicting the values of different variables of interest for several periods. The neural network then
updates network parameters to minimize the loss function, which can be derived from the equilibrium condition of the economic model. Repeating this process several times yields
strikingly close predictions. 

## Instructions for Running the Project
We specify a majority of the hyperparameters in main.py and we code the neural network, the economic model, and plotting in `train.py`. To train the model, run `main.py`. You can also specify the frequency of plotting and saving the model from command line. 

```
python main.py [--seed] SEED [--plot-interval] PLOT_INTERVAL [--save-interval] SAVE_INTERVAL
```

## Results
We present our results along with a detailed description of the economic model and the neural network in `GangolfGoyalRoss_CMSCH360_FinalProject.pdf`. We provide results for the baseline analytical model as specified by Azinovic et al. and modify it to allow for transition probabilities. 

