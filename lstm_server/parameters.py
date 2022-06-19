import numpy as np

initial_seed = 1337
np.random.seed(initial_seed)  # for reproducibility
# parameters
nb_epoch = 100  # number of epoch at training stage. To find a nice epochs in the valid dataset.
batch_size = 64  # batch size
