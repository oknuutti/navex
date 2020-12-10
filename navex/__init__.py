import random
import numpy as np
import torch

# random seed used
RND_SEED = 10

random.seed(RND_SEED)
np.random.seed(RND_SEED)
torch.manual_seed(RND_SEED)
