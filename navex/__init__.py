import warnings

# ignore warnings
warnings.filterwarnings("ignore", message="You want to use `wandb` which is not installed yet")
warnings.filterwarnings("ignore", message="You want to use `gym` which is not installed yet")


# random seed used
RND_SEED = 10

try:
    from pytorch_lightning import seed_everything
    seed_everything(RND_SEED)
except Exception as e:
    print('Could not set random seed!')
