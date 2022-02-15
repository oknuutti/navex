try:
    from pytorch_lightning import seed_everything

    # random seed used
    RND_SEED = 10
    seed_everything(RND_SEED)
except Exception as e:
    raise Exception('could not set random seed') from e
