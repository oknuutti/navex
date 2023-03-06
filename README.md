# navex

## setup
Setting up environment using conda:
```
conda create -n navex -c pytorch -c nvidia -c conda-forge --override-channels python==3.10 pip pytorch torchvision \
 pytorch-cuda==11.6 opencv pytorch-lightning==1.1.7 opencv paramiko quaternion scipy scikit-optimize pandas numba
conda activate navex
pip install pytorch-lightning-bolts adabelief-pytorch==0.2.0 ray==2.2.0 ray[default]
pip install git+https://github.com/jatentaki/unets
pip install git+https://github.com/jatentaki/torch-localize
pip install git+https://github.com/jatentaki/torch-dimcheck

# make functions node_ip_address_from_perspective and get_node_ip_address always return "127.0.0.1":
vim <env-path>/lib/python3.10/site-packages/ray/_private/services.py

# to test original r2d2 network
pip install git+https://github.com/oknuutti/r2d2.git

# to enable relative pose estimation during feature extractor evaluation:
pip install git+https://github.com/oknuutti/featstat.git

# to enable experimental MAC operation counting
pip install thop
```

<!-- pip install -U ray ray[tune] -->
<!-- ray install-nightly -->
<!-- pip install ray[tune] -->

or, for Windows, install the latest version or [find the correct ray wheel](https://s3-us-west-2.amazonaws.com/ray-wheels/?prefix=latest/) like this:

```
pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-win_amd64.whl
```
