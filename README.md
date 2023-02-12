# navex

## setup
Setting up environment using conda:
```
git clone https://github.com/oknuutti/navex.git navex
cd navex
git clone https://github.com/oknuutti/r2d2.git r2d2

conda create -n navex -c pytorch -c nvidia -c conda-forge --override-channels pip pytorch torchvision pytorch-cuda==11.6 \
 opencv pytorch-lightning==1.1.7 opencv paramiko quaternion scipy scikit-optimize pandas numba
conda activate navex
pip install pytorch-lightning-bolts adabelief-pytorch==0.2.0 ray==2.2.0 ray[default]
pip install git+https://github.com/jatentaki/unets
pip install git+https://github.com/jatentaki/torch-localize
pip install git+https://github.com/jatentaki/torch-dimcheck
pip install -e ./r2d2

# make functions node_ip_address_from_perspective and get_node_ip_address always return "127.0.0.1":
vim <env-path>/lib/python3.10/site-packages/ray/_private/services.py
```

<!-- might need to use pip instead of conda for pandas, numba -->
<!-- pip install -U ray ray[tune] -->
<!-- ray install-nightly -->
<!-- pip install ray[tune] -->

or for Windows, [find the correct ray wheel](https://s3-us-west-2.amazonaws.com/ray-wheels/?prefix=latest/) and install like this:

```
pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-1.10.0-cp38-cp38-win_amd64.whl
```
