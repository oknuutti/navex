# navex

## setup
Setting up environment using conda:
```
git clone https://github.com/oknuutti/navex.git navex
cd navex
git clone https://github.com/oknuutti/r2d2.git r2d2
conda create -n navex python=3.8 pip
conda activate navex
conda install -c pytorch pytorch cudatoolkit=11.0
conda install -c conda-forge pytorch-lightning==1.1.7 opencv paramiko quaternion scipy scikit-optimize pandas numba
pip install git+https://github.com/pytorch/vision.git#egg=torchvision
pip install pytorch-lightning-bolts adabelief-pytorch==0.2.0 ray==1.13.0
pip install git+https://github.com/jatentaki/unets
pip install git+https://github.com/jatentaki/torch-localize
pip install git+https://github.com/jatentaki/torch-dimcheck
pip install -e ./r2d2

# make functions node_ip_address_from_perspective and get_node_ip_address always return "127.0.0.1":
vim <env-path>/lib/python3.8/site-packages/ray/_private/services.py
```

<!-- might need to use pip instead of conda for pandas, numba -->
<!-- pip install -U ray ray[tune] -->
<!-- ray install-nightly -->
<!-- pip install ray[tune] -->

or for Windows, [find the correct ray wheel](https://s3-us-west-2.amazonaws.com/ray-wheels/?prefix=latest/) and install like this:

```
pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-1.10.0-cp38-cp38-win_amd64.whl
```
