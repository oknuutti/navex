# navex

## setup
Setting up environment using conda:
```
git clone https://github.com/oknuutti/navex.git navex
cd navex
git clone https://github.com/oknuutti/r2d2.git r2d2
conda create -n navex python=3.8 pip
conda activate navex
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install pytorch-lightning opencv pandas -c conda-forge
pip install -e ./r2d2
pip install -U ray ray[tune] ray[debug]
```

or [find most recent ray wheel](https://s3-us-west-2.amazonaws.com/ray-wheels/?prefix=latest/) and install like this:

```
pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-1.1.0.dev0-cp38-cp38-win_amd64.whl
pip install ray[tune]
```
