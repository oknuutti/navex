# navex

## setup
Setting up environment using conda:
```
git clone https://github.com/oknuutti/navex.git navex
cd navex
git clone https://github.com/oknuutti/r2d2.git r2d2
conda create -n navex python=3.8 pip
conda activate navex
pip install -e ./r2d2
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
