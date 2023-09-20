# CNN-based local feature extractors for Solar System small body imagery

This repository contains code for training and evaluating CNN-based local feature extractors targeted for optical 
navigation purposes in the vicinity of Solar System small bodies. In addition, the code base supports
hyperparameter optimization and fetching and preprocessing image data from original sources. Images from several
missions are supported, including Rosetta, OSIRIS-REx, Hayabusa, and NEAR Shoemaker.

The work is described in detail in the following [article](https://dx.doi.org/10.48550/arXiv.2309.xxxxx):
```
@article{
    title={{CNN}-based local features for navigation near an asteroid},
    author={Knuuttila, Olli and Kestil{\"a}, Antti and Kallio, Esa},
    journal={arXiv preprint},
    year={2023}
}
```
<!--  TODO: put correct doi above;  doi={10.48550/arXiv.2309.xxxxx} -->

Related data and fully trained feature extractors can be found in Zenodo: https://dx.doi.org/10.5281/zenodo.8319117.


## Getting started

The system works best on Linux, but can be run on Windows as well, except for hyperparameter optimization. 

The following instructions assume that you are using Linux.

Setup environment using conda:
```
conda create -n navex -c pytorch -c nvidia -c conda-forge --override-channels python==3.10 pip pytorch torchvision \
 pytorch-cuda==11.6 opencv pytorch-lightning==1.1.7 quaternion scipy paramiko scikit-optimize pandas numba
conda activate navex
pip install pytorch-lightning-bolts
pip install git+https://github.com/jatentaki/unets
pip install git+https://github.com/jatentaki/torch-localize
pip install git+https://github.com/jatentaki/torch-dimcheck

# to enable hyperparameter optimization:
pip install ray==2.2.0 ray[default]
# additionally, modify functions node_ip_address_from_perspective and get_node_ip_address to always return "127.0.0.1":
vim <env-path>/lib/python3.10/site-packages/ray/_private/services.py

# to enable testing the original R2D2 network
pip install git+https://github.com/oknuutti/r2d2.git

# to enable relative pose estimation during feature extractor evaluation:
pip install git+https://github.com/oknuutti/featstat.git

# to enable experimental MAC operation counting
pip install thop
```

The rest of the instructions assume that you are in the same folder as this `README.md` file and that there are
additional `data`, `models` and `results` folders. In addition, hyperparameter optimization uses `output` and `cache`
folders. The `data` folder is thought to contain the datasets (`.tar` files) downloaded from Zenodo and the `models`
folder is thought to contain the trained models (`.ckpt` files), also downloaded from Zenodo. Please note that the
file paths in e.g. `rot-cg67p-osinac.tar` and `cg67p-osinac-d.tar` archives are the same, so you need to either rename 
the extracted folder, extract them to different folders, or only extract the archive that you need.


## Feature extraction

E.g., assuming `data/hpatches/image_list_hpatches_sequences.txt` contains 
[a list](https://github.com/mihaidusmanu/d2-net/blob/master/image_list_hpatches_sequences.txt) of
[hpatches](https://github.com/hpatches/hpatches-dataset) images, the following command can be used to extract
features from the images and save them to `.lafe` files:

```
python -m navex.extract --images data/hpatches/image_list_hpatches_sequences.txt \
                        --model models/lafe.ckpt --tag lafe --top-k 2000
```

Alternatively, to extract features of all images in a folder (incl. subfolders), run:
```
python -m navex.extract --images data/cg67p/navcam \
                        --model models/lafe.ckpt --tag lafe --top-k 2000
```

The example above extracts LAFE features of all images belonging to the CG67P Navcam dataset (file
`cg67p-navcam.tar` in Zenodo).

The meaning of the command line arguments can be found by running `python -m navex.extract --help`.


## Evaluation

The following command can be used to extract and evaluate LAFE features on the Eros MSI, Itokawa AMICA,
C-G/67P OSINAC and synthetic datasets. Note that the evaluation code needs the contents of the `*-d.tar` archives 
instead of the `rot-*.tar` archives.

```
python -m navex.evaluate --root=./data --model=./models/extractor-lafe.ckpt \
                         --output=./results/lafe-eval-results.csv \
                         -d=eros -d=ito -d=67p -d=synth \
                         --gpu=1 --min-size=256 --max-size=512 --success-px-limit=5 \
                         --det-lim=0.5 --qlt-lim=0.5 --max-scale=1.0 --est-ori
```

The results are saved to `./results/lafe-eval-results.csv`. If `--est-ori` is given, the
`featstat` package is needed, see section "Getting started" above. The `--model` argument also accepts `akaze`, `orb`,
`sift`, `rsift` (for RootSIFT), and `surf` as values. Also `.onnx` models are accepted. However, in that case, 
`--min-size` and `--max-size` has to be equal as different models for different scales is not supported.

Giving `--help` as an argument prints out a list of possible arguments and their meanings.

Converting a `.ckpt` model into an ONNX model can be done with the following commands:
```
import torch
from navex.dataset.tools import load_model
path = "models/extractor-lafe.ckpt"
model = load_model(path, 'cpu')
torch.onnx.export(model, torch.randn(1, 1, 512, 512), path[:-5] + ".512x512.onnx",
                  input_names=['input'], output_names=['des', 'det', 'qlt'])
```


## Plotting evaluation results

To produce the plots in the article based on the evaluation output, run the following command varying the dataset 
argument to be each of `67p`, `eros`, `ito`, and `synth`:

```
python -m navex.visualizations.evaluation --path=./results/lafe-eval-results.csv --dataset=67p
```


## Training the models

To train a model from scratch, copy a suitable config file from `navex/experiments` to a new file, e.g. `myconfig.yaml`,
and modify the parameters as needed. The config file format is defined in `navex/experiments/definition.yaml`. 
The config file `ast_lafe_2.yaml` is the one used to train the LAFE-ME model in the article. 
The config file `ast_r2d2_v2_best.yaml` corresponds to the HAFE-R2D2 model in the article.

For example, to retrain the LAFE-ME model, you could use the following command:
```
python -m navex.train --config=./navex/experiments/ast_lafe_2.yaml -d=./data --cache=./cache -o=./output \
                      --loss--teacher=./models/extractor-hafe-r2d2.ckpt \
                      --training--batch-size=32 --data--image-size=448 \
                      --id=my-v1 -j=6
                      
```

The meaning of the command line arguments can be found by running `python -m navex.train --help`. Most of the arguments
are the same as in the config file, but can be overridden from the command line. However, due to the experimental nature of the
code, the help text may not be up-to-date and inspection of the source code might be necessary to understand the meaning
of some arguments. The arguments starting with `--search` are only used for hyperparameter optimization, see below.

Training expects the contents of the `rot-*.tar` archives. The contents of the `*-d.tar` archives are only used when
evaluating fully trained feature extractors.


## Running hyperparameter optimization

Hyperparameter optimization is done using [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) customized for the
HPC environments that we used: Triton of Aalto Science-IT project, and Puhti of CSC - IT Center for Science, Finland.

This functionality is very experimental and will likely need a lot of tweaking to get it to work in your environment.

To use the hyperparameter optimization functionality, you need to have access to a cluster with a SLURM job scheduler.
You also need to create a new worker `.sbatch` file with your settings at `navex/ray` and modify the method 
`raytune.ScheduledWorkerNode.schedule_slurm_node` to use your custom `.sbatch` file. The current `worker-csc.sbatch` 
and `worker-triton.sbatch` files are for Puhti and Triton, respectively.

After managing to set up everything, you can run the hyperparameter optimization that produced the HAFE-R2D2 model
with the following command, probably embedded within a custom `.sbatch` file and scheduled on a long-running,
non-GPU node:

```
srun python -m navex.raytune --config=./navex/experiments/ast_r2d2_ashabo_2.yaml \
            --out=./output --cache=./cache --host=`hostname` \
            -b=8 --epochs=24001 --bm=32000 --tf=1500 \
            --data--image-size=224 --lr=1e-3 -j=6 --nodes=6
```

The command line arguments are the same as for `navex.train`. The `--nodes=6` argument specifies that
`navex.raytune` should attempt to schedule work on 6 GPU nodes.

Tensorboard logs can be viewed during and after optimization by running

```
tensorboard --logdir=./output/ast_r2d2_ashabo_2 --port=16006 > tensorboard.log 2>&1 &
```

and then opening `localhost:16006` in a browser.

To plot partial dependency results of a hyperparameter search, you can use the following command:

```
python -m navex.visualizations.raytune --path=./output/ast_r2d2_ashabo_2
```

## Fetching and preprocessing the data from original sources

The data used in the article can be downloaded from [Zenodo](https://dx.doi.org/10.5281/zenodo.8319117). However,
if you wish to fetch the data from the original sources, you can use scripts in the `navex/datasets/preproc` folder.
Each script has a `--help` argument that explains its usage. The `--start` and `--end` arguments can be used to
limit the amount of data to be fetched (the range is from 0.0 to 1.0). The image indices are randomized once 
and persisted so that the same indices are used for consequtive runs, provided that the `.sqlite` 
database is preserved.

However, before you begin, create a separate conda environment for it. You can do it e.g. like this:
```
conda create -n data_io -c conda-forge "python==3.8" pip opencv matplotlib gdal geos \
                                       tqdm scipy pvl quaternion requests bs4 spiceypy
conda activate data_io
pip install pds4-tools
```

To fetch the Bennu TAGCAMS dataset (100% of suitable images):

```
python -m navex.datasets.preproc.bennu_tagcams --dst=./data/bennu/tagcams \
                                               --start=0.0 --end=1.0
```

For C-G/67P NAVCAM (50% of suitable images):

```
python -m navex.datasets.preproc.cg67p_rosetta --dst=./data/cg67p/navcam --instr=navcam \
                                               --start=0.0 --end=0.5
```

For C-G/67P OSINAC (20% of suitable images):

```
python -m navex.datasets.preproc.cg67p_rosetta --dst=./data/cg67p/osinac --instr=osinac \
                                               --min-matches=90000 --min-angle=10 --max-angle=30 \
                                               --start=0.0 --end=0.2
```

For Eros MSI (10% of suitable images):
```
python -m navex.datasets.preproc.eros_msi --dst=./data/eros --pairs-only \
                                          --min-matches=90000 --min-angle=10 --max-angle=30 \
                                          --start=0.0 --end=0.1
```

For Itokawa AMICA (100% of suitable images):
```
python -m navex.datasets.preproc.itokawa_amica --dst=./data/itokawa \
                                               --min-matches=10000 --min-angle=10 --max-angle=30 \
                                               --start=0.0 --end=1.0
```

The datasets obtained using the above commands contain apart from PNG images, image pair pixel correspondences
in the `aflow` folder, detailed metadata `*.lbl`, pixel depths `*.d`, fixed frame 3D coordinates `*.xyz`,
and a shadow mask `*.sdw`. The database `dataset_all.sqlite` contains structured metadata of the images, including poses
and light direction. However, these values were not used for the article as they were not reliable
enough. Instead, all relative poses are calculated based on the `*.xyz` files, derived from the available geometry 
backplanes. Better values should be obtained from e.g. the mission SPICE kernels. Apart from the file names of 
the pixel correspondences, pairing is contained in the `pairs.txt` file.

The C-G/67P NAVCAM and Bennu TAGCAMS datasets do not contain pairing, pixel depths or 3D coordinates,
as geometry backplanes were not available at the time when the data preprocessing related to the article was done.

Feature extractor training uses only the image files and the pixel correspondences at the `aflow` folder. However,
to conserve network capacity, the images of geo-referenced datasets are rotated so that the target body rotation axis
points upwards in each image (resulting in the `rot-*.tar` archives). This can be done for all geo-referenced datasets
used by an experiment by issuing the following command:

```
python -m navex.train --config=./navex/experiments/ast_lafe_2.yaml \
                      -d=./data --preproc-path=./data/rotated
```

The depth information in `.d` files are used during evaluation if the `--est-ori` argument is given. In addition,
evaluation expects non-rotated images as input. Thus, the need for both `rot-*.tar` and `*-d.tar` archives.