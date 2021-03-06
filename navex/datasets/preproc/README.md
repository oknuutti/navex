
#Usage
We decided to use a separate conda environment for data preprocessing. You can create it like this:
```
conda create -n data_io -c conda-forge "python=>3.8" pip opencv matplotlib gdal geos tqdm scipy pvl quaternion requests bs4
conda activate data_io
pip install pds4-tools
```
