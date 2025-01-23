# Overview
This code is used to align H&E and DAPI images using [palom](https://github.com/labsyspharm/palom)

# Preparing 
Install conda environment using yml file:

`conda env create -f environment.yml`

And activate it:

`conda activate palom`

Prepare csv file with paths for both images and names of the datasets. Columns should have exatcly the same name as in example *test_HE_DAPI_path.csv*

# Running 
Run the code with default parameters:

`python palom_HE_DAPI.py test_HE_DAPI_path.csv pat/to/out_dir`

## Additional parameters
There are 2 parameters in palom image alignment that can be changed:
 - *level* - pyramid level of images to be aligned. Keep it as 0 if you want to have original spatial resolution. Larger number will results in downscaled images aligned
 - *thumbnail* - pyramid level of image that is used for alignment. As I understand this - this is the level at which you can extract meaningful image features for alignment

# Output
As output program saves merged image with one channel corresponds to DAPI image and other 3 to H&E image (R,G,B)
