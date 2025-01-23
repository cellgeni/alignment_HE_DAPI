# Overview
This code is used to align H&E and DAPI images using [palom]([https://github.com/labsyspharm/palom](https://github.com/VasylVaskivskyi/microaligner)). There are two version of the code: one is using only feature registration (global registration using transformation matrix) - *microaligner_HE_DAPI.py*; second is using feature registration + optical flow registration from microaligner (local) - microaligner_HE_DAPI.py

# Preparing 
Install conda environment using yml file:

`conda env create -f environment.yml`

And activate it:

`conda activate microalginer`

Prepare csv file with paths for both images and names of the datasets. Columns should have exatcly the same name as in example *test_HE_DAPI_path.csv*. Please note, that you need to know orientation of each pair of the image. ROtation information (last 4 columns) is corresponds to H&E image and how it has to be rotated and flipped to match DAPI image. Firstly you need to "do" rotation and then flipping.

# Running 
Run the code with default parameters:

`python microaligner_HE_DAPI.py test_HE_DAPI_path.csv pat/to/out_dir`

## Additional parameters
There are next parameters that can be changed:
 - *n_pyr* - number of pyramid levels used for feature registration
 - *n_iter* - number of iterations used for each pyramid level in feature registration part
 - *max_int_clip* - maximum intensity clipped in DAPI image, after it been reformatted for uint8 type. It is used for preprocessing images that are used for alignment (but not for final warping). Can be [0-255]
*microaligner_HE_DAPI.py* has 2 additional parameters:
 - *n_pyr_of* - number of pyramid levels used for optical flow registration
 - *n_iter_of* - number of iterations used for each pyramid level in optical flow registration

# Output
As output program saves:
 - Registered H&E pyramidized ome.tif image
 - Registered DAPI pyramidized ome.tif image
 - txt file with transformation matrix
 - (in case of optical flow code) txt file with optical flow vector field
