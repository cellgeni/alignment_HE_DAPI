# Overview
This code is used to align H&E and DAPI images using [palom](https://github.com/labsyspharm/palom)

# Preparing 
Install conda environment using yml file:

`conda env create -f environment.yml`

And activate it:

`conda activate palom`

Prepare csv file with paths for both images and names of the datasets. Columns should have exatcly the same name as in example *test_HE_DAPI_path.csv*

# Running palom image alignment
Run the code with default parameters:

`python palom_HE_DAPI.py test_HE_DAPI_path.csv pat/to/out_dir`

## Additional parameters
There are 2 parameters in palom image alignment that can be changed:
 - **level** (*default = 0*) - pyramid level of images to be aligned. Keep it as 0 if you want to have original spatial resolution. Larger number will results in downscaled images aligned
 - **thumbnail** (*default = 5*) - pyramid level of image that is used for alignment. As I understand this - this is the level at which you can extract meaningful image features for alignment
 - **save_random_crop**  (*default = False*) - whether save random FOVs png crop images from aligned images
 - **N_crops** (*default = 10*) - number of image crops saved per one registered iamge
 - **crop_size** (*default = 2000*) - size in pixels of one crop (it has quadratic shape)
 - **registration_direction** (*default="DAPI_ref_HE_moving"*) - cshise whether you want DAPI or H&E to be reference image. Options are: "DAPI_ref_HE_moving", "HE_ref_DAPI_moving"

# Output
As output program saves merged image with one channel corresponds to DAPI image and other 3 to H&E image (R,G,B). It also saves npz file with transformation matrices and indexes (locations of tiles where those matrices should be used). Later it can be used to warp points


# Aligning Xenium dataset to H&E
If you want to align xenium object to H&E image use *palom_HE_DAPI* code with **registration_direction** = "HE_ref_DAPI_moving", and then output npz file can be used to warp xenium cell centroids and save h5ad anndata file with  

## Command Line Arguments

### Required Arguments

### `--csv`

Path to the input CSV table containing sample metadata.

The CSV should contain at least:

| Column                 | Description                         |
| ---------------------- | ----------------------------------- |
| Sample name column     | Sample identifier                   |
| Xenium bundle column   | Path to Xenium bundle directory     |
| PALOM transform column | Path to PALOM transform `.npz` file |



### `--out_dir`

Output directory where warped AnnData files and optional  plots will be saved.

---

## Input Table Column Names

These parameters allow custom column names in the input CSV.

### `--sample_col`

Column containing sample names.

Default:

```bash
Name
```

---

### `--bundle_col`

Column containing paths to Xenium bundle directories.

Default:

```bash
xenium_bundle
```

---

### `--transform_col`

Column containing paths to PALOM transform `.npz` files.

Default:

```bash
palom_transform_npz
```

---

### `--image_col`

Column containing paths to reference images used for verification plots.

Default:

```bash
image_path
```

---

### `--pixel_size_um_col`

Column containing Xenium pixel size in microns.

If absent or missing for a sample, `--default_pixel_size_um` is used.

Default:

```bash
pixel_size_um
```

---


### `--default_pixel_size_um`

Default Xenium pixel size (microns per pixel) used when the CSV does not provide a value.

Default:

```bash
0.2125
```

---

### `--direction`

Direction of coordinate transformation.

Options:

| Value           | Description                                                         |
| --------------- | ------------------------------------------------------------------- |
| `moving_to_ref` | Warp coordinates from moving image space into reference image space |
| `ref_to_moving` | Warp coordinates from reference image space into moving image space |

Default:

```bash
moving_to_ref
```

---

### `--warp_mode`

Controls which PALOM transform is applied.

Options:

| Value         | Description                                                     |
| ------------- | --------------------------------------------------------------- |
| `local`       | Use PALOM local block-wise affine transformations (recommended) |
| `affine_only` | Use only the global affine transformation for debugging         |

Default:

```bash
local
```

---

### `--post_shift_x`

Additional X-coordinate shift (pixels) applied after warping.

Useful for manual fine adjustment, but dont recommend to use it unless it is necessary

Default:

```bash
0
```

---

### `--post_shift_y`

Additional Y-coordinate shift (pixels) applied after warping.

Useful for manual fine adjustment, but dont recommend to use it unless it is necessary


Default:

```bash
0
```

---

# AnnData Output Parameters

### `--out_pixels_key`

Name of the AnnData `.obsm` entry used to store warped coordinates in pixel units.

Default:

```bash
spatial_palom_pixels
```

---

### `--out_microns_key`

Name of the AnnData `.obsm` entry used to store warped coordinates in micron units.

Default:

```bash
spatial_palom_microns
```

---

# Verification Plot Parameters

### `--skip_plots`

Disable generation of verification overlay plots.

When enabled, only warped AnnData files are produced.

---

### `--plot_downscale`

Requested image downsampling factor used during plotting. Larger values reduce memory usage and plotting time.

Default:

```bash
10
```

---

### `--plot_channel`

Image channel to display in verification plots.

Default:

```bash
0
```

---

### `--point_size`

Marker size used when plotting warped cell coordinates.

Default:

```bash
0.2
```

---

### `--plot_reader`

Image reader used for generating verification plots.

Options:

| Value      | Description                                              |
| ---------- | -------------------------------------------------------- |
| `tifffile` | Uses TIFF pyramids and zarr-backed reading (recommended) |
| `palom`    | Uses PALOM image reader                                  |

Default:

```bash
tifffile
```

---

### `--max_plot_pixels`

Maximum number of image pixels allowed in the plotting image after downsampling.

Used to prevent excessive memory usage.

Default:

```bash
16000000
```

---

### `--max_plot_points`

Maximum number of cells plotted.

If the dataset contains more cells than this value, a reproducible random subset is displayed.

Special value:

```bash
0
```

means plot all cells.

Default:

```bash
250000
```

---

# Outputs

For each sample the script generates:

## Warped AnnData

```text
<sample_name>_palom_warped.h5ad
```

Containing:

* Original Xenium data
* `obsm["spatial_pixels"]`
* Warped coordinates in `obsm[<out_pixels_key>]`
* Warped coordinates in `obsm[<out_microns_key>]`

---

## Verification Plot (optional)

```text
<sample_name>_warped_points_overlay.png
```

Overlay of warped cell centroids on the reference image (usually H&E) used to visually assess registration quality.

---

## Example of usage

```bash
python warp_xenium_with_palom.py \
    --csv samples.csv \
    --out_dir warped_output \
    --direction moving_to_ref \
    --warp_mode local \
    --plot_downscale 20
```

