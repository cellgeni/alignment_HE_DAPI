import palom
import gc
import numpy as np
import pandas as pd
import fire
import os
import tifffile
import dask.array as da
import random
import os
import traceback




def get_image_da_from_weird_tiff(path, channel = 1):

    with tifffile.TiffFile(path) as tif:
        frames = []
        try:
            for page in tif.pages:
                frames.append(page.asarray())
        except Exception:
            pass

    stack = np.stack(frames)
    
    if len(stack.shape)==4: stack = stack[0]
    if stack.shape[2] == 3:
        img = stack[:,:,channel]
    elif stack.shape[0] == 3:
        img = stack[channel]
    return da.from_array(img)

import tifffile
import numpy as np

def read_ome_tif_as_yxc(path):
    """
    Reads an OME-TIFF file, performs max projection over Z if needed, 
    and outputs the image in (y, x, c) order along with updated metadata.
    
    Handles single-channel images (adds channel dimension if needed).
    
    Parameters
    ----------
    path : str
        Path to the .ome.tif file.
    
    Returns
    -------
    img_yxc : np.ndarray
        Image array with shape (y, x, c). If single-channel, c = 1.
    metadata : dict
        Original metadata dictionary, with Z-plane info removed if projection was applied.
    """
    with tifffile.TiffFile(path) as tif:
        metadata = tif.ome_metadata
        axes = tif.series[0].axes   # e.g., "TCZYX", "ZCYX", "CYX", "YX"
        image_data = tif.series[0].asarray()

    print(f"[INFO] Original axes: {axes}, shape: {image_data.shape}")

    # Collapse time if present (just take the first timepoint)
    if "T" in axes:
        t_index = axes.index("T")
        image_data = np.take(image_data, indices=0, axis=t_index)
        axes = axes.replace("T", "")

    # Handle Z: max projection if Z > 1
    if "Z" in axes:
        z_index = axes.index("Z")
        if image_data.shape[z_index] > 1:
            image_data = np.max(image_data, axis=z_index)
        else:
            image_data = np.squeeze(image_data, axis=z_index)
        axes = axes.replace("Z", "")

    # Handle channel axis
    if "C" in axes:
        c_index = axes.index("C")
        # Remaining axes (Y,X)
        remaining_axes = [a for a in axes if a != "C"]
        # Reorder to (Y, X, C)
        desired_order = [axes.index("Y"), axes.index("X"), c_index]
        img_yxc = np.transpose(image_data, desired_order)
    else:
        # No channel axis: assume single-channel and add dimension
        # Remaining axes must be (Y, X)
        if axes != "YX":
            # Reorder to YX if needed
            order = [axes.index("Y"), axes.index("X")]
            image_data = np.transpose(image_data, order)
        img_yxc = np.expand_dims(image_data, axis=-1)

    # Update metadata: remove Z info if any
    metadata_dict = {"OME": metadata}

    return da.from_array(img_yxc), metadata_dict

def save_palom_transform(c21l, out_folder, name, level, thumbnail):
    os.makedirs(out_folder, exist_ok=True)

    mxs = c21l.block_affine_matrices_da
    if hasattr(mxs, "compute"):
        mxs = mxs.compute()

    out_path = os.path.join(out_folder, f"{name}_palom_transform.npz")

    np.savez_compressed(
        out_path,
        block_affine_matrices=mxs,
        level=level,
        thumbnail=thumbnail,
        scale=2 ** level,
        ref_shape=np.array(c21l.ref_img.shape),
        moving_shape=np.array(c21l.moving_img.shape),
    )

    print(f"Saved PALOM transform → {out_path}")
    return out_path


def GetImages(HE_image_path, DAPI_image_path, level, thumbnail):
    try:
        c2r = palom.reader.OmePyramidReader(HE_image_path)
    except:
        c2r = None
    c1r = palom.reader.OmePyramidReader(DAPI_image_path)
    
    img1, meta = read_ome_tif_as_yxc(DAPI_image_path) #here input image can have any number of planes or channels
    img1 = img1[::2**level, ::2**level]
    if c2r:
        img2 = c2r.read_level_channels(0, 1) #using G channel for better contrast
    else:
        img2 = get_image_da_from_weird_tiff(HE_image_path, 1)
    
    img2 = img2[::2**level, ::2**level]
    img1_thumbnail = img1[::2**thumbnail, ::2**thumbnail]
    img2_thumbnail = img2[::2**thumbnail, ::2**thumbnail]

    #make them all dask arrays
    #img1 = da.from_array(img1);img1_thumbnail = da.from_array(img1_thumbnail);
    #img2 = da.from_array(img2);img2_thumbnail = da.from_array(img2_thumbnail);
    return img1, img2, img1_thumbnail, img2_thumbnail, c1r, c2r, meta


def _build_pyramid_levels(img, downscale_factor=2, max_levels=5):
    """
    Build pyramid levels by iterative downscaling.

    Parameters
    ----------
    img : np.ndarray
        Base image (Y, X, C) or (Y, X).
    downscale_factor : int
        Downscaling factor between pyramid levels.
    max_levels : int, optional
        Maximum number of pyramid levels to generate (including the base image).
        If None, levels are generated until the image is smaller than downscale_factor.

    Returns
    -------
    list of np.ndarray
        Pyramid levels from full-resolution to smallest.
    """
    levels = [img]
    while True:
        # Check if we've reached max levels
        if max_levels is not None and len(levels) >= max_levels:
            break

        # Stop if the smallest dimension is smaller than downscale_factor
        if min(levels[-1].shape[:2]) // downscale_factor < 1:
            break

        # Downscale using simple decimation
        downscaled = levels[-1][::downscale_factor, ::downscale_factor, ...]
        levels.append(downscaled)

    return levels


def save_registered_images(
    ref_img,
    moving_img,
    out_folder,
    sample_name,
    metadata=None,
    use_pyramids=True,
    downscale_factor=2
):
    """
    Save reference and registered moving images separately as OME-TIFFs using tifffile.

    Parameters
    ----------
    ref_img : np.ndarray or dask.array
        Reference image (Y, X, C) or (Y, X).
    moving_img : np.ndarray or dask.array
        Registered moving image (Y, X, C) or (C, Y, X).
    out_folder : str
        Output folder.
    sample_name : str
        Prefix for output file names.
    metadata : dict, optional
        Metadata for OME-TIFF header (applied to reference image).
    use_pyramids : bool
        If True, write multi-resolution pyramids.
    downscale_factor : int
        Downscaling factor between pyramid levels.
    """
    os.makedirs(out_folder, exist_ok=True)

    # --- Convert to NumPy ---
    if isinstance(ref_img, da.Array):
        ref_img = ref_img.compute()
    if isinstance(moving_img, da.Array):
        moving_img = moving_img.compute()

    # --- Ensure (y, x, c) order ---
    if ref_img.ndim == 2:
        ref_img = ref_img[..., np.newaxis]  # single channel
    elif ref_img.ndim == 3 and ref_img.shape[0] <= 4:
        # Channel first -> (y, x, c)
        ref_img = np.transpose(ref_img, (1, 2, 0))

    if moving_img.ndim == 2:
        moving_img = moving_img[..., np.newaxis]
    elif moving_img.ndim == 3 and moving_img.shape[0] <= 4:
        moving_img = np.transpose(moving_img, (1, 2, 0))

    # --- Match shapes ---
    min_y = min(ref_img.shape[0], moving_img.shape[0])
    min_x = min(ref_img.shape[1], moving_img.shape[1])
    ref_img = ref_img[:min_y, :min_x, ...]
    moving_img = moving_img[:min_y, :min_x, ...]

    # Output paths
    ref_out = os.path.join(out_folder, f"{sample_name}_reference.ome.tif")
    moving_out = os.path.join(out_folder, f"{sample_name}_moving.ome.tif")

    # --- Save pyramids or single-level ---
    def _save_tiff(img, path, with_metadata=False):
        if use_pyramids:
            levels = _build_pyramid_levels(img, downscale_factor=downscale_factor)
            # tifffile expects (samples, y, x)
            levels = [np.moveaxis(level, -1, 0) for level in levels]

            with tifffile.TiffWriter(path, bigtiff=True, ome=True) as tif:
                tif.write(
                    data=levels[0],
                    subifds=len(levels) - 1,
                    photometric='minisblack' if levels[0].shape[0] == 1 else 'rgb',
                    description=metadata["OME"] if (with_metadata and metadata) else None
                )
                for sub in levels[1:]:
                    tif.write(
                        data=sub,
                        photometric='minisblack' if sub.shape[0] == 1 else 'rgb'
                    )
        else:
            data = np.moveaxis(img, -1, 0)  # always (samples, y, x)
            tifffile.imwrite(
                path,
                data,
                bigtiff=True,
                ome=True,
                photometric='minisblack' if data.shape[0] == 1 else 'rgb',
                description=metadata["OME"] if (with_metadata and metadata) else None
            )

    print(f"Saving reference image → {ref_out}")
    _save_tiff(ref_img, ref_out, with_metadata=True)

    print(f"Saving moving image → {moving_out}")
    _save_tiff(moving_img, moving_out, with_metadata=False)

    print(f"✅ Saved registered images: {ref_out} and {moving_out}")


def RegisterOneImage(
    HE_image_path,
    DAPI_image_path,
    out_folder,
    name,
    level=0,
    thumbnail=5,
    save_random_crops=False,
    N_crops=10,
    registration_direction="DAPI_ref_HE_moving",
):
    print('reading images')
    name = name.replace("/", "_")
    img1, img2, img1_thumbnail, img2_thumbnail, c1r, c2r, ref_meta = GetImages(HE_image_path, DAPI_image_path, level, thumbnail)
    
    if len(img1.shape)>2:
        img1_p = img1[:,:,0] #take the first channel - assume DAPI
        img1_thumbnail_p = img1_thumbnail[:,:,0]
    else:
        img1_p = img1; img1_thumbnail_p = img1_thumbnail
    '''
    print('image shapes after opening')
    print('img1')
    print(img1_p.shape)
    print(img1_thumbnail_p.dtype)
    print('img2')
    print(img2.shape)
    print(img2.dtype)
    print('img1_thumbnail')
    print(img1_thumbnail_p.shape)
    print(img1_thumbnail_p.dtype)
    print('img2_thumbnail')
    print(img2_thumbnail.shape)
    print(img2_thumbnail.dtype)
    '''
    if registration_direction == "DAPI_ref_HE_moving":
        ref_for_reg = img1_p
        moving_for_reg = img2
        ref_thumb_for_reg = img1_thumbnail_p
        moving_thumb_for_reg = img2_thumbnail

        ref_full_for_save = img1
        moving_full_for_warp = c2r.pyramid[level] if c2r else img2

    elif registration_direction == "HE_ref_DAPI_moving":
        ref_for_reg = img2
        moving_for_reg = img1_p
        ref_thumb_for_reg = img2_thumbnail
        moving_thumb_for_reg = img1_thumbnail_p

        ref_full_for_save = img2
        moving_full_for_warp = img1

    else:
        raise ValueError(
            "registration_direction must be either "
            "'DAPI_ref_HE_moving' or 'HE_ref_DAPI_moving'"
        )

    c21l = palom.align.Aligner(
        ref_img=ref_for_reg,
        moving_img=moving_for_reg,
        ref_thumbnail=ref_thumb_for_reg,
        moving_thumbnail=moving_thumb_for_reg,
        ref_thumbnail_down_factor=2**thumbnail/2**level,
        moving_thumbnail_down_factor=2**thumbnail/2**level,
    )
    c21l.coarse_register_affine(n_keypoints=4000)
    gc.collect()
    print('palom registration')
    c21l.block_size = 4096 ## hardcoded values for faster registration - using large blocks
    c21l.block_step = 2048 ## it should be ok if local warping is not very big
    c21l.compute_shifts()
    c21l.constrain_shifts()
    ## save transformation matrices
    transform_path = save_palom_transform(
        c21l=c21l,
        out_folder=out_folder,
        name=name,
        level=level,
        thumbnail=thumbnail,
    )

    if c2r:
        moving_img = c2r.pyramid[level]
        if not isinstance(moving_img, da.Array):
            moving_full_for_warp = da.from_array(moving_full_for_warp, chunks=(1024, 1024))
        #print("img1:", img1.shape, img1.dtype, isinstance(img1, da.Array))
        #print("moving_img:", moving_img.shape, moving_img.dtype, isinstance(moving_img, da.Array))
        #print("Matrices:", c21l.block_affine_matrices_da.shape)
        c2m = palom.align.block_affine_transformed_moving_img(
            ref_img=ref_for_reg,
            moving_img=moving_full_for_warp,
            mxs=c21l.block_affine_matrices_da,
        )
    else:
        if not isinstance(img2, da.Array):
            img2 = da.from_array(img2, chunks=(1024, 1024))
        c2m = palom.align.block_affine_transformed_moving_img(
            ref_img=img1,
            moving_img=img2,
            mxs=c21l.block_affine_matrices_da
        )
    print('saving images')
    out_path = os.path.join(out_folder, name + '_reg.ome.tif')
    '''
    if single_plane:
        palom.pyramid.write_pyramid(
            mosaics=[
                c1r.pyramid[level],
                c2m
            ],
            output_path=out_path,
            pixel_size=c1r.pixel_size*c1r.level_downsamples[level])
    else:
    
    # Ensure c2m matches img1's shape
    if c2m.ndim == 3 and c2m.shape[0] == img1.shape[-1]:
        # It's already channels last
        pass
    elif c2m.ndim == 3 and c2m.shape[0] <= 4:   # channels first (C, Y, X)
        c2m = np.transpose(c2m, (1, 2, 0))      # -> (Y, X, C)
    elif c2m.ndim == 2:
        # Single channel, just expand dims
        c2m = np.expand_dims(c2m, axis=-1)
    print("img1:", img1.shape)
    print("c2m:", c2m.shape)
    palom.pyramid.write_pyramid(
        mosaics=[
            img1,
            c2m
        ],
        output_path=out_path,
        pixel_size=c1r.pixel_size*c1r.level_downsamples[level])
    '''
    # Save both images separately
    save_registered_images(
        ref_img=ref_full_for_save,
        moving_img=c2m,
        out_folder=out_folder,
        sample_name=name,
        metadata=ref_meta,
        use_pyramids=True,
    )

    
    if save_random_crops:
        save_random_N_crops(img1, c2m, out_folder, name, N_crops, crop_size = 2000)
        
def save_random_N_crops(img_da_1ch, img_da_3ch, out_folder, sample_name, N, crop_size, dtype='uint8'):
    """
    Save N random crops as composite RGB images.
    
    img_da_1ch: reference image (should be single-channel or multi-channel)
    img_da_3ch: moving image (should be multi-channel: C,Y,X)
    """
    for i in range(N):
        shp = img_da_1ch.shape
        x, y = (
            random.randint(0, shp[0] - crop_size),
            random.randint(0, shp[1] - crop_size)
        )
        x0, x1 = x, x + crop_size
        y0, y1 = y, y + crop_size

        # Handle multi-channel reference (pick channel 0)
        crop_ref = img_da_1ch[x0:x1, y0:y1]
        if crop_ref.ndim == 3:
            crop_ref = crop_ref[..., 0]  # pick first channel

        crop_moving = img_da_3ch[:, x0:x1, y0:y1]  # moving image assumed (C, Y, X)

        # Normalize ref and scale
        ref_norm = crop_ref / (np.max(crop_ref) * 2) * 255 if np.max(crop_ref) > 0 else crop_ref

        crop_img = np.zeros((3, crop_size, crop_size), dtype=dtype)
        crop_img[0] = ref_norm.astype(dtype)
        crop_img[2] = crop_moving[1].compute() if hasattr(crop_moving[1], 'compute') else crop_moving[1]

        img_path = os.path.join(out_folder, f"{sample_name}_crop_img_{i}.png")
        tifffile.imwrite(img_path, crop_img, photometric='rgb')

def main(
        csv_table_path,
        out_folder,
        level=0,
        thumbnail=5,
        save_random_crops=True,
        N_crops=10,
        registration_direction="DAPI_ref_HE_moving",
    ):
    #csv file should have next columns: Name, HE_image_path, DAPI_image_path
    # at the moment I consider only one channel image for DAPI! (in future I want to consider N channels)
    table = pd.read_csv(csv_table_path)
    for i in range(table.shape[0]):
        print(table['Name'][i])
        
        try:
            RegisterOneImage(table['HE_image_path'][i], table['DAPI_image_path'][i], out_folder, table['Name'][i], level, thumbnail, save_random_crops, N_crops, registration_direction)
        except Exception as error:
            print(f"❌ Error while processing {table['Name'][i]}:")
            traceback.print_exc()  # <---- FULL traceback
            print(f"{table['Name'][i]} is failed!")
            pass

if __name__ == "__main__":
    fire.Fire(main)  

    
