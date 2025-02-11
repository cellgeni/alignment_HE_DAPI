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

def GetImages(HE_image_path, DAPI_image_path, level, thumbnail, single_plane):
    try:
        c2r = palom.reader.OmePyramidReader(HE_image_path)
    except:
        c2r = None
    c1r = palom.reader.OmePyramidReader(DAPI_image_path)
    
    if single_plane:
        img1 = c1r.read_level_channels(0, 0)
    else:
        with tifffile.TiffFile(DAPI_image_path) as tif:
            shape = tif.series[0].shape
        if len(shape) > 3:
            raise ValueError("Image has more than 3 dimensions: " + str(shape))
        elif len(shape) == 2:
            raise ValueError("Image has only 2 dimensions: " + str(shape) + '. If it is the case make sure to use flag single_plane = True')    
        elif len(shape) == 3:
            planes_da = [c1r.read_level_channels(0, i) for i in range(shape[0])]
            img1 = da.stack(planes_da, axis=0).max(axis=0)
        else:
             raise ValueError("Image has weird dimensions: " + str(shape))
        
    img1 = img1[::2**level, ::2**level]
    if c2r:
        img2 = c2r.read_level_channels(0, 1) #using G channel for better contrast
    else:
        img2 = get_image_da_from_weird_tiff(HE_image_path, 1)
    
    img2 = img2[::2**level, ::2**level]
    img1_thumbnail = img1[::2**thumbnail, ::2**thumbnail]
    img2_thumbnail = img2[::2**thumbnail, ::2**thumbnail]
    return img1, img2, img1_thumbnail, img2_thumbnail, c1r, c2r

def RegisterOneImage(HE_image_path, DAPI_image_path, out_folder, name, level=0, thumbnail=5, single_plane = True, save_random_crops = False, N_crops = 10):
    print('reading images')
    img1, img2, img1_thumbnail, img2_thumbnail, c1r, c2r = GetImages(HE_image_path, DAPI_image_path, level, thumbnail, single_plane)
    c21l = palom.align.Aligner(ref_img=img1, moving_img=img2, ref_thumbnail=img1_thumbnail, moving_thumbnail=img2_thumbnail,
                               ref_thumbnail_down_factor=2**thumbnail/2**level,
                               moving_thumbnail_down_factor=2**thumbnail/2**level)
    c21l.coarse_register_affine(n_keypoints=4000)
    gc.collect()
    
    c21l.compute_shifts()
    c21l.constrain_shifts()
    if c2r:
        c2m = palom.align.block_affine_transformed_moving_img(
            ref_img=img1,
            # select all the three channels (RGB) in moving image to transform
            moving_img=c2r.pyramid[level],
            mxs=c21l.block_affine_matrices_da)
    else:
        c2m = palom.align.block_affine_transformed_moving_img(
            ref_img=img1,
            # select all the three channels (RGB) in moving image to transform
            moving_img=img2,
            mxs=c21l.block_affine_matrices_da)
    
    print('saving images')
    out_path = os.path.join(out_folder, name + '_reg.ome.tif')
    if single_plane:
        palom.pyramid.write_pyramid(
            mosaics=[
                c1r.pyramid[level],
                c2m
            ],
            output_path=out_path,
            pixel_size=c1r.pixel_size*c1r.level_downsamples[level])
    else:
        palom.pyramid.write_pyramid(
            mosaics=[
                img1,
                c2m
            ],
            output_path=out_path,
            pixel_size=c1r.pixel_size*c1r.level_downsamples[level])
    
    if save_random_crops:
        save_random_N_crops(img1, c2m, out_folder, name, N_crops, crop_size = 2000)
        
def save_random_N_crops(img_da_1ch, img_da_3ch, out_folder, sample_name, N, crop_size, dtype = 'uint8'):
    #images has to be the same size
    for i in range(N):
        print(i)
        shp = img_da_1ch.shape
        x,y = (random.randint(0, shp[0] - 1), random.randint(0, shp[0] - 1))
        x0 = int(x-crop_size/2); x1 = int(x+crop_size/2)
        y0 = int(y-crop_size/2); y1 = int(y+crop_size/2)
        if x0<0: x0=0
        if y0<0: y0=0
        if x1>shp[0]: x1=shp[0]-1
        if y1>shp[1]: y1=shp[1]-1
        if x1>x0 and y1>y0:
            crop_img = np.zeros((3,x1-x0, y1-y0), dtype = dtype)
            #subimg1 = np.clip(img_da_1ch[x0:x1, y0:y1], np.min(img_da_1ch[x0:x1, y0:y1]),np.max(img_da_1ch[x0:x1, y0:y1]))*255
            subimg1 = img_da_1ch[x0:x1, y0:y1]/(np.max(img_da_1ch[x0:x1, y0:y1])*2)*255
            crop_img[0] = subimg1.compute().astype(dtype)
            crop_img[2] = img_da_3ch[1, x0:x1, y0:y1].compute()
            img_path = os.path.join(out_folder, sample_name + '_crop_img_' + str(i) +'.png')
            tifffile.imwrite(img_path, crop_img, photometric = 'rgb')

def main(csv_table_path, out_folder, single_plane = True, level = 0, thumbnail = 5, save_random_crops = True, N_crops = 10):
    #csv file should have next columns: Name, HE_image_path, DAPI_image_path
    # at the moment I consider only one channel image for DAPI! (in future I want to consider N channels)
    table = pd.read_csv(csv_table_path)
    for i in range(table.shape[0]):
        print(table['Name'][i])
        if 'single_plane' in table.columns: single_plane = table['single_plane'][i]
        print(single_plane)
        try:
            RegisterOneImage(table['HE_image_path'][i], table['DAPI_image_path'][i], out_folder, table['Name'][i], level, thumbnail, single_plane, save_random_crops, N_crops)
        except Exception as error:
            print(error)
            print(table['Name'][i] + str(' is failed!'))
            pass

if __name__ == "__main__":
    fire.Fire(main)  




    
if __name__ == "__main__":
    fire.Fire(main)    
    
