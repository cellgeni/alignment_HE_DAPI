import palom
import gc
import numpy as np
import pandas as pd
import fire
import os



def GetImages(HE_image_path, DAPI_image_path, level, thumbnail):
    c1r = palom.reader.OmePyramidReader(DAPI_image_path)
    c2r = palom.reader.OmePyramidReader(HE_image_path)
    img1 = c1r.read_level_channels(0, 0)
    img1 = img1[::2**level, ::2**level]
    img2 = c2r.read_level_channels(0, 1) #using G channel for better contrast
    img2 = img2[::2**level, ::2**level]
    img1_thumbnail = img1[::2**thumbnail, ::2**thumbnail]
    img2_thumbnail = img2[::2**thumbnail, ::2**thumbnail]
    return img1, img2, img1_thumbnail, img2_thumbnail, c1r, c2r

def RegisterOneImage(HE_image_path, DAPI_image_path, out_folder, name, level=0, thumbnail=5):
    print('reading images')
    img1, img2, img1_thumbnail, img2_thumbnail, c1r, c2r = GetImages(HE_image_path, DAPI_image_path, level, thumbnail)
    c21l = palom.align.Aligner(ref_img=img1, moving_img=img2, ref_thumbnail=img1_thumbnail, moving_thumbnail=img2_thumbnail,
                               ref_thumbnail_down_factor=2**thumbnail/2**level,
                               moving_thumbnail_down_factor=2**thumbnail/2**level)
    c21l.coarse_register_affine(n_keypoints=4000)
    gc.collect()
    
    c21l.compute_shifts()
    c21l.constrain_shifts()
    c2m = palom.align.block_affine_transformed_moving_img(
        ref_img=img1,
        # select all the three channels (RGB) in moving image to transform
        moving_img=c2r.pyramid[level],
        mxs=c21l.block_affine_matrices_da)
    print('saving images')
    out_path = os.path.join(out_folder, name + '_reg.ome.tif')
    palom.pyramid.write_pyramid(
        mosaics=[
            c1r.pyramid[level],
            c2m
        ],
        output_path=out_path,
        pixel_size=c1r.pixel_size*c1r.level_downsamples[level])

def main(csv_table_path, out_folder, level = 0, thumbnail = 5):
    #csv file should have next columns: Name, HE_image_path, DAPI_image_path
    # at the moment I consider only one channel image for DAPI! (in future I want to consider N channels)
    table = pd.read_csv(csv_table_path)
    for i in range(table.shape[0]):
        print(table['Name'][i])
        RegisterOneImage(table['HE_image_path'][i], table['DAPI_image_path'][i], out_folder, table['Name'][i], level, thumbnail)

if __name__ == "__main__":
    fire.Fire(main)  




    
if __name__ == "__main__":
    fire.Fire(main)    
    
