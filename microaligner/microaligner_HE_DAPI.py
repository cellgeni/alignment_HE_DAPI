import numpy as np
import pandas as pd
from tifffile import TiffFile, imwrite, TiffWriter
from microaligner import FeatureRegistrator, transform_img_with_tmat, OptFlowRegistrator, Warper 
import tifffile as tif
import gc
import os
import fire
import sys


def OpenTiff(file_path):
    with TiffFile(file_path) as fh:
        img_data = fh.asarray()
        metadata = fh.ome_metadata
    return img_data, metadata



def prep_images_for_registration(img_1ch, img_3ch, max_int_clip = 50):
    xsize = np.max([img_1ch.shape[1], img_3ch.shape[1]])
    ysize = np.max([img_1ch.shape[0], img_3ch.shape[0]])
    img1 = np.zeros((ysize, xsize), dtype = img_3ch.dtype)
    img1_orig = np.zeros((ysize, xsize), dtype = img_1ch.dtype)
    img2 = np.zeros((ysize, xsize), dtype = img_3ch.dtype)
    img2_3ch = np.zeros((ysize, xsize), dtype = img_1ch.dtype)
    img2_3ch_orig = np.zeros((ysize, xsize,3), dtype = img_3ch.dtype)
    img1[:img_1ch.shape[0], :img_1ch.shape[1]] = np.clip(img_1ch/np.max(img_1ch)*255, 0, max_int_clip)
    img2[:img_3ch.shape[0], :img_3ch.shape[1]] = 255-np.mean(img_3ch, axis = 2)
    img1_orig[:img_1ch.shape[0], :img_1ch.shape[1]] = img_1ch
    img2_3ch_orig[:img_3ch.shape[0], :img_3ch.shape[1], :] = img_3ch
    return img1, img2, img1_orig, img2_3ch_orig


def FeatRegistration(img1, img2, n_pyr = 5, n_iter = 3, tile_size = 5000):
    freg = FeatureRegistrator()
    freg.ref_img = img1
    freg.mov_img = img2
    freg.num_iterations = n_iter
    freg.use_full_res_img = False
    freg.tile_size = tile_size
    #freg._factors = [16,8,4,2] #pyramid levels               
    freg.num_pyr_lvl = n_pyr
    freg.use_dog = True
    transformation_matrix = freg.register()
    #img2_reg = transform_img_with_tmat(img2, img2.shape, transformation_matrix)
    return transformation_matrix

def find_all_positions(text, substring):
    positions = []
    start = 0
    while True:
        start = text.find(substring, start)
        if start == -1:
            break
        positions.append(start)
        start += 1  # Move past the last found position
    return positions

def get_metadata(ome_str):
    #1) dim order
    substr = ome_str[ome_str.index('DimensionOrder')+16:ome_str.index('DimensionOrder')+22]
    DimOrder = substr.split('"')[0]
    #2) pixel size
    substr = ome_str[ome_str.index('PhysicalSizeX')+15:ome_str.index('PhysicalSizeX')+22]
    PixelSize = substr.split('"')[0]
    #3) channel names
    channel_pos_starts = find_all_positions(ome_str, "Channel ID")
    channel_names = []
    for nch in range(len(channel_pos_starts)):
        substr = ome_str[channel_pos_starts[nch]:][ome_str[channel_pos_starts[nch]:].index('Name')+6:ome_str[channel_pos_starts[nch]:].index('Name')+25]
        channel_names.append(substr.split('"')[0])
    return DimOrder, float(PixelSize), channel_names

def warp_HE_image(img, trans_mat):
    img2_reg_3ch = np.zeros_like(img)
    for i in range(img.shape[2]):
        img2_reg_3ch[:,:,i] = transform_img_with_tmat(img[:,:,i], img[:,:,i].shape, trans_mat)
    return img2_reg_3ch
        

def write_pyr_image(volume, path_out, pixelsize, channel_names, dim_order, rgb = False, subresolutions = 5):
    if rgb:
        metadata = {'Channel': {'Name': channel_names}, 'PhysicalSizeX': pixelsize, 'PhysicalSizeXUnit': 'µm', 'PhysicalSizeY': pixelsize, 'PhysicalSizeYUnit': 'µm', 'axes': dim_order, 'photometric': 'rgb'}
    else:
        metadata = {'Channel': {'Name': channel_names}, 'PhysicalSizeX': pixelsize, 'PhysicalSizeXUnit': 'µm', 'PhysicalSizeY': pixelsize, 'PhysicalSizeYUnit': 'µm', 'axes': dim_order}
    with TiffWriter(path_out, bigtiff=True, ome = True) as tif:
        tif.write(
             volume,
             subifds=subresolutions,
             resolution=(1e4 / pixelsize, 1e4 / pixelsize),
             metadata = metadata,
             #description = to_xml(ome_meta).encode(),
        )

        for level in range(subresolutions):
            mag = 2**(level + 1)
            tif.write(
            volume[::mag, ::mag, ...],
            subfiletype=1,
            resolution=(1e4 / mag / pixelsize, 1e4 / mag / pixelsize),
            #**options
            )
    
    
def RegisterOneImage(img_HE_path, img_X_path, rot_clock, rot_anticlock, flip_x, flip_y, n_pyr, n_iter, max_int_clip):
    print('reading and rotating images')
    img_HE, poh = OpenTiff(img_HE_path)
    img_X, ome_meta = OpenTiff(img_X_path)

    try:
        dim_order, pixel_size, channel_names = get_metadata(ome_meta)
    except:
        print('No ome metadata available, using default parameteres!')
        dim_order = 'YX'; pixel_size = 0.2125; channel_names = ['DAPI']
        
    if rot_anticlock:
        img_HE = np.swapaxes(img_HE, 0, 1)
        img_HE = np.flip(img_HE, axis=0)
    if rot_clock:
        img_HE = np.swapaxes(img_HE, 0, 1)
        img_HE = np.flip(img_HE, axis=1)
    if flip_x:
        img_HE = np.flip(img_HE, axis=1)
    if flip_y:
        img_HE = np.flip(img_HE, axis=0)
    
    print('preprocessing of images')
    img1, img2, img1_orig, img2_orig = prep_images_for_registration(img_X, img_HE, max_int_clip)
    gc.collect()
    print('feature registration')
    transformation_matrix = FeatRegistration(img1, img2, n_pyr = n_pyr, n_iter = n_iter)
    
    del img1, img2, img_HE, img_X
    gc.collect()
    
    img2_reg = warp_HE_image(img2_orig, transformation_matrix)
    
    return img1_orig, img2_reg, transformation_matrix, dim_order, pixel_size, channel_names
    


def prepare_paths(out_folder, name):
    path_HE = os.path.join(out_folder,name + '_HE.ome.tif')
    path_DAPI = os.path.join(out_folder,name + '_fluo.ome.tif')
    path_trmat = os.path.join(out_folder,name + '_trans_mat.txt')
    return path_HE, path_DAPI, path_trmat


def main(csv_table_path, out_folder, n_pyr = 5, n_iter = 3, max_int_clip = 50):
    #csv file should have next columns: Name, HE_image_path, DAPI_image_path, rotate_clockwise, rotate_anticlockwise, flip_x, flip_y
    # last 4 columns should be either True or False
    # at the moment I consider only one channel image for DAPI! (in future I want to consider N channels)
    table = pd.read_csv(csv_table_path)
    for i in range(table.shape[0]):
        print(table['Name'][i])
        img1_orig, img2_reg, transformation_matrix, dim_order, pixel_size, channel_names = RegisterOneImage(table['HE_image_path'][i], table['DAPI_image_path'][i], bool(table['rotate_clockwise'][i]), bool(table['rotate_anticlockwise'][i]), bool(table['flip_x'][i]), bool(table['flip_y'][i]), n_pyr, n_iter, max_int_clip)
        print('saving images')
        
        path_HE, path_DAPI, path_trmat =  prepare_paths(out_folder, table['Name'][i])
        np.savetxt(path_trmat, transformation_matrix)
        write_pyr_image(img1_orig, path_DAPI, pixel_size, channel_names, 'YX')
        write_pyr_image(img2_reg, path_HE, pixel_size, ['R', 'G', 'B'], 'YXC', rgb = True)
        del img1_orig, img2_reg
        gc.collect()
        #sys.modules[__name__].__dict__.clear()
        
if __name__ == "__main__":
    fire.Fire(main)    
    
