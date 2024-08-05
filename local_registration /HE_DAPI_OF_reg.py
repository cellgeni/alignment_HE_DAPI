import tifffile
import numpy as np
import cv2 as cv
from microaligner import FeatureRegistrator, transform_img_with_tmat, OptFlowRegistrator, Warper 
import fire
import imagecodecs

def OptFlowRegistration(img1, img2):
    
    ofreg = OptFlowRegistrator()
    ofreg.ref_img = img1
    ofreg.mov_img = img2
    ofreg.num_pyr_lvl = 3
    ofreg.num_iterations = 10
    ofreg.tile_size = 20
    ofreg.use_full_res_img = False
    print(ofreg.tile_size)
    ofreg.use_dog = True
    flow_map = ofreg.register()
    return flow_map


def warp_one_ch(flow_map, img):
    warper = Warper()
    warper.flow = flow_map
    warper.image = img
    img_reg = warper.warp()
    return img_reg

def prepare_HE_image(img_HE):
    img_HE_fluo = img_HE[:,:,1]
    img_HE_fluo = 255-img_HE_fluo
    img_HE_fluo_filtered = cv.adaptiveThreshold(img_HE_fluo,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,201,-30)
    return img_HE_fluo_filtered


def main(path_img_HE, path_img_Xenium, output_folder):
    
    print('Openning and preprocessing images')
    with tifffile.TiffFile(path_img_HE) as tif:
        img_HE = tif.asarray()
   
    with tifffile.TiffFile(path_img_Xenium) as tif:
        img_X = tif.asarray()
        
   
    img_HE_grey = prepare_HE_image(img_HE)
    
    print('Registration')
    flow_map = OptFlowRegistration(img_HE_grey, img_X)
    
    print('Warping and saving results')
    img_X_reg = warp_one_ch(flow_map, img_X)
    
    path_flow_map = output_folder + '/' + 'flow_map.npy'
    path_img_reg = output_folder + '/' + 'img_X_reg.tif'
    
    
    np.save(path_flow_map, flow_map)
    tifffile.imwrite(path_img_reg,img_X_reg)
    
if __name__ == "__main__":
    fire.Fire(main)
