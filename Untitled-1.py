import numpy as np
from skimage import io
import matplotlib

b = io.imread('blue_channel.tif')
g = io.imread('green_channel.tif')
r = io.imread('red_channel.tif')

Npic = np.shape(b)[0]         #get number of images
sizex = np.shape(b)[1]        #get size in x
sizey = np.shape(b)[2]        #get size in y
Ncol = 3

merged_channels = np.zeros([Npic, sizex, sizey, Ncol])
merged_channels[:,:,:,0] = r
merged_channels[:,:,:,1] = g
merged_channels[:,:,:,2] = b

merged_channels = merged_channels.astype(np.float32)          #Cast Image data type      
merged_channels /= 255                                        #Scale value to float32 range 0-1

merged_channels_hsv = np.zeros([Npic, sizex, sizey, Ncol])

for i in tqdm(range(Npic)):
    merged_channels_hsv[i, :, :, :] = matplotlib.colors.rgb_to_hsv(merged_channels[i, :, :, :])

merged_channels_hsv *= 255                                          
merged_channels_hsv = merged_channels_hsv.astype(np.uint8)           

imsave('merged_channels_hsv.tif', merged_channels_hsv[:,:,:,:])