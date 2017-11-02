import scipy.io
import numpy as np
import h5py
import skimage
from skimage import filters
import matplotlib
import matplotlib.pyplot as plt
import os

path = "/home/loopasam/yeast_colonies/"

filenames = [f for f in os.listdir(path + "for_segmentation/") if f.endswith(".jpg")]

for filename in filenames:

    ## use h5py to import probabilities from Ilastik
    f = h5py.File(path + filename + "_Probabilities.h5", 'r')
    a_group_key = list(f.keys())[0]
    # Get the data
    data = np.array(f[a_group_key])

    h5Pred = np.squeeze(data[:,:,1]) # shows the predicted probabilities for pixels to be colonies

    # Morphological filtering
    Otsu_filt = skimage.filters.threshold_otsu(h5Pred)

    from skimage.segmentation import clear_border
    gray_cleared = clear_border(h5Pred >= Otsu_filt)

    morphFilt = skimage.morphology.closing(skimage.morphology.opening(gray_cleared, skimage.morphology.disk(3)), skimage.morphology.disk(3))

    ## Label colonies, define features for filtering

    ColPredLabeled = skimage.measure.label(morphFilt, background=0)
    ColPredProps = skimage.measure.regionprops(ColPredLabeled)
    areas = np.array([region.area for region in ColPredProps])
    ecc = np.array([region.eccentricity for region in ColPredProps])

    ## Export final selected colonies as individual jpgs

    directory = os.path.dirname(path + "test/unknown/")
    if not os.path.exists(directory):
        os.makedirs(directory)

    k = 0
    for region in ColPredProps:
            if region.area >= 500 and region.area <= areas.mean() + areas.std():
                if region.eccentricity <= 0.5:
                    k += 1
                    minr, minc, maxr, maxc = region.bbox
                    cropped = rgb_image[minr:maxr, minc:maxc, :]
                    skimage.io.imsave(path + "test/unknown/" + filename + "_" + str(k) + ".jpg", cropped)