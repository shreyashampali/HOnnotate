import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from onlineAug.utilsAug import *
from onlineAug.commonAug import *


def HKPSAug(data,
            crop_height,
            crop_width,

            setOOBPix = True,

            randomScaleMin = 1.2,
            randomScaleMax = 1.6,

            randomRotMin = -90*np.pi/180,
            randomRotMax = 90*np.pi/180,

            randomTransX = 5,
            randomTransY = 5,

            randomHue = 0.1,

            randomSatMin = 0.8,

            randomSatMax = 1.2,

            randomBright = 0.2,

            randomNoise = 5,

            gaussSize = 1,
            gaussStd = 0.75,

            noiseFiltProb = 0.5,

            isConstantPatchSize = False,

            patchSize = [256, 256],

            cropCenter = None,

            augment3D = True,

            camMat = None,

            isTraining = False,

	        isColorAug = True
          ):

    assert isinstance(data, networkData)


    # instantiate Augmentation object
    augTF = augmentationTF(data.image, data.label, data.kps2D, setOOBPix, data.width, data.height,
                           data.kps3D if augment3D else None,
                           camMat=camMat)


    if isTraining:
        # rotation angle
        rotAng = tf.random_uniform((1,), minval=randomRotMin, maxval=randomRotMax)
    else:
        rotAng = tf.random_uniform((1,), minval=0.0, maxval=0.0)


    # center of rotation
    if data.kps2D is not None:
        validMask = tf.equal(data.kps2D[:, 2], 1)
        validKps = tf.boolean_mask(data.kps2D, validMask)
        center = tf.reduce_mean(validKps[:, :2], axis=0)
    else:
        center = tf.stack([tf.cast(data.width, tf.float32), tf.cast(data.height, tf.float32)], axis=0)/2
        center.set_shape([2,])

    # rotate
    augTF.rotate(rotAng, center)


    if isTraining:
        extraScale = tf.random_uniform((1,), minval=0.6, maxval=0.85)
    else:
        extraScale = tf.random_uniform((1,), minval=0.7, maxval=0.7)
    # get the bounding box
    topLeft, bottomRight, center = augTF.get2DBoundingBoxAroundCenter(extraScale, preserveAspRat=True,
                                                                 outWidth=crop_width,
                                                                 outHeight=crop_height,
                                                                 isConstantPatchSize=isConstantPatchSize,
                                                                 patchSize=patchSize, center = cropCenter)
	

    # crop and resize
    augTF.cropAndResize(topLeft, bottomRight, crop_width, crop_height, randomTrans=isTraining,
                                               randomTransX = randomTransX, randomTransY = randomTransY)

    if isTraining:
        # color augmentation
        if isColorAug:
            augTF.hueAug(randomHue)
            augTF.saturationAug(randomSatMin, randomSatMax)
            augTF.brightnessAug(randomBright)



        # noise/filter augmentation
        random_value = tf.random_uniform([])
        isNoise = tf.less_equal(random_value, noiseFiltProb)
        #augTF.img = tf.cond(isNoise, lambda:augTF.gaussFiltImg(gaussSize, gaussStd), lambda:augTF.addImgNoise(randomNoise))


    dataAug = networkData(augTF.img,
                          augTF.seg if data.label is not None else None,
                          augTF.kps2D if augTF.validKps else None,
                          augTF.kps3D if augment3D else data.kps3D,
                          data.imageID, augTF.img.shape[0], augTF.img.shape[1],
                          data.outputType, data.datasetName, topLeft, bottomRight, extraScale)

    return dataAug













