import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions

# plt.ion()

def setBlackPixels(img, val):
    tf.assert_type(img, tf.uint8)
    tf.assert_type(val, tf.uint8)

    # mean pixel value
    # meanPixVal = tf.reduce_mean(tf.reshape(tf.to_float(img), [-1, 3]), axis=0)
    # meanPixVal = tf.cast(meanPixVal, tf.uint8)
    # meanPixVal = tf.zeros_like(img) + tf.cast(meanPixVal, tf.uint8)  # same size as img

    mask = tf.equal(tf.reduce_sum(tf.to_float(img), axis=2), 0.)
    mask = tf.stack([mask, mask, mask], axis=2)
    mask = tf.cast(mask, tf.uint8)
    img = mask * val + (1 - mask) * img

    return img

class augmentationTF():
    def __init__(self, img, seg=None, kps2D=None, setOOBPix=True, imgW=None, imgH=None, kps3D=None, camMat=None):
        self.validKps = True
        self.validSeg = True

        if kps2D is not None:
            assert len(kps2D.shape) == 2, 'keypoints should be rank 2'
        else:
            self.validKps = False
        if seg is not None:
            pass
            # assert len(seg.shape) == 2, 'seg should be single channel'
        else:
            self.validSeg = False
        assert len(img.shape) == 3, 'img should have 3 channels'

        self.kpsValidCol = False
        if self.validKps:
            if kps2D.shape[-1] == 3:
                self.kpsValidCol = True

        # create copies
        self.img = tf.identity(img)
        if self.validSeg:
            self.seg = tf.identity(seg)
        if self.validKps:
            self.kps2D = tf.identity(kps2D)

        self.meanPixVal = tf.constant([0, 0, 0], dtype=tf.uint8)
        self.setOOBPix = setOOBPix
        self.imgW = imgW
        self.imgH = imgH
        self.kps3D = kps3D
        self.camMat = camMat



    def get2DBoundingBoxFromKps(self, extraScale, preserveAspRat = True, outWidth=None, outHeight=None,
                                isConstantPatchSize = False, patchSize = [256, 256]):
        '''
        Given keypoints in image, gets the bounding box
        :param kps2D: This is Nx3 tensor. Last column is valid col
        :return: topleft and bottom right corners of BB
        '''

        if not self.validKps:
            raise Exception('Keypoints not avilable for augmentation')

        if self.kpsValidCol:
            validMask = tf.equal(self.kps2D[:,2], 1)
            validKps = tf.boolean_mask(self.kps2D, validMask)
        else:
            validKps = self.kps2D

        topLeft = tf.reduce_min(validKps[:,:2], axis=0)
        bottomRight = tf.reduce_max(validKps[:,:2], axis=0)

        assert topLeft.shape == (2,)
        assert bottomRight.shape == (2,)

        cropCenter = (topLeft + bottomRight) / 2.0


        if isConstantPatchSize:
            cropWidth = tf.cast(patchSize[1], tf.float32)
            cropHeight = tf.cast(patchSize[0], tf.float32)
        else:
            cropWidth = bottomRight[0] - topLeft[0]
            cropHeight = bottomRight[1] - topLeft[1]


        # change window size if need to preserve aspect ratio
        if preserveAspRat:
            assert outWidth is not None
            assert outHeight is not None
            def changeHeight():
                scaleH = tf.cast(outHeight, tf.float32)*cropWidth/(tf.cast(outWidth, tf.float32)*cropHeight)
                newCropHeight = scaleH*cropHeight
                return cropWidth, newCropHeight
            def changeWidth():
                scaleW = (tf.cast(outWidth, tf.float32)*cropHeight)/(tf.cast(outHeight, tf.float32)*cropWidth)
                newCropWidth = scaleW*cropWidth
                return newCropWidth, cropHeight

            newCropWidth, newCropHeight = tf.cond(tf.greater((tf.to_float(cropHeight)/tf.to_float(cropWidth)),
                                                             (tf.to_float(outHeight)/tf.to_float(outWidth)))
                                                  , lambda:changeWidth(), lambda:changeHeight())
        else:
            newCropWidth = cropWidth
            newCropHeight = cropHeight

        topLeft = cropCenter - tf.stack([newCropWidth/2., newCropHeight/2.])*extraScale
        bottomRight = cropCenter + tf.stack([newCropWidth / 2., newCropHeight / 2.])*extraScale

        # convert all to int
        topLeft = tf.to_int32(tf.round(topLeft))
        bottomRight = tf.to_int32(tf.round(bottomRight))
        center = tf.to_int32(tf.round(cropCenter))

        return topLeft, bottomRight, center


    def get2DBoundingBoxAroundCenter(self, extraScale, preserveAspRat = True, outWidth=None, outHeight=None,
                                isConstantPatchSize = False, patchSize = [256, 256], center = None):
        '''
        Given keypoints in image, gets the bounding box
        :param kps2D: This is Nx3 tensor. Last column is valid col
        :return: topleft and bottom right corners of BB
        '''

        if center is None:
            cropCenter = tf.stack([tf.to_float(self.imgW), tf.to_float(self.imgH)], axis=0)/2.0
            cropCenter.set_shape([2,])
        else:
            cropCenter = tf.convert_to_tensor(center)

        cropWinSize = tf.minimum(tf.to_float(self.imgW), tf.to_float(self.imgH))
        cropHeight = extraScale[0] * cropWinSize
        cropWidth = extraScale[0] * cropWinSize



        if isConstantPatchSize:
            cropWidth = tf.cast(patchSize[0], tf.float32)
            cropHeight = tf.cast(patchSize[1], tf.float32)

        # if change window size if need to preserve aspect ratio
        if preserveAspRat:
            assert outWidth is not None
            assert outHeight is not None
            def changeHeight():
                scaleH = tf.cast(outHeight, tf.float32)*cropWidth/(tf.cast(outWidth, tf.float32)*cropHeight)
                newCropHeight = scaleH*cropHeight
                return cropWidth, newCropHeight
            def changeWidth():
                scaleW = (tf.cast(outWidth, tf.float32)*cropHeight)/(tf.cast(outHeight, tf.float32)*cropWidth)
                newCropWidth = scaleW*cropWidth
                return newCropWidth, cropHeight

            newCropWidth, newCropHeight = tf.cond(tf.greater((tf.to_float(cropHeight)/tf.to_float(cropWidth)),
                                                             (tf.to_float(outHeight)/tf.to_float(outWidth)))
                                                  , lambda:changeWidth(), lambda:changeHeight())
        else:
            newCropWidth = cropWidth
            newCropHeight = cropHeight

        # newCropWidth = tf.Print(newCropWidth, [newCropWidth, newCropHeight])
        topLeft = cropCenter - tf.stack([newCropWidth/2., newCropHeight/2.])
        bottomRight = cropCenter + tf.stack([newCropWidth / 2., newCropHeight / 2.])

        # convert all to int
        topLeft = tf.to_int32(tf.round(topLeft))
        bottomRight = tf.to_int32(tf.round(bottomRight))
        center = tf.to_int32(tf.round(cropCenter))

        return topLeft, bottomRight, center




    def rotate(self, rotAng, center):
        '''
        Rotate image and KPs about center by rotAng radians
        :param img:
        :param kps:
        :param rotAng: rotation angle
        :param center: center of rotation
        :return:
        '''
        # rotAng = tf.constant([0.0])
        if center.shape == (2,):
            center = tf.expand_dims(center, 1)

        assert center.shape == (2,1)

        # mean pix value (to be used later)
        meanPixVal = tf.reduce_mean(tf.reshape(tf.to_float(self.img), [-1, 3]), axis=0)
        self.meanPixVal = tf.cast(meanPixVal, tf.uint8)

        # get transorm matrix from output to input (as required by tf.contrib.image.transform)
        rotMat = tf.stack([(tf.cos(rotAng), -tf.sin(rotAng)), (tf.sin(rotAng), tf.cos(rotAng))], axis=0)[:,:,0]
        transVec = center - tf.matmul(tf.transpose(rotMat), center)
        transormMat = tf.concat([tf.transpose(rotMat), transVec], axis=1)
        transormMat = tf.concat([transormMat, np.array([[0., 0., 1.]])], axis=0)

        # transform from input to output
        transormMatInv = tf.linalg.inv(transormMat)

        if self.validKps:
            # get homogeneous kps
            kpsHomo = tf.concat([self.kps2D[:,:2], tf.ones((self.kps2D.shape[0], 1), dtype=tf.float32)], axis=1)

            # transorm the keypoints
            kpsTrans = tf.matmul(kpsHomo, tf.transpose(transormMatInv))

            # transform the 3d keypoints
            kps3DOld = tf.identity(self.kps3D)
            self.kps3D = tf.matmul(kpsTrans * self.kps3D[:, 2:3], tf.transpose(tf.linalg.inv(self.camMat)))

            # debug
            # self.kps3D = tf.Print(self.kps3D, [self.kps3D, kps3DOld])
            # reproj = tf.matmul(self.kps3D, tf.transpose(self.camMat))
            # reproj = reproj[:,:2]/reproj[:,2:]
            # self.kps3D = tf.Print(self.kps3D, [kpsTrans[0], reproj[0]])

            # add the valid col back
            if self.kpsValidCol:
                self.kps2D = tf.concat([kpsTrans[:,:2], self.kps2D[:,2:]], axis=1)
            else:
                self.kps2D = kpsTrans

        # rotate the image
        transformVec = tf.reshape(transormMat, [-1])[:-1]
        self.img = tf.contrib.image.transform(tf.expand_dims(self.img, 0), tf.expand_dims(transformVec, 0),
                                   interpolation='BILINEAR')[0]

        # rotate the seg
        if self.validSeg:
            self.seg = tf.contrib.image.transform(tf.expand_dims(self.seg, 0), tf.expand_dims(transformVec, 0),
                                                  interpolation='NEAREST')[0]


        # set all black pixels (out of boundary) to mean pixel value
        self.img = tf.cond(self.setOOBPix, lambda:setBlackPixels(self.img, self.meanPixVal), lambda: self.img)
        # if self.setOOBPix:
        #     self.img = setBlackPixels(self.img, self.meanPixVal)

        # plt.imshow(mask.numpy() * 255)
        # plt.imshow(self.img.numpy()[:, :, [2, 1, 0]])

        return

    def cropAndResize(self, topLeft, bottomRight, outWidth, outHeight,
                      randomTrans = True, randomTransX = 5, randomTransY = 5):


        tf.assert_type(topLeft, tf.int32)
        tf.assert_type(bottomRight, tf.int32)

        # add random shift to bounding box
        if randomTrans:
            transX = tf.random_uniform((1,), -randomTransX, randomTransX, dtype=tf.int32)
            transY = tf.random_uniform((1,), -randomTransY, randomTransY, dtype=tf.int32)
            topLeft = topLeft + tf.concat([transX, transY], axis=0)
            bottomRight = bottomRight + tf.concat([transX, transY], axis=0)

        # a = [tf.to_float(topLeft[1])/tf.to_float(self.img.shape[0]),
        #                   tf.to_float(topLeft[0])/tf.to_float(self.img.shape[1]),
        #                   tf.to_float(bottomRight[1])/tf.to_float(self.img.shape[0]),
        #                   tf.to_float(bottomRight[0])/tf.to_float(self.img.shape[1])]
        boxes = tf.concat([(tf.to_float(topLeft[1])/tf.to_float(self.imgH),
                          tf.to_float(topLeft[0])/tf.to_float(self.imgW),
                          tf.to_float(bottomRight[1])/tf.to_float(self.imgH),
                          tf.to_float(bottomRight[0])/tf.to_float(self.imgW))], axis=0)
        boxes = tf.expand_dims(boxes, 0)

        # crop and resize the image
        self.img = tf.image.crop_and_resize(tf.expand_dims(self.img,0),
                                            boxes, crop_size=tf.to_int32(tf.stack([outHeight, outWidth])),
                                            box_ind=[0])[0]
        self.img = tf.cast(self.img, tf.uint8)
        self.img.set_shape([outHeight, outWidth, self.img.shape[2]])

        # set all black pixels (out of boundary) to mean pixel value
        self.img = tf.cond(self.setOOBPix, lambda: setBlackPixels(self.img, self.meanPixVal), lambda: self.img)
        # if self.setOOBPix:
        #     self.img = setBlackPixels(self.img, self.meanPixVal)

        # crop and resize the seg
        if self.validSeg:
            self.seg = tf.image.crop_and_resize(tf.expand_dims(self.seg,0), boxes,
                                                crop_size=tf.to_int32(tf.stack([outHeight, outWidth])), box_ind=[0],
                                                method='nearest')[0]
            self.seg = tf.cast(self.seg, tf.uint8)
            self.seg.set_shape([outHeight, outWidth, self.seg.shape[2]])

        # change kps accordingly
        if self.validKps:
            kpsCropped = self.kps2D[:,:2] - tf.to_float(topLeft)

            scaleW = tf.to_float(outWidth) / tf.to_float(bottomRight[0] - topLeft[0])
            scaleH = tf.to_float(outHeight) / tf.to_float(bottomRight[1] - topLeft[1])
            scale = tf.stack([scaleW, scaleH], axis=0)

            kpsScaled = kpsCropped * scale

            # add the valid col back
            if self.kpsValidCol:
                self.kps2D = tf.concat([kpsScaled[:, :2], self.kps2D[:, 2:]], axis=1)
                self.kps2D.set_shape([self.kps2D.shape[0], 3])
            else:
                self.kps2D = kpsScaled
                self.kps2D.set_shape([self.kps2D.shape[0], 2])

        return

    def addImgNoise(self, val):
        noiseImg = tf.random_uniform(self.img.shape, -val, val, dtype=tf.int32)
        self.img = noiseImg + tf.to_int32(self.img)
        self.img = tf.minimum(tf.maximum(self.img, 0), 255)
        self.img = tf.cast(self.img, tf.uint8)

        return self.img


    def getGaussianKernel(self, size=1, mean=0., std=0.75):
        dist = tf.distributions.Normal(tf.cast(mean, tf.float32), tf.cast(std, tf.float32))

        vals = dist.prob(tf.range(-tf.cast(size, tf.float32), tf.cast(size, tf.float32) + 1, dtype=tf.float32))

        gaussKernel = tf.einsum('i,j->ij',
                                vals,
                                vals)

        return gaussKernel / tf.reduce_sum(gaussKernel)


    def gaussFiltImg(self, size = 1, std=0.75):
        kernel = self.getGaussianKernel(size, 0, std)
        kernel = kernel[:, :, tf.newaxis, tf.newaxis]
        kernel = tf.tile(kernel, [1,1,3,1])

        pointwise_filter = tf.eye(3, batch_shape=[1, 1])
        self.img = tf.nn.separable_conv2d(tf.expand_dims(tf.to_float(self.img), 0), kernel, pointwise_filter,
                                       strides=[1, 1, 1, 1], padding='SAME')
        self.img = tf.squeeze(self.img)
        self.img = tf.cast(self.img, tf.uint8)

        return self.img


    def hueAug(self, val):
        self.img = tf.image.random_hue(self.img, val)

    def saturationAug(self, minval, maxval):
        self.img = tf.image.random_saturation(self.img, minval, maxval)

    def brightnessAug(self, val):
        self.img = tf.image.random_brightness(self.img, val)

def getOneGaussianHeatmap(inputs):
    grid = tf.to_float(inputs[0])
    mean = tf.to_float(inputs[1])
    std = tf.to_float(inputs[2])
    # assert std.shape == (1,)
    assert len(grid.shape) == 2
    assert grid.shape[-1] == 2

    mvn = tfd.MultivariateNormalDiag(
        loc=mean,
        scale_identity_multiplier=std)
    prob = mvn.prob(grid) * 2 * np.pi * std * std

    return prob

def computeHeatmaps(kps2D, patchSize, std=5.):
    '''
    gets the gaussian heat map for the keypoints
    :param kps2d:Nx2 tensor
    :param patchSize: hxw
    :param std: standard dev. for the gaussain
    :return:Nxhxw heatmap
    '''
    X, Y = tf.meshgrid(tf.range(patchSize[1]), tf.range(patchSize[0]))
    grid = tf.stack([X, Y], axis=2)
    grid = tf.reshape(grid, [-1, 2])
    grid_tile = tf.tile(tf.expand_dims(grid, 0), [kps2D.shape[0], 1, 1])
    heatmaps = tf.map_fn(getOneGaussianHeatmap, (grid_tile, kps2D[:, :2], tf.zeros(kps2D.shape[0], 1) + std), dtype=tf.float32)
    heatmaps = tf.reshape(heatmaps, [kps2D.shape[0], X.shape[0], X.shape[1]])

    return heatmaps