import tensorflow as tf
import numpy as np

class networkData():
    '''
    Data structure for hand Keypoints & segmentation network
    '''
    def __init__(self, image, label, kps2D, kps3D, imageID, h, w, outType, dsName,
                 topLeft = None,
                 bottomRight = None,
                 extraScale=None, camMat = None):
        if label is not None:
            if label.shape.ndims == 2:
                label = tf.expand_dims(label, 2)
            elif label.shape.ndims == 3 and label.shape.dims[2] == 1:
                pass
            else:
                raise ValueError('Input label shape must be [height, width], or '
                                 '[height, width, 1].')

        if label is not None:
            label.set_shape([None, None, 1])

        if kps2D is not None:
            kps2D = tf.reshape(kps2D,[-1, 3])
        if kps3D is not None:
            kps3D = tf.reshape(kps3D, [-1, 3])


        self.image = image
        self.label = label
        self.kps2D = kps2D
        self.kps3D = kps3D
        self.imageID = imageID
        self.width = w
        self.height = h
        self.outputType = outType
        self.datasetName = dsName

        self.topLeft = topLeft
        self.bottomRight = bottomRight
        self.extraScale = extraScale
        self.camMat = camMat