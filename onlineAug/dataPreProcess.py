from onlineAug.augmenter import *

from onlineAug.utilsAug import *

slim = tf.contrib.slim

dataset = slim.dataset

tfexample_decoder = slim.tfexample_decoder

dataset_data_provider = slim.dataset_data_provider


# Semantic segmentation item names.
LABELS_CLASS = 'labels_class'
KPS_2D = 'keypoints2D'
KPS_3D = 'keypoints3D'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'
DATAOUT_TYPE = 'outputType'
DATASET_NAME = 'datasetName'
TEST_SET = 'test'
HEATMAPS = 'heatmaps'
CAMMAT = 'camMat'

def preProcessData(data, dataAugment, is_training, crop_size, ignore_label,
                   isConstantCropPatchSize = False, cropPatchSize = [256, 256], cropCenter = None):
    assert isinstance(data, networkData)

    setOOBPix = tf.logical_not(tf.equal(data.datasetName, 1))
    gaussSize = tf.cond(tf.logical_or(tf.equal(data.datasetName, 3),
                                      tf.equal(data.datasetName, 4)),
                        lambda: 0, lambda: 1)


    camMatObman = lambda: np.array([[480, 0 ,128], [0, 480, 128], [0, 0, 1]], dtype=np.float32)
    camMatHo3d = lambda: np.array([[617.343, 0, 312.42], [0, 617.343, 241.42], [0, 0, 1]], dtype=np.float32)
    camMatAssert = lambda: np.array([[617.343, 0, 312.42], [0, 617.343, 241.42], [0, 0, 1]], dtype=np.float32)# TODO:fix this

    camMat = tf.case({tf.equal(data.datasetName, 1): camMatObman, tf.equal(data.datasetName, 2): camMatHo3d,
                     tf.greater_equal(data.datasetName, 4): lambda: tf.identity(data.camMat)},
                default=camMatAssert, exclusive=True)

    # change to opencv if required. 3D coords ALWAYS IN OPENCV
    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    data.kps3D = tf.cond(tf.less(data.kps3D[0,2], 0.), lambda: tf.matmul(data.kps3D, coordChangeMat), lambda: data.kps3D)

    # data augmentation
    if dataAugment:
        dataAug = HKPSAug(data, crop_height=crop_size[0], crop_width=crop_size[1], setOOBPix=setOOBPix,
                          gaussSize=gaussSize,
                          randomScaleMin=1.1, randomScaleMax=1.3, isTraining=is_training, augment3D=True, camMat=camMat,
                          isConstantPatchSize=isConstantCropPatchSize, patchSize=cropPatchSize, cropCenter=cropCenter)
    else:
        dataAug = data
        dataAug.image.set_shape([crop_size[0], crop_size[1], 3])
        dataAug.label.set_shape([crop_size[0], crop_size[1], 1])


    # set segmentation map to ignore label if dataout_type&1==0
    cond = tf.greater(tf.bitwise.bitwise_and(dataAug.outputType, 1), 0)
    dataAug.label = tf.cond(cond, lambda: dataAug.label, lambda: dataAug.label * 0 + ignore_label)

    # make 2d kps invalid if not in output
    cond = tf.greater(tf.bitwise.bitwise_and(dataAug.outputType, 2), 0)
    dataAug.kps2D = tf.cond(cond, lambda: dataAug.kps2D, lambda: dataAug.kps2D * 0)

    # normalize the depths
    norm_factor = tf.norm(dataAug.kps3D[5] - dataAug.kps3D[0])
    # norm_factor = tf.Print(norm_factor, ['norm factor', norm_factor])
    dataAug.kps3D = dataAug.kps3D / norm_factor

    # add valid column to 3d keypoints from 2d keypoints
    dataAug.kps3D = tf.concat([dataAug.kps3D, dataAug.kps2D[:, 2:]], axis=1)

    # get the heatmaps from 2d keypoints
    heatmaps = computeHeatmaps(dataAug.kps2D[:, :2], crop_size, std=5)
    heatmaps = tf.transpose(heatmaps, [1, 2, 0])

    # dataAug.kps3D = tf.Print(dataAug.kps3D, [dataAug.kps3D, dataAug.imageID])

    sample = {
        IMAGE: dataAug.image,
        IMAGE_NAME: dataAug.imageID,
        HEIGHT: dataAug.height,
        WIDTH: dataAug.width,
        KPS_2D: dataAug.kps2D,
        KPS_3D: dataAug.kps3D,
        LABEL: dataAug.label,
        DATAOUT_TYPE: dataAug.outputType,
        DATASET_NAME: dataAug.datasetName,
        HEATMAPS: heatmaps
    }

    # also return cropping config which will be useful during inference. Not useful during training
    return sample, dataAug.topLeft, dataAug.bottomRight, dataAug.extraScale


