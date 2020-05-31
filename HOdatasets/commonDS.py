import cv2
from enum import IntEnum
import numpy as np
from random import shuffle
import os
from os.path import join as join

MPII_HO3D_CROP_SIZE = 300

class datasetType(IntEnum):
    '''
    Enum for different datatypes
    '''

    OBMAN = 1

    HO3D = 2

    MPII = 3

    MTC = 4

    YCB = 5

    HO3D_CAMERA = 6

    HO3D_MULTICAMERA = 7

    FREIHAND = 8

    MANOHANDS = 9

class handType(IntEnum):
    '''
    Enum for right/left hand in dataset
    '''

    LEFT = 1

    RIGHT = 2

    LEFT_RIGHT = 3

class outputType(IntEnum):
    '''
    Enum for type of output for the sample
    '''

    SEG = 1
    KEYPOINTS_2D = 2
    KEYPOINT_3D = 4

class splitType(IntEnum):
    '''
    Enum for split type
    '''
    TRAIN = 1
    TEST = 2
    VAL = 3

class segIndex(IntEnum):
    '''
    Segmentation index for hand and object
    '''

    HAND_RIGHT = 1
    OBJECT = 2
    HAND_LEFT = 3



class dataSample():
    '''
    This class stores all the network data i.e., inputs and outptus
    '''
    def __init__(self, img, fName, dataset, outType, seg=None, pts2D=None, pts3D=None, camMat=None, depth=None):
        '''
        :param img: raw input image, 3channels
        :param seg: raw seg map, 1channel
        :param fName: filename
        :param pts2D: 2D keypoints
        :param pts3D: 3D keypoints
        :param dataset: dataset Enum
        :param outType: bitwise or operation of outputType enums, specifying which outputs are valid
        '''

        # some initial checks
        assert len(img.shape) == 3, 'Image should be 3 channels'
        assert img.dtype == np.uint8
        if not seg is None:
            assert len(seg.shape) == 2, 'Seg should have one channel'
            assert seg.dtype == np.uint8
        if not pts2D is None:
            if pts3D is not None:
                assert len(pts2D.shape) == len(pts3D.shape)
            assert len(pts2D.shape) == 2
            assert isinstance(pts2D, np.ndarray)
            assert pts2D.shape[-1] == 3, 'pts2D should have a valid column also'
        if not pts3D is None:
            assert isinstance(pts3D, np.ndarray)
            assert pts3D.shape[-1] == 3
        assert isinstance(fName, str)
        if seg is not None:
            assert img.shape[:2] == seg.shape[:2], 'img and seg should of same size'
        assert isinstance(dataset, datasetType)

        # set outputs to zero if they are not available
        if seg is None:
            seg = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        if pts2D is None:
            pts2D = np.zeros((1, 2), dtype=np.float32)
        if pts3D is None:
            pts3D = np.zeros((pts2D.shape[0], 3), dtype=np.float32)
        else:
            assert np.max(pts3D[:,2]) <= 0, '3D coordinates should be in openGL coordinate system'

        if depth is not None:
            assert depth.shape == img.shape
            self.depth = depth
        else:
            self.depth = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        assert pts2D.shape[0] == pts3D.shape[0]

        self.imgRaw = img
        self.segRaw = seg
        self.pts2D = pts2D.astype(np.float32)
        self.pts3D = pts3D.astype(np.float32)
        self.fileName = fName
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.imgChannels = 3
        self.dataset = dataset
        self.outputType = outType
        self.camMat = camMat

    def encodeImages(self, formatImg='.jpg', formatSeg='.png'):
        '''
        Encode the image and seg in the input format
        :param format:
        :return:
        '''
        if not '.' in formatImg:
            formatImg = '.' + formatImg

        if not '.' in formatSeg:
            formatSeg = '.' + formatSeg

        self.imgEnc = cv2.imencode(formatImg, self.imgRaw)[1].tostring()
        self.segEnc = cv2.imencode(formatSeg, self.segRaw)[1].tostring()
        self.encFormatImg = formatImg
        self.encFormatSeg = formatSeg


class datasetBase():
    def __init__(self, fileList, imgDir, segDir=None, metaDataDir=None, isSingleMetaFile=False,
                 imgFormat='png', segFormat='png', metaFormat='mat'):
        assert isinstance(fileList, list)

        if isinstance(fileList[0], str):
            fileList = [x.strip() for x in fileList]
        self.fileList = fileList
        self.numSamples = len(fileList)

        self.startIdx = 0
        self.endIdx = len(fileList)
        self.currFileInd = self.startIdx

        self.imgDir = imgDir
        self.segDir = segDir
        self.metaDataDir = metaDataDir

        self.isSingleMetaFile = isSingleMetaFile

        self.imgFormat = imgFormat
        self.segFormat = segFormat
        self.metaFormat = metaFormat

    def getNumFiles(self):
        return self.numSamples

    def setStartEndFiles(self, start, end):
        self.startIdx = start
        self.endIdx = end
        self.currFileInd = self.startIdx

    def shuffleFileList(self):
        shuffle(self.fileList)

    def getNextFileName(self):
        currInd = self.currFileInd
        self.currFileInd = self.currFileInd + 1
        self.currFileInd = self.currFileInd % self.endIdx
        if self.currFileInd == 0:
            self.currFileInd = self.startIdx
        return self.fileList[currInd]

    def readImg(self, fileName, isFullPath=False):
        if isFullPath:
            img = cv2.imread(fileName+'.'+self.imgFormat)
        else:
            img = cv2.imread(join(self.imgDir, fileName+ '.'+self.imgFormat))

        return img

    def readSeg(self, fileName, isFullPath=False):
        if isFullPath:
            seg = cv2.imread(fileName+'.'+self.segFormat)
        else:
            if self.segDir is None:
                raise Exception('Segmentation directory not set')
            else:
                seg = cv2.imread(join(self.segDir, fileName+ '.'+self.segFormat))
        return seg

    # def readMeta(self, fileName):
    #     raise NotImplementedError()

    def createTFExample(self, itemType=None, fileIn=None):
        raise NotImplementedError()

class camProps(object):
    '''
    Struct for saving properties of the camera
    '''
    def __init__(self, ID, f, c, near, far, frameSize, pose):
        self.ID = ID
        self.f = f
        self.c = c
        self.near = near
        self.far = far
        self.frameSize = frameSize
        self.pose = pose

    def getCamMat(self):
        camMat = np.array([[self.f[0], 0, self.c[0]],[0., self.f[1], self.c[1]],[0., 0., 1.]])
        return camMat

