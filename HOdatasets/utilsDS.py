import pickle
from HOdatasets.commonDS import *
from HOdatasets.mypaths import *
import json

depthScale = 0.00012498664727900177 # this will be the depth scale used everywhere for encoding the depth in dataSample class

def loadPickleData(fName):
    with open(fName, 'rb') as f:
        try:
            pickData = pickle.load(f, encoding='latin1')
        except:
            pickData = pickle.load(f)

    return pickData

def savePickleData(fname, dictIn):
    with open(fname, 'wb') as f:
        pickle.dump(dictIn, f, protocol=2)

def loadJsonData(fName):
    with open(fName, 'r') as f:
        jsonData = json.load(f)

    return jsonData

def cv2ProjectPoints(camProp, pts3D, isOpenGLCoords=True):
    '''
    TF function for projecting 3d points to 2d using CV2
    :param camProp:
    :param pts3D:
    :param isOpenGLCoords:
    :return:
    '''
    assert isinstance(camProp, camProps), 'camProp should belong to camProps class'
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if isOpenGLCoords:
        pts3D = pts3D.dot(coordChangeMat.T)

    fx = camProp.f[0]
    fy = camProp.f[1]
    cx = camProp.c[0]
    cy = camProp.c[1]

    camMat = np.array([[fx, 0, cx], [0, fy, cy], [0., 0., 1.]])

    projPts = pts3D.dot(camMat.T)
    projPts = np.stack([projPts[:,0]/projPts[:,2], projPts[:,1]/projPts[:,2]],axis=1)

    assert len(projPts.shape) == 2

    return projPts

def loadAllObjectCorners():
    objCorners = {}
    objCorners['003_cracker_box'] = np.load(os.path.join(YCB_OBJECT_CORNERS_DIR, '003_cracker_box', 'corners.npy'))
    objCorners['004_sugar_box'] = np.load(os.path.join(YCB_OBJECT_CORNERS_DIR, '004_sugar_box', 'corners.npy'))
    objCorners['006_mustard_bottle'] = np.load(os.path.join(YCB_OBJECT_CORNERS_DIR, '006_mustard_bottle', 'corners.npy'))
    objCorners['025_mug'] = np.load(os.path.join(YCB_OBJECT_CORNERS_DIR, '025_mug', 'corners.npy'))
    objCorners['019_pitcher_base'] = np.load(os.path.join(YCB_OBJECT_CORNERS_DIR, '019_pitcher_base', 'corners.npy'))
    objCorners['035_power_drill'] = np.load(os.path.join(YCB_OBJECT_CORNERS_DIR, '035_power_drill', 'corners.npy'))
    objCorners['037_scissors'] = np.load(os.path.join(YCB_OBJECT_CORNERS_DIR, '037_scissors', 'corners.npy'))

    return objCorners

class ImageOps:
    '''
    Class for preprocessing images before feeding to network
    '''
    def __init__(self, img, mask, keyPts=None):
        self.img  = img
        self.mask = mask
        self.keyPts = keyPts
        # self.imgAug = ImageAug(img, mask, keyPts)

    # def augmentImg(self, isRandomBG=False, bgImgPreferred=None, prefProb=0.05, isAffine=True, isColor=True):
    #     self.img, self.mask, self.keyPts = self.imgAug.augment_img(isRandomBG, bgImgPreferred, prefProb, isAffine=isAffine, isColor=isColor)

    def imgResizeAndCrop(self, rsImgW, rsImgH, imgPatchW, imgPatchH, getCenterFromSeg=True):
        '''
        resize the image, crop it and get the patch for network input. If the crop is beyond the boundary, clip it to boundary
        :param rsImgW: Rescaled image width
        :param rdImgH: Rescaled image height
        :param imgPatchW: output patch width
        :param imgPatchH: output patch height
        :return:
        '''

        imgResc = cv2.resize(self.img, (rsImgW, rsImgH), interpolation=cv2.INTER_CUBIC)
        maskResc = cv2.resize(self.mask, (rsImgW, rsImgH), interpolation=cv2.INTER_NEAREST)
        keyPtsResc = self.keyPts * float(rsImgW) / float(self.img.shape[1])

        if getCenterFromSeg:
            xx, yy = np.meshgrid(np.arange(0, rsImgW), np.arange(0, rsImgH))
            if len(maskResc.shape) == 3:
                maskRescAll = (np.sum(maskResc, axis=2)>0).astype(np.uint8)
            else:
                maskRescAll = (maskResc>0).astype(np.uint8)
            xmean = np.round(np.sum(xx * maskRescAll) / np.sum(maskRescAll)).astype(np.uint32)
            ymean = np.round(np.sum(yy * maskRescAll) / np.sum(maskRescAll)).astype(np.uint32)
        else:
            tl = np.min(self.keyPts, axis=0)
            br = np.max(self.keyPts, axis=0)
            xmean = int((tl[0] + br[0]) / 2.)
            ymean = int((tl[1] + br[1]) / 2.)

        objCenter = np.array([xmean, ymean])  # x is horizontal to right

        startY = ymean - imgPatchH / 2
        endY = ymean + imgPatchH / 2
        startX = xmean - imgPatchW / 2
        endX = xmean + imgPatchW / 2
        if startY < 0:
            endY = endY - startY
            startY = 0
        if startX < 0:
            endX = endX - startX
            startX = 0
        if endY > imgResc.shape[0]:
            startY = startY - (endY - imgResc.shape[0])
            endY = imgResc.shape[0]
        if endX > imgResc.shape[1]:
            startX = startX - (endX - imgResc.shape[1])
            endX = imgResc.shape[1]
        xmean = startX + imgPatchW / 2
        ymean = startY + imgPatchH / 2
        imgPatch = imgResc[int(startY):int(endY), int(startX):int(endX)]
        maskPatch = maskResc[int(startY):int(endY), int(startX):int(endX)]

        kpsAug = keyPtsResc - np.array([xmean - imgPatchW / 2, ymean - imgPatchH / 2])

        return imgPatch, maskPatch, xmean, ymean, kpsAug

    def imgResizeAndCropExact(self, rsImgW, rdImgH, imgPatchW, imgPatchH):
        '''
        resize the image, crop it and get the patch for network input. If the crop is beyond the boundary, pad the image with 'average' pixel value.
        Note that the keypoint locations change now, which is why they are sent as part of args
        :param rsImgW: Rescaled image width
        :param rdImgH: Rescaled image height
        :param imgPatchW: output patch width
        :param imgPatchH: output patch height
        :return:
        '''
        # resize the image, crop it and get the patch for network input
        imgResc = cv2.resize(self.img, (rsImgW, rdImgH), interpolation=cv2.INTER_CUBIC)
        maskResc = cv2.resize(self.mask, (rsImgW, rdImgH), interpolation=cv2.INTER_NEAREST)
        keyPtsResc = self.keyPts*float(rsImgW)/float(self.img.shape[1])

        xx, yy = np.meshgrid(np.arange(0, rsImgW), np.arange(0, rdImgH))
        xmean = np.round(np.sum(xx * maskResc) / np.sum(maskResc)).astype(np.uint32)
        ymean = np.round(np.sum(yy * maskResc) / np.sum(maskResc)).astype(np.uint32)
        objCenter = np.array([xmean, ymean])  # x is horizontal to right

        mean_img = np.round(np.mean(np.reshape(imgResc, [-1, 3]), axis=0)).astype(np.uint8)

        dx = np.random.randint(-10,10)
        dy = np.random.randint(-10, 10)

        startY = ymean - imgPatchH / 2 + dy
        endY = ymean + imgPatchH / 2 + dy
        startX = xmean - imgPatchW / 2 + dx
        endX = xmean + imgPatchW / 2 + dx

        deltaStartY = 0
        deltaStartX = 0
        kpsAug = np.copy(keyPtsResc)
        if startY < 0:
            imgResc = np.concatenate([mean_img+np.zeros([-startY, imgResc.shape[1], 3], dtype=np.uint8), imgResc], axis=0)
            maskResc = np.concatenate([np.zeros([-startY, maskResc.shape[1]], dtype=np.uint8), maskResc], axis=0)
            deltaStartY = -startY
            kpsAug[:, 1] = keyPtsResc[:, 1] - startY
            endY = endY - startY
            startY = 0
        if startX < 0:
            imgResc = np.concatenate([mean_img+np.zeros([imgResc.shape[0], -startX, 3], dtype=np.uint8), imgResc], axis=1)
            maskResc = np.concatenate([np.zeros([maskResc.shape[0], -startX], dtype=np.uint8), maskResc],
                                     axis=1)
            deltaStartX = -startX
            kpsAug[:, 0] = keyPtsResc[:, 0] - startX
            endX = endX - startX
            startX = 0
        if endY > imgResc.shape[0]:
            imgResc = np.concatenate([imgResc, mean_img + np.zeros([endY-(imgResc.shape[0]-1), imgResc.shape[1], 3], dtype=np.uint8)], axis=0)
            maskResc = np.concatenate(
                [maskResc, np.zeros([endY - (maskResc.shape[0] - 1), maskResc.shape[1]], dtype=np.uint8)],
                axis=0)
        if endX > imgResc.shape[1]:
            imgResc = np.concatenate([imgResc, mean_img + np.zeros([imgResc.shape[0], endX-(imgResc.shape[1]-1), 3], dtype=np.uint8)],
                                     axis=1)
            maskResc = np.concatenate(
                [maskResc, np.zeros([maskResc.shape[0], endX - (maskResc.shape[1] - 1)], dtype=np.uint8)],
                axis=1)
        xmean = startX + imgPatchW / 2
        ymean = startY + imgPatchH / 2
        imgPatch = imgResc[startY:endY, startX:endX]
        maskPatch = maskResc[startY:endY, startX:endX]
        kpsAug = kpsAug - np.array([xmean - imgPatchW / 2, ymean - imgPatchH / 2])

        xmeanForEval = xmean - deltaStartX
        ymeanForEval = ymean - deltaStartY

        return imgPatch, maskPatch, xmean, ymean, kpsAug, xmeanForEval, ymeanForEval

def encodeDepthImg(depth, outFileName=None):
    '''
    Encode the depth (in mts) to BGR image. B has the residual, G has the factor, R is zero.
    Also save the image
    :param depth: depth in float
    :param outFileName: image name for saving
    :return: encoded image in uint8
    '''
    depthInt = np.round(depth/depthScale)
    depthInt = depthInt.astype(np.uint32)

    gChannel = depthInt // 256
    bChannel = depthInt % 256
    depthImg = np.stack([bChannel, gChannel, np.zeros_like(bChannel, dtype=np.uint8)], axis=2)
    depthImg = depthImg.astype(np.uint8)

    if outFileName is not None:
        cv2.imwrite(outFileName, depthImg.astype(np.uint8))

    return depthImg

def decodeDepthImg(inFileName, dsize=None):
    '''
    Decode the depth image to depth map in meters
    :param inFileName: input file name
    :return: depth map (float) in meters
    '''
    if isinstance(inFileName, str):
        depthImg = cv2.imread(inFileName)
    else:
        depthImg = inFileName
    if dsize is not None:
        depthImg = cv2.resize(depthImg, dsize, interpolation=cv2.INTER_CUBIC)

    dpt = depthImg[:, :, 0] + depthImg[:, :, 1] * 256
    dpt = dpt * depthScale

    return dpt