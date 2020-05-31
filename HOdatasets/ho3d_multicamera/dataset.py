from HOdatasets.utilsDS import *
from HOdatasets.commonDS import *
from HOdatasets.mypaths import *
import yaml
import png
import itertools
import warnings
import sys

if sys.version_info >= (3, 0):
    warnings.simplefilter("ignore", ResourceWarning)


dscale = 1
w = 640 // dscale
h = 480 // dscale


jointsMap = [0,
             13, 14, 15, 16,
             1, 2, 3, 17,
             4, 5, 6, 18,
             10, 11, 12, 19,
             7, 8, 9, 20]

splitEnumMap = {
    splitType.TRAIN: 'train',
    splitType.TEST: 'test',
    splitType.VAL: 'val',
}



class datasetHo3dMultiCamera(datasetBase):
    def __init__(self, seq, camInd, isCropImg=False, isRemoveBG=False, fileListIn=None):
        if fileListIn is None:
            fileList = os.listdir(join(HO3D_MULTI_CAMERA_DIR, seq, 'rgb', str(camInd)))
            fileList = [join(seq, '0', f[:-4]) for f in fileList if 'png' in f]
            fileList = sorted(fileList)
        else:
            fileList = fileListIn

        self.seqDir = HO3D_MULTI_CAMERA_DIR
        imgDir = join(HO3D_MULTI_CAMERA_DIR, seq, 'rgb')
        segDir = join(HO3D_MULTI_CAMERA_DIR, seq, 'segmentation', str(camInd), 'raw_seg_results')
        # metaDataDir = join(HO3D_CAMERA_DIR, split, 'meta')

        if sys.version_info >= (3, 0):
            super(datasetHo3dMultiCamera, self).__init__(fileList=fileList, imgDir=imgDir, segDir=segDir,
                                               metaDataDir=None, isSingleMetaFile=False,
                                               imgFormat='png', segFormat='png', metaFormat='pkl')
        else:
            datasetBase.__init__(self, fileList=fileList, imgDir=imgDir, segDir=segDir,
                                               metaDataDir=None, isSingleMetaFile=False,
                                               imgFormat='png', segFormat='png', metaFormat='pkl')

        self.isCropImg = isCropImg

        self.removeBG = isRemoveBG


    def readMeta(self, fileName, seq):
        if not os.path.exists(fileName):
            return {'pts2DObj': np.zeros((8,2), dtype=np.float32),
                    'pts3DObj': np.zeros((8,3), dtype=np.float32) + np.array([0., 0., -0.5]),
                    'pts2DHand': np.zeros((21,2), dtype=np.float32),
                    'pts3DHand': np.zeros((21,3), dtype=np.float32) + np.array([0., 0., -0.5])}

        pklData = loadPickleData((fileName + '.' + self.metaFormat))

        pts3DHand = pklData['JTransformed'][0]
        camMat = self.getCamMat(seq)
        camProp = camProps(ID='cam1', f=np.array([camMat[0,0], camMat[1,1]], dtype=np.float32) / dscale,
                           c=np.array([camMat[0,2], camMat[1,2]], dtype=np.float32) / dscale,
                           near=0.001, far=2.0, frameSize=[w, h],
                           pose=np.eye(4, dtype=np.float32))
        pts2DHand = cv2ProjectPoints(camProp, pts3DHand, isOpenGLCoords=True)
        if 'objCornersTransormed' not in pklData.keys():
            objID = seqToObjID[seq]
            objCornersFilename = join(YCB_MODELS_DIR, objID, 'corners.npy')
            objCorners = np.load(objCornersFilename)
            pts3DObj = np.matmul(objCorners, cv2.Rodrigues(pklData['rotObj'])[0].T) + pklData['transObj']
        else:
            pts3DObj = pklData['objCornersTransormed']
        pts2DObj = cv2ProjectPoints(camProp, pts3DObj, isOpenGLCoords=True)

        return {'pts2DObj': pts2DObj, 'pts3DObj': pts3DObj,
                'pts2DHand': pts2DHand, 'pts3DHand': pts3DHand}

    def load_depth(self, path):
        # PyPNG library is used since it allows to save 16-bit PNG
        if sys.version_info >= (3, 0):
            warnings.simplefilter("ignore", ResourceWarning)
        r = png.Reader(filename=path)
        if sys.version_info[0] >= 3:
            im = np.vstack(map(np.uint16, r.asDirect()[2])).astype(np.float32)
        else:
            im = np.vstack(itertools.imap(np.uint16, r.asDirect()[2])).astype(np.float32)
        return im


    def getCamMat(self, seq):
        camInd = 0
        camMatFile = os.path.join(self.seqDir, seq, 'calibration', 'cam_%s_intrinsics.txt' % (camInd))
        depthScaleFile = os.path.join(self.seqDir, seq, 'calibration', 'cam_%s_depth_scale.txt' % (camInd))

        if not os.path.exists(camMatFile):
            raise Exception('Where is the camera intrinsics file???')
        with open(camMatFile, 'r') as f:
            line = f.readline()
        line = line.strip()
        items = line.split(',')
        for item in items:
            if 'fx' in item:
                fx = float(item.split(':')[1].strip())
            elif 'fy' in item:
                fy = float(item.split(':')[1].strip())
            elif 'ppx' in item:
                ppx = float(item.split(':')[1].strip())
            elif 'ppy' in item:
                ppy = float(item.split(':')[1].strip())

        camMat = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])

        with open(depthScaleFile, 'r') as f:
            line = f.readline()
        depthScale = float(line.strip())

        return camMat

    def createTFExample(self, itemType='hand', fileIn=None):
        if fileIn is None:
            fId = self.getNextFileName()
        else:
            fId = fileIn

        seq = fId.split('/')[0]
        camInd = fId.split('/')[1]
        id = fId.split('/')[2]
        img = self.readImg(join(self.seqDir, seq, 'rgb', camInd, id), True)

        if self.removeBG:
            r = png.Reader(filename=join(self.seqDir, seq, 'depth', camInd, id+'.png'))
            dep = np.vstack(map(np.uint16, r.asDirect()[2])).astype(np.float32)
            depScaleFile = os.path.join(self.seqDir, seq, 'cam_%s_depth_scale.txt' % (camInd))
            if not os.path.exists(depScaleFile):
                depScaleFile = os.path.join(self.seqDir, seq, 'calibration', 'cam_%s_depth_scale.txt' % (camInd))
            with open(depScaleFile, 'r') as f:
                line = f.readline()
            line = line.strip()
            depScale = float(line)
            dep = dep*depScale


            depMask = np.logical_or(dep>0.95, dep==0)
            depMask = np.logical_not(depMask)
            img = img*np.expand_dims(depMask, 2)

        objInd = 1
        handInd = 2

        seg = self.readSeg(join(self.seqDir, seq, 'segmentation', camInd, 'raw_seg_results', id), True)
        if seg is None:
            seg = np.zeros((img.shape[1], img.shape[0], 3), dtype=np.uint8)
        seg = cv2.resize(seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        meta = self.readMeta(join(self.seqDir, seq, 'meta', id), seq)


        if self.isCropImg:
            # Get the centroid of the seg mask, and crop a window of size MPII_HO3D_CROP_SIZE around it
            imOps = ImageOps(img, seg, np.concatenate([meta['pts2DHand'], meta['pts2DObj']], axis=0))
            imgPatch, maskPatch, _, _, kpsAug = imOps.imgResizeAndCrop(img.shape[1], img.shape[0], MPII_HO3D_CROP_SIZE, MPII_HO3D_CROP_SIZE)
        else:
            imgPatch = img
            maskPatch = seg
            kpsAug = np.concatenate([meta['pts2DHand'], meta['pts2DObj']], axis=0)

        # Convert seg into single channel image
        newSeg = np.zeros((imgPatch.shape[0], imgPatch.shape[1]), dtype=np.uint8)
        newSeg[maskPatch[:, :, 0] == handInd] = handInd
        newSeg[maskPatch[:, :, 0] == objInd] = objInd


        # add the valid column for 2D keypoints
        kpsAug = np.concatenate([kpsAug, np.ones((kpsAug.shape[0], 1), dtype=np.float32)], axis=1)

        camMat = self.getCamMat(seq)
        depthScaleFile = os.path.join(self.seqDir, seq, 'calibration', 'cam_%s_depth_scale.txt' % (camInd))
        with open(depthScaleFile, 'r') as f:
            line = f.readline()
        depthScale = float(line.strip())

        # read the depth file
        depth = self.load_depth(join(self.seqDir, seq, 'depth', camInd, id+'.png'))*depthScale

        depthEnc = encodeDepthImg(depth)

        kps2DHomoHand = np.concatenate([kpsAug[:21, :2], np.ones((21, 1), dtype=np.float32)], axis=1)
        kps3DAugHand = np.matmul(kps2DHomoHand * np.abs(meta['pts3DHand'][:, 2:3]), np.linalg.inv(camMat).T)

        kps2DHomoObj = np.concatenate([kpsAug[21:, :2], np.ones((8, 1), dtype=np.float32)], axis=1)
        kps3DAugobj = np.matmul(kps2DHomoObj * np.abs(meta['pts3DObj'][:, 2:3]), np.linalg.inv(camMat).T)

        if itemType == 'hand':
            # only hands for now
            kps2DAug = kpsAug[:21, :][jointsMap]
            kps3DAug = kps3DAugHand[jointsMap]
        elif itemType == 'object':
            kps2DAug = kpsAug[21:, :]
            kps3DAug = kps3DAugobj
        elif itemType == 'hand_object':
            kps2DAug = kpsAug
            kps3DAug = np.concatenate([kps3DAugHand, kps3DAugobj], axis=0)
        else:
            raise NotImplementedError

        coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        ds = dataSample(img=imgPatch, seg=newSeg, fName=fId, dataset=datasetType.HO3D,
                        outType=outputType.SEG | outputType.KEYPOINT_3D | outputType.KEYPOINTS_2D,
                        pts2D=kps2DAug, pts3D=kps3DAug.dot(coordChangeMat), camMat=camMat, depth=None)


        return None, ds

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    ds = datasetHo3dMultiCamera('releaseTest',  0, isCropImg=False)
    numFiles = ds.getNumFiles()



    fig = plt.figure()
    ax = fig.subplots(2)
    plt.ioff()

    objInds = []
    for i in range(500,2000,50):
        _, sample = ds.createTFExample('hand', fileIn='releaseTest/0/%05d'%(i))

        objInds.append(np.max(sample.segRaw))

        # img = plotLines(sample.imgRaw, sample.pts2D[:,:2])

        ax[0].imshow(sample.imgRaw[:,:,[2,1,0]])
        # ax[1].imshow(decodeDepthImg(sample.depth))
        ax[1].imshow(sample.segRaw)

        print(i)

        a = 10

        plt.show()

    objInds = np.array(objInds)



