from ghope.common import *
import cv2

class LossObservs():
    def __init__(self, observsVirt, observsReal, lossMode):
        self.observsVirt = observsVirt
        self.observsReal = observsReal
        self.lossMode = lossMode
        self.numViews = (self.observsVirt.seg.shape)[0]


        # make some(?) initial checks
        assert isinstance(self.lossMode, renderModeEnum), 'lossMode should be renderModeEnum class'
        assert (self.observsVirt.seg.shape)[0] == (self.observsReal.seg.shape)[0], 'num of views not matching in real and virtual observations'
        assert self.observsVirt.isReal == False
        assert self.observsReal.isReal == True
        assert (self.observsReal.frameID.shape[0]) == self.numViews, 'Frame ID not provided for all real observations'
        assert (self.observsReal.mask.shape)[0] == self.numViews, 'Num of Masks should equal number of views'
        if self.lossMode == renderModeEnum.SEG:

            assert self.observsVirt.seg is not None, 'seg map not available in virtual observations'
            assert self.observsReal.seg is not None, 'seg map not available in real observations'
            assert (self.observsVirt.seg.shape) == (self.observsReal.seg.shape), 'real and virtual seg maps should have same shape'

        elif self.lossMode == renderModeEnum.DEPTH:

            assert self.observsVirt.depth is not None, 'depth map not available in virtual observations'
            assert self.observsReal.depth is not None, 'depth map not available in real observations'
            assert (self.observsVirt.depth.shape) == (
                self.observsReal.depth.shape), 'real and virtual depth maps should have same shape'

        elif self.lossMode == renderModeEnum.COLOR:

            assert self.observsVirt.col is not None, 'col image not available in virtual observations'
            assert self.observsReal.col is not None, 'col image not available in real observations'
            assert (self.observsVirt.col.shape) == (
                self.observsReal.col.shape), 'real and virtual col images should have same shape'

        elif self.lossMode == renderModeEnum.SEG_COLOR:

            assert self.observsVirt.seg is not None, 'seg map not available in virtual observations'
            assert self.observsReal.seg is not None, 'seg map not available in real observations'
            assert (self.observsVirt.seg.shape) == (
                self.observsReal.seg.shape), 'real and virtual seg maps should have same shape'

            assert self.observsVirt.col is not None, 'col image not available in virtual observations'
            assert self.observsReal.col is not None, 'col image not available in real observations'
            assert (self.observsVirt.col.shape) == (
                self.observsReal.col.shape), 'real and virtual col images should have same shape'

        elif self.lossMode == renderModeEnum.SEG_DEPTH:

            assert self.observsVirt.seg is not None, 'seg map not available in virtual observations'
            assert self.observsReal.seg is not None, 'seg map not available in real observations'
            assert (self.observsVirt.seg.shape) == (
                self.observsReal.seg.shape), 'real and virtual seg maps should have same shape'

            assert self.observsVirt.depth is not None, 'depth map not available in virtual observations'
            assert self.observsReal.depth is not None, 'depth map not available in real observations'
            assert (self.observsVirt.depth.shape) == (
                self.observsReal.depth.shape), 'real and virtual depth maps should have same shape'

        elif self.lossMode == renderModeEnum.COLOR_DEPTH:

            assert self.observsVirt.col is not None, 'col image not available in virtual observations'
            assert self.observsReal.col is not None, 'col image not available in real observations'
            assert (self.observsVirt.col.shape) == (
                self.observsReal.col.shape), 'real and virtual col images should have same shape'

            assert self.observsVirt.depth is not None, 'depth map not available in virtual observations'
            assert self.observsReal.depth is not None, 'depth map not available in real observations'
            assert t(self.observsVirt.depth.shape) == (
                self.observsReal.depth.shape), 'real and virtual depth maps should have same shape'

        elif self.lossMode == renderModeEnum.SEG_COLOR_DEPTH:

            assert self.observsVirt.seg is not None, 'seg map not available in virtual observations'
            assert self.observsReal.seg is not None, 'seg map not available in real observations'
            assert (self.observsVirt.seg.shape) == (
                self.observsReal.seg.shape), 'real and virtual seg maps should have same shape'

            assert self.observsVirt.col is not None, 'col image not available in virtual observations'
            assert self.observsReal.col is not None, 'col image not available in real observations'
            assert (self.observsVirt.col.shape) == (
                self.observsReal.col.shape), 'real and virtual col images should have same shape'

            assert self.observsVirt.depth is not None, 'depth map not available in virtual observations'
            assert self.observsReal.depth is not None, 'depth map not available in real observations'
            assert (self.observsVirt.depth.shape) == (
                self.observsReal.depth.shape), 'real and virtual depth maps should have same shape'

        else:
            raise Exception('Invalid renderMode..')

    def gaussianPyr(self, err):
        '''
        Pyramidal downscaling and loss computaion
        :param err: error map which is already SQUARED
        :return: downscaled error map and loss value
        '''
        oneway = np.tile(cv2.getGaussianKernel(3, 1), (1, 3))
        gaussKernel = oneway * oneway.T
        gaussKernel = gaussKernel.astype(np.float32)
        # if len(err.shape) == 4:
        #     gaussKernel = np.stack([gaussKernel, gaussKernel, gaussKernel], axis=2)
        gaussKernel = np.expand_dims(gaussKernel, 3)
        gaussKernel = np.expand_dims(gaussKernel, 3)
        gaussKernel = tf.tile(gaussKernel, [1, 1, tf.shape(err)[-1], 1])

        gaussErr = tf.nn.conv2d(err, gaussKernel, strides=[1, 1, 1, 1], padding="VALID")

        gaussErrDown = gaussErr[:, ::2, ::2, :]

        gaussErrDownLoss = tf.reduce_mean(gaussErrDown)

        return gaussErrDown, gaussErrDownLoss

    def getL2Loss(self, isClipDepthLoss=False, pyrLevel = 0):
        '''
        Get the loss functions for seg, depth and col
        :param isClipDepthLoss: clip depth err to 0.5 mts?
        :param pyrLevel: num of pyramid levels at which the loss is calculated
        :return: seg loss, depth loss and col loss. Will be *None* value depending on lossMode
        '''
        # some inits
        errSegMap = None
        errDepthMap = None
        errColMap = None

        segLoss = None
        depthLoss = None
        colLoss = None

        # self.observsReal.depth = tf.log(self.observsReal.depth)
        # self.observsVirt.depth = tf.log(self.observsVirt.depth)

        # depedning on the loss mode calculate the error maps
        if self.lossMode == renderModeEnum.SEG:

            errSegMap = tf.abs(self.observsReal.seg - self.observsVirt.seg)*self.observsReal.mask

        elif self.lossMode == renderModeEnum.DEPTH:

            errDepthMap = tf.abs(self.observsReal.depth - self.observsVirt.depth) * self.observsReal.mask

        elif self.lossMode == renderModeEnum.COLOR:

            errColMap = tf.abs(self.observsReal.col - self.observsVirt.col) * self.observsReal.mask

        elif self.lossMode == renderModeEnum.SEG_COLOR:

            errSegMap = tf.abs(self.observsReal.seg - self.observsVirt.seg) * self.observsReal.mask
            errColMap = tf.abs(self.observsReal.col - self.observsVirt.col) * self.observsReal.mask

        elif self.lossMode == renderModeEnum.SEG_DEPTH:

            errSegMap = tf.abs(self.observsReal.seg - self.observsVirt.seg) * self.observsReal.mask
            errDepthMap = tf.abs(self.observsReal.depth - self.observsVirt.depth) * self.observsReal.mask

        elif self.lossMode == renderModeEnum.COLOR_DEPTH:

            errColMap = tf.abs(self.observsReal.col - self.observsVirt.col) * self.observsReal.mask
            errDepthMap = tf.abs(self.observsReal.depth - self.observsVirt.depth) * self.observsReal.mask

        elif self.lossMode == renderModeEnum.SEG_COLOR_DEPTH:

            errSegMap = tf.abs(self.observsReal.seg - self.observsVirt.seg) * self.observsReal.mask
            errColMap = tf.abs(self.observsReal.col - self.observsVirt.col) * self.observsReal.mask
            errDepthMap = tf.abs(self.observsReal.depth - self.observsVirt.depth) * self.observsReal.mask

        else:

            raise Exception('Invalid renderMode..')

        # clip depth err if required
        if isClipDepthLoss:
            errDepthMap = tf.minimum(errDepthMap, 0.5)

        # get the final loss scaler, do pyramidal downscaling if required
        if errSegMap is not None:
            errSegMap = tf.square(errSegMap)
            segLoss = tf.reduce_mean(errSegMap)
            if pyrLevel > 0:
                for _ in range(pyrLevel):
                    errSegMap, pyrLossSeg = self.gaussianPyr(errSegMap)
                    segLoss += pyrLossSeg

        if errDepthMap is not None:
            # errDepthMap = tf.square(errDepthMap)
            depthLoss = tf.reduce_mean(errDepthMap)
            if pyrLevel > 0:
                for _ in range(pyrLevel):
                    errDepthMap, pyrLossDepth = self.gaussianPyr(errDepthMap)
                    depthLoss += pyrLossDepth

        if errColMap is not None:
            errColMap = tf.square(errColMap)
            colLoss = tf.reduce_mean(errColMap)
            if pyrLevel > 0:
                for _ in range(pyrLevel):
                    errColMap, pyrLossCol = self.gaussianPyr(errColMap)
                    colLoss += pyrLossCol

        return segLoss, depthLoss, colLoss

    @staticmethod
    def getRealObservables(dataset, numViews, w, h):
        frameCntIntV = tf.Variable(0, name='FrameCnt', dtype=tf.int32)
        loadRealObservs = tf.placeholder(tf.bool, name='OBSERVS_GATE')

        fidV = tf.get_variable(initializer=np.array(['abcd']*numViews), name='frameID', dtype=tf.string)
        segV = tf.get_variable(initializer=np.zeros((numViews, h, w, 3), dtype=np.float32), name='seg', dtype=tf.float32)
        depthV = tf.get_variable(initializer=np.zeros((numViews, h, w, 3), dtype=np.float32), name='dep', dtype=tf.float32)
        colV = tf.get_variable(initializer=np.zeros((numViews, h, w, 3), dtype=np.float32), name='col', dtype=tf.float32)
        maskV = tf.get_variable(initializer=np.zeros((numViews, h, w, 3), dtype=np.float32), name='mask', dtype=tf.float32)

        def loadVars(fidV, segV, depthV, colV, maskV, frameCntIntV):
            frameID, seg, depth, col, mask = dataset.make_one_shot_iterator().get_next()
            fidV = tf.assign(fidV, frameID)
            segV = tf.assign(segV, seg)
            depthV = tf.assign(depthV, depth)
            colV = tf.assign(colV, col)
            maskV = tf.assign(maskV, mask)
            frameCntIntV = frameCntIntV + 1
            segV = tf.Print(segV, ['Loading New frame ', fidV])
            return fidV, segV, depthV, colV, maskV, frameCntIntV

        def dummyFunc(fidV, segV, depthV, colV, maskV, frameCntIntV):
            # segV = tf.Print(segV, ['Reusing frame ', fidV])
            return fidV, segV, depthV, colV, maskV, frameCntIntV

        frameID, seg, depth, col, mask, frameCntInt = tf.cond(loadRealObservs, lambda: loadVars(fidV, segV, depthV, colV, maskV, frameCntIntV),
                                                lambda: dummyFunc(fidV, segV, depthV, colV, maskV, frameCntIntV))

        realObservs = observables(frameID=frameID, seg=seg, depth=depth, col=col, mask=mask, isReal=True)

        return frameCntInt, loadRealObservs, realObservs

    @staticmethod
    def getVarInits(dataset, varList, loadInits):

        def loadVars(varListL):
            initList = dataset.make_one_shot_iterator().get_next()
            for i, var in enumerate(varListL):
                var = tf.assign(var, initList[i])
            varListL[0] = tf.Print(varListL[0], ['Loading New Variable inits'])
            return varListL

        def dummyFunc(varListL):
            # segV = tf.Print(segV, ['Reusing frame ', fidV])
            return varListL

        varList = tf.cond(loadInits, lambda: loadVars(varList), lambda: dummyFunc(varList))

        return


