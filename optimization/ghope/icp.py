from ghope.common import *
from ghope.utils import *

class Icp():
    def __init__(self, realObservs, camProp, optMode=optModeEnum.MULTIFRAME_JOINT, numCams = 1, distThresh=None):
        assert isinstance(realObservs, observables)
        # assert realObservs.seg.shape[0] == 1, 'Multi camera ICP not supported yet!!'
        # assert optMode != optModeEnum.MULTICAMERA, 'Multi camera ICP not supported yet!!'
        self.numViews = realObservs.seg.shape[0]
        self.optMode = optMode
        self.camProp = camProp
        if distThresh == None:
            self.distThresh = 999.
        else:
            self.distThresh = distThresh

        if optMode != optModeEnum.MULTICAMERA and optMode != optModeEnum.MUTLICAMERA_MULTIFRAME:
            assert isinstance(camProp, camProps)
            [px, py] = tf.meshgrid(np.arange(0, int(realObservs.seg.shape[2])), np.arange(0, int(realObservs.seg.shape[1])))
            # move the meshgrid to the center, same coordinate system as clipping space of renderer
            px = px - camProp.c[0]
            py = py - camProp.c[1]
            py = -py
            px = tf.cast(px, tf.float32)
            py = tf.cast(py, tf.float32)

            # get (x,y,z) point cloud coordinates from depth
            xx = px * realObservs.depth[:, :, :, 0] / camProp.f[0]
            yy = py * realObservs.depth[:, :, :, 0] / camProp.f[1]
            zz = -realObservs.depth[:, :, :, 0]
            ones = tf.ones((xx.shape[0], xx.shape[1], xx.shape[2]), dtype=tf.float32)
            self.pcl2DAllViews = tf.stack([xx, yy, zz, ones], axis=3)#
        elif optMode == optModeEnum.MULTICAMERA:
            assert isinstance(camProp, list)
            assert len(camProp) == self.numViews
            self.numCams = self.numViews
            pcl2DList = []
            for i in range(self.numViews):
                [px, py] = tf.meshgrid(np.arange(0, int(realObservs.seg.shape[2])),
                                       np.arange(0, int(realObservs.seg.shape[1])))
                # move the meshgrid to the center, same coordinate system as clipping space of renderer
                px = px - camProp[i].c[0]
                py = py - camProp[i].c[1]
                py = -py
                px = tf.cast(px, tf.float32)
                py = tf.cast(py, tf.float32)

                # get (x,y,z) point cloud coordinates from depth
                xx = px * realObservs.depth[i, :, :, 0] / camProp[i].f[0]
                yy = py * realObservs.depth[i, :, :, 0] / camProp[i].f[1]
                zz = -realObservs.depth[i, :, :, 0]
                ones = tf.ones((xx.shape[0], xx.shape[1]), dtype=tf.float32)
                pcl2D = tf.stack([xx, yy, zz, ones], axis=2)  #
                pcl2DList.append(pcl2D)
            self.pcl2DAllViews = tf.stack(pcl2DList, axis=0)
        elif optMode == optModeEnum.MUTLICAMERA_MULTIFRAME:
            #FRAMES:CAMERAS:H:W:C
            self.numCams = numCams
            assert numCams > 1
            self.numFrames = int(int(self.numViews)/int(numCams))
            assert isinstance(camProp, list)
            assert len(camProp) == numCams
            pcl2DListFrame = []
            for f in range(self.numFrames):
                currFrameIndx = f*numCams
                pcl2DListCam = []
                for i in range(numCams):
                    [px, py] = tf.meshgrid(np.arange(0, int(realObservs.seg.shape[2])),
                                           np.arange(0, int(realObservs.seg.shape[1])))
                    # move the meshgrid to the center, same coordinate system as clipping space of renderer
                    px = px - camProp[i].c[0]
                    py = py - camProp[i].c[1]
                    py = -py
                    px = tf.cast(px, tf.float32)
                    py = tf.cast(py, tf.float32)

                    # get (x,y,z) point cloud coordinates from depth
                    xx = px * realObservs.depth[currFrameIndx+i, :, :, 0] / camProp[i].f[0]
                    yy = py * realObservs.depth[currFrameIndx+i, :, :, 0] / camProp[i].f[1]
                    zz = -realObservs.depth[currFrameIndx+i, :, :, 0]
                    ones = tf.ones((xx.shape[0], xx.shape[1]), dtype=tf.float32)
                    pcl2D = tf.stack([xx, yy, zz, ones], axis=2)  #
                    pcl2DListCam.append(pcl2D)
                pcl2DAllCams = tf.stack(pcl2DListCam, axis=0)
                pcl2DListFrame.append(pcl2DAllCams)
            self.pcl2DAllViews = tf.stack(pcl2DListFrame, axis=0) #numFramesxnumCamsxHxW

        self.realObservs = realObservs



    def getLossPerFrame(self, inputs):
        '''
        Internal function. Calculates ICP loss for each frame
        :param inputs:
        :return:
        '''
        verts = inputs[0]
        pcl2D = inputs[1]
        seg = inputs[2]
        segColor = inputs[3]


        # get the point cloud
        pcl1D = tf.reshape(pcl2D, [-1, 4])  # N1x4
        mask1D = tf.reshape(tf.less(tf.reduce_sum(tf.abs(seg-segColor), axis=2), 0.05), [-1]) #N1,
        pclValid = tf.boolean_mask(pcl1D, mask1D)  # Nx4

        # get the nearest neighbor
        pclValid = tf.expand_dims(pclValid, 1)  # Nx1x4, N->Number of points in pcl
        nnArg = tf.stop_gradient(
            tf.arg_min(tf.reduce_sum(tf.abs(pclValid - verts), axis=2), dimension=1))  # N,
        nnVert = tf.gather(verts, nnArg)  # Nx4
        loss = tf.reduce_mean(tf.abs(pclValid[:, 0, :] - nnVert))
        return loss

    def getLossMulticam(self, inputs):
        verts = inputs[0]
        seg = inputs[1]
        segColor = inputs[2]
        pcl2DAllViewsL = inputs[3]

        verts = tf.expand_dims(verts, 0)#1xMx4
        # transform the point clouds
        pcl1DList = []
        mask1DList = []
        for i in range(self.numCams):
            pcl1DCam = tf.reshape(pcl2DAllViewsL[i], [-1, 4])  # N1x4
            pose = self.camProp[i].pose
            pcl1DCam = tf.transpose(tf.matmul(pose, tf.transpose(pcl1DCam)))
            mask1DCam = tf.reshape(tf.less(tf.reduce_sum(tf.abs(seg[i] - segColor), axis=2), 0.05), [-1])
            mask1DCam = tf.logical_and(mask1DCam, tf.reshape(tf.greater(self.realObservs.depth[i][:,:,0], 0.), [-1])) # remove holes
            mask1DList.append(mask1DCam)
            pcl1DList.append(pcl1DCam)
        pcl1D = tf.concat(pcl1DList, axis=0) #N2x4
        mask1D = tf.concat(mask1DList, axis=0) #N2
        if True:
            numPointsToSample = 5000
            randInds = tf.random_shuffle(
                tf.boolean_mask(tf.range(0, mask1D.shape[0], dtype=tf.int32), mask1D))
            pclValid = tf.cond(tf.reduce_sum(tf.cast(mask1D, tf.int32)) < numPointsToSample,
                                       lambda: tf.boolean_mask(pcl1D, mask1D),
                                       lambda: tf.gather(pcl1D, randInds[:numPointsToSample]))
        else:
            pclValid = tf.boolean_mask(pcl1D, mask1D)  # Nx4

        #data to model loss
        # get the nearest neighbor
        pclValid = tf.expand_dims(pclValid, 1)  # Nx1x4, N->Number of points in pcl
        nnArg = tf.stop_gradient(
            tf.arg_min(tf.reduce_sum(tf.abs(pclValid - verts), axis=2), dimension=1))  # N,
        nnVert = tf.gather(verts[0], nnArg)  # Nx4

        #remove far off points
        vertWithinThreshMask = tf.cast(tf.less(tf.sqrt(tf.reduce_sum(tf.square(nnVert - pclValid[:,0,:]),1)), self.distThresh), tf.float32)
        vertWithinThreshMask = tf.stop_gradient(tf.expand_dims(vertWithinThreshMask, 1)) #Nx1
        loss = tf.reduce_sum(tf.square(pclValid[:, 0, :] - nnVert)*vertWithinThreshMask)/tf.stop_gradient(tf.cast(tf.shape(pclValid)[0], tf.float32))

        # model to data loss
        if True:
            verts = tf.expand_dims(verts[0], 1)  # Mx1x4, M->Number of points in pcl
            nnRealPclArg = tf.stop_gradient(
                tf.arg_min(tf.reduce_sum(tf.abs(tf.expand_dims(tf.squeeze(pclValid),0) - verts), axis=2), dimension=1))  # M,
            nnRealPcl = tf.gather(tf.squeeze(pclValid), nnRealPclArg)  # Nx4

            # remove far off points
            realPclWithinThreshMask = tf.cast(
                tf.less(tf.sqrt(tf.reduce_sum(tf.square(nnRealPcl - verts[:, 0, :]), 1)), self.distThresh), tf.float32)
            realPclWithinThreshMask = tf.stop_gradient(tf.expand_dims(realPclWithinThreshMask, 1))  # Mx1
            lossMtoD = tf.reduce_sum(tf.square(verts[:, 0, :] - nnRealPcl) * realPclWithinThreshMask) / tf.stop_gradient(
                tf.cast(tf.shape(verts)[0], tf.float32))

            loss = lossMtoD + loss


        # self.pclReal = pclValid
        # self.nearestVert = nnRealPcl
        return loss

    def getLoss(self, finalMesh, segColor=np.array([1.,1.,1.], dtype=np.float32)):
        '''
        calculates ICP loss in batch (for all frames) for the item with segColor specified. Call multiple times for multiple items ICP
        :param finalMesh: NumViewsxNumVertx4 vertices array
        :param segColor: segColor for the chosen item
        :return: loss, a scalar value
        '''
        # assert hasattr(finalMesh, 'vUnClipped')
        assert len(finalMesh.shape) == 3
        if self.optMode != optModeEnum.MULTICAMERA and self.optMode != optModeEnum.MUTLICAMERA_MULTIFRAME:
            segColor = np.tile(np.expand_dims(segColor.astype(np.float32),0), [self.numViews, 1])
            loss = tf.map_fn(self.getLossPerFrame, (finalMesh, self.pcl2DAllViews, self.realObservs.seg, segColor), parallel_iterations=int(self.numViews), dtype=tf.float32)
            loss = tf.reduce_mean(loss)
        elif self.optMode == optModeEnum.MULTICAMERA:
            assert self.realObservs.seg.shape[0] == self.numCams
            assert finalMesh.shape[0] == 1, 'finalMesh.shape[0]'
            loss = self.getLossMulticam((finalMesh[0], self.realObservs.seg, segColor, self.pcl2DAllViews))
        elif self.optMode == optModeEnum.MUTLICAMERA_MULTIFRAME:
            assert self.realObservs.seg.shape[0] == self.numCams*self.numFrames
            assert finalMesh.shape[0] == self.numFrames, 'finalMesh.shape[0]'
            segColor = np.tile(np.expand_dims(segColor.astype(np.float32), 0), [self.numFrames , 1])
            loss = tf.map_fn(self.getLossMulticam, (finalMesh, tf.reshape(self.realObservs.seg,
                                                                          [self.numFrames, self.numCams,
                                                                           self.realObservs.seg.shape[1],
                                                                           self.realObservs.seg.shape[2],
                                                                           self.realObservs.seg.shape[3]]),
                                                    segColor, self.pcl2DAllViews),
                             parallel_iterations=int(self.numFrames), dtype=tf.float32)
            loss = tf.reduce_mean(loss)
        else:
            raise NotImplementedError

        return loss






