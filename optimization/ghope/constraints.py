
import tensorflow as tf
import numpy as np
from ghope.common import *
from HOdatasets.mypaths import *
from ghope.utils import *
from manoTF.batch_lbs import batch_rodrigues



class Constraints():
    def __init__(self):

        self.thetaLimits()

        # load SMPL data for future
        self.smpl_data = loadPickleData(MANO_MODEL_PATH)


    def thetaLimits(self):
        MINBOUND = -5.
        MAXBOUND = 5.
        self.validThetaIDs = np.array([0, 1, 2, 3, 4, 5, 6, 8, 11, 13, 14, 15, 17, 20, 21, 22, 23, 25, 26, 29,
                                       30, 31, 32, 33, 35, 38, 39, 40, 41, 42, 44, 46, 47], dtype=np.int32)
        self.validThetaIDs = np.array([0, 1, 2, 3, 4, 5, 6, 8, 11, 13, 14, 15, 17, 20, 21, 22, 23, 25, 26, 29,
                                       30, 31, 32, 33, 35, 38, 39, 40, 41, 42, 44, 46, 47], dtype=np.int32)
        # self.validThetaIDs = np.arange(0, 48, dtype=np.int32)
        invalidThetaIDsList = []
        for i in range(48):
            if i not in self.validThetaIDs:
                invalidThetaIDsList.append(i)
        # self.invalidThetaIDs = np.array([7, 9, 10, 12, 16, 18, 19, 24, #25, 26,
        #                                  27, 28, 34, 36, 37, 43, 45], dtype=np.int32)
        self.invalidThetaIDs = np.array(invalidThetaIDsList)
        self.minThetaVals = np.array([MINBOUND, MINBOUND, MINBOUND, # global rot
                            0, -0.15, 0.1, -0.3, MINBOUND, -0.0, MINBOUND, MINBOUND, 0, # index
                            MINBOUND, -0.15, 0.1, -0.5, MINBOUND, -0.0, MINBOUND, MINBOUND, 0, # middle
                            -1.5, -0.15, -0.1, MINBOUND, -0.5, -0.0, MINBOUND, MINBOUND, 0, # pinky
                            -0.5, -0.25, 0.1, -0.4, MINBOUND, -0.0, MINBOUND, MINBOUND, 0, # ring
                            0.0, -0.83, -0.0, -0.15, MINBOUND, 0, MINBOUND, -0.5, -1.57, ]) # thumb

        self.maxThetaVals = np.array([MAXBOUND, MAXBOUND, MAXBOUND, #global
                            0.45, 0.2, 1.8, 0.2, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25, # index
                            MAXBOUND, 0.15, 2.0, -0.2, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25, # middle
                            -0.2, 0.6, 1.6, MAXBOUND, 0.6, 2.0, MAXBOUND, MAXBOUND, 1.25, # pinky
                            -0.4, 0.10, 1.8, -0.2, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25, # ring
                            2.0, 0.66, 0.5, 1.6, MAXBOUND, 0.5, MAXBOUND, 0, 1.08]) # thumb


        self.fullThetaMat = np.zeros((48, len(self.validThetaIDs)), dtype=np.float32)  #48x25
        for i in range(len(self.validThetaIDs)):
            self.fullThetaMat[self.validThetaIDs[i], i] = 1.0

    def getFullThetafromValidTheta(self, validTheta): # validTheta is Nx25 or
        '''
        Converts valid theta vector to full theta vector by inserting zeros in invalid locations
        :param validTheta: shape is either (N,25) or (25,)
        :return: shape is either (N,48) or (48,) depending on input
        '''
        isNoBatch = False
        if len(validTheta.shape) == 1:
            validTheta = tf.expand_dims(validTheta, 0)
            isNoBatch = True
        assert len(validTheta.shape) == 2, 'validTheta should be 2 dimensional or 1 dimensional'

        thetaMat = np.expand_dims(self.fullThetaMat, 0) #1x48x25
        thetaMat = tf.tile(thetaMat, [tf.shape(validTheta)[0], 1, 1]) # Nx48x25
        fullTheta = tf.matmul(thetaMat, tf.expand_dims(validTheta, 2))[:,:,0] # Nx48x1

        if isNoBatch:
            fullTheta = fullTheta[0]

        return fullTheta

    def getHandThetaConstraints(self, theta, isValidTheta=False):
        '''
        get constraints on the joint angles when input is theta vector itself (first 3 elems are global rot)
        :param theta: Nx48 tensor if isValidTheta is False and Nx25 if isValidTheta is True
        :param isValidTheta:
        :return:
        '''
        if len((theta.shape)) == 1:
            theta = tf.expand_dims(theta, 0)
        assert len((theta.shape)) == 2, 'Theta should be 2 dimensional'

        if not isValidTheta:
            assert (theta.shape)[-1] == 48
            validTheta = tf.gather(theta, self.validThetaIDs,axis=1)
        else:
            assert (theta.shape)[-1] == len(self.validThetaIDs)
            validTheta = theta

        phyConst = tf.square(tf.maximum(self.minThetaVals[self.validThetaIDs] - validTheta, 0)) + \
                   tf.square(tf.maximum(validTheta - self.maxThetaVals[self.validThetaIDs], 0))
        phyConst = tf.reduce_sum(phyConst)

        return phyConst, validTheta

    def getHandJointConstraints(self, joints, isValidTheta=False):
        '''
        get constraints on the joint angles when input is joints vector (theta[3:])
        :param theta: Nx45 tensor if isValidTheta is False and Nx22 if isValidTheta is True
        :param isValidTheta:
        :return:
        '''
        if len((joints.shape)) == 1:
            joints = tf.expand_dims(joints, 0)
        assert len((joints.shape)) == 2, 'Theta should be 2 dimensional'

        if not isValidTheta:
            assert (joints.shape)[-1] == 45
            validJoints = tf.gather(joints, self.validThetaIDs[3:]-3, axis=1)
        else:
            assert (joints.shape)[-1] == len(self.validThetaIDs[3:])
            validJoints = joints

        phyConst = tf.square(tf.maximum(self.minThetaVals[self.validThetaIDs[3:]] - validJoints, 0)) + \
                   tf.square(tf.maximum(validJoints - self.maxThetaVals[self.validThetaIDs[3:]], 0))
        phyConst = tf.reduce_sum(phyConst)

        return phyConst, validJoints

    def getHandJointConstraintsCh(self, theta, isValidTheta=False):
        '''
        chumpy implementation of getHandJointConstraints
        :param theta: Nx48 tensor if isValidTheta is False and Nx25 if isValidTheta is True
        :param isValidTheta:
        :return:
        '''
        import chumpy as ch

        if not isValidTheta:
            assert (theta.shape)[-1] == 45
            validTheta = theta[self.validThetaIDs[3:] - 3]
        else:
            assert (theta.shape)[-1] == len(self.validThetaIDs[3:])
            validTheta = theta

        phyConstMax = (ch.maximum(self.minThetaVals[self.validThetaIDs[3:]] - validTheta, 0))
        phyConstMin = (ch.maximum(validTheta - self.maxThetaVals[self.validThetaIDs[3:]], 0))

        return phyConstMin, phyConstMax

    def getHandJointRestPos(self, handBeta):
        assert handBeta.shape == (10,), 'handBeta should be of shape (10,)'

        shapeDirs = np.reshape(self.smpl_data['shapedirs'].r.astype(np.float32), [-1, 10])
        v_shaped = tf.matmul(shapeDirs, tf.expand_dims(handBeta,1))[:,0]
        v_shaped = tf.reshape(v_shaped, [self.smpl_data['shapedirs'].r.shape[0], 3]) + self.smpl_data['v_template'].astype(np.float32)
        J_tmpx = tf.matmul(self.smpl_data['J_regressor'].todense().astype(np.float32), v_shaped[:, 0:1])
        J_tmpy = tf.matmul(self.smpl_data['J_regressor'].todense().astype(np.float32), v_shaped[:, 1:2])
        J_tmpz = tf.matmul(self.smpl_data['J_regressor'].todense().astype(np.float32), v_shaped[:, 2:3])
        JRest = tf.concat([J_tmpx, J_tmpy, J_tmpz], axis=1)
        assert JRest.shape == (16,3)

        return JRest

    def getHandObjRelPoseConstraint(self, handRot, handTrans, handBeta, objRot, objTrans):
        '''
        Computes rel. pose btw hand and object in each frame, and adds a loss term that makes this rel. pose to be same for all frames.
        R_Rel1.dot(R_Rel2.T) == I, is the constraint used for rel. rotation btw two frames
        :param handRot:
        :param handTrans:
        :param handBeta:
        :param objRot:
        :param objTrans:
        :return:
        '''
        assert len(handRot.shape) == 2
        assert len(handTrans.shape) == 2
        assert len(objRot.shape) == 2
        assert len(objTrans.shape) == 2
        assert len(handBeta.shape) == 2

        JRest = self.getHandJointRestPos(handBeta[0])[0]
        JRest = tf.expand_dims(JRest, 0)
        JRest = tf.expand_dims(JRest, 0)
        JRest = tf.tile(JRest, [handRot.shape[0], 1, 1]) #Nx3

        handRotMat = batch_rodrigues(handRot)
        objRotMat = batch_rodrigues(objRot)

        relRotMat = tf.matmul(tf.transpose(objRotMat,[0,2,1]), handRotMat)
        relTrans = tf.matmul(JRest + tf.expand_dims(handTrans, 1) - tf.expand_dims(objTrans, 1), objRotMat) - JRest
        relTrans = relTrans[:,0,:]
        # relRotMat = tf.Print(relRotMat, [relRotMat[0], relRotMat[1], 'rotmat'])
        # relTrans = tf.Print(relTrans, [relTrans[0], relTrans[1], 'trans'])
        # relTrans = tf.Print(relTrans, [JRest, 'JRest'])

        idenMat = np.expand_dims(np.eye(3, dtype=np.float32), 0)
        idenMat = np.tile(idenMat, [handRot.shape[0], 1, 1])

        rotLoss = tf.reduce_mean(tf.squared_difference(tf.matmul(relRotMat[:-1], tf.transpose(relRotMat[1:], [0,2,1])), idenMat[:-1])) \
                  + tf.reduce_mean(tf.squared_difference(tf.matmul(relRotMat[0], tf.transpose(relRotMat[-1], [1,0])), idenMat[0]))
        transLoss = tf.reduce_mean(tf.squared_difference(relTrans[:-1], relTrans[1:])) \
                     + tf.reduce_mean(tf.squared_difference(relTrans[-1], relTrans[0]))

        totalLoss = transLoss + 1e2*rotLoss

        return totalLoss

    def getHandObjRelDeltaPoseConstraint(self, handRot, handTrans, weight=None):
        rotLoss = tf.reduce_sum(tf.square(handRot))
        transLoss = tf.reduce_sum(tf.square(handTrans))

        if weight is None:
            totalLoss = rotLoss
        else:
            totalLoss = transLoss + weight * rotLoss

        return totalLoss

    def getTemporalConstraint(self, params, type='ZERO_VEL'):
        assert len(params.shape) == 2

        if type == 'ZERO_VEL':
            loss = tf.reduce_sum(tf.square(params[:-1,:] - params[1:,:]))
        elif type == 'ZERO_ACCL':
            vel = params[1:,:] - params[:-1,:]
            loss = tf.reduce_sum(tf.square(vel[:-1, :] - vel[1:, :]))
        else:
            raise NotImplementedError

        return loss


class ContactLoss():
    def __init__(self, objMesh, handMesh, contactPoints):
        assert hasattr(objMesh, 'vn'), 'need normal vectors in object mesh'
        assert len(objMesh.v.shape) == 3
        assert objMesh.v.shape[2] == 4
        assert len(handMesh.v.shape) == 3
        assert handMesh.v.shape[2] == 4
        assert len(contactPoints.shape) == 3
        assert contactPoints.shape[2] == 3
        assert contactPoints.shape[0] == objMesh.v.shape[0]
        assert handMesh.v.shape[0] == objMesh.v.shape[0]

        self.objMesh = objMesh
        self.handMesh = handMesh
        self.contPts = contactPoints

    def getNearestObjPts(self):
        objMeshNew = tf.expand_dims(self.objMesh.v, 1)[:,:,:,:3]  # Nx1xMx3
        contPtsNew = tf.expand_dims(self.contPts, 2)  # NxCx1x3

        nearObjPtsArg = tf.argmin(tf.norm(contPtsNew-objMeshNew, axis=3, keep_dims=False), axis=2)  #NxC

        numViews = self.objMesh.v.shape[0]
        numCps = nearObjPtsArg.shape[-1]
        inds1D = np.reshape(np.tile(np.expand_dims(np.arange(0, int(numViews)),1), [1, numCps]), [-1])
        nearObjPtsIndReshaped = tf.stack([inds1D, tf.reshape(nearObjPtsArg, [-1])], axis=1)
        self.nearObjPtsInd = tf.reshape(nearObjPtsIndReshaped, [numViews, numCps, 2])

        self.nearObjPts = tf.stop_gradient(tf.gather_nd(self.objMesh.v, nearObjPtsIndReshaped))
        self.nearObjPts = tf.reshape(self.nearObjPts, [numViews, numCps, self.objMesh.v.shape[-1]])

        self.nearObjPtsNormals = tf.stop_gradient(tf.gather_nd(self.objMesh.vn, nearObjPtsIndReshaped))
        self.nearObjPtsNormals = tf.reshape(self.nearObjPtsNormals, [numViews, numCps, self.objMesh.vn.shape[-1]])

        assert self.nearObjPts[:,:,:3].shape == self.contPts.shape

        return self.nearObjPtsInd, self.nearObjPts

    def getRepulsionLoss(self, lossType=contactLossTypeEnum.NORMAL_DOT_PRODUCT):
        if lossType == contactLossTypeEnum.NORMAL_DOT_PRODUCT:
            self.getNearestObjPts()
            dotProd = tf.reduce_sum((self.contPts - self.nearObjPts[:,:,:3])*self.nearObjPtsNormals[:,:,:3], axis=2) #NxC
            # dotProd = tf.Print(dotProd, [dotProd[0][:4]], summarize=4)
            # dotProd = tf.Print(dotProd, [self.objMesh.vn[0,293,:]], summarize=4)
            # dotProd = tf.Print(dotProd, [self.contPts[0, 0], self.nearObjPts[0,0,:3], self.nearObjPtsNormals[0,0,:3], self.nearObjPtsInd[0,0,:]], summarize=11)
            # loss = tf.reduce_sum(tf.nn.relu(-dotProd + 0.6e-2))
            loss = tf.reduce_sum(tf.exp(10*tf.nn.relu(-dotProd + 0.5e-2)) - 1.)
            return loss
        else:
            raise NotImplementedError

if __name__ == '__main__':
    tf.enable_eager_execution()
    class objMesh():
        pass
    om = objMesh()
    numPts = 20
    numFrames = 5
    v = np.ones((numFrames, numPts, 4))
    v[:,:,:3] = np.random.uniform(-10, 10, (numFrames, numPts, 3))
    om.v = v
    om.vn = v

    randInd = np.random.randint(0, numPts, (numFrames))
    contPts = np.tile(np.array([[[0., 0., 0.]]]), [numFrames, 1, 1])
    for i in range(numFrames):
        contPts[i, 0] = om.v[i, randInd[i], :3]
    print(randInd)
    CL = ContactLoss(om, om, contPts)

    mag = np.sqrt(np.sum(np.square(v[0,:,:3]),1))

    a, b = CL.getNearestObjPts()




