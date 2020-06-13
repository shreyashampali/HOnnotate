from ghope.common import *
from manoTF.batch_mano import MANO
from object.batch_object import objectModel
from HOdatasets.mypaths import *
from ghope.constraints import Constraints
import random
from ghope.utils import *
from manoTF.batch_lbs import batch_rodrigues


class Scene():
    def __init__(self, optMode, cameraCnt=1, frameCnt=1):
        '''

        :param optMode: spectifies the type of optimization. MULTIFRAME or MULTICAMERA or MUTLICAMERA_MULTIFRAME and so on
        :param cameraCnt: number of cameras in the scene
        :param frameCnt: number of frames to render
        '''

        self.optMode = optMode
        self.numCameras = cameraCnt
        self.numFrames = frameCnt

        self.handModel = MANO(MANO_MODEL_PATH)

        self.optVars = []
        self.constrs = Constraints()

        self.objModel = objectModel()

        self.itemPropsDict = {}
        self.camPropsList = []

    def genUniqueID(self, prefix):
        num = random.randint(1,51)
        id = prefix + str(num)
        while(id in self.itemPropsDict.keys()):
            num = random.randint(1, 51)
            id = prefix + str(num)

        return id

    def getCameraMat(self):
        '''
        get the full cam projection matrix including camera pose. Updates self.camProjMat
        :return:
        '''
        assert len(self.camPropsList) == self.numCameras, \
            'Num of cameras added not equal to %d'%self.numCameras

        if self.optMode == optModeEnum.MULTICAMERA:
            self.camProjMat = np.zeros((self.numCameras, 4, 4), dtype=np.float32)
            self.camFrameSize = np.array(self.camPropsList[0].frameSize)
            camProjMatList = []
            for i, camProp in enumerate(self.camPropsList):
                camProjMatList.append(creatCamMat(f = camProp.f, c = camProp.c,
                                                 near = camProp.near, far = camProp.far,
                                                 imShape = camProp.frameSize, pose = camProp.pose))
                # cam frame size should be same for all camera, because dirt cant support different frame sizes
                assert self.camFrameSize[0] == np.array(camProp.frameSize)[0], 'Camera frame sizes should be same for all cameras'
                assert self.camFrameSize[1] == np.array(camProp.frameSize)[1], 'Camera frame sizes should be same for all cameras'
            self.camProjMat = tf.stack(camProjMatList, axis=0)
        elif self.optMode == optModeEnum.MUTLICAMERA_MULTIFRAME:
            self.camProjMat = np.zeros((self.numCameras, 4, 4), dtype=np.float32)
            self.camFrameSize = np.array(self.camPropsList[0].frameSize)
            camProjMatList = []
            for _ in range(self.numFrames):
                for i, camProp in enumerate(self.camPropsList):
                    camProjMatList.append(creatCamMat(f=camProp.f, c=camProp.c,
                                                      near=camProp.near, far=camProp.far,
                                                      imShape=camProp.frameSize, pose=camProp.pose))
                    # cam frame size should be same for all camera, because dirt cant support different frame sizes
                    assert self.camFrameSize[0] == np.array(camProp.frameSize)[
                        0], 'Camera frame sizes should be same for all cameras'
                    assert self.camFrameSize[1] == np.array(camProp.frameSize)[
                        1], 'Camera frame sizes should be same for all cameras'
            self.camProjMat = tf.stack(camProjMatList, axis=0)
        else:
            projMat = creatCamMat(f=self.camPropsList[0].f, c=self.camPropsList[0].c,
                                 near=self.camPropsList[0].near, far=self.camPropsList[0].far,
                                 imShape=self.camPropsList[0].frameSize, pose=self.camPropsList[0].pose)
            self.camProjMat = tf.tile(tf.expand_dims(projMat, 0), [self.numFrames, 1, 1])
            self.camFrameSize = np.array(self.camPropsList[0].frameSize)


    def addCamera(self, f, c, near, far, frameSize, pose=np.eye(4, dtype=np.float32)):
        '''
        Set parameters of A camera
        :param f: ndarray, fx and fy
        :param c: ndarray, cx and cy
        :param near: ndarray, near value
        :param far: ndarray, far value
        :param frameSize: ndarray, w and h
        :return:
        '''
        # save the itemProps for this object, use it later for rendering
        camID = self.genUniqueID('cam')
        self.camPropsList.append(camProps(camID, f, c, near, far, frameSize, pose))


    def addObject(self, objMesh, paramInits, segColor=np.array([0., 0., 1.]), useInputVars=False):
        '''
        Adds the object to the scene. Creates tf variables for obj pose

        Variable name format : <objID>_rot_<frameCnt>, <objID>_trans_<frameCnt>
        :param objMesh: input object mesh
        :param paramInits: if useInputVars is true then this list of items belonging to objVars class else it is a list of items belonging to objParams class
        This input argument is used either initialized the new variables created in the scene are use the variables which are provided as input
        :param segColor: segcolor for this object
        :param useInputVars: create internal vars or reuse vars which are provided in paramInits
        :return: object id for this item
        '''


        if not isinstance(paramInits, list):
            paramInits = [paramInits]

        with tf.variable_scope('obj', reuse=tf.AUTO_REUSE):
            # save the itemProps for this object, use it later for rendering
            objID = self.genUniqueID('obj')
            self.itemPropsDict[objID] = itemProps(objID, segColor, itemTypeEnum.OBJECT, objMesh)

            if self.optMode != optModeEnum.MULTICAMERA:# and self.optMode != optModeEnum.MUTLICAMERA_MULTIFRAME:
                assert len(paramInits) == self.numFrames, 'num of init values not equal to num frames'
                # create the tf variables with appropriate names for each frame
                for cnt, elem in enumerate(paramInits):
                    if useInputVars:
                        assert isinstance(elem, objVars), 'Input vars should belong to handVars class...'
                        rot = elem.rot
                    else:
                        assert isinstance(elem, objParams), 'Init values should belong to objParams class...'
                        rot = tf.get_variable(name=objID + '_rot_' + str(cnt), initializer=elem.rot, dtype=tf.float32)
                    self.itemPropsDict[objID].addVarToList(rot)

                    if useInputVars:
                        trans = elem.trans
                    else:
                        trans = tf.get_variable(name=objID + '_trans_' + str(cnt), initializer=elem.trans, dtype=tf.float32)
                    self.itemPropsDict[objID].addVarToList(trans)

                    self.optVars.append(rot)
                    self.optVars.append(trans)
            elif self.optMode == optModeEnum.MULTICAMERA:
                assert len(paramInits) == 1, 'num of init values not equal to num cameras'
                if useInputVars:
                    assert isinstance(paramInits[0], objVars), 'Input vars should belong to handVars class...'
                    rot = paramInits[0].rot
                else:
                    assert isinstance(paramInits[0], objParams), 'Init values should belong to objParams class...'
                    rot = tf.get_variable(name=objID + '_rot_global', initializer=paramInits[0].rot, dtype=tf.float32)
                self.itemPropsDict[objID].addVarToList(rot)

                if useInputVars:
                    trans = paramInits[0].trans
                else:
                    trans = tf.get_variable(name=objID + '_trans_global', initializer=paramInits[0].trans, dtype=tf.float32)
                self.itemPropsDict[objID].addVarToList(trans)

                self.optVars.append(rot)
                self.optVars.append(trans)


            return objID



    def addHand(self, paramInits, segColor=np.array([1., 0., 0.]), useInputVars=False, baseItemID=None):
        '''
        Adds hand to the scene. Creates opt vars and inits them

        per frame Variable name format : <handID>_rot_<frameCnt>, <handID>_trans_<frameCnt>, <handID>_joint_<frameCnt>, <handID>_beta_<frameCnt>

        global variable name format : <handID>_joint_global, <handID>_beta_global

        :param paramInits: if useInputVars is true then this list of items belonging to handVars class else it is a list of items belonging to objParams class
        This input argument is used either initialized the new variables created in the scene are use the variables which are provided as input
        :param segColor: segcolor for this object
        :param useInputVars: create internal vars or reuse vars which are provided in paramInits
        :return: hand id for this item
        '''


        if not isinstance(paramInits, list):
            paramInits= [paramInits]

        with tf.variable_scope('hand', reuse=tf.AUTO_REUSE):
            # save the itemProps for this hand, use it later for rendering
            handID = self.genUniqueID('hand')
            self.itemPropsDict[handID] = itemProps(handID, segColor, itemTypeEnum.HAND)

            if self.optMode == optModeEnum.MULTIFRAME:
                assert len(paramInits) == self.numFrames, 'num of init values not equal to num frames'
                # different rot/trans for frames, same joint angle and beta
                for cnt, elem in enumerate(paramInits):

                    if useInputVars:
                        assert isinstance(elem, handVars), 'Input vars should belong to handVars class...'
                        rot = elem.rot
                    else:
                        assert isinstance(elem, handParams), 'Init values should belong to handParams class...'
                        rot = tf.get_variable(name=handID+'_rot_' + str(cnt), initializer=elem.theta[:3], dtype=tf.float32)
                    self.itemPropsDict[handID].addVarToList(rot)
                    self.optVars.append(rot)

                    if useInputVars:
                        trans = elem.trans
                    else:
                        trans = tf.get_variable(name=handID+'_trans_' + str(cnt), initializer=elem.trans, dtype=tf.float32)
                    self.itemPropsDict[handID].addVarToList(trans)
                    self.optVars.append(trans)

                # global joint angles var
                if useInputVars:
                    jointAngs = elem.jointAngs #TODO:hampali:this needs a fix
                else:
                    thetaValidInit = paramInits[0].theta[self.constrs.validThetaIDs]
                    jointAngs = tf.get_variable(name=handID+'_joint_global', initializer=thetaValidInit[3:], dtype=tf.float32)
                self.itemPropsDict[handID].addVarToList(jointAngs)
                self.optVars.append(jointAngs)

                # global beta var
                if useInputVars:
                    beta = elem.beta #TODO:hampali:this needs a fix
                else:
                    beta = tf.get_variable(name=handID+'_beta_global', initializer=paramInits[0].beta, dtype=tf.float32)
                self.itemPropsDict[handID].addVarToList(beta)
                self.optVars.append(beta)

            elif self.optMode == optModeEnum.MULTIFRAME_RIGID_HO_POSE:
                assert len(paramInits) == self.numFrames or len(paramInits) == 1, 'num of init values should be 1 or equal to num frames'

                # check if the item, relative to which the pose is calculates is available
                assert baseItemID in self.itemPropsDict.keys(), 'base item is not in the scene yet. First add the base item and the item whose pose is relative'
                assert self.itemPropsDict[baseItemID].itemType == itemTypeEnum.OBJECT, 'base item should be object.'
                self.itemPropsDict[handID].setBaseItemID(baseItemID)

                # different rot/trans delta for frames
                for cnt, elem in enumerate(paramInits):

                    if useInputVars:
                        assert isinstance(elem, handVars), 'Input vars should belong to handVars class...'
                        rot = elem.rotDelta #TODO:hampali:this needs a fix
                    else:
                        assert isinstance(elem, handParams), 'Init values should belong to handParams class...'
                        rot = tf.get_variable(name=handID+'_rot_delta_' + str(cnt), initializer=np.zeros((3,), dtype=np.float32), dtype=tf.float32)
                    self.itemPropsDict[handID].addVarToList(rot)
                    self.optVars.append(rot)

                    if useInputVars:
                        trans = elem.transDelta #TODO:hampali:this needs a fix
                    else:
                        trans = tf.get_variable(name=handID+'_trans_delta_' + str(cnt), initializer=np.zeros((3,), dtype=np.float32), dtype=tf.float32)
                    self.itemPropsDict[handID].addVarToList(trans)
                    self.optVars.append(trans)

                # global hand rot and trans, note this is relative to object
                if useInputVars:
                    assert isinstance(paramInits[0], handVars), 'Input vars should belong to handVars class...'
                    rot = paramInits[0].rot
                    trans = paramInits[0].trans
                else:
                    assert isinstance(paramInits[0], handParams), 'Init values should belong to handParams class...'
                    rot = tf.get_variable(name=handID + '_rot_global', initializer=paramInits[0].theta[:3],
                                          dtype=tf.float32)
                    trans = tf.get_variable(name=handID + '_trans_global', initializer=paramInits[0].trans,
                                            dtype=tf.float32)
                self.itemPropsDict[handID].addVarToList(rot)
                self.optVars.append(rot)

                self.itemPropsDict[handID].addVarToList(trans)
                self.optVars.append(trans)

                # global joint angles var
                if useInputVars:
                    jointAngs = paramInits[0].jointAngs
                else:
                    thetaValidInit = paramInits[0].theta[self.constrs.validThetaIDs]
                    jointAngs = tf.get_variable(name=handID+'_joint_global', initializer=thetaValidInit[3:], dtype=tf.float32)
                self.itemPropsDict[handID].addVarToList(jointAngs)
                self.optVars.append(jointAngs)

                # global beta var
                if useInputVars:
                    beta = paramInits[0].beta
                else:
                    beta = tf.get_variable(name=handID+'_beta_global', initializer=paramInits[0].beta, dtype=tf.float32)
                self.itemPropsDict[handID].addVarToList(beta)
                self.optVars.append(beta)

            elif self.optMode == optModeEnum.MULTIFRAME_RIGID_HO_POSE_JOINT:
                assert len(paramInits) == self.numFrames or len(paramInits) == 1, 'num of init values should be 1 or equal to num frames'

                # check if the item, relative to which the pose is calculates is available
                assert baseItemID in self.itemPropsDict.keys(), 'base item is not in the scene yet. First add the base item and the item whose pose is relative'
                assert self.itemPropsDict[baseItemID].itemType == itemTypeEnum.OBJECT, 'base item should be object.'
                self.itemPropsDict[handID].setBaseItemID(baseItemID)

                # different rot/trans delta for frames
                for cnt, elem in enumerate(paramInits):

                    if useInputVars:
                        assert isinstance(elem, handVars), 'Input vars should belong to handVars class...'
                        rot = elem.rotDelta #TODO:hampali:this needs a fix
                    else:
                        assert isinstance(elem, handParams), 'Init values should belong to handParams class...'
                        rot = tf.get_variable(name=handID+'_rot_delta_' + str(cnt), initializer=np.zeros((3,), dtype=np.float32), dtype=tf.float32)
                    self.itemPropsDict[handID].addVarToList(rot)
                    self.optVars.append(rot)

                    if useInputVars:
                        trans = elem.transDelta #TODO:hampali:this needs a fix
                    else:
                        trans = tf.get_variable(name=handID+'_trans_delta_' + str(cnt), initializer=np.zeros((3,), dtype=np.float32), dtype=tf.float32)
                    self.itemPropsDict[handID].addVarToList(trans)
                    self.optVars.append(trans)

                    if useInputVars:
                        jointAngs = elem.jointAngs
                    else:
                        thetaValidInit = elem.theta[self.constrs.validThetaIDs]
                        jointAngs = tf.get_variable(name=handID+'_joint_' + str(cnt), initializer=thetaValidInit[3:], dtype=tf.float32)
                    self.itemPropsDict[handID].addVarToList(jointAngs)
                    self.optVars.append(jointAngs)

                # global hand rot and trans, note this is relative to object
                if useInputVars:
                    assert isinstance(paramInits[0], handVars), 'Input vars should belong to handVars class...'
                    rot = paramInits[0].rot
                    trans = paramInits[0].trans
                else:
                    assert isinstance(paramInits[0], handParams), 'Init values should belong to handParams class...'
                    rot = tf.get_variable(name=handID + '_rot_global', initializer=paramInits[0].theta[:3],
                                          dtype=tf.float32)
                    trans = tf.get_variable(name=handID + '_trans_global', initializer=paramInits[0].trans,
                                            dtype=tf.float32)
                self.itemPropsDict[handID].addVarToList(rot)
                self.optVars.append(rot)

                self.itemPropsDict[handID].addVarToList(trans)
                self.optVars.append(trans)

                # global beta var
                if useInputVars:
                    beta = paramInits[0].beta
                else:
                    beta = tf.get_variable(name=handID+'_beta_global', initializer=paramInits[0].beta, dtype=tf.float32)
                self.itemPropsDict[handID].addVarToList(beta)
                self.optVars.append(beta)

            elif self.optMode == optModeEnum.MULTIFRAME_JOINT or self.optMode == optModeEnum.MUTLICAMERA_MULTIFRAME:
                assert len(paramInits) == self.numFrames, 'num of init values not equal to num frames'
                # different rot/trans for frames, same joint angle and beta
                for cnt, elem in enumerate(paramInits):

                    if useInputVars:
                        assert isinstance(elem, handVars), 'Input vars should belong to handVars class...'
                        rot = elem.rot
                    else:
                        assert isinstance(elem, handParams), 'Init values should belong to handParams class...'
                        rot = tf.get_variable(name=handID+'_rot_' + str(cnt), initializer=elem.theta[:3], dtype=tf.float32)
                    self.itemPropsDict[handID].addVarToList(rot)
                    self.optVars.append(rot)

                    if useInputVars:
                        trans = elem.trans
                    else:
                        trans = tf.get_variable(name=handID+'_trans_' + str(cnt), initializer=elem.trans, dtype=tf.float32)
                    self.itemPropsDict[handID].addVarToList(trans)
                    self.optVars.append(trans)

                    if useInputVars:
                        jointAngs = elem.jointAngs
                    else:
                        thetaValidInit = elem.theta[self.constrs.validThetaIDs]
                        jointAngs = tf.get_variable(name=handID+'_joint_' + str(cnt), initializer=thetaValidInit[3:], dtype=tf.float32)
                    self.itemPropsDict[handID].addVarToList(jointAngs)
                    self.optVars.append(jointAngs)

                # global beta var
                if useInputVars:
                    beta = elem.beta
                else:
                    beta = tf.get_variable(name=handID+'_beta_global', initializer=paramInits[0].beta, dtype=tf.float32)
                self.itemPropsDict[handID].addVarToList(beta)
                self.optVars.append(beta)

            elif self.optMode == optModeEnum.MULTICAMERA:
                assert len(paramInits) == 1, 'Only one set of init params required in the multicamera mode'

                # rot
                if useInputVars:
                    assert isinstance(paramInits[0], handVars), 'Input vars should belong to handVars class...'
                    rot = paramInits[0].rot
                else:
                    assert isinstance(paramInits[0], handParams), 'Init values should belong to handParams class...'
                    rot = tf.get_variable(name=handID+'_rot_global', initializer=paramInits[0].theta[:3], dtype=tf.float32)
                self.itemPropsDict[handID].addVarToList(rot)
                self.optVars.append(rot)

                # trans
                if useInputVars:
                    trans = paramInits[0].trans
                else:
                    trans = tf.get_variable(name=handID+'_trans_global', initializer=paramInits[0].trans, dtype=tf.float32)
                self.itemPropsDict[handID].addVarToList(trans)
                self.optVars.append(trans)

                # joint angs
                if useInputVars:
                    jointAngs = paramInits[0].jointAngs
                else:
                    thetaValidInit = paramInits[0].theta[self.constrs.validThetaIDs]
                    jointAngs = tf.get_variable(name=handID+'_joint_global', initializer=thetaValidInit[3:], dtype=tf.float32)
                self.itemPropsDict[handID].addVarToList(jointAngs)
                self.optVars.append(jointAngs)

                # beta
                if useInputVars:
                    beta = paramInits[0].beta
                else:
                    beta = tf.get_variable(name=handID+'_beta_global', initializer=paramInits[0].beta, dtype=tf.float32)
                self.itemPropsDict[handID].addVarToList(beta)
                self.optVars.append(beta)

            else:
                raise Exception('Invalid optimization Mode..')

            return handID

    def reorderHandVars(self, handID):
        '''
        1. concatenates rot with joint angles to form theta vector
        2. create batches of theta, beta and trans as required by MANO model api
        :return:
        '''
        thetaList = []
        betaList = []
        transList = []

        # if mode is MULTIFRAME_RIGID_HO_POSE, get the base item variables
        # if self.optMode == optModeEnum.MULTIFRAME_RIGID_HO_POSE:



        if self.optMode == optModeEnum.MULTIFRAME:
            with tf.variable_scope('hand', reuse=True):
                varBeta = tf.get_variable(name=handID+'_beta_global')
                varAng = tf.get_variable(name=handID+'_joint_global')
                for cnt in range(self.numFrames):
                    # theta
                    varRot = tf.get_variable(name=handID+'_rot_' + str(cnt))
                    theta = tf.concat([varRot, varAng], axis=0)
                    varTheta = self.constrs.getFullThetafromValidTheta(theta)
                    varTrans = tf.get_variable(name=handID+'_trans_' + str(cnt))

                    thetaList.append(varTheta)

                    # beta
                    betaList.append(varBeta)

                    # trans
                    transList.append(varTrans)

        elif self.optMode == optModeEnum.MULTIFRAME_RIGID_HO_POSE:
            baseItemID = self.itemPropsDict[handID].baseItemID

            with tf.variable_scope('hand', reuse=True):
                varBeta = tf.get_variable(name=handID+'_beta_global')
                varAng = tf.get_variable(name=handID+'_joint_global')
                varRotRel = tf.get_variable(name=handID + '_rot_global')
                varTransRel = tf.get_variable(name=handID + '_trans_global')

            J0 = self.handModel.getRestPoseJointLocs(tf.expand_dims(varBeta,0))[0,0:1,:]
            J0 = tf.transpose(J0)#3x1 vector
            for cnt in range(self.numFrames):
                # get base item rot and trans
                with tf.variable_scope('obj', reuse=True):
                    varRotBase = tf.get_variable(name=baseItemID + '_rot_' + str(cnt))
                    varTransBase = tf.get_variable(name=baseItemID + '_trans_' + str(cnt))
                    varTransBase = tf.transpose(tf.expand_dims(varTransBase, 0))# 3x1

                with tf.variable_scope('hand', reuse=True):
                    # theta
                    varRotDelta = tf.get_variable(name=handID+'_rot_delta_' + str(cnt))
                    varTransDelta = tf.get_variable(name=handID + '_trans_delta_' + str(cnt))
                    rotDeltaMat = batch_rodrigues(tf.expand_dims(varRotDelta,0))
                    idenMats = np.tile(np.expand_dims(np.eye(3),0),[15, 1, 1])
                    rotDeltaMat = tf.concat([rotDeltaMat, idenMats], axis=0)
                    assert rotDeltaMat.shape == (16,3,3)

                    theta = tf.concat([varRotRel, varAng], axis=0)
                    varTheta = self.constrs.getFullThetafromValidTheta(theta)
                    varTheta = tf.reshape(varTheta, [16,3])
                    fullRotMat = batch_rodrigues(varTheta) #16x3x3

                    # delta rot matrix multiplication
                    finalRelRotMat = tf.matmul(rotDeltaMat, fullRotMat)

                    # get the final (ABS) rot mat after base Item rotation
                    varRotBaseMat = batch_rodrigues(tf.expand_dims(varRotBase, 0))
                    varRotBaseMat = tf.concat([varRotBaseMat, idenMats], axis=0)
                    finalRotMat = tf.matmul(varRotBaseMat, finalRelRotMat)

                    thetaList.append(finalRotMat)

                    # beta
                    betaList.append(varBeta)

                    # trans with delta trans addition
                    finalRelTrans = varTransDelta + varTransRel
                    finalRelTrans = tf.transpose(tf.expand_dims(finalRelTrans, 0))  # 3x1

                    # get the final (ABS) trans after base Item translation (refer to code in absHandPoseFromRel())
                    TAbs = tf.matmul(varRotBaseMat[0], J0 + finalRelTrans) + varTransBase - J0
                    transList.append(TAbs[:,0])

        elif self.optMode == optModeEnum.MULTIFRAME_RIGID_HO_POSE_JOINT:
            baseItemID = self.itemPropsDict[handID].baseItemID

            with tf.variable_scope('hand', reuse=True):
                varBeta = tf.get_variable(name=handID+'_beta_global')
                varRotRel = tf.get_variable(name=handID + '_rot_global')
                varTransRel = tf.get_variable(name=handID + '_trans_global')

            J0 = self.handModel.getRestPoseJointLocs(tf.expand_dims(varBeta,0))[0,0:1,:]
            J0 = tf.transpose(J0)#3x1 vector
            for cnt in range(self.numFrames):
                # get base item rot and trans
                with tf.variable_scope('obj', reuse=True):
                    varRotBase = tf.get_variable(name=baseItemID + '_rot_' + str(cnt))
                    varTransBase = tf.get_variable(name=baseItemID + '_trans_' + str(cnt))
                    varTransBase = tf.transpose(tf.expand_dims(varTransBase, 0))# 3x1

                with tf.variable_scope('hand', reuse=True):
                    varAng = tf.get_variable(name=handID + '_joint_' + str(cnt))

                    # theta
                    varRotDelta = tf.get_variable(name=handID+'_rot_delta_' + str(cnt))
                    varTransDelta = tf.get_variable(name=handID + '_trans_delta_' + str(cnt))
                    rotDeltaMat = batch_rodrigues(tf.expand_dims(varRotDelta,0))
                    idenMats = np.tile(np.expand_dims(np.eye(3),0),[15, 1, 1])
                    rotDeltaMat = tf.concat([rotDeltaMat, idenMats], axis=0)
                    assert rotDeltaMat.shape == (16,3,3)

                    theta = tf.concat([varRotRel, varAng], axis=0)
                    varTheta = self.constrs.getFullThetafromValidTheta(theta)
                    varTheta = tf.reshape(varTheta, [16,3])
                    fullRotMat = batch_rodrigues(varTheta) #16x3x3

                    # delta rot matrix multiplication
                    finalRelRotMat = tf.matmul(rotDeltaMat, fullRotMat)

                    # get the final (ABS) rot mat after base Item rotation
                    varRotBaseMat = batch_rodrigues(tf.expand_dims(varRotBase, 0))
                    varRotBaseMat = tf.concat([varRotBaseMat, idenMats], axis=0)
                    finalRotMat = tf.matmul(varRotBaseMat, finalRelRotMat)

                    thetaList.append(finalRotMat)

                    # beta
                    betaList.append(varBeta)

                    # trans with delta trans addition
                    finalRelTrans = varTransDelta + varTransRel
                    finalRelTrans = tf.transpose(tf.expand_dims(finalRelTrans, 0))  # 3x1

                    # get the final (ABS) trans after base Item translation (refer to code in absHandPoseFromRel())
                    TAbs = tf.matmul(varRotBaseMat[0], J0 + finalRelTrans) + varTransBase - J0
                    transList.append(TAbs[:,0])

        elif self.optMode == optModeEnum.MULTIFRAME_JOINT:
            with tf.variable_scope('hand', reuse=True):
                varBeta = tf.get_variable(name=handID+'_beta_global')

                for cnt in range(self.numFrames):
                    # theta
                    varRot = tf.get_variable(name=handID+'_rot_' + str(cnt))
                    varAng = tf.get_variable(name=handID+'_joint_' + str(cnt))
                    theta = tf.concat([varRot, varAng], axis=0)
                    varTheta = self.constrs.getFullThetafromValidTheta(theta)
                    varTrans = tf.get_variable(name=handID+'_trans_' + str(cnt))

                    thetaList.append(varTheta)

                    # beta
                    betaList.append(varBeta)

                    # trans

                    transList.append(varTrans)

        elif self.optMode == optModeEnum.MUTLICAMERA_MULTIFRAME:
            with tf.variable_scope('hand', reuse=True):
                varBeta = tf.get_variable(name=handID+'_beta_global')

                for cnt in range(self.numFrames):
                    # theta
                    varRot = tf.get_variable(name=handID+'_rot_' + str(cnt))
                    varAng = tf.get_variable(name=handID+'_joint_' + str(cnt))
                    theta = tf.concat([varRot, varAng], axis=0)
                    varTheta = self.constrs.getFullThetafromValidTheta(theta)
                    varTrans = tf.get_variable(name=handID+'_trans_' + str(cnt))

                    for _ in range(self.numCameras):
                        thetaList.append(varTheta)

                    # beta
                    for _ in range(self.numCameras):
                        betaList.append(varBeta)

                    # trans

                    for _ in range(self.numCameras):
                        transList.append(varTrans)

        elif self.optMode == optModeEnum.MULTICAMERA:
            with tf.variable_scope('hand', reuse=True):
                # theta
                varAng = tf.get_variable(name=handID+'_joint_global')
                varRot = tf.get_variable(name=handID+'_rot_global')
                theta = tf.concat([varRot, varAng], axis=0)
                varTheta = self.constrs.getFullThetafromValidTheta(theta)
                for _ in range(self.numCameras):
                    thetaList.append(varTheta)

                # beta
                varBeta = tf.get_variable(name=handID+'_beta_global')
                for _ in range(self.numCameras):
                    betaList.append(varBeta)

                # trans
                varTrans = tf.get_variable(name=handID+'_trans_global')
                for _ in range(self.numCameras):
                    transList.append(varTrans)

        else:
            raise Exception('Invalid optimization Mode..')

        thetaAll = tf.stack(thetaList, axis=0)
        betaAll = tf.stack(betaList, axis=0)
        transAll = tf.stack(transList, axis=0)

        return thetaAll, betaAll, transAll

    def reorderObjVars(self, objID):
        '''
        concatenate rot and trans of all frames to form one tensor, which will be provided to object model
        :return:
        '''
        rotList = []
        transList = []

        with tf.variable_scope('obj', reuse=True):
            if self.optMode != optModeEnum.MULTICAMERA and self.optMode != optModeEnum.MUTLICAMERA_MULTIFRAME:
                for cnt in range(self.numFrames):
                    varRot = tf.get_variable(name=objID+'_rot_' + str(cnt))
                    varTrans = tf.get_variable(name=objID+'_trans_' + str(cnt))

                    # rot
                    rotList.append(varRot)

                    # trans
                    transList.append(varTrans)
            elif self.optMode == optModeEnum.MUTLICAMERA_MULTIFRAME:
                for cnt in range(self.numFrames):
                    varRot = tf.get_variable(name=objID+'_rot_' + str(cnt))
                    varTrans = tf.get_variable(name=objID+'_trans_' + str(cnt))

                    # rot
                    for _ in range(self.numCameras):
                        rotList.append(varRot)

                    # trans
                    for _ in range(self.numCameras):
                        transList.append(varTrans)
            else:
                varRot = tf.get_variable(name=objID + '_rot_global')
                varTrans = tf.get_variable(name=objID + '_trans_global')

                # rot
                for _ in range(self.numCameras):
                    rotList.append(varRot)

                # trans
                for _ in range(self.numCameras):
                    transList.append(varTrans)

            rotAll = tf.stack(rotList, axis=0)
            transAll = tf.stack(transList, axis=0)

            return rotAll, transAll


    def getFinalMesh(self):
        '''
        1. Merges the meshes of all the items in the scene
        2. Multiply by cam proj mat to get vertices in clipping space
        :itemIDList: list of itemsID which will added to the final mesh. If none all the available be added
        :return: finalMesh object
        '''

        itemsIDList = self.itemPropsDict.keys()

        # setup camera matrices with poses
        self.getCameraMat()

        # get the meshes by passing the variables through the models
        meshList = []
        for id in itemsIDList:
            prop = self.itemPropsDict[id]
            # for each item in the scene, get the mesh. No. of meshes per item = numCameras or numFrames
            if prop.itemType == itemTypeEnum.OBJECT:

                # get the tf variables in the correct order as required by the objModel class
                rotObj, transObj = self.reorderObjVars(prop.ID)

                # get the mesh
                objMesh = self.objModel(prop.mesh, rotObj, transObj, prop.segColor, name=prop.ID)

                # save the transformed mesh and pose parameters for later use
                prop.setTransformedMesh(objMesh)
                prop.setObjPoseParameters(rotObj, transObj)

                # check if no. of meshes = numCameras or numFrames
                if self.optMode != optModeEnum.MULTICAMERA and self.optMode != optModeEnum.MUTLICAMERA_MULTIFRAME:
                    assert (objMesh.v.shape)[0] == self.numFrames
                elif self.optMode == optModeEnum.MUTLICAMERA_MULTIFRAME:
                    assert (objMesh.v.shape)[0] == self.numFrames*self.numCameras
                else:
                    assert (objMesh.v.shape)[0] == self.numCameras

                # append the mesh to the list
                meshList.append(objMesh)
            elif prop.itemType == itemTypeEnum.HAND:

                # get the tf variables in the correct order as required by the objModel class
                theta, beta, transHand = self.reorderHandVars(prop.ID)

                # if providing theta in matrix format, change some settings in the model
                if len(theta.shape) == 4:
                    self.handModel.theta_in_rodrigues = False
                    self.handModel.theta_is_perfect_rotmtx = True
                else:
                    assert len(theta.shape) == 2
                # get the mesh
                handMesh = self.handModel(theta, beta, transHand, segColor=prop.segColor)

                # make the vertices 4 dimensional
                if len(handMesh.v.shape) == 3:
                    handMesh.v = tf.concat([handMesh.v, tf.ones([tf.shape(handMesh.v)[0], tf.shape(handMesh.v)[1], 1])], axis=2)

                # save the transformed mesh and pose parameters for later use
                prop.setTransformedMesh(handMesh)
                prop.setHandPoseParameters(theta, transHand, beta)
                if self.optMode == optModeEnum.MULTICAMERA:
                    # get the campose of all cameras
                    camPoseMatList = []
                    for i, camProp in enumerate(self.camPropsList):
                        camPoseMatList.append(np.linalg.inv(camProp.pose).T)
                    camPoseMatAll = np.stack(camPoseMatList, axis=0)

                    # transform the joints to the camera space
                    J_transformedHomo = tf.concat([handMesh.J_transformed,
                               np.ones((handMesh.J_transformed.shape[0], handMesh.J_transformed.shape[1], 1), dtype=np.float32)], axis=2)
                    prop.setTransformedJoints(tf.matmul(J_transformedHomo, camPoseMatAll)[:,:,:3])
                elif self.optMode == optModeEnum.MUTLICAMERA_MULTIFRAME:
                    # get the campose of all cameras
                    camPoseMatList = []
                    for _ in range(self.numFrames):
                        for i, camProp in enumerate(self.camPropsList):
                            camPoseMatList.append(np.linalg.inv(camProp.pose).T)
                    camPoseMatAll = np.stack(camPoseMatList, axis=0)

                    # transform the joints to the camera space
                    J_transformedHomo = tf.concat([handMesh.J_transformed,
                                                   np.ones((handMesh.J_transformed.shape[0],
                                                            handMesh.J_transformed.shape[1], 1), dtype=np.float32)],
                                                  axis=2)
                    prop.setTransformedJoints(tf.matmul(J_transformedHomo, camPoseMatAll)[:, :, :3])
                else:
                    prop.setTransformedJoints(handMesh.J_transformed)

                # check if no. of meshes = numCameras or numFrames
                if self.optMode != optModeEnum.MULTICAMERA:
                    if self.optMode == optModeEnum.MUTLICAMERA_MULTIFRAME:
                        assert (handMesh.v.shape)[0] == self.numFrames*self.numCameras
                    else:
                        assert (handMesh.v.shape)[0] == self.numFrames
                else:
                    assert (handMesh.v.shape)[0] == self.numCameras

                # append the mesh to the list
                meshList.append(handMesh)
            else:
                # can never come here
                raise Exception('Invalid itemType in itemPropDict')

        # meshList will have as many elements as number of items in the scene
        assert len(meshList) == len(self.itemPropsDict.keys())

        # merge all the meshes from meshList
        class finalMesh(object):
            pass

        # get final set of vertices
        finalMesh.v = tf.concat([mesh.v for mesh in meshList], axis=1)

        # get final set of faces
        finalFacesList = [meshList[0].f.astype(np.int32)]
        vertCnt = int(meshList[0].v.shape[1])
        for mesh in meshList[1:]:
            finalFacesList.append(mesh.f.astype(np.int32) + vertCnt)
            vertCnt += int(mesh.v.shape[1])
        finalMesh.f = np.concatenate(finalFacesList, axis=0)

        # get final vc
        finalMesh.vc = tf.concat([mesh.vc for mesh in meshList], axis=0)

        # get final vcSeg
        finalMesh.vcSeg = tf.concat([mesh.vcSeg for mesh in meshList], axis=0)


        finalMesh.vUnClipped = finalMesh.v
        # get vertices in clipping space for each camera
        assert (finalMesh.v.shape)[0] == (self.camProjMat.shape)[0]
        finalMesh.v = tf.matmul(finalMesh.v, self.camProjMat)
        finalMesh.frameSize = self.camFrameSize



        # make some final checks
        if self.optMode != optModeEnum.MULTICAMERA:
            assert (finalMesh.v.shape)[0] == self.numFrames*self.numCameras
        else:
            assert (finalMesh.v.shape)[0] == self.numCameras
        assert len((finalMesh.f.shape)) == 2
        assert len((finalMesh.vc.shape)) == 2
        assert len((finalMesh.vcSeg.shape)) == 2
        assert len((finalMesh.frameSize.shape)) == 1

        return finalMesh

    def getParamsByItemID(self, parTypeList, itemID):
        '''
        Gets the parameters for the input itemID
        :param parTypeList:
        :param itemID:
        :return:
        '''
        assert isinstance(parTypeList, list), 'parType should be a list'
        if len(parTypeList) == 0:
            if self.itemPropsDict[itemID].itemType == itemTypeEnum.HAND:
                parTypeList = [parTypeEnum.HAND_THETA, parTypeEnum.HAND_BETA, parTypeEnum.HAND_TRANS]
            else:
                parTypeList = [parTypeEnum.OBJ_ROT, parTypeEnum.OBJ_TRANS, parTypeEnum.OBJ_POSE_MAT]

        parList = []

        for parType in parTypeList:
            if parType==parTypeEnum.HAND_THETA:
                parList.append(self.itemPropsDict[itemID].theta)
            elif parType==parTypeEnum.HAND_TRANS:
                parList.append(self.itemPropsDict[itemID].trans)
            elif parType == parTypeEnum.HAND_BETA:
                parList.append(self.itemPropsDict[itemID].beta)
            elif parType == parTypeEnum.OBJ_TRANS:
                parList.append(self.itemPropsDict[itemID].trans)
            elif parType == parTypeEnum.OBJ_ROT:
                parList.append(self.itemPropsDict[itemID].rot)
            elif parType == parTypeEnum.OBJ_POSE_MAT:
                parList.append(self.itemPropsDict[itemID].getObjPoseMat())

        return parList


    def getVarsByItemID(self, itemID, varTypeList=[], frameList=[]):
        '''
        Fetches list of TF variables for optimization
        :param itemID: itemID whose variables need to be fetched
        :param varTypeList: List of variables corresponding to the item. Contents should belong to  varTypeEnum
        :param frameList: List of frames at whose variables need to be fetched. Not used in MULTICAMERA mode. If empty list returns vars for all frames
        :return: List of variables with names <itemID>_<varType>_<frameCnt>
        '''
        assert isinstance(varTypeList, list), 'varType should be a list'
        assert isinstance(frameList, list), 'frame numbers should be a list'
        if len(varTypeList) == 0:
            if self.itemPropsDict[itemID].itemType == itemTypeEnum.HAND:
                varTypeList = [varTypeEnum.HAND_THETA, varTypeEnum.HAND_BETA, varTypeEnum.HAND_TRANS]
            else:
                varTypeList = [varTypeEnum.OBJ_ROT, varTypeEnum.OBJ_TRANS]

        if len(frameList) == 0:
            if self.optMode != optModeEnum.MULTICAMERA:
                frameList = [i for i in range(self.numFrames)]
            else:
                frameList = [1] # doesnt matter only global variables, no framewise vars

        varList = []

        betaAppendDone = False
        jointAppendDone = False
        rotAppendDone = False
        transAppendDone = False
        for frame in frameList:
            for varType in varTypeList:
                if self.optMode == optModeEnum.MULTIFRAME:
                    if varType == varTypeEnum.HAND_JOINT:
                        if not jointAppendDone:
                            with tf.variable_scope('hand', reuse=True):
                                varAng = tf.get_variable(name=itemID + '_joint_global')
                                varList.append(varAng)
                                jointAppendDone = True
                    elif varType == varTypeEnum.HAND_ROT:
                        with tf.variable_scope('hand', reuse=True):
                            varRot = tf.get_variable(name=itemID + '_rot_' + str(frame))
                            varList.append(varRot)
                    elif varType == varTypeEnum.HAND_TRANS:
                        with tf.variable_scope('hand', reuse=True):
                            varTrans = tf.get_variable(name=itemID + '_trans_' + str(frame))
                            varList.append(varTrans)
                    elif varType == varTypeEnum.HAND_BETA:
                        if not betaAppendDone:
                            with tf.variable_scope('hand', reuse=True):
                                varBeta = tf.get_variable(name=itemID + '_beta_global')
                                varList.append(varBeta)
                                betaAppendDone = True
                    elif varType == varTypeEnum.OBJ_TRANS:
                        with tf.variable_scope('obj', reuse=True):
                            varTrans = tf.get_variable(name=itemID + '_trans_' + str(frame))
                            varList.append(varTrans)
                    elif varType == varTypeEnum.OBJ_ROT:
                        with tf.variable_scope('obj', reuse=True):
                            varRot = tf.get_variable(name=itemID + '_rot_' + str(frame))
                            varList.append(varRot)

                elif self.optMode == optModeEnum.MULTIFRAME_RIGID_HO_POSE:
                    if varType == varTypeEnum.HAND_JOINT:
                        if not jointAppendDone:
                            with tf.variable_scope('hand', reuse=True):
                                varAng = tf.get_variable(name=itemID + '_joint_global')
                                varList.append(varAng)
                                jointAppendDone = True
                    elif varType == varTypeEnum.HAND_ROT_REL_DELTA:
                        with tf.variable_scope('hand', reuse=True):
                            varRot = tf.get_variable(name=itemID + '_rot_delta_' + str(frame))
                            varList.append(varRot)
                    elif varType == varTypeEnum.HAND_TRANS_REL_DELTA:
                        with tf.variable_scope('hand', reuse=True):
                            varTrans = tf.get_variable(name=itemID + '_trans_delta_' + str(frame))
                            varList.append(varTrans)
                    elif varType == varTypeEnum.HAND_ROT:
                        if not rotAppendDone:
                            with tf.variable_scope('hand', reuse=True):
                                varRot = tf.get_variable(name=itemID + '_rot_global')
                                varList.append(varRot)
                                rotAppendDone = True
                    elif varType == varTypeEnum.HAND_TRANS:
                        if not transAppendDone:
                            with tf.variable_scope('hand', reuse=True):
                                varTrans = tf.get_variable(name=itemID + '_trans_global')
                                varList.append(varTrans)
                                transAppendDone = True
                    elif varType == varTypeEnum.HAND_BETA:
                        if not betaAppendDone:
                            with tf.variable_scope('hand', reuse=True):
                                varBeta = tf.get_variable(name=itemID + '_beta_global')
                                varList.append(varBeta)
                                betaAppendDone = True
                    elif varType == varTypeEnum.OBJ_TRANS:
                        with tf.variable_scope('obj', reuse=True):
                            varTrans = tf.get_variable(name=itemID + '_trans_' + str(frame))
                            varList.append(varTrans)
                    elif varType == varTypeEnum.OBJ_ROT:
                        with tf.variable_scope('obj', reuse=True):
                            varRot = tf.get_variable(name=itemID + '_rot_' + str(frame))
                            varList.append(varRot)

                elif self.optMode == optModeEnum.MULTIFRAME_RIGID_HO_POSE_JOINT:
                    if varType == varTypeEnum.HAND_JOINT:
                        with tf.variable_scope('hand', reuse=True):
                            varAng = tf.get_variable(name=itemID + '_joint_' + str(frame))
                            varList.append(varAng)
                    elif varType == varTypeEnum.HAND_ROT_REL_DELTA:
                        with tf.variable_scope('hand', reuse=True):
                            varRot = tf.get_variable(name=itemID + '_rot_delta_' + str(frame))
                            varList.append(varRot)
                    elif varType == varTypeEnum.HAND_TRANS_REL_DELTA:
                        with tf.variable_scope('hand', reuse=True):
                            varTrans = tf.get_variable(name=itemID + '_trans_delta_' + str(frame))
                            varList.append(varTrans)
                    elif varType == varTypeEnum.HAND_ROT:
                        if not rotAppendDone:
                            with tf.variable_scope('hand', reuse=True):
                                varRot = tf.get_variable(name=itemID + '_rot_global')
                                varList.append(varRot)
                                rotAppendDone = True
                    elif varType == varTypeEnum.HAND_TRANS:
                        if not transAppendDone:
                            with tf.variable_scope('hand', reuse=True):
                                varTrans = tf.get_variable(name=itemID + '_trans_global')
                                varList.append(varTrans)
                                transAppendDone = True
                    elif varType == varTypeEnum.HAND_BETA:
                        if not betaAppendDone:
                            with tf.variable_scope('hand', reuse=True):
                                varBeta = tf.get_variable(name=itemID + '_beta_global')
                                varList.append(varBeta)
                                betaAppendDone = True
                    elif varType == varTypeEnum.OBJ_TRANS:
                        with tf.variable_scope('obj', reuse=True):
                            varTrans = tf.get_variable(name=itemID + '_trans_' + str(frame))
                            varList.append(varTrans)
                    elif varType == varTypeEnum.OBJ_ROT:
                        with tf.variable_scope('obj', reuse=True):
                            varRot = tf.get_variable(name=itemID + '_rot_' + str(frame))
                            varList.append(varRot)

                elif self.optMode == optModeEnum.MULTIFRAME_JOINT or self.optMode == optModeEnum.MUTLICAMERA_MULTIFRAME:
                    if varType == varTypeEnum.HAND_JOINT:
                        with tf.variable_scope('hand', reuse=True):
                            varAng = tf.get_variable(name=itemID + '_joint_' + str(frame))
                            varList.append(varAng)
                    elif varType == varTypeEnum.HAND_ROT:
                        with tf.variable_scope('hand', reuse=True):
                            varRot = tf.get_variable(name=itemID + '_rot_' + str(frame))
                            varList.append(varRot)
                    elif varType == varTypeEnum.HAND_TRANS:
                        with tf.variable_scope('hand', reuse=True):
                            varTrans = tf.get_variable(name=itemID + '_trans_' + str(frame))
                            varList.append(varTrans)
                    elif varType == varTypeEnum.HAND_BETA:
                        if not betaAppendDone:
                            with tf.variable_scope('hand', reuse=True):
                                varBeta = tf.get_variable(name=itemID + '_beta_global')
                                varList.append(varBeta)
                                betaAppendDone = True
                    elif varType == varTypeEnum.OBJ_TRANS:
                        with tf.variable_scope('obj', reuse=True):
                            varTrans = tf.get_variable(name=itemID + '_trans_' + str(frame))
                            varList.append(varTrans)
                    elif varType == varTypeEnum.OBJ_ROT:
                        with tf.variable_scope('obj', reuse=True):
                            varRot = tf.get_variable(name=itemID + '_rot_' + str(frame))
                            varList.append(varRot)

                elif self.optMode == optModeEnum.MULTICAMERA:
                    if varType == varTypeEnum.HAND_JOINT:
                        with tf.variable_scope('hand', reuse=True):
                            varAng = tf.get_variable(name=itemID + '_joint_global')
                            varList.append(varAng)
                    elif varType == varTypeEnum.HAND_ROT:
                        with tf.variable_scope('hand', reuse=True):
                            varRot = tf.get_variable(name=itemID + '_rot_global')
                            varList.append(varRot)
                    elif varType == varTypeEnum.HAND_TRANS:
                        with tf.variable_scope('hand', reuse=True):
                            varTrans = tf.get_variable(name=itemID + '_trans_global')
                            varList.append(varTrans)
                    elif varType == varTypeEnum.HAND_BETA:
                        with tf.variable_scope('hand', reuse=True):
                            varBeta = tf.get_variable(name=itemID + '_beta_global')
                            varList.append(varBeta)
                    elif varType == varTypeEnum.OBJ_TRANS:
                        with tf.variable_scope('obj', reuse=True):
                            varTrans = tf.get_variable(name=itemID + '_trans_global')
                            varList.append(varTrans)
                    elif varType == varTypeEnum.OBJ_ROT:
                        with tf.variable_scope('obj', reuse=True):
                            varRot = tf.get_variable(name=itemID + '_rot_global')
                            varList.append(varRot)

                else:
                    raise Exception('Invalid optimization Mode..')



        return varList



