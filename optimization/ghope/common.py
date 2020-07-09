import tensorflow as tf
import numpy as np
from enum import Enum
from dirt.matrices import rodrigues

class observables():
    def __init__(self, frameID, seg, depth, col, mask, isReal):
        '''
        Observables, either from actual camera or virtual camera (renderer)
        :param seg: segmentation map
        :param depth: depth map
        :param col: color image
        :param mask: mask
        :param isReal: is real or virtual
        '''
        if seg is not None:
            assert len(seg.shape) == 4, 'Segmenation map must be 4-D and must have 3 channels'
            assert seg.shape[3] == 3, 'Segmenation map must be 4-D and must have 3 channels'
        if depth is not None:
            assert len(depth.shape) == 4, 'Depth map must be 4-D and must have 3 channels'
        if col is not None:
            assert len(col.shape) == 4, 'Color image must be 4-D and must have 3 channels'
            assert col.shape[3] == 3, 'Color Image must have 3 channels'
        if mask is not None:
            assert len(mask.shape) == 4, 'Mask should be 4-D and should have 3 channels'
        assert isinstance(isReal, bool), 'isReal should be of type bool'

        self.frameID = frameID
        self.seg = seg
        self.depth = depth
        self.col = col
        self.mask = mask
        self.isReal = isReal

class handVars():
    def __init__(self, jointAngs, rot, trans, beta):
        assert rot.shape == (3,), 'rot should of (3,) shape'
        assert trans.shape == (3,), 'rot should of (3,) shape'
        assert beta.shape == (10,), 'rot should of (10,) shape'

        self.jointAngs = jointAngs
        self.rot = rot
        self.trans = trans
        self.beta = beta

class objVars():
    def __init__(self, rot, trans):
        assert rot.shape == (3,), 'rot should of (3,) shape'
        assert trans.shape == (3,), 'rot should of (3,) shape'

        self.rot = rot
        self.trans = trans

class handParams():
    '''
    Struct for initializing hand model variables
    '''
    def __init__(self, theta = np.zeros((48,), dtype=np.float32),
                beta = np.zeros((10,), dtype=np.float32),
                 trans = np.array([0., 0., -0.4]),
                 JTransformed = np.zeros((21,3), dtype=np.float32),
                 JVis = np.zeros((21,), dtype=np.bool),
                 JTransformed2D = None):

        assert theta.shape==(48,), 'theta should be (48,) shape'
        assert beta.shape==(10,), 'beta should be (10,) shape'
        assert trans.shape==(3,), 'trans should be (3,) shape'
        assert JTransformed.shape==(21,3)
        assert JVis.shape==(21,)

        self.theta = theta.astype(np.float32).copy()
        self.beta = beta.astype(np.float32).copy()
        self.trans = trans.astype(np.float32).copy()
        self.JTransformed = JTransformed.astype(np.float32).copy()
        self.JVis = JVis.astype(np.float32).copy()
        self.JTransformed2D = JTransformed2D
        

class objParams():
    '''
    Struct for initializing object pose
    '''
    def __init__(self,
                 rot = np.zeros((3,), dtype=np.float32),
                 trans = np.array([0., 0., -0.4])):

        assert rot.shape == (3,), 'rot should be (3,) shape'
        assert trans.shape == (3,), 'beta should be (3,) shape'

        self.rot = rot.astype(np.float32)
        self.trans = trans.astype(np.float32)



class itemProps(object):
    '''
    Struct for saving some properties of the items in the scene
    '''
    def __init__(self, ID, segColor, itemType, mesh=None, transformedMesh=None, tfVarsList=[]):
        assert isinstance(itemType, itemTypeEnum), 'itemType should be itemTypeEnum class'
        self.ID = ID
        self.segColor = segColor.astype(np.float32)
        self.mesh = mesh
        self.itemType = itemType
        self.transformedMesh = transformedMesh
        if not isinstance(tfVarsList, list):
            tfVarsList = [tfVarsList]
        self.tfVarsList = tfVarsList

    def addVarToList(self, var):
        self.tfVarsList.append(var)

    def setBaseItemID(self, itemID):
        self.baseItemID = itemID

    def setTransformedMesh(self, transformedMesh):
        assert len(transformedMesh.v.shape) == 3
        assert transformedMesh.v.shape[2] == 4
        self.transformedMesh = transformedMesh

    def setTransformedJoints(self, transorfmedJs):
        assert len(transorfmedJs.shape) == 3
        assert transorfmedJs.shape[1] == 21
        assert transorfmedJs.shape[2] == 3
        self.transorfmedJs = transorfmedJs

    def setHandPoseParameters(self, theta, trans, beta):
        '''
        Set the hand pose parameters for all the views
        :param poseParams:
        :return:
        '''
        assert self.itemType == itemTypeEnum.HAND, 'Cant set hand parameters for itemType that is not Hand'
        assert len((theta.shape)) == 2 or len((theta.shape)) == 4, 'pose parameters should be 2D(in Rodrigues) or 4D(in rotation matrix). NumViews x NumParams x  (3 x 3)'
        assert len((trans.shape)) == 2, 'pose parameters should be 2D. NumViews x NumParams'
        assert len((beta.shape)) == 2, 'pose parameters should be 2D. NumViews x NumParams'

        self.theta = theta
        self.trans = trans
        self.beta = beta

    def setObjPoseParameters(self, rot, trans):
        '''
        Set the obj pose parameters for all the views
        :param poseParams:
        :return:
        '''
        assert self.itemType == itemTypeEnum.OBJECT, 'Cant set hand parameters for itemType that is not Object'
        assert len((rot.shape)) == 2, 'pose parameters should be 2D. NumViews x NumParams'
        assert len((trans.shape)) == 2, 'pose parameters should be 2D. NumViews x NumParams'

        self.rot = rot
        self.trans = trans

    def getObjPoseMat(self):
        # get rotation matrix
        view_matrix_1 = tf.transpose(rodrigues(self.rot), perm=[0, 2, 1])  # Nx4x4

        # get translation matrix
        trans4 = tf.concat([self.trans, tf.ones((tf.shape(self.trans)[0], 1), dtype=tf.float32)], axis=1)
        trans4 = tf.expand_dims(trans4, 1)
        trans4 = tf.concat([np.tile(np.array([[[0., 0., 1., 0.]]]), [self.trans.shape[0], 1, 1]), trans4], axis=1)
        trans4 = tf.concat([np.tile(np.array([[[0., 1., 0., 0.]]]), [self.trans.shape[0], 1, 1]), trans4], axis=1)
        view_matrix_2 = tf.concat([np.tile(np.array([[[1., 0., 0., 0.]]]), [self.trans.shape[0], 1, 1]), trans4],
                                  axis=1)  # Nx4x4

        objPoseMat = tf.matmul(view_matrix_1, view_matrix_2) # this is a right multiplication matrix of size Nx4x4

        return objPoseMat


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
        camMat = np.array([[self.f[0], 0, self.c[0]],[0., self.f[1], self.c[1]],[0., 0., 1.]]).astype(np.float32)
        return camMat

class itemTypeEnum(Enum):
    '''
    Enum for the type of item
    '''
    HAND = 1

    OBJECT = 2

class contactLossTypeEnum(Enum):
    '''
    Enum for type of contact loss
    '''

    NORMAL_DOT_PRODUCT = 1

class optModeEnum(Enum):
    '''
    Enums for the mode of optimization
    '''
    # optimize over multiple frames of seq. rot/trans changes across frames, joint angle and beta are fixed
    MULTIFRAME = 1

    # optimize over multiple frames of seq. rot/trans, joint angles change across frames, beta is fixed
    # for hand tracking
    MULTIFRAME_JOINT = 2

    # optimize over multiple frames of seq. rel. rot/trans between hand and object, joint angles and beta are all fixed across frames
    MULTIFRAME_RIGID_HO_POSE = 4

    # optimize over multiple frames of seq. rel. rot/trans between hand and object, joint angles change across frames, beta is fixed across frames
    # used in global refinement
    MULTIFRAME_RIGID_HO_POSE_JOINT = 5

    # optimize over multiple views. rot/trans, joint angles and beta are fixed for all views
    MULTICAMERA = 3

    MUTLICAMERA_MULTIFRAME = 6 



class sceneModeEnum(Enum):
    '''
    Enum for the type of scene
    '''
    # only one object in the scene
    OBJECT = 1

    # only hand in the scene
    HAND = 2

    # hand and object in the scene
    HAND_OBJECT = 3

class renderModeEnum(Enum):
    '''
    Enum for the type of rendering
    '''
    # segmentation only
    SEG = 1

    # depth only
    DEPTH = 2

    # color only
    COLOR = 3

    # seg and depth
    SEG_DEPTH = 4

    # seg and col
    SEG_COLOR = 5

    # col and depth
    COLOR_DEPTH = 6

    # seg, depth and col
    SEG_COLOR_DEPTH = 7

class parTypeEnum(Enum):
    '''
    Enum for the type of parameters
    '''
    # hand theta variable
    HAND_THETA = 1

    # hand trans variable
    HAND_TRANS = 3

    # hand beta variable
    HAND_BETA = 4

    # object rot variable
    OBJ_ROT = 5

    # object trans variable
    OBJ_TRANS = 6

    # object pose mat (multiplication of rot and trans matrices)
    OBJ_POSE_MAT = 7

class varTypeEnum(Enum):
    '''
    Enum for the type of variables (TF variables)
    '''
    # hand joint variable
    HAND_JOINT = 1

    # hand rot variable
    HAND_ROT = 2

    # hand trans variable
    HAND_TRANS = 3

    # hand beta variable
    HAND_BETA = 4

    # object rot variable
    OBJ_ROT = 5

    # object trans variable
    OBJ_TRANS = 6

    # hand rot rel delta variable
    HAND_ROT_REL_DELTA = 7

    # hand trans rel delta variable
    HAND_TRANS_REL_DELTA = 8
