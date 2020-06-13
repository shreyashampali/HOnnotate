import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import cv2
import pickle
import time
from HOdatasets.mypaths import *
from ghope.common import *
from sklearn.preprocessing import normalize
import math

depthScale = 0.00012498664727900177

jointsMapObmanToMano = [0,
                        5, 6, 7,
                        9, 10, 11,
                        17, 18, 19,
                        13, 14, 15,
                        1, 2, 3,
                        4, 8, 12, 16, 20]

jointsMapManoToObman = [0,
                        13, 14, 15, 16,
                        1, 2, 3, 17,
                        4, 5, 6, 18,
                        10, 11, 12, 19,
                        7, 8, 9, 20]

def creatCamMat(f, c, near, far, imShape, name=None, pose=np.eye(4, dtype=np.float32)):
    with ops.name_scope(name, 'PerspectiveProjection', [f, c, near, far, imShape]) as scope:
        near = tf.convert_to_tensor(near, name='near')
        far = tf.convert_to_tensor(far, name='far')
        fx = tf.convert_to_tensor(f[0], name='focalLengthx')
        fy = tf.convert_to_tensor(f[1], name='focalLengthy')
        cx = tf.convert_to_tensor(c[0], name='princiPointx')
        cy = tf.convert_to_tensor(c[1], name='princiPointy')

        left = -cx
        right = imShape[0] - cx
        top = cy
        bottom = -(imShape[1] - cy)

        elements1 = tf.convert_to_tensor([
            [fx, 0., 0., 0, ],
            [0., fy, 0., 0.],
            [0., 0., -(far + near) / (far - near), -2. * far * near / (far - near)],
            [0., 0., -1., 0.]
        ], dtype=tf.float32)  # indexed by x/y/z/w (out), x/y/z/w (in)
        normMat = tf.convert_to_tensor([
            [2/(right-left), 0., 0., -((left+right)/2.)*(2/(right-left))],
            [0., 2/(top-bottom), 0., -((top+bottom)/2.)*(2/(top-bottom))],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ], dtype=tf.float32)
        elements = tf.matmul(tf.matmul(normMat, elements1), np.linalg.inv(pose))

        return tf.transpose(elements)

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
    depthImg = cv2.imread(inFileName)
    if dsize is not None:
        depthImg = cv2.resize(depthImg, dsize, interpolation=cv2.INTER_CUBIC)

    dpt = depthImg[:, :, 0] + depthImg[:, :, 1] * 256
    dpt = dpt * depthScale

    return dpt


def printStats(totalLoss, imBatch, imGTSeg, imGTDpt, global_step):
    errSeg = imBatch[0] - imGTSeg
    meanSeg = tf.reduce_mean(tf.abs(errSeg))

    errDepth = tf.abs(imBatch[1] - imGTDpt)
    errDepth = tf.reshape(errDepth, [-1])
    errDepthMask = tf.less(errDepth, 0.5)
    maxDepth = tf.reduce_max(tf.boolean_mask(errDepth, errDepthMask))
    minDepth = tf.reduce_min(tf.boolean_mask(errDepth, errDepthMask))
    meanDepth = tf.reduce_mean(tf.boolean_mask(errDepth, errDepthMask))

    totalLoss = tf.Print(totalLoss, ['global_step', global_step, 'MaxDepth', maxDepth, 'MeanDepth', meanDepth*1000,
                                         'meanSeg', meanSeg, 'totalLoss', totalLoss])

    return totalLoss


def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else np.array(x.r)


def assignTFVars(varsList, init):
    # if init is str read the pickle file
    if not isinstance(init, dict):
        if isinstance(init, str):
            with open(init, 'rb') as f:
                try:
                    initDict = pickle.load(f, encoding='latin1')
                except:
                    initDict = pickle.load(f)
        else:
            raise Exception('Invalid input to assignTFVars')
    else:
        initDict = init

    # assert len(varsList) == len(initDict), 'Cannot initialize the TF variables...'

    for i in range(len(varsList)):
        varName = varsList[i].name.split(':')[0]
        if varName not in initDict.keys():
            raise Exception('Cant initialize variable %s'%varName)
        varsList[i] = tf.assign(varsList[i], undo_chumpy(initDict[varName]))




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


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

def getHandJointLocs(betas):
    smpl_data = loadPickleData(MANO_MODEL_PATH)
    smpl_data['v_shaped'] = smpl_data['shapedirs'].dot(betas) + smpl_data['v_template']
    v_shaped = smpl_data['v_shaped']
    J_tmpx = smpl_data['J_regressor'].dot(v_shaped[:, 0])
    J_tmpy = smpl_data['J_regressor'].dot(v_shaped[:, 1])
    J_tmpz = smpl_data['J_regressor'].dot(v_shaped[:, 2])
    smpl_data['J'] = np.vstack((J_tmpx, J_tmpy, J_tmpz)).T

    return smpl_data['J']

def showHandJoints(imgInOrg, gtIn, estIn=None, filename=None, upscale=1, lineThickness=3):
    '''
    Utility function for displaying hand annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param estIn: estimated keypoints
    :param filename: dump image name
    :param upscale: scale factor
    :param lineThickness:
    :return:
    '''

    imgIn = np.copy(imgInOrg)

    # Set color for each finger
    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]

    PYTHON_VERSION = 3

    for joint_num in range(gtIn.shape[0]):

        color_code_num = (joint_num // 4)
        if joint_num in [0, 4, 8, 12, 16]:
            if PYTHON_VERSION == 3:
                joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
            else:
                joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

            cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)
        else:
            if PYTHON_VERSION == 3:
                joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
            else:
                joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

            cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)

    for limb_num in range(len(limbs)):

        x1 = gtIn[limbs[limb_num][0], 1]
        y1 = gtIn[limbs[limb_num][0], 0]
        x2 = gtIn[limbs[limb_num][1], 1]
        y2 = gtIn[limbs[limb_num][1], 0]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if length < 150 and length > 5:
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 3),
                                       int(deg),
                                       0, 360, 1)
            color_code_num = limb_num // 4
            if PYTHON_VERSION == 3:
                limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))
            else:
                limb_color = map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num])

            cv2.fillConvexPoly(imgIn, polygon, color=limb_color)


    if filename is not None:
        cv2.imwrite(filename, imgIn)

    return imgIn

def showHandJointsOld(imgIn, gtIn, estIn=None, filename=None, upscale=1, lineThickness=3):
    jointConns = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18], [0, 10, 11, 12, 19], [0, 7, 8, 9, 20], [0, 13, 14, 15, 16]]
    jointColsGt = [(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255)]
    jointColsEst  = []
    for col in jointColsGt:
        newCol = (col[0]+col[1]+col[2])/3
        jointColsEst.append((newCol, newCol, newCol))
    # draws lines connected using jointConns
    img = np.zeros((imgIn.shape[0], imgIn.shape[1], imgIn.shape[2]), dtype=np.uint8)
    img[:, :, :] = (imgIn).astype(np.uint8)

    img = cv2.resize(img, (upscale * imgIn.shape[1], upscale * imgIn.shape[0]), interpolation=cv2.INTER_CUBIC)
    if gtIn is not None:
        gt = gtIn.copy() * upscale
    if estIn is not None:
        est = estIn.copy() * upscale

    for i in range(len(jointConns)):
        for j in range(len(jointConns[i]) - 1):
            jntC = jointConns[i][j]
            jntN = jointConns[i][j+1]
            if gtIn is not None:
                cv2.line(img, (int(gt[jntC,0]), int(gt[jntC,1])), (int(gt[jntN,0]), int(gt[jntN,1])), jointColsGt[i], lineThickness)
            if estIn is not None:
                cv2.line(img, (int(est[jntC,0]), int(est[jntC,1])), (int(est[jntN,0]), int(est[jntN,1])), jointColsEst[i], lineThickness)

    if filename is not None:
        cv2.imwrite(filename, img)

    return img

def showObjJoints(imgIn, gtIn, estIn=None, filename=None, upscale=1, lineThickness=3):
    jointConns = [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [2, 6], [3,7]]
    jointColsGt = (255,255,0)
    newCol = (jointColsGt[0] + jointColsGt[1] + jointColsGt[2]) / 3
    jointColsEst  = (newCol, newCol, newCol)

    # draws lines connected using jointConns
    img = np.zeros((imgIn.shape[0], imgIn.shape[1], imgIn.shape[2]), dtype=np.uint8)
    img[:, :, :] = (imgIn).astype(np.uint8)

    img = cv2.resize(img, (upscale * imgIn.shape[1], upscale * imgIn.shape[0]), interpolation=cv2.INTER_CUBIC)
    if gtIn is not None:
        gt = gtIn.copy() * upscale
    if estIn is not None:
        est = estIn.copy() * upscale

    for i in range(len(jointConns)):
        for j in range(len(jointConns[i]) - 1):
            jntC = jointConns[i][j]
            jntN = jointConns[i][j+1]
            if gtIn is not None:
                cv2.line(img, (int(gt[jntC,0]), int(gt[jntC,1])), (int(gt[jntN,0]), int(gt[jntN,1])), jointColsGt, lineThickness)
            if estIn is not None:
                cv2.line(img, (int(est[jntC,0]), int(est[jntC,1])), (int(est[jntN,0]), int(est[jntN,1])), jointColsEst, lineThickness)

    if filename is not None:
        cv2.imwrite(filename, img)

    return img

def getObjectCorners(v):
    xmax = np.max(v[:, 0])
    xmin = np.min(v[:, 0])
    ymax = np.max(v[:, 1])
    ymin = np.min(v[:, 1])
    zmax = np.max(v[:, 2])
    zmin = np.min(v[:, 2])
    corners = np.array([[xmin, ymin, zmin],
               [xmin, ymin, zmax],
               [xmin, ymax, zmin],
               [xmin, ymax, zmax],
               [xmax, ymin, zmin],
               [xmax, ymin, zmax],
               [xmax, ymax, zmin],
               [xmax, ymax, zmax]])

    return corners

def getHORelPose(handParamObj, objParamObj):
    '''
    Gets the rel pose btw hand and obj. [R_O | T_O].[R_rel | J-R_rel.J+T_rel] = [R_H | J-R_H.J+T_H]. R_rel and T_rel are the rel hand pose wrt object
    :param handParamObj:
    :param objParamObj:
    :return:
    '''
    assert isinstance(handParamObj, handParams)
    assert isinstance(objParamObj, objParams)

    J = getHandJointLocs(handParamObj.beta)[0:1,:].T

    RRelMat = cv2.Rodrigues(objParamObj.rot)[0].T.dot(cv2.Rodrigues(handParamObj.theta[:3])[0])
    RRelRod = cv2.Rodrigues(RRelMat)[0][:,0]

    TRel = cv2.Rodrigues(objParamObj.rot)[0].T.dot(J + np.expand_dims(handParamObj.trans,0).T - np.expand_dims(objParamObj.trans,0).T) - J

    theta = handParamObj.theta.copy()
    theta[:3] = RRelRod
    handParamRel = handParams(theta=theta, beta=handParamObj.beta.copy(), trans=TRel[:,0])

    return handParamRel

def getAbsHandPoseFromRel(handParamRel, objParam):
    '''
    Given relative hand pose wrt object, get the abs hand pose after rotating the object
    :param handParamRel:
    :param objParam:
    :return:
    '''
    assert isinstance(handParamRel, handParams)
    assert isinstance(objParam, objParams)

    J = getHandJointLocs(handParamRel.beta)[0:1, :].T

    RAbsMat = cv2.Rodrigues(objParam.rot)[0].dot(cv2.Rodrigues(handParamRel.theta[:3])[0])
    RAbsRod = cv2.Rodrigues(RAbsMat)[0][:,0]

    TAbs = cv2.Rodrigues(objParam.rot)[0].dot(J + np.expand_dims(handParamRel.trans,0).T) + np.expand_dims(objParam.trans,0).T - J

    theta = handParamRel.theta.copy()
    theta[:3] = RAbsRod
    handParamAbs = handParams(theta=theta, beta=handParamRel.beta.copy(), trans=TAbs[:, 0])

    return handParamAbs

def cv2BatchRodrigues(src):
    '''
    Computes Batch Rodrigues using cv2
    :param src:
    :return:
    '''
    rodList = []
    for i in range(src.shape[0]):
        rodList.append(cv2.Rodrigues(src[i])[0])

    if len(src.shape)==3:
        rodList = [r[:,0] for r in rodList]

    rodBatch = np.stack(rodList, axis=0)

    return rodBatch

def tfProjectPoints(camProp, pts3D, isOpenGLCoords=True):
    '''
    TF function for projecting 3d points to 2d using TF
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
        pts3D = tf.matmul(pts3D, coordChangeMat.T)

    fx = camProp.f[0]
    fy = camProp.f[1]
    cx = camProp.c[0]
    cy = camProp.c[1]

    camMat = np.array([[fx, 0, cx], [0, fy, cy], [0., 0., 1.]], dtype=np.float32)
    camMat = tf.convert_to_tensor(camMat)

    projPts = tf.matmul(pts3D, tf.transpose(camMat))
    projPts = tf.stack([projPts[:,0]/projPts[:,2], projPts[:,1]/projPts[:,2]],axis=1)
    # projPts = projPts[:,:2]/projPts[:,2]

    assert len(projPts.shape) == 2

    return projPts

def tfProjectPointsMulticam(camPropList, pts3D, isOpenGLCoords=True, usePose = True, isInversePose=True):
    '''
    TF function for projecting 3d points to 2d using TF
    :param camProp:
    :param pts3D:
    :param isOpenGLCoords: is input pts in openGL?
    :return:
    '''
    assert isinstance(camPropList, list), 'camProp should be a list'
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 3

    # get the pose matrix for multiplication
    pts3DHomo = tf.concat([pts3D, tf.ones((pts3D.shape[0], pts3D.shape[1], 1), dtype=tf.float32 )], axis=2) #Nx21x4
    camPoseList = []
    camMatList = []
    for camProp in camPropList:
        assert isinstance(camProp, camProps)
        if isInversePose:
            camPoseList.append(np.linalg.inv(camProp.pose).T)
        else:
            camPoseList.append(camProp.pose.T)
        camMatList.append(camProp.getCamMat())
    camPoseAll = np.stack(camPoseList, axis=0) #Nx4x4
    camMatAll = np.stack(camMatList, axis=0) #Nx3x3

    # get the pts in the cam coordinate frame (for each camera)
    if usePose:
        pts3DHomoCamRef = tf.matmul(pts3DHomo, camPoseAll)[:,:,:3] #Nx21x3
    else:
        pts3DHomoCamRef = pts3DHomo[:,:,:3]

    # change to opencv coordinate system
    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if isOpenGLCoords:
        pts3DCamRef = tf.matmul(pts3DHomoCamRef,
                                np.tile(np.expand_dims(coordChangeMat.T,0), [pts3DHomoCamRef.shape[0],1,1])) #Nx21x3

    # project the points to camera
    projPtsAll = tf.matmul(pts3DCamRef, tf.transpose(camMatAll, [0,2,1])) #Nx21x3
    projPtsAll = projPtsAll/projPtsAll[:,:,2:]
    projPtsAll = projPtsAll[:,:,:2]

    # projPtsAll = tf.Print(projPtsAll, ['ProjPts', projPtsAll[:,0,:]],summarize=6)

    return projPtsAll

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

def getBatch2DPtVisFromSeg(seg, pts2D, segColor):
    '''
    Checks if pts2D belongs o segColor in seg. This is useful while checking if a point in visible in the image
    :param seg: segmentation map, NxWxHx3
    :param pts2D: 2d points NxMx2, M is the number of pts per frame
    :param segColor: color to which pts2D should belong
    :return: a bool array of size NxM
    '''

    assert len(seg.shape) == 4
    assert len(pts2D.shape) == 3
    assert pts2D.shape[2] == 2
    assert len(segColor) == 3

    batchSize = seg.shape[0]
    pts2D = np.round(pts2D.copy()).astype(np.uint32)

    vis = np.zeros(pts2D.shape[:2], dtype=np.bool)
    for i in range(batchSize):
        projPtCol = seg[i][pts2D[i, :, 1], pts2D[i, :, 0]]
        vis[i] = np.sqrt(np.sum((projPtCol - segColor)**2, axis=1))<0.05

    return vis

def getBatch2DPtVisFromDep(dep, seg, pts2D, pts3D, segColor):
    '''
    Checks if pts3D is visible or occluded by the item itself or some other item
    :param dep: depth map, NxWxHx3
    :param seg: segmentation map, NxWxHx3
    :param pts2D: 2d points NxMx2, M is the number of pts per frame
    :param pts3D: 3d points NxMx3, M is the number of pts per frame
    :param segColor: color to which pts2D should belong
    :return: a bool array of size NxM
    '''

    assert len(dep.shape) == 4
    assert len(seg.shape) == 4
    assert len(pts2D.shape) == 3
    assert len(pts3D.shape) == 3
    assert pts2D.shape[2] == 2
    assert pts3D.shape[2] == 3
    assert len(segColor) == 3

    batchSize = dep.shape[0]
    pts2D = np.round(pts2D.copy()).astype(np.uint32)

    vis = np.zeros(pts2D.shape[:2], dtype=np.bool)
    for i in range(batchSize):
        projPtCol = seg[i][pts2D[i, :, 1], pts2D[i, :, 0]]
        projPtDep = -dep[i][pts2D[i, :, 1], pts2D[i, :, 0]][:,0]
        visDep = np.abs(projPtDep - pts3D[i, :, 2])<0.035 #1cm
        visCol = np.sqrt(np.sum((projPtCol - segColor) ** 2, axis=1)) < 0.01
        vis[i] = np.logical_and(visCol, visDep)

    return vis

def convertPoseOpenDR_DIRT(rot, trans):
    assert rot.shape == (3,)
    assert trans.shape == (3,)

    coordChangMat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    newRot = cv2.Rodrigues(coordChangMat.dot(cv2.Rodrigues(rot)[0]))[0][:,0]
    newTrans = trans.dot(coordChangMat.T)

    return newRot, newTrans

def inverseKinematicCh(pts3D, fullposeInit = np.zeros((48,), dtype=np.float32),
                       transInit = np.array([0., 0., -0.45]), betaInit = np.zeros((10,), dtype=np.float32)):
    '''
    Performs inverse kinematic optimization when given the 3d points
    3D points --> MANO params
    :param pts3D:
    :param fullposeInit:
    :param transInit:
    :param betaInit:
    :return: fullpose, trans and beta vector
    '''
    import chumpy as ch
    from mano.chumpy.smpl_handpca_wrapper_HAND_only import load_model, load_model_withInputs
    from ghope.constraints import Constraints

    assert pts3D.shape == (21,3)

    constraints = Constraints()
    validTheta = ch.zeros((len(constraints.validThetaIDs,)))
    fullposeCh = constraints.fullThetaMat.dot(ch.expand_dims(validTheta,1))[:,0]
    transCh = ch.array(undo_chumpy(transInit))
    betaCh = ch.array(undo_chumpy(betaInit))

    pts3DGT = pts3D

    m = load_model_withInputs(MANO_MODEL_PATH, fullposeCh, transCh, betaCh, ncomps=6, flat_hand_mean=True)

    if fullposeInit.shape == (16,3,3):
        fpList = []
        for i in range(16):
            fpList.append(cv2.Rodrigues(fullposeInit[i])[0][:, 0])
        validTheta[:] = np.concatenate(fpList, 0)[constraints.validThetaIDs]
    else:
        validTheta[:] = undo_chumpy(fullposeInit[constraints.validThetaIDs])

    loss = {}

    projPts = m.J_transformed
    projPtsGT = pts3DGT
    loss['joints'] = (projPts - projPtsGT)

    # m.fullpose[Constraints().invalidThetaIDs] = 0.
    thetaConstMin, thetaConstMax = Constraints().getHandJointConstraintsCh(fullposeCh[3:])
    # loss['constMin'] = 1e4 * thetaConstMin
    # loss['constMax'] = 1e4 * thetaConstMax
    freeVars = [validTheta, transCh]

    def cbPass(_):
        print(np.max(np.abs(m.J_transformed - pts3DGT)))
        pass

    ch.minimize(loss, x0=freeVars, callback=cbPass, method='dogleg', options={'maxiter': 10})

    if True:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(m.J_transformed[:, 0], m.J_transformed[:, 1], m.J_transformed[:, 2], c='r')
        ax.scatter(pts3DGT[:, 0], pts3DGT[:, 1], pts3DGT[:, 2], c='b')
        plt.show()

    return undo_chumpy(m.fullpose), undo_chumpy(m.trans), undo_chumpy(m.betas)

def convertFullposeMatToVec(fullposeMat):
    assert fullposeMat.shape == (16,3,3)
    myList = []
    for i in range(16):
        myList.append(cv2.Rodrigues(fullposeMat[i])[0][:,0])
    return np.concatenate(myList, axis=0)

def forwardKinematics(fullpose, trans, beta):
    '''
    MANO parameters --> 3D pts
    :param fullpose:
    :param trans:
    :param beta:
    :return: 3D pts of size (21,3)
    '''
    import chumpy as ch
    from mano.chumpy.smpl_handpca_wrapper_HAND_only import load_model

    assert fullpose.shape == (48,)
    assert trans.shape == (3,)
    assert beta.shape == (10,)

    m = load_model(MANO_MODEL_PATH, ncomps=6, flat_hand_mean=True, optwrt='fullpose')
    m.fullpose[:] = undo_chumpy(fullpose)
    m.trans[:] = undo_chumpy(trans)
    m.betas[:] = undo_chumpy(beta)

    return undo_chumpy(m.J_transformed), m


def convertPosecoeffToFullposeNp(posecoeff, flat_hand_mean=False):
    ncomps = posecoeff.shape[0]
    posecoeff = posecoeff.copy()
    smpl_data = pickle.load(open(MANO_MODEL_PATH, 'rb'), encoding='latin1')

    smpl_data['hands_components'] = normalize(smpl_data['hands_components'], axis=1)
    hands_components = smpl_data['hands_components']
    hands_mean = np.zeros(hands_components.shape[1]) if flat_hand_mean else smpl_data['hands_mean']

    selected_components = np.vstack((hands_components[:ncomps]))
    hands_mean = hands_mean.copy()

    full_hand_pose = posecoeff.dot(selected_components)

    fullpose = undo_chumpy(hands_mean + full_hand_pose)

    return fullpose

def convertFullposeToPosecoeffNp(fullpose, ncomps=30, flat_hand_mean=False):
    fullpose = fullpose.copy()
    smpl_data = pickle.load(open(MANO_MODEL_PATH, 'rb'), encoding='latin1')

    smpl_data['hands_components'] = normalize(smpl_data['hands_components'], axis=1)
    hands_components = smpl_data['hands_components']
    hands_mean = np.zeros(hands_components.shape[1]) if flat_hand_mean else smpl_data['hands_mean']

    selected_components = np.vstack((hands_components[:ncomps]))
    hands_mean = hands_mean.copy()

    posecoeff = (fullpose - hands_mean).dot(selected_components.T)

    posecoeff = undo_chumpy(posecoeff)

    return posecoeff









