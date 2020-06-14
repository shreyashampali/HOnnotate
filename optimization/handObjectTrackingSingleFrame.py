import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))

from ghope.common import *
from ghope.scene import Scene
from ghope.rendering import DirtRenderer
from ghope.loss import LossObservs
from ghope.optimization import Optimizer
from ghope.constraints import Constraints
from ghope.icp import Icp

from ghope.utils import *
from ext.mesh_loaders import *

import argparse, json, yaml
import os
import chumpy as ch
import matplotlib.pyplot as plt

from absl import flags
from absl import app
parser = argparse.ArgumentParser()
FLAGS = flags.FLAGS

from ghope.vis import renderScene

from HOdatasets.ho3d_multicamera.dataset import datasetHo3dMultiCamera
from HOdatasets.commonDS import datasetType, splitType
from HOdatasets.mypaths import *

depthScale = 0.00012498664727900177
bgDepth = 2.0
handLabel = 2
configFile = '/media/shreyas/ssd2/Dataset/HO3D_multicamera/Shreyas_Mug_4/configs/configHandPose.json'
handSegColor = np.array([0.,0.,1.])
objSegColor = np.array([1.,0.,0.])
DIRT_RENDERER = True
datasetName = datasetType.HO3D_MULTICAMERA
doPyRendFinalImage = False
DEPTH_THRESH = 0.75

parser.add_argument('--seq', default='0010', help='Sequence Name')
parser.add_argument('--camID', default='0', help='Sequence Name')
parser.add_argument('--numIter', type=int, default=50, help='Sequence Name')
parser.add_argument('--showFig', action='store_true', help='Show Visualization')
parser.add_argument('--doPyRender', action='store_true', help='Show object rendering, very slow!')
FLAGS = parser.parse_args()
showFig = FLAGS.showFig
doPyRendFinalImage = FLAGS.doPyRender

# tf.enable_eager_execution()


def handObjectTrack(w, h, objParamInit, handParamInit, objMesh, camProp, out_dir):
    ds = tf.data.Dataset.from_generator(lambda: dataGen(w, h),
                                        (tf.string, tf.float32, tf.float32, tf.float32, tf.float32),
                                        ((None,), (None, h, w, 3), (None, h, w, 3), (None, h, w, 3), (None, h, w, 3)))
    # assert len(objParamInitList)==len(handParamInitList)

    numFrames = 1

    # read real observations
    frameCntInt, loadData, realObservs = LossObservs.getRealObservables(ds, numFrames, w, h)
    icp = Icp(realObservs, camProp)

    # set up the scene
    scene = Scene(optModeEnum.MULTIFRAME_RIGID_HO_POSE, frameCnt=numFrames)
    objID = scene.addObject(objMesh, objParamInit, segColor=objSegColor)
    handID = scene.addHand(handParamInit, handSegColor, baseItemID=objID)
    scene.addCamera(f=camProp.f, c=camProp.c, near=camProp.near, far=camProp.far, frameSize=camProp.frameSize)
    finalMesh = scene.getFinalMesh()

    # render the scene
    renderer = DirtRenderer(finalMesh, renderModeEnum.SEG_COLOR_DEPTH)
    virtObservs = renderer.render()

    # get loss over observables
    observLoss = LossObservs(virtObservs, realObservs, renderModeEnum.SEG_COLOR_DEPTH)
    segLoss, depthLoss, colLoss = observLoss.getL2Loss(isClipDepthLoss=True, pyrLevel=2)

    # get parameters and constraints
    handConstrs = Constraints()
    paramListHand = scene.getVarsByItemID(handID, [varTypeEnum.HAND_JOINT, varTypeEnum.HAND_ROT])
    jointAngs = paramListHand[0]
    handRot = paramListHand[1]
    validTheta = tf.concat([handRot, jointAngs], axis=0)
    theta = handConstrs.getFullThetafromValidTheta(validTheta)
    thetaConstrs, _ = handConstrs.getHandThetaConstraints(validTheta, isValidTheta=True)

    paramListObj = scene.getParamsByItemID([parTypeEnum.OBJ_ROT, parTypeEnum.OBJ_TRANS, parTypeEnum.OBJ_POSE_MAT], objID)
    rotObj = paramListObj[0]
    transObj = paramListObj[1]
    poseMat = paramListObj[2]

    paramListHand = scene.getParamsByItemID([parTypeEnum.HAND_THETA, parTypeEnum.HAND_TRANS, parTypeEnum.HAND_BETA],
                                           handID)
    thetaMat = paramListHand[0]
    transHand = paramListHand[1]
    betaHand = paramListHand[2]

    # get icp losses
    icpLossHand = icp.getLoss(scene.itemPropsDict[handID].transformedMesh.v, handSegColor)
    icpLossObj = icp.getLoss(scene.itemPropsDict[objID].transformedMesh.v, objSegColor)

    # get rel hand obj pose loss
    handTransVars = tf.stack(scene.getVarsByItemID(handID, [varTypeEnum.HAND_TRANS_REL_DELTA]), axis=0)
    handRotVars = tf.stack(scene.getVarsByItemID(handID, [varTypeEnum.HAND_ROT_REL_DELTA]), axis=0)
    relPoseLoss = handConstrs.getHandObjRelDeltaPoseConstraint(handRotVars, handTransVars)

    # get final loss
    icpLoss = 1e3*icpLossHand + 1e3*icpLossObj
    totalLoss1 = 1.0e1*segLoss + 1e0*depthLoss + 0.0*colLoss + 1e2*thetaConstrs + icpLoss + 1e6*relPoseLoss
    totalLoss2 = 1.15 * segLoss + 5.0 * depthLoss + 0.0*colLoss + 1e2 * thetaConstrs

    # get the variables for opt
    optVarsHandList = scene.getVarsByItemID(handID, [#varTypeEnum.HAND_TRANS, varTypeEnum.HAND_ROT,
                                                     # varTypeEnum.HAND_ROT_REL_DELTA, varTypeEnum.HAND_TRANS_REL_DELTA,
                                                     varTypeEnum.HAND_JOINT], [])
    optVarsHandDelta = scene.getVarsByItemID(handID, [varTypeEnum.HAND_TRANS_REL_DELTA, varTypeEnum.HAND_ROT_REL_DELTA], [])
    optVarsHandJoint = scene.getVarsByItemID(handID, [varTypeEnum.HAND_JOINT], [])
    optVarsObjList = scene.getVarsByItemID(objID, [varTypeEnum.OBJ_TRANS, varTypeEnum.OBJ_ROT], [])
    optVarsList = optVarsObjList #+ optVarsHandList
    optVarsListNoJoints = optVarsObjList #+ optVarsHandList

    # get the initial val of variables for BFGS optimizer
    initVals = []
    for fID in range(len(objParamInitList)):
        initVals.append(handParamInitList[fID].trans)
        initVals.append(handParamInitList[fID].theta[:3])
    initVals.append(handParamInitList[0].theta[handConstrs.validThetaIDs][3:])
    for fID in range(len(objParamInitList)):
        initVals.append(objParamInitList[fID].trans)
        initVals.append(objParamInitList[fID].rot)
    initValsNp = np.concatenate(initVals, axis=0)


    # setup optimizer
    opti1 = Optimizer(totalLoss1, optVarsList, 'Adam', learning_rate=0.02/2.0, initVals=initValsNp)
    opti2 = Optimizer(totalLoss1, optVarsListNoJoints, 'Adam', learning_rate=0.01)

    # get the optimization reset ops
    resetOpt1 = tf.variables_initializer(opti1.optimizer.variables())
    resetOpt2 = tf.variables_initializer(opti2.optimizer.variables())

    # tf stuff
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    session = tf.Session(config=config)
    session.__enter__()
    tf.global_variables_initializer().run()

    # setup the plot window
    if showFig:
        plt.ion()
        fig = plt.figure()
        ax = fig.subplots(4, max(numFrames,1))
        axesList = [[],[],[],[]]
        for i in range(numFrames):
            axesList[0].append(ax[0].imshow(np.zeros((240,320,3), dtype=np.float32)))
            axesList[1].append(ax[1].imshow(np.zeros((240, 320, 3), dtype=np.float32)))
            axesList[2].append(ax[2].imshow(np.random.uniform(0,2,(240,320,3))))
            axesList[3].append(ax[3].imshow(np.random.uniform(0,1,(240,320,3))))
        plt.subplots_adjust(top=0.984,
                            bottom=0.016,
                            left=0.028,
                            right=0.99,
                            hspace=0.045,
                            wspace=0.124)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

    # python renderer for rendering object texture
    pyRend = renderScene(h, w)
    pyRend.addObjectFromMeshFile(modelPath, 'obj')
    pyRend.addCamera()
    pyRend.creatcamProjMat(camProp.f, camProp.c, camProp.near, camProp.far)

    segLossList = []
    depLossList = []
    icpLossList = []
    relPoseLossList = []

    while (True):
        session.run(resetOpt1)
        session.run(resetOpt2)

        # load new frame
        opti1.runOptimization(session, 1, {loadData: True})
        # print(icpLoss.eval(feed_dict={loadData: False}))
        # print(segLoss.eval(feed_dict={loadData: False}))
        # print(depthLoss.eval(feed_dict={loadData: False}))

        # run the optimization for new frame
        frameID = (realObservs.frameID.eval(feed_dict={loadData: False}))[0].decode('UTF-8')
        opti1.runOptimization(session, FLAGS.numIter, {loadData: False})


        segLossList.append(1.0*segLoss.eval(feed_dict={loadData: False}))
        depLossList.append(1.0*depthLoss.eval(feed_dict={loadData: False}))
        icpLossList.append(icpLoss.eval(feed_dict={loadData: False}))
        relPoseLossList.append(1e3*relPoseLoss.eval(feed_dict={loadData: False}))
        # icpLossList.append(1e2*icpLossObj.eval(feed_dict={loadData: False}))

        # show all the images for analysis
        plt.title(frameID)
        depRen = virtObservs.depth.eval(feed_dict={loadData: False})
        depGT = realObservs.depth.eval(feed_dict={loadData: False})
        segRen = virtObservs.seg.eval(feed_dict={loadData: False})
        segGT = realObservs.seg.eval(feed_dict={loadData: False})
        poseMatNp = poseMat.eval(feed_dict={loadData: False})
        colRen = virtObservs.col.eval(feed_dict={loadData: False})
        colGT = realObservs.col.eval(feed_dict={loadData: False})
        for f in range(numFrames):
            if doPyRendFinalImage:
                # render the obj col image
                pyRend.setObjectPose('obj', poseMatNp[f].T)
                cRend, dRend = pyRend.render()

                # blend with dirt rendered image to get full texture image
                dirtCol = colRen[f][:,:,[2,1,0]]
                objRendMask = (np.sum(np.abs(segRen[f] - objSegColor),2) < 0.05).astype(np.float32)
                objRendMask = np.stack([objRendMask,objRendMask,objRendMask], axis=2)
                finalCol = dirtCol*(1-objRendMask) + (cRend.astype(np.float32)/255.)*objRendMask

            if showFig:
                axesList[0][f].set_data(colGT[f])
                if doPyRendFinalImage:
                    axesList[1][f].set_data(finalCol)
                axesList[2][f].set_data(np.abs(depRen-depGT)[f,:,:,0])
                axesList[3][f].set_data(np.abs(segRen-segGT)[f,:,:,:])

            if f >= 0:
                coordChangMat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
                handJoints = scene.itemPropsDict[handID].transorfmedJs.eval(feed_dict={loadData: False})[f]
                camMat = camProp.getCamMat()
                handJointProj = cv2.projectPoints(handJoints.dot(coordChangMat), np.zeros((3,)), np.zeros((3,)), camMat, np.zeros((4,)))[0][:,0,:]
                imgIn = (colGT[f][:, :, [2, 1, 0]] * 255).astype(np.uint8).copy()
                imgIn = cv2.resize(imgIn, (imgIn.shape[1]*dscale, imgIn.shape[0]*dscale), interpolation=cv2.INTER_LANCZOS4)
                imgJoints = showHandJoints(imgIn, np.round(handJointProj).astype(np.int32)[jointsMapManoToObman]*dscale,
                                           estIn=None, filename=None, upscale=1, lineThickness=2)

                objCorners = getObjectCorners(mesh.v)
                rotObjNp = rotObj.eval(feed_dict={loadData: False})[f]
                transObjNp = transObj.eval(feed_dict={loadData: False})[f]
                objCornersTrans = np.matmul(objCorners, cv2.Rodrigues(rotObjNp)[0].T) + transObjNp
                objCornersProj = cv2.projectPoints(objCornersTrans.dot(coordChangMat), np.zeros((3,)), np.zeros((3,)), camMat, np.zeros((4,)))[0][:,0, :]
                imgJoints = showObjJoints(imgJoints, objCornersProj*dscale, lineThickness=2)

                alpha = 0.35
                rendMask = segRen[f]
                # rendMask[:,:,[1,2]] = 0
                rendMask = np.clip(255. * rendMask, 0, 255).astype('uint8')
                msk = rendMask.sum(axis=2) > 0
                msk = msk * alpha
                msk = np.stack([msk, msk, msk], axis=2)
                blended = msk * rendMask[:,:,[2,1,0]] + (1. - msk) * (colGT[f][:, :, [2, 1, 0]] * 255).astype(np.uint8)
                blended = blended.astype(np.uint8)

                cv2.imwrite(out_dir + '/annoVis_' + frameID + '.jpg', imgJoints)
                cv2.imwrite(out_dir + '/annoBlend_' + frameID + '.jpg', blended)
                cv2.imwrite(out_dir + '/maskOnly_' + frameID + '.jpg', (segRen[0] * 255).astype(np.uint8))
                depthEnc = encodeDepthImg(depRen[0,:,:,0])
                cv2.imwrite(out_dir + '/renderDepth_' + frameID + '.jpg', depthEnc)
                if doPyRendFinalImage:
                    cv2.imwrite(out_dir + '/renderCol_' + frameID + '.jpg', (finalCol[:, :, [2, 1, 0]]* 255).astype(np.uint8))



        if showFig:
            plt.savefig(out_dir + '/'+frameID+'.png')
            plt.waitforbuttonpress(0.01)


        # save all the vars
        optVarListNp = []
        for optVar in optVarsHandDelta:
            optVarListNp.append(optVar.eval())

        thetaNp = thetaMat.eval(feed_dict={loadData: False})[0]
        betaNp = betaHand.eval(feed_dict={loadData: False})[0]
        transNp = transHand.eval(feed_dict={loadData: False})[0]
        rotObjNp = rotObj.eval(feed_dict={loadData: False})[0]
        transObjNp = transObj.eval(feed_dict={loadData: False})[0]
        JTransformed = scene.itemPropsDict[handID].transorfmedJs.eval(feed_dict={loadData: False})
        handJproj = np.reshape(cv2ProjectPoints(camProp, np.reshape(JTransformed, [-1, 3])), [numFrames, JTransformed.shape[1], 2])
        # vis = getBatch2DPtVisFromDep(depRen, segRen, projPts, JTransformed, handSegColor)
        objCornersRest = np.load(os.path.join(YCB_OBJECT_CORNERS_DIR, obj.split('/')[0], 'corners.npy'))
        objCornersTransormed = objCornersRest.dot(cv2.Rodrigues(rotObjNp)[0].T) + transObjNp
        objCornersproj = np.reshape(cv2ProjectPoints(camProp, np.reshape(objCornersTransormed, [-1, 3])),
                               [objCornersTransormed.shape[0], 2])

        savePickleData(out_dir + '/' + frameID +'.pkl', {'beta':betaNp, 'fullpose': thetaNp, 'trans': transNp,
                                                        'rotObj':rotObjNp, 'transObj': transObjNp,
                                                        'JTransformed': JTransformed, 'objCornersRest': objCornersRest,
                                                        'objCornersTransormed': objCornersTransormed,
                                                        'objName': obj.split('/')[0], 'objLabel': objLabel})


def dataGen(w, h):

    with open(configFile) as config_file:
        data = yaml.safe_load(config_file)

    objAliasLabelList = []
    objLabel = data['objLabel']
    startAt = data['startAt']
    endAt = data['endAt']
    skip = data['skip']

    if 'camID' in data:
        camID = data['camID']
    else:
        camID = '0'

    handSegDir = os.path.join(HO3D_MULTI_CAMERA_DIR, FLAGS.seq, 'segmentation', FLAGS.camID, 'raw_seg_results')




    if datasetName == datasetType.HO3D_MULTICAMERA:
        files = os.listdir(join(base_dir, 'rgb', camID))
        files = [FLAGS.seq+'/'+camID+'/'+f1[:-4] for f1 in files if 'png' in f1]
        files = sorted(files)
        fileListIn = sorted(files)
        dataset = datasetHo3dMultiCamera(FLAGS.seq, 0, fileListIn=fileListIn)
        if isinstance(startAt, str):
            for i, f in enumerate(files):
                if f.split('/')[-1] == startAt:
                    break
            files = files[i::skip]
            # print(files)
        else:
            files = files[startAt:endAt + 1:skip]
    else:
        raise NotImplementedError


    for file in files:
        if datasetName == datasetType.HO3D_MULTICAMERA:
            seq = file.split('/')[0]
            camInd = file.split('/')[1]
            id = file.split('/')[2]
            _, ds = dataset.createTFExample(itemType='hand', fileIn=file)
            img = ds.imgRaw[:, :, [2, 1, 0]]
            dpt = ds.depth
            seg = cv2.imread(join(HO3D_MULTI_CAMERA_DIR, seq, 'segmentation', camInd, 'raw_seg_results', id + '.png'))[
                  :, :, 0]
            frameID = np.array([join(id)])
        else:
            raise NotImplementedError

        dpt = dpt[:, :, 0] + dpt[:, :, 1] * 256
        dpt = dpt * depthScale

        dptMask = np.logical_or(dpt > DEPTH_THRESH, dpt == 0.0)
        dpt[dptMask] = bgDepth

        seg[dpt > DEPTH_THRESH] = 0

        objMask = (seg == objLabel)
        handMask = (seg == handLabel)
        objImg = img * np.expand_dims(objMask, 2)
        handImg = img * np.expand_dims(handMask, 2)

        objDepth = dpt * objMask
        objDepth[np.logical_not(objMask)] = bgDepth
        handDepth = dpt * handMask
        handDepth[np.logical_not(handMask)] = bgDepth

        handMask = np.stack([handMask, handMask, handMask], axis=2).astype(np.float32) * handSegColor
        objMask = np.stack([objMask, objMask, objMask], axis=2).astype(np.float32) * objSegColor
        handObjMask = handMask + objMask
        handObjDepth = dpt * (np.sum(handObjMask, 2)>0).astype(np.float32)
        handObjDepth[np.sum(handObjMask, 2)==0] = bgDepth
        handObjImg = img #* np.expand_dims((np.sum(handObjMask, 2)>0).astype(np.float32), 2)


        mask = np.logical_not(np.zeros_like(objMask, dtype=np.bool))

        handObjDepth = cv2.resize(handObjDepth, (w, h), interpolation=cv2.INTER_NEAREST)
        handObjDepth = np.stack([handObjDepth, handObjDepth, handObjDepth], axis=2)
        handObjDepth = np.expand_dims(handObjDepth, 0).astype(np.float32)

        handObjMask = cv2.resize(handObjMask, (w, h), interpolation=cv2.INTER_NEAREST)
        handObjMask = np.expand_dims(handObjMask, 0).astype(np.float32)

        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, 0).astype(np.float32)

        handObjImg = cv2.resize(handObjImg, (w, h), interpolation=cv2.INTER_CUBIC)
        handObjImg = np.expand_dims(handObjImg, 0).astype(np.float32)
        handObjImg = handObjImg.astype(np.float32)/255.#np.zeros_like(mask, dtype=np.float32)


        yield (frameID, handObjMask, handObjDepth, handObjImg, mask)


if __name__ == '__main__':

    dscale = 2
    w = 640 // dscale
    h = 480 // dscale
    plt.ion()

    base_dir = os.path.join(HO3D_MULTI_CAMERA_DIR, FLAGS.seq)
    configFile = join(HO3D_MULTI_CAMERA_DIR, FLAGS.seq, 'configs/configHandObjPose.json')
    with open(configFile) as config_file:
        data = yaml.safe_load(config_file)
    obj = data['obj']
    objLabel = data['objLabel']
    startAt = data['startAt']
    endAt = data['endAt']
    skip = data['skip']
    camID = data['camID']

    out_dir = os.path.join(base_dir, 'dirt_hand_obj_pose')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if datasetName == datasetType.HO3D_MULTICAMERA:
        for i in range(1):
            if not os.path.exists(join(out_dir, str(i))):
                os.mkdir(join(out_dir, str(i)))
    if datasetName == datasetType.HO3D_MULTICAMERA:
            out_dir = join(out_dir, camID)

    ycbModelsDir = YCB_MODELS_DIR
    modelPath = os.path.join(ycbModelsDir, data['obj'])
    mesh = load_mesh(modelPath)

    if datasetName == datasetType.HO3D_MULTICAMERA:
        # dataset = datasetHo3dMultiCamera(splitType.TEST)
        files = os.listdir(join(base_dir, 'rgb', camID))
        files = [camID+'/'+f1[:-4] for f1 in files if 'png' in f1]
        if isinstance(startAt, str):
            for i, f in enumerate(files):
                if f.split('/')[-1] == startAt:
                    break
            files = files[i::skip]
            # print(files)
        else:
            files = files[startAt:endAt + 1:skip]

    else:
        raise NotImplementedError

    objParamInitList = []

    handParamInitList = []


    # load data other pickle file for visualtization outputs
    newInit = loadPickleData(base_dir+'/dirt_grasp_pose/graspPose.pkl')
    beta = newInit['beta'][0]#loadPickleData(betaFileName)['beta'][0]
    handParamAnchorFrame = handParams(theta=newInit['fullpose'][0], trans=newInit['trans'][0], beta=beta)
    objParamAnchorFrame = objParams(rot=newInit['rotObj'][0], trans=newInit['transObj'][0])
    handParamRel = getHORelPose(handParamAnchorFrame, objParamAnchorFrame)
    handParamInitList.append(handParamRel)

    assert os.path.exists(os.path.join(base_dir, 'dirt_obj_pose', files[0] + '.pkl')), 'Object pose not initialized for frame %s. Change ' \
                                                                                       '\'startAt\' in configHandObjPose.json to a frame for which' \
                                                                                       ' object pose initialization is available (check dirt_obj_pose folder)'%(files[0].split('/')[-1])


    initDict = loadPickleData(os.path.join(base_dir, 'dirt_obj_pose', files[0] + '.pkl'))
    rot = undo_chumpy(initDict['rot'])[0]
    trans = undo_chumpy(initDict['trans'])[0]
    paramInit = objParams(rot=rot, trans=trans)
    objParamInitList.append(paramInit)

    if datasetName == datasetType.HO3D_MULTICAMERA:
        camMat = datasetHo3dMultiCamera.getCamMat(FLAGS.seq)
        camProp = camProps(ID='cam1', f=np.array([camMat[0,0], camMat[1,1]], dtype=np.float32) / dscale,
                           c=np.array([camMat[0,2], camMat[1,2]], dtype=np.float32) / dscale,
                           near=0.001, far=2.0, frameSize=[w, h],
                           pose=np.eye(4, dtype=np.float32))
    else:
        raise NotImplementedError

    if False:
        myGen = dataGen(w, h)
        frameID, handMask, handDepth, col, mask = next(myGen)
        a = 10

    handObjectTrack(w, h, objParamInitList[0], handParamInitList[0], mesh, camProp, out_dir)






