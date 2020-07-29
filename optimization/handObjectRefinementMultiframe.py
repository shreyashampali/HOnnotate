import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))

from ghope.common import *
from ghope.scene import Scene
from ghope.rendering import DirtRenderer
from ghope.loss import LossObservs
from ghope.optimization import Optimizer
from ghope.constraints import Constraints, ContactLoss
from ghope.icp import Icp

from ghope.utils import *
from ext.mesh_loaders import *

import argparse, json, yaml
import chumpy as ch
import matplotlib.pyplot as plt
from shutil import copyfile

from ghope.vis import renderScene
from HOdatasets.ho3d_multicamera.dataset import datasetHo3dMultiCamera
from HOdatasets.commonDS import datasetType, splitType
from HOdatasets.mypaths import *

from absl import flags
from absl import app
parser = argparse.ArgumentParser()
FLAGS = flags.FLAGS

depthScale = 0.00012498664727900177
bgDepth = 2.0
handLabel = 2
handSegColor = np.array([0.,0.,1.])
objSegColor = np.array([1.,0.,0.])
batchSize = 20
showFigForIter = False
doPyRender = False
DEPTH_THRESH = 0.75

parser.add_argument('--seq', default='0010', help='Sequence Name')
parser.add_argument('--camID', default='0', help='Sequence Name')
parser.add_argument('--numIter', type=int, default=40, help='Sequence Name')
parser.add_argument('--batchSize', type=int, default=20, help='Batch size')
parser.add_argument('--showFig', action='store_true', help='Show Visualization')
parser.add_argument('--doPyRender', action='store_true', help='Show object rendering, very slow!')
FLAGS = parser.parse_args()
showFigForIter = FLAGS.showFig
doPyRender = FLAGS.doPyRender
numIter = FLAGS.numIter

PRELOAD = False # Load the initial pose parameters from latest files. Check code
DIRT_RENDERER = True #if False, assumes poses are from old openDR implementation, need some coordinate conversions then.
numIter = 40 # number of iteration in grad descent
datasetName = datasetType.HO3D_MULTICAMERA


def getVarInitsOp(dataset, scene, handID, objID, numViews):
    # joints are global variables
    opList = []
    handRelTheta, handRelTrans, handBeta, objRot, objTrans = dataset.make_one_shot_iterator().get_next()

    # initialize global variables for hand
    handGlobalVarList = scene.getVarsByItemID(handID, [varTypeEnum.HAND_ROT, varTypeEnum.HAND_JOINT, varTypeEnum.HAND_TRANS, varTypeEnum.HAND_BETA])
    thetaValidInit = handRelTheta[0][Constraints().validThetaIDs]
    opList.append(handGlobalVarList[0].assign(handRelTheta[0][:3]))
    opList.append(handGlobalVarList[1].assign(thetaValidInit))
    opList.append(handGlobalVarList[2].assign(handRelTrans[0]))
    opList.append(handGlobalVarList[3].assign(handBeta[0]))
    opList[-1] = tf.Print(opList[-1], ['Loading Variable %s' % (opList[-1].name)])

    # initialize frame wise variables for hand
    handFWiseVarList = scene.getVarsByItemID(handID,
                                              [varTypeEnum.HAND_ROT_REL_DELTA,
                                               varTypeEnum.HAND_TRANS_REL_DELTA])
    for v in range(numViews):
        ind = v * 2
        opList.append(handFWiseVarList[ind+0].assign(np.zeros((3,), dtype=np.float32))) # rot_delta
        opList.append(handFWiseVarList[ind+1].assign(np.zeros((3,), dtype=np.float32))) # trans_delta

    # initialize frame wise variables for object
    objFWiseVarList = scene.getVarsByItemID(objID,
                                             [varTypeEnum.OBJ_ROT,
                                              varTypeEnum.OBJ_TRANS])

    for v in range(numViews):
        ind = v * 2
        opList.append(objFWiseVarList[ind+0].assign(objRot[v]))
        opList.append(objFWiseVarList[ind+1].assign(objTrans[v]))
        opList[-1] = tf.Print(opList[-1], ['Loading Variable %s' % (opList[-1].name)])

    return opList

def getVarInitsOpJoints(dataset, scene, handID, objID, numViews):
    # joints are per frame variables
    # handRelTheta, handRelTrans, handBeta, objRot, objTrans
    opList = []
    handRelTheta, handRelTrans, handBeta, objRot, objTrans = dataset.make_one_shot_iterator().get_next()

    # initialize global variables for hand
    handGlobalVarList = scene.getVarsByItemID(handID, [varTypeEnum.HAND_ROT, varTypeEnum.HAND_TRANS, varTypeEnum.HAND_BETA])
    opList.append(handGlobalVarList[0].assign(handRelTheta[0][:3]))
    opList.append(handGlobalVarList[1].assign(handRelTrans[0]))
    opList.append(handGlobalVarList[2].assign(handBeta[0]))
    opList[-1] = tf.Print(opList[-1], ['Loading Variable %s' % (opList[-1].name)])

    # initialize frame wise variables for hand
    handFWiseVarList = scene.getVarsByItemID(handID,
                                              [varTypeEnum.HAND_JOINT,
                                               varTypeEnum.HAND_ROT_REL_DELTA,
                                               varTypeEnum.HAND_TRANS_REL_DELTA])
    for v in range(numViews):
        ind = v * 3
        # thetaValidInit = handRelTheta[v][Constraints().validThetaIDs]
        thetaValidInit = tf.gather(handRelTheta, Constraints().validThetaIDs, axis=1)[v]
        opList.append(handFWiseVarList[ind+0].assign(thetaValidInit[3:]))
        opList[-1] = tf.Print(opList[-1], ['Loading Variable %s' % (opList[-1].name)])
        opList.append(handFWiseVarList[ind+1].assign(np.zeros((3,), dtype=np.float32))) # rot_delta
        opList.append(handFWiseVarList[ind+2].assign(np.zeros((3,), dtype=np.float32))) # trans_delta

    # initialize frame wise variables for object
    objFWiseVarList = scene.getVarsByItemID(objID,
                                             [varTypeEnum.OBJ_ROT,
                                              varTypeEnum.OBJ_TRANS])

    for v in range(numViews):
        ind = v * 2
        # assOp = objFWiseVarList[ind+0].assign(objRot[v])
        # assOp = tf.Print(assOp, ['Rot', assOp])
        opList.append(objFWiseVarList[ind+0].assign(objRot[v]))
        opList.append(objFWiseVarList[ind+1].assign(objTrans[v]))
        opList[-1] = tf.Print(opList[-1], ['Loading Variable %s' % (opList[-1].name)])

    return opList

def handPoseMF(w, h, objParamInitList, handParamInitList, objMesh, camProp, out_dir):
    ds = tf.data.Dataset.from_generator(lambda: dataGen(w, h, batchSize),
                                        (tf.string, tf.float32, tf.float32, tf.float32, tf.float32),
                                        ((None,), (None, h, w, 3), (None, h, w, 3), (None, h, w, 3), (None, h, w, 3)))

    dsVarInit = tf.data.Dataset.from_generator(lambda: initVarGen(batchSize),
                                        (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                                        ((batchSize, 48), (batchSize, 3), (batchSize, 10), (batchSize, 3), (batchSize, 3)))

    assert len(objParamInitList)==len(handParamInitList)

    numFrames = len(objParamInitList)

    # read real observations
    frameCntInt, loadData, realObservs = LossObservs.getRealObservables(ds, numFrames, w, h)
    icp = Icp(realObservs, camProp)

    # set up the scene
    scene = Scene(optModeEnum.MULTIFRAME_RIGID_HO_POSE_JOINT, frameCnt=numFrames)
    objID = scene.addObject(objMesh, objParamInitList, segColor=objSegColor)
    handID = scene.addHand(handParamInitList, handSegColor, baseItemID=objID)
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

    # some variables for vis and analysis
    paramListObj = scene.getParamsByItemID([parTypeEnum.OBJ_ROT, parTypeEnum.OBJ_TRANS, parTypeEnum.OBJ_POSE_MAT], objID)
    rotObj = paramListObj[0]
    transObj = paramListObj[1]
    poseMat = paramListObj[2]

    paramListHand = scene.getParamsByItemID([parTypeEnum.HAND_THETA, parTypeEnum.HAND_TRANS, parTypeEnum.HAND_BETA],
                                           handID)
    thetaMat = paramListHand[0]
    transHand = paramListHand[1]
    betaHand = paramListHand[2]


    # contact loss
    hoContact = ContactLoss(scene.itemPropsDict[objID].transformedMesh, scene.itemPropsDict[handID].transformedMesh,
                            scene.itemPropsDict[handID].transorfmedJs)
    contLoss = hoContact.getRepulsionLoss()

    # get icp losses
    icpLossHand = icp.getLoss(scene.itemPropsDict[handID].transformedMesh.v, handSegColor)
    icpLossObj = icp.getLoss(scene.itemPropsDict[objID].transformedMesh.v, objSegColor)

    # get rel hand obj pose loss
    handTransVars = tf.stack(scene.getVarsByItemID(handID, [varTypeEnum.HAND_TRANS_REL_DELTA]), axis=0)
    handRotVars = tf.stack(scene.getVarsByItemID(handID, [varTypeEnum.HAND_ROT_REL_DELTA]), axis=0)
    relPoseLoss = handConstrs.getHandObjRelDeltaPoseConstraint(handRotVars, handTransVars)

    # get temporal loss
    handJointVars = tf.stack(scene.getVarsByItemID(handID, [varTypeEnum.HAND_JOINT]), axis=0)
    handRotVars = tf.stack(scene.getVarsByItemID(handID, [varTypeEnum.HAND_ROT]), axis=0)
    handTransVars = tf.stack(scene.getVarsByItemID(handID, [varTypeEnum.HAND_TRANS]), axis=0)
    objRotVars = tf.stack(scene.getVarsByItemID(objID, [varTypeEnum.OBJ_ROT]), axis=0)
    objTransVars = tf.stack(scene.getVarsByItemID(objID, [varTypeEnum.OBJ_TRANS]), axis=0)

    handJointsTempLoss = handConstrs.getTemporalConstraint(handJointVars, type='ZERO_ACCL')
    objRotTempLoss = handConstrs.getTemporalConstraint(objRotVars, type='ZERO_ACCL')
    handRotTempLoss = handConstrs.getTemporalConstraint(handRotVars, type='ZERO_ACCL')
    objTransTempLoss = handConstrs.getTemporalConstraint(objTransVars, type='ZERO_VEL')
    handTransTempLoss = handConstrs.getTemporalConstraint(handTransVars, type='ZERO_VEL')

    # get final loss
    segWt = 10.0
    depWt = 5.0
    colWt = 0.0
    thetaConstWt = 1e2
    icpHandWt = 1e2
    icpObjWt = 1e2
    relPoseWt  = 1e2
    contactWt = 0#1e-2

    handJointsTempWt = 1e1
    objRotTempWt = 1e1
    handRotTempWt = 0.
    objTransTempWt = 5e2
    handTransTempWt = 5e1
    totalLoss1 = segWt*segLoss + depWt*depthLoss + colWt*colLoss + thetaConstWt*thetaConstrs + icpHandWt*icpLossHand + icpObjWt*icpLossObj + \
                 relPoseWt*relPoseLoss + contactWt*contLoss + handJointsTempWt*handJointsTempLoss + \
                 objRotTempWt*objRotTempLoss + objTransTempWt*objTransTempLoss# + handRotTempWt*handRotTempLoss + handTransTempWt*handTransTempLoss
    totalLoss2 = 1.15 * segLoss + 5.0 * depthLoss + 0.0*colLoss + 1e2 * thetaConstrs

    # get the variables for opt
    optVarsHandList = scene.getVarsByItemID(handID, [varTypeEnum.HAND_TRANS, varTypeEnum.HAND_ROT,
                                                     #varTypeEnum.HAND_ROT_REL_DELTA, varTypeEnum.HAND_TRANS_REL_DELTA,
                                                     varTypeEnum.HAND_JOINT], [])
    optVarsHandDelta = scene.getVarsByItemID(handID, [varTypeEnum.HAND_TRANS_REL_DELTA, varTypeEnum.HAND_ROT_REL_DELTA], [])
    optVarsHandJoint = scene.getVarsByItemID(handID, [varTypeEnum.HAND_JOINT], [])
    optVarsObjList = scene.getVarsByItemID(objID, [varTypeEnum.OBJ_TRANS, varTypeEnum.OBJ_ROT], [])
    optVarsList = optVarsHandList + optVarsObjList
    optVarsListNoJoints = optVarsHandList + optVarsObjList

    # get var init op
    initOpList = getVarInitsOpJoints(dsVarInit, scene, handID, objID, numFrames)

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
    opti1 = Optimizer(totalLoss1, optVarsList, 'Adam', learning_rate=1.0*0.02/2.0, initVals=initValsNp)
    opti2 = Optimizer(totalLoss1, optVarsListNoJoints, 'Adam', learning_rate=0.01)

    # get the optimization reset ops
    resetOpt1 = tf.variables_initializer(opti1.optimizer.variables())
    resetOpt2 = tf.variables_initializer(opti2.optimizer.variables())

    # tf stuff
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    session = tf.Session(config=config)
    session.__enter__()
    tf.global_variables_initializer().run()

    # setup the plot window

    if showFigForIter:
        plt.ion()
        fig = plt.figure()
        ax = fig.subplots(4, max(numFrames,2))
        axesList = [[],[],[],[]]
        for i in range(numFrames):
            axesList[0].append(ax[0, i].imshow(np.zeros((240,320,3), dtype=np.float32)))
            axesList[1].append(ax[1, i].imshow(np.zeros((240, 320, 3), dtype=np.float32)))
            axesList[2].append(ax[2, i].imshow(np.random.uniform(0,2,(240,320,3))))
            axesList[3].append(ax[3, i].imshow(np.random.uniform(0,1,(240,320,3))))
        plt.subplots_adjust(top=0.984,
                            bottom=0.016,
                            left=0.028,
                            right=0.99,
                            hspace=0.045,
                            wspace=0.124)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
    else:
        plt.ioff()

    # some init runs
    session.run(resetOpt1)
    session.run(resetOpt2)
    # opti1.runOptimization(session, 1, {loadData: True})
    # tl = totalLoss1.eval(feed_dict={loadData: True})

    # python renderer for rendering object texture
    pyRend = renderScene(h, w)
    pyRend.addObjectFromMeshFile(modelPath, 'obj')
    pyRend.addCamera()
    pyRend.creatcamProjMat(camProp.f, camProp.c, camProp.near, camProp.far)

    while True:
        segLossList = []
        depLossList = []
        icpLossList = []
        relPoseLossList = []
        repulLossList = []
        jointTempLossList = []
        objRotTempLossList = []
        objTransTempLossList = []

        # load the init values for variables
        session.run(initOpList)

        # load the real observations
        tl = totalLoss1.eval(feed_dict={loadData: True})
        for i in range(numIter):
            print('iteration ',i)

            thumb13 = cv2.Rodrigues(thetaMat.eval(feed_dict={loadData: False})[0][7])[0]

            # run the optimization for new frame
            frameID = (realObservs.frameID.eval(feed_dict={loadData: False}))[0].decode('UTF-8')
            iterDir = join(out_dir, frameID)
            if not os.path.exists(iterDir):
                os.mkdir(iterDir)
            if i < 0:
                opti2.runOptimization(session, 1,
                                      {loadData: False})  # , logLossFunc=True, lossPlotName='handLoss/'+frameID+'_1.png')
            else:
                opti1.runOptimization(session, 1, {loadData: False})#, logLossFunc=True, lossPlotName='handLoss/'+frameID+'_1.png')


            segLossList.append(segWt*segLoss.eval(feed_dict={loadData: False}))
            depLossList.append(depWt*depthLoss.eval(feed_dict={loadData: False}))
            icpLossList.append(icpHandWt*icpLossHand.eval(feed_dict={loadData: False}))
            relPoseLossList.append(relPoseWt*relPoseLoss.eval(feed_dict={loadData: False}))
            repulLossList.append(contactWt*contLoss.eval(feed_dict={loadData: False}))
            jointTempLossList.append(handJointsTempWt*handJointsTempLoss.eval(feed_dict={loadData: False}))
            objRotTempLossList.append(objRotTempWt * objRotTempLoss.eval(feed_dict={loadData: False}))
            objTransTempLossList.append(objTransTempWt * objTransTempLoss.eval(feed_dict={loadData: False}))
            # icpLossList.append(1e2*icpLossObj.eval(feed_dict={loadData: False}))

            # show all the images for analysis

            plt.title(str(i))
            depRen = virtObservs.depth.eval(feed_dict={loadData: False})
            depGT = realObservs.depth.eval(feed_dict={loadData: False})
            segRen = virtObservs.seg.eval(feed_dict={loadData: False})
            segGT = realObservs.seg.eval(feed_dict={loadData: False})
            poseMatNp = poseMat.eval(feed_dict={loadData: False})
            colRen = virtObservs.col.eval(feed_dict={loadData: False})
            colGT = realObservs.col.eval(feed_dict={loadData: False})
            finalCol = np.zeros_like(colRen)
            for f in range(numFrames):


                if doPyRender:
                    # render the obj col image
                    pyRend.setObjectPose('obj', poseMatNp[f].T)
                    cRend, dRend = pyRend.render()
                    # blend with dirt rendered image to get full texture image
                    dirtCol = colRen[f][:,:,[2,1,0]]
                    objRendMask = (np.sum(np.abs(segRen[f] - objSegColor),2) < 0.05).astype(np.float32)
                    objRendMask = np.stack([objRendMask,objRendMask,objRendMask], axis=2)
                    finalCol[f] = dirtCol*(1-objRendMask) + (cRend.astype(np.float32)/255.)*objRendMask

                if showFigForIter:
                    axesList[0][f].set_data(colGT[f])
                    if doPyRender:
                        axesList[1][f].set_data(finalCol[f])
                    axesList[2][f].set_data(np.abs(depRen-depGT)[f,:,:,0])
                    axesList[3][f].set_data(np.abs(segRen-segGT)[f,:,:,:])


            if showFigForIter:
                plt.savefig(iterDir + '/'+frameID+'_'+str(i)+'.png')
                plt.waitforbuttonpress(0.01)

        frameID = (realObservs.frameID.eval(feed_dict={loadData: False}))  # [0].decode('UTF-8')
        frameID = [f.decode('UTF-8') for f in frameID]
        print(frameID)
        for f in range(numFrames):
            coordChangMat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
            handJoints = scene.itemPropsDict[handID].transorfmedJs.eval(feed_dict={loadData: False})[f]
            camMat = camProp.getCamMat()
            handJointProj = \
            cv2.projectPoints(handJoints.dot(coordChangMat), np.zeros((3,)), np.zeros((3,)), camMat, np.zeros((4,)))[0][
            :, 0, :]
            imgIn = (colGT[f][:, :, [2, 1, 0]] * 255).astype(np.uint8).copy()
            imgIn = cv2.resize(imgIn, (imgIn.shape[1] * dscale, imgIn.shape[0] * dscale),
                               interpolation=cv2.INTER_LANCZOS4)
            imgIn = cv2.imread(join(base_dir, 'rgb', camID, frameID[f] + '.png'))
            imgJoints = showHandJoints(imgIn,
                                       np.round(handJointProj).astype(np.int32)[jointsMapManoToObman] * dscale,
                                       estIn=None, filename=None, upscale=1, lineThickness=2)

            objCorners = getObjectCorners(mesh.v)
            rotObjNp = rotObj.eval(feed_dict={loadData: False})[f]
            transObjNp = transObj.eval(feed_dict={loadData: False})[f]
            objCornersTrans = np.matmul(objCorners, cv2.Rodrigues(rotObjNp)[0].T) + transObjNp
            objCornersProj = \
            cv2.projectPoints(objCornersTrans.dot(coordChangMat), np.zeros((3,)), np.zeros((3,)), camMat,
                              np.zeros((4,)))[0][:, 0, :]
            imgJoints = showObjJoints(imgJoints, objCornersProj * dscale, lineThickness=2)

            #bg = cv2.imread('/home/shreyas/Desktop/checkCrop.jpg')
            #bg = cv2.resize(bg, (320, 240))
            #mask = np.sum(segRen[f], 2) > 0
            #mask = np.stack([mask, mask, mask], axis=2)
            # newImg = (finalCol[f, :, :, [2, 1, 0]] * 255).astype(np.uint8) * mask + bg * (1 - mask)

            alpha = 0.35
            rendMask = segRen[f]
            # rendMask[:,:,[1,2]] = 0
            rendMask = np.clip(255. * rendMask, 0, 255).astype('uint8')
            msk = rendMask.sum(axis=2) > 0
            msk = msk * alpha
            msk = np.stack([msk, msk, msk], axis=2)
            blended = msk * rendMask[:, :, [2, 1, 0]] + (1. - msk) * (colGT[f][:, :, [2, 1, 0]] * 255).astype(np.uint8)
            blended = blended.astype(np.uint8)

            # cv2.imwrite(base_dir+'/' + str(f) + '_blend.png', imgJoints)
            cv2.imwrite(out_dir + '/annoVis_' + frameID[f] + '.jpg', imgJoints)
            cv2.imwrite(out_dir + '/annoBlend_' + frameID[f] + '.jpg', blended)
            cv2.imwrite(out_dir + '/maskOnly_' + frameID[f] + '.jpg', (segRen[f] * 255).astype(np.uint8))
            depthEnc = encodeDepthImg(depRen[f, :, :, 0])
            cv2.imwrite(out_dir + '/renderDepth_' + frameID[f] + '.jpg', depthEnc)
            if doPyRender:
                cv2.imwrite(out_dir + '/renderCol_' + frameID[f] + '.jpg',
                            (finalCol[f][:, :, [2, 1, 0]] * 255).astype(np.uint8))

        # dump loss plots intermittently
        if True:
            segLossAll = np.array(segLossList)
            depLossAll = np.array(depLossList)
            icpLossAll = np.array(icpLossList)
            relPoseLossAll = np.array(relPoseLossList)
            repulLossAll = np.array(repulLossList)
            jointTempLossAll = np.array(jointTempLossList)
            objRotTempLossAll = np.array(objRotTempLossList)
            objTransTempLossAll = np.array(objTransTempLossList)

            fig1 = plt.figure(2)
            plt.plot(np.arange(0, (len(segLossList))), segLossAll, 'r')
            fig1.savefig(iterDir + '/' + 'plotSeg_%s'%(frameID[0]) + '.png')
            plt.close(fig1)
            fig1 = plt.figure(2)
            plt.plot(np.arange(0, (len(depLossList))), depLossAll, 'g')
            fig1.savefig(iterDir + '/' + 'plotDep_%s'%(frameID[0]) + '.png')
            plt.close(fig1)
            fig1 = plt.figure(2)
            plt.plot(np.arange(0, (len(icpLossList))), icpLossAll, 'b')
            fig1.savefig(iterDir + '/' + 'plotIcp_%s'%(frameID[0]) + '.png')
            plt.close(fig1)
            fig1 = plt.figure(2)
            plt.plot(np.arange(0, (len(relPoseLossList))), relPoseLossAll, 'b')
            fig1.savefig(iterDir + '/' + 'plotRelPose_%s'%(frameID[0]) + '.png')
            plt.close(fig1)
            fig1 = plt.figure(2)
            plt.plot(np.arange(0, (len(repulLossAll))), repulLossAll, 'b')
            fig1.savefig(iterDir + '/' + 'plotRepul_%s'%(frameID[0]) + '.png')
            plt.close(fig1)
            fig1 = plt.figure(2)
            plt.plot(np.arange(0, (len(jointTempLossAll))), jointTempLossAll, 'b')
            fig1.savefig(iterDir + '/' + 'plotJointTemp_%s'%(frameID[0]) + '.png')
            plt.close(fig1)
            fig1 = plt.figure(2)
            plt.plot(np.arange(0, (len(objRotTempLossAll))), objRotTempLossAll, 'b')
            fig1.savefig(iterDir + '/' + 'plotObjRotTemp_%s'%(frameID[0]) + '.png')
            plt.close(fig1)
            fig1 = plt.figure(2)
            plt.plot(np.arange(0, (len(objTransTempLossAll))), objTransTempLossAll, 'b')
            fig1.savefig(iterDir + '/' + 'plotObjTransTemp_%s'%(frameID[0]) + '.png')
            plt.close(fig1)

        # save all the vars
        relPoseLossNp = relPoseLoss.eval(feed_dict={loadData: False})
        handJointNp = optVarsHandJoint[0].eval()
        optVarListNp = []
        for optVar in optVarsHandDelta:
            optVarListNp.append(optVar.eval())


        thetaMatNp = thetaMat.eval(feed_dict={loadData: False})
        thetaNp = np.reshape(cv2BatchRodrigues(np.reshape(thetaMatNp, [-1,3,3])), [numFrames, 48])
        betaNp = betaHand.eval(feed_dict={loadData: False})
        transNp = transHand.eval(feed_dict={loadData: False})
        rotObjNp = rotObj.eval(feed_dict={loadData: False})
        transObjNp = transObj.eval(feed_dict={loadData: False})
        JTransformed = scene.itemPropsDict[handID].transorfmedJs.eval(feed_dict={loadData: False})
        projPts = np.reshape(cv2ProjectPoints(camProp, np.reshape(JTransformed, [-1, 3])), [numFrames, JTransformed.shape[1], 2])
        # vis = getBatch2DPtVisFromDep(depRen, segRen, projPts, JTransformed, handSegColor)
        for f in range(numFrames):
            objCornersRest = np.load(os.path.join(YCB_OBJECT_CORNERS_DIR, obj.split('/')[0], 'corners.npy'))
            objCornersTransormed = objCornersRest.dot(cv2.Rodrigues(rotObjNp[f])[0].T) + transObjNp[f]
            savePickleData(out_dir + '/' + frameID[f] + '.pkl', {'beta': betaNp[f], 'fullpose': thetaNp[f], 'trans': transNp[f],
                                                              'rotObj': rotObjNp[f], 'transObj': transObjNp[f],
                                                              'JTransformed': JTransformed[f],
                                                              'objCornersRest': objCornersRest,
                                                              'objCornersTransormed': objCornersTransormed,
                                                              'objName': obj.split('/')[0], 'objLabel': objLabel})


def initVarGen(batchSize):
    with open(configFile) as config_file:
        data = yaml.safe_load(config_file)

    startAt = data['startAt']
    endAt = data['endAt']
    skip = data['skip']

    handObjPoseDir = os.path.join(base_dir, 'dirt_hand_obj_pose')
    if datasetName == datasetType.HO3D_MULTICAMERA:
        handObjPoseDir = os.path.join(base_dir, 'dirt_hand_obj_pose', camID)
    else:
        raise NotImplementedError

    files = (os.listdir(handObjPoseDir))
    files = [f[:-4] + '.png' for f in files if 'pkl' in f]
    files = sorted(files)
    if isinstance(startAt, str):
        for i, f in enumerate(files):
            if startAt in f.split('/')[-1]:
                break
        files = files[i::skip]
        # print(files)
    else:
        files = files[startAt:endAt:skip]
    numBatches = len(files)//batchSize


    for i in range(numBatches):
        currBatchFiles = files[i*batchSize:(i+1)*batchSize]
        handRelTheta = np.zeros((batchSize, 48))
        handRelTrans = np.zeros((batchSize, 3))
        handBeta = np.zeros((batchSize, 10))
        objRot = np.zeros((batchSize, 3))
        objTrans = np.zeros((batchSize, 3))
        for j, file in enumerate(currBatchFiles):
            # handRelRot, handRelTrans, handJoints, objRot, objTrans
            pklData = loadPickleData(join(handObjPoseDir, file[:-4]+'.pkl'))
            fullpose = convertFullposeMatToVec(pklData['fullpose'])
            handParamInit = handParams(theta=fullpose, trans=pklData['trans'], beta=pklData['beta'])
            objParamsInit = objParams(rot=pklData['rotObj'], trans=pklData['transObj'])

            # pose of hand in other frames is calculated using the relative pose btw hand and object
            handParamRelInit = getHORelPose(handParamInit, objParamsInit)

            handRelTheta[j] = handParamRelInit.theta
            handRelTrans[j] = handParamRelInit.trans
            handBeta[j] = handParamRelInit.beta
            objRot[j] = objParamsInit.rot
            objTrans[j] = objParamsInit.trans

        yield (handRelTheta, handRelTrans, handBeta, objRot, objTrans)



def dataGen(w, h, batchSize):

    with open(configFile) as config_file:
        data = yaml.safe_load(config_file)

    obj = data['obj']
    objAliasLabelList = []
    objLabel = data['objLabel']
    startAt = data['startAt']
    endAt = data['endAt']
    skip = data['skip']

    if datasetName == datasetType.HO3D_MULTICAMERA:
        files = os.listdir(join(base_dir, 'rgb', camID))
        files = [FLAGS.seq + '/' + camID + '/' + f1[:-4] for f1 in files if 'png' in f1]
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
            files = files[startAt:endAt:skip]
    else:
        raise NotImplementedError
    numBatches = len(files)//batchSize


    # w = 640 // dscale
    # h = 480 // dscale


    for i in range(numBatches):
        frameIDList = []
        handObjMaskList = []
        handObjDepthList = []
        handObjImgList = []
        maskList = []
        currBatchFiles = files[i*batchSize:(i+1)*batchSize]
        for file in currBatchFiles:
            if datasetName == datasetType.HO3D_MULTICAMERA:
                seq = file.split('/')[0]
                camInd = file.split('/')[1]
                id = file.split('/')[2]
                _, ds = dataset.createTFExample(itemType='hand', fileIn=file)
                img = ds.imgRaw[:, :, [2, 1, 0]]
                dpt = ds.depth
                # reading the seg from file because the seg in ds has no objAliasLabel info
                seg = cv2.imread(join(HO3D_MULTI_CAMERA_DIR, seq, 'segmentation', camInd, 'raw_seg_results', id + '.png'))[
                  :, :, 0]

                frameID = np.array([id])
            else:
                raise NotImplementedError

            dpt = dpt[:, :, 0] + dpt[:, :, 1] * 256
            dpt = dpt * depthScale

            dptMask = np.logical_or(dpt > DEPTH_THRESH, dpt == 0.0)
            dpt[dptMask] = bgDepth

            seg[dpt > DEPTH_THRESH] = 0
            for alias in objAliasLabelList:
                seg[seg == alias] = objLabel

            objMask = (seg == objLabel)
            handMask = (seg == handLabel)
            objImg = img * np.expand_dims(objMask, 2)
            handImg = img * np.expand_dims(handMask, 2)

            objDepth = dpt * objMask
            objDepth[np.logical_not(objMask)] = bgDepth
            handDepth = dpt * handMask
            handDepth[np.logical_not(handMask)] = bgDepth
            # handDepth = handDepth*np.logical_or(handMask,objMask)
            # handDepth[np.logical_not(np.logical_or(handMask,objMask))] = bgDepth

            handMask = np.stack([handMask, handMask, handMask], axis=2).astype(np.float32) * handSegColor
            objMask = np.stack([objMask, objMask, objMask], axis=2).astype(np.float32) * objSegColor
            handObjMask = handMask + objMask
            handObjDepth = dpt * (np.sum(handObjMask, 2)>0).astype(np.float32)
            handObjDepth[np.sum(handObjMask, 2)==0] = bgDepth
            handObjImg = img #* np.expand_dims((np.sum(handObjMask, 2)>0).astype(np.float32), 2)


            mask = np.logical_not(np.zeros_like(objMask, dtype=np.bool))

            handObjDepth = cv2.resize(handObjDepth, (w, h), interpolation=cv2.INTER_NEAREST)
            handObjMask = cv2.resize(handObjMask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            handObjImg = cv2.resize(handObjImg, (w, h), interpolation=cv2.INTER_CUBIC)


            handObjMaskList.append(handObjMask)

            handObjDepth = np.stack([handObjDepth, handObjDepth, handObjDepth], axis=2)
            handObjDepthList.append(handObjDepth)

            maskList.append(mask)

            frameIDList.append(frameID[0])

            handObjImg = handObjImg.astype(np.float32)/255.#np.zeros_like(mask, dtype=np.float32)
            handObjImgList.append(handObjImg)

            # realObservs = observables(frameID=file[:-4], seg=handMask, depth=handDepth, col=None, mask=mask, isReal=True)

        yield (np.stack(frameIDList,0), np.stack(handObjMaskList,0), np.stack(handObjDepthList,0), np.stack(handObjImgList,0), np.stack(maskList,0))


if __name__ == '__main__':

    dscale = 4
    w = 640 // dscale
    h = 480 // dscale
    plt.ion()

    base_dir = os.path.join(HO3D_MULTI_CAMERA_DIR, FLAGS.seq)
    configFile = join(HO3D_MULTI_CAMERA_DIR, FLAGS.seq, 'configs/configHandObjPose.json')


    with open(configFile) as config_file:
        data = yaml.safe_load(config_file)
    out_dir = os.path.join(base_dir, 'dirt_hand_obj_refine')
    obj = data['obj']
    objLabel = data['objLabel']

    if 'camID' in data:
        camID = data['camID']
    else:
        camID = '0'

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

    objParamInitList = []
    handParamInitList = []

    if datasetName == datasetType.HO3D_MULTICAMERA:
        for i in range(batchSize):
            rot = np.zeros((3,), dtype=np.float32)
            trans = np.array([0., 0., -0.5], dtype=np.float32)
            paramInit = objParams(rot=rot, trans=trans)
            objParamInitList.append(paramInit)

        beta = np.zeros((10,), dtype=np.float32)
        theta = np.zeros((48,), dtype=np.float32)
        trans = np.array([0., 0., -0.5], dtype=np.float32)
    else:
        raise  NotImplementedError

    handParamFirstFrame = handParams(theta=theta, trans=trans, beta=beta)

    # pose of hand in other frames is calculated using the relative pose btw hand and object
    handParamRel = getHORelPose(handParamFirstFrame, objParamInitList[0])
    for i in range(len(objParamInitList)):
        # handParamInitList.append(getAbsHandPoseFromRel(handParamRel, objParamInitList[i]))
        handParamInitList.append(handParamRel)


    if datasetName == datasetType.HO3D_MULTICAMERA:
        camMat = datasetHo3dMultiCamera.getCamMat(FLAGS.seq)
        camProp = camProps(ID='cam1', f=np.array([camMat[0,0], camMat[1,1]], dtype=np.float32) / dscale,
                           c=np.array([camMat[0,2], camMat[1,2]], dtype=np.float32) / dscale,
                           near=0.001, far=2.0, frameSize=[w, h],
                           pose=np.eye(4, dtype=np.float32))
    else:
        raise NotImplementedError

    if False:
        myGen = initVarGen(batchSize)
        handRelTheta, handRelTrans, handBeta, objRot, objTrans = next(myGen)
        a = 10

    if False:
        myGen = dataGen(w,h,batchSize)
        while(True):
            frameID, seg, depth, col, mask = next(myGen)
            a = 10

    handPoseMF(w, h, objParamInitList, handParamInitList, mesh, camProp, out_dir)






