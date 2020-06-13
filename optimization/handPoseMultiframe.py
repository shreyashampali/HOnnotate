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
import open3d as o3d

from ghope.vis import renderScene
from dirt.matrices import rodrigues
from copy import deepcopy

from HOdatasets.ho3d_multicamera.dataset import datasetHo3dMultiCamera
from HOdatasets.commonDS import datasetType, splitType
from HOdatasets.mypaths import *
datasetName = datasetType.HO3D_MULTICAMERA

depthScale = 0.00012498664727900177
bgDepth = 2.0
handLabel = 2
handSegColor = np.array([0.,0.,1.])
objSegColor = np.array([1.,0.,0.])

OBJ_POSE_FROM_DIRT = True #if False, assumes poses are from old openDR implementation, need some coordinate conversions then.
numIter = 350 # number of iteration in grad descent
useAutoInit = True
numAutoInitFiles = 11
use2DJointLoss = True

dscale = 2
w = 640 // dscale
h = 480 // dscale

from absl import flags
from absl import app
parser = argparse.ArgumentParser()
FLAGS = flags.FLAGS

parser.add_argument('--seq', default='0010', help='Sequence Name')
parser.add_argument('--camID', default='0', help='Sequence Name')
parser.add_argument('--numIter', type=int, default=150, help='Sequence Name')
parser.add_argument('--numOptFrames', type=int, default=11, help='Number frames to determine hand pose')
parser.add_argument('--showFig', action='store_true', help='Show Visualization')
parser.add_argument('--doPyRender', action='store_true', help='Show hand-object rendering, very slow!')
FLAGS = parser.parse_args()
# FLAGS.showFig=False
numIter = FLAGS.numIter
numAutoInitFiles = FLAGS.numOptFrames


# tf.enable_eager_execution()

def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback(pcd,
                                                              rotate_view)

def handPoseMF(w, h, objParamInitList, handParamInitList, objMesh, camProp, out_dir):
    ds = tf.data.Dataset.from_generator(lambda: dataGen(w, h),
                                        (tf.string, tf.float32, tf.float32, tf.float32, tf.float32),
                                        ((None,), (None, h, w, 3), (None, h, w, 3), (None, h, w, 3), (None, h, w, 3)))
    assert len(objParamInitList)==len(handParamInitList)

    numFrames = len(objParamInitList)

    # read real observations
    frameCntInt, loadData, realObservs = LossObservs.getRealObservables(ds, numFrames, w, h)
    icp = Icp(realObservs, camProp)

    # set up the scene
    scene = Scene(optModeEnum.MULTIFRAME_RIGID_HO_POSE, frameCnt=numFrames)
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

    if use2DJointLoss:
        # get 2d joint loss
        transJs = tf.reshape(scene.itemPropsDict[handID].transorfmedJs, [-1, 3])
        projJs = tfProjectPoints(camProp, transJs)
        projJs = tf.reshape(projJs, [numFrames, 21, 2])
        if handParamInitList[0].JTransformed[0, 2] < 0:
            isOpenGLCoords = True
        else:
            isOpenGLCoords = False
        joints2DGT = np.stack(
            [cv2ProjectPoints(camProp, hpi.JTransformed, isOpenGLCoords) for hpi in handParamInitList], axis=0)
        jointVisGT = np.stack([hpi.JVis for hpi in handParamInitList], axis=0)
        joints2DErr = tf.reshape(tf.reduce_sum(tf.square(projJs - joints2DGT), axis=2), [-1])
        joints2DErr = joints2DErr * tf.reshape(jointVisGT, [-1])#tf.boolean_mask(joints2DErr, tf.reshape(jointVisGT, [-1]))
        joints2DLoss = tf.reduce_sum(joints2DErr)
        wrist2DErr = tf.reshape(tf.reduce_sum(tf.square(projJs[:,0,:] - joints2DGT[:,0,:]), axis=1), [-1])
        wrist2DErr = wrist2DErr * tf.reshape(jointVisGT[:,0], [-1])
        wrist2DLoss = tf.reduce_sum(wrist2DErr)

    # get final loss
    icpWt = 5e3#1e3#1e2
    j2dWt = 0.#1e-5
    segWt = 5e1
    depWt = 1e1
    wristJWt = 1e-3#1e-1
    contactWt = 1e-1
    totalLoss1 = segWt*segLoss + depWt*depthLoss + 0.0*colLoss + 1e2*thetaConstrs + icpWt*icpLossHand + icpWt*icpLossObj + 1e6*relPoseLoss + contactWt*contLoss
    if use2DJointLoss:
        totalLoss1 = totalLoss1 + j2dWt*joints2DLoss + wristJWt*wrist2DLoss
    totalLoss2 = 1.15 * segLoss + 5.0 * depthLoss + 0.0*colLoss + 1e2 * thetaConstrs

    # get the variables for opt
    optVarsHandList = scene.getVarsByItemID(handID, [varTypeEnum.HAND_TRANS, varTypeEnum.HAND_ROT,
                                                     # varTypeEnum.HAND_ROT_REL_DELTA, varTypeEnum.HAND_TRANS_REL_DELTA,
                                                     # varTypeEnum.HAND_JOINT,
                                                     ], [])
    optVarsHandDelta = scene.getVarsByItemID(handID, [varTypeEnum.HAND_TRANS_REL_DELTA, varTypeEnum.HAND_ROT_REL_DELTA], [])
    optVarsHandJoint = scene.getVarsByItemID(handID, [varTypeEnum.HAND_JOINT], [])
    optVarsHandBeta = scene.getVarsByItemID(handID, [varTypeEnum.HAND_BETA], [])
    optVarsObjList = scene.getVarsByItemID(objID, [varTypeEnum.OBJ_TRANS, varTypeEnum.OBJ_ROT], [])
    optVarsList = optVarsHandList + optVarsObjList + optVarsHandJoint
    optVarsListNoJoints = optVarsHandList + optVarsObjList

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
    optiBeta = Optimizer(totalLoss1, optVarsHandBeta, 'Adam', learning_rate=0.05)


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
    if FLAGS.showFig:
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
    tl = totalLoss1.eval(feed_dict={loadData: True})

    # python renderer for rendering object texture
    configFile = join(HO3D_MULTI_CAMERA_DIR, FLAGS.seq, 'configs/configHandPose.json')
    with open(configFile) as config_file:
        data = yaml.safe_load(config_file)
    modelPath = os.path.join(YCB_MODELS_DIR, data['obj'])
    pyRend = renderScene(h, w)
    pyRend.addObjectFromMeshFile(modelPath, 'obj')
    pyRend.addCamera()
    pyRend.creatcamProjMat(camProp.f, camProp.c, camProp.near, camProp.far)

    segLossList = []
    depLossList = []
    icpLossList = []
    relPoseLossList = []
    repulLossList = []
    joints2DLossList = []


    for i in range(numIter):
        print('iteration ',i)

        thumb13 = cv2.Rodrigues(thetaMat.eval(feed_dict={loadData: False})[0][7])[0]


        if i < 35:
            opti2.runOptimization(session, 1, {loadData: False})  # , logLossFunc=True, lossPlotName='handLoss/'+frameID+'_1.png')
        elif i>350:
            optiBeta.runOptimization(session, 1, {loadData: False})  # , logLossFunc=True, lossPlotName='handLoss/'+frameID+'_1.png')
        else:
            opti1.runOptimization(session, 1, {loadData: False})#, logLossFunc=True, lossPlotName='handLoss/'+frameID+'_1.png')


        segLossList.append(segWt*segLoss.eval(feed_dict={loadData: False}))
        depLossList.append(depWt*depthLoss.eval(feed_dict={loadData: False}))
        icpLossList.append(icpWt*icpLossHand.eval(feed_dict={loadData: False}))
        relPoseLossList.append(1e6*relPoseLoss.eval(feed_dict={loadData: False}))
        repulLossList.append(contactWt*contLoss.eval(feed_dict={loadData: False}))
        if use2DJointLoss:
            joints2DLossList.append(j2dWt*joints2DLoss.eval(feed_dict={loadData: False}))
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
        frameIDList = (realObservs.frameID.eval(feed_dict={loadData: False}))
        frameIDList = [f.decode('UTF-8') for f in frameIDList]


        for f in range(numFrames):
            if (i % 1 == 0) and FLAGS.showFig:
                frameID = frameIDList[f]
                # frameIDList.append(frameID)
                # render the obj col image
                pyRend.setObjectPose('obj', poseMatNp[f].T)
                if FLAGS.doPyRender:
                    cRend, dRend = pyRend.render()

                # blend with dirt rendered image to get full texture image
                dirtCol = colRen[f][:,:,[2,1,0]]
                objRendMask = (np.sum(np.abs(segRen[f] - objSegColor),2) < 0.05).astype(np.float32)
                objRendMask = np.stack([objRendMask,objRendMask,objRendMask], axis=2)
                if FLAGS.doPyRender:
                    finalCol = dirtCol*(1-objRendMask) + (cRend.astype(np.float32)/255.)*objRendMask

                axesList[0][f].set_data(colGT[f])
                if FLAGS.doPyRender:
                    axesList[1][f].set_data(finalCol)
                axesList[2][f].set_data(np.abs(depRen-depGT)[f,:,:,0])
                axesList[3][f].set_data(np.abs(segRen-segGT)[f,:,:,:])


                coordChangMat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
                handJoints = scene.itemPropsDict[handID].transorfmedJs.eval(feed_dict={loadData: False})[f]
                camMat = camProp.getCamMat()
                handJointProj = cv2.projectPoints(handJoints.dot(coordChangMat), np.zeros((3,)), np.zeros((3,)), camMat, np.zeros((4,)))[0][:,0,:]
                imgIn = (colGT[f][:, :, [2, 1, 0]] * 255).astype(np.uint8).copy()
                imgIn = cv2.resize(imgIn, (dscale*imgIn.shape[1], dscale*imgIn.shape[0]), interpolation=cv2.INTER_CUBIC)
                imgJoints = showHandJoints(imgIn, np.round(handJointProj).astype(np.int32)[jointsMapManoToObman]*dscale,
                                           estIn=None, filename=None, upscale=1, lineThickness=2)

                objCorners = getObjectCorners(objMesh.v)
                rotObjNp = rotObj.eval(feed_dict={loadData: False})[f]
                transObjNp = transObj.eval(feed_dict={loadData: False})[f]
                objCornersTrans = np.matmul(objCorners, cv2.Rodrigues(rotObjNp)[0].T) + transObjNp
                objCornersProj = cv2.projectPoints(objCornersTrans.dot(coordChangMat), np.zeros((3,)), np.zeros((3,)), camMat, np.zeros((4,)))[0][:,0, :]
                imgJoints = showObjJoints(imgJoints, objCornersProj*dscale, lineThickness=2)

                mask = np.sum(segRen[f],2)>0
                mask = np.stack([mask, mask, mask], axis=2)

                alpha = 0.35
                rendMask = segRen[f]
                # rendMask[:,:,[1,2]] = 0
                rendMask = np.clip(255. * rendMask, 0, 255).astype('uint8')
                msk = rendMask.sum(axis=2) > 0
                msk = msk * alpha
                msk = np.stack([msk, msk, msk], axis=2)
                blended = msk * rendMask[:,:,[2,1,0]] + (1. - msk) * (colGT[f][:, :, [2, 1, 0]] * 255).astype(np.uint8)
                blended = blended.astype(np.uint8)
                cv2.imwrite(os.path.join(HO3D_MULTI_CAMERA_DIR, FLAGS.seq, 'dirt_grasp_pose', str(f) + '_blend.png'), imgJoints)


        rotObjNp = rotObj.eval(feed_dict={loadData: False})
        transObjNp = transObj.eval(feed_dict={loadData: False})

        if FLAGS.showFig:
            plt.savefig(out_dir + '/'+str(i)+'.png')
            plt.waitforbuttonpress(0.01)

        # dump loss plots intermittently
        if (i%25 == 0 or i == (numIter-1)) and (i>0):
            segLossAll = np.array(segLossList)
            depLossAll = np.array(depLossList)
            icpLossAll = np.array(icpLossList)
            relPoseLossAll = np.array(relPoseLossList)
            repulLossAll = np.array(repulLossList)
            if use2DJointLoss:
                joints2sLossAll = np.array(joints2DLossList)

            fig1 = plt.figure(2)
            plt.plot(np.arange(0, (len(segLossList))), segLossAll, 'r')
            fig1.savefig(out_dir + '/' + 'plotSeg_' + str(0) + '.png')
            plt.close(fig1)
            fig1 = plt.figure(2)
            plt.plot(np.arange(0, (len(depLossList))), depLossAll, 'g')
            fig1.savefig(out_dir + '/' + 'plotDep_' + str(0) + '.png')
            plt.close(fig1)
            fig1 = plt.figure(2)
            plt.plot(np.arange(0, (len(icpLossList))), icpLossAll, 'b')
            fig1.savefig(out_dir + '/' + 'plotIcp_' + str(0) + '.png')
            plt.close(fig1)
            fig1 = plt.figure(2)
            plt.plot(np.arange(0, (len(relPoseLossList))), relPoseLossAll, 'b')
            fig1.savefig(out_dir + '/' + 'plotRelPose_' + str(0) + '.png')
            plt.close(fig1)
            fig1 = plt.figure(2)
            plt.plot(np.arange(0, (len(repulLossAll))), repulLossAll, 'b')
            fig1.savefig(out_dir + '/' + 'plotRepul_' + str(0) + '.png')
            plt.close(fig1)
            if use2DJointLoss:
                fig1 = plt.figure(2)
                plt.plot(np.arange(0, (len(joints2sLossAll))), joints2sLossAll, 'b')
                fig1.savefig(out_dir + '/' + 'plotJoints2D_' + str(0) + '.png')
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
        savePickleData(out_dir + '/' + 'graspPose'+'.pkl', {'beta':betaNp, 'fullpose': thetaNp, 'trans': transNp,
                                                        'rotObj':rotObjNp, 'transObj': transObjNp,
                                                        'JTransformed': JTransformed,
                                                        'frameID': frameIDList})#, 'JVis': vis})

    finalHandVert = scene.itemPropsDict[handID].transformedMesh.v.eval(feed_dict={loadData: False})
    handFace = scene.itemPropsDict[handID].transformedMesh.f
    finalObjVert = scene.itemPropsDict[objID].transformedMesh.v.eval(feed_dict={loadData: False})
    objFace = scene.itemPropsDict[objID].transformedMesh.f
    finalHandMesh = o3d.geometry.TriangleMesh()
    finalHandMesh.vertices = o3d.utility.Vector3dVector(finalHandVert[0][:,:3])
    finalHandMesh.triangles = o3d.utility.Vector3iVector(handFace)
    finalHandMesh.vertex_colors = o3d.utility.Vector3dVector(np.reshape(np.random.uniform(0., 1., finalHandVert.shape[1]*3), (finalHandVert.shape[1],3)))
    finalObjMesh = o3d.geometry.TriangleMesh()
    finalObjMesh.vertices = o3d.utility.Vector3dVector(finalObjVert[0][:,:3])
    finalObjMesh.triangles = o3d.utility.Vector3iVector(objFace)
    finalObjMesh.vertex_colors = o3d.utility.Vector3dVector(
        np.reshape(np.random.uniform(0., 1., finalObjVert.shape[1] * 3), (finalObjVert.shape[1], 3)))

    o3d.io.write_triangle_mesh(out_dir+'/'+'hand.ply', finalHandMesh)
    o3d.io.write_triangle_mesh(out_dir + '/' + 'object.ply', finalObjMesh)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D', width=640, height=480, left=0, top=0,
                      visible=True)  # use visible=True to visualize the point cloud
    vis.get_render_option().light_on = False
    vis.add_geometry(finalHandMesh)
    vis.add_geometry(finalObjMesh)
    vis.run()

    return


def dataGen(w, h):
    configFile = join(HO3D_MULTI_CAMERA_DIR, FLAGS.seq, 'configs/configHandPose.json')
    with open(configFile) as config_file:
        data = yaml.safe_load(config_file)

    base_dir = os.path.join(HO3D_MULTI_CAMERA_DIR, FLAGS.seq)
    objAliasLabelList = []
    objLabel = data['objLabel']

    if 'camID' in data:
        camID = data['camID']
    else:
        camID = None

    camera_dir = os.path.join(base_dir, 'camera')
    # raw_seg_folder = os.path.join(base_dir, 'hand_only', 'mask')
    raw_seg_folder = os.path.join(base_dir, 'segmentation', 'raw_seg_results')





    handSegDir = os.path.join(HO3D_MULTI_CAMERA_DIR, FLAGS.seq, 'segmentation', FLAGS.camID, 'raw_seg_results')

    if useAutoInit:
        if datasetName == datasetType.HO3D_MULTICAMERA:
            autoInitDir = join(base_dir, 'handInit', camID, 'globalFit')
        elif datasetName == datasetType.HO3D:
            autoInitDir = join(base_dir, 'handInit', 'globalFit')
        else:
            raise NotImplementedError

        files = os.listdir(autoInitDir)
        files = [f[:-7] for f in files if 'pickle' in f]

        fileTuple = {f: np.sum(cv2.imread(join(base_dir, 'segmentation', camID, 'raw_seg_results', f + '.png'))==objLabel) for f in files}
        sorted_files = sorted(fileTuple.items(), key=lambda kv: kv[1], reverse=True)
        files = [f[0] for f in sorted_files]

        # files = sorted(files)
        files = files[:numAutoInitFiles]
        files = files


    if datasetName == datasetType.HO3D_MULTICAMERA:
        dataset = datasetHo3dMultiCamera(FLAGS.seq, FLAGS.camID)
    # w = 640 // dscale
    # h = 480 // dscale
    frameIDList = []
    handObjMaskList = []
    handObjDepthList = []
    handObjImgList = []
    maskList = []

    for file in files:
        if datasetName == datasetType.HO3D:
            imgName = os.path.join(camera_dir, 'color_' + file)
            dptName = os.path.join(camera_dir, 'depth_' + file)
            segName = os.path.join(raw_seg_folder, 'color_' + file)
            segHandName = os.path.join(raw_seg_folder, 'color_' + file)

            img = cv2.imread(imgName)[:, :, [2,1,0]]
            dpt = cv2.imread(dptName)[:, :, [2,1,0]]
            seg = cv2.imread(segName)[:, :, 0]
            segH = cv2.imread(segHandName)[:, :, 0]
            frameIDList.append(file[:-4])
        elif datasetName == datasetType.HO3D_MULTICAMERA:
            seq = base_dir.split('/')[-1]
            _, ds = dataset.createTFExample(itemType='hand', fileIn=join(seq, camID, file))
            img = ds.imgRaw[:, :, [2, 1, 0]]
            dpt = ds.depth
            # reading the seg from file because the seg in ds has no objAliasLabel info
            seg = cv2.imread(join(base_dir, 'segmentation', camID, 'raw_seg_results', file + '.png'))[:, :, 0]
            segH = cv2.imread(join(handSegDir, file + '.png'))[:,:,0]
            assert seg.shape == segH.shape
            segH[segH == 255] = handLabel
            frameIDList.append(file)
        else:
            raise NotImplementedError



        dpt = dpt[:, :, 0] + dpt[:, :, 1] * 256
        dpt = dpt * depthScale

        dptMask = np.logical_or(dpt > 0.75, dpt == 0.0)
        dpt[dptMask] = bgDepth

        seg[dptMask] = 0
        segH[dptMask] = 0
        seg[segH == handLabel] = handLabel  # because object segmetation is not accurate always, might say hand is part of object
        for alias in objAliasLabelList:
            seg[seg == alias] = objLabel

        objMask = (seg == objLabel)
        handMask = (segH == handLabel)
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

        handObjImg = handObjImg.astype(np.float32)/255.#np.zeros_like(mask, dtype=np.float32)
        handObjImgList.append(handObjImg)

        # realObservs = observables(frameID=file[:-4], seg=handMask, depth=handDepth, col=None, mask=mask, isReal=True)

    yield (np.stack(frameIDList,0), np.stack(handObjMaskList,0), np.stack(handObjDepthList,0), np.stack(handObjImgList,0), np.stack(maskList,0))


if __name__ == '__main__':

    plt.ion()

    configFile = join(HO3D_MULTI_CAMERA_DIR, FLAGS.seq, 'configs/configHandPose.json')


    with open(configFile) as config_file:
        data = yaml.safe_load(config_file)
    base_dir = os.path.join(HO3D_MULTI_CAMERA_DIR, FLAGS.seq)
    objLabel = data['objLabel']
    out_dir = os.path.join(base_dir, 'dirt_grasp_pose')
    camID = data['camID']

    betaFileName = data['betaFileName']
    assert os.path.exists(betaFileName), 'The provided beta parameters file %s does not exist'%(betaFileName)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    modelPath = os.path.join(YCB_MODELS_DIR, data['obj'])
    mesh = load_mesh(modelPath)

    objParamInitList = []
    handParamInitList = []



    beta = loadPickleData(betaFileName)['beta'][0]

    if useAutoInit:
        if datasetName == datasetType.HO3D_MULTICAMERA:
            autoInitDir = join(base_dir, 'handInit', camID, 'globalFit')
        else:
            raise NotImplementedError

        files = os.listdir(autoInitDir)
        files = [f[:-7]+'.png' for f in files if 'pickle' in f]

        fileTuple = {f: np.sum(cv2.imread(join(base_dir, 'segmentation', camID, 'raw_seg_results', f))[:,:,0] == objLabel)
                     for f in files}
        sorted_files = sorted(fileTuple.items(), key=lambda kv: kv[1], reverse=True)
        files = [f[0] for f in sorted_files]

        # files = sorted(files)
        files = files[:numAutoInitFiles]
        files = files #+ ['00538.png', '00569.png', '00587.png', '00598.png',]# '00433.png', '00469.png']
        print(files)

        anchorFile = files[0][:-4]
        newInit = loadPickleData(join(autoInitDir, anchorFile+'.pickle'))
        transConstr = newInit['trans']
        fullposeConstr = convertPosecoeffToFullposeNp(newInit['poseCoeff'][3:])
        fullposeConstr = np.concatenate([newInit['poseCoeff'][:3], fullposeConstr], axis=0)
    else:
        anchorFile = data['inputFiles'][0][:4]
        # get the rot and trans of hand in anchor frame in opengl coords
        newInit = loadPickleData(base_dir + '/configs/render_' + anchorFile + '.pickle')
        # pts3D = forwardKinematics(newInit['fullpose'], newInit['trans'], newInit['beta'])
        # fullposeConstr, transConstr, _ = inverseKinematicCh(pts3D, newInit['fullpose'], newInit['trans'], newInit['beta'])
        fullposeConstr, transConstr = newInit['fullpose'], newInit['trans']
        # beta = undo_chumpy(newInit['beta'])

    newFullposeList = []
    for i in range(16):
        if i == 0:
            newRot, newTrans = convertPoseOpenDR_DIRT(undo_chumpy(fullposeConstr[3 * i:3 * (i + 1)]),
                                                      undo_chumpy(transConstr))
        else:
            newRot = undo_chumpy(fullposeConstr[3 * i:3 * (i + 1)])
        newFullposeList.append(newRot)
    newFullpose = np.concatenate(newFullposeList, axis=0)
    handParamAnchorFrame = handParams(theta=newFullpose, trans=newTrans, beta=beta)


    if OBJ_POSE_FROM_DIRT:
        objInit = loadPickleData(os.path.join(base_dir, 'dirt_obj_pose', camID, anchorFile + '.pkl'))
        newObjRot = undo_chumpy(objInit['rot'])[0]
        newObjTrans = undo_chumpy(objInit['trans'])[0]
    else:
        raise Exception('Not supported!')
    objParamAnchorFrame = objParams(rot=newObjRot, trans=newObjTrans)

    # get relative pose
    handParamRel = getHORelPose(handParamAnchorFrame, objParamAnchorFrame)
    for i in range(len(files)):
        handParamInitList.append(deepcopy(handParamRel))
        if use2DJointLoss and useAutoInit:
            if datasetName == datasetType.HO3D_MULTICAMERA:
                joints2dDir = join(base_dir, 'handInit', camID, 'singleFrameFit')
                if os.path.exists(join(joints2dDir, files[i][:-4] + '.pickle')):
                    pklData = loadPickleData(join(joints2dDir, files[i][:-4] + '.pickle'))
                    handParamInitList[i].JTransformed = pklData['KPS3D'][jointsMapObmanToMano]
                    handParamInitList[i].JVis = (pklData['conf'][jointsMapObmanToMano])
                else:
                    handParamInitList[i].JTransformed = np.zeros((21,3), dtype=np.float32) + 1.0
                    handParamInitList[i].JVis = np.zeros((21,), dtype=np.float32)
            else:
                raise NotImplementedError

    objParamInitList = []
    # get pose of obj
    for file in files:
        if False:
            initDict = loadPickleData(base_dir + '/obj_pose/render_' + file[:4] + '.pickle')
            rotTransp = cv2.Rodrigues(cv2.Rodrigues(undo_chumpy(initDict['r']))[0].T)[0][:, 0]
            rot, trans = convertPoseOpenDR_DIRT(rotTransp, undo_chumpy(initDict['t']))
            paramInit = objParams(rot=rot, trans=trans)
        else:
            initDict = loadPickleData(os.path.join(base_dir, 'dirt_obj_pose', camID, file[:-4] + '.pkl'))
            paramInit = objParams(rot=initDict['rot'][0], trans=initDict['trans'][0])
        objParamInitList.append(paramInit)




    if datasetName == datasetType.HO3D_MULTICAMERA:
        camMat = datasetHo3dMultiCamera.getCamMat(FLAGS.seq)
        camProp = camProps(ID='cam1', f=np.array([camMat[0,0], camMat[1,1]], dtype=np.float32) / dscale,
                           c=np.array([camMat[0,2], camMat[1,2]], dtype=np.float32) / dscale,
                           near=0.001, far=2.0, frameSize=[w, h],
                           pose=np.eye(4, dtype=np.float32))
    else:
        raise NotImplementedError

    handPoseMF(w, h, objParamInitList, handParamInitList, mesh, camProp, out_dir)


    if False:
        myGen = dataGen(w, h)
        frameID, handMask, handDepth, col, mask = next(myGen)
        a = 10



