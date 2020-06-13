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
from ghope.vis import renderScene

from HOdatasets.ho3d_multicamera.dataset import datasetHo3dMultiCamera
from HOdatasets.commonDS import datasetType, splitType
from HOdatasets.mypaths import *

import argparse, yaml
from ext.mesh_loaders import *
import matplotlib.pyplot as plt

depthScale = 0.00012498664727900177 # constant depth scale used throughout the project
bgDepth = 2.0
handLabel = 2

datasetName = datasetType.HO3D_MULTICAMERA

from absl import flags
from absl import app
import argparse
parser = argparse.ArgumentParser()
FLAGS = flags.FLAGS

parser.add_argument('--seq', default='0010', help='Sequence Name')
parser.add_argument('--camID', default='0', help='Sequence Name')
parser.add_argument('--doPyRender', action='store_true', help='Show object rendering, very slow!')
FLAGS = parser.parse_args()

USE_PYTHON_RENDERER = FLAGS.doPyRender # for visualization, but slows down

def objectTracker(w, h, paramInit, camProp, objMesh, out_dir, configData):
    '''
    Generative object tracking
    :param w: width of the image
    :param h: height of the image
    :param paramInit: object of objParams class
    :param camProp: camera properties object
    :param objMesh: object mesh
    :param out_dir: out directory
    :return:
    '''
    ds = tf.data.Dataset.from_generator(lambda: dataGen(w, h, datasetName),
                                        (tf.string, tf.float32, tf.float32, tf.float32, tf.float32),
                                        ((None,), (None, h, w, 3), (None, h, w, 3), (None, h, w, 3), (None, h, w, 3)))
    numFrames = 1

    # read real observations
    frameCntInt, loadData, realObservs = LossObservs.getRealObservables(ds, numFrames, w, h)
    icp = Icp(realObservs, camProp)

    # set up the scene
    scene = Scene(optModeEnum.MULTIFRAME_JOINT, frameCnt=1)
    objID = scene.addObject(objMesh, paramInit, segColor=np.array([1.,1.,1.]))
    scene.addCamera(f=camProp.f, c=camProp.c, near=camProp.near, far=camProp.far, frameSize=camProp.frameSize)
    finalMesh = scene.getFinalMesh()

    # render the scene
    renderer = DirtRenderer(finalMesh, renderModeEnum.SEG_DEPTH)
    virtObservs = renderer.render()

    # get loss over observables
    observLoss = LossObservs(virtObservs, realObservs, renderModeEnum.SEG_DEPTH)
    segLoss, depthLoss, _ = observLoss.getL2Loss(isClipDepthLoss=True, pyrLevel=2)

    # get constraints
    handConstrs = Constraints()
    paramList = scene.getParamsByItemID([parTypeEnum.OBJ_ROT, parTypeEnum.OBJ_TRANS, parTypeEnum.OBJ_POSE_MAT], objID)
    rot = paramList[0]
    trans = paramList[1]
    poseMat = paramList[2]

    # get icp loss
    icpLoss = icp.getLoss(finalMesh.vUnClipped)

    # get final loss
    objImg = (realObservs.col)
    # totalLoss1 = 1.0*segLoss + 1e1*depthLoss + 1e4*icpLoss + 0.0*tf.reduce_sum(objImg-virtObservs.seg)
    totalLoss1 = 1.0e0 * segLoss  + 1e1 * depthLoss  + 1e2 * icpLoss + 0.0 * tf.reduce_sum(objImg - virtObservs.seg)
    totalLoss2 = 1.15 * segLoss + 5.0 * depthLoss + 500.0*icpLoss

    # get the variables for opt
    optVarsList = scene.getVarsByItemID(objID, [varTypeEnum.OBJ_ROT, varTypeEnum.OBJ_TRANS])

    # setup optimizer
    opti1 = Optimizer(totalLoss1, optVarsList, 'Adam', learning_rate=0.02/2.0)
    opti2 = Optimizer(totalLoss2, optVarsList, 'Adam', learning_rate=0.005)
    optiICP = Optimizer(1e1*icpLoss, optVarsList, 'Adam', learning_rate=0.01)

    # get the optimization reset ops
    resetOpt1 = tf.variables_initializer(opti1.optimizer.variables())
    resetOpt2 = tf.variables_initializer(opti2.optimizer.variables())


    # tf stuff
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    session = tf.Session(config=config)
    session.__enter__()
    tf.global_variables_initializer().run()

    # python renderer for rendering object texture
    pyRend = renderScene(h, w)
    modelPath = os.path.join(YCB_MODELS_DIR, configData['obj'])
    pyRend.addObjectFromMeshFile(modelPath, 'obj')
    pyRend.addCamera()
    pyRend.creatcamProjMat(camProp.f, camProp.c, camProp.near, camProp.far)

    # setup the plot window
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    lGT = ax1.imshow(np.zeros((240,320,3), dtype=np.float32))
    ax2 = fig.add_subplot(2, 2, 2)
    lRen = ax2.imshow(np.zeros((240, 320, 3), dtype=np.float32))
    ax3 = fig.add_subplot(2, 2, 3)
    lDep = ax3.imshow(np.random.uniform(0,2,(240,320)))
    ax4 = fig.add_subplot(2, 2, 4)
    lMask = ax4.imshow(np.random.uniform(0,2,(240,320,3)))



    while(True):
        session.run(resetOpt1)
        session.run(resetOpt2)

        # load new frame
        opti1.runOptimization(session, 1, {loadData:True})
        print(icpLoss.eval(feed_dict={loadData: False}))
        print(segLoss.eval(feed_dict={loadData: False}))
        print(depthLoss.eval(feed_dict={loadData: False}))

        # run the optimization for new frame
        frameID = (realObservs.frameID.eval(feed_dict={loadData: False}))[0].decode('UTF-8')
        # opti1.runOptimization(session, 200, {loadData: False})#, logLossFunc=True, lossPlotName=out_dir+'/LossFunc/'+frameID+'_1.png')
        opti2.runOptimization(session, 25, {loadData: False})#, logLossFunc=True, lossPlotName='handLoss/'+frameID+'_2.png')

        pyRend.setObjectPose('obj',poseMat.eval(feed_dict={loadData: False})[0].T)
        if USE_PYTHON_RENDERER:
            cRend, dRend = pyRend.render()


        plt.title(frameID)
        depRen = virtObservs.depth.eval(feed_dict={loadData: False})[0]
        depGT = realObservs.depth.eval(feed_dict={loadData: False})[0]
        segRen = virtObservs.seg.eval(feed_dict={loadData: False})[0]
        segGT = realObservs.seg.eval(feed_dict={loadData: False})[0]

        lGT.set_data(objImg.eval(feed_dict={loadData: False})[0]) # input image
        if USE_PYTHON_RENDERER:
            lRen.set_data(cRend) # object rendered in the optimized pose
        lDep.set_data(np.abs(depRen-depGT)[:,:,0]) # depth map error
        lMask.set_data(np.abs(segRen-segGT)[:,:,:]) # mask error
        plt.savefig(out_dir+'/'+frameID+'.png')
        plt.waitforbuttonpress(0.01)

        transNp = trans.eval(feed_dict={loadData: False})
        rotNp = rot.eval(feed_dict={loadData: False})
        savePickleData(out_dir+'/'+frameID+'.pkl', {'rot': rotNp, 'trans': transNp})

def dataGen(w, h, datasetName):
    '''
        Generator which provides rgb, depth and segmentation data for each frame
    '''
    configFile = join(HO3D_MULTI_CAMERA_DIR, FLAGS.seq, 'configs/configObjPose.json')

    # read the config file
    with open(configFile) as config_file:
        data = yaml.safe_load(config_file)

    base_dir = os.path.join(HO3D_MULTI_CAMERA_DIR, FLAGS.seq)
    obj = data['obj']
    objAliasLabelList = []
    objLabel = data['objLabel']
    startAt = data['startAt']
    endAt = data['endAt']
    skip = data['skip']


    # set some paths
    modelPath = os.path.join(YCB_MODELS_DIR, obj)

    plt.ion()

    # get list of filenames
    if datasetName == datasetType.HO3D_MULTICAMERA:
        fileListIn = os.listdir(join(HO3D_MULTI_CAMERA_DIR, FLAGS.seq, 'rgb', '0'))
        fileListIn = [join(FLAGS.seq, '0', f[:-4]) for f in fileListIn if 'png' in f]
        fileListIn = sorted(fileListIn)
        dataset = datasetHo3dMultiCamera(FLAGS.seq, 0, fileListIn=fileListIn)
        files = dataset.fileList
        if isinstance(startAt, str):
            for i, f in enumerate(files):
                if f.split('/')[-1] == startAt:
                    break
            files = files[i::skip]
            # print(files)
        else:
            files = files[startAt:endAt + 1:skip]
    else:
        raise Exception('Unsupported datasetName')

    for i, file in enumerate(files):

        # read RGB. depth and segmentations for current fileID
        seq = file.split('/')[0]
        camInd = file.split('/')[1]
        id = file.split('/')[2]
        _, ds = dataset.createTFExample(itemType='hand', fileIn=file)
        img = ds.imgRaw[:,:,[2,1,0]]
        dpt = ds.depth
        # reading the seg from file because the seg in ds has no objAliasLabel info
        seg = cv2.imread(join(HO3D_MULTI_CAMERA_DIR, seq, 'segmentation', camInd, 'raw_seg_results', id+'.png'))[:, :, 0]
        frameID = np.array([join(camInd, id)])


        # decode depth map
        dpt = dpt[:, :, 0] + dpt[:, :, 1] * 256
        dpt = dpt * depthScale

        # clean up depth map
        dptMask = np.logical_or(dpt > 0.7, dpt == 0.0)
        dpt[dptMask] = bgDepth

        # clean up seg map
        seg[dptMask] = 0
        for alias in objAliasLabelList:
            seg[seg == alias] = objLabel

        # Extract the object image in the image using the mask
        objMask = (seg == objLabel)
        handMask = (seg == handLabel)
        objImg = img * np.expand_dims(objMask, 2)
        handImg = img * np.expand_dims(handMask, 2)
        # plt.imshow(seg)
        # plt.show()

        # Extract the object depth in the depth map using the mask
        objDepth = dpt * objMask
        objDepth[np.logical_not(objMask)] = bgDepth
        handDepth = dpt * handMask
        handDepth[np.logical_not(handMask)] = bgDepth

        # skip the file if object is too small (occluded heavily)
        maskPc = float(np.sum(objMask)) / float((objMask.shape[0] * objMask.shape[0]))
        print('maskPC for Image %s is %f' % (file, maskPc))
        if maskPc < .005:
            continue

        # resizing
        objDepth = cv2.resize(objDepth, (w, h), interpolation=cv2.INTER_NEAREST)
        objMask = cv2.resize(objMask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        handMask = cv2.resize(handMask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        objImg = cv2.resize(objImg.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)

        # minor changes to the dimensions as required by TF
        objMask = np.stack([objMask, objMask, objMask], axis=2)
        objMask = np.expand_dims(objMask, 0).astype(np.float32)
        objDepth = np.stack([objDepth, objDepth, objDepth], axis=2)
        objDepth = np.expand_dims(objDepth, 0)
        handMask = np.stack([handMask, handMask, handMask], axis=2)
        handMask = np.expand_dims(handMask, 0).astype(np.float32)
        objImg = np.expand_dims(objImg, 0).astype(np.float32) / 255.  # np.zeros_like(mask, dtype=np.float32)

        yield (frameID, objMask, objDepth, objImg, handMask)

if __name__ == '__main__':

    dscale = 2
    w = 640 // dscale
    h = 480 // dscale
    plt.ion()

    # read the config file
    configFile = join(HO3D_MULTI_CAMERA_DIR, FLAGS.seq, 'configs/configObjPose.json')
    with open(configFile) as config_file:
        configData = yaml.safe_load(config_file)
    base_dir = os.path.join(HO3D_MULTI_CAMERA_DIR, FLAGS.seq)
    out_dir = os.path.join(base_dir, 'dirt_obj_pose')

    # create out dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # create more dir if required
    if datasetName == datasetType.HO3D_MULTICAMERA:
        for i in range(1):
            if not os.path.exists(join(out_dir, str(i))):
                os.mkdir(join(out_dir, str(i)))


    # initialization for rot and trans
    rot = np.array(configData['rotationInit'])
    trans = np.array(configData['translationInit'])

    modelPath = os.path.join(YCB_MODELS_DIR, configData['obj'])
    mesh = load_mesh(modelPath)

    # ready the arguments
    paramInit = objParams(rot=rot, trans=trans)
    if datasetName == datasetType.HO3D_MULTICAMERA:
        camMat = datasetHo3dMultiCamera.getCamMat(FLAGS.seq)
        camProp = camProps(ID='cam1', f=np.array([camMat[0,0], camMat[1,1]], dtype=np.float32) / dscale,
                           c=np.array([camMat[0,2], camMat[1,2]], dtype=np.float32) / dscale,
                           near=0.001, far=2.0, frameSize=[w, h],
                           pose=np.eye(4, dtype=np.float32))
    else:
        raise NotImplementedError

    # for debugging
    if False:
        myGen = dataGen(w, h, datasetName)
        frameID, handMask, handDepth, col, mask = next(myGen)

    objectTracker(w, h, paramInit, camProp, mesh, out_dir, configData)



