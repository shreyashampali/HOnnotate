from HOdatasets.commonDS import *
import models.deeplab.common as common
from eval import evalSeg
from eval import eval2DKps
from onlineAug.commonAug import networkData
from utils import inferenceUtils as infUti
import pickle
from utils.predict2DKpsObject import getNetSess
from HOdatasets.mypaths import *
import warnings
import tensorflow as tf
import time
import multiprocessing as mlp
from utils.pnp import PNP
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
os.environ["CUDA_VISIBLE_DEVICES"]='0'

from absl import flags
from absl import app
FLAGS = flags.FLAGS

flags.DEFINE_string('seq', 'releaseTest', 'Sequence Name')
flags.DEFINE_string('objectName', '035_power_drill', 'YCB name of object')
flags.DEFINE_string('camID', '0', 'Cam ID')
configDir = 'CPMObj'
baseDir = HO3D_MULTI_CAMERA_DIR

dataset_mix = infUti.datasetMix.HO3D_MULTICAMERA
w = 224
h = 224
LIFT2DTO3D = False
numConsThreads = 1
itemType='object'
numKps = 8

def postProcess(dummy, consQueue, numImgs, numConsThreads):
    while True:
        queueElem = consQueue.get()
        predsDict = queueElem[0]
        ds = queueElem[1]
        jobID = queueElem[2]

        global kps2DPredsSM, confidenceSM
        global savePatchDir, saveKps2DDir, saveResultsDir

        kps2DPreds = np.frombuffer(kps2DPredsSM, dtype=np.float32).reshape([numImgs, numKps, 2])

        confidence = np.frombuffer(confidenceSM, dtype=np.float32).reshape([numImgs, numKps])

        seq = ds.fileName.split('/')[0]
        camInd = ds.fileName.split('/')[1]
        id = ds.fileName.split('/')[2]

        if common.IMAGE in predsDict.keys():
            # dump the network input patch
            croppedImg = predsDict[common.IMAGE]
            if len(ds.fileName.split('/')) == 3:
                if not os.path.exists(os.path.join(savePatchDir, camInd)):
                    os.mkdir(os.path.join(savePatchDir, camInd))
                finalSaveDir = os.path.join(savePatchDir, camInd)
            else:
                raise NotImplementedError
            evalSeg.dump(croppedImg, finalSaveDir, ds.fileName.split('/')[-1], add_colormap=False)

        if common.KPS_2D in predsDict.keys():
            # dump the visualization of predicted kps
            predsDict[common.KPS_2D] = predsDict[common.KPS_2D][0]

            kps2DPreds[jobID, :, 0] = predsDict[common.KPS_2D + '_loc'][0, :, 1]
            kps2DPreds[jobID, :, 1] = predsDict[common.KPS_2D + '_loc'][0, :, 0]

            # get kps wrt full image.
            kps2DPreds[jobID] = eval2DKps.getKpsWrtImage(kps2DPreds[jobID], predsDict['topLeft'],
                                                     predsDict['bottomRight'], h, w)


            corners = np.load(join('objCorners', FLAGS.objectName, 'corners.npy'))
            poses = PNP(kps2DPreds[jobID], corners, ds.camMat).pnp_ransac()
            poses = np.expand_dims(poses, 0)
            cornersTrans = corners.dot(poses[0][:, :3].T) + poses[0][:, 3].reshape(1, 3)
            cornerProj, _ = cv2.projectPoints(cornersTrans, np.zeros((3,)), np.zeros((3,)), ds.camMat,
                                                      np.zeros((4,)))
            # kps2DPreds[jobID] = cornerProj[:,0,:]


            if len(ds.fileName.split('/')) == 3:
                if not os.path.exists(os.path.join(saveKps2DDir, camInd)):
                    os.mkdir(os.path.join(saveKps2DDir, camInd))
            else:
                raise NotImplementedError

            # visualize with connected lines
            eval2DKps.vis2DKps(kps2DPreds[jobID],
                               None,
                               ds.imgRaw,
                               saveKps2DDir, camInd+'/'+id)


            maxValsIndu = np.max(
                np.reshape(predsDict[common.KPS_2D], [-1, predsDict[common.KPS_2D].shape[-1]]), axis=0)



            confidence[jobID] = maxValsIndu

            # save the results
            if len(ds.fileName.split('/')) == 3:
                if not os.path.exists(os.path.join(saveResultsDir, camInd)):
                    os.mkdir(os.path.join(saveResultsDir, camInd))
                finalResultsDir = os.path.join(saveResultsDir, camInd)
            else:
                raise NotImplementedError

            coordChangMat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
            poseMatOpenCV = poses[0]
            rotOpenGL = cv2.Rodrigues(coordChangMat.dot(poseMatOpenCV[:3,:3]))[0][:, 0]
            transOpenGL = poseMatOpenCV[:3,3].dot(coordChangMat.T)

            with open(os.path.join(finalResultsDir, id+'.pickle'), 'wb') as f:
                pickle.dump({'KPS2D': kps2DPreds[jobID], 'conf': confidence[jobID], 'imgID': ds.fileName,
                             'poseMatOpenCV': poseMatOpenCV,
                             'rotOpenGL':rotOpenGL,
                             'transOpenGL': transOpenGL,
                             }, f)

        print('Frame %d of %d (%s)' % (jobID, numImgs, ds.fileName))
        if jobID>=(numImgs-numConsThreads):
            return


def runNetInLoop(fileListIn, numImgs):
    myG = tf.Graph()

    with myG.as_default():
        data = networkData(image=tf.placeholder(tf.uint8, shape=(None, None, 3)),
                           label=tf.placeholder(tf.uint8, shape=(None, None, 1)),
                           kps2D=tf.placeholder(tf.float32, (numKps, 3)),
                           kps3D=tf.placeholder(tf.float32, (numKps, 3)),
                           imageID=None,
                           h=tf.placeholder(tf.int32),
                           w=tf.placeholder(tf.int32),
                           outType=tf.placeholder(tf.int32),
                           dsName=tf.placeholder(tf.int32),
                           camMat=tf.placeholder(tf.float32))

        cropCenter = tf.placeholder(tf.float32, (2,))
        cropPatchSize = tf.placeholder(tf.float32, (2,))

    sess, g, predictions, dataPreProcDict, topLeft, bottomRight, extraScale = getNetSess(data, h, w, myG,
                                                                                         cropCenter=cropCenter,
                                                                                         cropPatchSize=cropPatchSize,)




    dsQueue, dsProcs = infUti.startInputQueueRunners(dataset_mix, splitType.TEST, numThreads=1, itemType=itemType, fileListIn=fileListIn)


    # start consumer threads
    consQueue = mlp.Queue(maxsize=100)
    procs = []
    for proc_index in range(numConsThreads):
        args = ([], consQueue, numImgs, numConsThreads)
        proc = mlp.Process(target=postProcess, args=args)

        proc.start()
        procs.append(proc)


    # start the network
    for i in range(numImgs):

        while(dsQueue.empty()):
            waitTime = 10*1e-3
            time.sleep(waitTime)

        ds = dsQueue.get()

        assert isinstance(ds, dataSample)



        predictions['topLeft'] = topLeft
        predictions['bottomRight'] = bottomRight
        predictions['extraScale'] = extraScale
        predictions[common.LABEL+'_GT'] = dataPreProcDict[common.LABEL]
        predictions[common.KPS_2D+'_GT'] = dataPreProcDict[common.KPS_2D]
        predictions[common.IMAGE] = dataPreProcDict[common.IMAGE]

        objIndex = 1
        rows, cols = np.where(ds.segRaw == objIndex)
        if rows.size == 0 or cols.size == 0:
            print('%s has no Object' % (ds.fileName))
            consQueue.put([{}, ds, i])
            continue
        tl_x, tl_y = np.min(cols), np.min(rows)
        br_x, br_y = np.max(cols), np.max(rows)

        # get the patch center and size from segmentations
        center = np.array([(tl_x+br_x) / 2., (tl_y+br_y) / 2.])
        cropPatchSizeNp = np.array([br_x-tl_x, br_y-tl_y])*2.0

        coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        predsDict = sess.run(predictions, feed_dict={data.image: cv2.resize(ds.imgRaw, (w,h), interpolation=cv2.INTER_CUBIC),
                                                     data.label: np.expand_dims(ds.segRaw, 2),
                                                     data.kps2D: ds.pts2D,
                                                     data.kps3D: ds.pts3D.dot(coordChangeMat),
                                                     data.outputType: ds.outputType,
                                                     data.datasetName: ds.dataset,
                                                     data.height: ds.height,
                                                     data.width: ds.width,
                                                     data.camMat: ds.camMat,
                                                     cropCenter: center,
                                                     cropPatchSize: cropPatchSizeNp},)



        consQueue.put([predsDict, ds, i])

    for proc in procs:
        proc.join()

    while(not consQueue.empty()):
        time.sleep(10*1e-3)

    consQueue.close()
    dsQueue.close()


def main(argv):
    global kps2DPredsSM, confidenceSM
    global savePatchDir, saveKps2DDir, saveResultsDir

    savePatchDir = os.path.join(baseDir, FLAGS.seq, configDir, 'patch')
    saveKps2DDir = os.path.join(baseDir, FLAGS.seq, configDir, 'KPS2DStick')
    saveResultsDir = os.path.join(baseDir, FLAGS.seq, configDir, 'Results')

    savePatchDir = savePatchDir + '_' + itemType
    saveKps2DDir = saveKps2DDir + '_' + itemType
    saveResultsDir = saveResultsDir + '_' + itemType

    if not os.path.exists(os.path.join(baseDir, FLAGS.seq, configDir)):
        os.mkdir(os.path.join(baseDir, FLAGS.seq, configDir))
    if not os.path.exists(savePatchDir):
        os.mkdir(savePatchDir)
    if not os.path.exists(saveKps2DDir):
        os.mkdir(saveKps2DDir)
    if not os.path.exists(saveResultsDir):
        os.mkdir(saveResultsDir)


    fileListIn = os.listdir(join(HO3D_MULTI_CAMERA_DIR, FLAGS.seq, 'rgb', '0'))
    fileListIn = [join(FLAGS.seq, '0', f[:-4]) for f in fileListIn if 'png' in f]
    fileListIn = sorted(fileListIn)

    numImgs = len(fileListIn)

    kps2DPredsL = np.zeros((numImgs, numKps, 2), dtype=np.float32)
    kps2DPredsSM = mlp.RawArray('f', kps2DPredsL.size)
    confidenceSM = mlp.RawArray('f', kps2DPredsL.size // 2)

    runNetInLoop(fileListIn, numImgs)


if __name__ == '__main__':
    app.run(main)
