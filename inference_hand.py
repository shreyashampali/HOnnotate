import matplotlib.pyplot as plt
from HOdatasets.commonDS import *
import models.deeplab.common as common
from eval import evalSeg
from eval import eval2DKps
from onlineAug.commonAug import networkData
from utils import inferenceUtils as infUti
import pickle
from HOdatasets.mypaths import *
from utils.predict2DKpsHand import getNetSess
from absl import flags
from absl import app
import tensorflow as tf
import time
import multiprocessing as mlp
import warnings
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
os.environ["CUDA_VISIBLE_DEVICES"]='0'

flags.DEFINE_string('seq', 'releaseTest', 'Sequence Name') # name ,default, help
flags.DEFINE_string('camID', '0', 'Cam ID') # name ,default, help
FLAGS = flags.FLAGS

dataset_mix = infUti.datasetMix.HO3D_MULTICAMERA
w = 256
h = 256
numConsThreads = 1
configDir = 'CPMHand'
itemType='hand'
baseDir = HO3D_MULTI_CAMERA_DIR
numKps = 21


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

        if len(ds.fileName.split('/')) == 3:
            if not os.path.exists(os.path.join(saveResultsDir, camInd)):
                os.mkdir(os.path.join(saveResultsDir, camInd))
            finalResultsDir = os.path.join(saveResultsDir, camInd)
        else:
            raise NotImplementedError

        if common.IMAGE in predsDict.keys():
            # dump the network input patches
            croppedImg = predsDict[common.IMAGE]
            if len(ds.fileName.split('/')) == 3:
                if not os.path.exists(os.path.join(savePatchDir, camInd)):
                    os.mkdir(os.path.join(savePatchDir, camInd))
                finalSaveDir = os.path.join(savePatchDir, camInd)
            else:
                raise NotImplementedError
            # save input patch
            evalSeg.dump(croppedImg, finalSaveDir, ds.fileName.split('/')[-1], add_colormap=False)

        if common.KPS_2D in predsDict.keys():
            # dump the visualization of predicted kps
            predsDict[common.KPS_2D] = predsDict[common.KPS_2D][0]

            kps2DPreds[jobID, :, 0] = predsDict[common.KPS_2D + '_loc'][0, :, 1]
            kps2DPreds[jobID, :, 1] = predsDict[common.KPS_2D + '_loc'][0, :, 0]

            # get kps wrt full image.
            kps2DPreds[jobID] = eval2DKps.getKpsWrtImage(kps2DPreds[jobID], predsDict['topLeft'],
                                                     predsDict['bottomRight'], h, w)


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


            with open(os.path.join(finalResultsDir, id+'.pickle'), 'wb') as f:
                pickle.dump({'KPS2D': kps2DPreds[jobID], 'conf': confidence[jobID], 'imgID': ds.fileName,
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
                                                                                         cropPatchSize=cropPatchSize)




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
    isPrevFrameValid = False
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
        predictions[common.IMAGE] = dataPreProcDict[common.IMAGE]

        # track the BB
        handIndex = 2
        rows, cols = np.where(ds.segRaw == handIndex)
        if rows.size == 0 or cols.size == 0:
            print('%s has no Hand'%(ds.fileName))
            consQueue.put([{}, ds, i])
            isPrevFrameValid = False
            continue

        if not isPrevFrameValid:
            # use segmentations for bounding box when previous frame has no keypoints
            tl_x, tl_y = np.min(cols), np.min(rows)
            br_x, br_y = np.max(cols), np.max(rows)

            center = np.array([(tl_x + br_x) / 2., (tl_y + br_y) / 2.])
            cropPatchSizeNp = np.array([br_x-tl_x, br_y-tl_y])*1.1
        else:
            kpsPrevFrame = np.zeros((21,2), dtype=np.float32)
            kpsPrevFrame[:, 0] = predsDict[common.KPS_2D + '_loc'][0, :, 1]
            kpsPrevFrame[:, 1] = predsDict[common.KPS_2D + '_loc'][0, :, 0]
            kpsPrevFrame[:, :2] = eval2DKps.getKpsWrtImage(kpsPrevFrame, predsDict['topLeft'],
                                                       predsDict['bottomRight'], h, w)

            tl_x = np.min(kpsPrevFrame,axis=0)[0]
            tl_y = np.min(kpsPrevFrame,axis=0)[1]
            br_x = np.max(kpsPrevFrame, axis=0)[0]
            br_y = np.max(kpsPrevFrame, axis=0)[1]

            cropPatchSizeNp = np.array([br_x - tl_x, br_y - tl_y]) * 1.2
            center = np.array([(tl_x + br_x) / 2., (tl_y + br_y) / 2.])



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
        isPrevFrameValid = True

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
    # create empty arrays

    kps2DPredsL = np.zeros((numImgs, numKps, 2), dtype=np.float32)
    kps2DPredsSM = mlp.RawArray('f', kps2DPredsL.size)
    confidenceSM = mlp.RawArray('f', kps2DPredsL.size // 2)


    runNetInLoop(fileListIn, numImgs)


if __name__ == '__main__':
    app.run(main)
