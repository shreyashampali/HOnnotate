import os
from enum import IntEnum
import numpy as np
from multiprocessing import Process, Queue


from HOdatasets.ho3d_multicamera.dataset import datasetHo3dMultiCamera

from HOdatasets.mypaths import MANO_MODEL_PATH
# import commonAug
import pickle
import cv2

import warnings

warnings.simplefilter("ignore", ResourceWarning)

# dsQueue = queue.Queue(maxsize=200)

class datasetMix(IntEnum):
    '''
    Enum for different datatypes
    '''

    HO3D = 1

    MPII = 2

    HO3D_MPII = 3

    OBMAN = 4

    HO3D_CAMERA = 5

    HO3D_MTC = 6

    MTC = 7

    HO3D_MULTICAMERA = 8

    FREIHANDS = 9

    SHALINI_DATASET = 10



def shardProc(dummy, shard_id, dataset_split, dataset_mix, numShards, dsQueue, itemType, isRemoveBG=False, fileListIn=None):

    if dataset_mix == datasetMix.HO3D_MULTICAMERA:
        dsCurr = datasetHo3dMultiCamera('', 0,  isRemoveBG=isRemoveBG, fileListIn=fileListIn)
    else:
        raise NotImplementedError

    num_images = dsCurr.getNumFiles()

    num_per_shard = int(np.ceil(float(num_images) / float(numShards)))

    start_idx = shard_id * num_per_shard
    end_idx = min((shard_id + 1) * num_per_shard, num_images)

    dsCurr.setStartEndFiles(start_idx, end_idx)

    print('Launching thread %d'%(shard_id))

    for i in range(end_idx-start_idx):
        _, ds = dsCurr.createTFExample(itemType=itemType)
        dsQueue.put(ds)



def startInputQueueRunners(dataset_mix, dataset_split, numThreads=10, queueSize=200, itemType='hand', isRemoveBG=False, fileListIn=None):
    dsQueue = Queue(maxsize=queueSize)
    procs = []
    for proc_index in range(numThreads):
        args = (
            [], proc_index, dataset_split, dataset_mix, numThreads, dsQueue, itemType, isRemoveBG, fileListIn)
        proc = Process(target=shardProc, args=args)
        # proc.daemon = True

        proc.start()
        procs.append(proc)

    # for proc in procs:
    #     proc.join()

    return dsQueue, procs



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


def get2DBoundingBoxCenterFromSegmap(seg, index):
    xx, yy = np.meshgrid(np.arange(0, seg.shape[1]), np.arange(0, seg.shape[0]))
    maskRescAll = (seg == index).astype(np.uint8)
    if np.sum(maskRescAll) > 0:
        xmean = np.round(np.sum(xx * maskRescAll) / np.sum(maskRescAll)).astype(np.uint32)
        ymean = np.round(np.sum(yy * maskRescAll) / np.sum(maskRescAll)).astype(np.uint32)
    else:
        xmean = seg.shape[1] / 2.
        ymean = seg.shape[0] / 2.

    return xmean, ymean

    if False:
        rows, cols = np.where(seg == index)
        tl_x, tl_y = np.min(cols), np.min(rows)
        br_x, br_y = np.max(cols), np.max(rows)
        bboxes = [tl_x, tl_y, br_x, br_y]

        return bboxes

def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else np.array(x.r)

def forwardKinematics(fullpose, trans, beta):
    '''
    MANO parameters --> 3D pts
    :param fullpose:
    :param trans:
    :param beta:
    :return: 3D pts of size (21,3)
    '''
    import chumpy as ch
    from manoCh.smpl_handpca_wrapper_HAND_only import load_model


    assert fullpose.shape == (48,)
    assert trans.shape == (3,)
    assert beta.shape == (10,)

    m = load_model(MANO_MODEL_PATH, ncomps=6, flat_hand_mean=True, optwrt='fullpose')
    m.fullpose[:] = undo_chumpy(fullpose)
    m.trans[:] = undo_chumpy(trans)
    m.betas[:] = undo_chumpy(beta)

    return undo_chumpy(m.J_transformed), m

def convertPosecoeffToFullposeNp(posecoeff, flat_hand_mean=False):
    from sklearn.preprocessing import normalize as normalize
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

def convertFullposeMatToVec(fullposeMat):
    if not fullposeMat.shape == (16,3,3):
        if fullposeMat.shape == (48,):
            return fullposeMat
        else:
            raise Exception('Invalid shape for fullposeMat')
    myList = []
    for i in range(16):
        myList.append(cv2.Rodrigues(fullposeMat[i])[0][:,0])
    return np.concatenate(myList, axis=0)