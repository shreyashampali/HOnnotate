import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import auc

from eval import utilsEval
import os


def get2DPCK(predictions, gt, figName=None, showFig=False):
    # calc. PCK
    interval = np.arange(0, 100 + 1, 1)
    errArr = np.zeros((len(interval),), dtype=np.float32)
    projErrs = np.nanmean(np.linalg.norm(gt - predictions, axis=2), axis=1)
    cntr = 0
    for i in interval:
        errArr[cntr] = float(np.sum(projErrs < i)) / float(gt.shape[0]) * 100.
        cntr += 1

    AUC = auc(interval, errArr)
    # plot it
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(interval,
            errArr,
            # c=colors[0],
            linestyle='-', linewidth=1)
    plt.xlabel('Distance threshold / px', fontsize=12)
    plt.ylabel('Fraction of frames within distance / %', fontsize=12)
    plt.xlim([0.0, 100])
    plt.ylim([0.0, 100.0])
    ax.grid(True)

    # save if required
    if figName is not None:
        fig.savefig(figName,
                    bbox_extra_artists=None,
                    bbox_inches='tight')
        with open(figName.split('.')[0] + '.pickle', 'wb') as f:
            pickle.dump({'x': interval, 'y': errArr, 'gt': gt, 'est': predictions}, f)

    # show if required
    if showFig:
        plt.show(block=False)
    # plt.close(fig)

    return interval, errArr, AUC

def get2DKpsFromHeatmap(heatmap):
    isNoBatch = True
    if len(heatmap.shape) == 4:
        batchSize = heatmap.shape[0]
        isNoBatch = False
    elif len(heatmap.shape) == 3:
        heatmap = np.expand_dims(heatmap, 0)
        batchSize = 1
    else:
        raise Exception('Invalid shape for heatmap')

    numKps = heatmap.shape[-1]
    kps2d = np.zeros((batchSize, numKps, 2), dtype=np.uint8)
    for i in range(batchSize):
        for kp in range(numKps):
            y, x = np.unravel_index(np.argmax(heatmap[i][:,:,kp], axis=None), heatmap[i][:,:,kp].shape)
            kps2d[i, kp, 0] = x
            kps2d[i, kp, 1] = y

    if isNoBatch:
        kps2d = kps2d[0]
    return kps2d

def vis2DKps(predictions, gt, img, saveDir, filename, visType=utilsEval.visTypeEnum.STICK_ANNO):
    assert isinstance(visType, utilsEval.visTypeEnum)

    if np.sum(np.isnan(predictions)) > 0:
        return


    if visType == utilsEval.visTypeEnum.STICK_ANNO:
        assert len(predictions.shape) == 2
        assert (predictions.shape[-1] == 2)

        if filename is not None:
            saveFilename = os.path.join(saveDir, filename)
            if not (('png' in saveFilename) or ('jpg' in saveFilename)):
                saveFilename = saveFilename + '.jpg'
        else:
            saveFilename = None


        if predictions.shape[0] == 21:
            imgOut = utilsEval.showHandJoints(img, predictions, estIn=gt,
                                 filename=saveFilename,
                                 upscale=1, lineThickness=2)
        else:
            imgOut = utilsEval.showHandJointsOld(img, predictions, estIn=gt,
                                              filename=saveFilename,
                                              upscale=1, lineThickness=2)
    elif visType == utilsEval.visTypeEnum.HEATMAP:
        assert len(predictions.shape) == 3
        saveFilename = os.path.join(saveDir, filename)
        if not (('png' in saveFilename) or ('jpg' in saveFilename)):
            saveFilename = saveFilename + '.jpg'
            # utilsEval.showHandHeatmap(img, predictions, saveFilename)
            utilsEval.showHeatmaps(predictions, gt, saveFilename)
    else:
        raise Exception

    return

def getKpsWrtImage(kps2d, topLeft, bottomRight, imgH, imgW):
    patchSize = bottomRight - topLeft
    assert patchSize[0] > 0
    assert patchSize[1] > 0

    scaleW = float(patchSize[0]) / float(imgW)
    scaleH = float(patchSize[1]) / float(imgH)

    scale = np.tile(np.array([[scaleW, scaleH]]), [kps2d.shape[0], 1])

    kps2dOrig = kps2d[:,:2]*scale

    kps2dOrig = kps2dOrig + np.array(topLeft)

    return kps2dOrig




