import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from evalNet import utilsEval, eval2DKps
import os
from sklearn.metrics import auc

def get3DPCK(predictionsIn, gt, figName=None, showFig=False, withDepCorrection=False):

    if withDepCorrection:
        predictions = np.zeros_like(predictionsIn)
        for i in range(predictionsIn.shape[0]):
            predCurr = predictionsIn[i].copy()
            s = np.sqrt(np.sum(np.square(predCurr[9] - predCurr[0])))
            if s > 0:
                predCurr = predCurr/s
            sGT = np.sqrt(np.sum(np.square(gt[i][9] - gt[i][0])))
            predCurr = predCurr*sGT
            predCurrRel = predCurr - predCurr[0:1,:]
            predCurrAbs = predCurrRel + gt[i][0:1,:]
            predictions[i] = predCurrAbs
    else:
        predictions = predictionsIn

    # calc. PCK
    interval = np.arange(0, 20 + 0.01, 0.01)
    errArr = np.zeros((len(interval),), dtype=np.float32)
    aa = np.linalg.norm(gt - predictions, axis=2)
    projErrs = np.nanmean(np.linalg.norm(gt - predictions, axis=2), axis=1)
    projErrs = projErrs[np.logical_not(np.isnan(projErrs))]
    meanProjErr = np.nanmean(projErrs)
    cntr = 0
    for i in interval:
        errArr[cntr] = (float(np.sum(projErrs < (i/100))) / float(gt.shape[0])) * 100.
        cntr += 1

    AUC = auc(interval, errArr)
    # plot it
    if figName is not None or showFig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(interval,
                errArr,
                # c=colors[0],
                linestyle='-', linewidth=1)
        plt.xlabel('Distance threshold / cm', fontsize=12)
        plt.ylabel('Fraction of frames within distance / %', fontsize=12)
        plt.xlim([0.0, np.max(interval)])
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

    projErrsRel = np.nanmean(np.linalg.norm((gt-gt[:,0:1,:]) - (predictions-predictions[:,0:1,:]), axis=2), axis=1)
    meanProjErrRel = np.nanmean(projErrsRel)

    return interval, errArr, AUC, meanProjErr, meanProjErrRel

def get3DKpsFrom2p5D(hm2D, hm3D, kps2DFull, gt3D, camMat, relNormDep = None):
    '''
    Calculates 3d coordinates from 2.5D map
    :param hm2D: heatmaps 2d at patch size
    :param hm3D: heatmaps 3d (rel depth) at patch size
    :param kps2DFull: kps at full image which is precomputed somewhere
    :param gt3D: ground truth 3d annotation for getting the scale
    :param camMat: camera matrix
    :return:
    '''

    if relNormDep is None:
        # get the normalized root relative depths
        kps2DScaled = eval2DKps.get2DKpsFromHeatmap(hm2D)
        relNormDep = np.zeros((kps2DScaled.shape[0],1), dtype=np.float32)
        for i in range(0, kps2DScaled.shape[0]):
            relNormDep[i] = hm3D[kps2DScaled[i,1], kps2DScaled[i,0], i]/10.0 # convert dms to mts
    else:
        assert relNormDep.shape == (gt3D.shape[0], 1)
        relNormDep = relNormDep/10 # convert dms to mts

    kps2D = kps2DFull.copy()

    # get the 2d pixel positions in mts (this is not mentioned in paper, WHAT THE HELL!!!)
    kps2D = kps2D[:,:2] - np.array([camMat[0,2], camMat[1,2]]) # subtract principal point
    kps2D = kps2D/(np.tile(np.array([[camMat[0,0], camMat[1,1]]]), [kps2D.shape[0],1])) # divide by focal length

    # get the rel norm dep in opencv coordinate system
    relNormDep = -relNormDep
    relNormDep[0] = 0.
    # print(relNormDep[:,0])

    # eqns. from paper for calc. norm. root depth
    a = np.sum(np.square(kps2D[5] - kps2D[0]))
    b = relNormDep[5]*(kps2D[5,0]**2 + kps2D[5,1]**2 - kps2D[5,0]*kps2D[0,0] - kps2D[5,1]*kps2D[0,1]) + \
        relNormDep[0] * (kps2D[0, 0] ** 2 + kps2D[0, 1] ** 2 - kps2D[5, 0] * kps2D[0, 0] - kps2D[5, 1] * kps2D[0, 1])
    c = (kps2D[5,0]*relNormDep[5] - kps2D[0,0]*relNormDep[0])**2 + (kps2D[5,1]*relNormDep[5] - kps2D[0,1]*relNormDep[0])**2 + \
        (relNormDep[5] - relNormDep[0])**2 - 1.
    chk = b**2 - 4*a*c
    # assert chk >= 0
    if chk<0:
        kps3D = np.zeros_like(gt3D) + float('nan')
        return kps3D

    normRootDep = 0.5*(-b + np.sqrt(b**2 - 4*a*c))/a

    # get normalization factor
    s = np.linalg.norm(gt3D[5] - gt3D[0], ord=2)

    # get norm abs. depths of all joints
    normDep = relNormDep + normRootDep

    # get abs. depths
    dep = normDep * s

    # get 3d kps from 2dkps and dep. Note the points are in opencv coordinate system
    kps2D = np.concatenate([kps2DFull, np.ones((kps2D.shape[0],1), dtype=np.uint8)], axis=1).astype(np.float32)
    kps2D = kps2D * np.tile(np.reshape(dep, [-1, 1]), [1,3])
    kps3D = kps2D.dot(np.linalg.inv(camMat).T)

    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    # print(kps3D.dot(coordChangeMat)-gt3D)

    return kps3D


def get2p5DepMatFromKps(kps3d, heatmaps):
    '''

    :param kps3d:
    :param heatmaps:
    :return:
    '''
    norm_factor = np.linalg.norm(kps3d[5] - kps3d[0], ord=2)
    kps3d = kps3d.copy()/norm_factor

    kps3d_rel = kps3d - np.tile(kps3d[0:1, :], [kps3d.shape[0], 1])
    rel_depth = kps3d_rel[:, 2] * 10  #  in dms
    rel_depth_vec = rel_depth.copy()
    rel_depth = np.expand_dims(rel_depth, 0)
    rel_depth = np.expand_dims(rel_depth, 0)  # 1x1xnum_kps

    a = np.max(heatmaps[:,:,0])
    rel_depth_maps = rel_depth * heatmaps

    return rel_depth_maps, rel_depth_vec

def vis3DKps(predictions3D, gt3D, img, camMat, saveDir, filename, visType=utilsEval.visTypeEnum.STICK_ANNO):
    assert isinstance(visType, utilsEval.visTypeEnum)
    assert visType == utilsEval.visTypeEnum.STICK_ANNO

    if np.sum(np.isnan(predictions3D)) > 0:
        return

    if np.sum(np.isnan(gt3D)) > 0:
        print('ignoring %s'%(filename))
        return


    predictions = utilsEval.cv2ProjectPoints(camMat, predictions3D, isOpenGLCoords=False)
    gt = utilsEval.cv2ProjectPoints(camMat, gt3D, isOpenGLCoords=False)

    if visType == utilsEval.visTypeEnum.STICK_ANNO:
        assert len(predictions.shape) == 2
        assert (predictions.shape[-1] == 2)

        if filename is not None:
            saveFilename = os.path.join(saveDir, filename)
            if not (('png' in saveFilename) or ('jpg' in saveFilename)):
                saveFilename = saveFilename + '.jpg'
        else:
            saveFilename = None

        imgOut = utilsEval.showHandJoints(img, gt, estIn=predictions,
                                          filename=saveFilename,
                                          upscale=1, lineThickness=3)
    elif visType == utilsEval.visTypeEnum.HEATMAP:
        assert len(predictions.shape) == 3
        saveFilename = os.path.join(saveDir, filename)
        if not (('png' in saveFilename) or ('jpg' in saveFilename)):
            saveFilename = saveFilename + '.jpg'
            utilsEval.showHeatmaps(img, predictions, saveFilename)
    else:
        raise Exception

    return
