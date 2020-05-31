import cv2
import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from enum import IntEnum
import math
import chumpy as ch

class visTypeEnum(IntEnum):
    '''
    Enum for right/left hand in dataset
    '''

    HEATMAP = 1

    STICK_ANNO = 2

def showHandJointsOld(imgIn, gtIn, estIn=None, filename=None, upscale=1, lineThickness=3):
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
    # jointConns = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18], [0, 10, 11, 12, 19], [0, 7, 8, 9, 20], [0, 13, 14, 15, 16]]
    if gtIn.shape[0] == 21:
        jointConns = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
    elif gtIn.shape[0] == 8:
        jointConns = [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    else:
        raise NotImplementedError


    jointColsGt = [(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255)]
    jointColsEst  = []
    for col in jointColsGt:
        newCol = (col[0]+col[1]+col[2])/3
        jointColsEst.append((newCol, newCol, newCol))

    if True:
        for i in range(len(jointColsGt)):
            jointColsGt[i] = (0,255,0)
            jointColsEst[i] = (255,0,0)

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
                cv2.line(img, (int(gt[jntC,0]), int(gt[jntC,1])), (int(gt[jntN,0]), int(gt[jntN,1])), jointColsGt[0], lineThickness)
            if estIn is not None:
                cv2.line(img, (int(est[jntC,0]), int(est[jntC,1])), (int(est[jntN,0]), int(est[jntN,1])), jointColsEst[0], lineThickness)

    if filename is not None:
        cv2.imwrite(filename, img)

    return img

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

def showHandHeatmap(img, predictions, filename):
    numKps = predictions.shape[-1]
    assert numKps == 21
    assert img.shape[:2] == predictions.shape[:2]

    blendFact = 0.0
    plt.ioff()
    fig = plt.figure()
    ax = fig.subplots(3, 7)
    axesList = [[], [], []]
    for i in range(7):
        axesList[0].append(ax[0, i].imshow(np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)))
        axesList[1].append(ax[1, i].imshow(np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)))
        axesList[2].append(ax[2, i].imshow(np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)))

    # plt.subplots_adjust(top=0.984,
    #                     bottom=0.016,
    #                     left=0.028,
    #                     right=0.99,
    #                     hspace=0.045,
    #                     wspace=0.124)
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()

    for i in range(3):
        for j in range(7):
            kpCnt = i*7 + j

            pred3 = np.tile(predictions[:,:,kpCnt:kpCnt+1], [1,1,3])*255
            pred3 = np.clip(np.round(pred3), 0, 255).astype(np.uint32)
            pred3[:,:,:2] = pred3[:,:,:2]*0# keep only r channel

            imgBlend = img.astype(np.uint32)
            imgBlend[:,:,2] = imgBlend[:,:,2]*blendFact + pred3[:,:,2]*(1-blendFact)
            #np.clip(imgBlend[:,:,2]+pred3[:,:,2], 0, 255)

            imgBlend = np.round(imgBlend).astype(np.uint8)

            axesList[i][j].set_data(imgBlend[:,:,[2,1,0]])

    plt.savefig(filename)
    plt.close(fig)



def showHanObjectHeatmap(img, predictions, filename):
    numKps = predictions.shape[-1]
    assert numKps == 8
    assert img.shape[:2] == predictions.shape[:2]



    blendFact = 0.0
    plt.ioff()
    fig = plt.figure()
    ax = fig.subplots(split[0], split[1])
    axesList = [[], [], []]
    for i in range(split[1]):
        axesList[0].append(ax[0, i].imshow(np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)))
        axesList[1].append(ax[1, i].imshow(np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)))
        axesList[2].append(ax[2, i].imshow(np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)))

    # plt.subplots_adjust(top=0.984,
    #                     bottom=0.016,
    #                     left=0.028,
    #                     right=0.99,
    #                     hspace=0.045,
    #                     wspace=0.124)
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()

    for i in range(split[0]):
        for j in range(split[1]):
            kpCnt = i*split[1] + j

            pred3 = np.tile(predictions[:,:,kpCnt:kpCnt+1], [1,1,3])*255
            pred3 = np.clip(np.round(pred3), 0, 255).astype(np.uint32)
            pred3[:,:,:2] = pred3[:,:,:2]*0# keep only r channel

            imgBlend = img.astype(np.uint32)
            imgBlend[:,:,2] = imgBlend[:,:,2]*blendFact + pred3[:,:,2]*(1-blendFact)
            #np.clip(imgBlend[:,:,2]+pred3[:,:,2], 0, 255)

            imgBlend = np.round(imgBlend).astype(np.uint8)

            axesList[i][j].set_data(imgBlend[:,:,[2,1,0]])

    plt.savefig(filename)
    plt.close(fig)

def showHeatmaps(predictions, gt, filename):
    numKps = predictions.shape[-1]
    # assert numKps == 21
    assert gt.shape[:2] == predictions.shape[:2]

    if numKps == 8:
        split = [2, 4]
    elif numKps == 21:
        split = [3, 7]
    else:
        raise NotImplementedError

    blendFact = 0.0
    plt.ioff()
    fig = plt.figure()
    ax = fig.subplots(split[0], split[1])
    axesList = [[], [], []]
    for i in range(split[1]):
        for j in range(split[0]):
            axesList[j].append(ax[j, i].imshow(np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)))
            # axesList[j].append(ax[1, i].imshow(np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)))
            # axesList[j].append(ax[2, i].imshow(np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)))

    # plt.subplots_adjust(top=0.984,
    #                     bottom=0.016,
    #                     left=0.028,
    #                     right=0.99,
    #                     hspace=0.045,
    #                     wspace=0.124)
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()

    for i in range(split[0]):
        for j in range(split[1]):
            kpCnt = i*split[1] + j

            pred3 = np.tile(predictions[:,:,kpCnt:kpCnt+1], [1,1,3])*255
            pred3 = np.clip(np.round(pred3), 0, 255).astype(np.uint32)
            pred3[:,:,:2] = pred3[:,:,:2]*0# keep only r channel

            gt3 = np.tile(gt[:, :, kpCnt:kpCnt + 1], [1, 1, 3]) * 255
            gt3 = np.clip(np.round(gt3), 0, 255).astype(np.uint32)
            gt3[:, :, 1:] = gt3[:, :, 1:] * 0  # keep only b channel

            imgBlend = gt3.astype(np.uint32)
            imgBlend[:,:,2] = pred3[:,:,2]
            #np.clip(imgBlend[:,:,2]+pred3[:,:,2], 0, 255)

            imgBlend = np.round(imgBlend).astype(np.uint8)

            axesList[i][j].set_data(imgBlend[:,:,[2,1,0]])

    plt.savefig(filename)
    plt.close(fig)

def cv2ProjectPoints(camMat, pts3D, isOpenGLCoords=True):
    '''
    TF function for projecting 3d points to 2d using CV2
    :param camProp:
    :param pts3D:
    :param isOpenGLCoords:
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if isOpenGLCoords:
        pts3D = pts3D.dot(coordChangeMat.T)

    projPts = pts3D.dot(camMat.T)
    projPts = np.stack([projPts[:,0]/projPts[:,2], projPts[:,1]/projPts[:,2]],axis=1)

    assert len(projPts.shape) == 2

    return projPts

def chProjectPoints(pts3D, camMat, isOpenGLCoords=True):
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    if isOpenGLCoords:
        coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        pts3D = pts3D.dot(coordChangeMat.T)

    # camMat = np.array([[617.343, 0, 312.42], [0, 617.343, 241.42], [0, 0, 1]])

    projPts = pts3D.dot(camMat.T)
    projPts = ch.vstack([projPts[:,0]/projPts[:,2], projPts[:,1]/projPts[:,2]]).T

    assert len(projPts.shape) == 2

    return projPts

