import numpy as np
import tensorflow as tf
import math
import cv2
import dirt.rasterise_ops
import dirt.lighting
import dirt.matrices
from ext.mesh_loaders import *
import os
from ghope.utils import *
import matplotlib.pyplot as plt
import time
from ghope.common import *
from manoTF.batch_mano import MANO
from HOdatasets.mypaths import *
from ghope.constraints import Constraints
from object.batch_object import objectModel
import random


def getOptimizer(loss, varList, global_step, type='Adam', learning_rate=0.01):
    if type=='Adam':
        adamOpt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads = 0#adamOpt.compute_gradients(loss, var_list=varList)
        optimizer = adamOpt.minimize(loss, var_list=varList, global_step=global_step)
    else:
        raise Exception(' Optimizer %s not Implemented!!'%(type))

    return optimizer, grads


def getDepthLossObj(gt, projected_vertices, faces, bgDepth, imgH, imgW, mask=1.0, isClipDepthLoss=False):

    # backgorund depth
    bgcolor = tf.Variable(np.array([bgDepth], dtype=np.float32), name='bgColor')
    bgcolor = tf.expand_dims(tf.expand_dims(bgcolor, 0), 0)
    bgcolor = tf.tile(bgcolor, [imgH, imgW, 1])

    # vertex colors (depth in this case)
    depth_vertices = tf.stop_gradient(projected_vertices[:, 3:4])

    # Rendering op
    im = dirt.rasterise_ops.rasterise(bgcolor, projected_vertices, depth_vertices, faces, height=imgH, width=imgW, channels=1)

    # define the loss
    if isClipDepthLoss:
        err = tf.abs(gt-im)
        err = tf.minimum(err, 0.5)
        err = tf.square(err)*mask
        depthLoss = tf.reduce_mean(err)
    else:
        depthLoss = tf.losses.mean_squared_error(gt, im, weights=mask)

    return depthLoss, im


def getSegLossObj(gt, projected_vertices, faces, bgColor, imgH, imgW, mask=1.0):

    # backgorund color
    bgcolor = tf.Variable(np.array(bgColor, dtype=np.float32), name='bgColor')
    bgcolor = tf.expand_dims(tf.expand_dims(bgcolor, 0), 0)
    bgcolor = tf.tile(bgcolor, [imgH, imgW, 1])

    # vertex colors
    vertex_color = tf.Variable(np.array([1., 1., 1.], dtype=np.float32), name='vertexColor')
    vertex_color = tf.expand_dims(vertex_color, 0)
    vertex_colors = tf.tile(vertex_color, [tf.shape(projected_vertices)[0], 1])

    # Rendering op
    im = dirt.rasterise_ops.rasterise(bgcolor, projected_vertices, vertex_colors, faces, height=imgH, width=imgW, channels=3)

    # define the loss
    segLoss = tf.losses.mean_squared_error(gt, im, weights=mask)

    return segLoss, im

def getDepthSegRendBatch(projected_vertices, faces, bgColor, bgDepth, imgH, imgW):
    '''
    Tiled rendering of seg and depth maps.
    :param projected_vertices:
    :param faces:
    :param bgColor:
    :param bgDepth:
    :param imgH:
    :param imgW:
    :param mask:
    :return: a tensor of shape [2,h,w,3], 1st subTensor is seg, 2nd subTensor is depth
    '''

    # backgorund color
    bgColorTF = tf.Variable(np.array(bgColor, dtype=np.float32), name='bgColor')
    bgColorTF = tf.expand_dims(tf.expand_dims(bgColorTF, 0), 0)
    bgColorTF = tf.tile(bgColorTF, [imgH, imgW, 1])

    # backgorund depth
    bgDepthTF = tf.Variable(np.array([bgDepth, bgDepth, bgDepth], dtype=np.float32), name='bgDepth')
    bgDepthTF = tf.expand_dims(tf.expand_dims(bgDepthTF, 0), 0)
    bgDepthTF = tf.tile(bgDepthTF, [imgH, imgW, 1])

    bgBatch = tf.stack([bgColorTF, bgDepthTF], axis = 0)



    # vertex colors
    vertex_color = tf.Variable(np.array([1., 1., 1.], dtype=np.float32), name='vertexColor')
    vertex_color = tf.expand_dims(vertex_color, 0)
    vertex_colors = tf.tile(vertex_color, [tf.shape(projected_vertices)[0], 1])

    # vertex depths
    vertex_depths = tf.stop_gradient(projected_vertices[:, 3:4])
    vertex_depths = tf.tile(vertex_depths, [1,3])

    vertexAttrBatch = tf.stack([vertex_colors, vertex_depths])

    facesBatch = np.stack([faces, faces], axis = 0)

    projected_verticesBatch = tf.stack([projected_vertices, projected_vertices], axis=0)

    imBatch = dirt.rasterise_ops.rasterise_batch(bgBatch, projected_verticesBatch, vertexAttrBatch, facesBatch, height=imgH, width=imgW,
                                      channels=3)

    return imBatch

def gaussianPyr(err):
    oneway = np.tile(cv2.getGaussianKernel(3,1), (1, 3))
    gaussKernel = oneway * oneway.T
    # if len(err.shape) == 4:
    #     gaussKernel = np.stack([gaussKernel, gaussKernel, gaussKernel], axis=2)
    gaussKernel = np.expand_dims(gaussKernel, 3)
    gaussKernel = np.expand_dims(gaussKernel, 3)

    gaussErr = tf.nn.conv2d(err, gaussKernel, strides=[1, 1, 1, 1], padding="VALID")

    gaussErrDown = gaussErr[:, ::2, ::2, :]

    gaussErrDownLoss = tf.reduce_mean(gaussErrDown)

    return gaussErrDown, gaussErrDownLoss



def getDepthSegLoss(imBatch, segGT, depthGT, mask, segWt, depthWt, isClipDepthLoss=False, pyrLevel = 0):

    if len(depthGT.shape) == 2:
        depthGT = np.stack([depthGT, depthGT, depthGT], axis=2)
    if depthGT.shape[2] == 1:
        depthGT = np.tile(depthGT, [1, 1, 3])

    depthGT = depthGT.astype(np.float32)
    assert depthGT.shape[2] == 3, 'Invalid Shape of ground depth map..'

    if len(mask.shape) == 2:
        mask = np.stack([mask, mask, mask], axis=2)
    if mask.shape[2] == 1:
        mask = np.tile(mask, [1, 1, 3])


    maskBatch = np.stack([mask, mask], axis=0)
    # gtBatch = tf.stack([segGT, tf.tile(depthGT, [1, 1, 3])], axis =0)

    # define the loss

    errDepth = tf.abs(depthGT - imBatch[1])
    if isClipDepthLoss:
        errDepth = tf.minimum(errDepth, 0.5)
    errDepth = tf.square(errDepth) * maskBatch[1]

    errSeg = tf.abs(segGT - imBatch[0])
    errSeg = tf.square(errSeg) * maskBatch[0]

    depthLoss = tf.reduce_mean(errDepth)
    segLoss = tf.reduce_mean(errSeg)
    if pyrLevel > 0:
        errDepth = tf.expand_dims(errDepth[:,:,0:1], 0)
        errSeg = tf.expand_dims(errSeg[:,:,0:1], 0)
        for _ in range(pyrLevel):
            errDepth, lossDep = gaussianPyr(errDepth)
            errSeg, lossSeg = gaussianPyr(errSeg)
            depthLoss += lossDep
            segLoss += lossSeg

    totalLoss = depthLoss*depthWt + segLoss*segWt

    return totalLoss





def getTransformationMat(translation, rotation, f, c, near, far, imShape):
    # get rotation matrix
    view_matrix_1 = tf.transpose(dirt.matrices.rodrigues(rotation))

    # get translation matrix
    view_matrix_2 = tf.stack([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        tf.concat([translation, [1.]], axis=0)
    ])

    # get projection matrix
    projection_matrix = creatCamMat(f,
                                    c, near, far, imShape)

    objPoseMat = tf.matmul(view_matrix_1, view_matrix_2)
    finalMatrix = tf.matmul(objPoseMat, projection_matrix)


    # return the final matrix
    return finalMatrix, objPoseMat, projection_matrix


class dirtRenderer_old():
    def __init__(self, projected_vertices, faces, imgH, imgW, bgColor=[0.,0.,0.], bgDepth=2.0, vertexColors=None):

        self.bgDict = {}
        self.vcDict = {}

        # backgorund color
        bgColorTF = tf.constant(np.array(bgColor, dtype=np.float32), name='bgColor')
        bgColorTF = tf.expand_dims(tf.expand_dims(bgColorTF, 0), 0)
        bgColorTF = tf.tile(bgColorTF, [imgH, imgW, 1])

        # backgorund depth
        bgDepthTF = tf.constant(np.array([bgDepth, bgDepth, bgDepth], dtype=np.float32), name='bgDepth')
        bgDepthTF = tf.expand_dims(tf.expand_dims(bgDepthTF, 0), 0)
        bgDepthTF = tf.tile(bgDepthTF, [imgH, imgW, 1])

        self.bgDict['seg'] = bgColorTF
        self.bgDict['col'] = bgColorTF
        self.bgDict['dep'] = bgDepthTF

        # seg colors
        seg_color = tf.constant(np.array([1., 1., 1.], dtype=np.float32), name='vertexColor')
        seg_color = tf.expand_dims(seg_color, 0)
        seg_colors = tf.tile(seg_color, [tf.shape(projected_vertices)[0], 1])

        # vertex colors
        vertex_colors = seg_colors
        if vertexColors is not None:
            vertex_colors = tf.constant(vertexColors, name='vertexColor')

        # vertex depths
        vertex_depths = tf.stop_gradient(projected_vertices[:, 3:4])
        vertex_depths = tf.tile(vertex_depths, [1, 3])

        self.vcDict['seg'] = seg_colors
        self.vcDict['col'] = vertex_colors
        self.vcDict['dep'] = vertex_depths

        self.faces = faces
        self.vertices = projected_vertices
        self.imgW = imgW
        self.imgH = imgH

    def render(self, rendType):
        if not isinstance(rendType, list):
            rendType = [rendType]

        for i in rendType:
            if i not in ['seg', 'col', 'dep']:
                raise Exception('Invalid render type %s. Should be seg, col or dep'%i)

        bgBatch = tf.stack([self.bgDict[i] for i in rendType], axis=0)

        vertexAttrBatch = tf.stack([self.vcDict[i] for i in rendType], axis=0)

        facesBatch = np.stack([self.faces for _ in rendType], axis=0)

        projected_verticesBatch = tf.stack([self.vertices for _ in rendType], axis=0)

        imBatch = dirt.rasterise_ops.rasterise_batch(bgBatch, projected_verticesBatch, vertexAttrBatch, facesBatch,
                                                     height=self.imgH, width=self.imgW,
                                                     channels=3)

        return imBatch


class DirtRenderer():
    def __init__(self, mesh, renderMode, bgColor=np.array([0.,0.,0.]), bgDepth=2.0):

        # make some initial check of mesh
        assert hasattr(mesh, 'v'), 'Input mesh has no vertices attribute'
        assert hasattr(mesh, 'f'), 'Input mesh has no faces attribute'
        assert hasattr(mesh, 'vc'), 'Input mesh has no vc attribute'
        assert hasattr(mesh, 'vcSeg'), 'Input mesh has no vcSeg attribute'
        assert hasattr(mesh, 'frameSize'), 'Input mesh has no frameSize attribute'
        assert len((mesh.v.shape)) == 3
        assert len(mesh.frameSize.shape) == 1
        assert (mesh.v.shape)[1] == (mesh.vc.shape)[0] == (mesh.vcSeg.shape)[0], 'shapes of v, vc, vcSeg do not match'

        self.renderMode = renderMode
        self.mesh = mesh
        self.imgW = self.mesh.frameSize[0]
        self.imgH = self.mesh.frameSize[1]
        self.numViews = self.mesh.v.shape[0]

        # batch out the mesh attributes
        self.verticesBatch = self.mesh.v
        self.fBatch = tf.tile(tf.expand_dims(self.mesh.f, 0), [tf.shape(self.mesh.v)[0], 1, 1])

        # batch out the 3 types of vertex colors
        vcColBatch = tf.tile(tf.expand_dims(self.mesh.vc, 0), [tf.shape(self.mesh.v)[0], 1, 1])
        vcSegBatch = tf.tile(tf.expand_dims(self.mesh.vcSeg, 0), [tf.shape(self.mesh.v)[0], 1, 1])
        vcDepthBatch = tf.stop_gradient(self.mesh.v[:, :, 3:4])
        vcDepthBatch = tf.tile(vcDepthBatch, [1, 1, 3])

        # batch out backgound
        bgColor = tf.constant(np.array(bgColor, dtype=np.float32), name='bgColor')
        bgColor = tf.expand_dims(tf.expand_dims(tf.expand_dims(bgColor, 0), 0), 0)
        bgColor = tf.tile(bgColor, [tf.shape(self.mesh.v)[0], self.imgH, self.imgW, 1])

        bgDepth = tf.constant(np.array([bgDepth, bgDepth, bgDepth], dtype=np.float32), name='bgDepth')
        bgDepth = tf.expand_dims(tf.expand_dims(tf.expand_dims(bgDepth, 0), 0), 0)
        bgDepth = tf.tile(bgDepth, [tf.shape(self.mesh.v)[0], self.imgH, self.imgW, 1])


        # get batches for vertex, face and colors depending on rendermode
        if self.renderMode == renderModeEnum.SEG:

            self.vertexAttrBatch = vcSegBatch
            self.bgBatch = bgColor

        elif self.renderMode == renderModeEnum.DEPTH:

            self.vertexAttrBatch = vcDepthBatch
            self.bgBatch = bgDepth

        elif self.renderMode == renderModeEnum.COLOR:

            self.vertexAttrBatch = vcColBatch
            self.bgBatch = bgColor

        elif self.renderMode == renderModeEnum.SEG_COLOR:

            self.verticesBatch = tf.tile(self.verticesBatch, [2, 1, 1])
            self.fBatch = tf.tile(self.fBatch, [2, 1, 1])
            self.vertexAttrBatch = tf.concat([vcSegBatch, vcColBatch], axis=0)
            self.bgBatch = tf.concat([bgColor, bgColor], axis=0)

        elif self.renderMode == renderModeEnum.SEG_DEPTH:

            self.verticesBatch = tf.tile(self.verticesBatch, [2, 1, 1])
            self.fBatch = tf.tile(self.fBatch, [2, 1, 1])
            self.vertexAttrBatch = tf.concat([vcSegBatch, vcDepthBatch], axis=0)
            self.bgBatch = tf.concat([bgColor, bgDepth], axis=0)

        elif self.renderMode == renderModeEnum.COLOR_DEPTH:

            self.verticesBatch = tf.tile(self.verticesBatch, [2, 1, 1])
            self.fBatch = tf.tile(self.fBatch, [2, 1, 1])
            self.vertexAttrBatch = tf.concat([vcColBatch, vcDepthBatch], axis=0)
            self.bgBatch = tf.concat([bgColor, bgDepth], axis=0)

        elif self.renderMode == renderModeEnum.SEG_COLOR_DEPTH:

            self.verticesBatch = tf.tile(self.verticesBatch, [3, 1, 1])
            self.fBatch = tf.tile(self.fBatch, [3, 1, 1])
            self.vertexAttrBatch = tf.concat([vcSegBatch, vcColBatch, vcDepthBatch], axis=0)
            self.bgBatch = tf.concat([bgColor, bgColor, bgDepth], axis=0)

        else:

            raise Exception('Invalid renderMode..')

        # make some final checks
        assert (self.verticesBatch.shape)[0] == (self.fBatch.shape)[0] == (self.vertexAttrBatch.shape)[0] == (self.bgBatch.shape)[0]
        assert (self.verticesBatch.shape[:-1]) == (self.vertexAttrBatch.shape[:-1])


    def render(self):


        imBatch = dirt.rasterise_ops.rasterise_batch(self.bgBatch, self.verticesBatch, self.vertexAttrBatch, self.fBatch,
                                                     height=self.imgH, width=self.imgW,
                                                     channels=3)

        # reorder the output of rasterize in the 'observables' format

        if self.renderMode == renderModeEnum.SEG:

            observsVirt = observables(frameID=[str(i) for i in range(self.numViews)], seg=imBatch, depth=None, col=None, mask=None, isReal=False)

        elif self.renderMode == renderModeEnum.DEPTH:

            observsVirt = observables(frameID=[str(i) for i in range(self.numViews)], seg=None, depth=imBatch, col=None, mask=None, isReal=False)

        elif self.renderMode == renderModeEnum.COLOR:

            observsVirt = observables(frameID=[str(i) for i in range(self.numViews)], seg=None, depth=None, col=imBatch, mask=None, isReal=False)

        elif self.renderMode == renderModeEnum.SEG_COLOR:

            observsVirt = observables(frameID=[str(i) for i in range(self.numViews)], seg=imBatch[:self.numViews], depth=None, col=imBatch[self.numViews:],
                                  mask=None, isReal=False)

        elif self.renderMode == renderModeEnum.SEG_DEPTH:

            observsVirt = observables(frameID=[str(i) for i in range(self.numViews)], seg=imBatch[:self.numViews], depth=imBatch[self.numViews:], col=None,
                                  mask=None, isReal=False)

        elif self.renderMode == renderModeEnum.COLOR_DEPTH:

            observsVirt = observables(frameID=[str(i) for i in range(self.numViews)], seg=None, depth=imBatch[self.numViews:], col=imBatch[:self.numViews],
                                  mask=None, isReal=False)

        elif self.renderMode == renderModeEnum.SEG_COLOR_DEPTH:

            observsVirt = observables(frameID=[str(i) for i in range(self.numViews)], seg=imBatch[:self.numViews], depth=imBatch[2*self.numViews:],
                                  col=imBatch[self.numViews:2*self.numViews], mask=None, isReal=False)

        else:

            raise Exception('Invalid renderMode..')


        return observsVirt





