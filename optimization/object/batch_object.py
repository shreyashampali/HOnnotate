import tensorflow as tf
import numpy as np
from ext.mesh_loaders import load_mesh
import os
from dirt.matrices import rodrigues
from HOdatasets.mypaths import YCB_MODELS_DIR as ycbModelsDir


YCBObjList = ['006_mustard_bottle', '003_cracker_box', '004_sugar_box', '011_banana']


def getMesh(objName, objType='YCB'):
    if objType == 'YCB':
        modelPath = os.path.join(ycbModelsDir, objName)

        if objName not in YCBObjList:
            raise Exception('%s object not supported..', objName)
    else:
        raise NotImplementedError('Only YCB objects supported for now...')

    mesh = load_mesh(modelPath)

    return mesh


class objectModel(object):
    def __init__(self):
        pass

    def setMesh(self, objMesh):
        self.mesh = objMesh
        # self.mesh.vcSeg = np.tile(np.expand_dims(segColor, 0), [self.mesh.v.shape[0], 1])

    def __call__(self, mesh, rot, trans, segColor, name=None):
        with tf.name_scope(name, "object_main", [rot, trans]):
            # get rotation matrix
            view_matrix_1 = tf.transpose(rodrigues(rot), perm=[0,2,1]) # Nx4x4

            # get translation matrix
            trans4 = tf.concat([trans, tf.ones((tf.shape(trans)[0], 1), dtype=tf.float32)], axis = 1)
            trans4 = tf.expand_dims(trans4, 1)
            trans4 = tf.concat([np.tile(np.array([[[0., 0., 1., 0.]]]), [trans.shape[0],1,1]), trans4], axis = 1)
            trans4 = tf.concat([np.tile(np.array([[[0., 1., 0., 0.]]]), [trans.shape[0],1,1]), trans4], axis = 1)
            view_matrix_2 = tf.concat([np.tile(np.array([[[1., 0., 0., 0.]]]), [trans.shape[0],1,1]), trans4], axis = 1)  # Nx4x4

            vertices = tf.concat([mesh.v, tf.ones([tf.shape(mesh.v)[0], 1])], axis=1) # Mx4
            vertices = tf.tile(tf.expand_dims(vertices, 0), [trans.shape[0], 1, 1])

            self.objPoseMat = tf.matmul(view_matrix_1, view_matrix_2) # Nx4x4

            verts = tf.matmul(vertices, self.objPoseMat) # NxMx4

            class objMesh(object):
                pass

            objMesh.v = verts
            objMesh.f = mesh.f #tf.tile(tf.expand_dims(self.f, 0), [tf.shape(verts_t)[0], 1, 1])
            objMesh.vcSeg = np.tile(np.expand_dims(segColor, 0), [mesh.v.shape[0], 1])
            if hasattr(mesh, 'vc'):
                objMesh.vc = mesh.vc
            else:
                objMesh.vc = objMesh.vcSeg

            if hasattr(mesh, 'vn'):
                vn = mesh.vn / np.expand_dims(np.linalg.norm(mesh.vn, ord=2, axis=1), 1) # normalize to unit vec
                vn = tf.concat([vn, tf.ones([tf.shape(vn)[0], 1])], axis=1)  # Mx4
                vn = tf.tile(tf.expand_dims(vn, 0), [trans.shape[0], 1, 1])
                objMesh.vn = tf.matmul(vn, view_matrix_1)

            return objMesh
