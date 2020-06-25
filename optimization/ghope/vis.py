import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import vtk
import cv2
from HOdatasets.ho3d_multicamera.dataset import datasetHo3dMultiCamera
from HOdatasets.commonDS import splitType
import ghope.utils as gutils
from os.path import join
from manoTF.batch_mano import getHandVertexCols
import open3d


class renderScene():
    def __init__(self, imgH, imgW):
        self.imgW = imgW
        self.imgH = imgH

        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0])
        self.nodesDict = {}

        self.light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                                        innerConeAngle=np.pi / 16.0, outerConeAngle=np.pi / 6.0)
        self.scene.add(self.light, pose=np.eye(4))

    def addObjectFromMeshFile(self, meshFileName, nodeName, pose=np.eye(4)):
        fuze_trimesh = trimesh.load(meshFileName)
        mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
        self.nodesDict[nodeName] = pyrender.Node(mesh=mesh, matrix=pose)
        self.scene.add_node(self.nodesDict[nodeName])

    def addObjectFromDict(self, dict, nodeName, pose=np.eye(4)):
        assert 'vertices' in dict.keys(), 'Vertices not present in the mesh dict...'
        assert 'faces' in dict.keys(), 'Faces not present in the mesh dict...'

        # fuze_trimesh = trimesh.load({}, None, None, dict)
        # mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
        if 'vertex_colors' in dict.keys():
            triMesh = trimesh.Trimesh(vertices=dict['vertices'], faces=dict['faces'], vertex_colors=dict['vertex_colors'])
        else:
            triMesh = trimesh.Trimesh(vertices=dict['vertices'], faces=dict['faces'])
        mesh = pyrender.Mesh.from_trimesh(triMesh)
        self.nodesDict[nodeName] = pyrender.Node(mesh=mesh, matrix=pose)
        self.scene.add_node(self.nodesDict[nodeName])

    def setObjectPose(self, nodeName, pose):
        if nodeName not in self.nodesDict.keys():
            raise Exception('Node %s node present in scene...' % (nodeName))

        self.nodesDict[nodeName].matrix = pose

    def getObjectPose(self, nodeName):
        if nodeName not in self.nodesDict.keys():
            raise Exception('Node %s node present in scene...' % (nodeName))

        return self.nodesDict[nodeName].matrix

    def addCamera(self, projMat=None):
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        if projMat is not None:
            self.camera.projMatrix = projMat
        self.scene.add(self.camera, pose=np.eye(4))

    def creatcamProjMat(self, f, c, near, far):
        fx = f[0]
        fy = f[1]
        cx = c[0]
        cy = c[1]
        imShape = [self.imgW, self.imgH]

        left = -cx
        right = imShape[0] - cx
        top = cy
        bottom = -(imShape[1] - cy)

        elements1 = np.array([
            [fx, 0., 0., 0, ],
            [0., fy, 0., 0.],
            [0., 0., -(far + near) / (far - near), -2. * far * near / (far - near)],
            [0., 0., -1., 0.]
        ], dtype=np.float32)  # indexed by x/y/z/w (out), x/y/z/w (in)
        normMat = np.array([
            [2 / (right - left), 0., 0., -((left + right) / 2.) * (2 / (right - left))],
            [0., 2 / (top - bottom), 0., -((top + bottom) / 2.) * (2 / (top - bottom))],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ], dtype=np.float32)
        elements = np.matmul(normMat, elements1)

        self.camera.projMatrix = elements
        # self.camera.set_projection_matrix(elements)

    def setCamProjMatrix(self, P):
        assert P.shape == (4, 4), 'Invalid projection matrix shape...'
        self.camera.projMatrix = P
        # self.camera.set_projection_matrix(P)

    def getCamProjMatrix(self):
        return self.camera.get_projection_matrix()

    def render(self, dumpFileName = None):
        r = pyrender.OffscreenRenderer(self.imgW, self.imgH)
        color, depth = r.render(self.scene)

        if dumpFileName is not None:
            plt.imsave(dumpFileName, color)

        r.delete()
        return color, depth

def visAndRenderHand(camMat, fullpose, trans, beta, img = None, J3D = None, camTransMat=np.eye(4, dtype=np.float32), w=640, h=480, isOpenGL=False,
               vertex_color = None, addBG=False):
    pyRend = renderScene(h, w)
    pyRend.addCamera()
    pyRend.creatcamProjMat([camMat[0, 0], camMat[1, 1]], [camMat[0, 2], camMat[1, 2]], 0.001, 2.0)

    bg = cv2.imread('/home/shreyas/Desktop/checkCrop.jpg')
    bg = cv2.resize(bg, (w, h))

    # superimpose joints on image
    coordChangMat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    if J3D is not None:
        handJoints3DHomo = np.concatenate([np.squeeze(J3D), np.ones((21, 1), dtype=np.float32)],
                                          axis=1).T
        handJointProj = \
            cv2.projectPoints(coordChangMat.dot(camTransMat.dot(handJoints3DHomo)[:3, :]).T,
                              np.zeros((3,)), np.zeros((3,)), camMat, np.zeros((4,)))[0][:, 0, :]
        img1Joints = gutils.showHandJoints(img.copy(),
                                           np.round(handJointProj[gutils.jointsMapManoToObman]).astype(np.int32),
                                           estIn=None, filename=None, upscale=1, lineThickness=2)

    # render the synthetic image
    if fullpose.shape == (16, 3, 3):
        fullpose = gutils.convertFullposeMatToVec(fullpose)
    _, handMesh = gutils.forwardKinematics(fullpose, trans, beta)
    if isOpenGL:
        handMesh = handMesh.dot(coordChangMat)
    handVertHomo = np.concatenate([handMesh.r.copy(), np.ones((handMesh.r.shape[0], 1), dtype=np.float32)], axis=1).T
    pyRend.addObjectFromDict({'vertices': camTransMat.dot(handVertHomo)[:3, :].T,
                              'faces': handMesh.f.copy(),
                              'vertex_colors': getHandVertexCols()[:, [2, 1, 0]] if vertex_color is None else vertex_color},
                              'hand')

    # add background
    cRend1, dRend1 = pyRend.render()
    pyRend.scene.remove_node(pyRend.nodesDict['hand'])

    if addBG:
        mask = (dRend1 == 0)
        mask = np.stack([mask, mask, mask], axis=2)
        cRend1 = bg * mask + cRend1 * (1 - mask)

    if J3D is not None:
        return cRend1, img1Joints
    else:
        return cRend1


def getJointsVisAndRend(fileList, camMat, hoPoseDir, objModelPath, camTransMat=np.eye(4, dtype=np.float32), w=640, h=480,
                        addBG=True, addObject=True):
    # camInd = cam2Ind
    # camTransMat = np.linalg.inv(camPose12)

    # python renderer for rendering object texture
    pyRend = renderScene(h, w)
    if addObject:
        pyRend.addObjectFromMeshFile(objModelPath, 'obj')
    pyRend.addCamera()
    pyRend.creatcamProjMat([camMat[0, 0], camMat[1, 1]], [camMat[0, 2], camMat[1, 2]], 0.001, 2.0)

    bg = cv2.imread('/home/shreyas/Desktop/checkCrop.jpg')
    bg = cv2.resize(bg, (w, h))
    bg = np.zeros_like(bg) + 255

    numImgs = len(fileList)
    rendImgs = np.zeros((numImgs,h,w,3), dtype=np.uint8)
    jointVisImgs = np.zeros((numImgs,h,w,3), dtype=np.uint8)

    dataset = datasetHo3dMultiCamera(splitType.TEST, fileListIn=fileList)
    for cntr, f in enumerate(fileList):
        fileID = f.split('/')[-1]
        _, ds = dataset.createTFExample()
        img1 = ds.imgRaw
        poseData = gutils.loadPickleData(join(hoPoseDir, fileID+'.pkl'))

        # superimpose joints on image
        coordChangMat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
        handJoints3DHomo = np.concatenate([np.squeeze(poseData['JTransformed']), np.ones((21,1), dtype=np.float32)], axis=1).T
        handJointProj = \
        cv2.projectPoints(coordChangMat.dot(camTransMat.dot(handJoints3DHomo)[:3,:]).T,
                          np.zeros((3,)), np.zeros((3,)), camMat, np.zeros((4,)))[0][:, 0, :]
        img1Joints = gutils.showHandJoints(img1.copy(), np.round(handJointProj[gutils.jointsMapManoToObman]).astype(np.int32),
                                    estIn=None, filename=None, upscale=1, lineThickness=2)

        objCorners3DHomo = np.concatenate([np.squeeze(poseData['objCornersTransormed']), np.ones((8, 1), dtype=np.float32)],
                                          axis=1).T
        objCornersProj = \
            cv2.projectPoints(coordChangMat.dot(camTransMat.dot(objCorners3DHomo)[:3,:]).T, np.zeros((3,)), np.zeros((3,)), camMat,
                              np.zeros((4,)))[0][:, 0, :]
        img1Joints = gutils.showObjJoints(img1Joints, objCornersProj, lineThickness=2)

        # render the synthetic image
        if addObject:
            poseMat = np.concatenate([cv2.Rodrigues(poseData['rotObj'])[0], np.reshape(poseData['transObj'], [3,1])], axis=1)
            poseMat = np.concatenate([poseMat, np.array([[0., 0., 0., 1.]])], axis=0)
            pyRend.setObjectPose('obj', camTransMat.dot(poseMat))
        if poseData['fullpose'].shape == (16,3,3):
            fullpose = gutils.convertFullposeMatToVec(poseData['fullpose'])
        else:
            fullpose = poseData['fullpose']
        # fullpose[5] = 0.8
        # fullpose[12] = -1.6
        # fullpose[14] = 0.4
        # fullpose[23] = 0.8
        # poseData['trans'][2] = poseData['trans'][2] + 0.05
        # fullpose[32] = 0.8
        _, handMesh = gutils.forwardKinematics(fullpose, poseData['trans'], poseData['beta'])
        handVertHomo = np.concatenate([handMesh.r.copy(), np.ones((handMesh.r.shape[0], 1), dtype=np.float32)], axis=1).T
        pyRend.addObjectFromDict({'vertices': camTransMat.dot(handVertHomo)[:3,:].T,
                                  'faces': handMesh.f.copy(),
                                  'vertex_colors': getHandVertexCols()[:,[2,1,0]]}, 'hand')

        # add background
        cRend1, dRend1 = pyRend.render()
        pyRend.scene.remove_node(pyRend.nodesDict['hand'])

        if addBG:
            mask = (dRend1==0)
            mask = np.stack([mask, mask, mask], axis=2)
            cRend1 = bg * mask + cRend1 * (1 - mask)

        rendImgs[cntr] = cRend1.copy()
        jointVisImgs[cntr] = img1Joints

        # plt.imshow(img1Joints[:,:,[2,1,0]])
        # plt.show()



        print(f)

    return rendImgs, jointVisImgs

def vis_mesh(ms):

    def mkVtkIdList(it):
        vil = vtk.vtkIdList()
        for i in it:
            vil.InsertNextId(int(i))
        return vil

    if not isinstance(ms, list):
        ms = [ms]

    # The usual rendering stuff.
    camera = vtk.vtkCamera()
    camera.SetPosition(1, 1, 1)
    camera.SetFocalPoint(0, 0, 0)

    renderer = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(renderer)

    colorTuples = [(255,255,0), (255,0,0)]
    totalVerts = 0
    for ii, m in enumerate(ms):
        if hasattr(m, 'v'):
            x = np.array(m.v, dtype='float').tolist()
        else:
            x = np.array(m, dtype='float').tolist()
        totalVerts = totalVerts + len(x)
    for ii, m in enumerate(ms):
        # x = array of 8 3-tuples of float representing the vertices of a cube:
        if hasattr(m, 'v'):
            x = np.array(m.v, dtype='float').tolist()
        else:
            x = np.array(m, dtype='float').tolist()

        # pts = array of 6 4-tuples of vtkIdType (int) representing the faces
        #     of the cube in terms of the above vertices
        pts = np.array(m.f, dtype='int').tolist()

        # We'll create the building blocks of polydata including data attributes.
        cube = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        scalars = vtk.vtkFloatArray()
        scalars.SetNumberOfValues(3)
        scalars.SetNumberOfTuples(totalVerts)

        if False:
            Colors = vtk.vtkUnsignedCharArray()
            Colors.SetNumberOfComponents(3)
            Colors.SetName("Colors")
            Colors.SetVectors(255, 0, 0)
            cube.GetPointData().SetVectors(Colors)

        # Load the point, cell, and data attributes.
        for i in range(len(x)):
            points.InsertPoint(i, x[i])
        for i in range(len(pts)):
            polys.InsertNextCell(mkVtkIdList(pts[i]))
        for i in range(len(x)):
            scalars.InsertTuple3(i, colorTuples[ii][0], colorTuples[ii][1], colorTuples[ii][2])

        # We now assign the pieces to the vtkPolyData.
        cube.SetPoints(points)
        del points
        cube.SetPolys(polys)
        del polys
        cube.GetPointData().SetScalars(scalars)
        del scalars

        # Now we'll look at it.
        cubeMapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            cubeMapper.SetInput(cube)
        else:
            cubeMapper.SetInputData(cube)
        cubeMapper.SetScalarRange(0, len(x)-1)
        cubeActor = vtk.vtkActor()
        cubeActor.SetMapper(cubeMapper)

        renderer.AddActor(cubeActor)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    renderer.SetActiveCamera(camera)
    renderer.ResetCamera()
    renderer.SetBackground(1, 1, 1)

    renWin.SetSize(300, 300)

    # interact with data
    renWin.Render()
    iren.Start()

    # Clean up
    del camera
    del renderer
    del renWin
    del iren

def open3dVisualizePcl(pclList, np_colorsList=[np.array([0, 0, 255])]):
    assert isinstance(pclList, list)
    assert isinstance(np_colorsList, list)
    o3dPclList = []
    for i, pcl in enumerate(pclList):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pcl)
        pcd.colors = open3d.utility.Vector3dVector(np_colorsList[i])
        o3dPclList.append(pcd)
    open3d.visualization.draw_geometries(o3dPclList)
