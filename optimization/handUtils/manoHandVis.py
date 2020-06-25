import open3d as o3d
from matplotlib.colors import rgb2hex
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np
from eval import utilsEval
import os
import chumpy as ch
import cv2
import pyrender
import trimesh
import math

jointsMap = [0,
             13, 14, 15, 16,
             1, 2, 3, 17,
             4, 5, 6, 18,
             10, 11, 12, 19,
             7, 8, 9, 20]

jointsMapManoToObman = [0,
                        13, 14, 15, 16,
                        1, 2, 3, 17,
                        4, 5, 6, 18,
                        10, 11, 12, 19,
                        7, 8, 9, 20]

vc_reg = np.reshape(np.random.uniform(0., 1., 778 * 3), (778, 3))


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
            # self.camera.set_projection_matrix(projMat)
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

def open3dVisualize(m):
    import open3d
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(np.copy(m.r))
    mesh.triangles = open3d.utility.Vector3iVector(np.copy(m.f))
    mesh.vertex_colors = open3d.utility.Vector3dVector(np.copy(vc_reg[:, [2, 1, 0]]))
    o3d.visualization.draw_geometries([mesh])

def open3dVisualizePcl(pcl, np_colors=np.array([0, 0, 255])):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    pcd.colors = o3d.utility.Vector3dVector(np_colors)
    o3d.visualization.draw_geometries([pcd])

def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)

def plot3dVisualize(ax, m, flip_x=False, c="viridis", alpha=0.1, camPose=np.eye(4, dtype=np.float32)):
    verts = np.copy(m.r)*1000
    vertsHomo = np.concatenate([verts, np.ones((verts.shape[0],1), dtype=np.float32)], axis=1)
    verts = vertsHomo.dot(camPose.T)[:,:3]

    faces = np.copy(m.f)
    ax.view_init(elev=90, azim=-90)
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    if c == "b":
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (0 / 255, 0 / 255, 112 / 255)
    elif c == "r":
        face_color = (226 / 255, 141 / 255, 141 / 255)
        edge_color = (112 / 255, 0 / 255, 0 / 255)
    elif c == "viridis":
        face_color = plt.cm.viridis(np.linspace(0, 1, faces.shape[0]))
        edge_color = None
        edge_color = (0 / 255, 0 / 255, 112 / 255)
    elif c == "plasma":
        face_color = plt.cm.plasma(np.linspace(0, 1, faces.shape[0]))
        edge_color = None
        # edge_color = (0 / 255, 0 / 255, 112 / 255)
    else:
        face_color = c
        edge_color = c

    mesh.set_color(np.concatenate([vc_reg[:,[2,1,0]], 0.5*np.ones((vc_reg.shape[0],1), dtype=np.float32)], axis=1))
    # mesh.set_edgecolor(vc_reg)
    mesh.set_alpha(0.0)
    # mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    cam_equal_aspect_3d(ax, verts, flip_x=flip_x)
    # plt.tight_layout()


def dump3DModel2DKpsHand(img, m, filename, camMat, est2DJoints = None, gt2DJoints = None, outDir=None, camPose=np.eye(4, dtype=np.float32)):
    '''
    Saves the following in this order depending on availibility
    1. GT 2D joint locations
    2. 2D joint locations as estimated by the CPM
    3. 2D joint locations after fitting the MANO model to the estimation 2D joints
    4. 3D model of the hand in the estimated pose
    :param img:
    :param m:
    :param filename:
    :param camMat:
    :param est2DJoints:
    :param gt2DJoints:
    :param outDir:
    :return:
    '''
    if outDir is not None:
        plt.ioff()
    fig = plt.figure(figsize=(2, 2))
    figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    plot3dVisualize(ax, m, flip_x=False, camPose=camPose)
    ax.title.set_text("3D mesh")

    J3DHomo = ch.concatenate([m.J_transformed, np.ones((21, 1))], axis=1)
    J3DTrans = J3DHomo.dot(camPose.T)[:, :3]
    projPts = utilsEval.chProjectPoints(J3DTrans, camMat, False)[jointsMap]

    axEst = fig.add_subplot(2, 2, 3)
    imgOutEst = utilsEval.showHandJoints(img.copy(), np.copy(projPts.r).astype(np.float32), estIn=None,
                                         filename=None,
                                         upscale=1, lineThickness=3)
    axEst.imshow(imgOutEst[:, :, [2, 1, 0]])
    axEst.title.set_text("After fitting")

    if est2DJoints is not None:
        axGT = fig.add_subplot(2, 2, 2)
        imgOutGt = utilsEval.showHandJoints(img.copy(), est2DJoints.astype(np.float32), estIn=None,
                                            filename=None,
                                            upscale=1, lineThickness=3)
        axGT.imshow(imgOutGt[:, :, [2, 1, 0]])
        axGT.title.set_text("Before fitting")

    if gt2DJoints is not None:
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(gt2DJoints[:, :, [2, 1, 0]])
        ax1.title.set_text("Ground Truth")

    if outDir is not None:
        if len(filename.split('/')) > 1:
            if len(filename.split('/')) == 2:
                if not os.path.exists(os.path.join(outDir, filename.split('/')[0])):
                    os.mkdir(os.path.join(outDir, filename.split('/')[0]))
            elif len(filename.split('/')) == 3:
                if not os.path.exists(os.path.join(outDir, filename.split('/')[0])):
                    os.mkdir(os.path.join(outDir, filename.split('/')[0]))
                if not os.path.exists(os.path.join(outDir, filename.split('/')[0], filename.split('/')[1])):
                    os.mkdir(os.path.join(outDir, filename.split('/')[0], filename.split('/')[1]))
            else:
                raise NotImplementedError

        fig.set_size_inches((11, 8.5), forward=False)
        # plt.show()
        plt.savefig(os.path.join(outDir, filename + '.jpg'), dpi=300)
        plt.close(fig)
    else:
        plt.show()
        
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









