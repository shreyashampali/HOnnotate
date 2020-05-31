import numpy

import cv2



class PNP(object):

    def __init__(self, pts2D, pts3D, K, eval_prefix='', reproj_err=8.0):
        self.points_3D = numpy.asarray(pts3D, dtype='float32')
        self.points_2D = numpy.asarray(pts2D, dtype='float32')
        self.K = numpy.asarray(K, dtype='float32')
        self.distCoeffs = numpy.zeros((8, 1), dtype='float32')
        self.eval_prefix = eval_prefix
        self.reproj_err = reproj_err
        self.iterationsCount = None
        self.minInliersCount = None

    def pnp(self):

        _, rvec, tvec = cv2.solvePnP(self.points_3D,
                                     self.points_2D,
                                     self.K,
                                     self.distCoeffs)
        R_pred, _ = cv2.Rodrigues(rvec)
        Rt = numpy.zeros((3, 4), dtype='float32')
        Rt[:, :3] = R_pred
        Rt[:, 3] = tvec.reshape(3, )
        return Rt

    def pnp_ransac(self):
        if int(cv2.__version__.split('.')[0]) < 3:
            rvec, tvec, _ = cv2.solvePnPRansac(self.points_3D,
                                               self.points_2D,
                                               self.K,
                                               self.distCoeffs,
                                               # minInliersCount=self.minInliersCount,
                                               # iterationsCount=self.iterationsCount,
                                               reprojectionError=self.reproj_err)
        else:
            _, rvec, tvec, _ = cv2.solvePnPRansac(self.points_3D,
                                                  self.points_2D,
                                                  self.K,
                                                  self.distCoeffs,
                                                  # minInliersCount=self.minInliersCount,
                                                  # iterationsCount=self.iterationsCount,
                                                  reprojectionError=self.reproj_err)

        R_pred, _ = cv2.Rodrigues(rvec)
        Rt = numpy.zeros((3, 4), dtype='float32')
        Rt[:, :3] = R_pred
        Rt[:, 3] = tvec.reshape(3, )
        return Rt

    @staticmethod
    def solveBatch(pts2D, pts3D, K, mode='pnp', reproj_errors=None, default_poses=None,
                   R_c=None, T_c=None):
        assert mode in ['pnp', 'ransac']
        if pts2D.shape[1] == pts3D.shape[0]:
            pts3D = pts3D[None, :, :].repeat(pts2D.shape[0], axis=0)
        elif pts2D.shape[0] == pts3D.shape[0]:
            assert pts2D.shape[1] == pts3D.shape[1]
        else:
            raise RuntimeError('Shapes do not match: {} != {}'.format(pts2D.shape, pts3D.shape))
        if R_c is not None or T_c is not None:
            assert R_c is not None and T_c is not None
        poses = []
        for i in range(pts2D.shape[0]):
            pnp = PNP(pts2D[i], pts3D[i], K)
            if mode == 'pnp':
                poses.append(pnp.pnp())
            elif mode == 'ransac':
                poses.append(pnp.pnp_ransac())
            else:
                raise NotImplementedError()
            imgpts, jac = cv2.projectPoints(pts3D[i], cv2.Rodrigues(poses[-1][:, :3])[0], poses[-1][:, 3], K, pnp.distCoeffs)
            reproj_err = numpy.linalg.norm(pts2D[i] - imgpts[:, 0, :], axis=1)
            if reproj_errors is not None:
                assert reproj_errors.shape[0] == pts2D.shape[0]
                reproj_errors[i] = numpy.mean(reproj_err)
            if default_poses is not None:
                assert default_poses.shape == pts2D.shape
                if poses[-1][2, 3] < 0.5 or poses[-1][2, 3] > 2.:  # numpy.any(reproj_err > 128.) or
                    poses[-1] = PNP(default_poses[i], pts3D[i], K).pnp_ransac()
                    imgpts, jac = cv2.projectPoints(pts3D[i], cv2.Rodrigues(poses[-1][:, :3])[0], poses[-1][:, 3], K, pnp.distCoeffs)
                    reproj_err = numpy.linalg.norm(default_poses[i] - imgpts[:, 0, :], axis=1)
                    if reproj_errors is not None:
                        reproj_errors[i] = numpy.mean(reproj_err)

        if R_c is not None:
            RT_c = numpy.eye(4)
            RT_c[:3, :3] = R_c
            RT_c[:3, 3] = T_c
            for i in range(len(poses)):
                poses[i] = numpy.dot(RT_c, numpy.vstack((poses[i], numpy.asarray([0., 0., 0., 1.]))))[:3, :]
        return numpy.asarray(poses)
