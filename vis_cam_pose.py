# This scripts is for draw the camera poses
import numpy as np
import cv2
import os


def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    T = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(T)

    return M_inv

def transform_to_matplotlib_frame(cMo, X, inverse=False):
    M = np.identity(4)
    # ! not need for me now
    # M[1, 1] = 0
    # M[1, 2] = 1
    # M[2, 1] = -1
    # M[2, 2] = 0

    if inverse:
        return M.dot(inverse_homogeneoux_matrix(cMo).dot(X))
    else:
        return M.dot(cMo.dot(X))

def create_camera_model(camera_matrix, width, height, scale_focal, draw_frame_axis=False):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    focal = 2 / (fx + fy)
    f_scale = scale_focal * focal

    # draw image plane
    X_img_plane = np.ones((4, 5))
    X_img_plane[0:3, 0] = [-width, height, f_scale]
    X_img_plane[0:3, 1] = [width, height, f_scale]
    X_img_plane[0:3, 2] = [width, -height, f_scale]
    X_img_plane[0:3, 3] = [-width, -height, f_scale]
    X_img_plane[0:3, 4] = [-width, height, f_scale]

    # draw triangle above the image plane
    X_triangle = np.ones((4, 3))
    X_triangle[0:3, 0] = [-width, -height, f_scale]
    X_triangle[0:3, 1] = [0, -2*height, f_scale]
    X_triangle[0:3, 2] = [width, -height, f_scale]

    # draw camera
    X_center1 = np.ones((4, 2))
    X_center1[0:3, 0] = [0, 0, 0]
    X_center1[0:3, 1] = [-width, height, f_scale]

    X_center2 = np.ones((4, 2))
    X_center2[0:3, 0] = [0, 0, 0]
    X_center2[0:3, 1] = [width, height, f_scale]

    X_center3 = np.ones((4, 2))
    X_center3[0:3, 0] = [0, 0, 0]
    X_center3[0:3, 1] = [width, -height, f_scale]

    X_center4 = np.ones((4, 2))
    X_center4[0:3, 0] = [0, 0, 0]
    X_center4[0:3, 1] = [-width, -height, f_scale]

    # draw camera frame axis
    X_frame1 = np.ones((4, 2))
    X_frame1[0:3, 0] = [0, 0, 0]
    X_frame1[0:3, 1] = [f_scale/2, 0, 0]

    X_frame2 = np.ones((4, 2))
    X_frame2[0:3, 0] = [0, 0, 0]
    X_frame2[0:3, 1] = [0, f_scale/2, 0]

    X_frame3 = np.ones((4, 2))
    X_frame3[0:3, 0] = [0, 0, 0]
    X_frame3[0:3, 1] = [0, 0, f_scale/2]

    if draw_frame_axis:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2, X_frame3]
    else:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]

def draw_camera(ax, camera_matrix, cam_width, cam_height, scale_focal,
                extrinsics,
                patternCentric=True,
                annotation=True):
    from matplotlib import cm

    min_values = np.zeros((3, 1))
    min_values = np.inf
    max_values = np.zeros((3, 1))
    max_values = -np.inf

    X_moving = create_camera_model(
        camera_matrix, cam_width, cam_height, scale_focal)

    cm_subsection = np.linspace(0.0, 1.0, extrinsics.shape[0])
    colors = [cm.jet(x) for x in cm_subsection]

    X_list = []
    for idx in range(extrinsics.shape[0]):
        cMo = extrinsics[idx]
        for i in range(len(X_moving)):
            X = np.zeros(X_moving[i].shape)
            for j in range(X_moving[i].shape[1]):
                X[0:4, j] = transform_to_matplotlib_frame(
                    cMo, X_moving[i][0:4, j], patternCentric)
            if i == 0:
                X_list.append(X[:, 0])
            ax.plot3D(X[0, :], X[1, :], X[2, :], color=colors[idx])
            min_values = np.minimum(min_values, X[0:3, :].min(1))
            max_values = np.maximum(max_values, X[0:3, :].max(1))
        # modified: add an annotation of number
        if annotation:
            X = transform_to_matplotlib_frame(
                cMo, X_moving[0][0:4, 0], patternCentric)
            ax.text(X[0], X[1], X[2], "{}".format(idx), color=colors[idx])

    X_list = np.stack(X_list, -1)  # [3, T]
    ax.plot3D(X_list[0, :], X_list[1, :], X_list[2, :], color='gray')
    return min_values, max_values
    
def draw_camera_traj(ax, camera_matrix, cam_width, cam_height, scale_focal,
                extrinsics,
                patternCentric=True,
                annotation=True):
    from matplotlib import cm

    min_values = np.zeros((3, 1))
    min_values = np.inf
    max_values = np.zeros((3, 1))
    max_values = -np.inf

    X_moving = create_camera_model(
        camera_matrix, cam_width, cam_height, scale_focal)

    cm_subsection = np.linspace(0.0, 1.0, extrinsics.shape[0])
    colors = [cm.jet(x) for x in cm_subsection]

    X_list = []
    for idx in range(extrinsics.shape[0]):
        cMo = extrinsics[idx]
        i = 0
        X = np.zeros(X_moving[i].shape)
        for j in range(X_moving[i].shape[1]):
            X[0:4, j] = transform_to_matplotlib_frame(
                cMo, X_moving[i][0:4, j], patternCentric)
        if i == 0:
            X_list.append(X[:, 0])
        min_values = np.minimum(min_values, X[0:3, :].min(1))
        max_values = np.maximum(max_values, X[0:3, :].max(1))
        # modified: add an annotation of number
        if annotation:
            X = transform_to_matplotlib_frame(
                cMo, X_moving[0][0:4, 0], patternCentric)
            ax.text(X[0], X[1], X[2], "{}".format(idx), color=colors[idx])

    X_list = np.stack(X_list, -1)  # [3, T]
    ax.plot3D(X_list[0, :], X_list[1, :], X_list[2, :], color='gray')
    return min_values, max_values


def visualize(camera_matrix, extrinsics, mesh=None, fname='./camera.png', traj=False):

    ########################    plot params     ########################
    cam_width = 0.064/2     # Width/2 of the displayed camera.
    cam_height = 0.048/2    # Height/2 of the displayed camera.
    scale_focal = 40        # Value to scale the focal length.

    ########################    original code    ########################
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-variable

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect("auto")

    if traj:
        min_values, max_values = draw_camera_traj(ax, camera_matrix, cam_width, cam_height,
                                         scale_focal, extrinsics, True)
    else:
        min_values, max_values = draw_camera(ax, camera_matrix, cam_width, cam_height,
                                         scale_focal, extrinsics, True)
    if mesh is not None:
        ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces)
    X_min = min_values[0]
    X_max = max_values[0]
    Y_min = min_values[1]
    Y_max = max_values[1]
    Z_min = min_values[2]
    Z_max = max_values[2]
    max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0
    # max_range = max(max_range, 2)

    mid_x = (X_max+X_min) * 0.5
    mid_y = (Y_max+Y_min) * 0.5
    mid_z = (Z_max+Z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    ax.set_title('Extrinsic Parameters Visualization')

    plt.show()
    print('Done')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    print('save to %s' % fname)
    plt.savefig(fname)


if __name__ == '__main__':
    seq_name = "MC6"
    from glob import glob
    import pickle as pkl
    import trimesh
    RT_list = []
    K_list = []
    pkl_lis = sorted(glob("/remote-home/jiangshijian/data/HO3D_v3/train/{}/meta/*.pkl".format(seq_name)))[:100]
    mesh = trimesh.load("/remote-home/jiangshijian/homan/local_data/datasets/ycbmodels/003_cracker_box/textured_simple_2000.obj", force="mesh")
    for pkl_path in pkl_lis:
        with open(pkl_path, "rb") as f:
            data = pkl.load(f, encoding="latin1")
        R_gt = data["objRot"]
        R_gt, _ = cv2.Rodrigues(R_gt)
        T_gt = data["objTrans"].reshape(3, 1)
        RT_gt = np.concatenate([R_gt, T_gt], axis=1)
        # transform the coordinate
        RT_gt[1:] *= -1
        RT_gt = np.concatenate([RT_gt, np.array([[0, 0, 0, 1]])], axis=0)
        RT_list.append(RT_gt)
        K_list.append(data['camMat'])
    extrinsics = np.array(RT_list)
    camera_matrix = K_list[0]

    visualize(camera_matrix, extrinsics, mesh=mesh)