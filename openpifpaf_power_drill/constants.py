import os

import numpy as np
try:
    import matplotlib.cm as mplcm
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    pass

import openpifpaf

from .transforms import transform_skeleton

POWER_DRILL_KEYPOINTS_24 = [
    '1',       # 1
    '2',        # 2
    '3',    # 3
    '4',     # 4
    '5',      # 5
    '6',       # 6
    '7',      # 7
    '8',     # 8
    '9',      # 9
]

SKELETON_ORIG = [
    [1, 4], [1, 5], [4, 5], [4, 7], [5, 8], [7, 8], [4, 6], [5, 6], [6, 9], [9, 3], [3, 2], [2, 9]
]


KPS_MAPPING = [1, 2, 3, 4, 5, 6, 7, 8, 9]

POWER_DRILL_SKELETON_24 = SKELETON_ORIG

POWER_DRILL_SIGMAS_24 = [0.05] * 9

split, error = divmod(len(POWER_DRILL_KEYPOINTS_24), 4)
POWER_DRILL_SCORE_WEIGHTS_24 = [10.0] * split + [3.0] * split + \
    [1.0] * split + [0.1] * split + [0.1] * error
assert len(POWER_DRILL_SCORE_WEIGHTS_24) == len(POWER_DRILL_KEYPOINTS_24)

HFLIP_24 = {
}

POWER_DRILL_CATEGORIES_24 = ['power_drill']

p = 0.25
FRONT = -6.0
BACK = 4.5

# POWER_DRILL POSE is used for joint rescaling. x = [-3, 3] y = [0,4]
POWER_DRILL_POSE_24 = np.array([
    [-2.9, 4.0, FRONT * 0.5],  # 'front_up_right',              # 1
    [2.9, 4.0, FRONT * 0.5],   # 'front_up_left',               # 2
    [-2.0, 2.0, FRONT],  # 'front_light_right',           # 3
    [2.0, 2.0, FRONT],  # 'front_light_left',             # 4
    [-2.5, 0.0, FRONT],  # 'front_low_right',             # 5
    [2.5, 0.0, FRONT],  # 'front_low_left',              # 6
    [2.6, 4.2, 0.0],  # 'central_up_left'     # 7
    [3.2, 0.2, FRONT * 0.7],  # 'front_wheel_left',           # 8
    [3.0, 0.3, BACK * 0.7],   # 'rear_wheel_left'      # 9
    [3.1, 2.1, BACK * 0.5],   # 'rear_corner_left',          # 10
    [2.4, 4.3, BACK * 0.35],  # 'rear_up_left',       # 11
    [-2.4, 4.3, BACK * 0.35],  # 'rear_up_right'      # 12
    [2.5, 2.2, BACK],   # 'rear_light_left',             # 13
    [-2.5, 2.2, BACK],  # 'rear_light_right',            # 14
    [2.1, 0.1, BACK],  # 'rear_low_left',            # 15
    [-2.1, 0.1, BACK],  # 'rear_low_right',          # 16
    [-2.6, 4.2, 0.0],  # 'central_up_right'    # 17
    [-3.1, 2.1, BACK * 0.5],  # 'rear_corner_right',         # 18
    [-3.0, 0.3, BACK * 0.7],  # 'rear_wheel_right'       # 19
    [-3.2, 0.2, FRONT * 0.7],  # 'front_wheel_right',          # 20
    [1.0, 1.3, BACK],  # 'rear_plate_left',              # 21
    [-1.0, 1.3, BACK],  # 'rear_plate_right',            # 22
    [2.8, 3, FRONT * 0.35],  # 'mirror_edge_left'          # 23
    [-2.8, 3, FRONT * 0.35],  # 'mirror_edge_right'        # 24
])

POWER_DRILL_POSE_FRONT_24 = np.array([
    [-2.0, 4.0, 2.0],  # 'front_up_right',         # 1
    [2.0, 4.0, 2.0],   # 'front_up_left',        # 2
    [-1.3, 2.0, 2.0],  # 'front_light_right',    # 3
    [1.3, 2.0, 2.0],  # 'front_light_left',     # 4
    [-2.2, 0.0, 2.0],  # 'front_low_right',       # 5
    [2.2, 0.0, 2.0],  # 'front_low_left',       # 6
    [2.0 - p / 2, 4.0 + p, 1.0],  # 'central_up_left',      # 7
    [2.0 + p, 0.1 - p / 2, 1.0],  # 'front_wheel_left',     # 8
    [2, 0.1, 0.0],  # 'rear_wheel_left',      # 9
    [2.6, 1.7, 0.0],   # 'rear_corner_left',          # 10
    [2.0, 4.1, 0.0],  # 'rear_up_left',         # 11
    [-2.0, 4.0, 0.0],  # 'rear_up_right',        # 12
    [2.1, 1.9, 0.0],   # 'rear_light_left',      # 13
    [-2.1, 1.9, 0.0],  # 'rear_right_right',     # 14
    [2.4, 0.1, 0.0],  # 'rear_low_left',        # 15
    [-2.4, 0.1, 0.0],  # 'rear_low_right',       # 16
    [-2.0 + p / 2, 4.0 + p, 1.0],  # 'central_up_right',     # 17
    [-2.6, 1.75, 0.0],  # 'rear_corner_right',           # 18
    [-2, 0.0, 0.0],  # 'rear_wheel_right',     # 19
    [-2 - p, 0.0 - p / 2, 1.0],  # 'front_wheel_right',     # 20
])

POWER_DRILL_POSE_REAR_24 = np.array([
    [-2.0, 4.0, 0.0],  # 'front_up_right',         # 1
    [2.0, 4.0, 0.0],   # 'front_up_left',        # 2
    [-1.3, 2.0, 0.0],  # 'front_light_right',    # 3
    [1.3, 2.0, 0.0],  # 'front_light_left',     # 4
    [-2.2, 0.0, 0.0],  # 'front_low_right',       # 5
    [2.2, 0.0, 0.0],  # 'front_low_left',       # 6
    [-2.0 + p, 4.0 + p, 2.0],  # 'central_up_left',      # 7
    [2, 0.0, 0.0],  # 'front_wheel_left',     # 8
    [2, 0.0, 0.0],  # 'rear_wheel_left',      # 9
    [-1.6 - p, 2.2 - p, 2.0],   # 'rear_corner_left',     # 10
    [-2.0, 4.0, 2.0],  # 'rear_up_left',         # 11
    [2.0, 4.0, 2.0],  # 'rear_up_right',        # 12
    [-1.6, 2.2, 2.0],   # 'rear_light_left',      # 13
    [1.6, 2.2, 2.0],  # 'rear_right_right',     # 14
    [-2.4, 0.0, 2.0],  # 'rear_low_left',        # 15
    [2.4, 0.0, 2.0],  # 'rear_low_right',       # 16
    [2.0 - p, 4.0 + p, 2.0],  # 'central_up_right',     # 17
    [1.6 + p, 2.2 - p, 2.0],  # 'rear_corner_right', # 18
    [-2, 0.0, 0.0],  # 'rear_wheel_right',     # 19
    [-2, 0.0, 0.0],  # 'front_wheel_right',     # 20
])

POWER_DRILL_POSE_LEFT_24 = np.array([
    [-2.0, 4.0, 0.0],  # 'front_up_right',         # 1
    [0 - 5 * p, 4.0 - p / 2, 2.0],   # 'front_up_left',        # 2
    [-1.3, 2.0, 0.0],  # 'front_light_right',    # 3
    [1.3, 2.0, 0.0],  # 'front_light_left',     # 4
    [-2.2, 0.0, 0.0],  # 'front_low_right',       # 5
    [-4 - 3 * p, 0.0, 2.0],   # 'front_low_left',       # 6
    [0, 4.0, 2.0],  # 'central_up_left',      # 7
    [-4, 0.0, 2.0],  # 'front_wheel_left',     # 8
    [4, 0.0, 2.0],  # 'rear_wheel_left',      # 9
    [5, 2, 2.0],  # 'rear_corner_left',     # 10
    [0 + 5 * p, 4.0 - p / 2, 2.0],  # 'rear_up_left',  # 11
    [2.0, 4.0, 0.0],  # 'rear_up_right',        # 12
    [5 + p, 2 + p, 1.0],   # 'rear_light_left',      # 13
    [1.6, 2.2, 0.0],  # 'rear_right_right',     # 14
    [-2.4, 0.0, 0.0],  # 'rear_low_left',        # 15
    [2.4, 0.0, 0.0],  # 'rear_low_right',       # 16
    [2.0, 4.0, 0.0],  # 'central_up_right',     # 17
    [1.6, 2.2, 0.0],  # 'rear_corner_right', # 18
    [-2, 0.0, 0.0],  # 'rear_wheel_right',     # 19
    [-2, 0.0, 0.0],  # 'front_wheel_right',     # 20
])


POWER_DRILL_POSE_RIGHT_24 = np.array([
[-0.101005911827,0.047009292990,0.003920878284],
[-0.037154082209,-0.107653088868,-0.013503172435],
[0.045423928648,-0.106373451650,-0.019496466964],
[0.002672282746,0.017537992448,-0.019606860355],
[0.001261682715,0.019342577085,0.018263733014],
[-0.015974249691,-0.012602778152,0.003121521091],
[0.074148103595,0.057585820556,-0.014557955787],
[0.074379354715,0.031028280035,-0.013706572354],
[0.052770089358,-0.070011772215,0.000104412517]])

POWER_DRILL_KEYPOINTS_66 = [
    '1',       # 1
    '2',        # 2
    '3',    # 3
    '4',     # 4
    '5',      # 5
    '6',       # 6
    '7',      # 7
    '8',     # 8
    '9',      # 9
]


HFLIP_ids = {
    
}

HFLIP_66 = {}

POWER_DRILL_CATEGORIES_66 = ['car']

SKELETON_ALL = [[1, 4], [1, 5], [4, 5], [4, 7], [5, 8], [7, 8], [4, 6], [5, 6], [6, 9], [9, 3], [3, 2], [2, 9]]

POWER_DRILL_SKELETON_66 = SKELETON_ALL  # COCO style skeleton

POWER_DRILL_SIGMAS_66 = [0.05] * len(POWER_DRILL_KEYPOINTS_66)

split, error = divmod(len(POWER_DRILL_KEYPOINTS_66), 4)
POWER_DRILL_SCORE_WEIGHTS_66 = [10.0] * split + [3.0] * split + \
    [1.0] * split + [0.1] * split + [0.1] * error
assert len(POWER_DRILL_SCORE_WEIGHTS_66) == len(POWER_DRILL_KEYPOINTS_66)


# number plate offsets
P_X = 0.3
P_Y_TOP = -0.2
P_Y_BOTTOM = -0.4

# z for front
FRONT_Z = -2.0
FRONT_Z_SIDE = -1.8
FRONT_Z_CORNER = -1.7
FRONT_Z_WHEEL = -1.4
FRONT_Z_DOOR = -1.0

# lights x offset
LIGHT_X_INSIDE = 0.8
X_OUTSIDE = 1.0

# y offsets
TOP_POWER_DRILL = 0.5
BOTTOM_LINE = -0.75
TOP_LINE = 0.1

# z for the back
BACK_Z_WHEEL = 1.0
BACK_Z = 1.5
BACK_Z_SIDE = 1.3

POWER_DRILL_POSE_HALF = np.array([
[-0.101005911827,0.047009292990,0.003920878284],
[-0.037154082209,-0.107653088868,-0.013503172435],
[0.045423928648,-0.106373451650,-0.019496466964],
[0.002672282746,0.017537992448,-0.019606860355],
[0.001261682715,0.019342577085,0.018263733014],
[-0.015974249691,-0.012602778152,0.003121521091],
[0.074148103595,0.057585820556,-0.014557955787],
[0.074379354715,0.031028280035,-0.013706572354],
[0.052770089358,-0.070011772215,0.000104412517]]) 

POWER_DRILL_POSE_66 = POWER_DRILL_POSE_HALF
assert not np.any(POWER_DRILL_POSE_66 == np.nan)

training_weights_local_centrality = [
    0.890968488270775,
    0.716506138617812,
    1.05674590410869,
    0.764774195768455,
    0.637682585483328,
    0.686680807728366,
    0.955422595797394,
    0.936714585642375,
    1.34823795445326,
]


def get_constants(num_kps):
    if num_kps == 24:
        POWER_DRILL_POSE_24[:, 2] = 2.0
        return [POWER_DRILL_KEYPOINTS_24, POWER_DRILL_SKELETON_24, HFLIP_24, POWER_DRILL_SIGMAS_24,
                POWER_DRILL_POSE_24, POWER_DRILL_CATEGORIES_24, POWER_DRILL_SCORE_WEIGHTS_24]
    if num_kps == 66:
        POWER_DRILL_POSE_66[:, 2] = 2.0
        return [POWER_DRILL_KEYPOINTS_66, POWER_DRILL_SKELETON_66, HFLIP_66, POWER_DRILL_SIGMAS_66,
                POWER_DRILL_POSE_66, POWER_DRILL_CATEGORIES_66, POWER_DRILL_SCORE_WEIGHTS_66]
    # using no if-elif-else construction due to pylint no-else-return error
    raise Exception("Only poses with 24 or 66 keypoints are available.")


def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    bbox = ann.bbox()
    xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
    ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
    if aspect == 'equal':
        fig_w = 5.0
    else:
        fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])

    with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if aspect is not None:
            ax.set_aspect(aspect)

        keypoint_painter.annotation(ax, ann)


def draw_skeletons(pose, sigmas, skel, kps, scr_weights):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    show.KeypointPainter.show_joint_scales = True
    keypoint_painter = show.KeypointPainter()
    ann = Annotation(keypoints=kps, skeleton=skel, score_weights=scr_weights)
    ann.set(pose, np.array(sigmas) * scale)
    os.makedirs('docs', exist_ok=True)
    draw_ann(ann, filename='docs/skeleton_car.png', keypoint_painter=keypoint_painter)


def plot3d_red(ax_2D, p3d, skeleton):
    skeleton = [(bone[0] - 1, bone[1] - 1) for bone in skeleton]

    rot_p90_x = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    p3d = p3d @ rot_p90_x

    fig = ax_2D.get_figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_axis_off()
    ax_2D.set_axis_off()

    ax.view_init(azim=-90, elev=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_range = np.array([p3d[:, 0].max() - p3d[:, 0].min(),
                          p3d[:, 1].max() - p3d[:, 1].min(),
                          p3d[:, 2].max() - p3d[:, 2].min()]).max() / 2.0
    mid_x = (p3d[:, 0].max() + p3d[:, 0].min()) * 0.5
    mid_y = (p3d[:, 1].max() + p3d[:, 1].min()) * 0.5
    mid_z = (p3d[:, 2].max() + p3d[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)  # pylint: disable=no-member

    for ci, bone in enumerate(skeleton):
        c = mplcm.get_cmap('tab20')((ci % 20 + 0.05) / 20)  # Same coloring as Pifpaf preds
        ax.plot(p3d[bone, 0], p3d[bone, 1], p3d[bone, 2], color=c)

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig

    return FuncAnimation(fig, animate, frames=360, interval=100)


def print_associations():
    print("\nAssociations of the car skeleton with 24 keypoints")
    for j1, j2 in POWER_DRILL_SKELETON_24:
        print(POWER_DRILL_KEYPOINTS_24[j1 - 1], '-', POWER_DRILL_KEYPOINTS_24[j2 - 1])
    print("\nAssociations of the car skeleton with 66 keypoints")
    for j1, j2 in POWER_DRILL_SKELETON_66:
        print(POWER_DRILL_KEYPOINTS_66[j1 - 1], '-', POWER_DRILL_KEYPOINTS_66[j2 - 1])


def main():
    print_associations()
# =============================================================================
#     draw_skeletons(POWER_DRILL_POSE_24, sigmas = POWER_DRILL_SIGMAS_24, skel = POWER_DRILL_SKELETON_24,
#                    kps = POWER_DRILL_KEYPOINTS_24, scr_weights = POWER_DRILL_SCORE_WEIGHTS_24)
#     draw_skeletons(POWER_DRILL_POSE_66, sigmas = POWER_DRILL_SIGMAS_66, skel = POWER_DRILL_SKELETON_66,
#                    kps = POWER_DRILL_KEYPOINTS_66, scr_weights = POWER_DRILL_SCORE_WEIGHTS_66)
# =============================================================================
    with openpifpaf.show.Canvas.blank(nomargin=True) as ax_2D:
        anim_66 = plot3d_red(ax_2D, POWER_DRILL_POSE_66, POWER_DRILL_SKELETON_66)
        anim_66.save('openpifpaf/plugins/apollocar3d/docs/POWER_DRILL_66_Pose.gif', fps=30)
    with openpifpaf.show.Canvas.blank(nomargin=True) as ax_2D:
        anim_24 = plot3d_red(ax_2D, POWER_DRILL_POSE_24, POWER_DRILL_SKELETON_24)
        anim_24.save('openpifpaf/plugins/apollocar3d/docs/POWER_DRILL_24_Pose.gif', fps=30)


if __name__ == '__main__':
    main()
