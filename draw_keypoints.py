import cv2
import matplotlib.pyplot as plt
import json
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np

rgb_blue = (0,176,240)
rgb_pink = (252,176,243)
rgb_purple = (175, 85, 255)
rgb_green = (169, 209, 142)
rgb_yellow = (255,255,0)
rgb_red = (255, 0, 0)

point_color_mpii = [rgb_pink,rgb_pink,rgb_pink,
            rgb_blue, rgb_blue, rgb_blue,
            rgb_purple, rgb_purple,
            rgb_red, rgb_red,
            rgb_yellow, rgb_yellow, rgb_yellow,
            rgb_green, rgb_green, rgb_green]

order = {}
order['ankle_r'] = 0
order['hip_r'] = 2
order['hip_l'] = 3
order['knee_r'] = 1
order['knee_l'] = 4
order['ankle_l'] = 5
order['pelvis'] = 6
order['thorax'] = 7
order['upper neck'] = 8
order['head top'] = 9
order['wrist_r'] = 10
order['elbow_r'] = 11
order['shoulder_r'] = 12
order['shoulder_l'] = 13
order['elbow_l'] = 14
order['wrist_l'] = 15

def draw_part(img, ax, skeleton, name_a, name_b, BGR=(0, 0, 0)):
    a = skeleton[name_a]
    b = skeleton[name_b]
    # if a[2]<0.5 or b[2]<0.5:
    # if a[1]< 1.0 or b[1] <1.0 or a[0]< 1.0 or b[0] <1.0:
    if abs(a[1] - b[1]) > 200 or abs(a[0] - b[0]) > 200:
        return
    # cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), BGR, 5, cv2.LINE_AA)
    line = mlines.Line2D(
        np.array([int(a[0]), int(b[0])]),
        np.array([int(a[1]), int(b[1])]),
        ls='-', lw=4, alpha=1, color=tuple(np.array(BGR)/255.), )
    line.set_zorder(0)
    ax.add_line(line)


def draw_skeleton(img, ax, kpts):
    # for coco

    # skeleton = {}
    # skeleton['nose']=kpts[0]
    # skeleton['l_eye']=  kpts[1]
    # skeleton['r_eye']=  kpts[2]
    # skeleton['l_ear']= kpts[3]
    # skeleton['r_ear']= kpts[4]
    # skeleton['l_shoulder']=kpts[5]
    # skeleton['r_shoulder']=kpts[6]
    # skeleton['l_elbow']=kpts[7]
    # skeleton['r_elbow']=kpts[8]
    # skeleton['l_wrist']=kpts[9]
    # skeleton['r_wrist']=kpts[10]
    # skeleton['l_hip']=kpts[11]
    # skeleton['r_hip']=kpts[12]
    # skeleton['l_knee']=kpts[13]
    # skeleton['r_knee']=kpts[14]
    # skeleton['l_ankle']=kpts[15]
    # skeleton['r_ankle']=kpts[16]
    #
    # rgb1 = (0,255,0)
    # rgb2 = (255,144,30)
    # rgb3 = (92,92,205)
    # draw_part(img,skeleton,'nose','l_eye',BGR=rgb1)
    # draw_part(img,skeleton,'nose','r_eye',BGR=rgb1)
    # draw_part(img,skeleton,'r_eye','r_ear',BGR=rgb1)
    # draw_part(img,skeleton,'l_eye','l_ear',BGR=rgb1)
    # draw_part(img,skeleton,'r_shoulder','r_ear',BGR=rgb1)
    # draw_part(img,skeleton,'l_shoulder','l_ear',BGR=rgb1)
    #
    # draw_part(img,skeleton,'r_shoulder','r_elbow',BGR=rgb2)
    # draw_part(img,skeleton,'r_elbow','r_wrist',BGR=rgb2)
    # draw_part(img,skeleton,'l_shoulder','l_elbow',BGR=rgb2)
    # draw_part(img,skeleton,'l_elbow','l_wrist',BGR=rgb2)
    #
    # draw_part(img,skeleton,'l_hip','l_knee',BGR=rgb3)
    # draw_part(img,skeleton,'l_ankle','l_knee',BGR=rgb3)
    # draw_part(img,skeleton,'r_hip','r_knee',BGR=rgb3)
    # draw_part(img,skeleton,'r_ankle','r_knee',BGR=rgb3)

    # for mpii

    skeleton = {}

    skeleton['ankle_r'] = kpts[0]
    skeleton['hip_r'] = kpts[2]
    skeleton['hip_l'] = kpts[3]
    skeleton['knee_r'] = kpts[1]
    skeleton['knee_l'] = kpts[4]
    skeleton['ankle_l'] = kpts[5]
    skeleton['pelvis'] = kpts[6]
    skeleton['thorax'] = kpts[7]
    skeleton['upper neck'] = kpts[8]
    skeleton['head top'] = kpts[9]
    skeleton['wrist_r'] = kpts[10]
    skeleton['elbow_r'] = kpts[11]
    skeleton['shoulder_r'] = kpts[12]
    skeleton['shoulder_l'] = kpts[13]
    skeleton['elbow_l'] = kpts[14]
    skeleton['wrist_l'] = kpts[15]

    #
    #
    draw_part(img, ax, skeleton, 'hip_l', 'pelvis', BGR=rgb_blue)
    draw_part(img, ax, skeleton, 'hip_r', 'pelvis', BGR=rgb_pink)
    draw_part(img, ax, skeleton, 'hip_r', 'knee_r', BGR=rgb_pink)
    draw_part(img, ax, skeleton, 'hip_l', 'knee_l', BGR=rgb_blue)
    draw_part(img, ax, skeleton, 'ankle_r', 'knee_r', BGR=rgb_pink)
    draw_part(img, ax, skeleton, 'ankle_l', 'knee_l', BGR=rgb_blue)

    draw_part(img, ax, skeleton, 'shoulder_r', 'thorax', BGR=rgb_yellow)
    draw_part(img, ax, skeleton, 'shoulder_l', 'thorax', BGR=rgb_green)
    draw_part(img, ax, skeleton, 'shoulder_r', 'elbow_r', BGR=rgb_yellow)
    draw_part(img, ax, skeleton, 'shoulder_l', 'elbow_l', BGR=rgb_green)
    draw_part(img, ax, skeleton, 'wrist_r', 'elbow_r', BGR=rgb_yellow)
    draw_part(img, ax, skeleton, 'wrist_l', 'elbow_l', BGR=rgb_green)

    draw_part(img, ax, skeleton, 'thorax', 'upper neck', BGR=rgb_red)
    draw_part(img, ax, skeleton, 'upper neck', 'head top', BGR=rgb_red)

    draw_part(img, ax, skeleton, 'thorax', 'pelvis', BGR=rgb_purple)

    # draw_part(img, skeleton, 'shoulder_r', 'hip_r', BGR=)
    # draw_part(img, skeleton, 'shoulder_l', 'hip_l', BGR=rgb2)



def draw(filename, img, kpts):
    # mpii
    ax = plt.subplot(1, 1, 1)
    bk = plt.imshow(img[:, :, ::-1])
    bk.set_zorder(-1)
    draw_skeleton(img, ax, kpts)
    for id, p in enumerate(kpts):
        if id == 15:
            circle = mpatches.Circle(tuple(p),
                                     radius=12,
                                     ec='black',
                                     fc=tuple(np.array(point_color_mpii[id]) / 255.),
                                     alpha=1,
                                     linewidth=2)
        else:
            circle = mpatches.Circle(tuple(p),
                                 radius=6,
                                 ec='black',
                                 fc=tuple(np.array(point_color_mpii[id])/255.),
                                 alpha=1,
                                 linewidth=1)
        circle.set_zorder(1)
        ax.add_patch(circle)
        # cv2.circle(img, (int(p[0]), int(p[1])), 5, point_color_mpii[id], 3)
        # cv2.circle(img, (int(p[0]), int(p[1])), 5, (255,255,255), 1)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.savefig('mpii_results/pred/{}'.format(filename), format='jpg',
    #                 bbox_inckes='tight', dpi=100)

    plt.savefig('mpii_results/pred/{}'.format(filename), format='jpg',
                    bbox_inckes='tight', dpi=100)
    # plt.show()
    plt.close()

    return img