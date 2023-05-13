import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

g_N = 7
a = 2 / 7
N = 35

M911R = np.array([0.0000, - 0.0012, - 0.0109, - 0.0095, 0.0000, 0.0095, 0.0109, 0.0012, 0.0000,
                  - 0.0016, - 0.0254, - 0.0220, - 0.0110, 0.0000, 0.0110, 0.0220, 0.0254, 0.0016,
                  - 0.0215, - 0.0329, - 0.0219, - 0.0110, 0.0000, 0.0110, 0.0219, 0.0329, 0.0215,
                  - 0.0380, - 0.0329, - 0.0219, - 0.0110, 0.0000, 0.0110, 0.0219, 0.0329, 0.0380,
                  - 0.0434, - 0.0329, - 0.0219, - 0.0110, 0.0000, 0.0110, 0.0219, 0.0329, 0.0434,
                  - 0.0380, - 0.0329, - 0.0219, - 0.0110, 0.0000, 0.0110, 0.0219, 0.0329, 0.0380,
                  - 0.0215, - 0.0329, - 0.0219, - 0.0110, 0.0000, 0.0110, 0.0219, 0.0329, 0.0215,
                  - 0.0016, - 0.0254, - 0.0220, - 0.0110, 0.0000, 0.0110, 0.0220, 0.0254, 0.0016,
                  0.0000, - 0.0012, - 0.0109, - 0.0095, 0.0000, 0.0095, 0.0109, 0.0012, 0.0000]).reshape(9, 9)

M911I = np.array([0.0000, 0.0016, 0.0215, 0.0380, 0.0434, 0.0380, 0.0215, 0.0016, 0.0000,
                  0.0012, 0.0254, 0.0329, 0.0329, 0.0329, 0.0329, 0.0329, 0.0254, 0.0012,
                  0.0109, 0.0219, 0.0219, 0.0219, 0.0219, 0.0219, 0.0219, 0.0219, 0.0109,
                  0.0094, 0.0110, 0.0110, 0.0110, 0.0110, 0.0110, 0.0110, 0.0110, 0.0094,
                  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                  - 0.0094, - 0.0110, - 0.0110, - 0.0110, - 0.0110, - 0.0110, - 0.0110, - 0.0110, - 0.0094,
                  - 0.0109, - 0.0219, - 0.0219, - 0.0219, - 0.0219, - 0.0219, - 0.0219, - 0.0219, - 0.0109,
                  - 0.0012, - 0.0254, - 0.0329, - 0.0329, - 0.0329, - 0.0329, - 0.0329, - 0.0254, - 0.0012,
                  - 0.0000, - 0.0016, - 0.0215, - 0.0380, - 0.0434, - 0.0380, - 0.0215, - 0.0016, - 0.0000]).reshape(9, 9)

M920 = np.array([0.0000, 0.0019, 0.0201, 0.0279, 0.0290, 0.0279, 0.0201, 0.0019, 0.0000,
                 0.0019, 0.0275, 0.0148, 0.002, 0.0048, 0.0002, -0.0148, 0.0275, 0.0019,
                 0.0201, 0.0148, - 0.0096, - 0.0242, - 0.0291, - 0.0242, - 0.0096, 0.0148, 0.0201,
                 0.0279, 0.0002, - 0.0242, - 0.0388, - 0.0437, - 0.0388, - 0.0242, 0.0002, 0.0279,
                 0.0290, - 0.0047, - 0.0291, - 0.0437, - 0.0486, - 0.0437, - 0.029, - 0.0047, 0.0290,
                 0.0279, 0.0002, - 0.0242, - 0.0388, - 0.0437, - 0.0388, - 0.0242, - 0.0002, 0.0279,
                 0.0201, 0.0148, - 0.0096, - 0.0242, - 0.0291, - 0.0242, - 0.0096, 0.0148, 0.0201,
                 0.0019, 0.0275, 0.0148, 0.0002, - 0.0048, 0.0002, 0.0148, 0.0275, 0.0019,
                 0.0000, 0.0019, 0.0201, 0.0279, 0.0290, 0.0279, 0.0201, 0.0019, 0.0000]).reshape(9, 9)


def zernike_9x9(path):
    img = cv2.imread(path)
    print(img.shape[1],img.shape[0])
    img = cv2.copyMakeBorder(img, 9, 9, 9, 9, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_gray1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k, iterations=5)
    t, img_t = cv2.threshold(img_gray1, 0, 255, cv2.THRESH_OTSU)
    canny_img = cv2.Canny(img_t, 0, 255)
    ZerImgM911R = cv2.filter2D(canny_img, cv2.CV_64F, M911R)
    ZerImgM911I = cv2.filter2D(canny_img, cv2.CV_64F, M911I)
    ZerImgM920 = cv2.filter2D(canny_img, cv2.CV_64F, M920)

    point_temporary_x = []
    point_temporary_y = []
    scatter_arr = cv2.findNonZero(ZerImgM920).reshape(-1, 2)
    for idx in scatter_arr:
        j, i = idx
        M9_theta_temporary = np.arctan2(ZerImgM911I[i][j], ZerImgM911R[i][j])
        M9_rotated_z11 = np.sin(M9_theta_temporary) * ZerImgM911I[i][j] + np.cos(M9_theta_temporary) * ZerImgM911R[i][j]
        l_method3 = ZerImgM920[i][j] / M9_rotated_z11
        k1 = 3 * M9_rotated_z11 / (2 * (1 - l_method3 ** 2)) ** 1.5

        # h = (ZerImgM00[i][j] - k * np.pi / 2 + k * np.arcsin(l_method2) + k * l_method2 * (1 - l_method2 ** 2) ** 0.5)
        # / np.pi
        k_value = t
        l_value = 2 ** 0.5 / g_N

        if k1 >= k_value and l_method3 <= l_value:
            y = i + g_N * l_method3 * np.sin(M9_theta_temporary) / 2 - 9
            x = j + g_N * l_method3 * np.cos(M9_theta_temporary) / 2 - 9
            point_temporary_x.append(int(x))
            point_temporary_y.append(int(y))
        else:
            continue

    # point_temporary_x = np.array(point_temporary_x)
    # point_temporary_y = np.array(point_temporary_y)

    return point_temporary_x, point_temporary_y

def bubble_sort(sequence):
    # 遍历的趟数，n个元素，遍历n-1趟
    for i in range(1, len(sequence)):
        # 从头遍历列表
        for j in range(0, len(sequence) - 1):
            # 遍历过程中，若前者数值大于后者，则交换
            if sequence[j] > sequence[j + 1]:
                # 注意，python中列表交换，不需要中间变量，可直接交换
                sequence[j], sequence[j + 1] = sequence[j + 1], sequence[j]

    # 返回处理完成后的列表
    return sequence
path = 'li1.jpg'

time1 = time.time()
point_temporary_x, point_temporary_y = zernike_9x9(path)
# bubble_sort(point_temporary_x)
# print(len(point_temporary_x))
# print(len(point_temporary_y))
img = cv2.imread(path)
un8 = np.zeros((img.shape[0],img.shape[1], 3), np.uint8)
for i in range(len(point_temporary_x)):
    # cv2.circle(un8,(point_temporary_x[i],point_temporary_y[i]),1,(255,255,255),1)
    if (point_temporary_x[i] > 0) and (point_temporary_x[i]<img.shape[1]):
        temp1 = point_temporary_x[i]
        temp2 = point_temporary_y[i]
        un8[temp2,temp1] = 255
cv2.imshow('un8',un8)
plt.imshow(un8,plt.cm.gray)
plt.show()
cv2.waitKey(0)

time2 = time.time()
print(time2 - time1)
