import numpy
import math
import cv2 as cv
from numpy import *
from scipy import optimize, odr
import functools
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题


def average_Grayval(img):  # 平均灰度值
    sum = 0
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            sum += img[i, j]
    average = sum / img.shape[0] / img.shape[1]
    return average


def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return (px, py)


image = cv.imread('li1.jpg', 1)
img1 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 转灰度图
_, otsu = cv.threshold(img1, None, 255, cv.THRESH_OTSU)  # OTSU阈值分割
otsu = cv.erode(otsu, kernel=numpy.ones((3, 3), numpy.uint8))  # 腐蚀去毛边
canny = cv.Canny(otsu, 100, 500, None, 3)  # 边缘检测
_, th2 = cv.threshold(canny, 100, 255, cv.THRESH_BINARY_INV)  # 颜色反转
set = []
for j in range(canny.shape[1]):
    for i in range(canny.shape[0]):
        if canny[i][j] == 0:
            continue
        else:
            set.append([j, i])
print(set)

un8 = numpy.zeros((image.shape[0], image.shape[1], 3), numpy.uint8) + 255
for i in set:
    un8[i[1]][i[0]] = [255, 0, 0]

bo_feng = [[58, 247], [132, 244], [217, 246], [290, 248], [357, 245], [426, 244], [505, 246], [580, 244], [664, 246]]
bo_gu = [[10, 262], [75, 304], [168, 263], [242, 267], [315, 264], [390, 264], [464, 266], [537, 273], [620, 303], [696, 265]]


flag1 = 0
flag2 = 0
gu_to_feng_point = []
feng_to_gu_point = []
for i in range(len(bo_feng) + len(bo_gu) - 1):
    # print('i=', i)
    if i % 2 == 0:
        # print(bo_gu[flag2][0], bo_gu[flag2][1], bo_feng[flag2][0], bo_feng[flag2][1])
        cv.line(canny, (bo_gu[flag2][0], bo_gu[flag2][1]), (bo_feng[flag2][0], bo_feng[flag2][1]), (255, 255, 255), 1)
        # 斜率指从波谷连波峰的斜率
        xielv_k_gu = (bo_feng[flag2][1] - bo_gu[flag2][1]) / (bo_feng[flag2][0] - bo_gu[flag2][0])
        mid_point_x_gu = (bo_gu[flag2][0] + bo_feng[flag2][0]) / 2
        mid_point_y_gu = (bo_gu[flag2][1] + bo_feng[flag2][1]) / 2
        zhongxian_k_gu = -1 / xielv_k_gu
        zhongxian_k_gu = round(zhongxian_k_gu, 3)
        chuixian_x1_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (190 - mid_point_y_gu)
        chuixian_x2_gu = mid_point_x_gu + (1 / zhongxian_k_gu) * (310 - mid_point_y_gu)
        cv.line(canny, (int(chuixian_x1_gu), 190), (int(chuixian_x2_gu), 310), (255, 255, 255), 1)
        flag2 += 1
        temp1 = []
        temp1.append(zhongxian_k_gu)
        temp1.append([mid_point_x_gu, mid_point_y_gu])
        gu_to_feng_point.append(temp1)


    else:
        # print(bo_feng[flag1][0], bo_feng[flag1][1], bo_gu[flag1 + 1][0], bo_gu[flag1 + 1][1])
        cv.line(canny, (bo_feng[flag1][0], bo_feng[flag1][1]), (bo_gu[flag1 + 1][0], bo_gu[flag1 + 1][1]), (255, 255, 255), 1)
        # 斜率指从波峰到波谷连线的斜率
        xielv_k_feng = (bo_feng[flag1][1] - bo_gu[flag1 + 1][1]) / (bo_feng[flag1][0] - bo_gu[flag1 + 1][0])
        mid_point_x_feng = (bo_feng[flag1][0] + bo_gu[flag1 + 1][0]) / 2
        mid_point_y_feng = (bo_feng[flag1][1] + bo_gu[flag1 + 1][1]) / 2
        zhongxian_k_feng = -1 / xielv_k_feng
        zhongxian_k_feng = round(zhongxian_k_feng, 3)
        chuixian_x1_feng = mid_point_x_feng + (1 / zhongxian_k_feng) * (190 - mid_point_y_feng)
        chuixian_x2_feng = mid_point_x_feng + (1 / zhongxian_k_feng) * (310 - mid_point_y_feng)
        cv.line(canny, (round(chuixian_x1_feng), 190), (round(chuixian_x2_feng), 310), (255, 255, 255), 1)
        flag1 += 1
        temp2 = []
        temp2.append(zhongxian_k_feng)
        temp2.append([mid_point_x_feng, mid_point_y_feng])
        feng_to_gu_point.append(temp2)

# print(gu_to_feng_point)
# print(feng_to_gu_point)

# [[3.2, [34.0, 254.5]],        [0.95, [103.5, 274.0]],     [2.882, [192.5, 254.5]],    [2.526, [266.0, 257.5]],
# [2.211, [336.0, 254.5]],      [1.8, [408.0, 254.0]],      [2.05, [484.5, 256.0]],     [1.483, [558.5, 258.5]],
# [0.772, [642.0, 274.5]]]

# [[-0.298, [66.5, 275.5]],     [-1.895, [150.0, 253.5]],   [-1.19, [229.5, 256.5]],    [-1.562, [302.5, 256.0]],
# [-1.737, [373.5, 254.5]],     [-1.727, [445.0, 255.0]],   [-1.185, [521.0, 259.5]],   [-0.678, [600.0, 273.5]],
# [-1.684, [680.0, 255.5]]]

FLAG1 = 0
FLAG2 = 0
last_bo_gu = []
last_bo_feng = []
for i in range(len(gu_to_feng_point)):
    x1 = gu_to_feng_point[FLAG1][1][0]
    y1 = gu_to_feng_point[FLAG1][1][1]
    k1 = gu_to_feng_point[FLAG1][0]
    x2 = feng_to_gu_point[FLAG1][1][0]
    y2 = feng_to_gu_point[FLAG1][1][1]
    k2 = feng_to_gu_point[FLAG1][0]
    x = (k1 * x1 - y1 - k2 * x2 + y2) / (k1 - k2)
    y = k1 * (x - x1) + y1
    last_bo_gu.append([round(x,3),round(y,3)])
    un8[int(y), int(x)] = [0, 0, 255]
    un8[int(y - 1), int(x)] = [0, 0, 255]
    un8[int(y + 1), int(x)] = [0, 0, 255]
    un8[int(y), int(x - 1)] = [0, 0, 255]
    un8[int(y), int(x + 1)] = [0, 0, 255]
    un8[int(y - 1), int(x - 1)] = [0, 0, 255]
    un8[int(y - 1), int(x + 1)] = [0, 0, 255]
    un8[int(y + 1), int(x - 1)] = [0, 0, 255]
    un8[int(y + 1), int(x + 1)] = [0, 0, 255]
    FLAG1 += 1
last_bo_gu = numpy.array(last_bo_gu)
print(last_bo_gu)
for i in range(len(feng_to_gu_point) - 1):
    x1 = feng_to_gu_point[FLAG2][1][0]
    y1 = feng_to_gu_point[FLAG2][1][1]
    k1 = feng_to_gu_point[FLAG2][0]
    x2 = gu_to_feng_point[FLAG2 + 1][1][0]
    y2 = gu_to_feng_point[FLAG2 + 1][1][1]
    k2 = gu_to_feng_point[FLAG2 + 1][0]
    x = (k1 * x1 - y1 - k2 * x2 + y2) / (k1 - k2)
    y = k1 * (x - x1) + y1
    last_bo_feng.append([round(x,3),round(y,3)])
    un8[int(y), int(x)] = [0, 255, 0]
    un8[int(y - 1), int(x)] = [0, 255, 0]
    un8[int(y + 1), int(x)] = [0, 255, 0]
    un8[int(y), int(x - 1)] = [0, 255, 0]
    un8[int(y), int(x + 1)] = [0, 255, 0]
    un8[int(y - 1), int(x - 1)] = [0, 255, 0]
    un8[int(y - 1), int(x + 1)] = [0, 255, 0]
    un8[int(y + 1), int(x - 1)] = [0, 255, 0]
    un8[int(y + 1), int(x + 1)] = [0, 255, 0]
    FLAG2 += 1
last_bo_feng = numpy.array(last_bo_feng)
print(last_bo_feng)
# cv.imshow('un8', un8)

# x = r_[14,   15,  15,  16,  16,  17,  17,  18,  19,  20,  20,  21,  22,  23,  24,  25,  26,  26,  27,  28,  29,  30,  31,  32]
# y = r_[-261, -260, -261, -259, -260, -258, -259, -258, -257, -256, -257, -255, -254, -254, -253, -253, -252, -253, -252, -251, -251, -250, -250, -250]
# method_1 = '代数逼近法    '
# # 质心坐标
# x_m = mean(x)
# y_m = mean(y)
# print(x_m, y_m)
# u = x - x_m
# v = y - y_m
# Suv = sum(u * v)
# Suu = sum(u ** 2)
# Svv = sum(v ** 2)
# Suuv = sum(u ** 2 * v)
# Suvv = sum(u * v ** 2)
# Suuu = sum(u ** 3)
# Svvv = sum(v ** 3)
# A = array([[Suu, Suv], [Suv, Svv]])
# B = array([Suuu + Suvv, Svvv + Suuv]) / 2.0
# uc, vc = linalg.solve(A, B)
# xc_1 = x_m + uc
# yc_1 = y_m + vc
# Ri_1 = sqrt((x - xc_1) ** 2 + (y - yc_1) ** 2)
# R_1 = mean(Ri_1)
# residu_1 = sum((Ri_1 - R_1) ** 2)
# ncalls_1 = 0
# residu2_2 = 0

plt.figure()
plt.grid(False)
plt.imshow(canny, cmap=plt.cm.gray)
plt.figure()
plt.grid(False)
plt.imshow(un8, cmap=plt.cm.gray)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
