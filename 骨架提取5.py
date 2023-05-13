'''

函数功能： 识别导线骨架后去除最小和最大的骨架曲线
           自写应骨架提取算法
'''
import numpy as np
import cv2 as cv
import math
im = cv.imread('./testphoto/9du.jpg', 0)


ret, im = cv.threshold(im, 127, 255, cv.THRESH_BINARY)


skel = np.zeros(im.shape, np.uint8)
erode = np.zeros(im.shape, np.uint8)
temp = np.zeros(im.shape, np.uint8)
i = 0
while(1):
    # cv.imshow('im %d' % (i), im)
    erode = cv.erode(im, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    temp = cv.dilate(erode, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    # 消失的像素是skeleton的一部分
    temp = cv.subtract(im, temp)
    # cv.imshow('open %d' % (i,), temp)
    skel = cv.bitwise_or(skel, temp)
    # cv.imshow('skel %d' % (i,), skel)
    im = erode.copy()

    if cv.countNonZero(im) == 0:
        break;
    i += 1

cv.imshow('Skeleton', skel)
# edges = cv.Canny(skel, 150, 550, apertureSize=3)
# cv.imshow('edges', edges)

# un8 = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
# un9 = un8.copy()
# lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=200, maxLineGap=150)  # 概率霍夫变换
# inf_angle = []  # 角度
# inf_x1 = []
# inf_y1 = []
# inf_x2 = []
# inf_y2 = []
#
# for line in lines:
#     for x1, y1, x2, y2 in line:
#         if abs(y2 - y1) > im.shape[1] / 2:
#             inf_x1.append(x1)
#             inf_y1.append(y1)
#             inf_x2.append(x2)
#             inf_y2.append(y2)
#
#             cv.line(un8, (x1, y1), (x2, y2), (250, 250, 0), 1)
#             cv.circle(un8, (x1, y1), 3, (255, 150, 0), 1)
#             angle = math.degrees(math.atan(((x2 - x1) / (y2 - y1))))
#             up_int = round(angle, 3)
#             slip_angle = float(str(angle)[0:5])
#             # print(up_int, '\t', slip_angle)
#             inf_angle.append(slip_angle)
#
# # cv.line(un9,(inf_x1[24],inf_y1[24]),(inf_x2[24],inf_y2[24]),(255,0,0),3)
# index_min = inf_x1.index(min(inf_x1))
# index_max = inf_x1.index(max(inf_x1))
#
# inf_x1.pop(index_min)
# inf_y1.pop(index_min)
# inf_x2.pop(index_min)
# inf_y2.pop(index_min)
#
# inf_x1.pop(index_max)
# inf_y1.pop(index_max)
# inf_x2.pop(index_max)
# inf_y2.pop(index_max)
#
# inf_x1.pop(index_min)
# inf_y1.pop(index_min)
# inf_x2.pop(index_min)
# inf_y2.pop(index_min)
#
# inf_x1.pop(index_max)
# inf_y1.pop(index_max)
# inf_x2.pop(index_max)
# inf_y2.pop(index_max)
# # print(inf_x1)
# # print(inf_y1)
# # print(inf_x2)
# # print(inf_y2)
# # print(len(inf_x1))
# # print(len(inf_y1))
# # print(len(inf_x2))
# # print(len(inf_y2))
# for i in range(len(inf_x1)):
#        cv.line(un9, (inf_x1[i], inf_y1[i]), (inf_x2[i], inf_y2[i]), (250, 250, 0), 1)
#
# for i in range(len(inf_x1)):
#     angle2 = math.degrees(math.atan(((inf_x2[i] - inf_x1[i]) / (inf_y2[i] - inf_y1[i]))))
#     print(float(str(angle2)[0:5]))
# cv.imshow('un8', un8)
#
# cv.imshow('un9', un9)
cv.waitKey()

