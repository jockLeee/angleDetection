import cv2 as cv
from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import *

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题


# 骨架提取
def get_skeleton(binary):
    binary[binary == 255] = 1
    skeleton0 = morphology.skeletonize(binary)  # 骨架提取
    # skel, distance = morphology.medial_axis(binary, return_distance=True)
    # skeleton0 = distance * skel
    skeleton = skeleton0.astype(np.uint8) * 255
    return skeleton


def medial_ax(binary):
    binary[binary == 255] = 1
    skel, distance = morphology.medial_axis(binary, return_distance=True)
    dist_on_skel = distance * skel
    dist_on_skel = dist_on_skel.astype(np.uint8) * 255
    return dist_on_skel


# 冒泡排序
def bubble_sort(sequence, sequence2, sequence3):
    # 遍历的趟数，n个元素，遍历n-1趟
    for i in range(1, len(sequence)):
        # 从头遍历列表
        for j in range(0, len(sequence) - 1):
            # 遍历过程中，若前者数值大于后者，则交换
            if sequence[j] > sequence[j + 1]:
                # 注意，python中列表交换，不需要中间变量，可直接交换
                sequence[j], sequence[j + 1] = sequence[j + 1], sequence[j]
                sequence2[j], sequence2[j + 1] = sequence2[j + 1], sequence2[j]
                sequence3[j], sequence3[j + 1] = sequence3[j + 1], sequence3[j]
    # 返回处理完成后的列表
    return sequence


# 排序去除重复元素
def delete_num(temp, temp1, temp2):
    k = 1
    for i in range(0, len(temp) - 1):
        for j in range(k, len(temp)):
            if temp[j] - temp[i] < 5:
                temp.pop(i)
                temp1.pop(i)
                temp2.pop(i)
                k += 1
                break
            else:
                k += 1
                break
    return temp


# 平均灰度值
def average_Grayval(img):
    sum = 0
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            sum += img[i, j]
    average = sum / img.shape[0] / img.shape[1] / 2
    return average


# 改进灰度均衡化
def get_strengthen_gray(image, precision, low_value):
    save_dir = ".\\save"
    path = "./save/"
    x = int(image.shape[0] / precision)  # 分块长度
    for i in range(0, precision + 1):
        un8 = np.zeros((x, image.shape[1], 3), np.uint8)
        un9 = image[i * x:(i + 1) * x, 0:image.shape[1]]
        sum = 0
        for k in range(0, un9.shape[0]):
            for j in range(0, un9.shape[1]):
                sum += un9[k][j]
        average_gray = sum / (un9.shape[0] * un9.shape[1])
        if average_gray < low_value:
            un7 = cv.equalizeHist(un9)
        else:
            un7 = un9
        cv.imwrite(save_dir + '/' + '%d.jpg' % (i), un7)
    save_path = path + str(0) + ".jpg"
    img_out = cv.imread(save_path)
    num = precision + 1
    for i in range(1, num):
        save_path = path + str(i) + ".jpg"
        img_tmp = cv.imread(save_path)
        img_out = np.concatenate((img_out, img_tmp), axis=0)
    img_out = cv.cvtColor(img_out, cv.COLOR_BGR2GRAY)
    cv.imwrite("%d.jpg" % (num), img_out)
    return img_out


# 读取图像
photo = './test/03252.jpg'
# photo = '../Windmachine221110/white_black_black_white3-3.jpg'
image = cv.imread(photo, 1)
img = cv.imread(photo, 0)
cv.imshow('img0', img)
temp1 = get_strengthen_gray(img, precision=79, low_value=200)
cv.imshow('temp1', temp1)

average = average_Grayval(img)
_, temp = cv.threshold(img, average, 255, cv.THRESH_BINARY)  # ← ← ← ← ← ← ← ← ← ← ← ← ← 二值化
cv.imshow('temp', temp)
# cv.imwrite('./test/'+'bin_1103.png', temp)
erosion = cv.dilate(temp, kernel=np.ones((1, 3), np.uint8))  # 腐蚀 膨胀
erosion = cv.erode(erosion, kernel=np.ones((3, 3), np.uint8))
# cv.imshow('erosion', erosion)

# 骨架提取
skel = get_skeleton(erosion)
cv.imshow('skel', skel)

edges = cv.Canny(skel, 150, 550, apertureSize=3)
cv.imshow('edges', edges)

un8 = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

lines = cv.HoughLinesP(skel, rho=1, theta=np.pi / 180, threshold=100, minLineLength=200, maxLineGap=200)  # 概率霍夫变换

inf_angle = []  # 角度
inf_x1 = []
inf_y1 = []
inf_x2 = []
inf_y2 = []
# print(lines)
for line in lines:
    for x1, y1, x2, y2 in line:
        if abs(y2 - y1) > img.shape[0] / 2:
            inf_x1.append(x1)
            inf_y1.append(y1)
            inf_x2.append(x2)
            inf_y2.append(y2)
            cv.line(un8, (x1, y1), (x2, y2), (250, 250, 0), 1)
            cv.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv.circle(un8, (x1, y1), 5, (255, 150, 0), 1)
            angle = math.degrees(math.atan(((x2 - x1) / (y2 - y1))))
            # up_int = round(angle, 3)
            slip_angle = float(str(angle)[0:5])
            # print(up_int, '\t', slip_angle)
            inf_angle.append(slip_angle)
print('inf_angle: ', inf_angle)

# index_min = inf_x1.index(min(inf_x1))
# index_max = inf_x1.index(max(inf_x1))
#
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

# inf_x1.pop(index_min)
# inf_y1.pop(index_min)
# inf_x2.pop(index_min)
# inf_y2.pop(index_min)
#
# inf_x1.pop(index_max)
# inf_y1.pop(index_max)
# inf_x2.pop(index_max)
# inf_y2.pop(index_max)

dele_angle = []  # 均方误差去除干扰值
dele_angle2 = []  # 均方误差去除后数据
for i in range(len(inf_x1)):
    angle2 = math.degrees(math.atan(((inf_x2[i] - inf_x1[i]) / (inf_y2[i] - inf_y1[i]))))
    dele_angle.append(float(str(angle2)[0:5]))
print('dele_angle: ', dele_angle)

demo = np.median(dele_angle)
print(demo)
for i in range(len(dele_angle)):
    if abs(dele_angle[i] - demo) > 0.4:
        continue
    else:
        dele_angle2.append(dele_angle[i])

print('dele_angle2: ', dele_angle2)
print(np.mean(dele_angle2), max(dele_angle2), min(dele_angle2))
cv.imshow('un8', un8)
cv.imshow('image', image)
cv.waitKey(0)
