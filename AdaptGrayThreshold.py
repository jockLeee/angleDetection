import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from skimage import morphology
import cv2 as cv
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

def get_skeleton(binary):
    binary[binary == 255] = 1
    skeleton0 = morphology.skeletonize(binary)  # 骨架提取
    # skel, distance = morphology.medial_axis(binary, return_distance=True)
    # skeleton0 = distance * skel
    skeleton = skeleton0.astype(np.uint8) * 255
    return skeleton

# 改进灰度均衡化
def get_strengthen_gray(image, precision, low_value):
    save_dir = "./save"
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

def average_Grayval(img):
    sum = 0
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            sum += img[i, j]
    average = sum / img.shape[0] / img.shape[1]
    return average


image = cv.imread('./testphoto/1.jpg', 1)
img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# cv.imshow('img',img)
eqimg = cv.equalizeHist(img)  # 自适应灰度均衡
temp2 = get_strengthen_gray(img, 79, 200)

# cv.imshow('temp2',temp2)

thres = round(average_Grayval(img) / 2)
print(thres)
_, th1 = cv.threshold(img, thres, 255, cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 69, 30)
_, th3 = cv.threshold(eqimg, thres, 255, cv.THRESH_BINARY)
_, th4 = cv.threshold(temp2, thres, 255, cv.THRESH_BINARY)
un8 = np.zeros((th2.shape[0], th2.shape[1], 3), np.uint8) + 255
for i in range(th2.shape[0]):
    for j in range(th2.shape[1]):
        if th2[i, j] == th1[i, j] == 0:
            un8[i, j] = 0
        else:
            un8[i, j] = 255
cv.imshow('un8', un8)

erosion = cv.dilate(un8, np.ones((1, 3), np.uint8))
# erosion = cv.erode(erosion, np.ones((1, 5), np.uint8))
skel = get_skeleton(erosion)
cv.imshow('skel', skel)

# sqar1 = 0
# sqar2 = 0
# sqar3 = 0
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         sqar1 = (img[i, j] - average_Grayval(img)) ** 2
#         sqar2 = (eqimg[i, j] - average_Grayval(img)) ** 2
#         sqar3 = (temp2[i, j] - average_Grayval(img)) ** 2
# sqar1 = sqar1 / img.shape[0] / img.shape[1]
# sqar2 = sqar2 / eqimg.shape[0] / eqimg.shape[1]
# sqar3 = sqar3 / temp2.shape[0] / temp2.shape[1]
# print(sqar1,sqar2,sqar3)


plt.subplot(2, 4, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB), cmap=plt.cm.gray)
plt.title("原图像")

plt.subplot(2, 4, 2)
plt.imshow(img, cmap=plt.cm.gray)
plt.title("灰度图像")

plt.subplot(2, 4, 3)
plt.imshow(eqimg, cmap=plt.cm.gray)
plt.title("自适应灰度化")

plt.subplot(2, 4, 4)
plt.imshow(temp2, cmap=plt.cm.gray)
plt.title("切片法")

plt.subplot(2, 4, 5)
plt.imshow(th1, cmap=plt.cm.gray)
plt.title("灰度图像+动态阈值")

plt.subplot(2, 4, 6)
plt.imshow(th2, cmap=plt.cm.gray)
plt.title("自适应二值化")

plt.subplot(2, 4, 7)
plt.imshow(th3, cmap=plt.cm.gray)
plt.title("直方图均衡化+动态阈值")

plt.subplot(2, 4, 8)
plt.imshow(th4, cmap=plt.cm.gray)
plt.title("切片法+动态阈值")
plt.show()
cv.waitKey(0)
