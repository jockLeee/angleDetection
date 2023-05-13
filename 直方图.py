import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
def Origin_histogram(img):
    # 建立原始图像各灰度级的灰度值与像素个数对应表
    histogram = {}
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            k = img[i][j]
            if k in histogram:
                histogram[k] += 1
            else:
                histogram[k] = 1

    sorted_histogram = {}  # 建立排好序的映射表
    sorted_list = sorted(histogram)  # 根据灰度值进行从低至高的排序

    for j in range(len(sorted_list)):
        sorted_histogram[sorted_list[j]] = histogram[sorted_list[j]]

    return sorted_histogram


def equalization_histogram(histogram, img):
    pr = {}  # 建立概率分布映射表

    for i in histogram.keys():      # 求各灰度所占百分比
        pr[i] = histogram[i] / (img.shape[0] * img.shape[1])

    tmp = 0
    for m in pr.keys():   # 求累积灰度百分比
        tmp += pr[m]

        pr[m] = max(histogram) * tmp  # 归一化

    new_img = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)

    for k in range(img.shape[0]):
        for l in range(img.shape[1]):
            new_img[k][l] = pr[img[k][l]]

    return new_img


def GrayHist(img):
    # 计算灰度直方图
    height, width = img.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(height):
        for j in range(width):
            grayHist[img[i][j]] += 1
    return grayHist


image1 = cv.imread('20.jpg', 1)
img1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
origin_grayHist1 = GrayHist(img1)

image2 = cv.imread('40.jpg', 1)
img2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
origin_grayHist2 = GrayHist(img2)

image3 = cv.imread('80.jpg', 1)
img3 = cv.cvtColor(image3, cv.COLOR_BGR2GRAY)
origin_grayHist3 = GrayHist(img3)

x = np.arange(256)
plt.subplot(2, 3, 1)
plt.imshow(img1, cmap=plt.cm.gray)
plt.title('Origin')

plt.figure(num=1)

plt.subplot(2,3, 4)
plt.plot(x, origin_grayHist1, 'r', linewidth=1, c='black')
plt.title("Origin")
plt.ylabel("number of pixels")

plt.subplot(2, 3, 2)
plt.imshow(img2, cmap=plt.cm.gray)
plt.title('Origin')

plt.subplot(2,3, 5)
plt.plot(x, origin_grayHist2, 'r', linewidth=1, c='black')
plt.title("Origin")
plt.ylabel("number of pixels")

plt.subplot(2, 3, 3)
plt.imshow(img3, cmap=plt.cm.gray)
plt.title('Origin')

plt.subplot(2,3, 6)
plt.plot(x, origin_grayHist3, 'r', linewidth=1, c='black')
plt.title("Origin")
plt.ylabel("number of pixels")

plt.show()






