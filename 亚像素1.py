import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import datetime

start = time.process_time()

kernels_Num = 8
kernels = ['_' for i in range(kernels_Num)]
kernels[0] = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=int)
kernels[1] = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=int)
kernels[2] = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=int)
kernels[3] = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=int)
kernels[4] = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=int)
kernels[5] = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=int)
kernels[6] = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=int)
kernels[7] = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=int)

img = cv2.imread('li1.jpg')
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转成RGB 方便后面显示
# 灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gradients = ['_' for i in range(kernels_Num)]
for i in range(kernels_Num):
    gradients[i] = cv2.filter2D(grayImage, cv2.CV_16S, kernels[i])
    """
    显示梯度图像  代码
    """
#     cv2.normalize(gradients[i],gradients[i],0,255,cv2.NORM_MINMAX)
#     abs = cv2.convertScaleAbs(gradients[i])
#     cv2.imshow(str(i),abs)
# cv2.imshow('222',img)
# cv2.waitKey(0)

angle_list = [270, 315, 0, 45, 90, 135, 180, 225]
amplitude = np.full(grayImage.shape,0)
angle = np.full(grayImage.shape,-64)

for r in range(grayImage.shape[0]):
    pAmp = amplitude[r]
    pAng = angle[r]

    pGrad = ['_' for i in range(kernels_Num)]
    for i in range(kernels_Num):
        pGrad[i] = gradients[i][r]
    for c in range(grayImage.shape[1]):
        for i in range(kernels_Num):
            if (pAmp[c] < pGrad[i][c]):
                pAmp[c] = pGrad[i][c]
                pAng[c] = angle_list[i]

"""
显示幅值图像  代码
"""
# cv2.normalize(amplitude,amplitude,0,255,cv2.NORM_MINMAX)
# abs = cv2.convertScaleAbs(amplitude)
# cv2.imshow('amplitude',abs)
# cv2.imshow('222',img)
# cv2.waitKey(0)

"""
显示角度图像  代码
"""
# cv2.normalize(angle,angle,0,255,cv2.NORM_MINMAX)
# abs = cv2.convertScaleAbs(angle)
# cv2.imshow('angle',abs)
# cv2.imshow('222',img)
# cv2.waitKey(0)

edge = np.full(grayImage.shape,0)
edge.astype('uint8')
thres = 100 #阈值  设置最小幅度值
for r in range(1, grayImage.shape[0]-1):
    pAmp1 = amplitude[r-1]
    pAmp2 = amplitude[r]
    pAmp3 = amplitude[r+1]

    pAng = angle[r]
    pEdge = edge[r]
    for c in range(1, grayImage.shape[1]-1):

        if (pAmp2[c] < thres):
            continue
        if pAng[c] == 270:
            if pAmp2[c] > pAmp1[c] and pAmp2[c] >= pAmp3[c]:
                pEdge[c] = 255
        elif pAng[c] == 90:
            if pAmp2[c] >= pAmp1[c] and pAmp2[c] > pAmp3[c]:
                pEdge[c] = 255
        elif pAng[c] == 315:
            if pAmp2[c] > pAmp1[c - 1] and pAmp2[c] >= pAmp3[c + 1]:
                pEdge[c] = 255
        elif pAng[c] == 135:
            if pAmp2[c] >= pAmp1[c - 1] and pAmp2[c] > pAmp3[c + 1]:
                pEdge[c] = 255
        elif pAng[c] == 0:
            if pAmp2[c] > pAmp2[c - 1] and pAmp2[c] >= pAmp2[c + 1]:
                pEdge[c] = 255
        elif pAng[c] == 180:
            if pAmp2[c] >= pAmp2[c - 1] and pAmp2[c] > pAmp2[c + 1]:
                pEdge[c] = 255
        elif pAng[c] == 45:
            if pAmp2[c] >= pAmp1[c + 1] and pAmp2[c] > pAmp3[c - 1]:
                pEdge[c] = 255
        elif pAng[c] == 225:
            if pAmp2[c] > pAmp1[c + 1] and pAmp2[c] >= pAmp3[c - 1]:
                pEdge[c] = 255
"""
显示单像素图像
"""
# edge = cv2.convertScaleAbs(edge)
# cv2.imshow('edge',edge)
# cv2.imwrite('edge.png', edge)
# cv2.waitKey(0)

"""
亚像素处理
"""
root2 = np.sqrt(2.0)
tri_list = [[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
for i in range(kernels_Num):
	tri_list[0][i] = np.cos(angle_list[i] * np.pi / 180.0)
	# sin前面的负号非常关键, 因为图像的y方向和直角坐标系的y方向相反
	tri_list[1][i] = -np.sin(angle_list[i] * np.pi / 180.0)
vPts = []
for r in range(1, grayImage.shape[0]-1):
    pAmp1 = amplitude[r - 1]
    pAmp2 = amplitude[r]
    pAmp3 = amplitude[r + 1]

    pAng = angle[r]
    pEdge = edge[r]
    for c in range(1, grayImage.shape[1]-1):
        if (pEdge[c]):
            nAngTmp = 0
            dTmp = 0
            if pAng[c] == 270:
                nAngTmp = 0
                dTmp = (pAmp1[c] - pAmp3[c]) / (pAmp1[c] + pAmp3[c] - 2 * pAmp2[c]) * 0.5
                # print([c + dTmp * tri_list[0][nAngTmp],r + dTmp * tri_list[1][nAngTmp]])
            elif pAng[c] == 90:
                nAngTmp = 4
                dTmp = -(pAmp1[c] - pAmp3[c]) / (pAmp1[c] + pAmp3[c] - 2 * pAmp2[c]) * 0.5
            elif pAng[c] == 315:
                nAngTmp = 1
                dTmp = (pAmp1[c - 1] - pAmp3[c + 1]) / (pAmp1[c - 1] + pAmp3[c + 1] - 2 * pAmp2[c]) * root2 * 0.5
            elif pAng[c] == 135:
                nAngTmp = 5
                dTmp = -(pAmp1[c - 1] - pAmp3[c + 1]) / (pAmp1[c - 1] + pAmp3[c + 1] - 2 * pAmp2[c]) * root2 * 0.5
            elif pAng[c] == 0:
                nAngTmp = 2
                dTmp = (pAmp2[c - 1] - pAmp2[c + 1]) / (pAmp2[c - 1] + pAmp2[c + 1] - 2 * pAmp2[c]) * 0.5
            elif pAng[c] == 180:
                nAngTmp = 6
                dTmp = -(pAmp2[c - 1] - pAmp2[c + 1]) / (pAmp2[c - 1] + pAmp2[c + 1] - 2 * pAmp2[c]) * 0.5
            elif pAng[c] == 45:
                nAngTmp = 3
                dTmp = (pAmp3[c - 1] - pAmp1[c + 1]) / (pAmp1[c + 1] + pAmp3[c - 1] - 2 * pAmp2[c]) * root2 * 0.5
            elif pAng[c] == 225:
                nAngTmp = 7
                dTmp = -(pAmp3[c - 1] - pAmp1[c + 1]) / (pAmp1[c + 1] + pAmp3[c - 1] - 2 * pAmp2[c]) * root2 * 0.5

            x = c + dTmp * tri_list[0][nAngTmp]
            y = r + dTmp * tri_list[1][nAngTmp]
            vPts.append([int(x),int(y)])
un8 = np.zeros((img.shape[0],img.shape[1], 3), np.uint8)
"""
输出亚像素坐标
"""
for x,y in vPts:
    print(x,y)
    # cv2.circle(un8,(x,y),1,(255,255,255),1)
    un8[y,x] = 255
cv2.imshow('un8',un8)
plt.imshow(un8,plt.cm.gray)
plt.show()
cv2.waitKey(0)

endtime = time.process_time()
print (str(endtime - start) + 's')
