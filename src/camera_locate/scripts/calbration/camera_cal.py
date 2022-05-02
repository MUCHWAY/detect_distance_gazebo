# -*- coding: UTF-8 -*-
from __future__ import print_function
import cv2
import numpy as np
import glob
from numpy import array as matrix, arange

# 找棋盘格角点
# 阈值
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# w h分别是棋盘格模板长边和短边规格（角点个数）
w = 11
h = 8

# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵，认为在棋盘格这个平面上Z=0
objp = np.zeros((w * h, 3), np.float32)  # 构造0矩阵，88行3列，用于存放角点的世界坐标
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)  # 三维网格坐标划分

# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像平面的二维点
record = []

images = glob.glob('./cal/*.jpg')
for fname in images:
    print(fname)
    img = cv2.imread(fname)
#     cv2.imshow("display",img)
#     cv2.waitKey(100)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 粗略找到棋盘格角点 这里找到的是这张图片中角点的亚像素点位置，共11×8 = 88个点，gray必须是8位灰度或者彩色图，（w,h）为角点规模
    ret, corners = cv2.findChessboardCorners(gray, (w, h))
    # 如果找到足够点对，将其存储起来
    if ret == True:
        record.append(fname)
        # 精确找到角点坐标
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 将正确的objp点放入objpoints中
        objpoints.append(objp)
        imgpoints.append(corners)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w, h), corners, ret)
        cv2.imshow('findCorners', img)
        cv2.waitKey(100)
cv2.destroyAllWindows()
# 标定 返回标定结果、相机的内参数矩阵、畸变系数、旋转矩阵和平移向量
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print('ret:', ret)
print('内参矩阵:', mtx)
print(type(mtx))
print('畸变系数:', dist)
print(type(dist))
print('旋转矩阵:', rvecs)
print('平移向量:', tvecs)

# 去畸变
img2 = cv2.imread('./cal/11.jpg')
h, w = img2.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # 自由比例参数
dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
# 根据前面ROI区域裁剪图片
# x, y, w, h = roi
# dst = dst[y:y + h, x:x + w]
# cv2.imshow('fin', dst)
cv2.imwrite('./fin.png', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 反投影误差
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error
print("total error: ", total_error / len(objpoints))



'''


'''
