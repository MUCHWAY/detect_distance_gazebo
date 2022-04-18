#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import math

# RESOLUTION=[3840,2160]
# FOV=[60,60]
# MTX=np.mat([[1/(np.tan(FOV[0]/2/180*np.pi)/(RESOLUTION[0]/2)),0,(RESOLUTION[0]/2)],[0,1/(np.tan(FOV[1]/2/180*np.pi)/(RESOLUTION[1]/2)),(RESOLUTION[1]/2)],[0,0,1]])
# INV_MTX=np.linalg.inv(MTX)
    

class Camera_pos:
    def __init__(self):
        # 内参矩阵:
        # 1024*1024
        # self.mtx = np.array([[886.6292067272493, 0.000000000000000, 511.6759266231074],
        #                      [0.000000000000000, 886.6393568235922, 511.7946062444309],
        #                      [0.000000000000000, 0.000000000000000, 1.000000000000000]])
        # self.dist = np.array([[0.00, 0.00, 0.00, 0.00, 0]])

        # 3840*2160
        self.mtx = np.array([[3326.701550, 0.000000000, 1917.775908],
                             [0.000000000, 3326.882329, 1080.536339],
                             [0.000000000, 0.000000000, 1.000000000]])
        self.dist = np.array([[0.00, 0.00, 0.00, 0.00, 0]])
        self.inv_mtx=np.linalg.inv(self.mtx)

        self.newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (1024, 1024), 1, (1024, 1024))  # 自由比例参数
        self.img_x_center=self.newcameramtx[0][2]
        self.img_y_center=self.newcameramtx[1][2]
        
        self.camera_pos = [0.00 , 0.00 ,0.00]
        self.line_distance=0

    def point2point(self,detect):
        a = []
        a.append(detect)
        b = []
        b.append(a)
        target = np.array(b)

        xy_undistorted = cv2.undistortPoints(target, self.newcameramtx, self.dist, None, self.newcameramtx) 
        return xy_undistorted[0][0]
    
    def pos(self,h,detect):
        a = []
        a.append(detect)
        b = []
        b.append(a)
        target = np.array(b)
        xy_undistorted = cv2.undistortPoints(target, self.newcameramtx, self.dist, None, self.newcameramtx) 

        self.camera_pos[0] = (h*( xy_undistorted[0][0][0] -self.img_x_center))/self.newcameramtx[0][0]
        self.camera_pos[1] = -1* (h*( xy_undistorted[0][0][1] -self.img_y_center))/self.newcameramtx[1][1]
        self.camera_pos[2] = h
        self.line_distance = math.sqrt(math.pow(math.sqrt( math.pow(h,2) + math.pow(self.camera_pos[1],2) ),2)+math.pow(self.camera_pos[0],2))

        return xy_undistorted[0][0]
    
    def getCameraPose(self,uav_pose,camera_pitch):

        x,y,height,yaw,pitch,roll=uav_pose

        yaw=yaw*0.017453292519943
        pitch=pitch*0.017453292519943
        roll=roll*0.017453292519943
        camera_pitch=camera_pitch*0.017453292519943

        rotation_mat=np.mat([[np.cos(-yaw),-np.sin(-yaw),0],[np.sin(-yaw),np.cos(-yaw),0],[0,0,1]])\
                    *np.mat([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])\
                    *np.mat([[np.cos(roll),0,np.sin(roll)],[0,1,0],[-np.sin(roll),0,np.cos(roll)]])\
                    *np.mat([[1,0,0],[0,np.cos(camera_pitch),-np.sin(camera_pitch)],[0,np.sin(camera_pitch),np.cos(camera_pitch)]])

        beta = np.arctan2(rotation_mat[2,1], np.sqrt((rotation_mat[0,1])**2 + (rotation_mat[1,1])**2))

        err = float(0.001)
        if beta >= np.pi/2-err and beta <= np.pi/2+err:
            beta = np.pi/2
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = np.arctan2(rotation_mat[0,2], rotation_mat[0,0])
        elif beta >= -(np.pi/2)-err and beta <= -(np.pi/2)+err:
            beta = -np.pi/2
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = np.arctan2(rotation_mat[0,2], rotation_mat[0,0])
        else:
            alpha = np.arctan2(-(rotation_mat[0,1])/(np.cos(beta)), (rotation_mat[1,1])/(np.cos(beta)))
            gamma = np.arctan2(-(rotation_mat[2,0])/(np.cos(beta)), (rotation_mat[2,2])/(np.cos(beta)))

        yaw=-alpha*57.295779513082
        pitch=beta*57.295779513082
        roll=gamma*57.295779513082

        return [x,y,height,yaw,pitch,roll]
    

    def getFixedPix(self,x,y,w,h,pitch):
        gamma=1.5   #h/w
        if np.cos(pitch*0.017453292519943)>0:     #>-90
            return [x+w/2,y+h-h/2*(1-gamma/(gamma+np.tan(-pitch*0.017453292519943)))]
        elif np.cos(pitch*0.017453292519943)<0:   #<-90
            return [x+w/2,y+h/2*(1+gamma/(np.tan(-pitch*0.017453292519943)-gamma))]
        else:
            return [x+w/2,y+h/2]

    def pix2pos_2(self,camera_pose:list,inmtx,pix:list,d=None,inv_inmtx=None):

        x,y,height,yaw,pitch,roll=camera_pose

        yaw=yaw*0.017453292519943
        pitch=pitch*0.017453292519943
        roll=roll*0.017453292519943

        rotation_mat=np.mat([[np.cos(-yaw),-np.sin(-yaw),0],[np.sin(-yaw),np.cos(-yaw),0],[0,0,1]])\
                    *np.mat([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])\
                    *np.mat([[np.cos(roll),0,np.sin(roll)],[0,1,0],[-np.sin(roll),0,np.cos(roll)]])\
                    *np.mat([[1,0,0],[0,0,1],[0,-1,0]])

        T=-np.linalg.inv(rotation_mat)*np.mat([x,y,height]).T
        if not isinstance(inv_inmtx,np.matrix):
            inv_inmtx=np.linalg.inv(inmtx)

        if not d:
            d=height/np.sin(-pitch)

        center=rotation_mat*(inv_inmtx*d*np.mat([inmtx[0,2],inmtx[1,2],1]).T-T)
        z=center[2,0]

        posNorm=rotation_mat*(inv_inmtx*np.mat([pix[0],pix[1],1]).T-T)

        pos=[x+(posNorm[0,0]-x)/(height-posNorm[2,0])*(height-z),y+(posNorm[1,0]-y)/(height-posNorm[2,0])*(height-z),0]
        return pos