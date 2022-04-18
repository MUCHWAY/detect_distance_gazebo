#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from numpy import save
import rospy
import math
import numpy as np
import threading
import cv2
import argparse
from time import sleep
import datetime
import socket
from time import sleep

from std_msgs.msg import *
from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *

from yolov5_detect.msg import detect
from camrea_locate.msg import DroneState

from cam_pos import *

# RESOLUTION=[1920,1080]
# FOV=[30,15]
# MTX=np.mat([[1/(np.tan(FOV[0]/2/180*np.pi)/(RESOLUTION[0]/2)),0,(RESOLUTION[0]/2)],[0,1/(np.tan(FOV[1]/2/180*np.pi)/(RESOLUTION[1]/2)),(RESOLUTION[1]/2)],[0,0,1]])
# INV_MTX=np.linalg.inv(MTX)
    

# class Camera_pos:
#     def __init__(self):
#         # 内参矩阵:
#         # 1024*1024
#         # self.mtx = np.array([[886.6292067272493, 0.000000000000000, 511.6759266231074],
#         #                      [0.000000000000000, 886.6393568235922, 511.7946062444309],
#         #                      [0.000000000000000, 0.000000000000000, 1.000000000000000]])

#         # self.dist = np.array([[0.00, 0.00, 0.00, 0.00, 0]])
#         # 3840*2160
#         self.mtx = np.array([[3326.701550, 0.000000000, 1917.775908],
#                              [0.000000000, 3326.882329, 1080.536339],
#                              [0.000000000, 0.000000000, 1.000000000]])

#         self.dist = np.array([[0.00, 0.00, 0.00, 0.00, 0]])

#         self.newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (1024, 1024), 1, (1024, 1024))  # 自由比例参数
#         # print(self.newcameramtx)
#         self.img_x_center=self.newcameramtx[0][2]
#         self.img_y_center=self.newcameramtx[1][2]
        
#         self.camera_pos = [0.00 , 0.00 ,0.00]
#         self.line_distance=0
    
#     def pos(self,h,detect):
#         a = []
#         a.append(detect)
#         b = []
#         b.append(a)
#         target = np.array(b)

#         xy_undistorted = cv2.undistortPoints(target, self.newcameramtx, self.dist, None, self.newcameramtx) 

#         self.camera_pos[0] = (h*( xy_undistorted[0][0][0] -self.img_x_center))/self.newcameramtx[0][0]
#         self.camera_pos[1] = -1* (h*( xy_undistorted[0][0][1] -self.img_y_center))/self.newcameramtx[1][1]
#         self.camera_pos[2] = h
#         self.line_distance = math.sqrt(math.pow(math.sqrt( math.pow(h,2) + math.pow(self.camera_pos[1],2) ),2)+math.pow(self.camera_pos[0],2))

#         return xy_undistorted[0][0]
    
#     def getCameraPose(self,uav_pose,camera_pitch):

#         x,y,height,yaw,pitch,roll=uav_pose

#         yaw=yaw*0.017453292519943
#         pitch=pitch*0.017453292519943
#         roll=roll*0.017453292519943
#         camera_pitch=camera_pitch*0.017453292519943

#         rotation_mat=np.mat([[np.cos(-yaw),-np.sin(-yaw),0],[np.sin(-yaw),np.cos(-yaw),0],[0,0,1]])\
#                     *np.mat([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])\
#                     *np.mat([[np.cos(roll),0,np.sin(roll)],[0,1,0],[-np.sin(roll),0,np.cos(roll)]])\
#                     *np.mat([[1,0,0],[0,np.cos(camera_pitch),-np.sin(camera_pitch)],[0,np.sin(camera_pitch),np.cos(camera_pitch)]])

#         beta = np.arctan2(rotation_mat[2,1], np.sqrt((rotation_mat[0,1])**2 + (rotation_mat[1,1])**2))

#         err = float(0.001)
#         if beta >= np.pi/2-err and beta <= np.pi/2+err:
#             beta = np.pi/2
#             # alpha + gamma is fixed
#             alpha = 0.0
#             gamma = np.arctan2(rotation_mat[0,2], rotation_mat[0,0])
#         elif beta >= -(np.pi/2)-err and beta <= -(np.pi/2)+err:
#             beta = -np.pi/2
#             # alpha - gamma is fixed
#             alpha = 0.0
#             gamma = np.arctan2(rotation_mat[0,2], rotation_mat[0,0])
#         else:
#             alpha = np.arctan2(-(rotation_mat[0,1])/(np.cos(beta)), (rotation_mat[1,1])/(np.cos(beta)))
#             gamma = np.arctan2(-(rotation_mat[2,0])/(np.cos(beta)), (rotation_mat[2,2])/(np.cos(beta)))

#         yaw=-alpha*57.295779513082
#         pitch=beta*57.295779513082
#         roll=gamma*57.295779513082

#         return [x,y,height,yaw,pitch,roll]
    

#     def getFixedPix(self,x,y,w,h,pitch):
#         gamma=1.5   #h/w
#         if np.cos(pitch)>0:     #>-90
#             return [x+w/2,y+h-h/2*(1-gamma/(gamma+np.tan(-pitch*0.017453292519943)))]
#         elif np.cos(pitch)<0:   #<-90
#             return [x+w/2,y+h/2*(1+gamma/(np.tan(-pitch*0.017453292519943)-gamma))]
#         else:
#             return [x+w/2,y+h/2]

#     def pix2pos_2(self,camera_pose:list,inmtx,pix:list,d=None,inv_inmtx=None):

#         x,y,height,yaw,pitch,roll=camera_pose

#         yaw=yaw*0.017453292519943
#         pitch=pitch*0.017453292519943
#         roll=roll*0.017453292519943

#         rotation_mat=np.mat([[np.cos(-yaw),-np.sin(-yaw),0],[np.sin(-yaw),np.cos(-yaw),0],[0,0,1]])\
#                     *np.mat([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])\
#                     *np.mat([[np.cos(roll),0,np.sin(roll)],[0,1,0],[-np.sin(roll),0,np.cos(roll)]])\
#                     *np.mat([[1,0,0],[0,0,1],[0,-1,0]])

#         T=-np.linalg.inv(rotation_mat)*np.mat([x,y,height]).T
#         if not isinstance(inv_inmtx,np.matrix):
#             inv_inmtx=np.linalg.inv(inmtx)

#         if not d:
#             d=height/np.sin(-pitch)

#         center=rotation_mat*(inv_inmtx*d*np.mat([inmtx[0,2],inmtx[1,2],1]).T-T)
#         z=center[2,0]

#         posNorm=rotation_mat*(inv_inmtx*np.mat([pix[0],pix[1],1]).T-T)

#         pos=[x+(posNorm[0,0]-x)/(height-posNorm[2,0])*(height-z),y+(posNorm[1,0]-y)/(height-posNorm[2,0])*(height-z),0]
#         return pos


class DelayedQueue:
    datas=[]
    delay=None
    last_data=None

    def __init__(self,delay:datetime.timedelta=datetime.timedelta()):
        self.datas=[]
        self.delay=delay

    def push(self,value):
        now=datetime.datetime.now()
        self.datas.append((now,value))
        while now - self.datas[0][0] > self.delay:
            self.datas.pop(0)

class Detect_Grtk():
    def __init__(self):

        # t = threading.Thread(target = self.sub_thread)
        # t.daemon = True
        # t.start()

        self.uav_attitude = [0.00,0.00,0.00]
        self.uav_pos = [0.00,0.00,0.00]
        self.gps_home = [0.00, 0.00, 0.00] 
        self.cam_to_world = [0.00, 0.00 ,0.00]
        self.cam_to_world_2 = [0.00,0.00,0.00] 
        self.car_pos = [0.00,0.00,0.00] 

        # self.car_pos_queue = DelayedQueue( delay=datetime.timedelta(seconds=1, milliseconds=200) )
        # self.uav_pos_queue = DelayedQueue( delay=datetime.timedelta(seconds=1, milliseconds=200) )
        # self.uav_yaw_queue = DelayedQueue( delay=datetime.timedelta(seconds=1, milliseconds=200) )

        self.cam_pos = Camera_pos()

        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest_addr = ('127.255.255.255', 5000)

        self.f=open('/home/nvidia/Develop_Project/detect_distance_gazebo/src/camrea_locate/data/dis_data.txt','w+')
        self.f.write("uav_pos.e"  + " " + "uav_pos.n"  + " " + "uav_pos.u"  + " " + 
                     "new_det.u"      + " " + "new_det.v"      + " " + 
                     "cam_to_world.e" + " " + "cam_to_world.n" + " " + "cam_to_world.u" + " " + 
                     "cam_to_world_2.e"      + " " + "cam_to_world_2.n"      + " " + "cam_to_world_2.u"      + " " + 
                     "car_pos.e"      + " " + "car_pos.n"      + " " + "car_pos.u"      + " " + 
                     "uav_roll"       + " " + "uav_pitch"      + " " + "uav_yaw"      + " " +  "\n"
    )

    def sub(self):
        rospy.Subscriber("/uav1/yolov5_detect_node/detect", detect , self.yolov5_sub)
        rospy.Subscriber("/uav1/prometheus/drone_state", DroneState , self.uav_attitude_sub)
        rospy.Subscriber("/uav1/mavros/local_position/pose", PoseStamped , self.local_pose_sub)
        rospy.Subscriber("/ugv_0/odom", Odometry , self.car_gps_sub)
        # rospy.spin()

    def local_pose_sub(self,msg):
        self.uav_pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        # self.uav_pos_queue.push(pos)
        
    def uav_attitude_sub(self,msg):
        # print(msg)
        self.uav_attitude[0] = msg.attitude[0]*57.3
        self.uav_attitude[1] = msg.attitude[1]*57.3
        self.uav_attitude[2] = ((-1 * msg.attitude[2]*57.3) + 90 + 360) % 360
        # self.uav_yaw_queue.push(msg)
    
    def car_gps_sub(self, msg):
        self.car_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]


    def yolov5_sub(self,data):
        if data:
            for i in range(len(data.num)):
                d = []
                d.append(float(data.box_x[i]))
                d.append(float(data.box_y[i]))

                new_det = self.cam_pos.pos(self.uav_pos[2], d)
                self.cam_to_world[0] = self.uav_pos[0] + self.cam_pos.camera_pos[0] * math.cos(self.uav_attitude[2]/57.3) + self.cam_pos.camera_pos[1] * math.sin(self.uav_attitude[2]/57.3)
                self.cam_to_world[1] = self.uav_pos[1] - self.cam_pos.camera_pos[0] * math.sin(self.uav_attitude[2]/57.3) + self.cam_pos.camera_pos[1] * math.cos(self.uav_attitude[2]/57.3)
                self.cam_to_world[2] = 0

                detect_x = data.box_x[i] - data.size_x[i]/2
                detect_y = data.box_y[i] - data.size_y[i]/2
                detect_w = data.size_x[i]
                detect_h = data.size_y[i]

                p = [self.uav_pos[0], self.uav_pos[1], self.uav_pos[2], self.uav_attitude[2], self.uav_attitude[1], self.uav_attitude[0]]
                camera_pose=self.cam_pos.getCameraPose(p, -90)
                pix=self.cam_pos.getFixedPix(detect_x,detect_y,detect_w,detect_h,camera_pose[4])    #修正斜视偏差
                pix=self.cam_pos.point2point(pix)                                                   #畸变矫正
                self.cam_to_world_2=self.cam_pos.pix2pos_2(camera_pose,self.cam_pos.mtx,pix,inv_inmtx=self.cam_pos.inv_mtx)  #解算目标坐标

                self.save_send(new_det)

    def save_send(self, new_det):
        print("{0:>8} {1:>6} {2:>6}".format("det_uv:",int(new_det[0]), int(new_det[1])))
        print("{0:>8} {1:>6} {2:>6} {3:>6}".format("uav_att:", round(self.uav_attitude[0], 2), round(self.uav_attitude[1], 2), round(self.uav_attitude[2], 2)))
        print("{0:>8} {1:>6} {2:>6} {3:>6}".format("cam_pos:",round(self.cam_pos.camera_pos[0],2), round(self.cam_pos.camera_pos[1],2), round(self.cam_pos.camera_pos[2],2)))
        print("{0:>8} {1:>6} {2:>6} {3:>6}".format("uav_pos:",round(self.uav_pos[0],2),round(self.uav_pos[1],2),round(self.uav_pos[2],2)))
        print("{0:>8} {1:>6} {2:>6} {3:>6}".format("c_to_w:",round(self.cam_to_world[0],2),round(self.cam_to_world[1],2),round(self.cam_to_world[2],2)))
        print("{0:>8} {1:>6} {2:>6} {3:>6}".format("c_to_w2:",round(self.cam_to_world_2[0],2),round(self.cam_to_world_2[1],2),round(self.cam_to_world_2[2],2)))
        print("{0:>8} {1:>6} {2:>6} {3:>6}".format("car_pos:",round(self.car_pos[0],2),round(self.car_pos[1],2),round(self.car_pos[2],2)))
        print('------------------------------')
                    
        uav = str(self.uav_pos[0]) + " " + str(self.uav_pos[1]) + " " + str(self.uav_pos[2])
        det = str(new_det[0]) + " " + str(new_det[1])
        c_w = str(self.cam_to_world[0]) + " " +  str(self.cam_to_world[1]) + " " + str(self.cam_to_world[2])
        c_w2 = str(self.cam_to_world_2[0]) + " " +  str(self.cam_to_world_2[1]) + " " + str(self.cam_to_world_2[2])
        car = str(self.car_pos[0]) + " " + str(self.car_pos[1]) + " " + str(self.car_pos[2])
        attitude = str(self.uav_attitude[0]) + " " + str(self.uav_attitude[1]) + " " + str(self.uav_attitude[2])
        data = uav + " " + det + " " + c_w + " " + c_w2 + " " + car + " " + attitude + "\n"

        # print(data)
        
        # data+=(1024-len(data.encode('utf-8')))*' '
        # self.udp_socket.sendto(data.encode('utf-8'), self.dest_addr)

        self.f.write(data)

    def caculate(self, start, end):
        #start[0] = latitude, start[1] = longitude, start[2] = altitude
        C_EARTH = 6378137.0
        pos = [0.00, 0.00, 0.00]
        #通过经纬度计算位置偏差
        deltaLon   = (start[1] - end[1]) / 57.3
        deltaLat   = (start[0] - end[0]) / 57.3

        pos[0] = -1 * deltaLon * C_EARTH * math.cos(start[0]/57.3)
        pos[1] = -1 * deltaLat * C_EARTH
        pos[2] = -1 * (start[2] - end[2])
        return pos
        

def parse_args():
    desc = 'save data'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--save', dest='save',
                        help='0 or 1',
                        default = 0, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    rospy.init_node("locate_node", anonymous=True)

    detect_grtk=Detect_Grtk()
    detect_grtk.sub()

    rospy.spin()
    detect_grtk.udp_socket.close()

