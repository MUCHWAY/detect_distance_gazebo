#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import rospy
import socket
from nav_msgs.msg import *
import json

ugv_num = 2
ugv_pos = []
for i in range(ugv_num):
    ugv_pos.append([0,0,0])

def ugv_pos_sub(msg, i):
    ugv_pos[i] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]

if __name__ == "__main__":
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    dest_addr = ('192.168.3.255', 6000)

    rospy.init_node('ugv_pos_node', anonymous= True)

    subscribers = []
    for i in range(ugv_num):
        subscriber = rospy.Subscriber("/ugv_" + str(i)+ "/odom", Odometry , ugv_pos_sub, i, 10)
        subscribers.append(subscriber)
    
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        data = json.dumps([{'x':ugv_pos[i][0],'y':ugv_pos[i][1]} for i in range(len(ugv_pos))]).encode('utf-8')
        print(data)
        udp_socket.sendto(data, dest_addr)
        rate.sleep()
