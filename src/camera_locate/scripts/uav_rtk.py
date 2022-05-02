#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy
import serial
import os
import time
from grtk.msg import GNGGA

if __name__ == "__main__":
  rospy.init_node('uav_rtk_pub',anonymous=True)

  #ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.2)

  pub = rospy.Publisher("/uav_rtk", GNGGA, queue_size=10)

  msg=GNGGA()

  rate = rospy.Rate(10)
  
  while not rospy.is_shutdown():
    # recv = ser.readline()
    # data = recv.decode('utf8','ignore')

    data = "$GNGGA,025754.00,4004.74102107,N,11614.19532779,E,1,18,0.7,63.3224,M,-9.7848,M,, *58"

    data_h = data.find('GNGGA')
    print(data_h)

    if(data_h != -1):
        data_list = data.split(',')
        print(data_list)
        msg.utc = float(data_list[1])
        msg.lat = float(data_list[2])/100
        msg.lon = float(data_list[4])/100
        msg.qual = float(data_list[6])
        msg.sats = int(data_list[7])
        msg.alt = float(data_list[9])

        pub.publish(msg)

    rate.sleep()