#ifndef VIDEO_H
#define VIDEO_H

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <chrono>
#include <cmath>

#include <pthread.h>
#include <thread>
#include <mutex>
#include <vector>
#include <ros/ros.h>

using namespace sensor_msgs;
using namespace std;

#define MTX_COEF 2.78880725e+03,0,2.02409862e+03,0,2.71633191e+03,9.84151725e+02,0,0,1.0
#define NEW_CAMERA_MTX_COEF 2.76520630e+03,0,2.02316235e+03,0,3.18689380e+03,1.15358220e+03,0,0,1.0
#define DIST_COEF 0.02834078,-0.25786759,-0.0052794,0.0007245,0.34212522

class Ros_image {
    private:
        ros::NodeHandle n_private;
        ros::Subscriber img_sub;
        pthread_t sub_pthread;

        void img_callback(const sensor_msgs::ImageConstPtr &msg);

    public:
        cv::Mat img;   
        string img_topic;
        int update;
        Ros_image(string &topic);
        void img_update() ;
};

class Img_update{

    private:
        std::string img_path;
    public:
        cv::Mat img;
        cv::Mat return_img;
        // cv::Mat r_img;
        int img_flag;
        cv::Mat mtx;
        cv::Mat newcameramtx;
        cv::Mat dist;
        int width;
        int height;
        
        Img_update(std::string path);
        void update();
        cv::Mat get_img();
        void undistortPoints(std::vector<cv::Point2f>& points);
};

class Img_split_focus{
    public:
        int split_size[2];
        int focus_size[2];
        int img_size[2];
        vector<int> split_x;
        vector<int> split_y;
        vector<int> focus_x;
        vector<int> focus_y;
        int x_num;
        int y_num;
        float overlap[2];
        Img_split_focus(const vector<int> &img_vec);
};

#endif