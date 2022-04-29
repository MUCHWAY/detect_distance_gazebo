# include "yolov5_detect/video.h"
std::mutex m;

Ros_image::Ros_image(string &topic) {
    img_topic = topic;
    update = 30;
}

void Ros_image::img_callback(const sensor_msgs::ImageConstPtr &msg)
{
  try
  {
    img = cv_bridge::toCvShare(msg, "bgr8")->image;
    update = 0;
    if(update > 30) update = 30;
    // cv::imshow("view", img);
    // cv::waitKey(1);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

void Ros_image::img_update() {
    img_sub = n_private.subscribe(img_topic, 1, &Ros_image::img_callback, this, ros::TransportHints().tcpNoDelay());
    ros::spin();
}

Img_update::Img_update(std::string path){
    img_path=path;
    img_flag=1;
    width = 0;
    height = 0;

    mtx=(cv::Mat_<double>(3, 3)<<MTX_COEF);
    newcameramtx=(cv::Mat_<double>(3, 3)<<NEW_CAMERA_MTX_COEF);
    dist=(cv::Mat_<double>(1, 5)<<DIST_COEF);
}

void Img_update::update(){
    cv::VideoCapture capture(img_path, cv::CAP_FFMPEG);
    
    height = capture.get(4);
    width = capture.get(3);
    
    if(!capture.isOpened()){
        cout<<"img open Error"<<endl;
        return;
    }
    else cout<<"img open successed"<<endl;

    while(ros::ok()){
        m.lock(); // 上锁
        if(!capture.read(img)) {
            img_flag=0;
            break;
        }else img_flag=1;
        return_img=img.clone();
        // img=cv::imread("../detect/test_image/DJI_0708.jpg");
        m.unlock(); // 解锁
    }
}

cv::Mat Img_update::get_img(){
    return return_img;
}

void Img_update::undistortPoints(std::vector<cv::Point2f>& points)
{
    cv::undistortPoints(points, points, mtx, dist, cv::Mat(), newcameramtx);
}

Img_split_focus::Img_split_focus(const vector<int> &img_vec){

    img_size[0] = img_vec[0]; img_size[1] = img_vec[1];

    split_size[0]=1024;split_size[1]=1024;
    focus_size[0]=1024;focus_size[1]=1024;

    // x_num=(int)(img_size[0]/split_size[0])+1;
    // y_num=(int)(img_size[1]/split_size[1])+1;
    x_num = img_vec[2];
    y_num = img_vec[3];

    cout<<"x_num: "<<x_num<<endl;
    cout<<"y_num: "<<y_num<<endl;

    int i;
    split_x.push_back(0);
    split_y.push_back(0);

    // for(i=1;i<=x_num-1;i++){
    //     split_x.push_back( ((img_size[0]-split_size[0])/(x_num-1))*i );
    //     cout<< "x: "<<((img_size[0]-split_size[0])/(x_num-1))*i<<endl;
    // }
    // overlap[0]=(float)(split_size[0]-split_x[1]);
    // // overlap[0]=(float)(split_size[0]-split_x[1])/split_size[0];

    // for(i=1;i<=y_num-1;i++){
    //     split_y.push_back( ((img_size[1]-split_size[1])/(y_num-1))*i );
    //     cout<<"y: "<< ((img_size[1]-split_size[1])/(y_num-1))*i<<endl;
    // }
    // overlap[1]=(float)(split_size[1]-split_y[1]);
    // // overlap[1]=(float)(split_size[1]-split_y[1])/split_size[1];
    // cout<<"overlap_x: "<<abs(overlap[0])<<" overlap_y:"<<abs(overlap[1])<<endl;

    overlap[0] = img_vec[4]; overlap[1] = img_vec[5];
    for(i=1; i<x_num; i++){
        split_x.push_back(split_x[i - 1] + split_size[0] - overlap[0]);
        cout<<split_x[i]<<endl;
    }

    for(i = 1; i < y_num; i ++){
        split_y.push_back(split_y[i - 1] + split_size[1] - overlap[1]);
        cout<<split_y[i]<<endl;
    }
}