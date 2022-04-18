#ifndef VideoCapture_class
#define VideoCapture_class

#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define RTSP_CAMERA_IP "192.168.42.108"

#define MTX_COEF 2.78880725e+03,0,2.02409862e+03,0,2.71633191e+03,9.84151725e+02,0,0,1.0
#define NEW_CAMERA_MTX_COEF 2.76520630e+03,0,2.02316235e+03,0,3.18689380e+03,1.15358220e+03,0,0,1.0
#define DIST_COEF 0.02834078,-0.25786759,-0.0052794,0.0007245,0.34212522

class CachedVideoCapture
{

public:
    CachedVideoCapture(std::string ip=RTSP_CAMERA_IP,int subtype=0,int width=4096,int height=2160,float freq=10)
    {
        this->ip=ip;
        this->subtype=subtype;
        this->width=width;
        this->height=height;
        this->freq=freq;
        init();
    }

    virtual ~CachedVideoCapture()
    {
        isEnding=true;
        delete cap;
    }

    bool isOpened()
    {
        return cap->isOpened();
    }

    CachedVideoCapture& operator >> (cv::Mat& frameDst)
    {
        //while(isWriting);
        //while(isReading);
        //isReading=true;
        //cout<<"1"<<endl;
        //cv::remap(frameCp,frameDst,map1,map2,INTER_LINEAR);
        //cv::undistort(frameCp,frameDst,mtx,dist,newcameramtx);
        //frameCp.copyTo(frameDst);
        //isReading=false;
        //isWriting=true;
        *cap >> frameDst;
        return *this;
    }

    void undistortPoints(std::vector<cv::Point2f>& points)
    {
        cv::undistortPoints(points, points, mtx, dist, cv::Mat(), newcameramtx);
    }

    void undistortFrame(cv::Mat& frame)
    {
        cv::remap(frame,frame,map1,map2,INTER_LINEAR);
    }

private:

    bool isEnding=false;
    bool isReading=false;
    bool isWriting=false;

    std::string ip;
    int subtype;
    int width,height;
    float freq;

    clock_t now,last;

    cv::VideoCapture* cap;
    pthread_t clock_pthread;
    cv::Mat frame;
    cv::Mat frameCp;

    cv::Mat mtx;
    cv::Mat newcameramtx;
    cv::Mat dist;

    cv::Mat map1,map2;

    void init()
    {
        mtx=(cv::Mat_<double>(3, 3)<<MTX_COEF);
        newcameramtx=(cv::Mat_<double>(3, 3)<<NEW_CAMERA_MTX_COEF);
        dist=(cv::Mat_<double>(1, 5)<<DIST_COEF);

        cv::initUndistortRectifyMap(mtx, dist, cv::Mat(), newcameramtx, cv::Size(width,height), CV_16SC2, map1, map2);

        cap = new cv::VideoCapture("rtsp://admin:admin@"+ip+":554/cam/realmonitor?channel=1&subtype="+std::to_string(subtype),cv::CAP_FFMPEG);
        *cap >> frame;
        frame.copyTo(frameCp);
        //pthread_create(&clock_pthread, NULL, &CachedVideoCapture::runByThread, this);
    }

    static void* runByThread(void* self) 
    {
        return static_cast<CachedVideoCapture*>(self)->run();
    }

    void* run()
    {
        long interval=(long)(CLOCKS_PER_SEC/freq*0.95);
        clock_t tick=clock(),tock;
        while(!isEnding)
        {
            do
            {
                usleep(1000);
                tock=clock();
            } while (tock-tick<=interval);
            
            tick=tock;
            *cap >> frame;
            //cv::remap(frame,frame,map1,map2,INTER_LINEAR);
            cout<<"2"<<endl;
            while(isReading);
            cout<<"3"<<endl;
            isReading=true;
            frame.copyTo(frameCp);
            isReading=false;
            isWriting=false;
        }
        return nullptr;
    }

};

#endif

