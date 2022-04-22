#include <iostream>
#include <chrono>
#include <cmath>

#include "yolov5_detect/cuda_utils.h"
#include "yolov5_detect/logging.h"
#include "yolov5_detect/common.hpp"
#include "yolov5_detect/utils.h"
#include "yolov5_detect/calibrator.h"
#include "yolov5_detect/preprocess.h"
#include "yolov5_detect/video.h"
#include "yolov5_detect/detect.h"
#include "yolov5_detect/sort.h"
#include "yolov5_detect/VideoCapture.hpp"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"


#include <thread>
#include <mutex>
#include <vector>
#include <ros/ros.h>

using namespace std;

using namespace std;
using namespace cv;

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // ensure it exceed the maximum size in the input images ! 

// stuff we know about the network and the input/output ,blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize) {
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);

    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);
}

int main(int argc, char** argv) {
    string node_name;
    node_name = "yolov5_detect_node";
    ros::init(argc, argv, node_name);
    ros::NodeHandle nh;

    ros::Publisher detect_pub = nh.advertise<yolov5_detect::detect>(node_name + "/detect", 1000);

    std::string engine_name;
    ros::param::get("~engine_name", engine_name);

    std::string video_name;
    ros::param::get("~video_name", video_name);

    std::string video_out_path;
    ros::param::get("~video_out_path", video_out_path);

    std::string img_topic;
    ros::param::get("~img_topic", img_topic);

    std::string uav_num;
    ros::param::get("~uav_num", uav_num);
    int num = 1;

    // Img_update img_update(video_name);
    // thread img_update_thread(&Img_update::update, &img_update);//传递初始函数作为线程的参数

    Ros_image ros_img(img_topic, num);
    thread ros_img_thread(&Ros_image::img_update, &ros_img); //传递初始函数作为线程的参数

    cudaSetDevice(DEVICE);

    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    static float prob[BATCH_SIZE * OUTPUT_SIZE];

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    float* buffers[2];
    
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);

    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;

    // prepare input data cache in pinned memory 
    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));

    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
        
    // cv::namedWindow("Display");
    cv::VideoWriter outputVideo;
    outputVideo.open( video_out_path + "detect.avi",  cv::VideoWriter::fourcc('M', 'P', '4', '2'), 10.0, cv::Size(1024, 1024));
    bool write = false;
    
    cv::Mat raw_img ;
    cv::Mat img;
    cv::Mat final;
    yolov5_detect::detect detect_msg;
    vector<Detect_Result> detect_result;
    Detect_Result dr;
    vector<TrackingBox> track_result;
    Sort sort_track;
    
    // Img_split_focus split_focus(img_update.width, img_update.height);

    float result_sum[61];
    int sum_index;

    std::vector<Yolo::Detection> final_res;
    std::vector<Yolo::Detection> res;

    std::string class_[1]={"car"};

    // cv::VideoCapture capture(video_name);

    while(ros::ok()) 
    {
        auto start = std::chrono::system_clock::now();

        // if(img_update.img_flag==0) break;
        // raw_img=img_update.get_img();

        img = ros_img.img.clone();
        
        // if(!capture.read(raw_img)) break;

        memset(result_sum, 0, sizeof(result_sum));
        sum_index=1;

        float* buffer_idx = (float*)buffers[inputIndex];
        size_t  size_image = img.cols * img.rows * 3;
        size_t  size_image_dst = INPUT_H * INPUT_W * 3;
        // --------------------------------------------------------------------------------------------21ms
        //copy data to pinned memory
        memcpy(img_host,img.data,size_image);
        // --------------------------------------------------------------------------------------------22ms
        //copy data to device memory
        CUDA_CHECK(cudaMemcpyAsync(img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));
        // --------------------------------------------------------------------------------------------23ms
        preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);   
        // --------------------------------------------------------------------------------------------23ms
        buffer_idx += size_image_dst;
        // Run inference
        doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE); //time：20ms
        // --------------------------------------------------------------------------------------------42ms

        final_res.clear();
        nms(final_res, prob, CONF_THRESH, NMS_THRESH); 

        if(final_res.size() != 0){
            for (size_t m = 0 ; m < final_res.size(); m++) {
                
                // detect_msg.num.push_back((int8_t)m);
                // detect_msg.class_name.push_back(class_[(int)final_res[m].class_id]);
                // detect_msg.conf.push_back((int8_t)(final_res[m].conf*100));
                // detect_msg.box_x.push_back((int16_t)final_res[m].bbox[0]); 
                // detect_msg.box_y.push_back((int16_t)final_res[m].bbox[1]);
                // detect_msg.size_x.push_back((int16_t)final_res[m].bbox[2]);
                // detect_msg.size_y.push_back((int16_t)final_res[m].bbox[3]);

                dr.target_location.x = (unsigned int)final_res[m].bbox[0];
                dr.target_location.y = (unsigned int)final_res[m].bbox[1];
                dr.target_location.width = (int)final_res[m].bbox[2];
                dr.target_location.height = (int)final_res[m].bbox[3];
                dr.target_location_confidence = final_res[m].conf;
                dr.target_name = class_[(int)final_res[m].class_id];

                detect_result.push_back(dr);

                // 在图上直接画出检测框
                // cv::Rect r( (int)(final_res[m].bbox[0]-final_res[m].bbox[2]/2),
                //             (int)(final_res[m].bbox[1]-final_res[m].bbox[3]/2),
                //             (int)final_res[m].bbox[2],
                //             (int)final_res[m].bbox[3]);

                // cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                // cv::putText(img, class_[(int)final_res[m].class_id], cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 0, 255), 2);
                // cv::putText(img, std::to_string(final_res[m].conf).substr(0,4), cv::Point(r.x+60, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 0, 255), 2);
                // cv::putText(img, std::to_string((int)final_res[m].bbox[2])+" "+std::to_string((int)final_res[m].bbox[3]), cv::Point(r.x+120, r.y), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 0, 255), 2);
                // cv::circle(img, cv::Point((int)final_res[m].bbox[0], (int)final_res[m].bbox[1]), 3, cv::Scalar(0, 0, 255), - 1);
            }
        }

        cv::line(img, cv::Point(511, img.rows), cv::Point(511, 0), cv::Scalar(0, 0, 255), 2, 4);
        cv::line(img, cv::Point(img.cols,511), cv::Point(0,511), cv::Scalar(0, 0, 255), 2, 4);

        track_result = sort_track.tracking(detect_result);
        
        if(track_result.size() != 0) {
            for(TrackingBox t : track_result) {
                cout<<t.box.x<<" "<<t.box.y<<" "<<t.box.width<<' '<<t.box.height<<endl;
                cout<<t.confidence<<endl;
                cout<<t.id<<endl;
                cout<<t.target_name<<endl;

                detect_msg.num.push_back((int8_t)t.id);
                detect_msg.class_name.push_back(t.target_name);
                detect_msg.conf.push_back((int8_t)(t.confidence*100));
                detect_msg.box_x.push_back((int16_t)t.box.x); 
                detect_msg.box_y.push_back((int16_t)t.box.y);
                detect_msg.size_x.push_back((int16_t)t.box.width);
                detect_msg.size_y.push_back((int16_t)t.box.height);

                // 在图上画出跟踪结果
                if(t.confidence != 0) {
                    cv::Rect r( (int)(t.box.x - t.box.width/2), (int)(t.box.y - t.box.height/2),(int)t.box.width,(int)t.box.height);
                    cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                    cv::putText(img, t.target_name, cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 0, 255), 2);
                    cv::putText(img, std::to_string(t.confidence).substr(0,4), cv::Point(r.x+40, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 0, 255), 2);
                    cv::putText(img, std::to_string(t.id), cv::Point(r.x+100, r.y), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 0, 255), 2);
                    cv::circle(img, cv::Point((int)t.box.x, (int)t.box.y), 3, cv::Scalar(0, 0, 255), - 1);
                }  
            }
        }
        detect_pub.publish(detect_msg);
        detect_result.clear();
        track_result.clear();

        detect_msg.num.clear();
        detect_msg.class_name.clear();
        detect_msg.conf.clear();
        detect_msg.box_x.clear();
        detect_msg.box_y.clear();
        detect_msg.size_x.clear();
        detect_msg.size_y.clear(); 

        // cv::resize(img, final, cv::Size(512,512));
        // cv::imshow("Display", img);
        int k = cv::waitKey(1); 
        static int num = 0;
        if(k == 115)  {
            write = true;
            cv::imwrite(to_string(num)+".jpg", raw_img);
            num ++;
        }

        if(write) cout<<write<<endl, outputVideo.write(img);
        
        auto end = std::chrono::system_clock::now();
        std::cout << "sum: " <<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<< "ms  "<<endl;
        cout<<"--------------------------"<<endl;
    }
    outputVideo.release();
    // cv::destroyWindow("Display");
    // img_update_thread.join();

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(img_device));
    CUDA_CHECK(cudaFreeHost(img_host));
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}


