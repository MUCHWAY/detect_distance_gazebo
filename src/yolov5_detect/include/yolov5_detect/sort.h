#ifndef SORT_H
#define SORT_H

#include "Hungarian.h"
#include "KalmanTracker.h"
#include <iostream>
#include <set>
#include <map>
#include <vector>

using namespace std;

typedef struct TrackingBox //跟踪框
{
	int frame;
	int id;
	std::string target_name;
	float confidence;
	Rect_<float> box;
} TrackingBox;

struct manage
{
	TrackingBox trackbox;
	string name;
	int recog_flag; //识别标志位，recog_flag为1时表示该图已经识别过，为0则还没有识别。
};

class Sort
{
	public:
		int frame_count; //帧数
		int max_age;	 //仅返回在当前帧出现且命中周期大于min_hits（除非跟踪刚开始）的跟踪结果；如果未命中时间大于max_age则删除跟踪器。
		int min_hits;	 //
		double iouThreshold;

		vector<KalmanTracker> trackers;		 //记录上一帧有效的跟踪框形成的双向链表
		vector<Rect_<float>> predictedBoxes; //跟踪预测框
		vector<vector<double>> iouMatrix;
		vector<int> assignment;			//assignment[i]表示与跟踪i匹配的检测框的id(下标).
		set<int> unmatchedDetections;	//未匹配的检测框，将生成新的跟踪ID
		set<int> unmatchedTrajectories; //未匹配的跟踪框，默认目标消失，将删除ID
		set<int> allItems;				//所有检测框的ID
		set<int> matchedItems;			//所有匹配成功的检测框的ID
		vector<cv::Point> matchedPairs; //匹配成功的检测框与跟踪框的ID点对
		vector<TrackingBox> frameTrackingResult;

		unsigned int trkNum; //同一帧中跟踪目标数量
		unsigned int detNum; //同一帧中检测目标数量

		Sort();													 //构造函数
		double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt); //计算跟踪框和检测框的交并比
		vector<TrackingBox> tracking(vector<Detect_Result> detections);
};

#endif
