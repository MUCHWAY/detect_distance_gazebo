#include "yolov5_detect/sort.h"

extern std::map<unsigned int, manage> mng;

Sort::Sort()
{
	frame_count = 0;
	max_age = 60; //仅返回在当前帧出现且命中周期大于min_hits（除非跟踪刚开始）的跟踪结果；如果未命中时间大于max_age则删除跟踪器。
	min_hits = 2;
	iouThreshold = 0.1;
	trkNum = 0; //同一帧中跟踪目标数量
	detNum = 0; //同一帧中检测目标数量
	KalmanTracker::kf_count = 0;
}

double Sort::GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt) //计算检测框与跟踪框的交并比（IOU）
{
	float in = (bb_test & bb_gt).area();		   //检测与跟踪框的交集
	float un = bb_test.area() + bb_gt.area() - in; //检测与跟踪框的并集
	if (un < DBL_EPSILON)
		return 0;
	return (double)(in / un);
}

vector<TrackingBox> Sort::tracking(vector<Detect_Result> detections)
{
	//std::cout<<"tracking start."<<std::endl;
	if (trackers.empty())
	{ //第一次开始跟踪，以检测结果初始化跟踪器
		for (int i = 0; i < detections.size(); i++)
		{
			KalmanTracker trk = KalmanTracker(detections[i].target_location);
			trackers.push_back(trk);

			TrackingBox temp;
			temp.box = trk.get_state();
			temp.id = trk.m_id;
			temp.target_name = detections[i].target_name;
			temp.confidence = detections[i].target_location_confidence;
			//temp.frame=frame_count;
			frameTrackingResult.push_back(temp); //将当前帧的跟踪结果放入frameTrackingResult
		}
		return frameTrackingResult;
	}

	predictedBoxes.clear();

	for (auto it = trackers.begin(); it != trackers.end();)
	{
		Rect_<float> pBox = it->predict(); //根据上一帧的跟踪框得到当前帧的预测框

		if (pBox.x >= 0 && pBox.y >= 0)
		{
			predictedBoxes.push_back(pBox); //将有效的预测框放入predictedBoxes
			it++;
		}
		else
		{
			it = trackers.erase(it); //如果预测框无效，则删除上一帧对应的的跟踪框
									 //cout<<trackers.head<<endl;
									 //cout<<it<<endl;
		}
	}

	trkNum = predictedBoxes.size(); //预测的有效窗口
	detNum = detections.size();		//检测的窗口
	iouMatrix.clear();
	iouMatrix.resize(trkNum, vector<double>(detNum, 0));

	for (int i = 0; i < trkNum; i++)
	{
		for (int j = 0; j < detNum; j++)
		{ //计算得到所有检测框与所有跟踪框形成的trkNum*detNum维的iou（交并比）矩阵iouMatrix
			iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detections[j].target_location);
		}
	}

	HungarianAlgorithm HungAlgo;
	assignment.clear();

	HungAlgo.Solve(iouMatrix, assignment); //匈牙利算法根据iou（交并比）矩阵iouMatrix得到跟踪框和检测框之间的匹配，匹配结果存入assignment

	unmatchedTrajectories.clear();
	unmatchedDetections.clear();
	allItems.clear();
	matchedItems.clear();

	if (detNum > trkNum)
	{
		for (int n = 0; n < detNum; n++)
		{
			allItems.insert(n); //将所有检测框的ID(下标)放入allItems集合
		}
		for (int i = 0; i < trkNum; i++)
		{
			matchedItems.insert(assignment[i]); //将所有匹配成功的检测框的ID(下标)放入matchedItems
		}
		set_difference(allItems.begin(), allItems.end(),
					   matchedItems.begin(), matchedItems.end(),
					   insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin())); //计算出：unmatchedDetections=allItems-matchedItems
	}
	else if (detNum < trkNum)
	{
		for (int i = 0; i < trkNum; i++)
		{
			if (assignment[i] == -1)
			{
				unmatchedTrajectories.insert(i); //将所有未匹配成功的跟踪框的ID(下标)放入unmatchedTrajectories
			}
		}
	}
	else
	{
		;
	}

	matchedPairs.clear();

	for (int i = 0; i < trkNum; i++)
	{
		if (assignment[i] == -1)
			continue;
		if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
		{
			unmatchedTrajectories.insert(i);		   // 将已经匹配成功但是iou小于阈值的跟踪框加入未匹配的跟踪框集合
			unmatchedDetections.insert(assignment[i]); // 将已经匹配成功但是iou小于阈值的检测框加入未匹配的检测框集合
		}
		else
		{
			matchedPairs.push_back(cv::Point(i, assignment[i])); // 建立匹配成功的检测框与跟踪框的ID(下标)点对
		}
	}

	int detIdx, trkIdx;

	for (int i = 0; i < matchedPairs.size(); i++)
	{
		trkIdx = matchedPairs[i].x; // 取跟踪框下标
		detIdx = matchedPairs[i].y; // 取检测框下标

		(trackers[trkIdx]).update(detections[detIdx].target_location); //用匹配的检测结果作为当前帧的跟踪结果，需要变成循环找链表
		(trackers[trkIdx]).update_class_info(detections[detIdx]);
	}

	for (auto umd : unmatchedDetections)
	{
		KalmanTracker tracker = KalmanTracker(detections[umd].target_location); //对于未匹配的检测框生成新的跟踪器
		trackers.push_back(tracker);											//将新的跟踪器加入有效跟踪序列
	}

	frameTrackingResult.clear();
	for (auto it = trackers.begin(); it != trackers.end();)
	{
		if ((it->m_time_since_update < 1) && (it->m_hit_streak >= min_hits || frame_count <= min_hits))
		{

			TrackingBox temp;
			temp.box = it->get_state(); //获取当前跟踪器的跟踪框
			temp.id = it->m_id;
			temp.target_name = it->m_class_name;
			temp.confidence = it->confidence;
			//temp.frame=frame_count;
			frameTrackingResult.push_back(temp); //将当前帧的跟踪结果放入frameTrackingResult
			it++;
		}
		else
		{
			it++;
		}
		if (it != trackers.end() && (it->m_time_since_update > max_age))
		{
			it = trackers.erase(it); //清除丢失的跟踪框
		}
	}

	return frameTrackingResult;
	//std::cout<<"tracking end."<<std::endl;
}
