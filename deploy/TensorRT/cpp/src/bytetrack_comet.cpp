#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "BYTETracker.h"
#include <map>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            cerr << "Cuda failure: " << ret << endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id

using namespace nvinfer1;
using namespace std;
using namespace cv;

static Logger gLogger;

// 类别映射表
const map<string, int> class_map = {
    {"car", 0},
    {"CAR", 0},
    {"truck", 1},
    {"TRUCK", 1},
    {"bus", 2},
    {"BUS", 2},
    {"bike", 3},
    {"BIKE", 3},
    {"bicycle", 4},
    {"BICYCLE", 4},
    {"pedestrian", 5},
    {"PEDESTRIAN", 5}
};

// 从文本文件读取检测结果的函数
vector<Object> load_detections_from_file(const string& det_file) {
    vector<Object> objects;
    ifstream file(det_file);
    string line;
    if (!file.is_open()) {
        cerr << "can not open file: " << det_file << endl;
        return objects;
    }

    
    while (getline(file, line)) {
        stringstream ss(line);
        string class_name;
        float conf;
        float x1, y1, x2, y2;
        
        // 文件格式：CLASS confidence x1 y1 x2 y2
        ss >> class_name >> conf >> x1 >> y1 >> x2 >> y2;
        
        // 检查是否是我们需要的类别
        auto it = class_map.find(class_name);
        if (it == class_map.end()) {
            continue;  // 跳过不需要的类别
        }
        
        // 转换为x,y,w,h格式
        float w = x2 - x1;
        float h = y2 - y1;
        
        Object obj;
        obj.rect = cv::Rect_<float>(x1, y1, w, h);
        obj.label = it->second;  // 使用映射后的类别ID
        obj.prob = conf;
        
        objects.push_back(obj);
    }
    
    return objects;
}

const float color_list[80][3] =
{
    {0.000, 0.447, 0.741},
    {0.850, 0.325, 0.098},
    {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188},
    {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184},
    {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000},
    {1.000, 0.500, 0.000},
    {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000},
    {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000},
    {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000},
    {1.000, 0.667, 0.000},
    {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500},
    {0.000, 0.667, 0.500},
    {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500},
    {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500},
    {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500},
    {0.667, 1.000, 0.500},
    {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500},
    {1.000, 0.667, 0.500},
    {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000},
    {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000},
    {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000},
    {0.667, 1.000, 1.000},
    {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000},
    {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000},
    {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000},
    {0.000, 0.167, 0.000},
    {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000},
    {0.000, 0.667, 0.000},
    {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167},
    {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500},
    {0.000, 0.000, 0.667},
    {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000},
    {0.000, 0.000, 0.000},
    {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286},
    {0.429, 0.429, 0.429},
    {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857},
    {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741},
    {0.50, 0.5, 0}
};

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "用法: " << argv[0] << " <视频路径> <检测结果目录>" << endl;
        return -1;
    }

    const string input_video_path = argv[1];
    const string det_dir = argv[2];

    VideoCapture cap(input_video_path);
    if (!cap.isOpened()) {
        cerr << "无法打开视频文件！" << endl;
        return -1;
    }

    int img_w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int img_h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    cout << "Total frames: " << nFrame << endl;

    VideoWriter writer("demo.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(img_w, img_h));

    Mat img;
    BYTETracker tracker(fps, 30);
    int num_frames = 0;
    int total_ms = 0;

    while (true) {
        if(!cap.read(img))
            break;
        
        num_frames++;
        if (num_frames % 20 == 0) {
            cout << "处理第 " << num_frames << " 帧 (" << num_frames * 1000000 / total_ms << " fps)" << endl;
        }
        
        if (img.empty())
            break;

        // 构造当前帧对应的检测结果文件名
        stringstream ss;
        ss << det_dir << "/frame_" << setw(6) << setfill('0') << num_frames << ".txt";
        string det_file = ss.str();

        auto start = chrono::system_clock::now();
        
        // 读取检测结果
        vector<Object> objects = load_detections_from_file(det_file);
        
        // 更新跟踪器
        vector<STrack> output_stracks = tracker.update(objects);
        
        auto end = chrono::system_clock::now();
        total_ms = total_ms + chrono::duration_cast<chrono::microseconds>(end - start).count();

        // 绘制跟踪结果
        for (int i = 0; i < int(output_stracks.size()); i++) {
            vector<float> tlwh = output_stracks[i].tlwh;
            bool vertical = tlwh[2] / tlwh[3] > 1.6;
            if (tlwh[2] * tlwh[3] > 20 && !vertical) {
                Scalar s = tracker.get_color(output_stracks[i].track_id);
                putText(img, format("%d", output_stracks[i].track_id), Point(tlwh[0], tlwh[1] - 5), 
                        0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
                rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
            }
        }

        putText(img, format("frame: %d fps: %d num: %d", num_frames, num_frames * 1000000 / total_ms, int(output_stracks.size())), 
                Point(0, 30), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
        writer.write(img);

        char c = waitKey(1);
        if (c > 0) {
            break;
        }
    }

    cap.release();
    cout << "FPS: " << num_frames * 1000000 / total_ms << endl;
    return 0;
}
