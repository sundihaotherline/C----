#include "common/common.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap("resources/task_3.mp4");
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频" << std::endl;
        return -1;
    }

    cv::Mat frame, hsv, maskLow, maskHigh, mask;
// 【关键1】：把核的大小从 (5,5) 缩小到 (3,3)，保护细线
cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

while (cap.read(frame)) {
    // 1. 颜色提取（放宽 S、V 下限，包容细线边缘的渐变色）
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    // 细线往往发光，S和V会很高，但边缘会衰减，所以下限设低一点（比如60）
    cv::inRange(hsv, cv::Scalar(0, 60, 60), cv::Scalar(15, 255, 255), maskLow);
    cv::inRange(hsv, cv::Scalar(160, 60, 60), cv::Scalar(179, 255, 255), maskHigh);
    cv::bitwise_or(maskLow, maskHigh, mask);

    // 2. 【关键2】：放弃开运算，改用闭运算（连接断裂的细线）
    // 如果你觉得连起来后噪点太多，可以把下面这行注释掉，完全不进行形态学操作！
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    
    // 【可选3】：如果线条实在太细，可以来一次微小的膨胀让它变粗，方便后续找轮廓
    // cv::dilate(mask, mask, kernel);

    // ... 后面的 findContours 和画圆逻辑保持不变 ...


cv::VideoCapture cap("resources/task_3.mp4"); // 记得改路径
    if (!cap.isOpened()) return -1;

    int fps = (int)cap.get(cv::CAP_PROP_FPS);
    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter writer("result/task3_windmill/task_3/recognition_overlay.mp4",
                           cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(w, h));

    cv::Mat frame, hsv, maskLow, maskHigh, mask;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    cv::Point2f centerR(w / 2.0, h / 2.0); // 暂用图像中心占位

    while (cap.read(frame)) {
        // 1. 颜色提取（收紧 H 上限为 15，排除 H=30 的黄光）
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(0, 80, 80), cv::Scalar(15, 255, 255), maskLow);
        cv::inRange(hsv, cv::Scalar(160, 80, 80), cv::Scalar(179, 255, 255), maskHigh);
        cv::bitwise_or(maskLow, maskHigh, mask);

        // 2. 形态学去噪
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

        // 3. 找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        bool found = false;
        // 4. 遍历轮廓：面积筛选 + 圆形框定
        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);

            // =============== 【面积筛选核心】 ===============
            // 根据你的视频分辨率，手动调整这两个数值
            // 太小（噪点）：比如 < 50
            // 太大（整个背景误识别）：比如 > 5000
            if (area < 50 || area > 5000) continue; 
            // ==============================================

            found = true;
            cv::Point2f targetCenter;
            float radius;
            // 最小外接圆：提取圆心和半径
            cv::minEnclosingCircle(contours[i], targetCenter, radius);

            // 画红色圆形框
            cv::circle(frame, targetCenter, (int)radius, cv::Scalar(0, 0, 255), 2);
            cv::circle(frame, targetCenter, 3, cv::Scalar(0, 0, 255), -1); // 圆心

            // 画连线
            cv::line(frame, centerR, targetCenter, cv::Scalar(0, 255, 255), 2);
            
            // 在目标旁边写上面积
            std::string areaText = "Area: " + std::to_string((int)area);
            cv::putText(frame, areaText, cv::Point(targetCenter.x - 30, targetCenter.y - 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        // 5. 绘制中心点 R
        cv::circle(frame, centerR, 5, cv::Scalar(255, 255, 255), -1);

        // 6. 状态显示
        if (found) {
            cv::putText(frame, "State: DETECTED", cv::Point(30, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(frame, "State: LOST", cv::Point(30, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        }

        // 【死命令】必须每帧都写！哪怕是丢失状态，也要保留原帧！
        writer.write(frame);
        cv::imshow("Tracking", frame);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    return 0;
}





























}