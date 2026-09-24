#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    // 1. 读取视频
    cv::VideoCapture cap("resources/task_3.mp4");
    if (!cap.isOpened()) {
        std::cerr << "❌ 无法打开视频，请检查路径！" << std::endl;
        return -1;
    }

    int fps = (int)cap.get(cv::CAP_PROP_FPS);
    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // 2. 准备视频写入
    cv::VideoWriter writer("result/task3_windmill/task_3/recognition_overlay.mp4",
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(w, h));

    cv::Mat frame, hsv, maskLow, maskHigh, mask;
    // 核大小用 7x7，专门对付同心圆中间的缝隙，让它们连成一片
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));

    // ========== 核心跟踪状态变量（必须在循环外） ==========
    int currentTargetId = -1;          // -1 表示没有锁定任何目标
    cv::Point2f lastCenter(-1, -1);    // 上一帧目标的中心
    int lostFrameCount = 0;            // 丢失帧计数器
    const int MAX_LOST_FRAMES = 30;    // 允许丢失的最大帧数（约0.5秒@60fps）
    const double MATCH_DIST_THRESH = 80.0; // 距离匹配阈值，防止跳到远处的目标

    // 假设图像中心是 R 标中心（任务书提示实际需检测，这里先用图像中心占位）
    cv::Point2f centerR(w / 2.0, h / 2.0); 

    while (cap.read(frame)) {
        // 1. 颜色提取（红色双区间，宽松一点保证边缘和断裂处不漏）
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(0, 60, 60), cv::Scalar(15, 255, 255), maskLow);
        cv::inRange(hsv, cv::Scalar(160, 60, 60), cv::Scalar(179, 255, 255), maskHigh);
        cv::bitwise_or(maskLow, maskHigh, mask);

        // 2. 形态学闭运算（先膨胀后腐蚀，填平同心圆环的断裂缝隙）
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        // 3. 找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 4. 遍历轮廓，应用几何过滤，收集候选目标
        std::vector<cv::Point2f> candidates;
        std::vector<float> radii;
        
        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            double perimeter = cv::arcLength(contours[i], true);

            // 【面积过滤】：排除太小的碎线、太大的背景（R标面积通常较小会被此条件过滤）
            if (area < 300 || area > 10000) continue; 
            if (perimeter == 0) continue;

            // 【圆度过滤】：圆度 = 4*PI*面积 / 周长^2，理想圆为1
            double circularity = 4 * M_PI * area / (perimeter * perimeter);
            // 同心圆即使断裂，圆度也较高；而R字母是不规则多边形，圆度很低
            if (circularity < 0.6) continue; 

            // 通过筛选，计算最小外接圆
            cv::Point2f center;
            float radius;
            cv::minEnclosingCircle(contours[i], center, radius);
            candidates.push_back(center);
            radii.push_back(radius);
        }

        // ==================== 状态机：稳定锁定逻辑 ====================
        bool found = false;
        cv::Point2f targetCenter;
        float targetRadius = 0;

        if (!candidates.empty()) {
            // 情况A：之前已经有目标，尝试找离上一帧最近的候选者
            if (currentTargetId != -1 && lastCenter.x >= 0) {
                double minDist = 1e9;
                int bestIdx = -1;
                for (size_t i = 0; i < candidates.size(); i++) {
                    double dist = cv::norm(candidates[i] - lastCenter);
                    if (dist < minDist && dist < MATCH_DIST_THRESH) {
                        minDist = dist;
                        bestIdx = (int)i;
                    }
                }
                if (bestIdx != -1) {
                    // 匹配成功，保持ID不变
                    targetCenter = candidates[bestIdx];
                    targetRadius = radii[bestIdx];
                    lastCenter = targetCenter;
                    lostFrameCount = 0;
                    found = true;
                }
            }

            // 情况B：没有历史目标，或者丢失超过容忍帧数，允许重新选择
            if (!found && (currentTargetId == -1 || lostFrameCount > MAX_LOST_FRAMES)) {
                // 选择面积最大的一个作为新的目标（通常扇叶面积最大）
                targetCenter = candidates[0];
                targetRadius = radii[0];
                lastCenter = targetCenter;
                currentTargetId = 1; // 分配 ID 为 1
                lostFrameCount = 0;
                found = true;
                std::cout << "重新锁定目标，分配 ID: " << currentTargetId << std::endl;
            }
        }

          // 5. 绘制部分
        if (found) {
            // 画目标圆轮廓（绿色）
            cv::circle(frame, targetCenter, (int)targetRadius, cv::Scalar(0, 255, 0), 2);
            // 画目标中心（红色实心点）
            cv::circle(frame, targetCenter, 3, cv::Scalar(0, 0, 255), -1);
            // 画连线（黄色）
            cv::line(frame, centerR, targetCenter, cv::Scalar(0, 255, 255), 2);

            // 显示 ID 和 DETECTED 状态
            std::string idText = "Target ID: " + std::to_string(currentTargetId);
            cv::putText(frame, idText, cv::Point(targetCenter.x - 30, targetCenter.y - 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            cv::putText(frame, "State: DETECTED", cv::Point(30, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        } else {
            // 丢失逻辑
            lostFrameCount++;
            if (lostFrameCount > MAX_LOST_FRAMES) {
                // 丢失太久，重置ID
                currentTargetId = -1;
                lastCenter = cv::Point2f(-1, -1);
            }
            cv::putText(frame, "State: LOST", cv::Point(30, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        }

        // 画 R 标中心（白色）
        cv::circle(frame, centerR, 5, cv::Scalar(255, 255, 255), -1);

        // 【死命令】：无论有没有找到目标，必须每帧都写入视频
        writer.write(frame);
        cv::imshow("Tracking", frame);

        // 按 ESC 退出
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    return 0;
}