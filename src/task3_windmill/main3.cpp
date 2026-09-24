#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    cv::VideoCapture cap("resources/task_3.mp4");
    if (!cap.isOpened()) { std::cerr << "❌ 无法打开视频" << std::endl; return -1; }

    int fps = (int)cap.get(cv::CAP_PROP_FPS);
    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter writer("result/task3_windmill/task_3/recognition_overlay.mp4",
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(w, h));

    cv::Mat frame, hsv, maskLow, maskHigh, mask;
    // 两个核：大核用于填补同心圆，小核用于保护 R 标细线
    cv::Mat kernelBig = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::Mat kernelSmall = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    // ========== 状态机变量 ==========
    int currentTargetId = -1;
    int nextTargetId = 1; // ID分配计数器
    cv::Point2f lastCenter(-1, -1);
    int lostFrameCount = 0;
    const int MAX_LOST_FRAMES = 30;
    const double MATCH_DIST_THRESH = 80.0;

    // R 标中心（初始设为画面中心附近，后续通过检测更新）
    cv::Point2f centerR(w / 2.0, h / 2.0); 
    bool rFoundInThisFrame = false; // R 标本帧是否找到

    while (cap.read(frame)) {
        // 1. 颜色提取（收紧H范围，提高S下限，拒绝黄色干扰）
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(0, 120, 80), cv::Scalar(12, 255, 255), maskLow);
        cv::inRange(hsv, cv::Scalar(168, 120, 80), cv::Scalar(179, 255, 255), maskHigh);
        cv::bitwise_or(maskLow, maskHigh, mask);

        // ========== 分离处理掩膜 ==========
        // A. 用于同心圆检测的大核掩膜（填成实心）
        cv::Mat maskCircle = mask.clone();
        cv::morphologyEx(maskCircle, maskCircle, cv::MORPH_CLOSE, kernelBig);
        
        // B. 用于 R 标检测的小核掩膜（保留细节）
        cv::Mat maskR = mask.clone();
        cv::morphologyEx(maskR, maskR, cv::MORPH_OPEN, kernelSmall); 

        // 2. 找轮廓
        std::vector<std::vector<cv::Point>> contoursCircle, contoursR;
        cv::findContours(maskCircle, contoursCircle, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::findContours(maskR, contoursR, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 3. 分离：同心圆候选池 vs R标候选池
        std::vector<cv::Point2f> circleCandidates;
        std::vector<float> circleRadii;
        std::vector<cv::Point2f> rCandidates;

        // 从 maskCircle 里找同心圆
        for (size_t i = 0; i < contoursCircle.size(); i++) {
            double area = cv::contourArea(contoursCircle[i]);
            if (area < 300 || area > 15000) continue; 
            
            std::vector<cv::Point> hull;
            cv::convexHull(contoursCircle[i], hull);
            double solidity = area / cv::contourArea(hull);

            if (solidity > 0.85) { // 实心度判断
                cv::Point2f center;
                float radius;
                cv::minEnclosingCircle(contoursCircle[i], center, radius);
                circleCandidates.push_back(center);
                circleRadii.push_back(radius);
            }
        }

        // 从 maskR 里找 R 标（不规则红色块）
        for (size_t i = 0; i < contoursR.size(); i++) {
            double area = cv::contourArea(contoursR[i]);
            
            // R 标面积范围：太小是噪点，太大是背景
            if (area < 80 || area > 3000) continue; 
            
            // 计算圆度，排除圆环
            double perimeter = cv::arcLength(contoursR[i], true);
            double circularity = 4 * M_PI * area / (perimeter * perimeter);
            if (circularity > 0.8) continue; // 太圆的不要
            
            // 计算长宽比（放宽，允许倾斜和变形）
            cv::Rect box = cv::boundingRect(contoursR[i]);
            double aspectRatio = (double)box.width / box.height;
            if (aspectRatio < 0.2 || aspectRatio > 5.0) continue; 

            // 记录 R 标候选者
            cv::Moments m = cv::moments(contoursR[i]);
            if (m.m00 > 0) {
                cv::Point2f center((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
                rCandidates.push_back(center);
                // 调试用：打印出所有可能被当成 R 标的候选者
                // std::cout << "R候选: 面积=" << area << " 长宽比=" << aspectRatio << " 坐标=(" << center.x << "," << center.y << ")" << std::endl;
            }
        }

        // ==================== 更新状态机：锁定同心圆 ====================
        bool found = false;
        cv::Point2f targetCenter;
        float targetRadius = 0;

        if (!circleCandidates.empty()) {
            if (currentTargetId != -1 && lastCenter.x >= 0) {
            // 优先匹配距离上一帧最近的目标
                double minDist = 1e9;
                int bestIdx = -1;
                for (size_t i = 0; i < circleCandidates.size(); i++) {
                    double dist = cv::norm(circleCandidates[i] - lastCenter);
                    if (dist < minDist && dist < MATCH_DIST_THRESH) {
                        minDist = dist; bestIdx = (int)i;
                    }
                }
                if (bestIdx != -1) {
                    targetCenter = circleCandidates[bestIdx];
                    targetRadius = circleRadii[bestIdx];
                    lastCenter = targetCenter;
                    lostFrameCount = 0;
                    found = true;
                }
            }
            if (!found && (currentTargetId == -1 || lostFrameCount > MAX_LOST_FRAMES)) {
                // 重新选择：选择面积最大的同心圆（更稳定）
                int bestIdx = 0;
                double maxArea = 0;
                for (size_t i = 0; i < contoursCircle.size(); i++) {
                    // 简单对应，实际应根据面积排序，这里简化为取第一个
                }
                targetCenter = circleCandidates[0];
                targetRadius = circleRadii[0];
                lastCenter = targetCenter;
                currentTargetId = nextTargetId; // 分配新 ID
                nextTargetId++;                 // 递增
                lostFrameCount = 0;
                found = true;
                std::cout << "重新锁定目标，分配 ID: " << currentTargetId << std::endl;
            }
        }

        // ==================== 绘制 ====================
        // 1. 更新 R 标中心
        rFoundInThisFrame = false;
        if (!rCandidates.empty() && found) {
            double minRDist = 1e9;
            cv::Point2f bestR;
            for (auto& rc : rCandidates) {
                double d = cv::norm(rc - targetCenter);
                if (d < minRDist && d < 250) { // 在同心圆附近250像素内找 R 标
                    minRDist = d;
                    bestR = rc;
                }
            }
            if (minRDist < 1e8) {
                centerR = bestR;
                rFoundInThisFrame = true;
            }
        }
        // 即使没找到，也画上一次的 R 标位置，保证画面稳定
        cv::circle(frame, centerR, 5, cv::Scalar(255, 255, 255), -1);
        cv::putText(frame, "R", cv::Point(centerR.x + 10, centerR.y - 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

        // 2. 画同心圆目标
        if (found) {
            cv::circle(frame, targetCenter, (int)targetRadius, cv::Scalar(0, 255, 0), 2);
            cv::circle(frame, targetCenter, 3, cv::Scalar(0, 0, 255), -1);
            cv::line(frame, centerR, targetCenter, cv::Scalar(0, 255, 255), 2);
            cv::putText(frame, "Target ID: " + std::to_string(currentTargetId),
                        cv::Point(targetCenter.x - 30, targetCenter.y - 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            cv::putText(frame, "State: DETECTED", cv::Point(30, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        } else {
            lostFrameCount++;
            if (lostFrameCount > MAX_LOST_FRAMES) {
                currentTargetId = -1;
                lastCenter = cv::Point2f(-1, -1);
            }
            cv::putText(frame, "State: LOST", cv::Point(30, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        }

        writer.write(frame);
        cv::imshow("Tracking", frame);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    return 0;
}       
        