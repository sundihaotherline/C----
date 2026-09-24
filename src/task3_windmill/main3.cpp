#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    
    cv::VideoCapture cap("resources/task_3.mp4");
    if (!cap.isOpened()) { std::cerr << "❌ 无法打开视频" << std::endl; return -1; }
    double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0 || fps > 120) fps = 60.0; // 防呆设计，防止读不到帧率
        int delay = cv::max(1, (int)(1000.0 / fps)); // 对于60fps，这里算出来是16毫秒
    
    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter writer("result/task3_windmill/task_3/recognition_overlay.mp4",
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(w, h));

    cv::Mat frame, hsv, maskLow, maskHigh, mask;
    cv::Mat kernelBig = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::Mat kernelSmall = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    // ========== 动态 R 标中心（不再是固定的 480,360） ==========
    cv::Point2f centerR(-1, -1);   // -1 表示尚未找到
    bool hasR = false;             // 标志位：是否已成功定位过 R 标
    
    // 已知的扇叶距离 R 标的物理半径（任务书提示为220像素）
    const float TARGET_RADIUS = 220.0f;
    const float RADIUS_TOLERANCE = 80.0f; // 允许误差（140~300像素内都算候选）

    // ========== 状态机变量 ==========
    int currentTargetId = -1;
    int nextTargetId = 1;
    cv::Point2f lastCenter(-1, -1);
    int lostFrameCount = 0;
    const int MAX_LOST_FRAMES = 30;
    const double MATCH_DIST_THRESH = 80.0;

    while (cap.read(frame)) {
        // 1. 颜色提取（收紧H范围，排除黄光干扰）
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(0, 120, 80), cv::Scalar(12, 255, 255), maskLow);
        cv::inRange(hsv, cv::Scalar(168, 120, 80), cv::Scalar(179, 255, 255), maskHigh);
        cv::bitwise_or(maskLow, maskHigh, mask);

        // 2. 形态学双轨处理
        cv::Mat maskCircle = mask.clone();
        cv::morphologyEx(maskCircle, maskCircle, cv::MORPH_CLOSE, kernelBig); // 用于找同心圆
        cv::Mat maskR = mask.clone();
        cv::morphologyEx(maskR, maskR, cv::MORPH_OPEN, kernelSmall); // 用于找 R 标

        // 3. 提取轮廓
        std::vector<std::vector<cv::Point>> contoursCircle, contoursR;
        cv::findContours(maskCircle, contoursCircle, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::findContours(maskR, contoursR, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // ================= 阶段A：先动态定位 R 标 =================
        std::vector<cv::Point2f> rCandidates;
        for (size_t i = 0; i < contoursR.size(); i++) {
            double area = cv::contourArea(contoursR[i]);
            if (area < 80 || area > 3000) continue; 
            double perimeter = cv::arcLength(contoursR[i], true);
            double circularity = 4 * M_PI * area / (perimeter * perimeter);
            if (circularity > 0.8) continue; // 排除圆环
            cv::Rect box = cv::boundingRect(contoursR[i]);
            double aspectRatio = (double)box.width / box.height;
            if (aspectRatio < 0.2 || aspectRatio > 5.0) continue; 

            cv::Moments m = cv::moments(contoursR[i]);
            if (m.m00 > 0) {
                cv::Point2f center((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
                rCandidates.push_back(center);
            }
        }

        // 从候选者中更新 centerR（挑选离上一帧最近，或离画面中心最近的）
        if (!rCandidates.empty()) {
            double minDist = 1e9;
            cv::Point2f bestR;
            for (auto& rc : rCandidates) {
                double d;
                if (hasR) {
                    d = cv::norm(rc - centerR); // 追踪上一帧的 R 标
                } else {
                    d = cv::norm(rc - cv::Point2f(w / 2.0, h / 2.0)); // 首次寻找靠近图像中心的
                }
                if (d < minDist) { minDist = d; bestR = rc; }
            }
            centerR = bestR;
            hasR = true;
        }

        // ================= 阶段B：基于 centerR 过滤同心圆 =================
        std::vector<cv::Point2f> circleCandidates;
        std::vector<float> circleRadii;
        for (size_t i = 0; i < contoursCircle.size(); i++) {
            double area = cv::contourArea(contoursCircle[i]);
            if (area < 300 || area > 15000) continue; 

            std::vector<cv::Point> hull;
            cv::convexHull(contoursCircle[i], hull);
            double solidity = area / cv::contourArea(hull);
            if (solidity < 0.85) continue; 

            cv::Point2f center;
            float radius;
            cv::minEnclosingCircle(contoursCircle[i], center, radius);

            // 【核心改进】：动态距离过滤
            if (hasR) {
                float distToR = cv::norm(center - centerR);
             // 距离 R 标太远或太近，都不可能是能量机关的扇叶
                if (distToR < TARGET_RADIUS - RADIUS_TOLERANCE || 
                    distToR > TARGET_RADIUS + RADIUS_TOLERANCE) {
                    continue;
                }
            }

            circleCandidates.push_back(center);
            circleRadii.push_back(radius);
        }

        // ================= 状态机：锁定同心圆 =================
        bool found = false;
        cv::Point2f targetCenter;
        float targetRadius = 0;

        if (!circleCandidates.empty()) {
            if (currentTargetId != -1 && lastCenter.x >= 0) {
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
                // 初次选择：选离预期距离(220)最近的那个
                int bestIdx = 0;
                float minDiff = 1e9;
                for (size_t i = 0; i < circleCandidates.size(); i++) {
                    float d = std::abs(cv::norm(circleCandidates[i] - centerR) - TARGET_RADIUS);
                    if (d < minDiff) { minDiff = d; bestIdx = (int)i; }
                }
                targetCenter = circleCandidates[bestIdx];
                targetRadius = circleRadii[bestIdx];
                lastCenter = targetCenter;
                currentTargetId = nextTargetId++;
                lostFrameCount = 0;
                found = true;
                std::cout << "锁定目标 ID: " << currentTargetId << std::endl;
            }
        }

        // ================= 绘制 =================
        // 画 R 标中心
        if (hasR) {
            cv::circle(frame, centerR, 5, cv::Scalar(255, 255, 255), -1);
            cv::putText(frame, "R", cv::Point(centerR.x + 10, centerR.y - 10), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        }

        // 画同心圆目标
        if (found) {
            cv::circle(frame, targetCenter, (int)targetRadius, cv::Scalar(0, 255, 0), 2);
            cv::circle(frame, targetCenter, 3, cv::Scalar(0, 0, 255), -1);
            if (hasR) {
                cv::line(frame, centerR, targetCenter, cv::Scalar(0, 255, 255), 2);
            }
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
        if (cv::waitKey(delay) == 27) break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    return 0;
}