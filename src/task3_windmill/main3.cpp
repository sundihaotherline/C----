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
    cv::Mat kernelBig = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::Mat kernelSmall = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    // ========== 动态 R 标中心（不再是固定的 480,360） ==========
    cv::Point2f centerR(-1, -1);   // -1 表示尚未找到
    bool hasR = false;             // 标志位：是否已成功定位过 R 标
    
    // 已知的扇叶距离 R 标的物理半径
    const float TARGET_RADIUS = 160.0f;
    const float RADIUS_TOLERANCE = 50.0f; // 允许误差（140~300像素内都算候选）

    // ========== 动态 R 标状态 ==========
    
    cv::Point2f lastValidCenterR(-1, -1);  // 上一帧有效的R标中心（用于容错）
   
    // ========== 目标锁定状态机 ==========
    int currentTargetId = -1;
    int nextTargetId = 1;
    cv::Point2f lastCenter(-1, -1);
    cv::Point2f lastValidTarget(-1, -1);   // 上一帧有效的目标中心（用于丢失预测）
    int lostFrameCount = 0;
    const int MAX_LOST_FRAMES = 30;        // 最多允许丢失30帧
    const double MATCH_DIST_THRESH = 80.0; // 匹配距离阈值

     // 物理约束（任务书已知半径220）
const float EXPECTED_RADIUS = 180.0f;   // 预期半径

while (cap.read(frame)) {
    // ...   



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
            if (area < 100 || area > 3000) continue; 
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

                // ================= 阶段B：同心圆检测与距离过滤 =================
        std::vector<cv::Point2f> circleCandidates;
        std::vector<float> circleRadii;

        // 获取参照中心
        cv::Point2f referenceCenter = hasR ? centerR : lastValidCenterR;

        for (size_t i = 0; i < contoursCircle.size(); i++) {
            double area = cv::contourArea(contoursCircle[i]);
            if (area < 1500 || area > 20000) continue; 

            std::vector<cv::Point> hull;
            cv::convexHull(contoursCircle[i], hull);
            double solidity = area / cv::contourArea(hull);
            if (solidity < 0.85) continue; 

            cv::Point2f center;
            float radius;
            cv::minEnclosingCircle(contoursCircle[i], center, radius);

            // 【核心修正1】：如果 R 标还没出现（无效），直接丢弃这个候选者，坚决不锁定！
            if (referenceCenter.x < 0) {
                continue; 
            }

            // 距离过滤（只有 R 标有效时才执行）
            float distToR = cv::norm(center - referenceCenter);
            if (distToR < EXPECTED_RADIUS - RADIUS_TOLERANCE || 
                distToR > EXPECTED_RADIUS + RADIUS_TOLERANCE) {
                continue; 
            }

            circleCandidates.push_back(center);
            circleRadii.push_back(radius);
        }

        // ================= 状态机：锁定同心圆 =================
        bool found = false;
        cv::Point2f targetCenter;
        float targetRadius = 0;

        // 【核心修正2】：如果 R 标无效，直接跳过整个目标锁定逻辑，进入等待状态
        if (referenceCenter.x < 0) {
            currentTargetId = -1;       // 不分配 ID
            lastCenter = cv::Point2f(-1, -1); 
            lastValidTarget = cv::Point2f(-1, -1);
            lostFrameCount = 0;         // 不增加丢失计数，避免触发全图搜索
            found = false;
        } 
        else if (!circleCandidates.empty()) {
            // 情况A：已有目标，尝试匹配
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
                    lastValidTarget = targetCenter;
                    lostFrameCount = 0;
                    found = true;
                }
            }
            
            // 情况B：初次选择，或丢失重选
            if (!found && (currentTargetId == -1 || lostFrameCount > MAX_LOST_FRAMES)) {
                int bestIdx = 0;
                float minDiff = 1e9;
                for (size_t i = 0; i < circleCandidates.size(); i++) {
                    float d = std::abs(cv::norm(circleCandidates[i] - referenceCenter) - EXPECTED_RADIUS);
                    if (d < minDiff) { minDiff = d; bestIdx = (int)i; }
                }
                
                targetCenter = circleCandidates[bestIdx];
                targetRadius = circleRadii[bestIdx];
                lastCenter = targetCenter;
                lastValidTarget = targetCenter;
                currentTargetId = nextTargetId++;
                lostFrameCount = 0;
                found = true;
                std::cout << "锁定目标 ID: " << currentTargetId << std::endl;
            }
        }

        // ================= 绘制 =================
        // 画 R 标中心
        if (hasR || lastValidCenterR.x > 0) {
            cv::Point2f refCenter = hasR ? centerR : lastValidCenterR;
            cv::circle(frame, refCenter, 5, cv::Scalar(255, 255, 255), -1);
            cv::putText(frame, "R", cv::Point(refCenter.x + 10, refCenter.y - 10), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        }

        // 画同心圆目标
        if (found) {
            cv::circle(frame, targetCenter, (int)targetRadius, cv::Scalar(0, 255, 0), 2);
            cv::circle(frame, targetCenter, 3, cv::Scalar(0, 0, 255), -1);
            if (hasR || lastValidCenterR.x > 0) {
                cv::Point2f refCenter = hasR ? centerR : lastValidCenterR;
                cv::line(frame, refCenter, targetCenter, cv::Scalar(0, 255, 255), 2);
            }
            cv::putText(frame, "Target ID: " + std::to_string(currentTargetId),
                        cv::Point(targetCenter.x - 30, targetCenter.y - 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            cv::putText(frame, "State: DETECTED", cv::Point(30, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        } else {
            // 【核心修正3】：如果是没有 R 标导致的等待，不要增加丢失帧数，显示等待提示
            if (referenceCenter.x < 0) {
                cv::putText(frame, "State: WAITING FOR R", cv::Point(30, 50),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 0), 2);
                // 也可以画一个提示圆环
                cv::circle(frame, cv::Point(w/2, h/2), 50, cv::Scalar(255, 255, 0), 2);
            } else {
                lostFrameCount++;
                // 丢失预测（画虚框）
                if (lostFrameCount <= MAX_LOST_FRAMES && lastValidTarget.x > 0) {
                    cv::circle(frame, lastValidTarget, 30, cv::Scalar(0, 0, 255), 2);
                    cv::putText(frame, "State: PREDICTING (Lost)", cv::Point(30, 50),
                                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 165, 255), 2);
                } else {
                    currentTargetId = -1; // 真正放弃
                    lastCenter = cv::Point2f(-1, -1);
                    lastValidTarget = cv::Point2f(-1, -1);
                    cv::putText(frame, "State: LOST", cv::Point(30, 50),
                                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                }
            }
        }

        // 写入视频
        writer.write(frame);
        cv::imshow("Tracking", frame);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    return 0;
}}