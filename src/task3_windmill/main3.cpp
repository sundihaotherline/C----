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
            // ================= 新增：计算目标相对 R 的角度 =================
                    if (hasR || lastValidCenterR.x > 0) {
                        cv::Point2f refCenter = hasR ? centerR : lastValidCenterR;
                        float dx = targetCenter.x - refCenter.x;
                        float dy = targetCenter.y - refCenter.y;
                        // 图像 y 向下，用 -dy 转成“向上为正”，再从正右方起算、逆时针为正
                        float angle_rad = std::atan2(-dy, dx);
                        float angle_deg = angle_rad * 180.0f / (float)M_PI;

                        char angleText[64];
                        snprintf(angleText, sizeof(angleText), "Angle: %.1f deg", angle_deg);
                        cv::putText(frame, angleText,
                                    cv::Point(30, 90),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                                    cv::Scalar(255, 255, 255), 2);
                    }
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
    
    
    
    
    
    //任务三第二部分
    
    
    
    


    
  
    // ===================================================================
    // ================== 第二部分：处理 task_4.mp4 (多目标) ==============
    // ===================================================================
       // ===================================================================
    // ============== 第二部分：task_4.mp4 (简化版：找R + 找目标) ==========
    // ===================================================================
    {
        cv::VideoCapture cap4("resources/task_4.mp4");
        if (!cap4.isOpened()) {
            std::cerr << "❌ 无法打开 task_4.mp4" << std::endl;
        } else {
            int fps4 = (int)cap4.get(cv::CAP_PROP_FPS);
            int w4 = (int)cap4.get(cv::CAP_PROP_FRAME_WIDTH);
            int h4 = (int)cap4.get(cv::CAP_PROP_FRAME_HEIGHT);

            cv::VideoWriter writer4("result/task3_windmill/task_4/recognition_overlay.mp4",
                                    cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                    fps4, cv::Size(w4, h4));

            cv::Mat frame4, hsv4, maskLow4, maskHigh4, mask4;
            cv::Mat kernelBig4 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
            cv::Mat kernelDilate4 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

            // R 标状态
            cv::Point2f centerR4(-1, -1);
            cv::Point2f lastValidCenterR4(-1, -1);
            bool hasR4 = false;
            int rStableCounter4 = 0;
            cv::Point2f lastRGuess4(-1, -1);

            // 目标状态机
            int currentTargetId4 = -1;
            int nextTargetId4 = 1;
            cv::Point2f lastCenter4(-1, -1);
            bool hasTarget4 = false;

            const float EXPECTED_RADIUS4 = 180.0f;
            const float RADIUS_TOLERANCE4 = 50.0f;

            while (cap4.read(frame4)) {
                // 1. 红色掩膜
                cv::cvtColor(frame4, hsv4, cv::COLOR_BGR2HSV);
                cv::inRange(hsv4, cv::Scalar(0, 120, 80), cv::Scalar(15, 255, 255), maskLow4);
                cv::inRange(hsv4, cv::Scalar(168, 120, 80), cv::Scalar(179, 255, 255), maskHigh4);
                cv::bitwise_or(maskLow4, maskHigh4, mask4);

                // 2. 形态学：大核闭运算用于同心圆；膨胀用于 R 标
                cv::Mat maskCircle4 = mask4.clone();
                cv::morphologyEx(maskCircle4, maskCircle4, cv::MORPH_CLOSE, kernelBig4);

                cv::Mat maskR4;
                cv::dilate(mask4, maskR4, kernelDilate4);

                // ================= 阶段A：定位 R 标 =================
                std::vector<std::vector<cv::Point>> contoursR4;
                cv::findContours(maskR4, contoursR4, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                std::vector<cv::Point2f> rCandidates4;
                for (size_t i = 0; i < contoursR4.size(); i++) {
                    double area = cv::contourArea(contoursR4[i]);
                    if (area < 80 || area > 2000) continue;

                    double perimeter = cv::arcLength(contoursR4[i], true);
                    if (perimeter == 0) continue;
                    double circularity = 4 * M_PI * area / (perimeter * perimeter);
                    if (circularity > 0.75) continue;

                    cv::Rect box = cv::boundingRect(contoursR4[i]);
                    double aspectRatio = (double)box.width / box.height;
                    if (aspectRatio < 0.2 || aspectRatio > 5.0) continue;

                    cv::Moments m = cv::moments(contoursR4[i]);
                    if (m.m00 > 0) {
                        cv::Point2f candidate((float)(m.m10 / m.m00), (float)(m.m01 / m.m00));
                        if (cv::norm(candidate - cv::Point2f(w4 / 2.0, h4 / 2.0)) > 150.0) continue;
                        rCandidates4.push_back(candidate);
                    }
                }

                if (!rCandidates4.empty()) {
                    double minDist = 1e9;
                    cv::Point2f bestR4;
                    for (auto& rc : rCandidates4) {
                        double d = hasR4 ? cv::norm(rc - centerR4)
                                         : cv::norm(rc - cv::Point2f(w4 / 2.0, h4 / 2.0));
                        if (d < minDist) { minDist = d; bestR4 = rc; }
                    }
                    if (lastRGuess4.x < 0 || cv::norm(bestR4 - lastRGuess4) < 100.0) {
                        rStableCounter4++;
                    } else {
                        rStableCounter4 = 1;
                    }
                    lastRGuess4 = bestR4;

                    if (rStableCounter4 >= 10 && !hasR4) {
                        centerR4 = bestR4;
                        lastValidCenterR4 = centerR4;
                        hasR4 = true;
                        std::cout << "✅ [task_4] R 标已稳定锁定！" << std::endl;
                    } else if (rStableCounter4 >= 10) {
                        centerR4 = bestR4;
                        lastValidCenterR4 = centerR4;
                    } else {
                        hasR4 = false;
                    }
                } else {
                    rStableCounter4 = 0;
                    hasR4 = false;
                }

                // ================= 阶段B：找同心圆目标候选 =================
                std::vector<std::vector<cv::Point>> contoursCircle4;
                cv::findContours(maskCircle4, contoursCircle4, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                std::vector<cv::Point2f> candidates4;
                std::vector<float> radii4;
                cv::Point2f refCenter4 = hasR4 ? centerR4 : lastValidCenterR4;

                for (size_t i = 0; i < contoursCircle4.size(); i++) {
                    double area = cv::contourArea(contoursCircle4[i]);
                    if (area < 1500 || area > 20000) continue;

                    std::vector<cv::Point> hull4;
                    cv::convexHull(contoursCircle4[i], hull4);
                    double solidity = area / cv::contourArea(hull4);
                    if (solidity < 0.85) continue;

                    cv::Point2f center;
                    float radius;
                    cv::minEnclosingCircle(contoursCircle4[i], center, radius);

                    if (refCenter4.x < 0) continue;
                    float distToR = cv::norm(center - refCenter4);
                    if (distToR < EXPECTED_RADIUS4 - RADIUS_TOLERANCE4 ||
                        distToR > EXPECTED_RADIUS4 + RADIUS_TOLERANCE4) continue;

                    candidates4.push_back(center);
                    radii4.push_back(radius);
                }

                // ================= 状态机：有目标就跟踪，没目标就找一个 =================
                cv::Point2f drawTarget4;
                float drawRadius4 = 0;
                bool drawFound4 = false;

                if (refCenter4.x < 0) {
                    // R 标还没出现，等待
                    hasTarget4 = false;
                    currentTargetId4 = -1;
                } else if (hasTarget4 && lastCenter4.x >= 0) {
                    // 已锁定目标：在候选池里找离上次最近的
                    double minDist = 1e9;
                    int bestIdx = -1;
                    for (size_t i = 0; i < candidates4.size(); i++) {
                        double d = cv::norm(candidates4[i] - lastCenter4);
                        if (d < minDist && d < 150.0) { minDist = d; bestIdx = (int)i; }
                    }
                    if (bestIdx != -1) {
                        // 找到，保持 ID
                        lastCenter4 = candidates4[bestIdx];
                        drawTarget4 = lastCenter4;
                        drawRadius4 = radii4[bestIdx];
                        drawFound4 = true;
                    } else {
                        // 丢失，清空目标，允许下一帧重选
                        std::cout << "⚠️ [task_4] 目标 " << currentTargetId4 << " 丢失" << std::endl;
                        hasTarget4 = false;
                        currentTargetId4 = -1;
                    }
                } else if (!candidates4.empty()) {
                    // 没有目标，找一个最近的作为新目标
                    int bestIdx = 0;
                    float minDiff = 1e9;
                    for (size_t i = 0; i < candidates4.size(); i++) {
                        float d = std::abs(cv::norm(candidates4[i] - refCenter4) - EXPECTED_RADIUS4);
                        if (d < minDiff) { minDiff = d; bestIdx = (int)i; }
                    }
                    lastCenter4 = candidates4[bestIdx];
                    drawTarget4 = lastCenter4;
                    drawRadius4 = radii4[bestIdx];
                    drawFound4 = true;
                    hasTarget4 = true;
                    currentTargetId4 = nextTargetId4++;
                    std::cout << "✅ [task_4] 锁定目标 ID: " << currentTargetId4 << std::endl;
                }

                // ================= 绘制 =================
                // 画 R 标
                if (hasR4 || lastValidCenterR4.x > 0) {
                    cv::Point2f refDraw = hasR4 ? centerR4 : lastValidCenterR4;
                    cv::circle(frame4, refDraw, 5, cv::Scalar(255, 255, 255), -1);
                    cv::putText(frame4, "R",
                                cv::Point(refDraw.x + 10, refDraw.y - 10),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                                cv::Scalar(255, 255, 255), 2);
                }

                // 画目标 + 连线 + ID
                if (drawFound4) {
                    cv::circle(frame4, drawTarget4, (int)drawRadius4, cv::Scalar(0, 255, 0), 2);
                    cv::circle(frame4, drawTarget4, 3, cv::Scalar(0, 0, 255), -1);

                    if (hasR4 || lastValidCenterR4.x > 0) {
                        cv::Point2f refDraw = hasR4 ? centerR4 : lastValidCenterR4;
                        cv::line(frame4, refDraw, drawTarget4, cv::Scalar(0, 255, 255), 2);
                    }

                    cv::putText(frame4, "Target ID: " + std::to_string(currentTargetId4),
                                cv::Point(drawTarget4.x - 30, drawTarget4.y - 40),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                                cv::Scalar(255, 255, 255), 2);

                    cv::putText(frame4, "State: DETECTED", cv::Point(30, 50),
                                cv::FONT_HERSHEY_SIMPLEX, 1.0,
                                cv::Scalar(0, 255, 0), 2);
                } else {
                    if (refCenter4.x < 0) {
                        cv::putText(frame4, "State: WAITING FOR R", cv::Point(30, 50),
                                    cv::FONT_HERSHEY_SIMPLEX, 1.0,
                                    cv::Scalar(255, 255, 0), 2);
                    } else {
                        cv::putText(frame4, "State: SEARCHING", cv::Point(30, 50),
                                    cv::FONT_HERSHEY_SIMPLEX, 1.0,
                                    cv::Scalar(0, 165, 255), 2);
                    }
                }

                writer4.write(frame4);
                cv::imshow("Tracking task_4", frame4);
                if (cv::waitKey(1) == 27) break;
            }

            cap4.release();
            writer4.release();
            cv::destroyAllWindows();
        }
    }}
    
    
    
    
    
    
    
    
    
    
    
 











