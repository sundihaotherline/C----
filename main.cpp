#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <vector>
#include <cmath>


int main() {
    cv::Mat gray,binary,adaptive,smooth,edges,maskHigh,maskLow,mask;
    cv::Mat equailized,dilated,eroded,opened,closed,rotated;
    
    //读入图片
    cv::Mat img = cv::imread("resources/test_image.jpg");
    if (img.empty()) { 
    std::cout << "找不到图片" << std::endl; 
    return -1; 
}   //cv::imshow("Original", img);
    cv::imwrite("result/task1_images/original.png", img);//原图
    
    
    cv::Mat meanImg,gaussianImg,medianImg;
    cv::cvtColor(img,gray,cv::COLOR_BGR2GRAY);
    cv::imwrite("result/task1_images/gray.png",gray);;//灰度化
    
    
    cv::Mat hsv, channels[3];
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::split(hsv, channels);
    //cv::imshow("H", channels[0]);
    // cv::imshow("S", channels[1]);
    // cv::imshow("V", channels[2]);
    cv::imwrite("result/task1_images/hsv_h.png", channels[0]);
    cv::imwrite("result/task1_images/hsv_s.png", channels[1]);
    cv::imwrite("result/task1_images/hsv_v.png", channels[2]);//单通道图
    //cv::cvtColor(img,hsv,cv::COLOR_BGR2HSV);
    
    
    
    //开始滤波
    blur(gray,meanImg,cv::Size(5,5));
    GaussianBlur(gray,gaussianImg,cv::Size(5,5),1.5);
    cv::medianBlur(gray,medianImg,5);
    //cv::imshow("meanImg",meanImg);
    //cv::imshow("Gaussian",gaussianImg);
    //cv::imshow("Median",medianImg);
    cv::imwrite("result/task1_images/mean.png", meanImg);
    cv::imwrite("result/task1_images/gaussian.png", gaussianImg);
    cv::imwrite("result/task1_images/median.png",medianImg); 
    //滤波结束
    //cv::adaptiveThreshold(gray,adaptive,255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY,11,2);
   
    //imshow("Adaptive",adaptive);
    
    
    //开始生成掩膜
    inRange(hsv,cv::Scalar(0,80,100),cv::Scalar(17,255,255),maskLow);
    inRange(hsv,cv::Scalar(160,80,100),cv::Scalar(179,255,255),maskHigh);
    bitwise_or(maskLow,maskHigh,mask);
    // imshow("Redmask",mask);
    cv::imwrite("result/task1_images/redmask.png", mask);

    //掩膜结束



    // Canny(gaussianImg,edges,100,200);
    // imshow("Canny",edges);
    //equalizeHist(gray,equailized);
    //cv::imshow("equailized",equailized);
    //cv::waitKey(0);
    
    
    cv::Mat kernel =cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    //生成形态学操作的核

    dilate(mask, dilated, kernel);
    erode(mask, eroded, kernel);
    morphologyEx(mask, opened,cv::MORPH_OPEN, kernel);
    morphologyEx(mask, closed,cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, opened, cv::MORPH_OPEN, kernel);
    cv::imwrite("result/task1_images/erode.png", eroded);
    cv::imwrite("result/task1_images/dilate.png", dilated);
    cv::imwrite("result/task1_images/open.png", opened);
    cv::imwrite("result/task1_images/close.png", closed);
    
    
    cv::Mat result = img.clone();
     std::vector<std::vector<cv::Point>> contours;
     cv::findContours(closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area < 500.0) continue;
        cv::Rect box = cv::boundingRect(contours[i]);
        cv::rectangle(result, box, cv::Scalar(0, 0, 255), 2);
        cv::drawContours(result, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);  
        std::string areaText = "Area: " + std::to_string((int)area);//图形上方标注
        cv::putText(result, areaText, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);//文字位置
    }
    cv::circle(result, cv::Point(img.cols / 2, img.rows / 2), 100, cv::Scalar(255, 0, 0), 3);//绘画一个圆
    cv::imwrite("result/task1_images/contours_boxes.png", result);
    // imshow("Open", opened);
    // imshow("Close", closed);
    
    cv::Rect cropRect(0, 0, img.cols / 2, img.rows / 2);  // x=0, y=0, 宽=原宽/2, 高=原高/2
    cv::Mat cropped = img(cropRect);                       // 用矩形ROI截取
    cv::imwrite("result/task1_images/crop_top_left.png", cropped);//截取左上角四分之一
    
    cv::Point2f center(img.cols / 2.0, img.rows / 2.0);
    double angle = 35.0;
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(img, rotated, rotMat, img.size());
    imwrite("result/task1_images/rotated.png", rotated);//旋转图像  
    
    
    
    //task 2------------------------------------------------------------------------------------//
    
    
    
    
    
    
    
    
    // ==================== 模块1：颜色检测 ====================
        cv::Point2f detectCyanTarget(const cv::Mat& frame) {
        cv::Mat hsv, mask;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        // 青色 H 范围约 80~100
        cv::inRange(hsv, cv::Scalar(80, 100, 100), cv::Scalar(100, 255, 255), mask);
        // 开运算去噪
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (contours.empty()) return cv::Point2f(-1, -1);

        // 找最大轮廓
        double maxArea = 0;
        int maxIdx = -1;
        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > maxArea) { maxArea = area; maxIdx = (int)i; }
        }
        if (maxIdx < 0 || maxArea < 100) return cv::Point2f(-1, -1);

        // 计算质心
        cv::Moments m = cv::moments(contours[maxIdx]);
        float cx = (float)(m.m10 / m.m00);
        float cy = (float)(m.m01 / m.m00);
        return cv::Point2f(cx, cy);
    }

    // ==================== 模块2：角度计算 + 解包裹 ====================
    class AngleTracker {
    public:
        AngleTracker(double cx, double cy) : cx_(cx), cy_(cy), last_angle_(0), accumulated_(0) {}
        // 输入目标坐标，返回累加角度（弧度）
        double update(double x, double y) {
            double raw = std::atan2(cy_ - y, x - cx_); // 图像y轴向下，需取反
            double delta = raw - last_angle_;
            // 解包裹
            if (delta > M_PI) delta -= 2 * M_PI;
            if (delta < -M_PI) delta += 2 * M_PI;
            accumulated_ += delta;
            last_angle_ = raw;
            return accumulated_;
        }
    private:
        double cx_, cy_;
        double last_angle_;
        double accumulated_;
    };

    // ==================== 模块3：参数拟合（Eigen + Ω扫描） ====================
    struct FitResult {
        double A, b, Omega, phi;
        double rmse;
    };

    FitResult fitAngularVelocity(const std::vector<double>& times,
                                const std::vector<double>& omegas) {
        FitResult best{0, 0, 0, 0, 1e30};

        // 估算时间步长
        double dt = times[1] - times[0];
        int N = (int)times.size();

        // 扫描 Omega 从 0.1 到 3.0，步长 0.01
        for (double Omega = 0.1; Omega <= 3.0; Omega += 0.01) {
            // 构造线性最小二乘 A*x = y
            // 模型: omega = b + C1*sin(Omega*t) + C2*cos(Omega*t)
            Eigen::MatrixXd A(N, 3);
            Eigen::VectorXd y(N);
            for (int i = 0; i < N; i++) {
                double t = times[i];
                A(i, 0) = 1.0;
                A(i, 1) = std::sin(Omega * t);
                A(i, 2) = std::cos(Omega * t);
                y(i) = omegas[i];
            }
            Eigen::Vector3d x = A.colPivHouseholderQr().solve(y);
            Eigen::VectorXd residual = A * x - y;
            double sse = residual.squaredNorm();

            if (sse < best.rmse * best.rmse * N) {
                best.b     = x(0);
                best.A     = std::sqrt(x(1) * x(1) + x(2) * x(2));
                best.phi   = std::atan2(x(2), x(1)); // 注意顺序
                best.Omega = Omega;
                best.rmse  = std::sqrt(sse / N);
            }
        }
        return best;
    }

        // ==================== 模块4：可视化 ====================
        void drawOverlay(cv::Mat& frame, const cv::Point2f& center,
                        const cv::Point2f& target, double angle) {
            cv::circle(frame, cv::Point((int)center.x, (int)center.y), 5, cv::Scalar(255, 255, 255), -1);
            if (target.x >= 0) {
                cv::circle(frame, cv::Point((int)target.x, (int)target.y), 8, cv::Scalar(255, 255, 0), 2);
                cv::line(frame, center, target, cv::Scalar(0, 255, 255), 2);
            }
            char text[128];
            snprintf(text, sizeof(text), "Angle: %.2f rad", angle);
            cv::putText(frame, text, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                        cv::Scalar(0, 255, 0), 2);
        }

        // ==================== 主流程 ====================
        int main() {
            const double cx = 480, cy = 360; // 已知旋转中心
            cv::VideoCapture cap("resources/task_2.mp4");
            if (!cap.isOpened()) { std::cerr << "无法打开视频" << std::endl; return -1; }

            int fps = (int)cap.get(cv::CAP_PROP_FPS);
            int w   = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
            int h   = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            cv::VideoWriter writer("result/task2_fit/tracking_overlay.mp4",
                                cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(w, h));

            AngleTracker tracker(cx, cy);
            std::vector<double> times, angles;

            cv::Mat frame;
            int frameI
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        return 0;
    }
