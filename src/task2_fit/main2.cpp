#include "common/common.hpp"
//定义任务二函数
//==================== 模块1：目标检测 ====================

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
        return cv::Point2f(cx, cy);//加权算质心
    }
    struct FitResult {
        double A, b, Omega, phi;
        double rmse;
    };
    FitResult fitAngularVelocity(const std::vector<double>& times,const std::vector<double>& omegas) {
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



        void drawCurve(const std::string& filename,
               const std::vector<double>& xs,
               const std::vector<double>& ys1,
               const std::vector<double>& ys2,
               const std::string& title) {
    int W = 900, H = 600;
    cv::Mat graph(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

    // 坐标轴
    cv::line(graph, cv::Point(60, H-60), cv::Point(W-30, H-60), cv::Scalar(0,0,0), 2);
    cv::line(graph, cv::Point(60, H-60), cv::Point(60, 30), cv::Scalar(0,0,0), 2);

    if (xs.empty()) return;

    // 找 y 的最大最小值，用于归一化
    double ymin = 1e9, ymax = -1e9;
    for (double v : ys1) { ymin = std::min(ymin, v); ymax = std::max(ymax, v); }
    for (double v : ys2) { ymin = std::min(ymin, v); ymax = std::max(ymax, v); }
    if (ymax - ymin < 1e-6) ymax = ymin + 1.0;

    double xmin = xs.front(), xmax = xs.back();
    if (xmax - xmin < 1e-6) xmax = xmin + 1.0;

    auto toPx = [&](double x, double y) {
        int px = 60 + (int)((x - xmin) / (xmax - xmin) * (W - 90));
        int py = H - 60 - (int)((y - ymin) / (ymax - ymin) * (H - 90));
        return cv::Point(px, py);
    };

    // 画曲线1（红点）
    for (size_t i = 0; i < xs.size(); i++) {
        cv::circle(graph, toPx(xs[i], ys1[i]), 2, cv::Scalar(0, 0, 255), -1);
    }
    // 画曲线2（绿线）
    for (size_t i = 1; i < xs.size(); i++) {
        cv::line(graph, toPx(xs[i-1], ys2[i-1]), toPx(xs[i], ys2[i]),
                 cv::Scalar(0, 255, 0), 2);
    }

    cv::putText(graph, title, cv::Point(200, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
    cv::imwrite(filename, graph);
}















     // ==================== 主流程 ====================
int main(){
    const double cx = 480, cy = 360;
    cv::VideoCapture cap("resources/task_2.mp4", cv::CAP_FFMPEG); // 强制 FFMPEG
    if (!cap.isOpened()) { std::cerr << "无法打开视频" << std::endl; return -1; }

    int fps = (int)cap.get(cv::CAP_PROP_FPS);
    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter writer("result/task2_fit/tracking_overlay.mp4",
                           cv::VideoWriter::fourcc('m','p','4','v'),
                           fps, cv::Size(w, h));

    AngleTracker tracker(cx, cy);
    std::vector<double> times, angles;

    // ========== 2. 逐帧循环 ==========
    cv::Mat frame;
    int frameIdx = 0;
    while (cap.read(frame)) {
        double t = frameIdx / (double)fps;
        cv::Point2f target = detectCyanTarget(frame);

        double angle = 0;
        if (target.x >= 0) {
            angle = tracker.update(target.x, target.y);
            times.push_back(t);
            angles.push_back(angle);
        }

        drawOverlay(frame, cv::Point2f((float)cx, (float)cy), target, angle);
        writer.write(frame);   // 【关键】写入视频
        frameIdx++;
    }
    cap.release();
    writer.release();          // 【关键】释放视频，否则文件损坏

    std::cout << "视频处理完成，共 " << frameIdx << " 帧" << std::endl;

    // ========== 3. 求角速度 ==========
    std::vector<double> omegas, omegaTimes;
    for (size_t i = 1; i < times.size(); i++) {
        double dt = times[i] - times[i-1];
        if (dt > 0) {
            omegas.push_back((angles[i] - angles[i-1]) / dt);
            omegaTimes.push_back(times[i]);
        }
    }

    // ========== 4. 拟合 ==========
    FitResult fit = fitAngularVelocity(omegaTimes, omegas);
    std::cout << "A=" << fit.A << " b=" << fit.b
              << " Omega=" << fit.Omega << " phi=" << fit.phi
              << " RMSE=" << fit.rmse << std::endl;

    // ========== 5. 生成 3 张图 ==========
    // 拟合曲线
    std::vector<double> fitCurve(omegaTimes.size());
    for (size_t i = 0; i < omegaTimes.size(); i++) {
        fitCurve[i] = fit.b + fit.A * std::sin(fit.Omega * omegaTimes[i] + fit.phi);
    }
    // 残差
    std::vector<double> residuals(omegaTimes.size());
    for (size_t i = 0; i < omegaTimes.size(); i++) {
        residuals[i] = omegas[i] - fitCurve[i];
    }

    // 画图函数（自定义，见下文）
    drawCurve("result/task2_fit/fit_comparison.png",
              omegaTimes, omegas, fitCurve, "Observation vs Fit");
    drawCurve("result/task2_fit/angular_velocity.png",
              omegaTimes, fitCurve, fitCurve, "Angular Velocity");
    drawCurve("result/task2_fit/residuals.png",
              omegaTimes, residuals, residuals, "Residuals");

    // ========== 6. 保存参数到 md ==========
    std::ofstream f("result/task2_fit_result.md");
    f << "# Task2 Fit Result\n\n";
    f << "## 模型\nω(t) = b + A·sin(Ωt + φ)\n\n";
    f << "## 参数\n";
    f << "- A = " << fit.A << " rad/s\n";
    f << "- b = " << fit.b << " rad/s\n";
    f << "- Ω = " << fit.Omega << " rad/s\n";
    f << "- φ = " << fit.phi << " rad\n\n";
    f << "## 误差\n- RMSE = " << fit.rmse << " rad/s\n";
    f << "- 有效样本数: " << omegas.size() << "\n";
    f << "- 帧范围: 0 ~ " << frameIdx - 1 << "\n";
    f.close();

    return 0;
}
            