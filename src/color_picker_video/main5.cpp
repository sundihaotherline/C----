#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat current_frame, current_hsv;
bool is_paused = true;

// 鼠标回调函数
void onMouse(int event, int x, int y, int flags, void* param) {
    if (event == cv::EVENT_LBUTTONDOWN && is_paused) {
        cv::Vec3b bgr = current_frame.at<cv::Vec3b>(y, x);
        cv::Vec3b hsv = current_hsv.at<cv::Vec3b>(y, x);

        std::cout << "坐标: (" << x << ", " << y << ") ";
        std::cout << "BGR: (" << (int)bgr[0] << "," << (int)bgr[1] << "," << (int)bgr[2] << ") ";
        std::cout << "HSV: (" << (int)hsv[0] << "," << (int)hsv[1] << "," << (int)hsv[2] << ")" << std::endl;

        // 在画面点击处画一个小红点
        cv::circle(current_frame, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
        cv::imshow("Video Picker", current_frame);
    }
}

int main() {
    // 改成你要取色的视频路径（比如 task_3.mp4 或 task_4.mp4）
    cv::VideoCapture cap("resources/task_3.mp4");
    if (!cap.isOpened()) {
        std::cerr << "❌ 无法打开视频！" << std::endl;
        return -1;
    }

    cv::namedWindow("Video Picker");
    cv::setMouseCallback("Video Picker", onMouse, nullptr);

    std::cout << "操作说明：\n"
              << "1. 按【空格键】暂停/继续播放视频\n"
              << "2. 在暂停状态下，用鼠标左键点击画面取色\n"
              << "3. 按【ESC键】退出程序\n" << std::endl;

    while (true) {
        if (!is_paused) {
            cap.read(current_frame);
            if (current_frame.empty()) {
                std::cout << "视频播放完毕。" << std::endl;
                break;
            }
            cv::cvtColor(current_frame, current_hsv, cv::COLOR_BGR2HSV);
        }

        // 如果视频读到了，就显示
        if (!current_frame.empty()) {
            cv::imshow("Video Picker", current_frame);
        }

        // 等待按键，30ms 一帧（控制播放速度）
        int key = cv::waitKey(30);

        if (key == 32) { // 空格键的 ASCII 码是 32
            is_paused = !is_paused;
            if (is_paused) std::cout << "⏸️ 已暂停" << std::endl;
            else std::cout << "▶️ 继续播放" << std::endl;
        }
        else if (key == 27) { // ESC 键的 ASCII 码是 27
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}