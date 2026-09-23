#include <opencv2/opencv.hpp>
#include <iostream>

// 全局变量，用来保存鼠标点击处的 HSV 值
cv::Scalar clicked_hsv;
cv::Mat img, hsv;

// 鼠标回调函数（相当于中断服务程序）
void onMouse(int event, int x, int y, int flags, void* param) {
    // 只响应鼠标左键按下的事件
    if (event == cv::EVENT_LBUTTONDOWN) {
        // 边界检查，防止点击越界
        if (x < 0 || x >= img.cols || y < 0 || y >= img.rows) return;

        // 读取原图像素值 (BGR)
        cv::Vec3b bgr = img.at<cv::Vec3b>(y, x);
        // 读取 HSV 像素值
        cv::Vec3b hsv_val = hsv.at<cv::Vec3b>(y, x);

        // 打印到终端
        std::cout << "========== 鼠标点击位置 ==========" << std::endl;
        std::cout << "坐标: (" << x << ", " << y << ")" << std::endl;
        std::cout << "BGR 值: (" << (int)bgr[0] << ", " << (int)bgr[1] << ", " << (int)bgr[2] << ")" << std::endl;
        std::cout << "HSV 值: (" << (int)hsv_val[0] << ", " << (int)hsv_val[1] << ", " << (int)hsv_val[2] << ")" << std::endl;
        std::cout << "可直接用的 Scalar 范围参考：上限 (" << (int)hsv_val[0] + 10 << ", 255, 255)" << std::endl;

        // 在图像上点击处画一个小红点，方便看到点在哪了
        cv::circle(img, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
        cv::imshow("取色器 - 点击花朵边缘", img);
    }
}

int main() {
    // 1. 读取图像
    img = cv::imread("resources/test_image.jpg");
    if (img.empty()) {
        std::cerr << "❌ 无法读取图片，请检查路径！" << std::endl;
        return -1;
    }

    // 2. 转换到 HSV 空间
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // 3. 绑定鼠标回调函数
    cv::namedWindow("取色器 - 点击花朵边缘", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("取色器 - 点击花朵边缘", onMouse, nullptr);

    // 4. 显示图像并等待按键
    cv::imshow("取色器 - 点击花朵边缘", img);
    cv::waitKey(0);
    return 0;
}