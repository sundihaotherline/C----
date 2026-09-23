#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
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
    
    
    return 0;
}
