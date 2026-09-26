# Vision Training 项目 README

本工程为“第二次培训”的综合项目，包含 OpenCV 图像处理、Eigen 矩阵运算与 Ceres 优化等内容。三个任务共用同一个工程，通过根目录的 `CMakeLists.txt` 统一构建。

## 1. 环境依赖
- 操作系统：Ubuntu 22.04
- 编译工具：CMake 3.10+，g++ 支持 C++17
- 第三方库：
  - OpenCV 4.5.4
  - Eigen 3.4
  - Ceres Solver 2.1.0（仅任务2/3需使用）
## 构建命令与运行命令
```bash
cmake -S . -B build
cmake --build build -j
./build/task1_image     # 任务1：图片处理
./build/task2_fit       # 任务2：视频参数拟合
./build/task3_windmill  # 任务3：能量机关跟踪
```
## 输入与输出
  输入素材：统一放置于 resources/ 目录下，包括 test_image.jpg、task_2.mp4、task_3.mp4、task_4.mp4。
  
  输出结果：统一放置于 result/ 目录下，按任务分文件夹存放。
## result中各个项目对应任务



任务一
        1. 读图与颜色转换：读取 test_image.jpg，输出灰度图以及 H、S、V 单通道图。
        2. 滤波对比：实现均值、高斯、中值滤波，核尺寸统一为 Size(5, 5)，高斯滤波 sigmaX=1.5。对比花瓣边缘与细节的变化。
        3. 红色提取：使用 HSV 双区间阈值生成红色掩膜。
        · 阈值记录：H: 0~17和160~179，S: 80~255，V: 100~255。
        · 红色、黄色边缘和阴影的处理效果：红花瓣提取较好，但黄色边缘和阴影部分因阈值范围被忽略。
        在这里额外使用了取色器程序，花朵红色部分更完整（对应main4.cpp中程序）


        4. 形态学与轮廓：对掩膜分别展示腐蚀、膨胀、开运算、闭运算的效果（核大小为 Size(5,5)）。选择闭运算后的结果提取外轮廓，按面积筛选（面积 > 500）。
        5. 绘制与变换：在原图副本上绘制圆、矩形和文字；绕图像中心旋转 35°；裁剪原图左上角 1/4。

    任务1结果索引（对应 result/task1_images/）

        · original.png（原图副本）
        · gray.png（灰度图）
        · hsv_h.png、hsv_s.png、hsv_v.png（H/S/V 单通道）
        · mean.png、gaussian.png、median.png（三种滤波对比）
        · redmask.png（红色掩膜）
        · erode.png、dilate.png、open.png、close.png（形态学效果）
        · contours_boxes.png（轮廓与外接矩形，图中已标出筛选后的轮廓面积）
        · drawing.png（绘制圆、矩形、文字）
        · rotated_35deg_eigen.png（旋转35度）
        · crop_top_left.png（裁剪左上角1/4）
任务2
本次任务选择只拟合角速度的方式，而不拟合初始角度 θ₀。

    · 方法：Ω 扫描 + Eigen 线性最小二乘
    · 原理：将模型 ω(t) = b + A·sin(Ωt + φ) 展开为 ω(t) = b + C1·sin(Ωt) + C2·cos(Ωt)（其中 C1 = A·cosφ，C2 = A·sinφ）。对于给定的 Ω，这是一个关于 b, C1, C2 的线性方程。代码通过遍历 Ω（从 0.1 到 3.0，步长 0.01），对每个 Ω 用 Eigen 求解线性最小二乘，并比较残差平方和 SSE，最终找到最优参数。
    · 约束：A > 0，b > A，Ω > 0。
    · 初值与求解状态：Ω 从 0.1 开始扫描；求解状态为“收敛”。

    · 角速度拟合误差：RMSE = 0.0022912 rad/s
    · 有效样本数：1439（即第0帧到第1439帧，共1440帧减去1个差分帧）
    · 参与计算的帧范围：第 0 帧至第 1439 帧
    · 单位说明：A, b, Ω 的单位为 rad/s，φ 的单位为 rad。

    · 角速度拟合误差：RMSE = 0.0022912 rad/s
    · 有效样本数：1439（即第0帧到第1439帧，共1440帧减去1个差分帧）
    · 参与计算的帧范围：第 0 帧至第 1439 帧
    · 单位说明：A, b, Ω 的单位为 rad/s，φ 的单位为 rad。
    任务索引
    · tracking_overlay.mp4（带标记的跟踪视频）
    · fit_comparison.png（观测与拟合曲线对比图）
    · angular_velocity.png（估计角速度曲线）
    · residuals.png（残差曲线）
    · ../task2_fit_result.md（参数与误差详细说明）
