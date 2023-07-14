//
// Created by nuc on 23-7-14.
//

#include "LocalWarping.h"

LocalWarping::LocalWarping(cv::Mat &source_img) {
    _source_img = source_img.clone();
    _local_wraping_img = source_img.clone();

    while(true){
        for(int i = 0; i < 4; i++){
            switch (i) {
                case 0:{
                    imageShift(Top);
                    break;
                }
                case 1:{
                    imageShift(Bottom);
                    break;
                }
                case 2:{
                    imageShift(Left);
                    break;
                }
                case 3:{
                    imageShift(Right);
                    break;
                }
                default: break;
            }
        }
    }
}

Rect LocalWarping::getSubImageRect(Direction direction) {
    Mat empty_position_mask(_local_wraping_img.size(), CV_8UC1);
    getEmptyPositionMask(_local_wraping_img, empty_position_mask);

    Rect roi;
    // 获取图像宽度和高度
    int width = empty_position_mask.cols;
    int height = empty_position_mask.rows;

    if(direction == Top || direction == Bottom){
        int longestLength = 0;  // 最长连续像素点集合的长度
        int startPoint = -1;  // 起点位置
        int endPoint = -1;    // 终点位置
        int currentStart = -1;  // 当前连续像素点集合的起点
        int currentLength = 0;  // 当前连续像素点集合的长度

        uchar* rowData;

        if(direction == Top){
            rowData = empty_position_mask.ptr<uchar>(0);
        }
        else{
            rowData = empty_position_mask.ptr<uchar>(height-1);
        }
        for (int i = 0; i < width; ++i)
        {
            if (rowData[i] == 255)  // 当前像素为白色(255)
            {
                if (currentStart == -1)  // 当前没有连续像素点集合，设置起点
                    currentStart = i;

                currentLength++;  // 增加连续像素点集合的长度

                // 更新最长连续像素点集合的信息
                if (currentLength > longestLength)
                {
                    longestLength = currentLength;
                    startPoint = currentStart;
                    endPoint = i;
                }
            }
            else  // 当前像素为黑色(0)
            {
                currentStart = -1;  // 重置连续像素点集合的起点
                currentLength = 0;  // 重置连续像素点集合的长度
            }
        }
        roi = Rect(startPoint,0,longestLength,height);
    }
    else{
        int longestLength = 0;  // 最长连续像素点集合的长度
        int startPoint = -1;  // 起点位置
        int endPoint = -1;    // 终点位置
        int currentStart = -1;  // 当前连续像素点集合的起点
        int currentLength = 0;  // 当前连续像素点集合的长度

        int col_index;
        if(direction == Left){
            col_index = 0;
        }
        else{
            col_index = width-1;
        }
        // 遍历最左列的像素
        for (col_index = 0; col_index < height; ++col_index)
        {
            if (empty_position_mask.at<uchar>(col_index, 0) == 255)  // 当前像素为白色(255)
            {
                if (currentStart == -1)  // 当前没有连续像素点集合，设置起点
                    currentStart = col_index;

                currentLength++;  // 增加连续像素点集合的长度

                // 更新最长连续像素点集合的信息
                if (currentLength > longestLength)
                {
                    longestLength = currentLength;
                    startPoint = currentStart;
                    endPoint = col_index;
                }
            }
            else  // 当前像素为黑色(0)
            {
                currentStart = -1;  // 重置连续像素点集合的起点
                currentLength = 0;  // 重置连续像素点集合的长度
            }
        }
        roi = Rect(0,startPoint,width,longestLength);
    }

    return roi;
}

void LocalWarping::getEmptyPositionMask(Mat &input, Mat &output) {
    int width = input.cols;
    int height = input.rows;

    // 获取图像数据指针
    uchar* rgbData = input.data;
    uchar* binaryData = output.data;

    // 遍历图像数据
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            // 计算当前像素的索引
            int index = i * width + j;

            // 获取当前像素的RGB值
            uchar r = rgbData[3 * index];
            uchar g = rgbData[3 * index + 1];
            uchar b = rgbData[3 * index + 2];

            // 判断RGB值是否为(255, 255, 255)，若是则在二值图像中设为白色(255)，否则设为黑色(0)
            if (r == 255 && g == 255 && b == 255)
                binaryData[index] = 255;
            else
                binaryData[index] = 0;
        }
    }
}

void LocalWarping::imageShift(LocalWarping::Direction direction) {
    Rect roi = getSubImageRect(direction);
    Mat image_roi = Mat(this->_local_wraping_img, roi);
    calculateSeam(image_roi);
}

void LocalWarping::calculateSeam(Mat &image) {
    Mat energy_image;
    calculateEnergyImage(image,energy_image);
}

void LocalWarping::calculateEnergyImage(Mat &input_image, Mat &energy_image) {
    // 转换为灰度图像
    cv::Mat grayImage;
    cv::cvtColor(input_image, grayImage, cv::COLOR_BGR2GRAY);

    // 计算梯度
    cv::Mat gradientX, gradientY;
    cv::Sobel(grayImage, gradientX, CV_32F, 1, 0);
    cv::Sobel(grayImage, gradientY, CV_32F, 0, 1);

    // 计算梯度幅值
    cv::magnitude(gradientX, gradientY, energy_image);

    // 将 RGB 图像中像素值为 (255, 255, 255) 的点在梯度图像中对应的值设置为 10^8
    for (int i = 0; i < input_image.rows; ++i)
    {
        for (int j = 0; j < input_image.cols; ++j)
        {
            cv::Vec3b rgbPixel = input_image.at<cv::Vec3b>(i, j);
            float gradientValue = energy_image.at<float>(i, j);

            if (rgbPixel[0] == 255 && rgbPixel[1] == 255 && rgbPixel[2] == 255)
            {
                energy_image.at<float>(i, j) = 1e8;  // 设置为 10 的 8 次方
                energy_image.at<float>(i, j) = 0;  // 设置为 10 的 8 次方
            }
        }
    }

    // 归一化到[0, 255]范围
    cv::normalize(energy_image, energy_image, 0, 255, cv::NORM_MINMAX);

    // 转换为8位无符号整型
    cv::Mat gradientImage;
    energy_image.convertTo(gradientImage, CV_8U);

    // 显示原始RGB图像和梯度图像
    cv::imshow("RGB Image", input_image);
    cv::imshow("Gradient Image", gradientImage);
    cv::waitKey(0);
}
