//
// Created by nuc on 23-7-14.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include "LocalWarping.h"
#include "MeshWarping.h"

using namespace std;
using namespace cv;

void getMask(Mat &input_img, Mat &whiteMask){
    cv::Scalar lowerWhite = cv::Scalar(245, 245, 245);
    cv::Scalar upperWhite = cv::Scalar(255, 255, 255);

    cv::inRange(input_img, lowerWhite, upperWhite, whiteMask);

    Mat element = getStructuringElement(MORPH_RECT, Size(6, 6));
    morphologyEx(whiteMask, whiteMask, MORPH_OPEN, element);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(whiteMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//    cv::drawContours(input_img, contours, -1, cv::Scalar(0,0,255), cv::FILLED);

    vector<vector<Point>> contours_in_region;
    for (const auto& contour : contours) {
        // 检查轮廓中的每个点是否在图像边界附近
        bool containsNearBoundary = false;
        for (const auto& point : contour) {
            if (point.x <= 2 || point.x >= input_img.cols - 3 || point.y <= 2 || point.y >= input_img.rows - 3) {
                containsNearBoundary = true;
                break;
            }
        }

        if (!containsNearBoundary) {
            contours_in_region.emplace_back(contour);
        }
    }


    for (const auto& contour : contours_in_region) {
        // 在掩膜图像上填充轮廓
        cv::fillPoly(whiteMask, contour, cv::Scalar(0));
    }

}

int main(){
    Mat rgbImage = imread("/home/nuc/workspace/ImageWarping/data/building.png");
    resize(rgbImage,rgbImage,Size(rgbImage.cols/1,rgbImage.rows/1));

    Mat mask(rgbImage.size(), CV_8UC1);
    getMask(rgbImage, mask);
//    imshow("mask",mask);
//    imshow("rgb",rgbImage);
//    waitKey(0);

    Mat expand_img;
    LocalWarping localWarping(rgbImage, mask);
    localWarping.getExpandImage(expand_img);
    MeshWarping meshWarping(rgbImage, expand_img, mask);

    cout<<"hello warping"<<endl;
}