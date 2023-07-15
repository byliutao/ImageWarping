//
// Created by nuc on 23-7-14.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include "LocalWarping.h"
#include "MeshWarping.h"

using namespace std;
using namespace cv;

const int super_parameter_mesh_quad_length = 26;

void cropImage(Mat &input_image, Mat &result_image, int quad_length){

    int cols = input_image.cols;
    int rows = input_image.rows;

    // 计算列数除以 quad_length 的商 cols_new，确保列数可以被 quad_length 整除
    int cols_new = cols / quad_length;
    cols = quad_length * cols_new;

    // 计算行数除以 quad_length 的商 rows_new，确保行数可以被 quad_length 整除
    int rows_new = rows / quad_length;
    rows = quad_length * rows_new;

    // 使用 cv::Rect 指定裁剪区域的左上角点和宽度高度
    cv::Rect cropRect(0, 0, cols, rows);

    // 获取图像的子图像（裁剪图像）
    result_image = input_image(cropRect);
}

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
    resize(rgbImage,rgbImage,Size(rgbImage.cols/2,rgbImage.rows/2));

    cropImage(rgbImage,rgbImage,super_parameter_mesh_quad_length);

    Mat mask(rgbImage.size(), CV_8UC1);
    getMask(rgbImage, mask);
//    imshow("mask",mask);
//    imshow("rgb",rgbImage);
//    waitKey(0);

    Mat expand_img;
    vector<vector<Point2i>> seams_top, seams_bottom, seams_left, seams_right;
    LocalWarping localWarping(rgbImage, mask);
    localWarping.getExpandImage(expand_img);
    localWarping.getSeams(seams_left,seams_right,seams_top,seams_bottom);
    MeshWarping meshWarping(rgbImage, expand_img, mask, super_parameter_mesh_quad_length,
                            seams_left,seams_right,seams_top,seams_bottom);

    cout<<"hello warping"<<endl;
}