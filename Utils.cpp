//
// Created by nuc on 23-7-16.
//

#include "Utils.h"

void drawGrids(vector<Grid> grids, string window_name, Mat &paint_img, bool is_wait) {
    // 显示网格
    Mat paint = paint_img.clone();
    for (int i = 0; i < grids.size(); i++) {
        // 绘制网格边界线
        Grid grid = grids[i];
        cv::line(paint, grid.top_left, grid.top_right, cv::Scalar(255, (i*33)%256, (i*11)%256), 1);
        cv::line(paint, grid.top_right, grid.bottom_right, cv::Scalar(255, (i*33)%256, (i*11)%256), 1);
        cv::line(paint, grid.bottom_right, grid.bottom_left, cv::Scalar(255, (i*33)%256, (i*11)%256), 1);
        cv::line(paint, grid.bottom_left, grid.top_left, cv::Scalar(255, (i*33)%256, (i*11)%256), 1);
    }

    // 显示带有网格的图像
    namedWindow(window_name,WINDOW_NORMAL);
    cv::imshow(window_name, paint);
    if(is_wait) cv::waitKey(0);
    else cv::waitKey(1);
}

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
    cv::Rect cropRect(0, 0, cols+1, rows+1);

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

