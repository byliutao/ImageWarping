//
// Created by nuc on 23-7-15.
//

#include "MeshWarping.h"



MeshWarping::MeshWarping(Mat &source_img, Mat &expand_img, Mat &mask, int mesh_quad_length,
                         vector<vector<Point2i>> &displacement_map) {
    _source_img = source_img.clone();
    _expand_img = expand_img.clone();
    _mask = mask.clone();
    _mesh_quad_length = mesh_quad_length;
    _displacement_map = displacement_map;


    initRectangleImageMesh();
    warpBackToSourceImageMesh();

#ifdef MESH_SHOW
    drawGrids(_warped_back_grids, "_warped_back_grids", _source_img, true);
#endif
}

void MeshWarping::initRectangleImageMesh() {
    _mesh_cols = _expand_img.cols / _mesh_quad_length;
    _mesh_rows = _expand_img.rows / _mesh_quad_length;

// 生成网格
    for (int row = 0; row < _mesh_rows; ++row) {
        for (int col = 0; col < _mesh_cols; ++col) {
            // 计算网格在图像中的四个顶点的坐标
            cv::Point2i tl(col * _mesh_quad_length, row * _mesh_quad_length);
            cv::Point2i tr((col + 1) * _mesh_quad_length, row * _mesh_quad_length);
            cv::Point2i br((col + 1) * _mesh_quad_length, (row + 1) * _mesh_quad_length);
            cv::Point2i bl(col * _mesh_quad_length, (row + 1) * _mesh_quad_length);

            // 创建网格对象并保存
            Grid grid(tl, tr, br, bl);
            _rectangle_grids.emplace_back(grid);
        }
    }

#ifdef SHOW_MESH
    drawGrids(_rectangle_grids, "Image with Grid", _expand_img);
#endif
}

void MeshWarping::warpBackToSourceImageMesh() {

    for(int i = 0; i < _rectangle_grids.size(); i++){
#ifdef MESH_SHOW_STEP
        drawGrids(_warped_back_grids, "_warped_back_grids", _source_img, false);
        waitKey(10);
#endif
        Point2i displaced_top_left = _rectangle_grids[i].top_left;
        Point2i original_top_left = displaced_top_left + _displacement_map[displaced_top_left.y][displaced_top_left.x];

        Point2i displaced_top_right = _rectangle_grids[i].top_right;
        Point2i original_top_right = displaced_top_right + _displacement_map[displaced_top_right.y][displaced_top_right.x];

        Point2i displaced_bottom_left = _rectangle_grids[i].bottom_left;
        Point2i original_bottom_left = displaced_bottom_left + _displacement_map[displaced_bottom_left.y][displaced_bottom_left.x];

        Point2i displaced_bottom_right = _rectangle_grids[i].bottom_right;
        Point2i original_bottom_right = displaced_bottom_right + _displacement_map[displaced_bottom_right.y][displaced_bottom_right.x];

        _warped_back_grids.emplace_back(original_top_left, original_top_right, original_bottom_right, original_bottom_left);
    }



}

//void MeshWarping::drawGrids(vector<Grid> grids, string window_name, Mat &paint_img) {
//    // 显示网格
//    Mat paint = paint_img.clone();
//    for (const Grid& grid : grids) {
//        // 绘制网格边界线
//        cv::line(paint, grid.top_left, grid.top_right, cv::Scalar(0, 0, 255), 1);
//        cv::line(paint, grid.top_right, grid.bottom_right, cv::Scalar(0, 0, 255), 1);
//        cv::line(paint, grid.bottom_right, grid.bottom_left, cv::Scalar(0, 0, 255), 1);
//        cv::line(paint, grid.bottom_left, grid.top_left, cv::Scalar(0, 0, 255), 1);
//    }
//
//    // 显示带有网格的图像
//    namedWindow(window_name,WINDOW_NORMAL);
//    cv::imshow(window_name, paint);
//    cv::waitKey(0);
//}

void MeshWarping::getWarpedBackGridsInfo(vector<Grid> &rectangle_grids, vector<Grid> &warped_back_grids, int &mesh_rows,
                                         int &mesh_cols) {
    rectangle_grids = _rectangle_grids;
    warped_back_grids = _warped_back_grids;
    mesh_rows = _mesh_rows;
    mesh_cols = _mesh_cols;
}

