//
// Created by nuc on 23-7-15.
//

#include "MeshWarping.h"



MeshWarping::MeshWarping(Mat &source_img, Mat &expand_img, Mat &mask, int mesh_quad_length, vector<vector<Point2i>> &seams_left,vector<vector<Point2i>> &seams_right,
                         vector<vector<Point2i>> &seams_top,vector<vector<Point2i>> &seams_bottom) {
    _source_img = source_img.clone();
    _expand_img = expand_img.clone();
    _mask = mask.clone();
    _mesh_quad_length = mesh_quad_length;


    _seams_left = seams_left;
    _seams_right = seams_right;
    _seams_top = seams_top;
    _seams_bottom = seams_bottom;

    initExpandImageMesh();
    initSourceImageMesh();
//    imshow("_source_img",_source_img);
//    imshow("_expand_img",_expand_img);
//    imshow("_mask1",_mask);
//    waitKey(0);
}

void MeshWarping::initExpandImageMesh() {
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
            _expand_img_grids.emplace_back(grid);
        }
    }

#ifdef SHOW_MESH
    drawGrids(_expand_img_grids,"Image with Grid");
#endif
}

void MeshWarping::initSourceImageMesh() {
#ifdef SHOW_MESH
    Mat paint = _expand_img.clone();
    for(const auto& seam : _seams_bottom){
        for(auto point : seam){
            paint.at<Vec3b>(point) = Vec3b(0,0,255);
        }
    }
    for(const auto& seam : _seams_top){
        for(auto point : seam){
            paint.at<Vec3b>(point) = Vec3b(255,0,0);
        }
    }
    for(const auto& seam : _seams_left){
        for(auto point : seam){
            paint.at<Vec3b>(point) = Vec3b(0,255,0);
        }
    }
    for(const auto& seam : _seams_right){
        for(auto point : seam){
            paint.at<Vec3b>(point) = Vec3b(255,0,255);
        }
    }
    imshow("mesh_point",paint);
    waitKey(0);
#endif

    _source_img_grids = _expand_img_grids;

    for(const auto& seam : _seams_top){
        vector<Grid> contained_grids;
        vector<Point2i> grids_position;
        vector<bool> isBeenPut(_expand_img_grids.size(), false);
        for(auto point : seam){
            for(int i = 0; i < _expand_img_grids.size(); i++){
                if(_expand_img_grids[i].isContainPoint(point)){
                    if(!isBeenPut[i]){
                        isBeenPut[i] = true;
                        contained_grids.push_back(_expand_img_grids[i]);
                        grids_position.emplace_back((int)i % _mesh_cols, (int)i / _mesh_cols);
                    }
                    break;
                }
            }
        }
        Mat paint = _expand_img.clone();
        for (int i = 0; i < contained_grids.size(); i++) {
            // 绘制网格边界线
            Grid grid = contained_grids[i];
            cv::line(paint, grid.top_left, grid.top_right, cv::Scalar(0, 0, 255), 1);
            cv::line(paint, grid.top_right, grid.bottom_right, cv::Scalar(0, 0, 255), 1);
            cv::line(paint, grid.bottom_right, grid.bottom_left, cv::Scalar(0, 0, 255), 1);
            cv::line(paint, grid.bottom_left, grid.top_left, cv::Scalar(0, 0, 255), 1);
            circle(paint, Point2i(grids_position[i].x * _mesh_quad_length + 13, grids_position[i].y * _mesh_quad_length + 13), 2, Scalar(0, 255, 0));
        }

        // 显示带有网格的图像
        cv::imshow("window_name", paint);
        cv::waitKey(0);
        updateMesh(contained_grids, grids_position, Top);

    }



}

void MeshWarping::drawGrids(vector<Grid> grids, string window_name) {
    // 显示网格
    Mat paint = _expand_img.clone();
    for (const Grid& grid : grids) {
        // 绘制网格边界线
        cv::line(paint, grid.top_left, grid.top_right, cv::Scalar(0, 0, 255), 1);
        cv::line(paint, grid.top_right, grid.bottom_right, cv::Scalar(0, 0, 255), 1);
        cv::line(paint, grid.bottom_right, grid.bottom_left, cv::Scalar(0, 0, 255), 1);
        cv::line(paint, grid.bottom_left, grid.top_left, cv::Scalar(0, 0, 255), 1);
    }

    // 显示带有网格的图像
    cv::imshow(window_name, paint);
    cv::waitKey(0);
}

void MeshWarping::updateMesh(vector<Grid> grids, vector<Point2i> grids_position, MeshWarping::Position position) {
    //把所有包含的gird往下平移一格
    vector<bool> is_move_of_grids(_source_img_grids.size(),false);
    if(position == Top){
        for(int i = 0; i < grids.size(); i++){
            drawGrids(_source_img_grids,"grids");
            int y = grids_position[i].y;
            int x = grids_position[i].x;
            if(!is_move_of_grids[y * _mesh_cols + x]){
                is_move_of_grids[y * _mesh_cols + x] = true;
                _source_img_grids[y * _mesh_cols + x].top_left.y += 1;
                _source_img_grids[y * _mesh_cols + x].top_right.y += 1;
            }
            while(--y >= 0){
                if(!is_move_of_grids[y * _mesh_cols + x]){
                    is_move_of_grids[y * _mesh_cols + x] = true;
                    _source_img_grids[y * _mesh_cols + x].top_left.y += 1;
                    _source_img_grids[y * _mesh_cols + x].top_right.y += 1;
                    _source_img_grids[y * _mesh_cols + x].bottom_left.y += 1;
                    _source_img_grids[y * _mesh_cols + x].bottom_right.y += 1;
                }
            }
        }
    }
    else if(position == Bottom){

    }
    else if(position == Left){

    }
    else{

    }


}
