//
// Created by nuc on 23-7-15.
//

#ifndef CONFORMALRESIZING_MESHWARPING_H
#define CONFORMALRESIZING_MESHWARPING_H
#include <opencv2/opencv.hpp>
#include <iostream>

#define SHOW_MESH
using namespace std;
using namespace cv;

class Grid {
public:
    cv::Point2i top_left;     // 左上角顶点
    cv::Point2i top_right;    // 右上角顶点
    cv::Point2i bottom_right; // 右下角顶点
    cv::Point2i bottom_left;  // 左下角顶点

    Grid(cv::Point2i tl, cv::Point2i tr, cv::Point2i br, cv::Point2i bl)
            : top_left(tl), top_right(tr), bottom_right(br), bottom_left(bl) {

    }

    bool isContainPoint(Point2i point){
        cv::Rect gridRect(top_left, bottom_right);
        return gridRect.contains(point);
    }

};


class MeshWarping {
    typedef enum
    {
        Top	= 0,
        Bottom	= 1,
        Left = 2,
        Right = 3
    } Position;

    Mat _source_img;
    Mat _expand_img;
    Mat _mask;
    vector<Grid> _expand_img_grids;
    vector<Grid> _source_img_grids;
    vector<vector<Point2i>> _seams_left, _seams_right, _seams_top, _seams_bottom;
    int _mesh_quad_length;
    int _mesh_rows;
    int _mesh_cols;

    void updateMesh(vector<Grid> grids, vector<Point2i> grids_position, Position position);
    void initExpandImageMesh();
    void initSourceImageMesh();
    void drawGrids(vector<Grid> grids, string window_name);
public:
    MeshWarping(Mat &source_img, Mat &expand_img, Mat &mask, int mesh_quad_length, vector<vector<Point2i>> &seams_left,vector<vector<Point2i>> &seams_right,
                vector<vector<Point2i>> &seams_top,vector<vector<Point2i>> &seams_bottom);
};


#endif //CONFORMALRESIZING_MESHWARPING_H
