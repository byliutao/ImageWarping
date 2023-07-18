//
// Created by nuc on 23-7-15.
//

#ifndef CONFORMALRESIZING_MESHWARPING_H
#define CONFORMALRESIZING_MESHWARPING_H

#define RESULT_SHOW_MESH
//#define SINGLE_STEP_SHOW_MESH

#include <opencv2/opencv.hpp>
#include <iostream>
#include "Utils.h"

using namespace std;
using namespace cv;




class MeshWarping {
    Mat _source_img;
    Mat _expand_img;
    Mat _mask;
    vector<Grid> _rectangle_grids;
    vector<Grid> _warped_back_grids;
    vector<vector<Point2i>> _displacement_map;
    int _mesh_quad_length;
    int _mesh_rows;
    int _mesh_cols;

    void initRectangleImageMesh();
    void warpBackToSourceImageMesh();
//    void drawGrids(vector<Grid> grids, string window_name, Mat &paint_img);
public:
    MeshWarping(Mat &source_img, Mat &expand_img, Mat &mask, int mesh_quad_length,
                vector<vector<Point2i>> &displacement_map);
    void getWarpedBackGridsInfo(vector<Grid> &rectangle_grids, vector<Grid> &warped_back_grids, int &mesh_rows,
                                int &mesh_cols);
};


#endif //CONFORMALRESIZING_MESHWARPING_H
