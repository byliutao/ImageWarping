//
// Created by nuc on 23-7-16.
//

#ifndef CONFORMALRESIZING_GLOBALWARPING_H
#define CONFORMALRESIZING_GLOBALWARPING_H
#define GLOBAL_SHOW

#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glut.h>

#include "Utils.h"
#include "lsd.h"

using namespace std;
using namespace cv;

class GlobalWarping {
    vector<Grid> _rectangle_grids;
    vector<Grid> _warped_back_grids;
    vector<vector<Point2i>> _warped_back_coordinates;
    vector<Grid> _optimized_grids;
    vector<vector<Point2i>> _optimized_coordinates;
    Mat _source_img;
    Mat _mask;
    Mat _render_img;
    int _mesh_rows;
    int _mesh_cols;
    int _rectangle_width;
    int _rectangle_height;
    const int _iter_times = 10;
    const int _M = 50;
    const double _lambda_L = 100;
    const double _lambda_B = 1e8;

    int _argc;
    char **_argv;

    void verifyOptimizedGrids();
    void optimizeEnergyFunction();
    void generateCoordinates();
    void calculateShapeEnergy(Eigen::MatrixXd &shape_matrix_A);
    void calculateBoundaryEnergy(Eigen::MatrixXd &boundary_matrix_A, Eigen::VectorXd &boundary_vector_b);
    void getOptimizedGridsFromX(Eigen::VectorXd &X);

public:
    GlobalWarping(Mat &source_img, Mat &mask, vector<Grid> rectangle_grids,
                  vector<Grid> warped_back_grids, int mesh_rows, int mesh_cols, int argc, char **argv);
    void getOptimizedGrids(vector<Grid> &optimized_grids);
};


#endif //CONFORMALRESIZING_GLOBALWARPING_H
