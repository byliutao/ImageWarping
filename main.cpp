//
// Created by nuc on 23-7-14.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include "LocalWarping.h"
#include "MeshWarping.h"
#include "GlobalWarping.h"
#include "Utils.h"

using namespace std;
using namespace cv;

const int super_parameter_mesh_quad_length = 26;


int main(int argc, char* argv[]){
    Mat rgbImage = imread("/home/nuc/workspace/ImageWarping/data/building.png");
    resize(rgbImage,rgbImage,Size(rgbImage.cols/3,rgbImage.rows/3));

    cropImage(rgbImage,rgbImage,super_parameter_mesh_quad_length);

    Mat mask(rgbImage.size(), CV_8UC1);
    getMask(rgbImage, mask);
//    imshow("mask",mask);
//    imshow("rgb",rgbImage);
//    waitKey(0);

    Mat expand_img;
    vector<vector<Point2i>> displacement_map;
    LocalWarping localWarping(rgbImage, mask);
    localWarping.getExpandImage(expand_img);
    localWarping.getSeams(displacement_map);
    MeshWarping meshWarping(rgbImage, expand_img, mask, super_parameter_mesh_quad_length,
                            displacement_map);

    vector<Grid> warped_back_grids;
    vector<Grid> rectangle_grids;
    int mesh_rows, mesh_cols;
    meshWarping.getWarpedBackGridsInfo(rectangle_grids, warped_back_grids, mesh_rows, mesh_cols);

    GlobalWarping globalWarping(rgbImage, mask, rectangle_grids, warped_back_grids, mesh_rows, mesh_cols, argc, argv);

    cout<<"hello warping"<<endl;
}