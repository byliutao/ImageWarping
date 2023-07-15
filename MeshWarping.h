//
// Created by nuc on 23-7-15.
//

#ifndef CONFORMALRESIZING_MESHWARPING_H
#define CONFORMALRESIZING_MESHWARPING_H
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class MeshWarping {
    Mat _source_img;
    Mat _expand_img;
    Mat _mask;
    vector<vector<Point2i>> mesh_vertices_expand_img;
    vector<vector<Point2i>> mesh_vertices_source_img;
    const int mesh_length = 26;
    const int mesh_width = 14;

    void initExpandImageMesh();
public:
    MeshWarping(Mat &source_img, Mat &expand_img, Mat &mask);
};


#endif //CONFORMALRESIZING_MESHWARPING_H
