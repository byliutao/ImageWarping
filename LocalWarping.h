//
// Created by nuc on 23-7-14.
//

#ifndef CONFORMALRESIZING_LOCALWARPING_H
#define CONFORMALRESIZING_LOCALWARPING_H
//#define SHOW
#include <opencv2/opencv.hpp>
#include <iostream>
#include <limits>

using namespace std;
using namespace cv;


class LocalWarping {
    typedef enum
    {
        Top	= 0,
        Bottom	= 1,
        Left = 2,
        Right = 3
    } Position;

    typedef enum
    {
        UpToDown	= 0,
        LeftToRight	= 1,
    } Direction;

    Mat _mask; //undefined region
    Mat _source_img;
    Mat _local_wraping_img;

    vector<vector<Point2i>> _seams_left;
    vector<vector<Point2i>> _seams_right;
    vector<vector<Point2i>> _seams_top;
    vector<vector<Point2i>> _seams_bottom;

    Rect getSubImageRect(Position position);
    bool imageShift(Position position);
    void singleShift(Position position, const vector<Point2i> &seam);
    void getEmptyPositionMask(Mat &input, Mat &output);
    void calculateSeam(Mat &src, Mat &mask, Direction direction, vector<Point2i> &seam, Rect roi);
    void calculateCostImage(Mat &input_image, Mat &cost_image, Mat &mask);
    void getTheBiggestPosition(LocalWarping::Position &result_position, const vector<bool> &position_flag);
public:
    LocalWarping(Mat &source_img, Mat &mask);
    void getExpandImage(Mat &image);
    void getSeams(vector<vector<Point2i>> &seams_left,vector<vector<Point2i>> &seams_right,
                  vector<vector<Point2i>> &seams_top,vector<vector<Point2i>> &seams_bottom);
};


#endif //CONFORMALRESIZING_LOCALWARPING_H
