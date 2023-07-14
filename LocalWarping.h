//
// Created by nuc on 23-7-14.
//

#ifndef CONFORMALRESIZING_LOCALWARPING_H
#define CONFORMALRESIZING_LOCALWARPING_H
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;


class LocalWarping {
    typedef enum
    {
        Top	= 0,
        Bottom	= 1,
        Left = 2,
        Right = 3
    } Direction;

    Mat _source_img;
    Mat _local_wraping_img;
    Rect getSubImageRect(Direction direction);
    void imageShift(Direction direction);
    void getEmptyPositionMask(Mat &input, Mat &output);
    void calculateSeam(Mat &image);
    void calculateEnergyImage(Mat &input_image, Mat &energy_image);
public:
    LocalWarping(Mat &source_img);
    void warpImage(Mat &input_img, Mat &output_img);


};


#endif //CONFORMALRESIZING_LOCALWARPING_H
