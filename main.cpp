//
// Created by nuc on 23-7-14.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include "LocalWarping.h"

using namespace std;
using namespace cv;
int main(){
    Mat rgbImage = imread("/home/nuc/workspace/ImageWarping/data/panorama.png");
    resize(rgbImage,rgbImage,Size(rgbImage.cols/4,rgbImage.rows/4));
    LocalWarping localWarping(rgbImage);

    cout<<"hello warping"<<endl;
}