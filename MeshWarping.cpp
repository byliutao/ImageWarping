//
// Created by nuc on 23-7-15.
//

#include "MeshWarping.h"

MeshWarping::MeshWarping(Mat &source_img, Mat &expand_img, Mat &mask) {
    _source_img = source_img.clone();
    _expand_img = expand_img.clone();
    _mask = mask.clone();

//    imshow("_source_img",_source_img);
//    imshow("_expand_img",_expand_img);
//    imshow("_mask1",_mask);
//    waitKey(0);
}

void MeshWarping::initExpandImageMesh() {
    
}
