//
// Created by nuc on 23-7-16.
//

#ifndef CONFORMALRESIZING_UTILS_H
#define CONFORMALRESIZING_UTILS_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

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

void drawGrids(vector<Grid> grids, string window_name, Mat &paint_img, bool is_wait);

void cropImage(Mat &input_image, Mat &result_image, int quad_length);

void getMask(Mat &input_img, Mat &whiteMask);

GLuint matToTexture(const cv::Mat &mat, GLenum minFilter, GLenum magFilter, GLenum wrapFilter);

void error_callback(int error, const char* description);

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

void init_opengl(int w, int h);

#endif //CONFORMALRESIZING_UTILS_H
