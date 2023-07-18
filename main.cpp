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

const int super_parameter_mesh_quad_length = 28;

int g_window_width  = 640;
int g_window_height = 480;
int g_mesh_rows;
int g_mesh_cols;
vector<Grid> g_optimized_grids;
vector<Grid> g_warped_back_grids;


static void draw_frame(cv::Mat& frame) {
    // Clear color and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);     // Operate on model-view matrix

    glEnable(GL_TEXTURE_2D);
    GLuint image_tex = matToTexture(frame, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_CLAMP);


    for(int row = 0; row < g_mesh_rows; row++){
        for(int col = 0; col < g_mesh_cols; col++){
            // 定义源图像和目标图像的纹理坐标
            Point2f tl_target, tr_target, bl_target, br_target;
            Point2f tl_source, tr_source, bl_source, br_source;
            Grid target_grid = g_optimized_grids[row*g_mesh_cols + col];
            Grid source_grid = g_warped_back_grids[row*g_mesh_cols + col];
//            drawGrids({target_grid}, "target", frame, false);
//            drawGrids({source_grid}, "local", frame, true);
            float window_ratio_col = (float)frame.cols / (float)g_window_width;
            float window_ratio_row = (float)frame.rows / (float)g_window_height;
            tl_target.x = (float)target_grid.top_left.x / window_ratio_col;
            tr_target.x = (float)target_grid.top_right.x / window_ratio_col;
            bl_target.x = (float)target_grid.bottom_left.x / window_ratio_col;
            br_target.x = (float)target_grid.bottom_right.x / window_ratio_col;
            tl_target.y = (float)g_window_height - (float)target_grid.top_left.y / window_ratio_row;
            tr_target.y = (float)g_window_height - (float)target_grid.top_right.y / window_ratio_row;
            bl_target.y = (float)g_window_height - (float)target_grid.bottom_left.y / window_ratio_row;
            br_target.y = (float)g_window_height - (float)target_grid.bottom_right.y / window_ratio_row;

            tl_source.x = (float)source_grid.top_left.x / (float)frame.cols;
            tr_source.x = (float)source_grid.top_right.x / (float)frame.cols;
            bl_source.x = (float)source_grid.bottom_left.x / (float)frame.cols;
            br_source.x = (float)source_grid.bottom_right.x / (float)frame.cols;
            tl_source.y = 1.0f - (float)source_grid.top_left.y / (float)frame.rows;
            tr_source.y = 1.0f - (float)source_grid.top_right.y / (float)frame.rows;
            bl_source.y = 1.0f - (float)source_grid.bottom_left.y / (float)frame.rows;
            br_source.y = 1.0f - (float)source_grid.bottom_right.y / (float)frame.rows;

            glBegin(GL_QUADS);
            glTexCoord2f(bl_source.x, bl_source.y);glVertex2f(bl_target.x, (float)g_window_height - 1 - bl_target.y);
            glTexCoord2f(tl_source.x, tl_source.y);glVertex2f(tl_target.x, (float)g_window_height - 1 - tl_target.y);
            glTexCoord2f(tr_source.x, tr_source.y);glVertex2f(tr_target.x, (float)g_window_height - 1 - tr_target.y);
            glTexCoord2f(br_source.x, br_source.y);glVertex2f(br_target.x, (float)g_window_height - 1 - br_target.y);
            glEnd();
        }
    }


    glDeleteTextures(1, &image_tex);
    glDisable(GL_TEXTURE_2D);
}

static void resize_callback(GLFWwindow* window, int new_width, int new_height) {
    glViewport(0, 0, g_window_width = new_width, g_window_height = new_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, g_window_width, g_window_height, 0.0, 0.0, 100.0);
    glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char* argv[]){
    Mat rgbImage = imread("/home/nuc/workspace/ImageWarping/data/panorama.png");
    resize(rgbImage,rgbImage,Size(1000,1000*rgbImage.rows/rgbImage.cols));
    cropImage(rgbImage,rgbImage,super_parameter_mesh_quad_length);

    Mat renderSource = rgbImage.clone();

    Mat mask(rgbImage.size(), CV_8UC1);
    getMask(rgbImage, mask);

    Mat expand_img;
    vector<vector<Point2i>> displacement_map;
    LocalWarping localWarping(rgbImage, mask);
    localWarping.getExpandImage(expand_img);
    localWarping.getSeams(displacement_map);
    MeshWarping meshWarping(rgbImage, expand_img, mask, super_parameter_mesh_quad_length,
                            displacement_map);

    vector<Grid> warped_back_grids;
    vector<Grid> rectangle_grids;
    vector<Grid> optimized_grids;
    int mesh_rows, mesh_cols;
    meshWarping.getWarpedBackGridsInfo(rectangle_grids, warped_back_grids, mesh_rows, mesh_cols);

    GlobalWarping globalWarping(rgbImage, mask, rectangle_grids, warped_back_grids, mesh_rows, mesh_cols, argc, argv);
    globalWarping.getOptimizedGrids(optimized_grids);

    g_warped_back_grids = warped_back_grids;
    g_optimized_grids = optimized_grids;
    g_mesh_rows = mesh_rows;
    g_mesh_cols = mesh_cols;

//opengl render
    g_window_width = rgbImage.cols;
    g_window_height = rgbImage.rows;

    GLFWwindow* window;

    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    window = glfwCreateWindow(g_window_width, g_window_height, "Opengl Render", NULL, NULL);
    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwSetKeyCallback(window, key_callback);
    glfwSetWindowSizeCallback(window, resize_callback);

    glfwMakeContextCurrent(window);

    glfwSwapInterval(1);

    //  Initialise glew (must occur AFTER window creation or glew will error)
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        cout << "GLEW initialisation error: " << glewGetErrorString(err) << endl;
        exit(-1);
    }
    cout << "GLEW okay - using version: " << glewGetString(GLEW_VERSION) << endl;

    /* set opengl store format to the same as opencv dose */
    //use fast 4-byte alignment (default anyway) if possible
    glPixelStorei(GL_UNPACK_ALIGNMENT, (renderSource.step & 3) ? 1 : 4);
    //set length of one complete row in data (doesn't need to equal image.cols)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, renderSource.step/renderSource.elemSize());
    // OpenGL中图像数据的行是从下往上存储的，所以需要将行反转
    cv::flip(renderSource, renderSource, 0);

    init_opengl(g_window_width, g_window_height);

    while (!glfwWindowShouldClose(window)) {
        draw_frame(renderSource);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    exit(EXIT_SUCCESS);
}