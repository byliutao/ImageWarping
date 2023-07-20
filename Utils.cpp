//
// Created by nuc on 23-7-16.
//

#include "Utils.h"

void drawGrids(vector<Grid> grids, string window_name, Mat &paint_img, bool is_wait) {
    // 显示网格
    Mat paint = paint_img.clone();
    for (int i = 0; i < grids.size(); i++) {
        // 绘制网格边界线
        Grid grid = grids[i];
        cv::line(paint, grid.top_left, grid.top_right, cv::Scalar(255, (i*11)%256), 2);
        cv::line(paint, grid.top_right, grid.bottom_right, cv::Scalar(255, (i*33)%256, (i*11)%256), 2);
        cv::line(paint, grid.bottom_right, grid.bottom_left, cv::Scalar(255, (i*33)%256, (i*11)%256), 2);
        cv::line(paint, grid.bottom_left, grid.top_left, cv::Scalar(255, (i*33)%256, (i*11)%256), 2);
    }

    // 显示带有网格的图像
    namedWindow(window_name,WINDOW_NORMAL);
    cv::imshow(window_name, paint);
    if(is_wait) cv::waitKey(0);
    else cv::waitKey(1);
//    destroyAllWindows();
}

void cropImage(Mat &input_image, Mat &result_image, int quad_length){

    int cols = input_image.cols - 1;
    int rows = input_image.rows - 1;

    // 计算列数除以 quad_length 的商 cols_new，确保列数可以被 quad_length 整除
    int cols_new = cols / quad_length;
    cols = quad_length * cols_new;

    // 计算行数除以 quad_length 的商 rows_new，确保行数可以被 quad_length 整除
    int rows_new = rows / quad_length;
    rows = quad_length * rows_new;

    // 使用 cv::Rect 指定裁剪区域的左上角点和宽度高度
    cv::Rect cropRect(0, 0, cols+1, rows+1);

    // 获取图像的子图像（裁剪图像）
    result_image = input_image(cropRect);
}

void getMask(Mat &input_img, Mat &whiteMask){
    cv::Scalar lowerWhite = cv::Scalar(245, 245, 245);
    cv::Scalar upperWhite = cv::Scalar(255, 255, 255);

    cv::inRange(input_img, lowerWhite, upperWhite, whiteMask);

    Mat element = getStructuringElement(MORPH_RECT, Size(6, 6));
    morphologyEx(whiteMask, whiteMask, MORPH_OPEN, element);
    dilate(whiteMask,whiteMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(12, 12)));
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(whiteMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//    cv::drawContours(input_img, contours, -1, cv::Scalar(0,0,255), cv::FILLED);

    vector<vector<Point>> contours_in_region;
    for (const auto& contour : contours) {
        // 检查轮廓中的每个点是否在图像边界附近
        bool containsNearBoundary = false;
        for (const auto& point : contour) {
            if (point.x <= 2 || point.x >= input_img.cols - 3 || point.y <= 2 || point.y >= input_img.rows - 3) {
                containsNearBoundary = true;
                break;
            }
        }

        if (!containsNearBoundary) {
            contours_in_region.emplace_back(contour);
        }
    }


    for (const auto& contour : contours_in_region) {
        // 在掩膜图像上填充轮廓
        cv::fillPoly(whiteMask, contour, cv::Scalar(0));
    }



}

// Function turn a cv::Mat into a texture, and return the texture ID as a GLuint for use
GLuint matToTexture(const cv::Mat &mat, GLenum minFilter, GLenum magFilter, GLenum wrapFilter) {
    // Generate a number for our textureID's unique handle
    GLuint textureID;
    glGenTextures(1, &textureID);

    // Bind to our texture handle
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Catch silly-mistake texture interpolation method for magnification
    if (magFilter == GL_LINEAR_MIPMAP_LINEAR  ||
        magFilter == GL_LINEAR_MIPMAP_NEAREST ||
        magFilter == GL_NEAREST_MIPMAP_LINEAR ||
        magFilter == GL_NEAREST_MIPMAP_NEAREST)
    {
        cout << "You can't use MIPMAPs for magnification - setting filter to GL_LINEAR" << endl;
        magFilter = GL_LINEAR;
    }

    // Set texture interpolation methods for minification and magnification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);

    // Set texture clamping method
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapFilter);

    // Set incoming texture format to:
    // GL_BGR       for CV_CAP_OPENNI_BGR_IMAGE,
    // GL_LUMINANCE for CV_CAP_OPENNI_DISPARITY_MAP,
    // Work out other mappings as required ( there's a list in comments in main() )
    GLenum inputColourFormat = GL_BGR;
    if (mat.channels() == 1)
    {
        inputColourFormat = GL_LUMINANCE;
    }

    // Create the texture
    glTexImage2D(GL_TEXTURE_2D,     // Type of texture
                 0,                 // Pyramid level (for mip-mapping) - 0 is the top level
                 GL_RGB,            // Internal colour format to convert to
                 mat.cols,          // Image width  i.e. 640 for Kinect in standard mode
                 mat.rows,          // Image height i.e. 480 for Kinect in standard mode
                 0,                 // Border width in pixels (can either be 1 or 0)
                 inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
                 GL_UNSIGNED_BYTE,  // Image data type
                 mat.ptr());        // The actual image data itself

    // If we're using mipmaps then generate them. Note: This requires OpenGL 3.0 or higher
    if (minFilter == GL_LINEAR_MIPMAP_LINEAR  ||
        minFilter == GL_LINEAR_MIPMAP_NEAREST ||
        minFilter == GL_NEAREST_MIPMAP_LINEAR ||
        minFilter == GL_NEAREST_MIPMAP_NEAREST)
    {
        glGenerateMipmap(GL_TEXTURE_2D);
    }

    return textureID;
}

void error_callback(int error, const char* description) {
    fprintf(stderr, "Error: %s\n", description);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

void init_opengl(int w, int h) {
    glViewport(0, 0, w, h); // use a screen size of WIDTH x HEIGHT

    glMatrixMode(GL_PROJECTION);     // Make a simple 2D projection on the entire window
    glLoadIdentity();
    glOrtho(0.0, w, h, 0.0, 0.0, 100.0);

    glMatrixMode(GL_MODELVIEW);    // Set the matrix mode to object modeling

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear the window
}


double isInsideGrid(Point2i point, Grid grid){
    vector<Point2f> contour = {grid.top_left, grid.top_right, grid.bottom_right, grid.bottom_left};
    return pointPolygonTest(contour,point,true);
}

float cross( Point2f a, Point2f b ) { return a.x*b.y - a.y*b.x; }

bool getInvBilinearWeight(Point2i p, Grid grid, pair<double, double> &res_w) {
    Point2f e = grid.top_right - grid.top_left;
    Point2f f = grid.bottom_left - grid.top_left;
    Point2f g = grid.top_left - grid.top_right + grid.bottom_right - grid.bottom_left;
    Point2f h = p - grid.top_left;

    if(e == f || e == g || e == h || f == g || f == h || g == h){
        res_w.first = 1.0;
        res_w.second = 0.0;
        return false;
    }


    float k2 = cross(g, f );
    float k1 = cross(e, f ) + cross(h, g );
    float k0 = cross(h, e );

    // if edges are parallel, this is a linear equation
    if( abs(k2)<0.001 )
    {
        res_w = pair<double,double>((h.x * k1 + f.x * k0) / (e.x * k1 - g.x * k0), -k0 / k1 );
        return true;
    }
    // otherwise, it's a quadratic
    else
    {
        float w = k1*k1 - 4.0*k0*k2;
        if( w<0.0 ) {
            res_w = pair<double,double>(-1.0,-1.0);
            return false;
        }
        w = sqrt( w );

        float ik2 = 0.5/k2;
        float v = (-k1 - w)*ik2;
        float u = (h.x - f.x*v)/(e.x + g.x*v);

        if( u<0.0 || u>1.0 || v<0.0 || v>1.0 )
        {
            v = (-k1 + w)*ik2;
            u = (h.x - f.x*v)/(e.x + g.x*v);
        }
        res_w = pair<double,double>( u, v );
    }
    return true;
}

Eigen::MatrixXd bilinearWeightsToMatrix(pair<double,double> w) {
    Eigen::MatrixXd mat(2,8);
    double v1w= 1 - w.first - w.second + w.first*w.second;
    double v2w = w.first - w.first*w.second;
    double v3w = w.second - w.first*w.second;
    double v4w = w.first*w.second;
    mat << v1w, 0, v2w, 0, v3w, 0, v4w, 0,
            0, v1w, 0, v2w, 0, v3w, 0, v4w;
    return mat;
}


