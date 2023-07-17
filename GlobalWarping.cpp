//
// Created by nuc on 23-7-16.
//

#include "GlobalWarping.h"


GLuint g_source_img_texture;
int g_mesh_rows;
int g_mesh_cols;
Mat g_source_img;
vector<Grid> g_optimized_grids;
vector<Grid> g_warped_back_grids;
vector<vector<Point2i>> g_warped_back_coordinates;


void display() {
    glLoadIdentity();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // 绑定纹理
    glBindTexture(GL_TEXTURE_2D, g_source_img_texture);
    for(int row = 0; row < g_mesh_rows; row++){
        for(int col = 0; col < g_mesh_cols; col++){
//            row = g_mesh_rows - 1;
//            col = g_mesh_cols - 1;
            // 定义源图像和目标图像的纹理坐标
            Point2f tl_target, tr_target, bl_target, br_target;
            Point2f tl_source, tr_source, bl_source, br_source;
            Grid target_grid = g_optimized_grids[row*g_mesh_cols + col];
            Grid source_grid = g_warped_back_grids[row*g_mesh_cols + col];


//            drawGrids({target_grid}, "target", g_source_img, false);
//            drawGrids({source_grid}, "local", g_source_img, true);
            tl_target.x = 2 * ((float)target_grid.top_left.x / (float)g_source_img.cols) - 1;
            tr_target.x = 2 * ((float)target_grid.top_right.x / (float)g_source_img.cols) - 1;
            bl_target.x = 2 * ((float)target_grid.bottom_left.x / (float)g_source_img.cols) - 1;
            br_target.x = 2 * ((float)target_grid.bottom_right.x / (float)g_source_img.cols) - 1;
            tl_target.y = 1 - 2 * ((float)target_grid.top_left.y / (float)g_source_img.rows);
            tr_target.y = 1 - 2 * ((float)target_grid.top_right.y / (float)g_source_img.rows);
            bl_target.y = 1 - 2 * ((float)target_grid.bottom_left.y / (float)g_source_img.rows);
            br_target.y = 1 - 2 * ((float)target_grid.bottom_right.y / (float)g_source_img.rows);

            tl_source.x = (float)source_grid.top_left.x / (float)g_source_img.cols;
            tr_source.x = (float)source_grid.top_right.x / (float)g_source_img.cols;
            bl_source.x = (float)source_grid.bottom_left.x / (float)g_source_img.cols;
            br_source.x = (float)source_grid.bottom_right.x / (float)g_source_img.cols;
            tl_source.y = 1.0f - (float)source_grid.top_left.y / (float)g_source_img.rows;
            tr_source.y = 1.0f - (float)source_grid.top_right.y / (float)g_source_img.rows;
            bl_source.y = 1.0f - (float)source_grid.bottom_left.y / (float)g_source_img.rows;
            br_source.y = 1.0f - (float)source_grid.bottom_right.y / (float)g_source_img.rows;


            glBegin(GL_QUADS);
//            glTexCoord2f(bl_source.x, bl_source.y);glVertex2f(bl_target.x, bl_target.y);
//            glTexCoord2f(br_source.x, br_source.y);glVertex2f(br_target.x, br_target.y);
//            glTexCoord2f(tr_source.x, tr_source.y);glVertex2f(tr_target.x, tr_target.y);
//            glTexCoord2f(tl_source.x, tl_source.y);glVertex2f(tl_target.x, tl_target.y);
            glTexCoord2f(0.0f, 0.0f);glVertex3f(-1.0f, -1.0f, 0.0f);
            glTexCoord2f(1.0f, 0.0f);glVertex3f(1.0f, -1.0f, 0.0f);
            glTexCoord2f(1.0f, 1.0f);glVertex3f(1.0f, 1.0f, 0.0f);
            glTexCoord2f(0.0f, 1.0f);glVertex3f(-1.0f, 1.0f, 0.0f);
            glEnd();
        }
    }

//    glBindTexture(GL_TEXTURE_2D, 0);
//
//    glFlush();
    glutSwapBuffers();

}

void GlobalWarping::loadTexture() {
    // 创建纹理
    glGenTextures(1, &g_source_img_texture);
    glBindTexture(GL_TEXTURE_2D, g_source_img_texture);


//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, g_source_img.cols, g_source_img.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, g_source_img.data);
//    glBindTexture(GL_TEXTURE_2D, 0);
    // 设置纹理参数
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // 加载输入图像作为纹理
//    cv::flip(g_source_img, g_source_img, 0);  // 翻转图像以匹配OpenGL的坐标系
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, g_source_img.cols, g_source_img.rows, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, g_source_img.data);
}

void GlobalWarping::saveResultImage() {
    // 获取OpenGL窗口的宽度和高度
    int width = glutGet(GLUT_WINDOW_WIDTH);
    int height = glutGet(GLUT_WINDOW_HEIGHT);

    // 创建一个Mat对象用于存储结果图像
    Mat resultImage(height, width, CV_8UC3);

    // 读取OpenGL渲染的像素数据
    glReadPixels(0, 0, width, height, GL_BGR_EXT, GL_UNSIGNED_BYTE, resultImage.data);

    // 翻转图像以匹配OpenCV的坐标系
    cv::flip(resultImage, resultImage, 0);

    _render_img = resultImage.clone();
}


GlobalWarping::GlobalWarping(Mat &source_img, Mat &mask, vector<Grid> rectangle_grids,
                             vector<Grid> warped_back_grids, int mesh_rows, int mesh_cols, int argc, char **argv) {
    _source_img = source_img.clone();
    _mask = mask.clone();
    _warped_back_grids = warped_back_grids;
    _rectangle_grids = rectangle_grids;
    _mesh_cols = mesh_cols;
    _mesh_rows = mesh_rows;
    _rectangle_width = _source_img.cols;
    _rectangle_height = _source_img.rows;
    _argc = argc;
    _argv = argv;
    generateCoordinates();
    optimizeEnergyFunction();
    openGLRender();
}

void GlobalWarping::optimizeEnergyFunction() {
    //最小化能量函数：即求解一个二次型的最小值点问题
    //二次型 Q(x) 表示为标准形式：Q(x) = x^T * A * x + b^T * x + c
    //x = Matrix[x_num,1] x_num: 变量个数

    Eigen::MatrixXd energy_shape_matrix_A((_mesh_cols + 1) * (_mesh_rows + 1) * 2, (_mesh_cols + 1) * (_mesh_rows + 1) * 2);
    energy_shape_matrix_A.setZero();
    calculateShapeEnergy(energy_shape_matrix_A);

    Eigen::MatrixXd energy_boundary_matrix_A((_mesh_cols + 1) * (_mesh_rows + 1) * 2, (_mesh_cols + 1) * (_mesh_rows + 1) * 2);
    Eigen::VectorXd energy_boundary_matrix_b((_mesh_cols + 1) * (_mesh_rows + 1) * 2);
    energy_boundary_matrix_A.setZero();
    energy_boundary_matrix_b.setZero();
    calculateBoundaryEnergy(energy_boundary_matrix_A,energy_boundary_matrix_b);


    Eigen::MatrixXd energy_matrix_A((_mesh_cols + 1) * (_mesh_rows + 1) * 2, (_mesh_cols + 1) * (_mesh_rows + 1) * 2);
    Eigen::VectorXd energy_matrix_b((_mesh_cols + 1) * (_mesh_rows + 1) * 2);
    energy_matrix_A = energy_shape_matrix_A + _lambda_B*energy_boundary_matrix_A;
    energy_matrix_b = _lambda_B*energy_boundary_matrix_b;

    Eigen::VectorXd x = energy_matrix_A.colPivHouseholderQr().solve(-0.5*energy_matrix_b);
    getOptimizedGridsFromX(x);
    for(int i = 0; i < _iter_times; i++){
        //iter without line constraints

    }


}

void GlobalWarping::generateCoordinates() {
    for(int i = 0; i <= _mesh_rows; i++){
        vector<Point2i> col_coordinates;
        for(int j = 0; j <= _mesh_cols; j++){
            if(i == _mesh_rows){
                if(j == _mesh_cols) col_coordinates.emplace_back(_warped_back_grids[(i-1)*_mesh_cols+j-1].bottom_right);
                else col_coordinates.emplace_back(_warped_back_grids[(i-1)*_mesh_cols+j].bottom_left);
            }
            else{
                if(j == _mesh_cols) col_coordinates.emplace_back(_warped_back_grids[i*_mesh_cols+j-1].top_right);
                else col_coordinates.emplace_back(_warped_back_grids[i*_mesh_cols+j].top_left);
            }
        }
        _warped_back_coordinates.emplace_back(col_coordinates);
    }

#ifdef GLOBAL_SHOW
//    Mat paint = _source_img.clone();
//    for(auto points : _warped_back_coordinates){
//        for(auto point : points){
//            circle(paint,point,1,Scalar(0,255,0));
//        }
//    }
//    namedWindow("paint_global",WINDOW_NORMAL);
//    imshow("paint_global",paint);
//    waitKey(0);
#endif

}

void GlobalWarping::calculateShapeEnergy(Eigen::MatrixXd &shape_matrix_A) {
    for(int row = 0; row < _mesh_rows; row++){
        for(int col = 0; col < _mesh_cols; col++){
            Eigen::MatrixXd current_quad_coff_Matrix((_mesh_cols+1) * (_mesh_rows+1) * 2, (_mesh_cols+1) * (_mesh_rows+1) * 2);
            current_quad_coff_Matrix.setZero();
            Eigen::MatrixXd Aq(8,4);
            Point2i top_left = _warped_back_grids[row*_mesh_cols + col].top_left;
            Point2i top_right = _warped_back_grids[row*_mesh_cols + col].top_right;
            Point2i bottom_left = _warped_back_grids[row*_mesh_cols + col].bottom_left;
            Point2i bottom_right = _warped_back_grids[row*_mesh_cols + col].bottom_right;
            Aq << top_left.x, -top_left.y, 1, 0,
                top_left.y, top_left.x, 0, 1,
                top_right.x, -top_right.y, 1, 0,
                top_right.y, top_right.x, 0, 1,
                bottom_left.x, -bottom_left.y, 1, 0,
                bottom_left.y, bottom_left.x, 0, 1,
                bottom_right.x, -bottom_right.y, 1, 0,
                bottom_right.y, bottom_right.x, 0, 1;

            Eigen::MatrixXd I = Eigen::MatrixXd::Identity(8, 8);
            Eigen::MatrixXd Aq_transpose = Aq.transpose();
            Eigen::MatrixXd Aq_trans_mul_Aq_reverse = (Aq_transpose * Aq).inverse();
            Eigen::MatrixXd Bq = (Aq * (Aq_trans_mul_Aq_reverse) * Aq_transpose - I);
            Eigen::MatrixXd Bq_transpose = Bq.transpose();
            Eigen::MatrixXd Coff = Bq_transpose * Bq;

            vector<int> index(8,0);
            for(int i = 0; i < 8; i++){
                if(i < 4) index[i] = 2*row*(_mesh_cols+1)+col*2+i;
                else index[i] = 2*(row+1)*(_mesh_cols+1)+col*2+(i-4);
            }
            for(int i = 0; i < 8; i++){
                for(int j = 0; j < 8; j++){
                    current_quad_coff_Matrix(index[i],index[j]) = Coff(i,j);
                }
            }
            shape_matrix_A += current_quad_coff_Matrix;
        }
    }

//    for(int i = 0; i < energy_matrix.rows(); i++){
//        for(int j = 0; j < energy_matrix.cols(); j++){
//            cout<<fixed<<setprecision(1)<<energy_matrix(i,j)<<" ";
//        }
//        cout<<endl;
//    }
//    waitKey(0);
}

void GlobalWarping::calculateBoundaryEnergy(Eigen::MatrixXd &boundary_matrix_A, Eigen::VectorXd &boundary_vector_b) {
    //Left
    for(int row = 0; row <= _mesh_rows; row++){
        Eigen::MatrixXd current_point_Matrix_A((_mesh_cols + 1) * (_mesh_rows + 1) * 2, (_mesh_cols + 1) * (_mesh_rows + 1) * 2);
        current_point_Matrix_A.setZero();
        int index = 2*row*(_mesh_cols+1);
        current_point_Matrix_A(index, index) = 1;
        boundary_matrix_A += current_point_Matrix_A;
    }
    //Right
    for(int row = 0; row <= _mesh_rows; row++){
        Eigen::MatrixXd current_point_Matrix_A((_mesh_cols + 1) * (_mesh_rows + 1) * 2, (_mesh_cols + 1) * (_mesh_rows + 1) * 2);
        current_point_Matrix_A.setZero();
        int index = 2*row*(_mesh_cols+1) + _mesh_cols*2;
        current_point_Matrix_A(index, index) = 1;
        boundary_matrix_A += current_point_Matrix_A;

        Eigen::VectorXd current_point_Matrix_b((_mesh_cols + 1) * (_mesh_rows + 1) * 2);
        current_point_Matrix_b.setZero();
        current_point_Matrix_b(index) = -2*(_rectangle_width-1);
        boundary_vector_b += current_point_Matrix_b;
    }
    //Top
    for(int col = 0; col <= _mesh_cols; col++){
        Eigen::MatrixXd current_point_Matrix_A((_mesh_cols + 1) * (_mesh_rows + 1) * 2, (_mesh_cols + 1) * (_mesh_rows + 1) * 2);
        current_point_Matrix_A.setZero();
        int index = 2*col + 1;
        current_point_Matrix_A(index, index) = 1;
        boundary_matrix_A += current_point_Matrix_A;
    }
    //Bottom
    for(int col = 0; col <= _mesh_cols; col++){
        Eigen::MatrixXd current_point_Matrix_A((_mesh_cols + 1) * (_mesh_rows + 1) * 2, (_mesh_cols + 1) * (_mesh_rows + 1) * 2);
        current_point_Matrix_A.setZero();
        int index = 2*col + 1 + _mesh_rows*(_mesh_cols+1)*2;
        current_point_Matrix_A(index, index) = 1;
        boundary_matrix_A += current_point_Matrix_A;

        Eigen::VectorXd current_point_Matrix_b((_mesh_cols + 1) * (_mesh_rows + 1) * 2);
        current_point_Matrix_b.setZero();
        current_point_Matrix_b(index) = -2*(_rectangle_height-1);
        boundary_vector_b += current_point_Matrix_b;
    }

//    for(int i = 0; i < boundary_vector_b.size(); i++){
//        cout<<boundary_vector_b(i)<<endl;
//    }
}

void GlobalWarping::getOptimizedGridsFromX(Eigen::VectorXd &X) {
    for(int row = 0; row <= _mesh_rows; row++){
        vector<Point2i> col_points;
        for(int col = 0; col <= _mesh_cols; col++){
            int index = 2*row*(_mesh_cols+1) + 2*col;
            col_points.emplace_back(X[index],X[index+1]);
        }
        _optimized_coordinates.push_back(col_points);
    }

    for(int row = 0; row < _mesh_rows; row++){
        for(int col = 0; col < _mesh_cols; col++){
            Point2i top_left, top_right, bottom_left, bottom_right;
            top_left = _optimized_coordinates[row][col];
            top_right = _optimized_coordinates[row][col+1];
            bottom_left = _optimized_coordinates[row+1][col];
            bottom_right = _optimized_coordinates[row+1][col+1];
            _optimized_grids.emplace_back(top_left,top_right,bottom_right,bottom_left);
        }
    }
    verifyOptimizedGrids();
    drawGrids(_optimized_grids, "optimized_grids", _source_img, true);
}

void GlobalWarping::verifyOptimizedGrids() {
    for(int col = 0; col < _mesh_cols; col++){
        if(_optimized_grids[(_mesh_rows-1)*_mesh_cols+col].bottom_left.y != _source_img.rows - 1)
        {
            _optimized_grids[(_mesh_rows-1)*_mesh_cols+col].bottom_left.y = _source_img.rows - 1;
            _optimized_coordinates[_mesh_rows][col].y = _source_img.rows - 1;
        }
        if(_optimized_grids[(_mesh_rows-1)*_mesh_cols+col].bottom_right.y != _source_img.rows - 1)
        {
            _optimized_grids[(_mesh_rows-1)*_mesh_cols+col].bottom_right.y = _source_img.rows - 1;
            _optimized_coordinates[_mesh_rows][col+1].y = _source_img.rows - 1;
        }
    }
}

void GlobalWarping::openGLRender() {
    initGlobal();
    // 初始化OpenGL和GLUT
//    int argc = 0;
    glutInit(&_argc, _argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(g_source_img.cols, g_source_img.rows);
    glutCreateWindow("Texture Mapping");
//    GLenum err = glewInit();
//    if (GLEW_OK != err)
//    {
//        std::cerr << "GLEW Error: " << glewGetErrorString(err) << std::endl;
//        return;
//    }
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    // 加载纹理
    loadTexture();
    glutDisplayFunc(display);

    // 开始主循环
    glutMainLoop();

    // 保存结果图像
//    saveResultImage();

}

void GlobalWarping::initGlobal() {
    g_source_img = _source_img.clone();
    g_mesh_cols = _mesh_cols;
    g_mesh_rows = _mesh_rows;
    g_optimized_grids = _optimized_grids;
    g_warped_back_grids = _warped_back_grids;
}



