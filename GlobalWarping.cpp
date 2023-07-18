//
// Created by nuc on 23-7-16.
//

#include "GlobalWarping.h"



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
//    openGLRender();
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


void GlobalWarping::getOptimizedGrids(vector<Grid> &optimized_grids) {
    optimized_grids = _optimized_grids;
}



