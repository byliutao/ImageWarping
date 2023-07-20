//
// Created by nuc on 23-7-16.
//

#include "GlobalWarping.h"



GlobalWarping::GlobalWarping(Mat &source_img, Mat &mask, vector<Grid> rectangle_grids, vector<Grid> warped_back_grids,
                             int mesh_rows, int mesh_cols) {
    _source_img = source_img.clone();
    _mask = mask.clone();
    _warped_back_grids = warped_back_grids;
    _rectangle_grids = rectangle_grids;
    _mesh_cols = mesh_cols;
    _mesh_rows = mesh_rows;
    _rectangle_width = _source_img.cols;
    _rectangle_height = _source_img.rows;
    _N = mesh_cols * mesh_rows;
    for(int i = 0; i < _M; i++){
        double theta = -M_PI/2.0 + (i / 50.0) * M_PI;
        _theta_bins.emplace_back(theta,0);
    }
    generateCoordinates();
#ifndef GLOBAL_WITHOUT_LINE
    lineDetect();
#endif
    optimizeEnergyFunction();
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
#ifdef GLOBAL_WITHOUT_LINE
    // without line constrain
    energy_matrix_A =  energy_shape_matrix_A + _lambda_B * energy_boundary_matrix_A;
    energy_matrix_b = _lambda_B * energy_boundary_matrix_b;

    Eigen::VectorXd x = energy_matrix_A.colPivHouseholderQr().solve(-0.5*energy_matrix_b);
    getUpdatedGridsFromX(x,_optimized_grids);
#else
    Eigen::MatrixXd energy_line_matrix_A((_mesh_cols + 1) * (_mesh_rows + 1) * 2, (_mesh_cols + 1) * (_mesh_rows + 1) * 2);
    vector<Grid> updated_grids;
    double start = cv::getTickCount();
    for(int iter = 0; iter < _iter_times; iter++){
        double t1 = cv::getTickCount();
        energy_line_matrix_A.setZero();
        updated_grids.clear();
        calculateLineEnergy(energy_line_matrix_A);

        // Fix {theta_m} update V
        energy_matrix_A = energy_shape_matrix_A + _lambda_L * energy_line_matrix_A + _lambda_B * energy_boundary_matrix_A;
        energy_matrix_b = _lambda_B * energy_boundary_matrix_b;
        Eigen::VectorXd x = energy_matrix_A.colPivHouseholderQr().solve(-0.5*energy_matrix_b);
        getUpdatedGridsFromX(x, updated_grids);

        // Fix V update {theta_m}
        updateThetaMByV(updated_grids);
        double t2 = cv::getTickCount();
//        cout<<"iter"<<iter<<"_consume_time: "<< (t2 - t1) / cv::getTickFrequency() * 1000 << "ms "<<endl;
#ifdef GLOBAL_SHOW_STEP
        drawGrids(updated_grids,"updated_grids",_source_img,true);
#endif
    }
    double end = cv::getTickCount();
    cout<<"total_consume_time: "<< (end - start) / cv::getTickFrequency() * 1000 << "ms "<<endl;

    _optimized_grids = updated_grids;
#endif

#ifdef GLOBAL_SHOW
    drawGrids(_optimized_grids,"optimized_grids",_source_img,true);
#endif

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

            //calculate quad related index of Coff_Matrix_A
            vector<int> index(8,0);
            for(int i = 0; i < 8; i++){
                if(i < 4) index[i] = 2*row*(_mesh_cols+1)+col*2+i;
                else index[i] = 2*(row+1)*(_mesh_cols+1)+col*2+(i-4);
            }
            for(int i = 0; i < 8; i++){
                for(int j = 0; j < 8; j++){
                    shape_matrix_A(index[i],index[j]) += Coff(i,j);
                }
            }
        }
    }

    shape_matrix_A /= (double)_N;
}

void GlobalWarping::calculateBoundaryEnergy(Eigen::MatrixXd &boundary_matrix_A, Eigen::VectorXd &boundary_vector_b) {
    //Left
    for(int row = 0; row <= _mesh_rows; row++){
        int index = 2*row*(_mesh_cols+1);
        boundary_matrix_A(index, index) += 1;
    }
    //Right
    for(int row = 0; row <= _mesh_rows; row++){
        int index = 2*row*(_mesh_cols+1) + _mesh_cols*2;
        boundary_matrix_A(index, index) += 1;
        boundary_vector_b(index) += -2*(_rectangle_width-1);
    }
    //Top
    for(int col = 0; col <= _mesh_cols; col++){
        int index = 2*col + 1;
        boundary_matrix_A(index, index) += 1;
    }
    //Bottom
    for(int col = 0; col <= _mesh_cols; col++){
        int index = 2*col + 1 + _mesh_rows*(_mesh_cols+1)*2;
        boundary_matrix_A(index, index) += 1;
        boundary_vector_b(index) += -2*(_rectangle_height-1);
    }
}

void GlobalWarping::getUpdatedGridsFromX(Eigen::VectorXd &X, vector<Grid> &grids_of_mesh) {
    vector<vector<Point2i>> coordinates_of_mesh;
    for(int row = 0; row <= _mesh_rows; row++){
        vector<Point2i> col_points;
        for(int col = 0; col <= _mesh_cols; col++){
            int index = 2*row*(_mesh_cols+1) + 2*col;
            col_points.emplace_back(X[index],X[index+1]);
        }
        coordinates_of_mesh.push_back(col_points);
    }

    for(int row = 0; row < _mesh_rows; row++){
        for(int col = 0; col < _mesh_cols; col++){
            Point2i top_left, top_right, bottom_left, bottom_right;
            top_left = coordinates_of_mesh[row][col];
            top_right = coordinates_of_mesh[row][col+1];
            bottom_left = coordinates_of_mesh[row+1][col];
            bottom_right = coordinates_of_mesh[row+1][col+1];
            grids_of_mesh.emplace_back(top_left,top_right,bottom_right,bottom_left);
        }
    }
    verifyGrids(grids_of_mesh);
}

void GlobalWarping::verifyGrids(vector<Grid> &grids_of_mesh) {
    for(int col = 0; col < _mesh_cols; col++){
        if(grids_of_mesh[(_mesh_rows-1)*_mesh_cols+col].bottom_left.y != _source_img.rows - 1)
        {
            grids_of_mesh[(_mesh_rows-1)*_mesh_cols+col].bottom_left.y = _source_img.rows - 1;
//            _optimized_coordinates[_mesh_rows][col].y = _source_img.rows - 1;
        }
        if(grids_of_mesh[(_mesh_rows-1)*_mesh_cols+col].bottom_right.y != _source_img.rows - 1)
        {
            grids_of_mesh[(_mesh_rows-1)*_mesh_cols+col].bottom_right.y = _source_img.rows - 1;
//            _optimized_coordinates[_mesh_rows][col+1].y = _source_img.rows - 1;
        }
    }
}


void GlobalWarping::getOptimizedGrids(vector<Grid> &optimized_grids) {
    optimized_grids = _optimized_grids;
}

void GlobalWarping::lineDetect() {
    Mat gray_source_img;
    cvtColor(_source_img,gray_source_img,COLOR_BGR2GRAY);

    double* input_image = new double[gray_source_img.rows * gray_source_img.cols];
    double* output;
    // 将灰度图像的像素值复制到double类型数组中
    for (int i = 0; i < gray_source_img.rows; i++)
    {
        for (int j = 0; j < gray_source_img.cols; j++)
        {
            input_image[i * gray_source_img.cols + j] = static_cast<double>(gray_source_img.at<uchar>(i, j));
        }
    }

    int line_num;
    vector<pair<Point2i,Point2i>> lines;
    vector<vector<pair<Point2i,Point2i>>> lines_of_mesh(_warped_back_grids.size());
    output = lsd(&line_num,input_image,gray_source_img.cols,gray_source_img.rows);
    printf("%d line segments found\n",line_num);
    for(int i = 0; i < line_num; i++)
    {
        lines.emplace_back(Point2i((int)output[7*i],(int)output[7*i+1]),Point2i((int)output[7*i+2],(int)output[7*i+3]));
    }
    // cut the line using the quad's edge, and push them into their belong quads
    for(auto line : lines){
        for(int row = 0; row < _mesh_rows; row++){
            for(int col = 0; col < _mesh_cols; col++){
                Grid grid = _warped_back_grids[row*_mesh_cols+col];
                vector<Point2i> region = {grid.top_right, grid.top_left, grid.bottom_left, grid.bottom_right};
                if(isInsideGrid(line.first,grid) >= 0 && isInsideGrid(line.second,grid) >= 0){
                    //both inside the quad
                    lines_of_mesh[row*_mesh_cols+col].emplace_back(line);
                }
                else if(isInsideGrid(line.first,grid) >= 0 && isInsideGrid(line.second,grid) < -1){
                    //one inside one outside the quad
                    bool is_find_intersect = false;
                    int between_point_num = max(abs(line.first.x - line.second.x), abs(line.first.y - line.second.y)) - 1;
                    Point2f start_point = line.first;
                    Point2f end_point = line.second;
                    Point2f direction_vector = end_point - start_point;
                    for(int i = 1; i <= between_point_num; i++){
                        Point2f point = start_point + i * direction_vector / (between_point_num + 1);
                        double dis_to_start = cv::norm(point-start_point);
                        if(abs(isInsideGrid(point,grid)) < 1 && dis_to_start > 5){
                            end_point = Point2i(point);
                            is_find_intersect = true;
                            break;
                        }
                    }
                    if(is_find_intersect) lines_of_mesh[row*_mesh_cols+col].emplace_back(start_point,end_point);
                }
                else if(isInsideGrid(line.first,grid) < -1 && isInsideGrid(line.second,grid) >= 0){
                    //one inside one outside the quad
                    bool is_find_intersect = false;
                    int between_point_num = max(abs(line.first.x - line.second.x), abs(line.first.y - line.second.y)) - 1;
                    Point2f start_point = line.second;
                    Point2f end_point = line.first;
                    Point2f direction_vector = end_point - start_point;
                    for(int i = 1; i <= between_point_num; i++){
                        Point2f point = start_point + i * direction_vector / (between_point_num + 1);
                        double dis_to_start = cv::norm(point-start_point);
                        if(abs(isInsideGrid(point,grid)) < 1 && dis_to_start > 5){
                            end_point = Point2i(point);
                            is_find_intersect = true;
                            break;
                        }
                    }
                    if(is_find_intersect) lines_of_mesh[row*_mesh_cols+col].emplace_back(start_point,end_point);
                }
                else{
                    //both outside the quad
                    int between_point_num = max(abs(line.first.x - line.second.x), abs(line.first.y - line.second.y)) - 1;
                    Point2f result_start, result_end;
                    Point2f start_point = line.second;
                    Point2f end_point = line.first;
                    Point2f direction_vector = end_point - start_point;
                    bool find_first_intersect = false;
                    bool find_second_intersect = false;
                    for(int i = 1; i <= between_point_num; i++){
                        Point2f point = start_point + i * direction_vector / (between_point_num + 1);
                        if((abs(isInsideGrid(point,grid)) < 1) && !find_first_intersect){
                            result_start = Point2i(point);
                            find_first_intersect = true;
                        }
                        if(find_first_intersect){
                            double dis_to_start = norm(point-result_start);
                            if(dis_to_start > 5 && (abs(isInsideGrid(point,grid)) < 1)){
                                result_end = Point2i(point);
                                find_second_intersect = true;
                                break;
                            }
                        }
                    }
                    if(find_first_intersect && find_second_intersect){
                        lines_of_mesh[row*_mesh_cols+col].emplace_back(result_start,result_end);
                    }
                }
            }
        }
    }

    _lines_of_mesh = lines_of_mesh;
    for(int row = 0; row < _mesh_rows; row++){
        for(int col = 0; col < _mesh_cols; col++){
            vector<pair<Point2i,Point2i>> quad_lines = _lines_of_mesh[row*_mesh_cols+col];
            vector<int> lines_angle_index_of_quad;
            for(int i = 0; i < quad_lines.size(); i++){
                bool find = false;
                pair<Point2d,Point2d> line = quad_lines[i];
                double theta = atan((line.first.y-line.second.y)/(line.first.x-line.second.x));
                for(int j = 0; j < _M; j++){
                    double min = _theta_bins[j].first;
                    double max = _theta_bins[j].first + ((j+1) / 50.0) * M_PI;
                    if(theta >= min && theta < max){
                        lines_angle_index_of_quad.emplace_back(j);
                        find = true;
                        break;
                    }
                }
                CV_Assert(find);
            }
            _lines_bin_index_of_mesh.emplace_back(lines_angle_index_of_quad);
        }
    }

#ifdef GLOBAL_SHOW
    Mat paint = _source_img.clone();
    for(auto pair : lines){
        cv::line(paint,pair.first,pair.second,Scalar(0,255,0),1);
    }
    namedWindow("line_detect_result",WINDOW_NORMAL);
    imshow("line_detect_result",paint);


    Mat paint__ = _source_img.clone();
    for(int row = 0; row < _mesh_rows; row++){
        for(int col = 0; col < _mesh_cols; col++){
            Grid grid = _warped_back_grids[row*_mesh_cols+col];
            vector<pair<Point2i,Point2i>> quad_lines = _lines_of_mesh[row*_mesh_cols+col];
            for(auto pair : quad_lines){
                cv::line(paint__,pair.first,pair.second,Scalar(0,255,0),2);
            }
            cv::line(paint__, grid.top_left, grid.top_right, cv::Scalar(255, 0, 0),2);
            cv::line(paint__, grid.top_right, grid.bottom_right, cv::Scalar(255, 0, 0), 2);
            cv::line(paint__, grid.bottom_right, grid.bottom_left, cv::Scalar(255, 0, 0), 2);
            cv::line(paint__, grid.bottom_left, grid.top_left, cv::Scalar(255, 0, 0), 2);
        }
    }
    namedWindow("line_detect_quad",WINDOW_NORMAL);
    imshow("line_detect_quad",paint__);
    waitKey(0);
#endif

#ifdef GLOBAL_SHOW_STEP
    Mat paint_ = _source_img.clone();
    for(int row = 0; row < _mesh_rows; row++){
        for(int col = 0; col < _mesh_cols; col++){
            Grid grid = _warped_back_grids[row*_mesh_cols+col];
            vector<pair<Point2i,Point2i>> quad_lines = _lines_of_mesh[row*_mesh_cols+col];
            for(auto pair : quad_lines){
                cv::line(paint_,pair.first,pair.second,Scalar(0,255,0),2);
            }
            cv::line(paint_, grid.top_left, grid.top_right, cv::Scalar(255, 0, 0),2);
            cv::line(paint_, grid.top_right, grid.bottom_right, cv::Scalar(255, 0, 0), 2);
            cv::line(paint_, grid.bottom_right, grid.bottom_left, cv::Scalar(255, 0, 0), 2);
            cv::line(paint_, grid.bottom_left, grid.top_left, cv::Scalar(255, 0, 0), 2);
            namedWindow("line_detect_quad",WINDOW_NORMAL);
            imshow("line_detect_quad",paint_);
            waitKey(0);
        }
    }
#endif
}

void GlobalWarping::calculateLineEnergy(Eigen::MatrixXd &line_matrix_A) {
    _N_L = 0;

    for(int row = 0; row < _mesh_rows; row++){
        for(int col = 0; col < _mesh_cols; col++){
            int grid_index = row*_mesh_cols+col;
            Grid grid = _warped_back_grids[grid_index];
            vector<pair<Point2i,Point2i>> quad_lines = _lines_of_mesh[grid_index];
            vector<int> quad_bins_index = _lines_bin_index_of_mesh[grid_index];
            for(int line_index = 0; line_index < quad_lines.size(); line_index++){

                pair<Point2i,Point2i> line = quad_lines[line_index];


                pair<double,double> w1;
                pair<double,double> w2;
                bool w1_flag = getInvBilinearWeight(line.first, grid, w1);
                bool w2_flag = getInvBilinearWeight(line.second, grid, w2);

                if(w1_flag && w2_flag){
                    _N_L++;
                    Eigen::MatrixXd e_hat(2,1);
                    e_hat << line.first.x-line.second.x, line.first.y-line.second.y;

                    Eigen::MatrixXd R(2,2);
                    int index_theta = quad_bins_index[line_index];
                    double theta_m = _theta_bins[index_theta].second;
                    R << cos(theta_m), -sin(theta_m),
                            sin(theta_m), cos(theta_m);

                    Eigen::MatrixXd inverse_tmp = (e_hat.transpose()*e_hat).inverse();
                    Eigen::MatrixXd C_mat = R * e_hat * inverse_tmp * ( e_hat.transpose() )*( R.transpose() ) - Eigen::Matrix2d::Identity();


                    Eigen::MatrixXd start_mat = bilinearWeightsToMatrix(w1);
                    Eigen::MatrixXd end_mat = bilinearWeightsToMatrix(w2);
                    Eigen::MatrixXd difference_mat = end_mat - start_mat;
                    Eigen::MatrixXd Coff = difference_mat.transpose()*C_mat.transpose()*C_mat*difference_mat;

                    vector<int> index(8,0);
                    for(int i = 0; i < 8; i++){
                        if(i < 4) index[i] = 2 * row * (_mesh_cols + 1) + col * 2 + i;
                        else index[i] = 2 * (row + 1) * (_mesh_cols + 1) + col * 2 + (i - 4);
                    }
                    for(int i = 0; i < 8; i++){
                        for(int j = 0; j < 8; j++){
                            line_matrix_A(index[i], index[j]) += Coff(i,j);
                        }
                    }
                }
            }
        }
    }

    line_matrix_A /= (double)_N_L;
}

void GlobalWarping::updateThetaMByV(const vector<Grid> &updated_grids) {
    //update _lines_bin_index_of_mesh
    vector<vector<double>> lines_angle_of_bin(_M);
    for(int row = 0; row < _mesh_rows; row++){
        for(int col = 0; col < _mesh_cols; col++){
            int quad_index= row*_mesh_cols+col;
            Grid original_grid = _warped_back_grids[quad_index];
            Grid updated_grid = updated_grids[quad_index];
            vector<pair<Point2i,Point2i>> lines = _lines_of_mesh[quad_index];
            vector<int> lines_bin_index = _lines_bin_index_of_mesh[quad_index];
            for(int i = 0; i < lines.size(); i++){
                int bin_index = lines_bin_index[i];
                if(bin_index != 0){
                    int a = 0;
                }
                pair<Point2d,Point2d> original_line = lines[i];
                pair<Point2d,Point2d> updated_line;
                double original_theta = atan((original_line.first.y - original_line.second.y) / (original_line.first.x - original_line.second.x));

                pair<double,double> w1;
                pair<double,double> w2;
                bool w1_flag = getInvBilinearWeight(original_line.second, original_grid, w1);
                bool w2_flag = getInvBilinearWeight(original_line.first, original_grid, w2);
                if(w1_flag && w2_flag){
                    double u1 = w1.first, v1 = w1.second, u2 = w2.first, v2 = w2.second;
                    Point2d A = updated_grid.top_left, B = updated_grid.top_right, C = updated_grid.bottom_right, D = updated_grid.bottom_left;
                    updated_line.first = A + (B - A) * u1 + (D - A) * v1 + (A - B + C - D) * u1 * v1;
                    updated_line.second = A + (B - A) * u2 + (D - A) * v2 + (A - B + C - D) * u2 * v2;
                    double updated_theta = atan((updated_line.first.y - updated_line.second.y) / (updated_line.first.x - updated_line.second.x));
                    double delta_theta = updated_theta - original_theta;
                    if (delta_theta > (M_PI / 2)) {
                        delta_theta -= M_PI;
                    }
                    if (delta_theta < (-M_PI / 2)) {
                        delta_theta += M_PI;
                    }

                    lines_angle_of_bin[bin_index].emplace_back(delta_theta);
                }
            }
        }
    }

    for(int i = 0; i < _M; i++){
        if(lines_angle_of_bin[i].empty()) continue;
        double total_delta_angle = 0;
        for(auto delta_angle : lines_angle_of_bin[i]) total_delta_angle += delta_angle;
        double avg_delta_angle = total_delta_angle / (double)lines_angle_of_bin[i].size();
        _theta_bins[i].second = avg_delta_angle;
    }
//    int a = 0;
}



