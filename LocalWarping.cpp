//
// Created by nuc on 23-7-14.
//

#include "LocalWarping.h"

void LocalWarping::show_displacement_map(vector<vector<Point2i>> displacement_map, string window_name, bool is_wait) {
    int cols_num = displacement_map[0].size();
    int rows_num = displacement_map.size();

    Mat x_map(Size(cols_num,rows_num),CV_32SC1);
    Mat y_map(Size(cols_num,rows_num),CV_32SC1);

    for(int row = 0; row < rows_num; row++){
        for(int col = 0; col < cols_num; col++){
            int x_value = displacement_map.at(row)[col].x;
            int y_value = displacement_map.at(row)[col].y;
            x_map.at<int>(row,col) = x_value;
            y_map.at<int>(row,col) = y_value;
        }
    }

    double minValueX, maxValueX, minValueY, maxValueY;
    cv::minMaxLoc(x_map, &minValueX, &maxValueX);
    cv::minMaxLoc(y_map, &minValueY, &maxValueY);

    // 计算映射的比例因子
    double scaleX = 255.0 / (maxValueX - minValueX);
    double scaleY = 255.0 / (maxValueY - minValueY);

    // 创建 CV_8UC1 类型的图像
    cv::Mat ucharXImage(x_map.size(), CV_8UC1);
    cv::Mat ucharYImage(y_map.size(), CV_8UC1);

    // 对矩阵中的每个元素进行映射
    for (int row = 0; row < x_map.rows; ++row) {
        for (int col = 0; col < x_map.cols; ++col) {
            int x_value = x_map.at<int>(row, col);
            int y_value = y_map.at<int>(row, col);
            ucharXImage.at<uchar>(row, col) = static_cast<uchar>((x_value - minValueX) * scaleX);
            ucharYImage.at<uchar>(row, col) = static_cast<uchar>((y_value - minValueY) * scaleY);
        }
    }

    namedWindow("ImageX_"+window_name,WINDOW_NORMAL);
    namedWindow("ImageY_"+window_name,WINDOW_NORMAL);
    cv::imshow("ImageX_"+window_name, ucharXImage);
    cv::imshow("ImageY_"+window_name,ucharYImage);
    if(is_wait) cv::waitKey(0);
    else waitKey(1);
}


LocalWarping::LocalWarping(Mat &source_img, Mat &mask) {
    _source_img = source_img.clone();
    _mask = mask.clone();
    _local_wraping_img = source_img.clone();
    for(int row = 0; row < _source_img.rows; row++){
        vector<Point2i> col_points(_source_img.cols,Point2i(0,0));
        _displacement_map.emplace_back(col_points);
    }

    vector<bool> position_flag(4);
    for(int i = 0; i < 4; i++) position_flag[i] = true;
    while(true){
        Position position;
        getTheBiggestPosition(position, position_flag);
//        position = Bottom;
        position_flag[position] = imageShift(position);
//        for(int i = 0; i < 4; i++){
//            position_flag[i] = imageShift(Position(i));
//        }
        if(!position_flag[0] && !position_flag[1] && !position_flag[2] && !position_flag[3]) break;

    }
#ifdef LOCAL_SHOW
    draw_all_seams();
#endif
}

Rect LocalWarping::getSubImageRect(Position position) {
    Rect roi;
    // 获取图像宽度和高度
    int width = _mask.cols;
    int height = _mask.rows;

    if(position == Top || position == Bottom){
        int longestLength = 0;  // 最长连续像素点集合的长度
        int startPoint = -1;  // 起点位置
        int endPoint = -1;    // 终点位置
        int currentStart = -1;  // 当前连续像素点集合的起点
        int currentLength = 0;  // 当前连续像素点集合的长度

        uchar* rowData;

        if(position == Top){
            rowData = _mask.ptr<uchar>(0);
        }
        else{
            rowData = _mask.ptr<uchar>(height-1);
        }
        for (int i = 0; i < width; ++i)
        {
            if (rowData[i] == 255)  // 当前像素为白色(255)
            {
                if (currentStart == -1)  // 当前没有连续像素点集合，设置起点
                    currentStart = i;

                currentLength++;  // 增加连续像素点集合的长度

                // 更新最长连续像素点集合的信息
                if (currentLength > longestLength)
                {
                    longestLength = currentLength;
                    startPoint = currentStart;
                    endPoint = i;
                }
            }
            else  // 当前像素为黑色(0)
            {
                currentStart = -1;  // 重置连续像素点集合的起点
                currentLength = 0;  // 重置连续像素点集合的长度
            }
        }
        roi = Rect(startPoint,0,longestLength,height);
    }
    else{
        int longestLength = 0;  // 最长连续像素点集合的长度
        int startPoint = -1;  // 起点位置
        int endPoint = -1;    // 终点位置
        int currentStart = -1;  // 当前连续像素点集合的起点
        int currentLength = 0;  // 当前连续像素点集合的长度

        int col_index;
        if(position == Left){
            col_index = 0;
        }
        else{
            col_index = width - 1;
        }
        // 遍历最左列的像素
        for (int row_index = 0; row_index < height; ++row_index)
        {
            if (_mask.at<uchar>(row_index, col_index) == 255)  // 当前像素为白色(255)
            {
                if (currentStart == -1)  // 当前没有连续像素点集合，设置起点
                    currentStart = row_index;

                currentLength++;  // 增加连续像素点集合的长度

                // 更新最长连续像素点集合的信息
                if (currentLength > longestLength)
                {
                    longestLength = currentLength;
                    startPoint = currentStart;
                    endPoint = row_index;
                }
            }
            else  // 当前像素为黑色(0)
            {
                currentStart = -1;  // 重置连续像素点集合的起点
                currentLength = 0;  // 重置连续像素点集合的长度
            }
        }
        roi = Rect(0,startPoint,width,longestLength);
    }

    return roi;
}

void LocalWarping::getEmptyPositionMask(Mat &input, Mat &output) {
    int width = input.cols;
    int height = input.rows;

    // 获取图像数据指针
    uchar* rgbData = input.data;
    uchar* binaryData = output.data;

    // 遍历图像数据
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            // 计算当前像素的索引
            int index = i * width + j;

            // 获取当前像素的RGB值
            uchar r = rgbData[3 * index];
            uchar g = rgbData[3 * index + 1];
            uchar b = rgbData[3 * index + 2];

            // 判断RGB值是否为(255, 255, 255)，若是则在二值图像中设为白色(255)，否则设为黑色(0)
            if (r == 255 && g == 255 && b == 255)
                binaryData[index] = 255;
            else
                binaryData[index] = 0;
        }
    }
}

bool LocalWarping::imageShift(Position position) {
    Rect roi = getSubImageRect(position);
    if(roi.width == 0 || roi.height == 0) return false;

    Mat image_roi = Mat(this->_local_wraping_img, roi);
    Mat mask_roi = Mat(this->_mask,roi);
    Direction direction;
    vector<Point2i> seam;

    if(position == Left || position == Right) direction = UpToDown;
    else direction = LeftToRight;

    calculateSeam(image_roi, mask_roi, direction, seam, roi);
    insertSeamAndUpdateDisplacementMap(position, seam);


    return true;
}





void LocalWarping::calculateSeam(Mat &src, Mat &mask, Direction direction, vector<Point2i> &seam, Rect roi) {
    Mat cost_image(src.size(), CV_32FC1);
    calculateCostImage(src, cost_image, mask);

    Mat path_map;
    if(direction == LeftToRight){
        cv::rotate(cost_image, cost_image, cv::ROTATE_90_CLOCKWISE);
    }


    path_map = Mat(cost_image.size(), CV_32SC1);
    //M(i,j)=E(i,j)+min(M(i−1,j−1),M(i−1,j),M(i−1,j+1))
    for (int i = 0; i < cost_image.rows; ++i)
    {
        for (int j = 0; j < cost_image.cols; ++j)
        {
            if(i == 0){
                path_map.at<int>(i,j) = 0;
            }
            else{
                float up_left, up, up_right;
                up = cost_image.at<float>(i - 1, j);
                if(j-1 < 0){
                    up_left = std::numeric_limits<float>::max();
                }
                else{
                    up_left = cost_image.at<float>(i - 1, j - 1);
                }
                if(j+1 == cost_image.cols){
                    up_right = std::numeric_limits<float>::max();
                }
                else{
                    up_right = cost_image.at<float>(i - 1, j + 1);
                }

                if(up <= min(up_left,up_right)){
                    path_map.at<int>(i,j) = 0;
                    cost_image.at<float>(i, j) += up;
                }
                else if(up_left <= min(up,up_right)){
                    path_map.at<int>(i,j) = -1;
                    cost_image.at<float>(i, j) += up_left;
                }
                else if(up_right <= min(up,up_left)){
                    path_map.at<int>(i,j) = 1;
                    cost_image.at<float>(i, j) += up_right;
                }
                else{
                    CV_Assert(false);
                }
            }

        }
    }

    // 获取最下面一行的像素数据指针
    float* rowData = cost_image.ptr<float>(cost_image.rows - 1);

    int minIndex = 0;      // 最小值点的索引
    float minValue = rowData[0];  // 最小值点的像素值

    // 遍历最下面一行的像素，寻找最小值点的位置
    for (int i = 1; i < cost_image.cols; ++i)
    {
        float pixelValue = rowData[i];
        if (pixelValue < minValue)
        {
            minIndex = i;
            minValue = pixelValue;
        }
    }

    seam.emplace_back(minIndex,path_map.rows-1);
    int last_col_index = minIndex;
    for (int i = path_map.rows - 1; i >= 1; i--)
    {
        int move_direction =  path_map.at<int>(Point2i(last_col_index,i ));
        int col_pos = last_col_index + move_direction;
        seam.emplace_back(col_pos,i-1 );
        last_col_index = col_pos;
    }

    if(direction == LeftToRight){
        for(int i = 0; i < seam.size(); i++){
            Point2i temp_point(seam[i].y, src.rows - 1 - seam[i].x);
            seam[i] = temp_point;
        }
    }

    for(int i = 0; i < seam.size(); i++){
        seam[i].x += roi.x;
        seam[i].y += roi.y;
    }

    _all_seams.push_back(seam);

}

void LocalWarping::calculateCostImage(Mat &input_image, Mat &cost_image, Mat &mask) {

    // 转换为灰度图像
    cv::Mat grayImage;
    cv::cvtColor(input_image, grayImage, cv::COLOR_BGR2GRAY);

    // 计算梯度
    cv::Mat gradientX, gradientY;
    cv::Sobel(grayImage, gradientX, CV_32F, 1, 0,3);
    cv::Sobel(grayImage, gradientY, CV_32F, 0, 1,3);

    // 计算梯度幅值
    cv::convertScaleAbs(gradientX, gradientX);
    cv::convertScaleAbs(gradientY, gradientY);
    cv::addWeighted(gradientX, 0.5, gradientY, 0.5, 0, cost_image, CV_32FC1);

//show
//    cv::Mat dst;
//    cv::normalize(cost_image, dst, 0, 1, cv::NORM_MINMAX);
//    cv::imshow("test", dst);
//    cv::waitKey(0);
//


    // 将 mask 图像中像素值为 255 的点在cost_image中对应的值设置为 10^8
    for (int i = 0; i < cost_image.rows; ++i)
    {
        for (int j = 0; j < cost_image.cols; ++j)
        {
            uchar value = mask.at<uchar>(i,j);
            if (value == 255)
            {
                cost_image.at<float>(i, j) = 1e8;  // 设置为 10 的 8 次方
            }
        }
    }

}

void LocalWarping::insertSeamAndUpdateDisplacementMap(Position position, const vector<Point2i> &seam) {
#ifdef LOCAL_SHOW_STEP
    show_displacement_map(_displacement_map, "Image", false);

    Mat paint = _local_wraping_img.clone();
    for(int i = 0; i < seam.size(); i++){
        paint.at<Vec3b>(seam[i]) = Vec3b(0,0,255);
    }
    namedWindow("_mask",WINDOW_NORMAL);
    imshow("_mask",_mask);
    namedWindow("paint_local_wraping_img",WINDOW_NORMAL);
    imshow("paint_local_wraping_img",paint);
    waitKey(0);
#endif


    vector<vector<Point2i>> current_displacement_map = _displacement_map;

    if(position == Bottom){
        for(int i = 0; i < seam.size(); i++){
            int x = seam[i].x;
            int y = seam[i].y;
            while(_mask.at<uchar>(Point2i(x,y)) == 0){
                y++;
            }
            _mask.at<uchar>(Point2i(x,y)) = 0;
            for(int j = y; j != seam[i].y; j--){
                //x: col_index y(j): row_index
                _local_wraping_img.at<Vec3b>(Point2i(x,j)) = _local_wraping_img.at<Vec3b>(Point(x,j-1));
                _displacement_map[j][x].y = current_displacement_map[j-1][x].y - 1;
                _displacement_map[j][x].x = current_displacement_map[j-1][x].x;
            }
        }
    }
    else if(position == Top){
        for(int i = 0; i < seam.size(); i++){
            int x = seam[i].x;
            int y = seam[i].y;
            while(_mask.at<uchar>(Point2i(x,y)) == 0){
                y--;
            }
            _mask.at<uchar>(Point2i(x,y)) = 0;
            for(int j = y; j != seam[i].y; j++){
                _local_wraping_img.at<Vec3b>(Point2i(x,j)) = _local_wraping_img.at<Vec3b>(Point(x,j+1));
                _displacement_map[j][x].y = current_displacement_map[j+1][x].y + 1;
                _displacement_map[j][x].x = current_displacement_map[j+1][x].x;
            }
        }
    }
    else if(position == Right){
        for(int i = 0; i < seam.size(); i++){
            int x = seam[i].x;
            int y = seam[i].y;
            while(_mask.at<uchar>(Point2i(x,y)) == 0){
                x++;
            }
            _mask.at<uchar>(Point2i(x,y)) = 0;
            for(int j = x; j != seam[i].x; j--){
                _local_wraping_img.at<Vec3b>(Point2i(j,y)) = _local_wraping_img.at<Vec3b>(Point(j-1,y));
                _displacement_map[y][j].y = current_displacement_map[y][j-1].y;
                _displacement_map[y][j].x = current_displacement_map[y][j-1].x - 1;
            }
        }
    }
    else{
        for(int i = 0; i < seam.size(); i++){
            int x = seam[i].x;
            int y = seam[i].y;
            while(_mask.at<uchar>(Point2i(x,y)) == 0){
                x--;
            }
            _mask.at<uchar>(Point2i(x,y)) = 0;
            for(int j = x; j != seam[i].x; j++){
                _local_wraping_img.at<Vec3b>(Point2i(j,y)) = _local_wraping_img.at<Vec3b>(Point(j+1,y));
                _displacement_map[y][j].y = current_displacement_map[y][j+1].y;
                _displacement_map[y][j].x = current_displacement_map[y][j+1].x + 1;
            }
        }
    }
}

void LocalWarping::getTheBiggestPosition(LocalWarping::Position &result_position, const vector<bool> &position_flag) {
    int max_roi_length = 0;
    Position max_roi_position;
    for(int i = 0; i < 4; i++){
        if(!position_flag[i]) continue;
        max_roi_position = Position(i);
    }
    for(int i = 0; i < 4; i++){
        if(!position_flag[i]) continue;
        Position position = Position(i);
        Rect roi = getSubImageRect(position);
        int length = ((position == Top || position == Bottom) ? roi.width : roi.height);
        if(length > max_roi_length){
            max_roi_length = length;
            max_roi_position = position;
        }
    }
    result_position = max_roi_position;
}

void LocalWarping::getExpandImage(Mat &image) {
    image = _local_wraping_img.clone();
}

void LocalWarping::getSeams(vector<vector<Point2i>> &displacement_map) {
    displacement_map = _displacement_map;
}

void LocalWarping::draw_all_seams() {
    Mat paint = _local_wraping_img.clone();
    for(const auto& seam : _all_seams){
        for(auto point : seam){
            paint.at<Vec3b>(point) = Vec3b(0,0,255);
        }
    }
    namedWindow("all_seams",WINDOW_NORMAL);
    imshow("all_seams", paint);
    waitKey(0);
}
