//
// Created by nuc on 23-7-14.
//

#include "LocalWarping.h"

LocalWarping::LocalWarping(Mat &source_img, Mat &mask) {
    _source_img = source_img.clone();
    _mask = mask.clone();
    _local_wraping_img = source_img.clone();

    while(true){
        for(int i = 0; i < 4; i++){
            switch (i) {
                case 0:{
                    imageShift(Top);
                    break;
                }
                case 1:{
                    imageShift(Bottom);
                    break;
                }
                case 2:{
                    imageShift(Left);
                    break;
                }
                case 3:{
                    imageShift(Right);
                    break;
                }
                default: break;
            }
        }
    }
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

void LocalWarping::imageShift(Position position) {
    Rect roi = getSubImageRect(position);
    if(roi.width == 0 || roi.height == 0) return;
    Mat image_roi = Mat(this->_local_wraping_img, roi);
    Mat mask_roi = Mat(this->_mask,roi);
    Direction direction;
    if(position == Left || position == Right) direction = UpToDown;
    else direction = LeftToRight;
    vector<Point2i> seam;
    calculateSeam(image_roi, mask_roi, direction, seam, roi);
    singleShift(position, seam);
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
        int move_direction =  path_map.at<int>(i, last_col_index);
        int col_pos = last_col_index + move_direction;
        seam.emplace_back(col_pos,i-1 );
        last_col_index = col_pos;
    }

    if(direction == LeftToRight){
        for(int i = 0; i < seam.size(); i++){
            Point2i temp_point(seam[i].y, src.rows - seam[i].x);
            seam[i] = temp_point;
        }
    }

    for(int i = 0; i < seam.size(); i++){
        seam[i].x += roi.x;
        seam[i].y += roi.y;
    }

#ifdef SHOW
//    Mat paint = _source_img.clone();
//    for(int i = 0; i < seam.size(); i++){
//        paint.at<Vec3b>(seam[i]) = Vec3b(255,0,0);
//    }
//    namedWindow("paint",WINDOW_NORMAL);
//    imshow("paint",paint);
//    waitKey(0);
#endif
}

void LocalWarping::calculateCostImage(Mat &input_image, Mat &cost_image, Mat &mask) {
    // 转换为灰度图像
    cv::Mat grayImage;
    cv::cvtColor(input_image, grayImage, cv::COLOR_BGR2GRAY);

    // 计算梯度
    cv::Mat gradientX, gradientY;
    cv::Sobel(grayImage, gradientX, CV_32F, 1, 0);
    cv::Sobel(grayImage, gradientY, CV_32F, 0, 1);

    // 计算梯度幅值
    cv::magnitude(gradientX, gradientY, cost_image);

    // 将 mask 图像中像素值为 255 的点在cost_image中对应的值设置为 10^8
    for (int i = 0; i < mask.rows; ++i)
    {
        for (int j = 0; j < mask.cols; ++j)
        {
            uchar value = mask.at<uchar>(i,j);
            if (value == 255)
            {
                cost_image.at<float>(i, j) = 1e8;  // 设置为 10 的 8 次方
            }
        }
    }
}

void LocalWarping::singleShift(Position position, const vector<Point2i> &seam) {
#ifdef SHOW
    for(int i = 0; i < seam.size(); i++){
        _local_wraping_img.at<Vec3b>(seam[i]) = Vec3b(255,0,255);
    }
    namedWindow("_mask",WINDOW_NORMAL);
    imshow("_mask",_mask);
    namedWindow("_local_wraping_img",WINDOW_NORMAL);
    imshow("_local_wraping_img",_local_wraping_img);
    waitKey(0);
#endif

    if(position == Bottom){
        for(int i = 0; i < seam.size(); i++){
            int x = seam[i].x;
            int y = seam[i].y;
            while(_mask.at<uchar>(Point2i(x,y)) == 0){
                y++;
            }
            _mask.at<uchar>(Point2i(x,y)) = 0;
            for(int j = y; j != seam[i].y; j--){
                _local_wraping_img.at<Vec3b>(Point2i(x,j)) = _local_wraping_img.at<Vec3b>(Point(x,j-1));
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
            }
        }
    }
}
