# My implementation of "Rectangling Panoramic Images via Warping"

## Assignment
第二轮考察要求相似，需要用C++实现：https://mmcheng.net/imageresizing/ 这个页面下面链接的这个论文 Rectangling Panoramic Images via Warping。同样准备PPT面试。交互界面和图像变形部分建议调用OpenGL的纹理贴图来实现。正常的程序执行时间5s以内，请不要明显超过这个时间。
请在2周之内完成这个测试。  

## bilinear interpolation
https://theailearner.com/2018/12/29/image-processing-nearest-neighbour-interpolation/  
https://theailearner.com/2018/12/29/image-processing-bilinear-interpolation/
https://iquilezles.org/articles/ibilinear/  
For the situation in the paper, the quad is a trapezoid, we can use Inverse Bilinear Interpolation to get the weight of four vertices.  


## Find the minimum of a quadratic form


## Line Detection
for the line detection, we use the code from here(an improved version of LSD algorithm)  
http://www.ipol.im/pub/art/2012/gjmr-lsd/


## OpenGL Render Image

## Dependencies
### install opengl
```angular2html
sudo apt-get install libglew-dev freeglut3-dev
```
### install opencv
```
# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip pkg-config
sudo apt-get install libavformat-dev libavcodec-dev libswscale-dev
# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip
# Create build directory and switch into it
mkdir -p build && cd build
# Configure
cmake -DWITH_FFMPEG=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
# Build
cmake --build .
# Install 
sudo make install 
```
| 1 | 2 | 3 | 4 | 5                     | 6                     | 7                     | 8                     | 
|------|------|------|---|-----------------------|-----------------------|-----------------------|-----------------------|
| (c1+c3)*a^2 + (c1+c2+c3+c4)*a^2   | (c1+c2+c3+c4)*a*b                 | (c1+c2+c3+c4)*a*c                 | (c1+c2+c3+c4)*a*d                 | 0                                | 0                                | 0                                | 0                                |
| (c1+c2+c3+c4)*a*b                 | (c1+c3)*b^2 + (c1+c2+c3+c4)*b^2   | (c1+c2+c3+c4)*b*c                 | (c1+c2+c3+c4)*b*d                 | 0                                | 0                                | 0                                | 0                                |
| (c1+c2+c3+c4)*a*c                 | (c1+c2+c3+c4)*b*c                 | (c1+c3)*c^2 + (c1+c2+c3+c4)*c^2   | (c1+c2+c3+c4)*c*d                 | 0                                | 0                                | 0                                | 0                                |
| (c1+c2+c3+c4)*a*d                 | (c1+c2+c3+c4)*b*d                 | (c1+c2+c3+c4)*c*d                 | (c1+c3)*d^2 + (c1+c2+c3+c4)*d^2   | 0                                | 0                                | 0                                | 0                                |
| 0                                | 0                                | 0                                | 0                                | (c1+c2+c3+c4)*a^2                | (c1+c2+c3+c4)*a*b                | (c1+c2+c3+c4)*a*c                | (c1+c2+c3+c4)*a*d                |
| 0                                | 0                                | 0                                | 0                                | (c1+c2+c3+c4)*a*b                | (c1+c2+c3+c4)*b^2                | (c1+c2+c3+c4)*b*c                | (c1+c2+c3+c4)*b*d                |
| 0                                | 0                                | 0                                | 0                                | (c1+c2+c3+c4)*a*c                | (c1+c2+c3+c4)*b*c                | (c1+c2+c3+c4)*c^2                | (c1+c2+c3+c4)*c*d                |
| 0                                | 0                                | 0                                | 0                                | (c1+c2+c3+c4)*a*d                | (c1+c2+c3+c4)*b*d                | (c1+c2+c3+c4)*c*d                | (c1+c2+c3+c4)*d^2                |

