#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

using namespace std;

void magnitudeSobel()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1gray.png");

    // convert image to grayscale
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // apply smoothing using the GaussianBlur() function from the OpenCV
    cv::Mat img_blur;
    cv::GaussianBlur(imgGray, img_blur, cv::Size(5, 5), 0, 0, 4);

    // create filter kernels using the cv::Mat datatype both for x and y
    float sobel_x[9] = {-1, 0, +1,
                        -2, 0, +2, 
                        -1, 0, +1};
    cv::Mat kernel_x = cv::Mat(3, 3, CV_32F, sobel_x);

    float sobel_y[9] = {-1, -2, -1,
                        0, 0, 0, 
                        +1, +2, +1};
    cv::Mat kernel_y = cv::Mat(3, 3, CV_32F, sobel_y);


    // apply filter using the OpenCv function filter2D()
    // ToDo : Add your code here
    cv::Mat result_x;
    cv::Mat result_y;
    cv::filter2D(img_blur, result_x, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(img_blur, result_y, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    // compute magnitude image based on the equation presented in the lesson 
    // ToDo : Add your code here
    cv::Mat magnitude;
    magnitude = img_blur.clone();
    
    for (int r = 0; r < magnitude.rows; r++)
    {
        for (int c = 0; c < magnitude.cols; c++)
        {
            magnitude.at<uchar>(r, c) = sqrt(result_x.at<uchar>(r, c) * result_x.at<uchar>(r, c) +
                                                     result_y.at<uchar>(r, c) * result_y.at<uchar>(r, c));
        }
    }

    // show result
    string windowName = "Gaussian Blurring";
    cv::namedWindow(windowName, 1); // create window
    cv::imshow(windowName, magnitude);
    cv::waitKey(0); // wait for keyboard input before continuing
}

int main()
{
    magnitudeSobel();
}