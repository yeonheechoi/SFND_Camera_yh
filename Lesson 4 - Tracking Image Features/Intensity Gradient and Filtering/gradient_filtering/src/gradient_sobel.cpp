#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

void gradientSobel()
{
    // TODO: Based on the image gradients in both x and y, compute an image 
    // which contains the gradient magnitude according to the equation at the 
    // beginning of this section for every pixel position. Also, apply different 
    // levels of Gaussian blurring before applying the Sobel operator and compare the results.
    
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1.png");

    // convert image to grayscale
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    //guassian smoothing
        // create filter kernel
    float gauss_data[25] = {1, 4, 7, 4, 1,
                            4, 16, 26, 16, 4,
                            7, 26, 41, 26, 7,
                            4, 16, 26, 16, 4,
                            1, 4, 7, 4, 1};

        // slove Gaussian smoothing code error
    for (int i = 0; i < 25; i++)
    {
        auto gauss_sum = accumulate(gauss_data, gauss_data+25, 0);
        gauss_data[i] /= gauss_sum;
    }

    cv::Mat kernel_gaussian = cv::Mat(5, 5, CV_32F, gauss_data);


        // apply filter
    cv::Mat result_gaussian;
    cv::filter2D(imgGray, result_gaussian, -1, kernel_gaussian, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    // create sobel filter kernel
    float sobel_x[9] = {-1, 0, +1,
                        -2, 0, +2, 
                        -1, 0, +1};
    cv::Mat kernel_x = cv::Mat(3, 3, CV_32F, sobel_x);

    float sobel_y[9] = {-1, -2, -1,
                        0, 0, 0, 
                        +1, +2, +1};
    cv::Mat kernel_y = cv::Mat(3, 3, CV_32F, sobel_y);

    //cv::Mat kernel;
    //kernel = (abs(kernel_x) + abs(kernel_y))*1/2;

    // apply filter
    cv::Mat result_x;
    cv::Mat result_xy;
    cv::filter2D(result_gaussian, result_x, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(result_x, result_xy, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    // show result
    //string windowName = "Sobel operator (x-direction)";
    string windowName = "Sobel operator";
    cv::namedWindow( windowName, 1 ); // create window 
    cv::imshow(windowName, result_xy);
    cv::waitKey(0); // wait for keyboard input before continuing
}

int main()
{
    gradientSobel();
}