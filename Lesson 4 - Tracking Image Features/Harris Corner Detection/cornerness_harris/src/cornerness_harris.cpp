#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

void cornernessHarris()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // convert to grayscale

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    // (입력이미지, 출력이미지, 인접 픽셀 크기, Sobel Ksize, Harris parameter, 픽셀 보간법)
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    // 정규화 (입력 이미지, 출력 이미지, normalize range(lovw), normalize range(high), 픽셀 보간법)
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    // 미분한 결과에 절대값을 적용해서 8 bit unsigned int 형으로 바꿈

    // visualize results
    //string windowName = "Harris Corner Detector Response Matrix";
    //cv::namedWindow(windowName, 4);
    //cv::imshow(windowName, dst_norm_scaled);
    //cv::waitKey(0);

    // TODO: Your task is to locate local maxima in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local neighborhood around 
    // each maximum. The resulting coordinates shall be stored in a list of keypoints 
    // of the type `vector<cv::KeyPoint>`.
    double maxOverlap = 0.0;
    vector<cv::KeyPoint> keypoints;

        for (int r = 0; r < dst_norm.rows; r++)
    {
        for (int c = 0; c < dst_norm.cols; c++)
        {
            int response = (int)dst_norm.at<float>(r, c);
            if (response > minResponse)
            { 
                cv::KeyPoint newKeyPoint; //newKeypoint: 특징점 정보를 담는 객체 (minResponse 이상인 keypoint 들만 모음)
                newKeyPoint.pt = cv::Point2f(r, c); //.pt: 특징점 좌표(x, y), float 타입으로 정수 변환 필요
                newKeyPoint.size = 2 * apertureSize; //.size: 의미 있는 특징점 이웃의 반지름
                newKeyPoint.response = response; //특징점 반응 강도 (추출기에 따라 다름)


                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool Overlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    //This method computes overlap for pair of keypoints. Overlap is the ratio between area of keypoint regions' intersection and area of keypoint regions' union (considering keypoint region as circle). If they don't overlap, we get zero. If they coincide at same location with same size, we get 1.
                    if (kptOverlap > maxOverlap) //newKeyPoint 랑 *it 이 영역이 겹치지 않으면 0이 나오고 그렇지 않으면 ~ 1 값이 나오게 되기 때문에 maxOverlap (=0) 보다 크다는 말은 겹치는 영역이 있다는 의미
                    {
                        Overlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!Overlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } 
    }     

    string windowName = "Harris Corner Detection Results";
    cv::namedWindow(windowName, 5);
    cv::Mat Result_Image = dst_norm_scaled.clone();
    cv::drawKeypoints(dst_norm_scaled, keypoints, Result_Image, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT); 
    // (입력 이미지, 표시할 특징점 리스트, 특징점이 그려진 결과 이미지, 표시할 생상 (default:랜덤), 표시할 방법(cv2.DRAW_MATCHES_FLAGS_DEFAULT: 좌표 중심에 동그라미만 그림(default), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: 동그라미의 크기를 size와 angle을 반영해서 그림))
    cv::imshow(windowName, Result_Image);
    cv::waitKey(0);
}

int main()
{
    cornernessHarris();
}