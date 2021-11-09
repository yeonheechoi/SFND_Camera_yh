#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "structIO.hpp"

using namespace std;

void matchDescriptors(cv::Mat &imgSource, cv::Mat &imgRef, vector<cv::KeyPoint> &kPtsSource, vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      vector<cv::DMatch> &matches, string descriptorType, string matcherType, string selectorType)
{

    // configure matcher
    bool crossCheck = true;
    //기본값은 false
    //True로 지정하면 A라는 이미지의 어떤 하나의 특징점을 B라는 이미지의 모든 특징점과 비교하는 것에서 끝나지 않고, 다시 B라는 이미지에서 찾은 가장 유사한 특징점을 A라는 모든 특징점과 비교하여 그 결과가 같은지를 검사하라는 옵션
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)// 문자열 비교: 같으면
    {

        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2; //normType: 거리 측정 방식 선택
        matcher = cv::BFMatcher::create(normType, crossCheck);
        //첫 번째 세트 속 하나의 특성의 디스크립터를 취하고, 그리고 두 번째 세트의 다른 특성들과 거리 계산을 사용하여 매칭이 된다.
        // 그리고 가까운 것이 반환
        cout << "BF matching cross-check=" << crossCheck;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        //... TODO : implement FLANN matching
        // Fast Library for Approximate Nearest Neighbors
        //대규모 데이터 세트 및 고차원features에서 nearest neighbor search 을 위해 최적화 된 알고리즘 모음이 포함
        matcher = cv::FlannBasedMatcher::create();
        cout << "FLANN matching";
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // TODO : implement k-nearest-neighbor matching
        int k = 2; // k nearest neighbors = 2
        vector<vector<cv::DMatch>> knn_matches;
        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, k); // finds the (k=2) best matches descSource 가 1 순위, descRef 가 2 순위
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

        // TODO : filter matches using descriptor distance ratio test
        double thresholdv = 0.8; //threshold value = 0.8
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {

            if ((*it)[0].distance < thresholdv * (*it)[1].distance)
            {
                matches.push_back((*it)[0]); 
            }
        }
        cout << "(filter descriptor distance ratio) discarded matches keypoints = " << knn_matches.size() - matches.size() << endl;
    }

    // visualize results
    cv::Mat matchImg = imgRef.clone();
    cv::drawMatches(imgSource, kPtsSource, imgRef, kPtsRef, matches,
                    matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    string windowName = "Matching keypoints between two camera images (best 50)";
    cv::namedWindow(windowName, 7);
    cv::imshow(windowName, matchImg);
    cv::waitKey(0);
}

int main()
{
    cv::Mat imgSource = cv::imread("../images/img1gray.png");
    cv::Mat imgRef = cv::imread("../images/img2gray.png");

    vector<cv::KeyPoint> kptsSource, kptsRef; 
    readKeypoints("../dat/C35A5_KptsSource_BRISK_large.dat", kptsSource);
    readKeypoints("../dat/C35A5_KptsRef_BRISK_large.dat", kptsRef);

    cv::Mat descSource, descRef; 
    readDescriptors("../dat/C35A5_DescSource_BRISK_large.dat", descSource);
    readDescriptors("../dat/C35A5_DescRef_BRISK_large.dat", descRef);

    vector<cv::DMatch> matches;
    string matcherType = "MAT_FLANN"; // MAT_BF or MAT_FLANN
    string descriptorType = "DES_BINARY"; 
    string selectorType = "SEL_NN"; //SEL_NN or SEL_KNN
    matchDescriptors(imgSource, imgRef, kptsSource, kptsRef, descSource, descRef, matches, descriptorType, matcherType, selectorType);
}