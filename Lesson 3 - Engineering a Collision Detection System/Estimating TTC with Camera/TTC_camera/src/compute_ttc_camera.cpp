#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>

// these includes provide the data structures for managing 3D Lidar points and 2D keypoints
#include "dataStructures.h" // you do not need to look into this file
#include "structIO.hpp" // you do not need to look into this file

using namespace std;

// Purpose: Compute time-to-collision (TTC) based on keypoint correspondences in successive images
// Notes: 
// - please take a look at the main()-function first
// - kptsPrev and kptsCurr are the input keypoint sets, kptMatches are the matches between the two sets,
//   frameRate is required to compute the delta time between frames and TTC will hold the result of the computation
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer keypoint loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner keypoint loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt); //norm 은 벡터의 길이 혹은 크기를 측정하는 함수
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    /*double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size(); // (누적합 / 전체 갯수)

    double dT = 1 / frameRate;
    TTC = -dT / (1 - meanDistRatio); */

    // TODO: STUDENT TASK (replacement for meanDistRatio)
    /*
    double medianDistRatio;
    if (distRatios.size()% 2 == 0) // if even
    {
        auto m1_it = distRatios.begin() + distRatios.size()/2 -1 ;
        auto m2_it = distRatios.begin() + distRatios.size()/2 ;

        std::nth_element(distRatios.begin(),m1_it,distRatios.end() );
        auto e1 = *m1_it;

        std::nth_element(distRatios.begin(),m2_it,distRatios.end() );
        auto e2 = *m2_it;

        medianDistRatio = (e1 + e2)/2;
    }
    else
    {
        auto median_it = distRatios.begin() + distRatios.size()/2;
        std::nth_element(distRatios.begin(),median_it,distRatios.end() );
        medianDistRatio = *median_it;
    }
    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianDistRatio);  
    }*/
    double medianDistRatio;
    if (distRatios.size()% 2 == 0) // if even
    {
        std::nth_element(distRatios.begin(),distRatios.begin() + distRatios.size()/2 -1,distRatios.end() );
        auto m1 = distRatios[distRatios.size()/2 -1];

        std::nth_element(distRatios.begin(),distRatios.begin() + distRatios.size()/2,distRatios.end() );
        auto m2 = distRatios[distRatios.size()/2];

        medianDistRatio = (m1 + m2)/2;
    }
    else
    {
        std::nth_element(distRatios.begin(),distRatios.begin() + distRatios.size()/2,distRatios.end() );
        medianDistRatio = distRatios[distRatios.size()/2];
    }
    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianDistRatio);  
    
}

int main()
{
    // step 1: read pre-recorded keypoint sets from file
    // Note that the function "readKeypoints" is a helper function that is able to read pre-saved results from disk
    // so that you can focus on TTC computation based on a defined set of keypoints and matches. 
    // The task you need to solve in this example does not require you to look into the data structures.  
    vector<cv::KeyPoint> kptsSource, kptsRef;
    readKeypoints("../dat/C23A5_KptsSource_AKAZE.dat", kptsSource); // readKeypoints("./dat/C23A5_KptsSource_SHI-BRISK.dat"
    readKeypoints("../dat/C23A5_KptsRef_AKAZE.dat", kptsRef); // readKeypoints("./dat/C23A5_KptsRef_SHI-BRISK.dat"

    // step 2: read pre-recorded keypoint matches from file
    vector<cv::DMatch> matches;
    readKptMatches("../dat/C23A5_KptMatches_AKAZE.dat", matches); // readKptMatches("./dat/C23A5_KptMatches_SHI-BRISK.dat", matches);
    
    // step 3: compute the time-to-collision based on the pre-recorded data
    double ttc; 
    computeTTCCamera(kptsSource, kptsRef, matches, 10.0, ttc);
    cout << "ttc = " << ttc << "s" << endl;
}