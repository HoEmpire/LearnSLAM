#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "usage: feature_matching img1 img2" << endl;
    return 1;
  }
  //-- 读取图像
  Mat img_1_original = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  Mat img_2_original = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  assert(img_1_original.data != nullptr && img_2_original.data != nullptr);
  
  Mat img_1, img_2;
  if(img_1_original.rows > 1000)
  {
    resize(img_1_original, img_1, Size(800,600), CV_INTER_AREA);
    resize(img_2_original, img_2, Size(800,600), CV_INTER_AREA);
  }
  else
  {
    img_1 = img_1_original.clone();
    img_2 = img_2_original.clone();         
  }
  

  //-- 初始化
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;

  int option = 0;
  
  cout << "input feature option(0 = ORB, 1 = SURF):" << endl;
  cin >> option;
   
  //if(option == 0)
  //{
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    //-- 第一步:检测 Oriented FAST 角点位置
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

    Mat outimg1;
    drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    namedWindow("ORB features", 0);
    imshow("ORB features", outimg1);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> matches;
    t1 = chrono::steady_clock::now();
    matcher->match(descriptors_1, descriptors_2, matches);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;
    cout << "number of ORB features in img1 = " << descriptors_1.rows << endl;
    cout << "number of ORB features in img2 = " << descriptors_2.rows << endl;

    //-- 第四步:匹配点对筛选
    // 计算最小距离和最大距离
    auto min_max = minmax_element(matches.begin(), matches.end(),
                                  [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
      if (matches[i].distance <= max(2 * min_dist, 30.0)) {
        good_matches.push_back(matches[i]);
      }
    }
  //}
  //else if(option == 1)
  //{
    // Ptr<FeatureDetector> detector = ORB::create();
    // Ptr<DescriptorExtractor> descriptor = ORB::create();
    // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // //-- 第一步:检测 Oriented FAST 角点位置
    // chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    // detector->detect(img_1, keypoints_1);
    // detector->detect(img_2, keypoints_2);

    // //-- 第二步:根据角点位置计算 BRIEF 描述子
    // descriptor->compute(img_1, keypoints_1, descriptors_1);
    // descriptor->compute(img_2, keypoints_2, descriptors_2);
    // chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    // chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    // cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

    // Mat outimg1;
    // drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    // namedWindow("ORB features", 0);
    // imshow("ORB features", outimg1);

    // //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    // vector<DMatch> matches;
    // t1 = chrono::steady_clock::now();
    // matcher->match(descriptors_1, descriptors_2, matches);
    // t2 = chrono::steady_clock::now();
    // time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    // cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;
    // cout << "number of ORB features in img1 = " << descriptors_1.rows << endl;
    // cout << "number of ORB features in img2 = " << descriptors_2.rows << endl;

    // //-- 第四步:匹配点对筛选
    // // 计算最小距离和最大距离
    // auto min_max = minmax_element(matches.begin(), matches.end(),
    //                               [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    // double min_dist = min_max.first->distance;
    // double max_dist = min_max.second->distance;

    // printf("-- Max dist : %f \n", max_dist);
    // printf("-- Min dist : %f \n", min_dist);

    // //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    // std::vector<DMatch> good_matches;
    // for (int i = 0; i < descriptors_1.rows; i++) {
    //   if (matches[i].distance <= max(2 * min_dist, 30.0)) {
    //     good_matches.push_back(matches[i]);
    //   }
    // }
  //}

  //-- 第五步:绘制匹配结果
  Mat img_match;
  Mat img_goodmatch;
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
  namedWindow("all matches", 0);
  imshow("all matches", img_match);
  namedWindow("good matches", 0);
  imshow("good matches", img_goodmatch);
  waitKey(0);

  return 0;
}