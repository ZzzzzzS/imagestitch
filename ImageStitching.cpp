#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>

#include "ImageStitching.h"

using namespace cv::xfeatures2d;
using namespace cv;
using namespace std;

//顶点结构体
typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}four_corners_t;

four_corners_t corners;

//计算顶点
void CalcCorners(const Mat& H, const Mat& src)
{
	double v2[] = { 0, 0, 1 };//左上角
	double v1[3];//变换后的坐标值
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量

	V1 = H * V2;
	//左上角(0,0,1)
	cout << "V2: " << V2 << endl;
	cout << "V1: " << V1 << endl;
	corners.left_top.x = v1[0] / v1[2];
	corners.left_top.y = v1[1] / v1[2];

	//左下角(0,src.rows,1)
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.left_bottom.x = v1[0] / v1[2];
	corners.left_bottom.y = v1[1] / v1[2];

	//右上角(src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_top.x = v1[0] / v1[2];
	corners.right_top.y = v1[1] / v1[2];

	//右下角(src.cols,src.rows,1)
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_bottom.x = v1[0] / v1[2];
	corners.right_bottom.y = v1[1] / v1[2];

}

//优化两图的连接处，使得拼接自然
//使用渐入渐出的拼接方式
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
{
	int start = MIN(corners.left_top.x, corners.left_bottom.x);//开始位置，即重叠区域的左边界  

	double processWidth = (img1.cols - start);//重叠区域的宽度  
	int rows = dst.rows;
	int cols = img1.cols; //注意，是列数*通道数
	double alpha = 1;//img1中像素的权重  
	for (int i = 0; i < rows; i++)
	{
		uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
		uchar* t = trans.ptr<uchar>(i);
		uchar* d = dst.ptr<uchar>(i);
		for (int j = start; j < cols; j++)
		{
			//如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
			{
				alpha = 1;
			}

			else if (p[j * 3] == 0 && p[j * 3 + 1] == 0 && p[j * 3 + 2] == 0)
			{
				alpha = 0;
			}
			else
			{
				//img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好  
				alpha = (processWidth - (j - start)) / processWidth;
			}

			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);

		}
	}

}

Mat ImageStitching( Mat& img1, Mat& img2,bool isDebug=false)
{
	Mat KeyPointImage1, KeyPointImage2;
	//Ptr<SurfFeatureDetector> detector = SurfFeatureDetector::create(); //surf检测子
	//Ptr<HarrisLaplaceFeatureDetector> detector = HarrisLaplaceFeatureDetector::create();//harris检测子
	Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(); //fast检测子

	Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create(); //sift展开子，opencv只有这个展开子

	vector<cv::KeyPoint> KeyPoint1, KeyPoint2; //特征点

	Mat KeyVector1, KeyVector2; //展开的特征向量

	detector->detect(img1, KeyPoint1); //计算特征点
	detector->detect(img2, KeyPoint2);//计算特征点
	extractor->compute(img1, KeyPoint1, KeyVector1); //计算特征点的向量
	extractor->compute(img2, KeyPoint2, KeyVector2);//计算特征点的向量

	drawKeypoints(img1, KeyPoint1, KeyPointImage1);//画特征点
	drawKeypoints(img2, KeyPoint2, KeyPointImage2);//画特征点

	if (isDebug)
	{
		imshow("KeyPointImage1", KeyPointImage1);
		imshow("KeyPointImage2", KeyPointImage2);
		imshow("vector", KeyVector1);
	}
	
	//创建暴力匹配器
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::MatcherType::BRUTEFORCE);

	vector<DMatch> match, GoodMatch; //匹配的点和优秀的匹配点

	matcher->match(KeyVector2, KeyVector1, match); //计算匹配点

	double Max_dist = 0;
	double Min_dist = 100000000;
	for (int i = 0; i < img1.rows; i++) //找到距离最短的匹配点
	{
		double dist = match[i].distance;
		if (dist < Min_dist)Min_dist = dist;
		if (dist > Max_dist)Max_dist = dist;
	}

	for (int i = 0; i < img1.rows; i++) //利用最小距离法寻找优秀的匹配点
	{
		if (match[i].distance < 2 * Min_dist)
			GoodMatch.push_back(match[i]);
	}

	
	if (isDebug)
	{
		Mat ImageMatch;
		drawMatches(img1, KeyPoint1, img2, KeyPoint2, GoodMatch, ImageMatch);//画出匹配的图
		imshow("ImageMatch", ImageMatch);
		waitKey();
	}
	
	//特征点特征转换
	vector<Point2f> imagePoints1, imagePoints2;

	for (int i = 0; i < GoodMatch.size(); i++)
	{
		imagePoints2.push_back(KeyPoint2[GoodMatch[i].queryIdx].pt); //顺序增大
		imagePoints1.push_back(KeyPoint1[GoodMatch[i].trainIdx].pt);  //跳变
	}

	Mat homo = findHomography(imagePoints2, imagePoints1, RANSAC);//计算单应性矩阵

	CalcCorners(homo, img2); //计算转换后的图片的顶点

	cout << homo << endl;
	cout << "顶点坐标" << endl;
	cout << corners.right_top << endl;
	cout << corners.right_bottom << endl;
	cout << corners.left_top << endl;
	cout << corners.left_bottom << endl;

	Mat imageTransform1;//透视变换后的图像

	//变换图像
	warpPerspective(img2, imageTransform1, homo, Size(MAX(MIN(corners.right_top.x, corners.right_bottom.x), MIN(corners.left_top.x, corners.left_bottom.x)), img1.rows));
	
	if (isDebug)
	{
		imshow("imageTransform", imageTransform1);
		cout << imageTransform1.size();
		cout << img2.size();
		waitKey();
	}
		
	Mat FinalImage(imageTransform1.rows, imageTransform1.cols, CV_8UC3); //最终图像的矩阵
	FinalImage.setTo(0);//初始清零

	//进行图像拼接
	imageTransform1.copyTo(FinalImage(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
	img1.copyTo(FinalImage(Rect(0, 0, img1.cols, img1.rows)));

	if (isDebug)
		imshow("FinalImage", FinalImage);

	OptimizeSeam(img1, imageTransform1, FinalImage); //拼接缝优化

	return FinalImage;
}