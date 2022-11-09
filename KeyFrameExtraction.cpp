//#include <stdio.h>
//#include <stdlib.h>
//#include <iostream>
//#include <fstream>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/ml/ml.hpp>
//#include <string.h>
//
//using namespace std;
//using namespace cv;
//
//int main()
//{
//	system("color 2F");
//	long currentFrame = 1;
//	float p;
//	VideoCapture cap;
//	//这里放置需要提取关键字的视频
//	cap.open("exp1(1).avi");
//	if (!cap.isOpened())//如果视频不能正常打开则返回
//	{
//		cout << "cannot open video!" << endl;
//		return 0;
//	}
//	
//	Mat frame_key;
//	cap >> frame_key;
//	if (frame_key.empty())
//		cout << "frame_key is empty!" << endl;
//	imshow("fram_1", frame_key);
//	waitKey(20);
//	stringstream str;
//	str << "keyframes_pixels_cha/" << currentFrame << ".jpg";
//	cout << str.str() << endl;
//	imwrite(str.str(), frame_key);
//	Mat frame;
//	Mat previousImage, currentImage, resultImage;
//	while (1)
//	{
//		currentFrame++;
//		Mat frame;
//		cap >> frame;
//		if (frame.empty())
//		{
//			cout << "frame is empty!" << endl;
//			break;
//		}
//		imshow("fram_1", frame);
//		waitKey(20);
//
//		Mat srcImage_base, hsvImage_base;
//		Mat srcImage_test1, hsvImage_test1;
//		srcImage_base = frame_key;
//		srcImage_test1 = frame;
//		//将图像从BGR色彩空间转换到 HSV色彩空间
//		cvtColor(srcImage_base, previousImage, CV_BGR2GRAY);
//		cvtColor(srcImage_test1, currentImage, CV_BGR2GRAY);
//
//		absdiff(currentImage, previousImage, resultImage);  //帧差法，相减
//
//		threshold(resultImage, resultImage, 40, 255.0, CV_THRESH_BINARY); //二值化，像素值相差大于20则置为255，其余为0
//
//		imshow("keyframe", frame_key);
//
//		float counter = 0;
//		float num = 0;
//		// 统计两帧相减后图像素
//		for (int i = 0; i < resultImage.rows; i++)
//		{
//			uchar* data = resultImage.ptr<uchar>(i); //获取每一行的指针
//			for (int j = 0; j < resultImage.cols; j++)
//			{
//				num = num + 1;
//				if (data[j] == 255) //访问到像素值
//				{
//					counter = counter + 1;
//				}
//			}
//		}
//		p = counter / num;
//		// counter  num  p 分别为  有变化的像素点数  总像素点数目  比例
//		printf(">>>>>>counter>>>>num>>>>p: %f  %f  %f  \n", counter, num, p);
//		if (p > 0.38) //达到阈值的像素数达到一定的数量则保存该图像
//		{
//			//printf(">>>>>>>>>>>>>6");
//			cout << ">>>>>>>>>>>>>>>>>>>>.>>>>>>>.this frame is keyframe!" << endl;
//			
//			frame_key = frame;
//			imshow("keyframe", frame_key);
//			
//			cout << "正在写第" << currentFrame << "帧" << endl;
//			stringstream str;
//			//写视频保存目录,我的是  ./keyframes_pixels_cha/xx.jpg  xx为当前帧的序号
//			str << "keyframes_pixels_cha/" << currentFrame << ".jpg";
//			cout << str.str() << endl;
//			imwrite(str.str(), frame_key);
//		}
//		else
//		{
//			cout << ">>>>>>>>>>>>.this frame is not keyframe!" << endl;
//		}
//
//	}
//}
