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
//	//���������Ҫ��ȡ�ؼ��ֵ���Ƶ
//	cap.open("exp1(1).avi");
//	if (!cap.isOpened())//�����Ƶ�����������򷵻�
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
//		//��ͼ���BGRɫ�ʿռ�ת���� HSVɫ�ʿռ�
//		cvtColor(srcImage_base, previousImage, CV_BGR2GRAY);
//		cvtColor(srcImage_test1, currentImage, CV_BGR2GRAY);
//
//		absdiff(currentImage, previousImage, resultImage);  //֡������
//
//		threshold(resultImage, resultImage, 40, 255.0, CV_THRESH_BINARY); //��ֵ��������ֵ������20����Ϊ255������Ϊ0
//
//		imshow("keyframe", frame_key);
//
//		float counter = 0;
//		float num = 0;
//		// ͳ����֡�����ͼ����
//		for (int i = 0; i < resultImage.rows; i++)
//		{
//			uchar* data = resultImage.ptr<uchar>(i); //��ȡÿһ�е�ָ��
//			for (int j = 0; j < resultImage.cols; j++)
//			{
//				num = num + 1;
//				if (data[j] == 255) //���ʵ�����ֵ
//				{
//					counter = counter + 1;
//				}
//			}
//		}
//		p = counter / num;
//		// counter  num  p �ֱ�Ϊ  �б仯�����ص���  �����ص���Ŀ  ����
//		printf(">>>>>>counter>>>>num>>>>p: %f  %f  %f  \n", counter, num, p);
//		if (p > 0.38) //�ﵽ��ֵ���������ﵽһ���������򱣴��ͼ��
//		{
//			//printf(">>>>>>>>>>>>>6");
//			cout << ">>>>>>>>>>>>>>>>>>>>.>>>>>>>.this frame is keyframe!" << endl;
//			
//			frame_key = frame;
//			imshow("keyframe", frame_key);
//			
//			cout << "����д��" << currentFrame << "֡" << endl;
//			stringstream str;
//			//д��Ƶ����Ŀ¼,�ҵ���  ./keyframes_pixels_cha/xx.jpg  xxΪ��ǰ֡�����
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
