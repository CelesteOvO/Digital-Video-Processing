#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

//int main()
//{
//	Mat image = imread("1.jpg");
//	double m0, m1, m2, m3;
//	m0 = 1;
//	int N = image.rows * image.cols;
//	for (int i = 0; i < image.rows; i++)
//	{
//		for (int j = 0; j < image.cols; j++)
//		{
//			m1 += (i - image.rows / 2) * (i - image.rows / 2);
//			m2 += (j - image.cols / 2) * (j - image.cols / 2);
//			m3 += (i - image.rows / 2) * (j - image.cols / 2);
//		}
//	}
//	m1 = m1 / N;
//	m2 = m2 / N;
//	m3 = m3 / N;
//	double m00, m11, m22, m33;
//	double p0, p1;
//	double z00, z10, z01, z11, z02, z12, z03, z13;
//	
//}