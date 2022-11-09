#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <string.h>

using namespace std;
using namespace cv;

//Otsu��ֵ�ָ�
Mat OtsuAlgThreshold(Mat& image)
{
	if (image.channels() != 1)
	{
		cout << "Please input Gray-image!" << endl;
	}
	int T = 0; //Otsu�㷨��ֵ  
	double varValue = 0; //��䷽���м�ֵ����
	double w0 = 0; //ǰ�����ص�����ռ����  
	double w1 = 0; //�������ص�����ռ����  
	double u0 = 0; //ǰ��ƽ���Ҷ�  
	double u1 = 0; //����ƽ���Ҷ�  
	double Histogram[256] = { 0 }; //�Ҷ�ֱ��ͼ���±��ǻҶ�ֵ�����������ǻҶ�ֵ��Ӧ�����ص�����  
	uchar* data = image.data;

	double totalNum = image.rows * image.cols; //��������

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			if (image.at<uchar>(i, j) != 0) Histogram[data[i * image.step + j]]++;
		}
	}
	int minpos, maxpos;
	for (int i = 0; i < 255; i++)
	{
		if (Histogram[i] != 0)
		{
			minpos = i;
			break;
		}
	}
	for (int i = 255; i > 0; i--)
	{
		if (Histogram[i] != 0)
		{
			maxpos = i;
			break;
		}
	}

	for (int i = minpos; i <= maxpos; i++)
	{
		//ÿ�α���֮ǰ��ʼ��������  
		w1 = 0;       u1 = 0;       w0 = 0;       u0 = 0;
		//***********����������ֵ����**************************  
		for (int j = 0; j <= i; j++) //�������ָ�ֵ����  
		{
			w1 += Histogram[j];   //�����������ص�����  
			u1 += j * Histogram[j]; //�������������ܻҶȺ�  
		}
		if (w1 == 0) //�����������ص���Ϊ0ʱ�˳�  
		{
			break;
		}
		u1 = u1 / w1; //��������ƽ���Ҷ�  
		w1 = w1 / totalNum; // �����������ص�����ռ����
		//***********����������ֵ����**************************  

		//***********ǰ��������ֵ����**************************  
		for (int k = i + 1; k < 255; k++)
		{
			w0 += Histogram[k];  //ǰ���������ص�����  
			u0 += k * Histogram[k]; //ǰ�����������ܻҶȺ�  
		}
		if (w0 == 0) //ǰ���������ص���Ϊ0ʱ�˳�  
		{
			break;
		}
		u0 = u0 / w0; //ǰ������ƽ���Ҷ�  
		w0 = w0 / totalNum; // ǰ���������ص�����ռ����  
		//***********ǰ��������ֵ����**************************  

		//***********��䷽�����******************************  
		double varValueI = w0 * w1 * (u1 - u0) * (u1 - u0); //��ǰ��䷽�����  
		if (varValue < varValueI)
		{
			varValue = varValueI;
			T = i;
		}
	}
	Mat dst;
	threshold(image, dst, T, 255, CV_THRESH_OTSU);
	return dst;
}

//�������ֵ�ָ
Mat EntropySeg(Mat src)
{
	int tbHist[256] = { 0 };
	int index = 0;
	double Property = 0.0;
	double maxEntropy = -1.0;
	double frontEntropy = 0.0;
	double backEntropy = 0.0;
	int TotalPixel = 0;
	int nCol = src.cols * src.channels();
	for (int i = 0; i < src.rows; i++)
	{
		uchar* pData = src.ptr<uchar>(i);
		for (int j = 0; j < nCol; j++)
		{
			++TotalPixel;
			tbHist[pData[j]] += 1;
		}
	}

	for (int i = 0; i < 256; i++)
	{
		double backTotal = 0;
		for (int j = 0; j < i; j++)
		{
			backTotal += tbHist[j];
		}

		for (int j = 0; j < i; j++)
		{
			if (tbHist[j] != 0)
			{
				Property = tbHist[j] / backTotal;
				backEntropy += -Property * logf((float)Property);
			}
		}

		for (int k = i; k < 256; k++)
		{
			if (tbHist[k] != 0)
			{
				Property = tbHist[k] / (TotalPixel - backTotal);
				frontEntropy += -Property * logf((float)Property);
			}
		}

		if (frontEntropy + backEntropy > maxEntropy)
		{
			maxEntropy = frontEntropy + backEntropy;
			index = i;
		}

		frontEntropy = 0.0;
		backEntropy = 0.0;
	}

	Mat dst;
	threshold(src, dst, index, 255, 0);
	return dst;
}

//����Ӧ��ֵ�ָ�
void myadaptive(InputArray _src, OutputArray _dst, double maxValue,
	int method, int type, int blockSize, double delta)
{
	Mat src = _src.getMat();

	CV_Assert(src.type() == CV_8UC1);
	CV_Assert(blockSize % 2 == 1 && blockSize > 1);
	Size size = src.size();

	_dst.create(size, src.type());
	Mat dst = _dst.getMat();

	if (maxValue < 0)
	{
		dst = Scalar(0);
		return;
	}

	Mat mean;
	if (src.data != dst.data)
		mean = dst;
	if (method == ADAPTIVE_THRESH_GAUSSIAN_C)
	{
		GaussianBlur(src, mean, Size(blockSize, blockSize), 0, 0, BORDER_REPLICATE);
	}
	else if (method == ADAPTIVE_THRESH_MEAN_C)
	{
		boxFilter(src, mean, src.type(), Size(blockSize, blockSize),
			Point(-1, -1), true, BORDER_REPLICATE);
	}
	else
	{
		CV_Error(CV_StsBadFlag, "Unknown/unsupported adaptive threshold method");
	}

	int i, j;
	uchar imaxval = saturate_cast<uchar>(maxValue);
	int idelta = type == THRESH_BINARY ? cvCeil(delta) : cvFloor(delta);
	uchar tab[768];

	if (type == CV_THRESH_BINARY)
		for (i = 0; i < 768; i++)
			tab[i] = (uchar)(i - 255 > -idelta ? imaxval : 0);
	else if (type == CV_THRESH_BINARY_INV)
		for (i = 0; i < 768; i++)
			tab[i] = (uchar)(i - 255 <= -idelta ? imaxval : 0);
	else
	{
		CV_Error(CV_StsBadFlag, "Unknown/unsupported threshold type");
	}

	if (src.isContinuous() && mean.isContinuous() && dst.isContinuous())
	{
		size.width *= size.height;
		size.height = 1;
	}

	for (i = 0; i < size.height; i++)
	{
		const uchar* sdata = src.data + src.step * i;
		const uchar* mdata = mean.data + mean.step * i;
		uchar* ddata = dst.data + dst.step * i;

		for (j = 0; j < size.width; j++)
			// ��[-255, 255] ӳ�䵽[0, 510]Ȼ����
			ddata[j] = tab[sdata[j] - mdata[j] + 255];
	}
}

//������ֵ�ָ�
Mat IterationThreshold(Mat src)
{
	int width = src.cols;
	int height = src.rows;
	int hisData[256] = { 0 };
	for (int j = 0; j < height; j++)
	{
		uchar* data = src.ptr<uchar>(j);
		for (int i = 0; i < width; i++)
			hisData[data[i]]++;
	}

	int T0 = 0;
	for (int i = 0; i < 256; i++)
	{
		T0 += i * hisData[i];
	}
	T0 /= width * height;

	int T1 = 0, T2 = 0;
	int num1 = 0, num2 = 0;
	int T = 0;
	while (1)
	{
		for (int i = 0; i < T0 + 1; i++)
		{
			T1 += i * hisData[i];
			num1 += hisData[i];
		}
		if (num1 == 0)
			continue;
		for (int i = T0 + 1; i < 256; i++)
		{
			T2 += i * hisData[i];
			num2 += hisData[i];
		}
		if (num2 == 0)
			continue;

		T = (T1 / num1 + T2 / num2) / 2;

		if (T == T0)
			break;
		else
			T0 = T;
	}

	Mat dst;
	threshold(src, dst, T, 255, 0);
	return dst;
}

//void main()
//{
//	Mat src = imread("test1.png");
//	cvtColor(src, src, COLOR_RGB2GRAY);
//
//	Mat bw1, bw2, bw3, bw4;
//	myadaptive(src, bw1, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 15, 10);
//	bw2 = EntropySeg(src);
//	bw3 = OtsuAlgThreshold(src);
//	bw4 = IterationThreshold(src);
//
//	imshow("source", src);
//	imshow("����Ӧ��ֵ�ָ�", bw1);
//	imshow("�������ֵ�ָ�", bw2);
//	imshow("Otsu��ֵ�ָ�", bw3);
//	imshow("������ֵ�ָ�", bw4);
//	waitKey(60000);
//}