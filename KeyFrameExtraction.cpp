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

Vec3b RandomColor(int value) //<span style = "line-height: 20.8px; font-family: sans-serif;" >//���������ɫ����</span>
{
	value = value % 255;  //����0~255�������
	RNG rng;
	int aa = rng.uniform(0, value);
	int bb = rng.uniform(0, value);
	int cc = rng.uniform(0, value);
	return Vec3b(aa, bb, cc);
}

void WaterShed(Mat image)
{
	//�ҶȻ����˲���Canny��Ե���
	Mat imageGray;
	cvtColor(image, imageGray, CV_RGB2GRAY);//�Ҷ�ת��
	GaussianBlur(imageGray, imageGray, Size(3, 3), 2);   //��˹�˲�
	//imshow("Gray Image", imageGray);
	Canny(imageGray, imageGray, 80, 150);
	//imshow("Canny Image", imageGray);

	//��������
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imageGray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	Mat imageContours = Mat::zeros(image.size(), CV_8UC1);  //����	
	Mat marks(image.size(), CV_32S);   //Opencv��ˮ��ڶ����������
	marks = Scalar::all(0);
	int index = 0;
	int compCount = 0;
	for (; index >= 0; index = hierarchy[index][0], compCount++)
	{
		//��marks���б�ǣ��Բ�ͬ������������б�ţ��൱������עˮ�㣬�ж������������ж���עˮ��
		drawContours(marks, contours, index, Scalar::all(compCount + 1), 1, 8, hierarchy);
		drawContours(imageContours, contours, index, Scalar(255), 1, 8, hierarchy);
	}

	//��������һ�´���ľ���marks����ʲô����
	Mat marksShows;
	convertScaleAbs(marks, marksShows);
	//imshow("marksShow", marksShows);
	//imshow("����", imageContours);
	watershed(image, marks);

	//����������һ�·�ˮ���㷨֮��ľ���marks����ʲô����
	Mat afterWatershed;
	convertScaleAbs(marks, afterWatershed);
	//imshow("After Watershed", afterWatershed);

	//��ÿһ�����������ɫ���
	Mat PerspectiveImage = Mat::zeros(image.size(), CV_8UC3);
	for (int i = 0; i < marks.rows; i++)
	{
		for (int j = 0; j < marks.cols; j++)
		{
			int index = marks.at<int>(i, j);
			if (marks.at<int>(i, j) == -1)
			{
				PerspectiveImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			else
			{
				PerspectiveImage.at<Vec3b>(i, j) = RandomColor(index);
			}
		}
	}
	/*imshow("After ColorFill", PerspectiveImage);*/

	//�ָ�����ɫ�Ľ����ԭʼͼ���ں�
	Mat wshed;
	addWeighted(image, 0.4, PerspectiveImage, 0.6, 0, wshed);
	imshow("WaterShed", wshed);

	waitKey();
}

void MomentPreserving(Mat img1)
{
	int twhreshld;
	float h = 0;
	float p0;
	int z0, z1;
	long long  P_sum1 = 0, P_sum2 = 0, P_sum3 = 0;
	int m_0, m_1, m_2, m_3;
	int m0 = 1;
	Mat img;
	cvtColor(img1, img, CV_BGR2GRAY); 
	// תΪ�Ҷ�ͼ��
	int pixel = img.rows * img.cols;
	// ����ͼ�����������
	int height = img.rows;
	int width = img.cols;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			P_sum1 += (int)img.at<uchar>(i, j); // P_sum1��ͼƬ�ڣ�i��j�����ĻҶ�ֵ֮��
		}
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			P_sum2 += ((int)img.at<uchar>(i, j) * (int)img.at<uchar>(i, j));// P_sum2��ͼƬ�ڣ�i��j�����ĻҶ�ֵ��ƽ����
		}
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			P_sum3 += ((int)img.at<uchar>(i, j) * (int)img.at<uchar>(i, j) * (int)img.at<uchar>(i, j)); // P_sum3���ڣ�i��j�����ĻҶ�ֵ��������
		}
	}

	int m1 = P_sum1 / (img.rows * img.cols); // m1��ͼ���һ�׾�
	int m2 = P_sum2 / (img.rows * img.cols); // m2��ͼ��Ķ��׾�
	int m3 = P_sum3 / (img.rows * img.cols); // m3��ͼ������׾�

	int c0 = (m1 * m3 - m2 * m2) / (m2 - m1 * m1);
	int c1 = (m1 * m2 - m3) / (m2 - m1 * m1);


	int a = c1 * c1 - 4 * c0;
	int b = sqrt(a);
	int c = b - c1;

	int G = 0.5 * c;
	p0 = (G + m1) / b;

	float histogram[256] = { 0 };
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			histogram[img.at<uchar>(i, j)]++;
		}
	}

	for (int i = 0; i < 256; i++)
	{
		histogram[i] = histogram[i] / 256;
	}


	for (twhreshld = 0; twhreshld < 256; twhreshld++)
	{
		for (int j = 0; j < twhreshld; j++)
		{
			h = 0;
			h += histogram[j];
		}
		if (p0 - h < 1e-6)
			break;

	}
	cout << "������ֵΪ" << twhreshld << endl;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) >= twhreshld)
				img.at<uchar>(i, j) = 255;
			else
				img.at<uchar>(i, j) = 0;
		}
	}
	imshow("�ز�����ֵ��", img);
}

//Otsu��ֵ�ָ�
Mat OtsuAlgThreshold(Mat image1)
{
	Mat image;
	cvtColor(image1, image, CV_BGR2GRAY);
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
	imshow("Otsu��ֵ�ָ�",dst);
	return dst;
}

//�������ֵ�ָ
Mat EntropySeg(Mat src1)
{
	Mat src;
	cvtColor(src1, src, CV_BGR2GRAY);
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
	imshow("����ط���ֵ�ָ�", dst);
	return dst;
}

//����Ӧ��ֵ�ָ�
void myadaptive(InputArray _src, OutputArray _dst, double maxValue,
	int method, int type, int blockSize, double delta)
{
	
	Mat src1 = _src.getMat();
	Mat src;
	cvtColor(src1, src, CV_BGR2GRAY);
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
	imshow("����Ӧ��ֵ�ָ�", dst);
}

//������ֵ�ָ�
Mat IterationThreshold(Mat src1)
{
	Mat src;
	cvtColor(src1, src, CV_BGR2GRAY);
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
	imshow("��������ֵ�ָ�",dst);
	return dst;
}

int main()
{
	system("color 2F");
	long currentFrame = 1;
	float p;
	VideoCapture cap;
	//���������Ҫ��ȡ�ؼ��ֵ���Ƶ
	cap.open("exp1(1).avi");
	if (!cap.isOpened())//�����Ƶ�����������򷵻�
	{
		cout << "cannot open video!" << endl;
		return 0;
	}
	
	Mat frame_key;
	cap >> frame_key;
	Mat out;
	imshow("fram_1", frame_key);
	WaterShed(frame_key);
	MomentPreserving(frame_key);
	OtsuAlgThreshold(frame_key);
	EntropySeg(frame_key);
	myadaptive(frame_key, out, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 15, 10);
	IterationThreshold(frame_key);
	stringstream str;
	str << "keyframes_pixels_cha/" << currentFrame << ".jpg";
	cout << str.str() << endl;
	imwrite(str.str(), frame_key);
	Mat frame;
	Mat previousImage, currentImage, resultImage;
	while (1)
	{
		currentFrame++;
		Mat frame;
		cap >> frame;
		if (frame.empty())
		{
			cout << "frame is empty!" << endl;
			break;
		}
		imshow("fram_1", frame);
		
		waitKey(20);

		Mat srcImage_base, hsvImage_base;
		Mat srcImage_test1, hsvImage_test1;
		srcImage_base = frame_key;
		srcImage_test1 = frame;
		//��ͼ���BGRɫ�ʿռ�ת���� HSVɫ�ʿռ�
		cvtColor(srcImage_base, previousImage, CV_BGR2GRAY);
		cvtColor(srcImage_test1, currentImage, CV_BGR2GRAY);

		absdiff(currentImage, previousImage, resultImage);  //֡������

		threshold(resultImage, resultImage, 40, 255.0, CV_THRESH_BINARY); //��ֵ��������ֵ������20����Ϊ255������Ϊ0

		imshow("keyframe", frame_key);

		float counter = 0;
		float num = 0;
		// ͳ����֡�����ͼ����
		for (int i = 0; i < resultImage.rows; i++)
		{
			uchar* data = resultImage.ptr<uchar>(i); //��ȡÿһ�е�ָ��
			for (int j = 0; j < resultImage.cols; j++)
			{
				num = num + 1;
				if (data[j] == 255) //���ʵ�����ֵ
				{
					counter = counter + 1;
				}
			}
		}
		p = counter / num;
		
		

		if (p > 0.38) 
		{
			
			frame_key = frame;
			
			Mat out;
			imshow("keyframe", frame_key);
			WaterShed(frame_key);
			MomentPreserving(frame_key);
			OtsuAlgThreshold(frame_key);
			EntropySeg(frame_key);
			myadaptive(frame_key, out, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 15, 10);
			IterationThreshold(frame_key);
			
			stringstream str;

			str << "keyframes_pixels_cha/" << currentFrame << ".jpg";
			cout << str.str() << endl;
			imwrite(str.str(), frame_key);
		}

	}
}
