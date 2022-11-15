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

Vec3b RandomColor(int value) //<span style = "line-height: 20.8px; font-family: sans-serif;" >//生成随机颜色函数</span>
{
	value = value % 255;  //生成0~255的随机数
	RNG rng;
	int aa = rng.uniform(0, value);
	int bb = rng.uniform(0, value);
	int cc = rng.uniform(0, value);
	return Vec3b(aa, bb, cc);
}

void WaterShed(Mat image)
{
	//灰度化，滤波，Canny边缘检测
	Mat imageGray;
	cvtColor(image, imageGray, CV_RGB2GRAY);//灰度转换
	GaussianBlur(imageGray, imageGray, Size(3, 3), 2);   //高斯滤波
	//imshow("Gray Image", imageGray);
	Canny(imageGray, imageGray, 80, 150);
	//imshow("Canny Image", imageGray);

	//查找轮廓
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imageGray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	Mat imageContours = Mat::zeros(image.size(), CV_8UC1);  //轮廓	
	Mat marks(image.size(), CV_32S);   //Opencv分水岭第二个矩阵参数
	marks = Scalar::all(0);
	int index = 0;
	int compCount = 0;
	for (; index >= 0; index = hierarchy[index][0], compCount++)
	{
		//对marks进行标记，对不同区域的轮廓进行编号，相当于设置注水点，有多少轮廓，就有多少注水点
		drawContours(marks, contours, index, Scalar::all(compCount + 1), 1, 8, hierarchy);
		drawContours(imageContours, contours, index, Scalar(255), 1, 8, hierarchy);
	}

	//我们来看一下传入的矩阵marks里是什么东西
	Mat marksShows;
	convertScaleAbs(marks, marksShows);
	//imshow("marksShow", marksShows);
	//imshow("轮廓", imageContours);
	watershed(image, marks);

	//我们再来看一下分水岭算法之后的矩阵marks里是什么东西
	Mat afterWatershed;
	convertScaleAbs(marks, afterWatershed);
	//imshow("After Watershed", afterWatershed);

	//对每一个区域进行颜色填充
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

	//分割并填充颜色的结果跟原始图像融合
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
	// 转为灰度图像
	int pixel = img.rows * img.cols;
	// 计算图像的像素总数
	int height = img.rows;
	int width = img.cols;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			P_sum1 += (int)img.at<uchar>(i, j); // P_sum1是图片在（i，j）处的灰度值之和
		}
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			P_sum2 += ((int)img.at<uchar>(i, j) * (int)img.at<uchar>(i, j));// P_sum2是图片在（i，j）处的灰度值的平方和
		}
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			P_sum3 += ((int)img.at<uchar>(i, j) * (int)img.at<uchar>(i, j) * (int)img.at<uchar>(i, j)); // P_sum3是在（i，j）处的灰度值的立方和
		}
	}

	int m1 = P_sum1 / (img.rows * img.cols); // m1是图像的一阶矩
	int m2 = P_sum2 / (img.rows * img.cols); // m2是图像的二阶矩
	int m3 = P_sum3 / (img.rows * img.cols); // m3是图像的三阶矩

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
	cout << "最优阈值为" << twhreshld << endl;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) >= twhreshld)
				img.at<uchar>(i, j) = 255;
			else
				img.at<uchar>(i, j) = 0;
		}
	}
	imshow("矩不变阈值法", img);
}

//Otsu阈值分割
Mat OtsuAlgThreshold(Mat image1)
{
	Mat image;
	cvtColor(image1, image, CV_BGR2GRAY);
	if (image.channels() != 1)
	{
		cout << "Please input Gray-image!" << endl;
	}
	int T = 0; //Otsu算法阈值  
	double varValue = 0; //类间方差中间值保存
	double w0 = 0; //前景像素点数所占比例  
	double w1 = 0; //背景像素点数所占比例  
	double u0 = 0; //前景平均灰度  
	double u1 = 0; //背景平均灰度  
	double Histogram[256] = { 0 }; //灰度直方图，下标是灰度值，保存内容是灰度值对应的像素点总数  
	uchar* data = image.data;

	double totalNum = image.rows * image.cols; //像素总数

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
		//每次遍历之前初始化各变量  
		w1 = 0;       u1 = 0;       w0 = 0;       u0 = 0;
		//***********背景各分量值计算**************************  
		for (int j = 0; j <= i; j++) //背景部分各值计算  
		{
			w1 += Histogram[j];   //背景部分像素点总数  
			u1 += j * Histogram[j]; //背景部分像素总灰度和  
		}
		if (w1 == 0) //背景部分像素点数为0时退出  
		{
			break;
		}
		u1 = u1 / w1; //背景像素平均灰度  
		w1 = w1 / totalNum; // 背景部分像素点数所占比例
		//***********背景各分量值计算**************************  

		//***********前景各分量值计算**************************  
		for (int k = i + 1; k < 255; k++)
		{
			w0 += Histogram[k];  //前景部分像素点总数  
			u0 += k * Histogram[k]; //前景部分像素总灰度和  
		}
		if (w0 == 0) //前景部分像素点数为0时退出  
		{
			break;
		}
		u0 = u0 / w0; //前景像素平均灰度  
		w0 = w0 / totalNum; // 前景部分像素点数所占比例  
		//***********前景各分量值计算**************************  

		//***********类间方差计算******************************  
		double varValueI = w0 * w1 * (u1 - u0) * (u1 - u0); //当前类间方差计算  
		if (varValue < varValueI)
		{
			varValue = varValueI;
			T = i;
		}
	}
	Mat dst;
	threshold(image, dst, T, 255, CV_THRESH_OTSU);
	imshow("Otsu阈值分割",dst);
	return dst;
}

//最大熵阈值分割法
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
	imshow("最大熵法阈值分割", dst);
	return dst;
}

//自适应阈值分割
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
			// 将[-255, 255] 映射到[0, 510]然后查表
			ddata[j] = tab[sdata[j] - mdata[j] + 255];
	}
	imshow("自适应阈值分割", dst);
}

//迭代阈值分割
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
	imshow("迭代法阈值分割",dst);
	return dst;
}

int main()
{
	system("color 2F");
	long currentFrame = 1;
	float p;
	VideoCapture cap;
	//这里放置需要提取关键字的视频
	cap.open("exp1(1).avi");
	if (!cap.isOpened())//如果视频不能正常打开则返回
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
		//将图像从BGR色彩空间转换到 HSV色彩空间
		cvtColor(srcImage_base, previousImage, CV_BGR2GRAY);
		cvtColor(srcImage_test1, currentImage, CV_BGR2GRAY);

		absdiff(currentImage, previousImage, resultImage);  //帧差法，相减

		threshold(resultImage, resultImage, 40, 255.0, CV_THRESH_BINARY); //二值化，像素值相差大于20则置为255，其余为0

		imshow("keyframe", frame_key);

		float counter = 0;
		float num = 0;
		// 统计两帧相减后图像素
		for (int i = 0; i < resultImage.rows; i++)
		{
			uchar* data = resultImage.ptr<uchar>(i); //获取每一行的指针
			for (int j = 0; j < resultImage.cols; j++)
			{
				num = num + 1;
				if (data[j] == 255) //访问到像素值
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
