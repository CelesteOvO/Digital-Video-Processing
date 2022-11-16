//#include<iostream>
//#include<opencv2/opencv.hpp>
//#include<ctime>
//using namespace std;
//using namespace cv;
//
//cv::Mat ideal_low_kernel(cv::Mat& scr, float sigma);
//cv::Mat ideal_low_pass_filter(cv::Mat& src, float sigma);
//cv::Mat ideal_high_kernel(cv::Mat& scr, float sigma);
//cv::Mat ideal_high_pass_filter(cv::Mat& src, float sigma);
//cv::Mat frequency_filter(cv::Mat& scr, cv::Mat& blur);
//cv::Mat image_make_border(cv::Mat& src);
//void fftshift(cv::Mat& plane0, cv::Mat& plane1);
//
//int main(void)
//{
//	VideoCapture cap;
//	cap.open("exp2.avi");
//	Mat frame, test;
//	cap >> frame;
//	cvtColor(frame, test, CV_RGB2GRAY);
//	//Mat test = imread("test1.png", 0);
//	float D0 = 50.0f;
//	float D1 = 5.0f;
//	Mat lowpass = ideal_low_pass_filter(test, D0);
//	Mat highpass = ideal_high_pass_filter(test, D1);
//
//	imshow("original", test);
//	imshow("low pass", lowpass / 255);
//	imshow("high pass", highpass / 255);
//	waitKey(0);
//
//	system("pause");
//	return 0;
//}
//
//// 理想低通滤波器
//cv::Mat ideal_low_pass_filter(cv::Mat& src, float sigma)
//{
//	cv::Mat padded = image_make_border(src);
//	cv::Mat ideal_kernel = ideal_low_kernel(padded, sigma);
//	cv::Mat result = frequency_filter(padded, ideal_kernel);
//	return result;
//}
//
//// 理想低通滤波核函数
//cv::Mat ideal_low_kernel(cv::Mat& scr, float sigma)
//{
//	cv::Mat ideal_low_pass(scr.size(), CV_32FC1); //，CV_32FC1
//	float d0 = sigma;//半径D0越小，模糊越大；半径D0越大，模糊越小
//	for (int i = 0; i < scr.rows; i++) {
//		for (int j = 0; j < scr.cols; j++) {
//			float d = sqrt(pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2));//分子,计算pow必须为float型
//			if (d <= d0) {
//				ideal_low_pass.at<float>(i, j) = 1;
//			}
//			else {
//				ideal_low_pass.at<float>(i, j) = 0;
//			}
//		}
//	}
//	return ideal_low_pass;
//}
//
//// 理想高通滤波核函数
//cv::Mat ideal_high_kernel(cv::Mat& scr, float sigma)
//{
//	cv::Mat ideal_high_pass(scr.size(), CV_32FC1); //，CV_32FC1
//	float d0 = sigma;//半径D0越小，模糊越大；半径D0越大，模糊越小
//	for (int i = 0; i < scr.rows; i++) {
//		for (int j = 0; j < scr.cols; j++) {
//			float d = sqrt(pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2));//分子,计算pow必须为float型
//			if (d <= d0) {
//				ideal_high_pass.at<float>(i, j) = 0;
//			}
//			else {
//				ideal_high_pass.at<float>(i, j) = 1;
//			}
//		}
//	}
//	return ideal_high_pass;
//}
//
//// 理想高通滤波
//cv::Mat ideal_high_pass_filter(cv::Mat& src, float sigma)
//{
//	cv::Mat padded = image_make_border(src);
//	cv::Mat ideal_kernel = ideal_high_kernel(padded, sigma);
//	cv::Mat result = frequency_filter(padded, ideal_kernel);
//	return result;
//}
//
//// 频率域滤波
//cv::Mat frequency_filter(cv::Mat& scr, cv::Mat& blur)
//{
//	cv::Mat mask = scr == scr;
//	scr.setTo(0.0f, ~mask);
//
//	//创建通道，存储dft后的实部与虚部（CV_32F，必须为单通道数）
//	cv::Mat plane[] = { scr.clone(), cv::Mat::zeros(scr.size() , CV_32FC1) };
//
//	cv::Mat complexIm;
//	cv::merge(plane, 2, complexIm); // 合并通道 （把两个矩阵合并为一个2通道的Mat类容器）
//	cv::dft(complexIm, complexIm); // 进行傅立叶变换，结果保存在自身
//
//	// 分离通道（数组分离）
//	cv::split(complexIm, plane);
//
//	// 以下的操作是频域迁移
//	fftshift(plane[0], plane[1]);
//
//	// *****************滤波器函数与DFT结果的乘积****************
//	cv::Mat blur_r, blur_i, BLUR;
//	cv::multiply(plane[0], blur, blur_r);  // 滤波（实部与滤波器模板对应元素相乘）
//	cv::multiply(plane[1], blur, blur_i);  // 滤波（虚部与滤波器模板对应元素相乘）
//	cv::Mat plane1[] = { blur_r, blur_i };
//
//	// 再次搬移回来进行逆变换
//	fftshift(plane1[0], plane1[1]);
//	cv::merge(plane1, 2, BLUR); // 实部与虚部合并
//
//	cv::idft(BLUR, BLUR);       // idft结果也为复数
//	BLUR = BLUR / BLUR.rows / BLUR.cols;
//
//	cv::split(BLUR, plane);//分离通道，主要获取通道
//
//	return plane[0];
//}
//
//// 图像边界处理
//cv::Mat image_make_border(cv::Mat& src)
//{
//	int w = cv::getOptimalDFTSize(src.cols); // 获取DFT变换的最佳宽度
//	int h = cv::getOptimalDFTSize(src.rows); // 获取DFT变换的最佳高度
//
//	cv::Mat padded;
//	// 常量法扩充图像边界，常量 = 0
//	cv::copyMakeBorder(src, padded, 0, h - src.rows, 0, w - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
//	padded.convertTo(padded, CV_32FC1);
//
//	return padded;
//}
//
//// fft变换后进行频谱搬移
//void fftshift(cv::Mat& plane0, cv::Mat& plane1)
//{
//	// 以下的操作是移动图像  (零频移到中心)
//	int cx = plane0.cols / 2;
//	int cy = plane0.rows / 2;
//	cv::Mat part1_r(plane0, cv::Rect(0, 0, cx, cy));  // 元素坐标表示为(cx, cy)
//	cv::Mat part2_r(plane0, cv::Rect(cx, 0, cx, cy));
//	cv::Mat part3_r(plane0, cv::Rect(0, cy, cx, cy));
//	cv::Mat part4_r(plane0, cv::Rect(cx, cy, cx, cy));
//
//	cv::Mat temp;
//	part1_r.copyTo(temp);  //左上与右下交换位置(实部)
//	part4_r.copyTo(part1_r);
//	temp.copyTo(part4_r);
//
//	part2_r.copyTo(temp);  //右上与左下交换位置(实部)
//	part3_r.copyTo(part2_r);
//	temp.copyTo(part3_r);
//
//	cv::Mat part1_i(plane1, cv::Rect(0, 0, cx, cy));  //元素坐标(cx,cy)
//	cv::Mat part2_i(plane1, cv::Rect(cx, 0, cx, cy));
//	cv::Mat part3_i(plane1, cv::Rect(0, cy, cx, cy));
//	cv::Mat part4_i(plane1, cv::Rect(cx, cy, cx, cy));
//
//	part1_i.copyTo(temp);  //左上与右下交换位置(虚部)
//	part4_i.copyTo(part1_i);
//	temp.copyTo(part4_i);
//
//	part2_i.copyTo(temp);  //右上与左下交换位置(虚部)
//	part3_i.copyTo(part2_i);
//	temp.copyTo(part3_i);
//}