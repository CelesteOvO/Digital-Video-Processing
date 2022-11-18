//#include<iostream>
//#include<opencv2/opencv.hpp>
//#include<ctime>
//using namespace std;
//using namespace cv;
//
//cv::Mat butterworth_low_kernel(cv::Mat& scr, float sigma, int n);
//cv::Mat butterworth_low_pass_filter(cv::Mat& src, float d0, int n);
//cv::Mat butterworth_high_kernel(cv::Mat& scr, float sigma, int n);
//cv::Mat butterworth_high_pass_filter(cv::Mat& src, float d0, int n);
//cv::Mat frequency_filter(cv::Mat& scr, cv::Mat& blur);
//cv::Mat image_make_border(cv::Mat& src);
//void fftshift(cv::Mat& plane0, cv::Mat& plane1);
//void getcart(int rows, int cols, cv::Mat& x, cv::Mat& y);
//Mat powZ(cv::InputArray src, double power);
//Mat sqrtZ(cv::InputArray src);
//
//int main(void)
//{
//	Mat test = imread("test1.png", 0);
//	float D0 = 50.0f;
//	float D1 = 5.0f;
//	Mat lowpass = butterworth_low_pass_filter(test, D0, 2);
//	Mat highpass = butterworth_high_pass_filter(test, D1, 2);
//
//	imshow("original", test);
//	imshow("low pass", lowpass / 255);     // lowpass的数据都比较大，0-255，imshow对于float型Mat显示需要除以255
//	imshow("high pass", highpass / 255);   // highpass的数据都比较大，0-255，imshow对于float型Mat显示需要除以255
//	waitKey(0);
//
//	system("pause");
//	return 0;
//}
//
//// 巴特沃斯低通滤波核函数
//cv::Mat butterworth_low_kernel(cv::Mat& scr, float sigma, int n)
//{
//	cv::Mat butterworth_low_pass(scr.size(), CV_32FC1); //，CV_32FC1
//	float D0 = sigma;//半径D0越小，模糊越大；半径D0越大，模糊越小
//	for (int i = 0; i < scr.rows; i++) {
//		for (int j = 0; j < scr.cols; j++) {
//			float d = sqrt(pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2));//分子,计算pow必须为float型
//			butterworth_low_pass.at<float>(i, j) = 1.0f / (1.0f + pow(d / D0, 2 * n));
//		}
//	}
//	return butterworth_low_pass;
//}
//
//// 巴特沃斯低通滤波
//cv::Mat butterworth_low_pass_filter(cv::Mat& src, float d0, int n)
//{
//	// H = 1 / (1+(D/D0)^2n)   n表示巴特沃斯滤波器的次数
//	// 阶数n=1 无振铃和负值    阶数n=2 轻微振铃和负值  阶数n=5 明显振铃和负值   阶数n=20 与ILPF相似
//	cv::Mat padded = image_make_border(src);
//	cv::Mat butterworth_kernel = butterworth_low_kernel(padded, d0, n);
//	cv::Mat result = frequency_filter(padded, butterworth_kernel);
//	return result;
//}
//
//// 巴特沃斯高通滤波核函数
//cv::Mat butterworth_high_kernel(cv::Mat& scr, float sigma, int n)
//{
//	cv::Mat butterworth_high_pass(scr.size(), CV_32FC1); //，CV_32FC1
//	float D0 = (float)sigma;  // 半径D0越小，模糊越大；半径D0越大，模糊越小
//	for (int i = 0; i < scr.rows; i++) {
//		for (int j = 0; j < scr.cols; j++) {
//			float d = sqrt(pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2));//分子,计算pow必须为float型
//			butterworth_high_pass.at<float>(i, j) = 1.0f - 1.0f / (1.0f + pow(d / D0, 2 * n));
//		}
//	}
//	return butterworth_high_pass;
//}
//
//// 巴特沃斯高通滤波
//cv::Mat butterworth_high_pass_filter(cv::Mat& src, float d0, int n)
//{
//	cv::Mat padded = image_make_border(src);
//	cv::Mat butterworth_kernel = butterworth_high_kernel(padded, d0, n);
//	cv::Mat result = frequency_filter(padded, butterworth_kernel);
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
//// 实现频域滤波器的网格函数
//void getcart(int rows, int cols, cv::Mat& x, cv::Mat& y) {
//	x.create(rows, cols, CV_32FC1);
//	y.create(rows, cols, CV_32FC1);
//	//设置边界
//
//	//计算其他位置的值
//	for (int i = 0; i < rows; ++i) {
//		if (i <= rows / 2) {
//			x.row(i) = i;
//		}
//		else {
//			x.row(i) = i - rows;
//		}
//	}
//	for (int i = 0; i < cols; ++i) {
//		if (i <= cols / 2) {
//			y.col(i) = i;
//		}
//		else {
//			y.col(i) = i - cols;
//		}
//	}
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
//
//Mat powZ(cv::InputArray src, double power) {
//	cv::Mat dst;
//	cv::pow(src, power, dst);
//	return dst;
//}
//
//Mat sqrtZ(cv::InputArray src) {
//	cv::Mat dst;
//	cv::sqrt(src, dst);
//	return dst;
//}
