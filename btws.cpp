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
//	imshow("low pass", lowpass / 255);     // lowpass�����ݶ��Ƚϴ�0-255��imshow����float��Mat��ʾ��Ҫ����255
//	imshow("high pass", highpass / 255);   // highpass�����ݶ��Ƚϴ�0-255��imshow����float��Mat��ʾ��Ҫ����255
//	waitKey(0);
//
//	system("pause");
//	return 0;
//}
//
//// ������˹��ͨ�˲��˺���
//cv::Mat butterworth_low_kernel(cv::Mat& scr, float sigma, int n)
//{
//	cv::Mat butterworth_low_pass(scr.size(), CV_32FC1); //��CV_32FC1
//	float D0 = sigma;//�뾶D0ԽС��ģ��Խ�󣻰뾶D0Խ��ģ��ԽС
//	for (int i = 0; i < scr.rows; i++) {
//		for (int j = 0; j < scr.cols; j++) {
//			float d = sqrt(pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2));//����,����pow����Ϊfloat��
//			butterworth_low_pass.at<float>(i, j) = 1.0f / (1.0f + pow(d / D0, 2 * n));
//		}
//	}
//	return butterworth_low_pass;
//}
//
//// ������˹��ͨ�˲�
//cv::Mat butterworth_low_pass_filter(cv::Mat& src, float d0, int n)
//{
//	// H = 1 / (1+(D/D0)^2n)   n��ʾ������˹�˲����Ĵ���
//	// ����n=1 ������͸�ֵ    ����n=2 ��΢����͸�ֵ  ����n=5 ��������͸�ֵ   ����n=20 ��ILPF����
//	cv::Mat padded = image_make_border(src);
//	cv::Mat butterworth_kernel = butterworth_low_kernel(padded, d0, n);
//	cv::Mat result = frequency_filter(padded, butterworth_kernel);
//	return result;
//}
//
//// ������˹��ͨ�˲��˺���
//cv::Mat butterworth_high_kernel(cv::Mat& scr, float sigma, int n)
//{
//	cv::Mat butterworth_high_pass(scr.size(), CV_32FC1); //��CV_32FC1
//	float D0 = (float)sigma;  // �뾶D0ԽС��ģ��Խ�󣻰뾶D0Խ��ģ��ԽС
//	for (int i = 0; i < scr.rows; i++) {
//		for (int j = 0; j < scr.cols; j++) {
//			float d = sqrt(pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2));//����,����pow����Ϊfloat��
//			butterworth_high_pass.at<float>(i, j) = 1.0f - 1.0f / (1.0f + pow(d / D0, 2 * n));
//		}
//	}
//	return butterworth_high_pass;
//}
//
//// ������˹��ͨ�˲�
//cv::Mat butterworth_high_pass_filter(cv::Mat& src, float d0, int n)
//{
//	cv::Mat padded = image_make_border(src);
//	cv::Mat butterworth_kernel = butterworth_high_kernel(padded, d0, n);
//	cv::Mat result = frequency_filter(padded, butterworth_kernel);
//	return result;
//}
//
//// Ƶ�����˲�
//cv::Mat frequency_filter(cv::Mat& scr, cv::Mat& blur)
//{
//	cv::Mat mask = scr == scr;
//	scr.setTo(0.0f, ~mask);
//
//	//����ͨ�����洢dft���ʵ�����鲿��CV_32F������Ϊ��ͨ������
//	cv::Mat plane[] = { scr.clone(), cv::Mat::zeros(scr.size() , CV_32FC1) };
//
//	cv::Mat complexIm;
//	cv::merge(plane, 2, complexIm); // �ϲ�ͨ�� ������������ϲ�Ϊһ��2ͨ����Mat��������
//	cv::dft(complexIm, complexIm); // ���и���Ҷ�任���������������
//
//	// ����ͨ����������룩
//	cv::split(complexIm, plane);
//
//	// ���µĲ�����Ƶ��Ǩ��
//	fftshift(plane[0], plane[1]);
//
//	// *****************�˲���������DFT����ĳ˻�****************
//	cv::Mat blur_r, blur_i, BLUR;
//	cv::multiply(plane[0], blur, blur_r);  // �˲���ʵ�����˲���ģ���ӦԪ����ˣ�
//	cv::multiply(plane[1], blur, blur_i);  // �˲����鲿���˲���ģ���ӦԪ����ˣ�
//	cv::Mat plane1[] = { blur_r, blur_i };
//
//	// �ٴΰ��ƻ���������任
//	fftshift(plane1[0], plane1[1]);
//	cv::merge(plane1, 2, BLUR); // ʵ�����鲿�ϲ�
//
//	cv::idft(BLUR, BLUR);       // idft���ҲΪ����
//	BLUR = BLUR / BLUR.rows / BLUR.cols;
//
//	cv::split(BLUR, plane);//����ͨ������Ҫ��ȡͨ��
//
//	return plane[0];
//}
//
//// ͼ��߽紦��
//cv::Mat image_make_border(cv::Mat& src)
//{
//	int w = cv::getOptimalDFTSize(src.cols); // ��ȡDFT�任����ѿ��
//	int h = cv::getOptimalDFTSize(src.rows); // ��ȡDFT�任����Ѹ߶�
//
//	cv::Mat padded;
//	// ����������ͼ��߽磬���� = 0
//	cv::copyMakeBorder(src, padded, 0, h - src.rows, 0, w - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
//	padded.convertTo(padded, CV_32FC1);
//
//	return padded;
//}
//
//// ʵ��Ƶ���˲�����������
//void getcart(int rows, int cols, cv::Mat& x, cv::Mat& y) {
//	x.create(rows, cols, CV_32FC1);
//	y.create(rows, cols, CV_32FC1);
//	//���ñ߽�
//
//	//��������λ�õ�ֵ
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
//// fft�任�����Ƶ�װ���
//void fftshift(cv::Mat& plane0, cv::Mat& plane1)
//{
//	// ���µĲ������ƶ�ͼ��  (��Ƶ�Ƶ�����)
//	int cx = plane0.cols / 2;
//	int cy = plane0.rows / 2;
//	cv::Mat part1_r(plane0, cv::Rect(0, 0, cx, cy));  // Ԫ�������ʾΪ(cx, cy)
//	cv::Mat part2_r(plane0, cv::Rect(cx, 0, cx, cy));
//	cv::Mat part3_r(plane0, cv::Rect(0, cy, cx, cy));
//	cv::Mat part4_r(plane0, cv::Rect(cx, cy, cx, cy));
//
//	cv::Mat temp;
//	part1_r.copyTo(temp);  //���������½���λ��(ʵ��)
//	part4_r.copyTo(part1_r);
//	temp.copyTo(part4_r);
//
//	part2_r.copyTo(temp);  //���������½���λ��(ʵ��)
//	part3_r.copyTo(part2_r);
//	temp.copyTo(part3_r);
//
//	cv::Mat part1_i(plane1, cv::Rect(0, 0, cx, cy));  //Ԫ������(cx,cy)
//	cv::Mat part2_i(plane1, cv::Rect(cx, 0, cx, cy));
//	cv::Mat part3_i(plane1, cv::Rect(0, cy, cx, cy));
//	cv::Mat part4_i(plane1, cv::Rect(cx, cy, cx, cy));
//
//	part1_i.copyTo(temp);  //���������½���λ��(�鲿)
//	part4_i.copyTo(part1_i);
//	temp.copyTo(part4_i);
//
//	part2_i.copyTo(temp);  //���������½���λ��(�鲿)
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
