////ͷ�ļ�
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <iostream>
//using namespace std;
//using namespace cv;
////�궨��
//#define CONTRAST_BAR "�Աȶ�"
//#define BRIGHT_BAR "��  ��"
//#define WINDOW_NAME "��Ч��ͼ���ڡ�"
//#define MAX_CONTRAST_VALUE 300
//#define MAX_BRIGHT_VALUE 200
// 
////ȫ�ֺ�������
//static void on_ContrastAndBright(int, void *);//�ص�����ԭ�ͱ�����void XXXX(int,void*)
////���е�һ�������ǹ켣����λ�ã��ڶ����������û�����
// 
////ȫ�ֱ�������
//int g_nContrastValue;//�Աȶ�ֵ
//int g_nBrightValue;//����ֵ
//Mat g_srcImage, g_dstImage;
// 
//int main(int argc, char** argv) {
//	
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
//	
//	while (1) {
//		Mat frame;
//		cap >> frame;
//		//��ȡ����ͼ��
//		g_srcImage = frame;
//		g_dstImage = Mat::zeros(g_srcImage.size(), g_srcImage.type());
//
//		//�趨�ԱȶȺ����ȳ�ֵ
//		g_nContrastValue = 100;//a=100*0.01=1
//		g_nBrightValue = 0;//b=0
//
//		//����Ч��ͼ����
//		namedWindow(WINDOW_NAME, 1);
//
//		//�����켣��
//		createTrackbar(CONTRAST_BAR, WINDOW_NAME, &g_nContrastValue, MAX_CONTRAST_VALUE, on_ContrastAndBright);
//		createTrackbar(BRIGHT_BAR, WINDOW_NAME, &g_nBrightValue, MAX_BRIGHT_VALUE, on_ContrastAndBright);
//
//		//���лص�������ʼ��
//		on_ContrastAndBright(g_nContrastValue, 0);//�����ص������ԱȶȻ����ֵ��userdata = 0
//		on_ContrastAndBright(g_nBrightValue, 0);//�����ص��������Ȼ����ֵ��userdata = 0
//
//		//����Q���ǳ����˳�
//		while (char(waitKey(1)) != 'q') {}
//		//waitKey(0);
//	}
//	return 0;
//}
// 
////------------------------��on_ContrastAndBright()������-------------
//static void on_ContrastAndBright(int, void *) {
//	//��������
//	namedWindow("ԭʼͼ����", 1);
//	//3��forѭ����ִ������g_dstImage(i,j) = a * g_srcImage(i,j) + b
//	for (int y = 0; y < g_srcImage.rows; y++) {
//		for (int x = 0; x < g_srcImage.cols; x++) {
//			for (int c = 0; c < 3; c++) {
//				g_dstImage.at<Vec3b>(y, x)[c] = saturate_cast<uchar>((g_nContrastValue*0.01)*(g_srcImage.at<Vec3b>(y, x)[c]) + g_nBrightValue);
//			}
//		}
//	}
// 
//	//��ʾͼ��
//	imshow("ԭʼͼ����", g_srcImage);
//	imshow(WINDOW_NAME, g_dstImage);
//}