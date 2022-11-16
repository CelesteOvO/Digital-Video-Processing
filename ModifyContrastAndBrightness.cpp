////头文件
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <iostream>
//using namespace std;
//using namespace cv;
////宏定义
//#define CONTRAST_BAR "对比度"
//#define BRIGHT_BAR "亮  度"
//#define WINDOW_NAME "【效果图窗口】"
//#define MAX_CONTRAST_VALUE 300
//#define MAX_BRIGHT_VALUE 200
// 
////全局函数声明
//static void on_ContrastAndBright(int, void *);//回调函数原型必须是void XXXX(int,void*)
////其中第一个参数是轨迹条的位置，第二个参数是用户数据
// 
////全局变量声明
//int g_nContrastValue;//对比度值
//int g_nBrightValue;//亮度值
//Mat g_srcImage, g_dstImage;
// 
//int main(int argc, char** argv) {
//	
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
//	
//	while (1) {
//		Mat frame;
//		cap >> frame;
//		//读取输入图像
//		g_srcImage = frame;
//		g_dstImage = Mat::zeros(g_srcImage.size(), g_srcImage.type());
//
//		//设定对比度和亮度初值
//		g_nContrastValue = 100;//a=100*0.01=1
//		g_nBrightValue = 0;//b=0
//
//		//创建效果图窗口
//		namedWindow(WINDOW_NAME, 1);
//
//		//创建轨迹条
//		createTrackbar(CONTRAST_BAR, WINDOW_NAME, &g_nContrastValue, MAX_CONTRAST_VALUE, on_ContrastAndBright);
//		createTrackbar(BRIGHT_BAR, WINDOW_NAME, &g_nBrightValue, MAX_BRIGHT_VALUE, on_ContrastAndBright);
//
//		//进行回调函数初始化
//		on_ContrastAndBright(g_nContrastValue, 0);//传给回调函数对比度滑块初值，userdata = 0
//		on_ContrastAndBright(g_nBrightValue, 0);//传给回调函数亮度滑块初值，userdata = 0
//
//		//按下Q键是程序退出
//		while (char(waitKey(1)) != 'q') {}
//		//waitKey(0);
//	}
//	return 0;
//}
// 
////------------------------【on_ContrastAndBright()函数】-------------
//static void on_ContrastAndBright(int, void *) {
//	//创建窗口
//	namedWindow("原始图窗口", 1);
//	//3个for循环，执行运算g_dstImage(i,j) = a * g_srcImage(i,j) + b
//	for (int y = 0; y < g_srcImage.rows; y++) {
//		for (int x = 0; x < g_srcImage.cols; x++) {
//			for (int c = 0; c < 3; c++) {
//				g_dstImage.at<Vec3b>(y, x)[c] = saturate_cast<uchar>((g_nContrastValue*0.01)*(g_srcImage.at<Vec3b>(y, x)[c]) + g_nBrightValue);
//			}
//		}
//	}
// 
//	//显示图像
//	imshow("原始图窗口", g_srcImage);
//	imshow(WINDOW_NAME, g_dstImage);
//}