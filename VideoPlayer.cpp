#include<iostream>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define MAX_CONTRAST_VALUE 300
#define MAX_BRIGHT_VALUE 200

//由于需要使用滑动条功能，故设置全局变量
VideoCapture cap;
int g_pos = 0; //当前视频帧索引
int g_nContrastValue = 100;//对比度值
int g_nBrightValue = 0;//亮度值

Mat g_srcImage;

void func(int, void*)
{
	cap.set(CAP_PROP_POS_FRAMES, g_pos);//将视频切换到当前帧
}

static void on_ContrastAndBright(int, void *) {
	//创建窗口
	//3个for循环，执行运算g_dstImage(i,j) = a * g_srcImage(i,j) + b
	for (int y = 0; y < g_srcImage.rows; y++) {
		for (int x = 0; x < g_srcImage.cols; x++) {
			for (int c = 0; c < 3; c++) {
				g_srcImage.at<Vec3b>(y, x)[c] = saturate_cast<uchar>((g_nContrastValue*0.01)*(g_srcImage.at<Vec3b>(y, x)[c]) + g_nBrightValue);
			}
		}
	}
}

int main()
{
	//读取视频文件
	cap.open("exp2.avi");
	if (!cap.isOpened())
	{
		cout << "can not open the video..." << endl;
		system("pause");
		return -1;
	}

	//获取视频帧数
	int frame_count = cap.get(CAP_PROP_FRAME_COUNT);

	Mat frame;
	while (cap.read(frame))
	{
		int key = waitKeyEx(30);  //键盘响应
		if (key == 27)
		{
			break; //按下ESC键退出循环、即视频播放结束
		}
		if (key == 32)
		{
			waitKey(0); //按下空格键暂停视频播放
		}
		if (key == 2424832)
		{
			//键盘←键向后快进
			g_pos -= 30;
			cap.set(CAP_PROP_POS_FRAMES, g_pos);
		}
		if (key == 2555904)
		{
			//键盘→键向前快进
			g_pos += 30;
			cap.set(CAP_PROP_POS_FRAMES, g_pos);
		}

		g_pos = cap.get(CAP_PROP_POS_FRAMES); //获取当前视频帧所在位置

		namedWindow("视频播放器", WINDOW_AUTOSIZE);
		createTrackbar("frame", "视频播放器", &g_pos, frame_count, func); //通过拖动滑动条控制视频播放画面
		
		createTrackbar("contrast", "视频播放器", &g_nContrastValue, MAX_CONTRAST_VALUE, on_ContrastAndBright);
		
		g_srcImage = frame;
		
		//设定对比度和亮度初值

		createTrackbar("contrast", "视频播放器", &g_nContrastValue, MAX_CONTRAST_VALUE, on_ContrastAndBright);
		createTrackbar("bright", "视频播放器", &g_nBrightValue, MAX_BRIGHT_VALUE, on_ContrastAndBright);

		//进行回调函数初始化
		on_ContrastAndBright(g_nContrastValue, 0);//传给回调函数对比度滑块初值，userdata = 0
		on_ContrastAndBright(g_nBrightValue, 0);//传给回调函数亮度滑块初值，userdata = 0
		
		frame = g_srcImage;
		
		imshow("视频播放器", g_srcImage);
	}

	cap.release();
	destroyAllWindows();
	system("pause");
	return 0;
}

