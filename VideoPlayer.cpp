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
int g_flag = false; //是否进行直方图均衡化
int g_boxFiltering = false; // 是否进行方框滤波
int g_averageFiltering = false; // 是否进行均值滤波
int g_gaussianFiltering = false; // 是否进行高斯滤波
int g_lowPassFiltering = false; // 是否进行低通滤波
int g_highPassFiltering = false; // 是否进行高通滤波
int g_medianFiltering = false; // 是否进行中值滤波

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

Mat showHistogram(Mat img) 
{
	// 创建3个矩阵来处理每个通道输入图像通道
	vector<Mat>bgr;
	split(img, bgr);

	//定义直方图的区间数
	int numbers = 256;
	
	//定义变量范围并创建3个矩阵来存储每个直方图
	float range[] = { 0,255 };
	const float* histRange = { range };
	Mat b_hist, g_hist, r_hist;
	
	//计算直方图
	calcHist(&bgr[0], 1, 0, Mat(), b_hist, 1, &numbers, &histRange, true, false);
	calcHist(&bgr[1], 1, 0, Mat(), g_hist, 1, &numbers, &histRange, true, false);
	calcHist(&bgr[2], 1, 0, Mat(), r_hist, 1, &numbers, &histRange, true, false);
	
	//定义直方图的宽度和高度
	int width = 512, height = 400;
	Mat histImage(height, width, CV_8UC3, Scalar(20, 20, 20));
	
	//定义直方图的最大值
	normalize(b_hist, b_hist, 0, height, NORM_MINMAX);
	normalize(g_hist, g_hist, 0, height, NORM_MINMAX);
	normalize(r_hist, r_hist, 0, height, NORM_MINMAX);
	
	//使用彩色通道绘制直方图
	int binStep = cvRound((float)width / (float)numbers);
	
	for (int i = 1; i < numbers; i++) {
		line(histImage, Point(binStep * (i - 1), height - cvRound(b_hist.at<float>(i - 1))),
			Point(binStep * (i), height - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(binStep * (i - 1), height - cvRound(g_hist.at<float>(i - 1))),
			Point(binStep * (i), height - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(binStep * (i - 1), height - cvRound(r_hist.at<float>(i - 1))),
			Point(binStep * (i), height - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}
	return histImage;
}
void on_HistogramEqualization(int, void*)
{
	if(g_flag == 0)
		return;
	Mat result, ycrcb;
	
	cvtColor(g_srcImage, ycrcb, COLOR_BGR2YCrCb);
	
	//将图像转换为单通道
	vector<Mat> channels;
	split(ycrcb, channels);
	
	//直方图均衡化
	equalizeHist(channels[0], channels[0]);
	
	//合并通道
	merge(channels, ycrcb);
	cvtColor(ycrcb, result, COLOR_YCrCb2BGR);
	
	g_srcImage = result;
}

void on_BoxFiltering(int, void*) {
	if (g_boxFiltering == 0)
		return;
	Mat result;
	boxFilter(g_srcImage, result, -1, Size(3, 3), Point(-1, -1), true);
	g_srcImage = result;
}

void on_averageFiltering(int, void*) {
	if (g_averageFiltering == 0)
		return;
	Mat result;
	blur(g_srcImage, result, Size(3, 3), Point(-1, -1)); // 3x3卷积核
	//blur(g_srcImage, result, Size(9, 9), Point(-1, -1)); // 9x9卷积核
	g_srcImage = result;
}

void on_gaussianFiltering(int, void*)
{
	if (g_gaussianFiltering == 0)
		return;
	Mat result;
	GaussianBlur(g_srcImage, result, Size(3, 3), 0, 0); // 3x3卷积核
	g_srcImage = result;
}

Mat image_make_border(Mat& src)
{
	int w = getOptimalDFTSize(src.cols); // 获取DFT变换的最佳宽度
	int h = getOptimalDFTSize(src.rows); // 获取DFT变换的最佳高度

	Mat padded;
	// 常量法扩充图像边界，常量 = 0
	copyMakeBorder(src, padded, 0, h - src.rows, 0, w - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	padded.convertTo(padded, CV_32FC1);

	return padded;
}
Mat ideal_low_kernel(Mat& scr, float sigma)
{
	Mat ideal_low_pass(scr.size(), CV_32FC1); //，CV_32FC1
	float d0 = sigma;//半径D0越小，模糊越大；半径D0越大，模糊越小
	for (int i = 0; i < scr.rows; i++) {
		for (int j = 0; j < scr.cols; j++) {
			float d = sqrt(pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2));//分子,计算pow必须为float型
			if (d <= d0) {
				ideal_low_pass.at<float>(i, j) = 1;
			}
			else {
				ideal_low_pass.at<float>(i, j) = 0;
			}
		}
	}
	return ideal_low_pass;
}
void fftshift(Mat& plane0, Mat& plane1)
{
	// 以下的操作是移动图像  (零频移到中心)
	int cx = plane0.cols / 2;
	int cy = plane0.rows / 2;
	Mat part1_r(plane0, Rect(0, 0, cx, cy));  // 元素坐标表示为(cx, cy)
	Mat part2_r(plane0, Rect(cx, 0, cx, cy));
	Mat part3_r(plane0, Rect(0, cy, cx, cy));
	Mat part4_r(plane0, Rect(cx, cy, cx, cy));

	Mat temp;
	part1_r.copyTo(temp);  //左上与右下交换位置(实部)
	part4_r.copyTo(part1_r);
	temp.copyTo(part4_r);

	part2_r.copyTo(temp);  //右上与左下交换位置(实部)
	part3_r.copyTo(part2_r);
	temp.copyTo(part3_r);

	Mat part1_i(plane1, Rect(0, 0, cx, cy));  //元素坐标(cx,cy)
	Mat part2_i(plane1, Rect(cx, 0, cx, cy));
	Mat part3_i(plane1, Rect(0, cy, cx, cy));
	Mat part4_i(plane1, Rect(cx, cy, cx, cy));

	part1_i.copyTo(temp);  //左上与右下交换位置(虚部)
	part4_i.copyTo(part1_i);
	temp.copyTo(part4_i);

	part2_i.copyTo(temp);  //右上与左下交换位置(虚部)
	part3_i.copyTo(part2_i);
	temp.copyTo(part3_i);
}
Mat frequency_filter(Mat& scr, Mat& blur)
{
	Mat mask = scr == scr;
	scr.setTo(0.0f, ~mask);

	//创建通道，存储dft后的实部与虚部（CV_32F，必须为单通道数）
	Mat plane[] = { scr.clone(), Mat::zeros(scr.size() , CV_32FC1) };

	Mat complexIm;
	merge(plane, 2, complexIm); // 合并通道 （把两个矩阵合并为一个2通道的Mat类容器）
	dft(complexIm, complexIm); // 进行傅立叶变换，结果保存在自身

	// 分离通道（数组分离）
	split(complexIm, plane);

	// 以下的操作是频域迁移
	fftshift(plane[0], plane[1]);

	// *****************滤波器函数与DFT结果的乘积****************
	Mat blur_r, blur_i, BLUR;
	multiply(plane[0], blur, blur_r);  // 滤波（实部与滤波器模板对应元素相乘）
	multiply(plane[1], blur, blur_i);  // 滤波（虚部与滤波器模板对应元素相乘）
	Mat plane1[] = { blur_r, blur_i };

	// 再次搬移回来进行逆变换
	fftshift(plane1[0], plane1[1]);
	merge(plane1, 2, BLUR); // 实部与虚部合并

	idft(BLUR, BLUR);       // idft结果也为复数
	BLUR = BLUR / BLUR.rows / BLUR.cols;

	split(BLUR, plane);//分离通道，主要获取通道

	return plane[0];
}
Mat ideal_low_pass_filter(Mat& src, float sigma)
{
	Mat padded = image_make_border(src);
	Mat ideal_kernel = ideal_low_kernel(padded, sigma);
	Mat result = frequency_filter(padded, ideal_kernel);
	return result;
}
void on_lowPassFiltering(int, void*)
{
	if (g_lowPassFiltering == 0)
		return; 
	Mat result, gray;
	cvtColor(g_srcImage, gray, CV_RGB2GRAY);
	result = ideal_low_pass_filter(gray, 50.0f);
	g_srcImage = result / 255;
}


Mat ideal_high_kernel(Mat& scr, float sigma)
{
	Mat ideal_high_pass(scr.size(), CV_32FC1); //，CV_32FC1
	float d0 = sigma;//半径D0越小，模糊越大；半径D0越大，模糊越小
	for (int i = 0; i < scr.rows; i++) {
		for (int j = 0; j < scr.cols; j++) {
			float d = sqrt(pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2));//分子,计算pow必须为float型
			if (d <= d0) {
				ideal_high_pass.at<float>(i, j) = 0;
			}
			else {
				ideal_high_pass.at<float>(i, j) = 1;
			}
		}
	}
	return ideal_high_pass;
}
Mat ideal_high_pass_filter(Mat& src, float sigma)
{
	Mat padded = image_make_border(src);
	Mat ideal_kernel = ideal_high_kernel(padded, sigma);
	Mat result = frequency_filter(padded, ideal_kernel);
	return result;
}
void on_highPassFiltering(int, void*)
{
	if (g_highPassFiltering == 0)
		return;
	Mat result, gray;
	cvtColor(g_srcImage, gray, CV_RGB2GRAY);
	result = ideal_high_pass_filter(gray, 50.0f);
	g_srcImage = result / 255;
}

void on_medianFiltering(int, void*)
{
	if (g_medianFiltering == 0)
		return;
	Mat result;
	medianBlur(g_srcImage, result, 3); // 3x3卷积核
	g_srcImage = result;
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
		resizeWindow("视频播放器", 800, 200);
		
		//控制视频播放进度
		createTrackbar("frame", "视频播放器", &g_pos, frame_count, func); //通过拖动滑动条控制视频播放画面
		
		//调节当前视频的对比度与亮度
		g_srcImage = frame;
		
		//设定对比度和亮度初值

		createTrackbar("对比度", "视频播放器", &g_nContrastValue, MAX_CONTRAST_VALUE, on_ContrastAndBright);
		createTrackbar("亮度", "视频播放器", &g_nBrightValue, MAX_BRIGHT_VALUE, on_ContrastAndBright);

		//进行回调函数初始化
		on_ContrastAndBright(g_nContrastValue, 0);//传给回调函数对比度滑块初值，userdata = 0
		on_ContrastAndBright(g_nBrightValue, 0);//传给回调函数亮度滑块初值，userdata = 0
		
		//直方图均衡化
		Mat before, after;
		before = showHistogram(g_srcImage);
		/*imshow("直方图均衡化前", before);*/
		createTrackbar("直方图均衡化", "视频播放器", &g_flag, 1, on_HistogramEqualization);
		
		on_HistogramEqualization(g_flag,0);
		
		after = showHistogram(g_srcImage);
		/*imshow("直方图均衡化后", after);*/
		
		//视频增强
		//方框滤波
		createTrackbar("方框滤波", "视频播放器", &g_boxFiltering, 1, on_BoxFiltering);
		on_BoxFiltering(g_boxFiltering, 0);
		
		//均值滤波
		createTrackbar("均值滤波", "视频播放器", &g_averageFiltering, 1, on_averageFiltering);
		on_averageFiltering(g_averageFiltering, 0);

		//高斯滤波
		createTrackbar("高斯滤波", "视频播放器", &g_gaussianFiltering, 1, on_gaussianFiltering);
		on_gaussianFiltering(g_gaussianFiltering, 0);
		
		//中值滤波
		createTrackbar("中值滤波", "视频播放器", &g_medianFiltering, 1, on_medianFiltering);
		on_medianFiltering(g_medianFiltering, 0);
		
		//理想低通滤波
		createTrackbar("理想低通滤波", "视频播放器", &g_lowPassFiltering, 1, on_lowPassFiltering);
		on_lowPassFiltering(g_lowPassFiltering, 0);

		//理想高通滤波
		createTrackbar("理想高通滤波", "视频播放器", &g_highPassFiltering, 1, on_highPassFiltering);
		on_highPassFiltering(g_highPassFiltering, 0);
		
		frame = g_srcImage;
		imshow("视频播放器", frame);
	}

	cap.release();
	destroyAllWindows();
	system("pause");
	return 0;
}

