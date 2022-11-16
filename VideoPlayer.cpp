#include<iostream>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define MAX_CONTRAST_VALUE 300
#define MAX_BRIGHT_VALUE 200

//������Ҫʹ�û��������ܣ�������ȫ�ֱ���
VideoCapture cap;
int g_pos = 0; //��ǰ��Ƶ֡����
int g_nContrastValue = 100;//�Աȶ�ֵ
int g_nBrightValue = 0;//����ֵ
int g_flag = false; //�Ƿ����ֱ��ͼ���⻯
int g_boxFiltering = false; // �Ƿ���з����˲�
int g_averageFiltering = false; // �Ƿ���о�ֵ�˲�
int g_gaussianFiltering = false; // �Ƿ���и�˹�˲�
int g_lowPassFiltering = false; // �Ƿ���е�ͨ�˲�
int g_highPassFiltering = false; // �Ƿ���и�ͨ�˲�
int g_medianFiltering = false; // �Ƿ������ֵ�˲�

Mat g_srcImage;

void func(int, void*)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
{
	cap.set(CAP_PROP_POS_FRAMES, g_pos);//����Ƶ�л�����ǰ֡
}

static void on_ContrastAndBright(int, void *) {
	//��������
	//3��forѭ����ִ������g_dstImage(i,j) = a * g_srcImage(i,j) + b
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
	// ����3������������ÿ��ͨ������ͼ��ͨ��
	vector<Mat>bgr;
	split(img, bgr);

	//����ֱ��ͼ��������
	int numbers = 256;
	
	//���������Χ������3���������洢ÿ��ֱ��ͼ
	float range[] = { 0,255 };
	const float* histRange = { range };
	Mat b_hist, g_hist, r_hist;
	
	//����ֱ��ͼ
	calcHist(&bgr[0], 1, 0, Mat(), b_hist, 1, &numbers, &histRange, true, false);
	calcHist(&bgr[1], 1, 0, Mat(), g_hist, 1, &numbers, &histRange, true, false);
	calcHist(&bgr[2], 1, 0, Mat(), r_hist, 1, &numbers, &histRange, true, false);
	
	//����ֱ��ͼ�Ŀ�Ⱥ͸߶�
	int width = 512, height = 400;
	Mat histImage(height, width, CV_8UC3, Scalar(20, 20, 20));
	
	//����ֱ��ͼ�����ֵ
	normalize(b_hist, b_hist, 0, height, NORM_MINMAX);
	normalize(g_hist, g_hist, 0, height, NORM_MINMAX);
	normalize(r_hist, r_hist, 0, height, NORM_MINMAX);
	
	//ʹ�ò�ɫͨ������ֱ��ͼ
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
	
	//��ͼ��ת��Ϊ��ͨ��
	vector<Mat> channels;
	split(ycrcb, channels);
	
	//ֱ��ͼ���⻯
	equalizeHist(channels[0], channels[0]);
	
	//�ϲ�ͨ��
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
	blur(g_srcImage, result, Size(3, 3), Point(-1, -1)); // 3x3�����
	//blur(g_srcImage, result, Size(9, 9), Point(-1, -1)); // 9x9�����
	g_srcImage = result;
}

void on_gaussianFiltering(int, void*)
{
	if (g_gaussianFiltering == 0)
		return;
	Mat result;
	GaussianBlur(g_srcImage, result, Size(3, 3), 0, 0); // 3x3�����
	g_srcImage = result;
}

Mat image_make_border(Mat& src)
{
	int w = getOptimalDFTSize(src.cols); // ��ȡDFT�任����ѿ��
	int h = getOptimalDFTSize(src.rows); // ��ȡDFT�任����Ѹ߶�

	Mat padded;
	// ����������ͼ��߽磬���� = 0
	copyMakeBorder(src, padded, 0, h - src.rows, 0, w - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	padded.convertTo(padded, CV_32FC1);

	return padded;
}
Mat ideal_low_kernel(Mat& scr, float sigma)
{
	Mat ideal_low_pass(scr.size(), CV_32FC1); //��CV_32FC1
	float d0 = sigma;//�뾶D0ԽС��ģ��Խ�󣻰뾶D0Խ��ģ��ԽС
	for (int i = 0; i < scr.rows; i++) {
		for (int j = 0; j < scr.cols; j++) {
			float d = sqrt(pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2));//����,����pow����Ϊfloat��
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
	// ���µĲ������ƶ�ͼ��  (��Ƶ�Ƶ�����)
	int cx = plane0.cols / 2;
	int cy = plane0.rows / 2;
	Mat part1_r(plane0, Rect(0, 0, cx, cy));  // Ԫ�������ʾΪ(cx, cy)
	Mat part2_r(plane0, Rect(cx, 0, cx, cy));
	Mat part3_r(plane0, Rect(0, cy, cx, cy));
	Mat part4_r(plane0, Rect(cx, cy, cx, cy));

	Mat temp;
	part1_r.copyTo(temp);  //���������½���λ��(ʵ��)
	part4_r.copyTo(part1_r);
	temp.copyTo(part4_r);

	part2_r.copyTo(temp);  //���������½���λ��(ʵ��)
	part3_r.copyTo(part2_r);
	temp.copyTo(part3_r);

	Mat part1_i(plane1, Rect(0, 0, cx, cy));  //Ԫ������(cx,cy)
	Mat part2_i(plane1, Rect(cx, 0, cx, cy));
	Mat part3_i(plane1, Rect(0, cy, cx, cy));
	Mat part4_i(plane1, Rect(cx, cy, cx, cy));

	part1_i.copyTo(temp);  //���������½���λ��(�鲿)
	part4_i.copyTo(part1_i);
	temp.copyTo(part4_i);

	part2_i.copyTo(temp);  //���������½���λ��(�鲿)
	part3_i.copyTo(part2_i);
	temp.copyTo(part3_i);
}
Mat frequency_filter(Mat& scr, Mat& blur)
{
	Mat mask = scr == scr;
	scr.setTo(0.0f, ~mask);

	//����ͨ�����洢dft���ʵ�����鲿��CV_32F������Ϊ��ͨ������
	Mat plane[] = { scr.clone(), Mat::zeros(scr.size() , CV_32FC1) };

	Mat complexIm;
	merge(plane, 2, complexIm); // �ϲ�ͨ�� ������������ϲ�Ϊһ��2ͨ����Mat��������
	dft(complexIm, complexIm); // ���и���Ҷ�任���������������

	// ����ͨ����������룩
	split(complexIm, plane);

	// ���µĲ�����Ƶ��Ǩ��
	fftshift(plane[0], plane[1]);

	// *****************�˲���������DFT����ĳ˻�****************
	Mat blur_r, blur_i, BLUR;
	multiply(plane[0], blur, blur_r);  // �˲���ʵ�����˲���ģ���ӦԪ����ˣ�
	multiply(plane[1], blur, blur_i);  // �˲����鲿���˲���ģ���ӦԪ����ˣ�
	Mat plane1[] = { blur_r, blur_i };

	// �ٴΰ��ƻ���������任
	fftshift(plane1[0], plane1[1]);
	merge(plane1, 2, BLUR); // ʵ�����鲿�ϲ�

	idft(BLUR, BLUR);       // idft���ҲΪ����
	BLUR = BLUR / BLUR.rows / BLUR.cols;

	split(BLUR, plane);//����ͨ������Ҫ��ȡͨ��

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
	Mat ideal_high_pass(scr.size(), CV_32FC1); //��CV_32FC1
	float d0 = sigma;//�뾶D0ԽС��ģ��Խ�󣻰뾶D0Խ��ģ��ԽС
	for (int i = 0; i < scr.rows; i++) {
		for (int j = 0; j < scr.cols; j++) {
			float d = sqrt(pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2));//����,����pow����Ϊfloat��
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
	medianBlur(g_srcImage, result, 3); // 3x3�����
	g_srcImage = result;
}

int main()
{
	//��ȡ��Ƶ�ļ�
	cap.open("exp2.avi");
	if (!cap.isOpened())
	{
		cout << "can not open the video..." << endl;
		system("pause");
		return -1;
	}

	//��ȡ��Ƶ֡��
	int frame_count = cap.get(CAP_PROP_FRAME_COUNT);

	Mat frame;
	while (cap.read(frame))
	{
		int key = waitKeyEx(30);  //������Ӧ
		if (key == 27)
		{
			break; //����ESC���˳�ѭ��������Ƶ���Ž���
		}
		if (key == 32)
		{
			waitKey(0); //���¿ո����ͣ��Ƶ����
		}
		if (key == 2424832)
		{
			//���̡��������
			g_pos -= 30;
			cap.set(CAP_PROP_POS_FRAMES, g_pos);
		}
		if (key == 2555904)
		{
			//���̡�����ǰ���
			g_pos += 30;
			cap.set(CAP_PROP_POS_FRAMES, g_pos);
		}

		g_pos = cap.get(CAP_PROP_POS_FRAMES); //��ȡ��ǰ��Ƶ֡����λ��

		namedWindow("��Ƶ������", WINDOW_AUTOSIZE);
		resizeWindow("��Ƶ������", 800, 200);
		
		//������Ƶ���Ž���
		createTrackbar("frame", "��Ƶ������", &g_pos, frame_count, func); //ͨ���϶�������������Ƶ���Ż���
		
		//���ڵ�ǰ��Ƶ�ĶԱȶ�������
		g_srcImage = frame;
		
		//�趨�ԱȶȺ����ȳ�ֵ

		createTrackbar("�Աȶ�", "��Ƶ������", &g_nContrastValue, MAX_CONTRAST_VALUE, on_ContrastAndBright);
		createTrackbar("����", "��Ƶ������", &g_nBrightValue, MAX_BRIGHT_VALUE, on_ContrastAndBright);

		//���лص�������ʼ��
		on_ContrastAndBright(g_nContrastValue, 0);//�����ص������ԱȶȻ����ֵ��userdata = 0
		on_ContrastAndBright(g_nBrightValue, 0);//�����ص��������Ȼ����ֵ��userdata = 0
		
		//ֱ��ͼ���⻯
		Mat before, after;
		before = showHistogram(g_srcImage);
		/*imshow("ֱ��ͼ���⻯ǰ", before);*/
		createTrackbar("ֱ��ͼ���⻯", "��Ƶ������", &g_flag, 1, on_HistogramEqualization);
		
		on_HistogramEqualization(g_flag,0);
		
		after = showHistogram(g_srcImage);
		/*imshow("ֱ��ͼ���⻯��", after);*/
		
		//��Ƶ��ǿ
		//�����˲�
		createTrackbar("�����˲�", "��Ƶ������", &g_boxFiltering, 1, on_BoxFiltering);
		on_BoxFiltering(g_boxFiltering, 0);
		
		//��ֵ�˲�
		createTrackbar("��ֵ�˲�", "��Ƶ������", &g_averageFiltering, 1, on_averageFiltering);
		on_averageFiltering(g_averageFiltering, 0);

		//��˹�˲�
		createTrackbar("��˹�˲�", "��Ƶ������", &g_gaussianFiltering, 1, on_gaussianFiltering);
		on_gaussianFiltering(g_gaussianFiltering, 0);
		
		//��ֵ�˲�
		createTrackbar("��ֵ�˲�", "��Ƶ������", &g_medianFiltering, 1, on_medianFiltering);
		on_medianFiltering(g_medianFiltering, 0);
		
		//�����ͨ�˲�
		createTrackbar("�����ͨ�˲�", "��Ƶ������", &g_lowPassFiltering, 1, on_lowPassFiltering);
		on_lowPassFiltering(g_lowPassFiltering, 0);

		//�����ͨ�˲�
		createTrackbar("�����ͨ�˲�", "��Ƶ������", &g_highPassFiltering, 1, on_highPassFiltering);
		on_highPassFiltering(g_highPassFiltering, 0);
		
		frame = g_srcImage;
		imshow("��Ƶ������", frame);
	}

	cap.release();
	destroyAllWindows();
	system("pause");
	return 0;
}

