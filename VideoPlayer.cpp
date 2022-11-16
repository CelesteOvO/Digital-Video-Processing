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
		createTrackbar("frame", "��Ƶ������", &g_pos, frame_count, func); //ͨ���϶�������������Ƶ���Ż���
		
		createTrackbar("contrast", "��Ƶ������", &g_nContrastValue, MAX_CONTRAST_VALUE, on_ContrastAndBright);
		
		g_srcImage = frame;
		
		//�趨�ԱȶȺ����ȳ�ֵ

		createTrackbar("contrast", "��Ƶ������", &g_nContrastValue, MAX_CONTRAST_VALUE, on_ContrastAndBright);
		createTrackbar("bright", "��Ƶ������", &g_nBrightValue, MAX_BRIGHT_VALUE, on_ContrastAndBright);

		//���лص�������ʼ��
		on_ContrastAndBright(g_nContrastValue, 0);//�����ص������ԱȶȻ����ֵ��userdata = 0
		on_ContrastAndBright(g_nBrightValue, 0);//�����ص��������Ȼ����ֵ��userdata = 0
		
		frame = g_srcImage;
		
		imshow("��Ƶ������", g_srcImage);
	}

	cap.release();
	destroyAllWindows();
	system("pause");
	return 0;
}

