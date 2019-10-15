#include <opencv2/opencv.hpp>
#include <iostream>    
#include <opencv2\core\core.hpp>
#include <iostream>  
#include <stdio.h>
#include <string>
#include <sstream>

using namespace cv;
using namespace std;


// 用于矫正
Mat transformCorner(Mat src, RotatedRect rect);
Mat transformQRcode(Mat src, RotatedRect rect, double angle);
// 用于判断角点
bool IsQrPoint(vector<Point>& contour, Mat& img);
bool isCorner(Mat &image);
double Rate(Mat &count);
int leftTopPoint(vector<Point> centerPoint);
vector<int> otherTwoPoint(vector<Point> centerPoint, int leftTopPointIndex);
double rotateAngle(Point leftTopPoint, Point rightTopPoint, Point leftBottomPoint);


int main()
{
	//VideoCapture cap;
	//Mat src;
	//cap.open(0);                             //打开相机，电脑自带摄像头一般编号为0，外接摄像头编号为1，主要是在设备管理器中查看自己摄像头的编号。

	//cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);  //设置捕获视频的宽度
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 400);  //设置捕获视频的高度

	//if (!cap.isOpened())                         //判断是否成功打开相机
	//{
	//	cout << "摄像头打开失败!" << endl;
	//	return -1;
	//}
	while (1) {
		Mat src;
		src = imread("QRcode.jpg");
		//cap >> src;                                //从相机捕获一帧图像

		Mat srcCopy = src.clone();

		//canvas为画布 将找到的定位特征画出来
		Mat canvas;
		canvas = Mat::zeros(src.size(), CV_8UC3);

		Mat srcGray;

		//center_all获取特性中心
		vector<Point> center_all;

		// 转化为灰度图
		cvtColor(src, srcGray, COLOR_BGR2GRAY);

		// 3X3模糊
		//blur(srcGray, srcGray, Size(3, 3));

		// 计算直方图
		//equalizeHist(srcGray, srcGray);
		//int s = srcGray.at<Vec3b>(0, 0)[0];
		// 设置阈值根据实际情况 如视图中已找不到特征 可适量调整
		//threshold(srcGray, srcGray, 0, 255, THRESH_BINARY | THRESH_OTSU);
		
		Canny(srcGray, srcGray, 50, 150, 3);
		Mat erodeStruct = getStructuringElement(MORPH_RECT, Size(3, 3));
		morphologyEx(srcGray, srcGray, MORPH_CLOSE, erodeStruct);
		imshow("threshold", srcGray);
		/*contours是第一次寻找轮廓*/
		/*contours2是筛选出的轮廓*/
		vector<vector<Point>> contours;

		//	用于轮廓检测
		vector<Vec4i> hierarchy;
		findContours(srcGray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

		
		// 小方块的数量
		int numOfRec = 0;

		// 检测方块
		int ic = 0;
		int parentIdx = -1;
		for (int i = 0; i < contours.size(); i++)
		{
			if (hierarchy[i][2] != -1 && ic == 0)
			{
				parentIdx = i;
				ic++;
			}
			else if (hierarchy[i][2] != -1)
			{
				ic++;
			}
			else if (hierarchy[i][2] == -1)
			{
				parentIdx = -1;
				ic = 0;
			}
			if (ic == 5)
			{
				if (IsQrPoint(contours[parentIdx], src)) {
					RotatedRect rect = minAreaRect(Mat(contours[parentIdx]));

					// 画图部分
					Point2f points[4];
					rect.points(points);
					for (int j = 0; j < 4; j++) {
						line(src, points[j], points[(j + 1) % 4], Scalar(0, 255, 0), 2);
					}
					drawContours(canvas, contours, parentIdx, Scalar(0, 0, 255), -1);

					// 如果满足条件则存入
					center_all.push_back(rect.center);
					numOfRec++;
				}
				ic = 0;
				parentIdx = -1;
			}
		}
		
		// 连接三个正方形的部分
		for (int i = 0; i < center_all.size(); i++)
		{
			line(canvas, center_all[i], center_all[(i + 1) % center_all.size()], Scalar(255, 0, 0), 3);
		}

		vector<vector<Point>> contours3;
		Mat canvasGray;
		cvtColor(canvas, canvasGray, COLOR_BGR2GRAY);
		findContours(canvasGray, contours3, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		vector<Point> maxContours;
		double maxArea = 0;
		// 在原图中画出二维码的区域

		for (int i = 0; i < contours3.size(); i++)
		{
			RotatedRect rect = minAreaRect(contours3[i]);
			Point2f boxpoint[4];
			rect.points(boxpoint);
			for (int i = 0; i < 4; i++)
				line(src, boxpoint[i], boxpoint[(i + 1) % 4], Scalar(0, 0, 255), 3);

			double area = contourArea(contours3[i]);
			if (area > maxArea) {
				maxContours = contours3[i];
				maxArea = area;
			}
		}
		imshow("src", src);
		if (numOfRec < 3) {
			waitKey(10);
			continue;
		}
		// 计算“回”的次序关系
		int leftTopPointIndex = leftTopPoint(center_all);
		vector<int> otherTwoPointIndex = otherTwoPoint(center_all, leftTopPointIndex);
		// canvas上标注三个“回”的次序关系
		circle(canvas, center_all[leftTopPointIndex],10,Scalar(255,0,255),-1);
		circle(canvas, center_all[otherTwoPointIndex[0]], 10, Scalar(0, 255, 0),-1);
		circle(canvas, center_all[otherTwoPointIndex[1]], 10, Scalar(0, 255, 255),-1);

		// 计算旋转角
		double angle = rotateAngle(center_all[leftTopPointIndex], center_all[otherTwoPointIndex[0]], center_all[otherTwoPointIndex[1]]);

		// 拿出之前得到的最大的轮廓
		RotatedRect rect = minAreaRect(Mat(maxContours));
		Mat image = transformQRcode(srcCopy, rect, angle);

		// 展示图像
		imshow("QRcode", image);
		imshow("canvas", canvas);
		waitKey(10);
	}
	return 0;
}

Mat transformCorner(Mat src, RotatedRect rect)
{
	// 获得旋转中心
	Point center = rect.center;
	// 获得左上角和右下角的角点，而且要保证不超出图片范围，用于抠图
	Point TopLeft = Point(cvRound(center.x), cvRound(center.y)) - Point(rect.size.height / 2, rect.size.width / 2);  //旋转后的目标位置
	TopLeft.x = TopLeft.x > src.cols ? src.cols : TopLeft.x;
	TopLeft.x = TopLeft.x < 0 ? 0 : TopLeft.x;
	TopLeft.y = TopLeft.y > src.rows ? src.rows : TopLeft.y;
	TopLeft.y = TopLeft.y < 0 ? 0 : TopLeft.y;

	int after_width, after_height;
	if (TopLeft.x + rect.size.width > src.cols) {
		after_width = src.cols - TopLeft.x - 1;
	}
	else {
		after_width = rect.size.width - 1;
	}
	if (TopLeft.y + rect.size.height > src.rows) {
		after_height = src.rows - TopLeft.y - 1;
	}
	else {
		after_height = rect.size.height - 1;
	}
	// 获得二维码的位置
	Rect RoiRect = Rect(TopLeft.x, TopLeft.y, after_width, after_height);

	//	dst是被旋转的图片 roi为输出图片 mask为掩模
	double angle = rect.angle;
	Mat mask, roi, dst;
	Mat image;
	// 建立中介图像辅助处理图像

	vector<Point> contour;
	// 获得矩形的四个点
	Point2f points[4];
	rect.points(points);
	for (int i = 0; i < 4; i++)
		contour.push_back(points[i]);

	vector<vector<Point>> contours;
	contours.push_back(contour);
	// 再中介图像中画出轮廓
	drawContours(mask, contours, 0, Scalar(255, 255, 255), -1);
	// 通过mask掩膜将src中特定位置的像素拷贝到dst中。
	src.copyTo(dst, mask);
	// 旋转
	Mat M = getRotationMatrix2D(center, angle, 1);
	warpAffine(dst, image, M, src.size());
	// 截图
	roi = image(RoiRect);

	return roi;
}

// 该部分用于检测是否是角点，与下面两个函数配合
bool IsQrPoint(vector<Point>& contour, Mat& img) {
	double area = contourArea(contour);
	// 角点不可以太小
	if (area < 30)
		return 0;
	RotatedRect rect = minAreaRect(Mat(contour));
	double w = rect.size.width;
	double h = rect.size.height;
	double rate = min(w, h) / max(w, h);
	if (rate > 0.7)
	{
		// 返回旋转后的图片，用于把“回”摆正，便于处理
		Mat image = transformCorner(img, rect); 
		if (isCorner(image))
		{
			return 1;
		}
	}
	return 0;
}

// 计算内部所有白色部分占全部的比率
double Rate(Mat &count)
{
	int number = 0;
	int allpixel = 0;
	for (int row = 0; row < count.rows; row++)
	{
		for (int col = 0; col < count.cols; col++)
		{
			if (count.at<uchar>(row, col) == 255)
			{
				number++;
			}
			allpixel++;
		}
	}
	//cout << (double)number / allpixel << endl;
	return (double)number / allpixel;
}

// 用于判断是否属于角上的正方形
bool isCorner(Mat &image)
{
	// 定义mask
	Mat imgCopy, dstCopy;
	Mat dstGray;
	imgCopy = image.clone();
	
	// 转化为灰度图像
	cvtColor(image, dstGray, COLOR_BGR2GRAY);
	// 进行二值化

	threshold(dstGray, dstGray, 0, 255, THRESH_BINARY | THRESH_OTSU);
	dstCopy = dstGray.clone();  //备份

	Mat erodeStruct = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(dstCopy, dstCopy, MORPH_OPEN, erodeStruct);
	// 找到轮廓与传递关系
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(dstCopy, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	
	for (int i = 0; i < contours.size(); i++)
	{
		//cout << i << endl;
		if (hierarchy[i][2] == -1 && hierarchy[i][3] != -1)
		{
			
			Rect rect = boundingRect(Mat(contours[i]));
			rectangle(image, rect, Scalar(0, 0, 255), 2);
			
			// 最里面的矩形与最外面的矩形的对比
			if (rect.width < imgCopy.cols * 2 / 7)      //2/7是为了防止一些微小的仿射
				continue;
			if (rect.height < imgCopy.rows * 2 / 7)      //2/7是为了防止一些微小的仿射
				continue;
			
			// 判断其中黑色与白色的部分的比例
			if (Rate(dstGray) > 0.20)
			{
				return true;
			}
		}
	}
	return  false;
}

int leftTopPoint(vector<Point> centerPoint) {
	int minIndex = 0;
	int multiple = 0;
	int minMultiple = 10000;
	multiple = (centerPoint[1].x - centerPoint[0].x)*(centerPoint[2].x - centerPoint[0].x) + (centerPoint[1].y - centerPoint[0].y)*(centerPoint[2].y - centerPoint[0].y);
	if (minMultiple > multiple){
		minIndex = 0;
		minMultiple = multiple;
	}
	multiple = (centerPoint[0].x - centerPoint[1].x)*(centerPoint[2].x - centerPoint[1].x) + (centerPoint[0].y - centerPoint[1].y)*(centerPoint[2].y - centerPoint[1].y);
	if (minMultiple > multiple) {
		minIndex = 1;
		minMultiple = multiple;
	}
	multiple = (centerPoint[0].x - centerPoint[2].x)*(centerPoint[1].x - centerPoint[2].x) + (centerPoint[0].y - centerPoint[2].y)*(centerPoint[1].y - centerPoint[2].y);
	if (minMultiple > multiple) {
		minIndex = 2;
		minMultiple = multiple;
	}
	return minIndex;
}

vector<int> otherTwoPoint(vector<Point> centerPoint,int leftTopPointIndex) {
	vector<int> otherIndex;
	double waiji = (centerPoint[(leftTopPointIndex + 1) % 3].x- centerPoint[(leftTopPointIndex) % 3].x)*
		(centerPoint[(leftTopPointIndex + 2) % 3].y - centerPoint[(leftTopPointIndex) % 3].y) -
		(centerPoint[(leftTopPointIndex + 2) % 3].x - centerPoint[(leftTopPointIndex) % 3].x)*
		(centerPoint[(leftTopPointIndex + 1) % 3].y - centerPoint[(leftTopPointIndex) % 3].y);
	if (waiji > 0) {
		otherIndex.push_back((leftTopPointIndex + 1) % 3);
		otherIndex.push_back((leftTopPointIndex + 2) % 3);
	}
	else {
		otherIndex.push_back((leftTopPointIndex + 2) % 3);
		otherIndex.push_back((leftTopPointIndex + 1) % 3);
	}
	return otherIndex;
}

double rotateAngle(Point leftTopPoint, Point rightTopPoint, Point leftBottomPoint) {
	double dy = rightTopPoint.y - leftTopPoint.y;
	double dx = rightTopPoint.x - leftTopPoint.x;
	double k = dy / dx;
	double angle = atan(k) * 180 / CV_PI;//转化角度
	if (leftBottomPoint.y < leftTopPoint.y)
		angle -= 180;
	return angle;
}

Mat transformQRcode(Mat src, RotatedRect rect,double angle)
{
	// 获得旋转中心
	Point center = rect.center;
	// 获得左上角和右下角的角点，而且要保证不超出图片范围，用于抠图
	Point TopLeft = Point(cvRound(center.x), cvRound(center.y)) - Point(rect.size.height / 2, rect.size.width / 2);  //旋转后的目标位置
	TopLeft.x = TopLeft.x > src.cols ? src.cols : TopLeft.x;
	TopLeft.x = TopLeft.x < 0 ? 0 : TopLeft.x;
	TopLeft.y = TopLeft.y > src.rows ? src.rows : TopLeft.y;
	TopLeft.y = TopLeft.y < 0 ? 0 : TopLeft.y;

	int after_width, after_height;
	if (TopLeft.x + rect.size.width > src.cols) {
		after_width = src.cols - TopLeft.x - 1;
	}
	else {
		after_width = rect.size.width - 1;
	}
	if (TopLeft.y + rect.size.height > src.rows) {
		after_height = src.rows - TopLeft.y - 1;
	}
	else {
		after_height = rect.size.height - 1;
	}
	// 获得二维码的位置
	Rect RoiRect = Rect(TopLeft.x, TopLeft.y, after_width, after_height);   

	// dst是被旋转的图片，roi为输出图片，mask为掩模
	Mat mask, roi, dst;                
	Mat image;
	// 建立中介图像辅助处理图像

	vector<Point> contour;
	// 获得矩形的四个点
	Point2f points[4];
	rect.points(points);
	for (int i = 0; i < 4; i++)
		contour.push_back(points[i]);

	vector<vector<Point>> contours;
	contours.push_back(contour);
	// 再中介图像中画出轮廓
	drawContours(mask, contours, 0, Scalar(255,255,255), -1);
	// 通过mask掩膜将src中特定位置的像素拷贝到dst中。
	src.copyTo(dst, mask);
	// 旋转
	Mat M = getRotationMatrix2D(center, angle, 1);
	warpAffine(dst, image, M, src.size());
	// 截图
	roi = image(RoiRect);

	return roi;
}