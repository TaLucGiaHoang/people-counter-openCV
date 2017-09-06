#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>

using namespace std;
using namespace cv;



	int keyboard = 0;
	char* savedFileName = "D:\\Program Files\\visual studio\\sample\\image.jpg";
	char* backGroundName = "D:\\Program Files\\visual studio\\sample\\background.jpg";
	char* windowName = "Camera Photograph";
	char* grayWindowName = "Gray Camera Photograph";
	char* binWindowName = "Binary Camera Photograph";


void backgroundSubtract(IplImage* backgroundImage, IplImage* currentImage);
int main(int argc, char** argv) {
	//IplImage* grayBackground = cvLoadImage("D:\\Program Files\\visual studio\\sample\\background.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//IplImage* frame = cvLoadImage("D:\\Program Files\\visual studio\\sample\\image.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	CvCapture *capture = cvCreateCameraCapture(0);
	//CvCapture *capture = cvCreateFileCapture("D:\\Program Files\\visual studio\\sample\\People_counter_OpenCV.avi");
	IplImage* frame;
	
	//take the background
	cout << "Initializing..." << endl;
	cvNamedWindow(windowName);
	cout << "Press 'c' to get the background" << endl;
	while (1) {
		frame = cvQueryFrame(capture);
		//cvFlip(frame, frame, 1);			//////////////////delete when using api file
		cvShowImage(windowName, frame);
		if (cvWaitKey(10) == 'c')
		{
			cvSaveImage(backGroundName, frame);
			cout << "Saved background" << endl;
			break;
		}
	}
	IplImage* backgroundImage = cvLoadImage(backGroundName, CV_LOAD_IMAGE_COLOR);
	CvSize size = cvGetSize(backgroundImage);

	IplImage* grayBackground = cvCreateImage(size, 8, 1);
	IplImage* grayFrame		= cvCreateImage(size, 8, 1);
	IplImage* absDiffDst	= cvCreateImage(size, 8, 1);
	IplImage* temp			= cvCreateImage(size, 8, 1);
	//IplImage* MorPho	    = cvCreateImage(size, 8, 1);
	IplImage* bin	        = cvCreateImage(size, 8, 1);


	cvCvtColor(backgroundImage, grayBackground, CV_BGR2GRAY);		//ham convert phai dung 2 ma tran src, dst khac nhau



	cout << "running" << endl;

	while (1) {
		if (!cvGrabFrame(capture)) break;
		frame = cvRetrieveFrame(capture);
		//frame = cvQueryFrame(capture);
		cvShowImage("video", frame);

		//cvFlip(frame, frame, 1);	//////////////////delete

		cvCvtColor(frame, grayFrame, CV_BGR2GRAY);
		
		cvAbsDiff(grayBackground, grayFrame, absDiffDst);
		//cvShowImage("foreground", absDiffDst);


		cvSmooth(absDiffDst, absDiffDst, CV_BLUR);


		//THRESHOLD
		cvThreshold(absDiffDst, bin, 15, 255, CV_THRESH_BINARY);	//value of noise usully less than 15
		
		cvMorphologyEx(bin, bin, temp, NULL, CV_MOP_OPEN, 4);
		cvMorphologyEx(bin, bin, temp, NULL, CV_MOP_CLOSE, 4);
/*
		cvDilate(bin, bin, NULL, 2);
		cvErode(bin, bin, NULL, 2);
		cvMorphologyEx(bin, bin, temp, NULL, CV_MOP_OPEN, 2);
		cvMorphologyEx(bin, bin, temp, NULL, CV_MOP_CLOSE, 2);
		
*/
		cvShowImage("binary", bin);

		int key = cvWaitKey(10);
		if (key == 'c')
		{
			cvSaveImage(savedFileName, frame);
			cout << "Saved image" << endl;
		}
		if (key == 27) break;
	}
	return 0;
}
/*
void drawingContours() {
	CvSeq* contours = NULL;
	CvMemStorage* storage = cvCreateMemStorage(0);
	cvFindContours(bin, storage, &contours);
	cout << "number of contours: " << contours->total << endl;
	IplImage* drawContours = cvCreateImage(size, 8, 3);
	cvZero(drawContours);
	cvDrawContours(drawContours, contours, CV_RGB(0, 0, 255), CV_RGB(255, 0, 0), 100, 1);
	for (int i = 0;contours != 0;contours = contours->h_next) {
		CvRect rect = cvBoundingRect(contours, 0);
		cout << i++ << endl;
		cvRectangleR(drawContours, rect, CV_RGB(100, 100, 100));
	}
	cvShowImage("draw contours", drawContours);
}*/
