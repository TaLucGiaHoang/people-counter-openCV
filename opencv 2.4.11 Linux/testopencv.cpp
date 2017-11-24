#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

void silhoutte_reduction(IplImage* src, IplImage* dst, int codeCvt )
{
	if (codeCvt != CV_BGR2HSV)
		cvCvtColor(src, src, codeCvt);
	IplImage* h = cvCreateImage(cvGetSize(src), 8, 1);
	IplImage* s = cvCreateImage(cvGetSize(src), 8, 1);
	IplImage* v = cvCreateImage(cvGetSize(src), 8, 1);
	cvSplit(src, h, s, v, 0);
	
	//cvThreshold(h, h, h_min, h_max, CV_THRESH_TOZERO);	//CV_THRESH_TOZERO = 3,  /* value = value > threshold ? value : 0           */
	//cvThreshold(h, h, h_max, h_max, CV_THRESH_TOZERO_INV);	//CV_THRESH_TRUNC = 2,  /* value = value > threshold ? threshold : value   */
	//cvThreshold(s, s, s_min, s_max, CV_THRESH_TOZERO);	//CV_THRESH_TOZERO_INV  =4,  /* value = value > threshold ? 0 : value           */
	//cvThreshold(s, s, s_max, s_max, CV_THRESH_TOZERO_INV);
	//cvThreshold(v, v, v_min, v_max, CV_THRESH_TOZERO);
	//cvThreshold(v, v, v_max, v_max, CV_THRESH_TOZERO_INV);

	cvReleaseImage(&h);
	cvReleaseImage(&s);
	cvReleaseImage(&v);
	cvMerge(0, 0, 0, 0, dst);
}

void show (const char* name, IplImage* img)
{
	cvNamedWindow(name,CV_WINDOW_AUTOSIZE);
    cvShowImage(name,img);
}

int main()
{
    IplImage* src = cvLoadImage("image1.png",CV_LOAD_IMAGE_COLOR);
    IplImage* dst = cvCreateImage(cvGetSize(src), 8, 3);
    IplImage* h = cvCreateImage(cvGetSize(src), 8, 1);
	IplImage* s = cvCreateImage(cvGetSize(src), 8, 1);
	IplImage* v = cvCreateImage(cvGetSize(src), 8, 1);
	cvSplit(src, h, s, v, 0);
	cvMerge(0, s, 0, 0, dst);
	
	show("h", h);
    show("s", s);
    show("v", v);
    show("opencvtest", src);
    show("Merge", dst);
    //cvNamedWindow("opencvtest",CV_WINDOW_AUTOSIZE);
    //cvShowImage("opencvtest",img);
    cvWaitKey(0);
    cvReleaseImage(&src);
    return 0;
}


//int main()
//{
    //IplImage* img = cvLoadImage("image1.png",CV_LOAD_IMAGE_COLOR);
    //silhoutte_reduction(img, img, CV_BGR2HSV);
    //cvNamedWindow("opencvtest",CV_WINDOW_AUTOSIZE);
    //cvShowImage("opencvtest",img);
    //cvWaitKey(0);
    //cvReleaseImage(&img);
    //return 0;
//}
