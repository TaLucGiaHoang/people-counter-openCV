#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

int main()
{
    IplImage* img = cvLoadImage("/home/hoang/NRF24L01-1.jpg",CV_LOAD_IMAGE_COLOR);
    cvNamedWindow("opencvtest",CV_WINDOW_AUTOSIZE);
    cvShowImage("opencvtest",img);
    cvWaitKey(0);
    cvReleaseImage(&img);
    return 0;
}