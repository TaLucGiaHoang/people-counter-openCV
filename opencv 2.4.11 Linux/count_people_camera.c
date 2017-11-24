#include <opencv/cv.h>
#include <opencv/highgui.h>
//#include <opencv/cvaux.h>// nho uncomment
#include <string.h>
#include <time.h>

/* Linux */
/* opnecv 2.4.13 */
#include <opencv2/core/types_c.h>
#include <opencv2/legacy/legacy.hpp>   /* Background codebook */
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <stdbool.h>  // true false
//#include <opencv2/video/background_segm.hpp>


//#define CAM_WIDTH 320//640
//#define CAM_HEIGHT 240//480
#define CHANNELS 3
#define UPDATE_PERIOD 2000
#define CLEAR_PERIOD  4000
#define LEARNING_TIME 100
#define NUM_OF_PIXEL_ON_LINE 500
#define WIDTH_CENTER_LINE	10

#define RED    cvScalar(0, 0, 255, 0)
#define GREEN  cvScalar(0, 255, 0, 0)
#define BLUE   cvScalar(255, 0, 0, 0)
#define WHITE  cvScalar(255, 255, 255, 0)
#define BLACK  cvScalarAll(0)

int CamWidth = 0;// CAM_WIDTH;
int CamHeight = 0;// CAM_HEIGHT;
int center_x = 0;// CamWidth / 2;
int center_y = 0;// CamHeight / 2;
int nframes = 0, nframesToLearnBG = 0;
bool useAVIfile = false;
bool firstInMode = true;
int pos = 0;
int fps = 0;
CvCapture *capture;
int maxFrame = 0;


//typedef struct _Line {
//	CvPoint pt1;
//	CvPoint pt2;
//	CvPoint center = cvPoint((int)((pt1.x + pt2.x) / 2), (int)((pt1.y + pt2.y) / 2));
//	int width;
//} Line;


int count_in = 0, count_out = 0;
int mode = 2;	//HORIZONTAL MODE (DEFAULT)

//VARIABLES for CODEBOOK METHOD:
IplImage* tempImage = 0;
IplImage* rawImage = 0, *hsvImage = 0; //yuvImage is for codebook method
IplImage *ImaskCodeBook = 0, *ImaskCodeBookCC = 0;
CvBGCodeBookModel *model = 0;
const int NCHANNELS = 3;

IplImage* circleImage = 0;

// various tracking parameters (in seconds)
const double MHI_DURATION = 0.5;//1;
const double MAX_TIME_DELTA = 0.5;
const double MIN_TIME_DELTA = 0.05;
int duration = 1000;
int seg_thresh = 1500;
// number of cyclic frame buffer used for motion detection
// (should, probably, depend on FPS)
const int N = 4;

// ring image buffer
IplImage **buf = 0;
int last = 0;


// temporary images
IplImage* motion = 0;

IplImage* silh = 0;// 8U, 1-channel
IplImage *mhi = 0; // MHI 32F, 1-channel
IplImage *orient = 0; // orientation 32F, 1-channel
IplImage *mask = 0; // valid orientation mask 8U, 1-channel
IplImage *segmask = 0; // motion segmentation map 32F, 1-channel

CvMemStorage* storage = 0; // temporary storage
void  update_mhi(IplImage* img, IplImage* dst, int diff_threshold);


int idx = 0;
int saiso = 40;

typedef struct _Tracking {
	bool value;
	double exist_time;	// life time
	double start_time;
	double time_count;
}Tracking;
Tracking *countOnLine;// [NUM_OF_PIXEL_ON_LINE];//[CAM_HEIGHT];//[CamHeight];

static struct _Tracking createTracking(void)
{
	struct _Tracking trk;
	trk.value =  false;
	trk.exist_time = 10;	//	exist_time < line_w/2 ,  exist_time ~= update
	trk.start_time = (double)clock() / CLOCKS_PER_SEC;	//get current time (second);
	trk.time_count = 0;
	return trk;
}

struct _Tracking* initCountLine(unsigned int count)
{
	Tracking* arr;
	arr = (Tracking*)malloc(count);
	int i;
	for(i = 0; i<count; i++)
	{
		arr[i] = createTracking();
	}
	return arr;
}

void releaseTracking (Tracking* track)
{
	if(track)
		free(track);
}

/////////////////////////////////////////////
//void max_mod_Trackbar(int pos);
//void min_mod_Trackbar(int pos);

int select_object = 0;
int track_object = 0;

CvPoint origin;
CvRect selection;
CvRect track_window;
CvBox2D track_box;
void on_mouse(int event, int x, int y, int flags, void* param);

void InitImages(CvSize sz);
void ReleaseImages();

int modMin = 50, modMax = 10;	//codebook configuration
int h_min  = 10, h_max  = 155;	//hue mau sac
int s_min  = 0,  s_max  = 255;	//saturation do tuong phan
int v_min  = 0,  v_max  = 255;		//value do sang

void threshold_trackbar(const char* name);
void silhoutte_reduction(IplImage* src, IplImage* dst, int codeCvt);
void control_trackbar(const char* name);
void onTrackbarSlide(int n) {
	cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, n);
}

CvFont g_font;

//int update_period = 200, clear_peridod = 300;
int update_duration = 6, clear_duration = 60;// seconds
time_t start_update_time, start_clear_time, current_time;


int poly1Hull0 = 0;
int line_w = 20;

int min_len  = 650;
int max_len  = 5000;			//< 2000 - 5000
int min_area = 6000;
int max_area = 70000;//640 * 480;	// < 30 000 - 50 000
CvPoint center [50];

int reject_small_and_large_object(IplImage* src, IplImage* dst, IplImage* circleImage, CvPoint* center, int /*&numOfCenters*/ *numOfCenters, CvMemStorage* storage);

void update_mhi(IplImage* img, IplImage* dst, int diff_threshold) {

	double timestamp = (double)clock() / CLOCKS_PER_SEC; // get current time in seconds
	int i;
	//bo sung lai
	double seg_thresh, duration;
	if (useAVIfile) {
		seg_thresh = 1.5;
		duration = 1.0;
	}else{
		seg_thresh = 1.75;		//use camera
		duration = 1.5;

	}
	
	if (img->nChannels != 1)
		cvCvtColor(img, buf[last], CV_BGR2GRAY); // convert frame to grayscale
	else
		cvCopyImage(img, buf[last]);
	int idx1 = last;
	int idx2 = (last + 1) % N; // index of (last - (N-1))th frame
	last = idx2;
	silh = buf[idx2];
	cvAbsDiff(buf[idx1], buf[idx2], silh); // get difference between frames
	cvThreshold(silh, silh, 50, 255, CV_THRESH_BINARY); // and threshold it
	// ket thuc bo sung

	//cvThreshold(ImaskCodeBookCC, silh, 50, 255, CV_THRESH_BINARY); // and threshold it
	cvUpdateMotionHistory(silh, mhi, timestamp, 1.0); //MHI_DURATION);// update MHI	= 1ms	//thoi gian luu cac bong
	cvCvtScale(mhi, mask, 255, 0);//((double)(duration / 1000) - timestamp) * 255);// convert MHI to blue 8u image
	cvZero(dst);
	cvMerge(mask, 0, 0, 0, dst);
	// calculate motion gradient orientation and valid orientation mask
	cvCalcMotionGradient(mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3);
	if (!storage)
		storage = cvCreateMemStorage(0);
	else
		cvClearMemStorage(storage);

	// segment motion: get sequence of motion components
	// segmask is marked motion components map. It is not used further
	CvSeq* seq;

	
	seq = cvSegmentMotion(mhi, segmask, storage, timestamp, seg_thresh);//(double)(seg_thresh / 1000));//seg_thresh = MAX_TIME_DELTA = 0.5;//	seg_thresh = 1.5 times the average difference in sillouete time stamps
	

	
	int detect_object(
		///	return 1: in	2: out		0: null  
		int detectable_direction,
		int object_center,
		int center_of_detect_line,
		int num_of_pixel_on_line,
		int direction_point,
		Tracking *track_line,
		int index_of_detected_object,
		double exist_time	//	exist_time < line_w/2
	);

	// iterate through the motion components,
	// One more iteration (i == -1) corresponds to the whole image (global motion)
	CvRect comp_rect;
	double count;
	double angle;
	//CvPoint momentcenter;
	double magnitude = 40;;
	for (i = -1; i < seq->total; i++) {

		if (i < 0) { // case of the whole image
			continue;
		}
		else { // i-th motion component
			comp_rect = ((CvConnectedComp*)cvGetSeqElem(seq, i))->rect;	//take the bounding rectangle for each motion
		}
		// select component ROI
		cvSetImageROI(silh, comp_rect);
		cvSetImageROI(mhi, comp_rect);
		cvSetImageROI(orient, comp_rect);
		cvSetImageROI(mask, comp_rect);
		//calculate moment
		//cvMoments(silh, &moments, 1);
		//double M00 = cvGetSpatialMoment(&moments, 0, 0);
		//double M01 = cvGetSpatialMoment(&moments, 0, 1);
		//double M10 = cvGetSpatialMoment(&moments, 1, 0);
		//momentcenter.x = (int)(M10 / M00);
		//momentcenter.y = (int)(M01 / M00);
		// calculate orientation
		
		angle = cvCalcGlobalOrientation(orient, mask, mhi, timestamp, duration);	//calculate orientation for each object
		angle = 360.0 - angle;  // adjust for images with top-left origin
		count = cvNorm(silh, 0, CV_L1, 0); // calculate number of points within silhouette ROI
		cvResetImageROI(mhi);
		cvResetImageROI(orient);
		cvResetImageROI(mask);
		cvResetImageROI(silh);
		// check for the case of little motion
		if (count < comp_rect.width*comp_rect.height * 0.05)
			continue;
		CvPoint directionPoint;
		//momentcenter = cvPoint((comp_rect.x + momentcenter.x), (comp_rect.y + momentcenter.y));
		/*directionPoint = cvPoint(cvRound(momentcenter.x + magnitude*cos(angle*CV_PI / 180)),
			cvRound(momentcenter.y - magnitude*sin(angle*CV_PI / 180)));*/
		directionPoint = cvPoint(cvRound(center[i].x + magnitude*cos(angle*CV_PI / 180)),
			cvRound(center[i].y - magnitude*sin(angle*CV_PI / 180)));

		//cvCircle(dst, momentcenter, 2, CV_RGB(255, 0, 0), -1, 8, 0);
		cvRectangleR(dst, comp_rect, CV_RGB(255, 0, 0), 1, 8, 0);
		cvLine(rawImage, center[i], directionPoint, CV_RGB(100, 255, 0), 2, 8 ,0);

		int isDetected = 0;
		if (mode == 1) {
			//vertical detective line
			if (firstInMode) {
				countOnLine = initCountLine(CamHeight);
				firstInMode = false;
			}			
			isDetected = detect_object(
				//momentcenter.x, momentcenter.y,
				center[i].x, center[i].y,
				CamWidth / 2,
				CamHeight,
				directionPoint.x,
				countOnLine,
				idx,
				line_w / 2	);
		}
		
		if (mode == 2) {			
			//horizontal detective line
			if (firstInMode) {
				countOnLine = initCountLine(CamWidth);
				firstInMode = false;
			}
			isDetected = detect_object(
				//momentcenter.y, momentcenter.x,
				center[i].y, center[i].x,
				CamHeight / 2,
				CamWidth,
				directionPoint.y,
				countOnLine,
				idx,
				line_w / 2	);
		}

		if (isDetected == 1) {
			//input 	
			count_in++;		printf("IN: %d\n", count_in);
		}
		if (isDetected == 2) {
			//output	
			count_out++;	printf("OUT: %d\n", count_out);
		}
	}

}

void on_mouse(int event, int x, int y, int flags, void* param)
{
	if (select_object)
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = selection.x + CV_IABS(x - origin.x);
		selection.height = selection.y + CV_IABS(y - origin.y);

		selection.x = MAX(selection.x, 0);	// = |x| or 0
		selection.y = MAX(selection.y, 0);
		selection.width = MIN(selection.width, rawImage->width);
		selection.height = MIN(selection.height, rawImage->height);
		selection.width -= selection.x;
		selection.height -= selection.y;
		printf("%d,%d w: %d h: %d \n", selection.x, selection.y, selection.width, selection.height);
	}

	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		origin = cvPoint(x, y);
		selection = cvRect(x, y, 0, 0);
		select_object = 1;
		break;
	case CV_EVENT_LBUTTONUP:
		select_object = 0;
		if (selection.width > 0 && selection.height > 0)
			track_object = -1;
		break;
	}
}

/*
 * rawImage        : IPL_DEPTH_8U, RGB image
 * hsvImage        : IPL_DEPTH_8U, HSV image
 * ImaskCodeBook   : IPL_DEPTH_8U, gray-scale image
 * ImaskCodeBookCC : IPL_DEPTH_8U, gray-scale image
 * circleImage     : IPL_DEPTH_8U, gray-scale image
 */
void InitImages(CvSize sz)
{
	int i;
	//	SILHOUTTE REDUCTION	
	tempImage = cvCloneImage(rawImage);

	// CODEBOOK METHOD ALLOCATION
	hsvImage = cvCloneImage(rawImage);
	cvZero(hsvImage);
	ImaskCodeBook = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 1);
	ImaskCodeBookCC = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 1);
	cvSet(ImaskCodeBook, WHITE, NULL );
	
	circleImage = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 1);

	silh = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 1);	// 8U, 1-channel
	mhi = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_32F, 1);
	cvZero(mhi); // clear MHI at the beginning
	orient = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_32F, 1);
	segmask = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_32F, 1);
	mask = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 1);

	if (buf == 0) {
		buf = (IplImage**)malloc(N * sizeof(buf[0]));
		memset(buf, 0, N * sizeof(buf[0]));

	}
	for (i = 0; i < 4; i++) {	//4 images
		cvReleaseImage(&buf[i]);
		buf[i] = cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 1); //cvCreateImage(size, IPL_DEPTH_8U, 1);
		cvZero(buf[i]);
	}

	motion = cvCloneImage(rawImage);
	cvZero(motion);
}

void ReleaseImages()
{
	cvReleaseImage(&tempImage);
	
	cvReleaseImage(&silh);
	cvReleaseImage(&mhi);
	cvReleaseImage(&orient);
	cvReleaseImage(&segmask);
	/*for (int i = 0; i < 4; i++) {	//4 images
		cvReleaseImage(&buf[i]);
	}*/
	if(motion)	cvReleaseImage(&motion);
	cvReleaseImage(&hsvImage);
	cvReleaseImage(&ImaskCodeBook);
	cvReleaseImage(&ImaskCodeBookCC);

	cvReleaseImage(&circleImage);
}

#define RECORD_DETAILS
CvVideoWriter* writer = 0;
#ifdef RECORD_DETAILS
CvVideoWriter *writer0 = 0, *writer1 = 0, *writer2 = 0, *writer3 = 0, *writer4 = 0;
#endif
void releaseAllVideos(void)
{
	cvReleaseVideoWriter(&writer);
#ifdef RECORD_DETAILS
	cvReleaseVideoWriter(&writer0);
	cvReleaseVideoWriter(&writer1);
	cvReleaseVideoWriter(&writer2);
	cvReleaseVideoWriter(&writer3);
	cvReleaseVideoWriter(&writer4);
#endif
}

int recordVideo(IplImage* img, CvVideoWriter* video)
{
	IplImage* tmp;
	int r;
	if(!img)
		return -1;
	if(!video)
		return -1;
	if(img->nChannels == 1)
	{
		tmp = cvCreateImage(cvGetSize(img), 8, 3);
		cvCvtColor(img, tmp, CV_GRAY2BGR);
		r = cvWriteFrame(video, tmp);
		cvReleaseImage(&tmp);
	}
	else if(img->nChannels == 3)
	{
		r = cvWriteFrame(video, img);
	}
	return r;
}

void silhoutte_reduction(IplImage* src, IplImage* dst, int codeCvt )
{
	if (codeCvt != CV_BGR2HSV)
		cvCvtColor(src, src, codeCvt);
	IplImage* h = cvCreateImage(cvGetSize(src), 8, 1);
	IplImage* s = cvCreateImage(cvGetSize(src), 8, 1);
	IplImage* v = cvCreateImage(cvGetSize(src), 8, 1);
	cvSplit(src, h, s, v, 0);
	
	cvThreshold(h, h, h_min, h_max, CV_THRESH_TOZERO);	//CV_THRESH_TOZERO = 3,  /* value = value > threshold ? value : 0           */
	cvThreshold(h, h, h_max, h_max, CV_THRESH_TOZERO_INV);	//CV_THRESH_TRUNC = 2,  /* value = value > threshold ? threshold : value   */
	cvThreshold(s, s, s_min, s_max, CV_THRESH_TOZERO);	//CV_THRESH_TOZERO_INV  =4,  /* value = value > threshold ? 0 : value           */
	cvThreshold(s, s, s_max, s_max, CV_THRESH_TOZERO_INV);
	cvThreshold(v, v, v_min, v_max, CV_THRESH_TOZERO);
	cvThreshold(v, v, v_max, v_max, CV_THRESH_TOZERO_INV);

	cvReleaseImage(&h);
	cvReleaseImage(&s);
	cvReleaseImage(&v);
	cvMerge(h, s, v, 0, dst);
}

static void drawItem(CvArr* img, char* name, int value, CvPoint point, const CvFont* font, CvScalar color)
{
	char text[50];
	sprintf(text, "%s: %d  ", name, value);
	cvPutText(img, text, point, font, color);
}

/*
 *
 */
static void drawLines (CvArr* img, CvPoint pt1, CvPoint pt2, CvScalar color, int dx, int dy )
{
	/* Main line */
	cvLine(img, pt1, pt2, color, 1, 8, 0);
	/* 2 sub parallel line */
	cvLine(img, cvPoint(pt1.x - dx, pt1.y - dy), cvPoint(pt2.x - dx, pt2.y - dy), color, 1, 8, 0);
	cvLine(img, cvPoint(pt1.x + dx, pt1.y + dy), cvPoint(pt2.x + dx, pt2.y + dy), color, 1, 8, 0);
}

/*
 * Draw information: frame, number of In/Out people, detection line
 * mode1: draw vertical line
 * mode2: draw horizontal line
 */
void mode1(IplImage* image)
{
	CvScalar textColor = cvScalar(0, 0, 255, 0);    // red
	drawLines(image, cvPoint(center_x, 0), cvPoint(center_x, CamHeight), CV_RGB(255, 0, 0), line_w, 0);
	drawItem(image, "frame", nframes, cvPoint(0, 30), &g_font, textColor);
	drawItem(image, "OUT", count_out, cvPoint(0, 60), &g_font, textColor);
	drawItem(image, "IN", count_in, cvPoint(CamWidth - 90, 60), &g_font, textColor);

}

void mode2(IplImage* image)
{
	CvScalar textColor = cvScalar(0, 0, 255, 0);    // red
	drawLines(image, cvPoint(0, center_y), cvPoint(CamWidth, center_y), CV_RGB(255, 0, 0), 0, line_w);
	drawItem(image, "frame", nframes, cvPoint(0, 30), &g_font, textColor);
	drawItem(image, "OUT", count_out, cvPoint(0, 60), &g_font, textColor);
	drawItem(image, "IN", count_in, cvPoint(0, CamHeight - 20), &g_font, textColor);
}

/*
 * Create control table to:
 * name = "Control"
 * - "update(s)"   : modify time to update codebook (in seconds)
 * - "clear(s)"    : modify time to clear codebook (in seconds)
 * - "poly1_hull0" : switch to Poly(1) or Hull(0) for cvSegmentFGMask function
 * - "min len" "max len"   : modify perimeter condition of object recognition (in pixels)
 * - "min area" "max area" : modify area condition of object recognition (in pixels)
 */
void control_trackbar(const char* name)
{
	cvNamedWindow(name, CV_WINDOW_AUTOSIZE);
	cvResizeWindow(name, 300, 540);
	cvCreateTrackbar("update(s)", "Control", &update_duration, 120, NULL);	//seconds
	cvCreateTrackbar("clear(s)", "Control", &clear_duration, 5*60, NULL);		//seconds
	cvCreateTrackbar("poly1_hull0", "Control", &poly1Hull0, 1, NULL);
	cvCreateTrackbar("width", "Control", &line_w, 80, NULL);
	//cvCreateTrackbar("duration (ms)", "Control", &duration, 2000, NULL);
	//cvCreateTrackbar("seg_thresh (ms)", "Control", &seg_thresh, 2000, NULL);
	cvCreateTrackbar("min len", "Control", &min_len, 10000, NULL);
	cvCreateTrackbar("max len", "Control", &max_len, 10000, NULL);
	cvCreateTrackbar("min area", "Control", &min_area, 320*240, NULL);
	cvCreateTrackbar("max area", "Control", &max_area, 480*640, NULL);

}

/*
 * Create threshold table to:
 * name = "Set YUV Background"
 * - "modMin" "modMax" : modify max/min of codebook configuration
 * - "h_min" "h_max"   : modify max/min threshold on H-plane
 * - "s_min" "s_max"   : modify max/min threshold on S-plane
 * - "v_min" "v_max"   : modify max/min threshold on V-plane
 */
void threshold_trackbar(const char* name)
{
	cvNamedWindow(name , CV_WINDOW_AUTOSIZE);
	cvResizeWindow(name, 260, 450);
	cvCreateTrackbar("modMin", name, &modMin, 255, NULL);
	cvCreateTrackbar("modMax", name, &modMax, 255, NULL);
	cvCreateTrackbar("h_min", name, &h_min, 255, NULL);
	cvCreateTrackbar("h_max", name, &h_max, 255, NULL);
	cvCreateTrackbar("s_min", name, &s_min, 255, NULL);
	cvCreateTrackbar("s_max", name, &s_max, 255, NULL);
	cvCreateTrackbar("v_min", name, &v_min, 255, NULL);
	cvCreateTrackbar("v_max", name, &v_max, 255, NULL);
}

/*
 * Check perimeter
 * Return 1 if true, 0 if false
 */
static int checkContourPerimeter(CvSeq* c, double min, double max)
{
	double c_len = cvContourPerimeter(c);
	if ((double)min < c_len || c_len < (double)max)
		return 1;
	return 0;
}

/*
 * Check area
 * Return 1 if true, 0 if false
 */
static int checkContourAre(CvSeq* c, double min, double max)
{
	double c_area = cvContourArea(c, CV_WHOLE_SEQ, 0);
	if ((double)min < c_area || c_area < (double)max)
		return 1;
	return 0;
}

/*
 * Calculate center point of rectangle
 */
static CvPoint calcMomentCenterPoint(IplImage* image, CvRect rect)
{
	CvPoint center;
	CvMoments moments;
	cvSetImageROI(image, rect);
	cvMoments(image, &moments, 1);
	double M00 = cvGetSpatialMoment(&moments, 0, 0);
	double M01 = cvGetSpatialMoment(&moments, 0, 1);
	double M10 = cvGetSpatialMoment(&moments, 1, 0);
	center.x = (int)(M10 / M00);	// x of ROI
	center.y = (int)(M01 / M00);	// y of ROI
	center = cvPoint((rect.x + center.x), (rect.y + center.y));  // (x,y) of image
	cvResetImageROI(image);
	return center;
}
/*
 * Filter: reject too small and too large object
 * Return -1 if false
 */

/*
 *
 */
static void drawCoordinate(CvArr* img, CvPoint point, const CvFont* font, CvScalar color)
{
			char text[50];
			sprintf(text, "%d,%d", point.x, point.y);
			cvPutText(img, text, point, font, color);
}

int reject_small_and_large_object(
		IplImage* src, IplImage* dst,
		IplImage* circleImage,
		CvPoint* center, 
		int /*&numOfCenters*/ *numOfCenters,	/* output */
		CvMemStorage* storage) 
{
	IplImage *temp;

//	CvSeq* contours = 0;
	if (!storage)
		storage = cvCreateMemStorage(0);
	else cvClearMemStorage(storage);

	if(!src || !dst || !circleImage)
	{
		printf("ERROR: reject_small_and_large_object NULL input\n");
		return -1;
	}
	
	cvZero(dst);			//output image
	cvZero(circleImage);
	// Check channel
	if(src->nChannels == 1 && dst->nChannels == 1 && circleImage->nChannels != 1)
	{
		printf("ERROR: nChannels != 1\n");
		return -1;
	}

	temp = cvCreateImage(cvGetSize(src), 8, 1);
	cvCopyImage(src, temp);	//input image

	
//	IplConvKernel *element = cvCreateStructuringElementEx(10, 30, 5, 14, CV_SHAPE_RECT, 0);
	cvMorphologyEx(temp, temp, 0, 0, CV_MOP_OPEN, 2);
	cvMorphologyEx(temp, temp, 0, 0, CV_MOP_CLOSE, 2);

	CvContourScanner scanner = cvStartFindContours(temp, storage, sizeof(CvContour), 0, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
	CvSeq *c;
	int numContour = 0;
	
	while ((c = cvFindNextContour(scanner)) != NULL) {
		if ( checkContourPerimeter(c, min_len, max_len) == false 
			|| checkContourAre(c, min_area, max_area) == false){
			continue;	// ignore
		}
		else {
			cvDrawContours(dst, c, WHITE, BLACK, 0, -1, 8, cvPoint(0, 0));
			CvRect rect = cvBoundingRect(c, 0);
			center[numContour] = calcMomentCenterPoint(dst, rect);
			
			cvCircle(circleImage, center[numContour], 20, WHITE, -1, 8, 0);  // gray-scale
			cvCircle(rawImage, center[numContour], 10, BLUE, -1, 8, 0);	// color
			cvRectangleR(rawImage, rect, WHITE, 1, 8, 0);	//  color
			drawCoordinate(rawImage, center[numContour], &g_font, BLUE);

			numContour++;
			if (numContour > *numOfCenters) {
				numContour = *numOfCenters;
				break;
			}
		}
	}
	*numOfCenters = numContour;
	cvEndFindContours(&scanner);
	if(temp) cvReleaseImage(&temp);
	return 0;
}



//function in mhi_update()
/*
 * Return:
 *    1 = in, 2 = out, 0 = NULL
 */
int detect_object(
	int detectable_direction,
	int object_center,
	int center_of_detect_line,
	int num_of_pixel_on_line,
	int direction_point,
	Tracking *track_line,
	int index_of_detected_object,
	double exist_time	//	exist_time < line_w/2 ,  exist_time ~= update
) {
	int i;
	int _index = index_of_detected_object;	
	bool _isDetected = false;
	
	for (i = 0; i < num_of_pixel_on_line; i++) {
		if (track_line[i].value) {	//kiem tra cac diem cu vat da di qua
			//if (useAVIfile) {
				track_line[i].time_count += 1;
				if (track_line[i].time_count > track_line[i].exist_time) {
					track_line[i].time_count = 0;
					track_line[i].value = false;
				}
			//}
			/*else {
				track_line[i].time_count = (double)clock() / CLOCKS_PER_SEC;	//get current time (second)
				//if (track_line[i].time_count - track_line[i].exist_time > 3.0) {
				if (track_line[i].time_count - track_line[i].start_time  > track_line[i].exist_time > 3.0) {
					track_line[i].time_count = 0;
					track_line[i].value = false;
				}
			}*/
		}
	}
	if (center_of_detect_line - line_w < detectable_direction  && detectable_direction < center_of_detect_line + line_w) {
		_index = object_center;	//luu toa do diem moi bi phat hien
		if (track_line[_index].value == false) {
			for (i = -saiso; i < saiso; i++) {
				if ((_index + i <0) || (_index + i >= num_of_pixel_on_line))
					continue;	// tranh truong hop bi loi cac diem nam ben ngoai khung hinh
				
				// khoi tao cac diem duoc danh dau xuat hien vat
				track_line[_index + i].value = true;
				track_line[_index + i].time_count = 0;	//bat dau dem thoi gian ton tai
				
				//if (useAVIfile) 			
					track_line[_index + i].exist_time = exist_time;
				/*else {
					// use camera
					//track_line[_index + i].exist_time = (double)clock() / CLOCKS_PER_SEC;	//get current time (second)
					track_line[_index + i].exist_time = 2.0;	//thoi gian ton tai (second)
					track_line[_index + i].start_time = (double)clock() / CLOCKS_PER_SEC;	//get current time (second)
				}*/
				_isDetected = true;
			}
		}
		// Confirm moving orientation
		if (_isDetected) {
			if (direction_point - detectable_direction > 0) { return 1; }	//input
			else { return 2; }	//output
		}
	}
	return 0;
}

/*
 * Return display mode
 */
int selectDisplay (IplImage* display, int mode)
{
	cvZero(display);
	switch (mode)
	{
	case 1:
		cvMerge(ImaskCodeBook, 0, 0, 0, display);
		break;
	case 2:
		cvMerge(0, 0, ImaskCodeBookCC, 0, display);
		break;
	case 3:
		cvMerge(0, circleImage, 0, 0, display);
		break;
	case 4:
		cvCopyImage(motion, display);
		break;
	case 5:
		cvCopyImage(hsvImage, display);
		break;
	case 0:
	default:
		cvCopyImage(rawImage, display);
		mode = 0;
		break;
	}
	return mode;
}

int window_mode = 0;
bool pause = false;
bool auto_update = false;
bool update = false, clear = false;
bool temp_image = false;
bool color_window = true;

/*
 *
 */
void updateBackgroundCodeBook (void)
{
	current_time = (double)clock() / CLOCKS_PER_SEC; // get current time in seconds
	if (current_time - start_update_time >= update_duration)
	{
		start_update_time = (double)clock() / CLOCKS_PER_SEC;
		update = true;
		printf("update\n");
	}

	if (current_time - start_clear_time >= clear_duration)
	{
		start_clear_time = (double)clock() / CLOCKS_PER_SEC;
		clear = true;
		printf("clear\n");
	}
}

static void saveAllImages(void)
{
	cvSaveImage("ImaskCodeBook.jpg", ImaskCodeBook, 0);
	cvSaveImage("ImaskCodeBookCC.jpg", ImaskCodeBookCC, 0);
	cvSaveImage("circleImage.jpg", circleImage, 0);
	cvSaveImage("hsvImage.jpg", hsvImage, 0);
	cvSaveImage("rawImage.jpg", rawImage, 0);
}

void controller(char key)
{
	switch (key) {
		printf("key: %c\n", key);
	case 'a':
		auto_update = !auto_update;
		printf("auto update: %d\n", auto_update);
		break;
	case 'c':
		clear = true;
		printf("clear codebook\n");
		break;
	case 'u':
		update = true;
		printf("update codebook\n");
		break;
	case '1':
		window_mode = 1;
		printf("foreground image\n");
		break;
	case '2':
		window_mode = 2;
		printf("connected component image\n");
		break;
	case '3':
		window_mode = 3;
		printf("circle image\n");
		break;
	case '4':
		window_mode = 4;
		printf("motion image\n");
		break;
	case '5':
		window_mode = 5;
		printf("hsv image\n");
		break;
	case '6':
		window_mode = 0;
		printf("color image\n");
		break;
	case 's':
		saveAllImages();
		printf("save frame\n");
		break;
	case 'r':
		cvBGCodeBookClearStale(model, 0, cvRect(0,0,0,0), 0 );
		//cvClearMemStorage(storage);
		ReleaseImages();
		nframes = 0;
		releaseTracking(countOnLine);
		printf("reset\n");
		break;
	case 't':
		control_trackbar("Control");
		threshold_trackbar("Set YUV Background");
		break;
	case 'q':
		temp_image = !temp_image;
		printf("Process Window\n");
		break;
	case 'w':
		color_window = !color_window;
		printf("color_window = %d\n", color_window);
		break;
	case 'v':
		mode = 1;
		releaseTracking(countOnLine);
		firstInMode = true;
		printf("mode: %d-vertical\n", mode);
		break;
	case 'h':	//DEFAULT
		mode = 2;
		releaseTracking(countOnLine);
		firstInMode = true;
		printf("mode: %d-horizontal\n", mode);
		break;
	case 'p':
		pause = !pause;
		printf("pause = %d\n",pause);
		break;
	}
}

int main(int argc, char *argv[]) {

	//capture = cvCreateCameraCapture(0);
	if (capture){
		// Capture from Camera
		//cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, CamWidth);
		//cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, CamHeight);*/
		;
	}
	else {
		// Capture from Video.avi
		capture = cvCreateFileCapture("sample-video.avi");
		maxFrame = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT);
		useAVIfile = true;
		fps = cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
		printf("fps: %d\n", fps);
		cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 320);
		cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 240);
	}
	nframes = 0;
	nframesToLearnBG = LEARNING_TIME;

	//initialize codebook
	model = cvCreateBGCodeBookModel();

//	bool reset = false;
//	bool hsv = false;
//	bool trackbar = true;
//	bool save = false;

	g_font = cvFont(2.0, 2);

	cvNamedWindow("Camera", CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback("Camera", on_mouse, 0);
	control_trackbar("Control");
	threshold_trackbar("Set YUV Background");

	int fcc = CV_FOURCC('D', 'I', 'V', '3');

	while (1)
	{
		if (!pause) {
			rawImage = cvQueryFrame(capture);
			if (!rawImage) break;
			++nframes;
		}

		if (nframes == 1) {
			CvSize sz = cvGetSize(rawImage);
			InitImages(sz);

			//Set color thresholds to default values
			model->cbBounds[0] = model->cbBounds[1] = model->cbBounds[2] = 10;	//10
			model->modMin[0] = 70;	model->modMax[0] = 70;//H
			model->modMin[1] = 10;	model->modMax[1] = 40;	//S
			model->modMin[2] = modMin;	model->modMax[2] = modMax;	//V

			selection.height = sz.height/2;
			selection.width = sz.width/2;

			CamWidth = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
			CamHeight = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
			center_x = CamWidth / 2;
			center_y = CamHeight / 2;

			count_in = 0;
			count_out = 0;

			if (writer) {
				cvReleaseVideoWriter(&writer);
				printf("start record new video\n");
			}
			writer = cvCreateVideoWriter("Video Record.avi", fcc, 24, sz, 1);

			// record processing steps
#ifdef RECORD_DETAILS
			writer0 = cvCreateVideoWriter("foreground.avi", fcc, 24, sz, 1);
			writer1 = cvCreateVideoWriter("connected component.avi", fcc, 24, sz, 1);
			writer2 = cvCreateVideoWriter("circle.avi", fcc, 24, sz, 1);
			writer3 = cvCreateVideoWriter("motion.avi", fcc, 24, sz, 1);
			writer4 = cvCreateVideoWriter("hsv.avi", fcc, 24, sz, 1);
#endif
		}

		if (rawImage)
		{
			cvCvtColor(rawImage, hsvImage, CV_BGR2HSV);		//BGR -> HSV
			silhoutte_reduction(hsvImage, hsvImage, CV_BGR2HSV);

			if (update || (nframes - 1 < nframesToLearnBG)) {
				cvBGCodeBookUpdate(model, hsvImage, cvRect(0,0,0,0), 0);
				update = false;										//update codebook
				start_update_time = (double)clock() / CLOCKS_PER_SEC;
			}

			if (clear || (nframes - 1 == nframesToLearnBG)) {
				cvBGCodeBookClearStale(model, model->t / 2, cvRect(0,0,0,0), 0);
				clear = false;										//clear codebook
				start_clear_time = (double)clock() / CLOCKS_PER_SEC;
			}

			//Find the foreground if any
			if (nframes - 1 >= nframesToLearnBG)
			{

				// Find foreground by codebook method
				cvBGCodeBookDiff(model, hsvImage, ImaskCodeBook, cvRect(0,0,0,0));
				int numOfObject = 50;
				//reject_small_and_large_object(ImaskCodeBook, ImaskCodeBook, circleImage, center, numOfObject, storage);
				// This part just to visualize bounding boxes and centers if desired
				//cvCopy(ImaskCodeBook, ImaskCodeBookCC);
				//cvSegmentFGMask(ImaskCodeBookCC, poly1Hull0, 4.0, 0, cvPoint(0, 0));
				reject_small_and_large_object(ImaskCodeBook, ImaskCodeBookCC, circleImage, center, &numOfObject, storage);
				update_mhi(circleImage, motion, 50);
			}

			if (select_object && selection.width > 0 && selection.height > 0)
			{
				cvRectangleR(rawImage, selection, cvScalar(0, 255, 255, 0), 3, 8, 0);
				cvSetImageROI(rawImage, selection);
				cvXorS(rawImage, cvScalarAll(125), rawImage, 0);
				cvResetImageROI(rawImage);
			}

			// show on screen
			window_mode = selectDisplay(tempImage, window_mode);
			if (temp_image)	cvShowImage("Process Window", tempImage);
			else cvDestroyWindow("Process Window");


			// draw information
			if (mode == 1)	mode1(rawImage);	 //VERTICAL MODE

			if (mode == 2)	mode2(rawImage);		//HORIZONTAL MODE

			if (color_window)	cvShowImage("Camera", rawImage);
			else cvDestroyWindow("Camera");

			/* Record video */
			recordVideo(rawImage, writer);          // BGR
#ifdef RECORD_DETAILS
			recordVideo(ImaskCodeBook, writer0);	// gray-scale --> BGR , "foreground.avi"
			recordVideo(ImaskCodeBookCC, writer1);  // gray-scale --> BGR , "connected component.avi"
			recordVideo(circleImage, writer2);      // gray-scale --> BGR , "circle.avi"
			recordVideo(motion, writer3);           // BGR , "motion.avi"
			recordVideo(hsvImage, writer4);         // BGR , "hsv.avi"
#endif

			if (auto_update) {
				updateBackgroundCodeBook();
			}

		}

		if (useAVIfile) {
			//pos = cvGetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES);
			//cvCreateTrackbar("frame", "Camera", &pos, maxFrame, onTrackbarSlide);
		}

		// User input:
		int key = cvWaitKey(10);
		//key = tolower(key);
		if (key == 27)	 break;	//end processing on ESC
		controller(key);

	}

	releaseAllVideos();

	return 0;
}
