#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp" // Kalman Filter
#include <stdio.h>

#define DEBUG_PRINT
#define DEBUG_VIDEO

using namespace std;
using namespace cv;

static void help()
{
 printf("\nDo background segmentation, especially demonstrating the use of cvUpdateBGStatModel().\n"
"Learns the background at the start and then segments.\n"
"Learning is togged by the space key. Will read from file or camera\n"
"Usage: \n"
"			./bgfg_segm [--camera]=<use camera, if this key is present>, [--file_name]=<path to movie file> \n\n");
}

const char* keys =
{
    "{c |camera   |true    | use camera or not}"
    "{fn|file_name|tree.avi | movie file             }"
};



int silhoutte_reduction(Mat& src, Mat& dst)
{
	int code;
	int h_min  = 10, h_max  = 155;	//hue mau sac
	int s_min  = 0,  s_max  = 255;	//saturation do tuong phan
	int v_min  = 0,  v_max  = 255;		//value do sang
	Mat hsv;
	vector<Mat> planes;     // Use the STL's vector structure to store multiple Mat objects
	if(src.empty())
		return -1;
	
	/// Transform it to HSV
    cvtColor( src, hsv, COLOR_BGR2HSV );

	split(hsv, planes);  // split the image into separate color planes (Y U V)
	
	 // 0: Binary  /* value = value > threshold ? max_value : 0       */
     // 1: Binary Inverted /* value = value > threshold ? 0 : max_value       */
     // 2: Threshold Truncated /* value = value > threshold ? threshold : value   */
     // 3: Threshold to Zero /* value = value > threshold ? value : 0           */
     // 4: Threshold to Zero Inverted  /* value = value > threshold ? 0 : value           */
   
	threshold( planes[0], planes[0], h_min, h_max, THRESH_TOZERO );
	threshold( planes[0], planes[0], h_max, h_max, THRESH_TOZERO_INV );
	threshold( planes[1], planes[1], s_min, s_max, THRESH_TOZERO );
	threshold( planes[1], planes[1], s_max, s_max, THRESH_TOZERO_INV );
	threshold( planes[2], planes[2], v_min, v_max, THRESH_TOZERO );
	threshold( planes[2], planes[2], v_max, v_max, THRESH_TOZERO_INV );
	
	merge(planes , dst);
	return 0;
}

static void OpenClose(Mat& src, Mat& dst)
{
	int element_shape = MORPH_RECT;//MORPH_RECT; // MORPH_CROSS
	int iterations = 1;
	Mat element_1 = getStructuringElement(element_shape, Size(3,3), Point(-1,-1) );
	
	Mat element_2 = getStructuringElement(element_shape, Size(3,3), Point(-1,-1) );
	
	morphologyEx(src, dst, CV_MOP_OPEN, element_1,
					Point(-1,-1), iterations, BORDER_CONSTANT, morphologyDefaultBorderValue());
	
#ifdef DEBUG_VIDEO
	namedWindow("Open",0);
	imshow("Open",dst);
#endif
	morphologyEx(dst, dst, CV_MOP_CLOSE, element_2, 
					Point(-1,-1), iterations+10, BORDER_CONSTANT, morphologyDefaultBorderValue());
#ifdef DEBUG_VIDEO
	namedWindow("Open/Close", 0);
	imshow("Open/Close",dst);
#endif
}

/*
 * Check perimeter
 * Return length if true, -1 if false
 */
static double checkContourPerimeter(const vector<Point>& c, double min, double max)
{
	double c_len = arcLength(Mat(c), true);
	if ((double)min < c_len && c_len < (double)max)
	{
		return c_len;
	}

	return -1;
}

/*
 * Check area
 * Return area if true, -1 if false
 */
static double checkContourAre(const vector<Point>& c, double min, double max)
{
	double c_area = fabs(contourArea(Mat(c), false));
	if ((double)min < c_area && c_area < (double)max)
	{
		return c_area;
	}
		
	return -1;
}

static int rejectSmallAndLargeObject(const vector<Point>& c,
		int min_len , int max_len,
		int min_area , int max_area)
{
	// min_len = 650;
	// max_len = 5000;	// 2000 - 5000
	// min_area = 6000;
	// max_area = 70000;//640 * 480;	// < 30 000 - 50 000
	double length, area;
	length = checkContourPerimeter(c, min_len, max_len);
	area = checkContourAre(c, min_area, max_area);
	
	if ( (length == -1 ) || (area == -1)){
		return false;	
	}
#ifdef DEBUG_PRINT
	printf("contour: length=%f , area=%f\n", length, area);
#endif
	return true;
}

static void refineSegments_1(const Mat& img, Mat& mask, vector<vector<Point> > & contours_poly	)
{
	vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
	
	Mat temp = mask;
	Size size = img.size();	
	/// Find contours
	findContours( temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);//CV_CHAIN_APPROX_SIMPLE );
#ifdef DEBUG_PRINT
	printf("contours=%d , hierarchy=%d \n", (int)contours.size(), (int)hierarchy.size() );
#endif
	
	if( contours.size() == 0 )
    {
		;
	}  
	else{
	    // iterate through all the top-level contours,
		// draw each connected component with its own random color

		/// Approximate contours to polygons
		// vector<vector<Point> > contours_poly;//( contours.size() );

		for( int i = 0; i < contours.size(); i++ )
		{ 
			vector<Point> c ;//= contours_poly[i];
			approxPolyDP( Mat(contours[i]), c, 3, true ); //! approximates contour or a curve using Douglas-Peucker algorithm
			
			// if(rejectSmallAndLargeObject(c,
				// 640, 5000,  /* min_len, max_len*/
				// 6000,70000 /* min_area, max_area*/
				// ) == false)
				// continue; // skip this contour							
			contours_poly.push_back(c);
		}
		printf("contours_poly.size = %d\n", (int)contours_poly.size());
	}
 
}

/// Get bounding rects and circles
static void getBoundaries(vector<vector<Point> > & contours,
 		vector<Rect>& out_boundRect,
		vector<Point2f>& out_center,
		vector<float>& out_radius )
{
	if(contours.size() > 0)
	{
		vector<Rect> boundRect( contours.size() );
		vector<Point2f> center( contours.size() );
		vector<float> radius( contours.size() );
			
		for( int i = 0; i < contours.size(); i++ )
		{ 									
			boundRect[i] = boundingRect( Mat(contours[i]) ); //! computes the bounding rectangle for a contour
			minEnclosingCircle( (Mat)contours[i], center[i], radius[i] ); //! computes the minimal enclosing circle for a set of points
		}
			
		// Save detected
		out_boundRect = boundRect;
		out_center = center;
		out_radius = radius;		
	} 
#ifdef DEBUG_PRINT
	else {	
		printf("[getBounding]Error contours.size() == 0\n") ;
	}
#endif

}

RNG rng(12345);

static void drawBoundaries( Mat& img,
		const vector<vector<Point> >& contours,
 		const vector<Rect>& boundRect,
		const vector<Point2f>& center,
		const vector<float>& radius  )
{
	// img = Mat::zeros(size, CV_8UC3);
	if(contours.size() > 0)
	{			
		/// Draw polygonal contour + bonding rects + circles
		for( int i = 0; i< contours.size(); i++ )
		{
			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			drawContours( img, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
			// rectangle( img, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
			circle( img, center[i], 5 /*(int)radius[i]*/, color, 2, 8, 0 );
		}
	} 
#ifdef DEBUG_PRINT
	// else {	
		// printf("[drawBounding]Error contours.size() == 0\n") ;
	// }
#endif

#ifdef DEBUG_VIDEO
	namedWindow("Boundaries", 0);
	imshow("Boundaries", img);	
#endif
}	
	
void refineSegments(const Mat& img, Mat& mask, Mat& dst,
		vector<Rect>& out_boundRect,
		vector<Point2f>& out_center,
		vector<float>& out_radius	)
{
	vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
	
	Mat temp = mask;
	Size size = img.size();
	/// Find contours
	findContours( temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);//CV_CHAIN_APPROX_SIMPLE );
#ifdef DEBUG_PRINT
	printf("contours=%d , hierarchy=%d \n", (int)contours.size(), (int)hierarchy.size() );
#endif
	dst = Mat::zeros(size, CV_8UC3);
	
	if( contours.size() == 0 )
    {
		;
	}  
	else{
	    // iterate through all the top-level contours,
		// draw each connected component with its own random color

		/// Approximate contours to polygons + get bounding rects and circles
		vector<vector<Point> > contours_poly( contours.size() );
		vector<Rect> boundRect( contours.size() );
		vector<Point2f> center( contours.size() );
		vector<float> radius( contours.size() );
  
		for( int i = 0; i < contours.size(); i++ )
		{ 
	
			approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true ); //! approximates contour or a curve using Douglas-Peucker algorithm
			
			const vector<Point>& c = contours_poly[i];
			if(rejectSmallAndLargeObject(c,
				640, 5000,  /* min_len, max_len*/
				6000,70000 /* min_area, max_area*/
				) == false)
				continue; // skip this contour							
			boundRect[i] = boundingRect( Mat(contours_poly[i]) ); //! computes the bounding rectangle for a contour
			minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] ); //! computes the minimal enclosing circle for a set of points
		}
	 
		
		/// Draw polygonal contour + bonding rects + circles
		for( int i = 0; i< contours.size(); i++ )
		{

						const vector<Point>& c = contours_poly[i];
			if(rejectSmallAndLargeObject(c,
				640, 5000,  /* min_len, max_len*/
				6000,70000 /* min_area, max_area*/
				) == false)
				continue; // skip this contour		
			Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			drawContours( dst, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
			rectangle( dst, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
			circle( dst, center[i], 5 /*(int)radius[i]*/, color, 2, 8, 0 );
		}
		
		// Save detected
		out_boundRect = boundRect;
		out_center = center;
		out_radius = radius;
	 
	}

#ifdef DEBUG_VIDEO
	namedWindow("Contours", 0);
	imshow("Contours",dst);	
#endif
 
}

static void drawItem(Mat& img, string name, int value, Point point, int font , Scalar& color)
{
	stringstream text;
	text << name << ": " << value;
	putText(img, text.str(), point, FONT_HERSHEY_SIMPLEX , 0.8f, color, 2, 8, false);
}

static void drawLines (Mat& img, Point pt1, Point pt2, Scalar& color, int dx, int dy )
{
	/* Main line */
	line(img, pt1, pt2, color, 1, 8, 0);
	/* 2 sub parallel line */
	line(img, Point(pt1.x - dx, pt1.y - dy), Point(pt2.x - dx, pt2.y - dy), color, 1, 8, 0);
	line(img, Point(pt1.x + dx, pt1.y + dy), Point(pt2.x + dx, pt2.y + dy), color, 1, 8, 0);
}

void drawInfo (Mat& img)
{
	Scalar color(130,130,0);
	drawLines(img, Point(0, 480/2), Point(640, 480/2), color , 0, 20 );
	drawItem(img, "frame", 22 /*nframes*/, Point(0, 30), FONT_HERSHEY_SIMPLEX, color );
	drawItem(img, "OUT", 2/* count_out */, Point(0, 60), 0, color );
	drawItem(img, "IN", 2/* count_in */, Point(0, 480 - 20), 0, color );	
	
}

/* Kalman Filter */
KalmanFilter KF(2, 1, 0);
Mat state(2, 1, CV_32F); /* (phi, delta_phi) */
Mat processNoise(2, 1, CV_32F);
Mat measurement = Mat::zeros(1, 1, CV_32F);
	
	
	
int main(int argc, const char** argv)
{
    help();

    CommandLineParser parser(argc, argv, keys);
    bool useCamera = parser.get<bool>("camera");
    string file = parser.get<string>("file_name");
    VideoCapture cap;
    bool update_bg_model = true;

    // if( argc < 2 )
        // cap.open(0);
    // else
        // cap.open(std::string(argv[1]));

    // if( !cap.isOpened() )
    // {
        // printf("\nCan not open camera or video file\n");
        // return -1;
    // }
	
	cap.open("sample-video.avi");
    if( !cap.isOpened() )
    {
        printf("can not open camera or video file\n");
        return -1;
    }

    namedWindow("image", 0);
    namedWindow("foreground mask", WINDOW_NORMAL);
    // namedWindow("foreground image", WINDOW_NORMAL);
    // namedWindow("mean background image", WINDOW_NORMAL);
    namedWindow("foreground mask 2", WINDOW_NORMAL);
	namedWindow("Draw and Display", WINDOW_NORMAL);
		namedWindow("Kalman Filter", WINDOW_NORMAL);
	
    BackgroundSubtractorMOG2 bg_model;//(100, 3, 0.3, 5);

    Mat img, fgmask, fgimg;
	Mat rawframe;


		
    for(;;)
    {
        cap >> img;

        if( img.empty() )
            break;

        //cvtColor(_img, img, COLOR_BGR2GRAY);
        if( fgimg.empty() )
          fgimg.create(img.size(), img.type());

        //update the model
        bg_model(img, fgmask, update_bg_model ? -1 : 0);

        // fgimg = Scalar::all(0);
        // img.copyTo(fgimg, fgmask);

		

        imshow("image", img);
		
        imshow("foreground mask", fgmask);
		
		Mat fgmask_2 = Mat::zeros(img.size(), CV_8UC3);
		threshold( fgmask, fgmask_2, 200, 255, THRESH_BINARY );
		imshow("foreground mask 2", fgmask_2);
		
		Mat openclose;
		OpenClose(fgmask_2, openclose);

		vector<Rect> boundRect;
		vector<Point2f> center;
		vector<float> radius;
		vector<vector<Point> > contours_poly;
		Mat contour_out;
		refineSegments_1(img, openclose, contours_poly);
		
		Mat img_KF = Mat::zeros(img.size(), CV_8UC3);
		if(contours_poly.size() > 0)
		{
			getBoundaries(contours_poly, boundRect, center, radius);	
			
					
		}
		drawBoundaries(img_KF, contours_poly , boundRect, center, radius);
		
		// refineSegments(img, openclose, contour_out,	boundRect, center, radius);
			
		// Mat img_KF = Mat::zeros(img.size(), CV_8UC3);
		// Scalar color (0,255,0);
		// for(int j = 0; j< boundRect.size(); j++)
		// {
			// rectangle( img_KF, boundRect[j].tl(), boundRect[j].br(), color, 2, 8, 0 );
			// circle( img_KF, center[j], 5/*(int)radius[j]*/, color, 2, 8, 0 );
		// }
		// imshow("Kalman Filter", img_KF);

		
		
		// Mat disp_draw;
		// img.copyTo(disp_draw);
		drawInfo(img);
		imshow("Draw and Display", img);

		
        // imshow("foreground image", fgimg);

		// Mat bgimg;
        // bg_model.getBackgroundImage(bgimg);
        // if(!bgimg.empty())
          // imshow("mean background image", bgimg );

        char k = (char)waitKey(30);
        if( k == 27 ) break;
        if( k == ' ' )
        {
            update_bg_model = !update_bg_model;
            if(update_bg_model)
                printf("Background update is on\n");
            else
                printf("Background update is off\n");
        }
    }

    return 0;
}
