#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>

using namespace cv;

static inline Point calcPoint(Point2f center, double x, double y = 0)
{
	return center + Point2f( x , 0);
}

static void help()
{
    printf( "\nExample of c calls to OpenCV's Kalman filter.\n"
"   Tracking of rotating point.\n"
"   Rotation speed is constant.\n"
"   Both state and measurements vectors are 1D (a point angle),\n"
"   Measurement is the real point angle + gaussian noise.\n"
"   The real and the estimated points are connected with yellow line segment,\n"
"   the real and the measured points are connected with red line segment.\n"
"   (if Kalman filter works correctly,\n"
"    the yellow segment should be shorter than the red one).\n"
            "\n"
"   Pressing any key (except ESC) will reset the tracking with a different speed.\n"
"   Pressing ESC will stop the program.\n"
            );
}

int main(int, char**)
{
    help();
    Mat img(500, 500, CV_8UC3);
    KalmanFilter KF(4, 2, 0, CV_32F);
    Mat state (4, 1, CV_32F); /* [x, y, v_x, v_y]' */
	Mat processNoise(4, 1, CV_32F);
	Mat measurement = Mat::zeros(2, 1, CV_32F);  // z(2x1) = [z_x , z_y] (x_measured , y_meausred)
    char code = (char)-1;

    for(;;)
    {
        randn( state, Scalar::all(0), Scalar::all(0.1) );
        KF.transitionMatrix = *(Mat_<float>(4, 4) << 1, 0, 1*20, 0,
													0, 1, 0, 1*10,
													0, 0, 1, 0,
													0, 0, 0, 1	);

		// measurementMatrix [ [1,0,0,0] , [0,1,0,0] ]
        setIdentity(KF.measurementMatrix); // Mat (2,4) 
		
        setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
        setIdentity(KF.measurementNoiseCov, Scalar::all(10*1e-1));
        setIdentity(KF.errorCovPost, Scalar::all(1));

        randn(KF.statePost, Scalar::all(0), Scalar::all(1));

        for(;;)
        {
            Point2f center(img.cols*0.5f, img.rows*0.5f);
            float R = img.cols/3.f;
			double state_x = state.at<float>(0); // real x
			double state_y = state.at<float>(1); // real y
			Point statePt = calcPoint(center, state_x, state_y);

            Mat prediction = KF.predict();  // return Mat (4,1)
			double predict_x = prediction.at<float>(0); // [x, y, v_x, v_y]'
			double predict_y = prediction.at<float>(1);
			Point predictPt = calcPoint(center, predict_x, predict_y);

            randn( measurement, Scalar::all(0), Scalar::all(KF.measurementNoiseCov.at<float>(0))); // measurementNoiseCov Mat(2,2)

            // generate measurement, In this simulated case, we generate the measurements from an underlying “real” data model by adding random noise ourselves
            measurement += KF.measurementMatrix*state; // fake measurement z_k (2x1) = H*x_k ; (2x4)*(4x1) 

			double meas_x = measurement.at<float>(0); // z_x
			double meas_y = measurement.at<float>(1); // z_y
            Point measPt = calcPoint(center, meas_x, meas_y);
 

            // plot points
            #define drawCross( center, color, d )                                 \
                line( img, Point( center.x - d, center.y - d ),                \
                             Point( center.x + d, center.y + d ), color, 1, CV_AA, 0); \
                line( img, Point( center.x + d, center.y - d ),                \
                             Point( center.x - d, center.y + d ), color, 1, CV_AA, 0 )

            img = Scalar::all(0);
			
			drawCross( center, Scalar(255,255,255), 3 );	// Start point
            drawCross( statePt, Scalar(255,255,255), 3 );		// WHITE : real state
            drawCross( measPt, Scalar(0,0,255), 3 );			// RED   : measurement by sensor (have noise)
            drawCross( predictPt, Scalar(0,255,0), 3 );			// GREEN : prediction by Kalman
            //line( img, statePt, measPt, Scalar(0,0,255), 3, CV_AA, 0 );
            //line( img, statePt, predictPt, Scalar(0,255,255), 3, CV_AA, 0 );
			
			printf("real state : x=%0.2f, y=%0.2f, v_x=%0.2f, v_y=%0.2f\n", state.at<float>(0), state.at<float>(1), state.at<float>(2), state.at<float>(3));
			printf("measurement: z_x=%0.2f, z_y=%0.2f\n", measurement.at<float>(0), measurement.at<float>(1));
			printf("prediction : y_x=%0.2f, y_y=%0.2f, v_x=%0.2f, v_y=%0.2f\n", prediction.at<float>(0), prediction.at<float>(1), prediction.at<float>(2), prediction.at<float>(3));
			printf("\n");
			
            if(theRNG().uniform(0,4) != 0)
                KF.correct(measurement);

            randn( processNoise, Scalar(0), Scalar::all(sqrt(KF.processNoiseCov.at<float>(0, 0))));
            state = KF.transitionMatrix*state + processNoise; // [angle, velocity angle] ; x_k+1 = F*x_k + w_k

            imshow( "Kalman", img );
            code = (char)waitKey(100);

            if( code > 0 )
                break;
        }
        if( code == 27 || code == 'q' || code == 'Q' )
            break;
    }

    return 0;
}
