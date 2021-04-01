//============================================================================
// Name        : main.cpp
// Author      : Florian
// Version     : 1.0
// Copyright   : -
// Description : Lets focus with the kinect 360 camera on a read obejct. For testing purpose a red lighter was used.
//               There is just one angle we can control, so only the height dimension was used.
//               Parts of the code are from  jayrambhia from github.
//               link: https://gist.github.com/jayrambhia/5677608
//============================================================================

#include <iostream>
#include <math.h>
#include <pthread.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include <libfreenect.h>

#include<unistd.h>

#define FREENECTOPENCV_WINDOW_D "Depthimage"
#define FREENECTOPENCV_WINDOW_N "Normalimage"
#define FREENECTOPENCV_RGB_DEPTH 3
#define FREENECTOPENCV_DEPTH_DEPTH 1
#define FREENECTOPENCV_RGB_WIDTH 640
#define FREENECTOPENCV_RGB_HEIGHT 480
#define FREENECTOPENCV_DEPTH_WIDTH 640
#define FREENECTOPENCV_DEPTH_HEIGHT 480


using namespace std;
using namespace cv;

Mat depthimg, rgbimg, tempimg, canny_temp, canny_img;

pthread_mutex_t mutex_depth = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex_rgb = PTHREAD_MUTEX_INITIALIZER;
pthread_t cv_thread;
pthread_t tilt_thread;

double tilt_degree = 0;

int Red_Area = 0;
Point e;

freenect_context *f_ctx;
freenect_device *f_dev;

unsigned int microsecond = 1000000;

// callback for depthimage, called by libfreenect
void depth_cb(freenect_device *dev, void *depth, uint32_t timestamp)
 
{
    Mat depth8;
    Mat mydepth = Mat( FREENECTOPENCV_DEPTH_WIDTH,FREENECTOPENCV_DEPTH_HEIGHT, CV_16UC1, depth);

    mydepth.convertTo(depth8, CV_8UC1, 1.0/4.0);
    // lock mutex for opencv depth image
    pthread_mutex_lock( &mutex_depth );
    memcpy(depthimg.data, depth8.data, 640*480);
    // unlock mutex
    pthread_mutex_unlock( &mutex_depth );
 
}

// callback for rgbimage, called by libfreenect
void rgb_cb(freenect_device *dev, void *rgb, uint32_t timestamp)
{
 
    // lock mutex for opencv rgb image
    pthread_mutex_lock( &mutex_rgb );
    memcpy(rgbimg.data, rgb, (FREENECTOPENCV_RGB_WIDTH+0)*(FREENECTOPENCV_RGB_HEIGHT+950));
    // unlock mutex
    pthread_mutex_unlock( &mutex_rgb );
}
/*
 * thread for displaying the opencv content
 */
void *cv_threadfunc (void *ptr) {

    depthimg = Mat(FREENECTOPENCV_DEPTH_HEIGHT, FREENECTOPENCV_DEPTH_WIDTH, CV_8UC1);
    rgbimg = Mat(FREENECTOPENCV_RGB_HEIGHT, FREENECTOPENCV_RGB_WIDTH, CV_8UC3);
    tempimg = Mat(FREENECTOPENCV_RGB_HEIGHT, FREENECTOPENCV_RGB_WIDTH, CV_8UC3);
    canny_img = Mat(FREENECTOPENCV_RGB_HEIGHT, FREENECTOPENCV_RGB_WIDTH, CV_8UC1);
    canny_temp = Mat(FREENECTOPENCV_DEPTH_HEIGHT, FREENECTOPENCV_DEPTH_WIDTH, CV_8UC3);
    Mat frame_HSV, frame_threshold, frame_binary;

    //define the point in the middle of the object
    Point middle_point(FREENECTOPENCV_RGB_WIDTH/2,FREENECTOPENCV_RGB_HEIGHT/2);
    
    // Preparing the kernel matrix object
    Mat kernel = Mat::ones( 4, 4, CV_32F );
    while (1)
    {

        //lock mutex for rgb image
        pthread_mutex_lock( &mutex_rgb );

        cvtColor(rgbimg,tempimg,COLOR_BGR2RGB);

        
        //Scalar(155, 0, 0) - Scalar(255, 100, 100) is the range for RED color
        inRange(rgbimg, Scalar(155, 0, 0), Scalar(255, 100, 100), frame_threshold);
        //delete single small dots which are not part of the main object (red lighter in my case)
        cv::erode(frame_threshold,frame_threshold,kernel);
        cv::dilate(frame_threshold,frame_threshold,kernel);
        threshold( frame_threshold, frame_binary, 100,255,THRESH_BINARY );

        // find moments of the image
        Moments m = moments(frame_binary,true);
        Point p(m.m10/m.m00, m.m01/m.m00);

        //start motion when red dots are in a meaningful range (inside the image)
        if(p.y <0 || p.y > 480){
            p = middle_point;
        }

        e = middle_point - p;

        //pixelsize of the red area
        Red_Area = cv::sum( frame_binary )[0];

        // show the image with a point mark at the centroid
        circle(tempimg, p, 5, Scalar(128,0,0), -1);

        imshow(FREENECTOPENCV_WINDOW_N, tempimg);
        imshow("frame_threshold", frame_threshold);
        // unlock mutex
        pthread_mutex_unlock( &mutex_rgb );

        // wait for quit key
        if(waitKey(15) == 27)
            break;

    }
    pthread_exit(NULL);

    return NULL;

}
/*
 * thread for tilting the opencv camera
 */
void *tilt_threadfunc (void *ptr) {


    while(true){
        usleep(0.01* microsecond);
        // tilt degree is proportional to the error in the y-dimension. 
        // this is a underdamped case, solutions for the citically damped case are warmly welcomed!
        tilt_degree += 0.5/240 * e.y;

        if(tilt_degree > 27) tilt_degree = 27;
        else if(tilt_degree < -27) tilt_degree = -27;

        //size of the Red_area is not used here. You could start tiling only if the area is in some range.
        cout<< "e.y: " << e.y << " tilt_degree: " << tilt_degree << " delta_tilt_degree: " << 5.0/240 * e.y << endl;
        cout << "Red_Area: " << Red_Area << endl;

        //Tilt
        freenect_set_tilt_degs(f_dev,tilt_degree);
    }
    pthread_exit(NULL);

    return NULL;

}
int main(int argc, char** argv) {


    int res = 0;
    int die = 0;

    printf("Kinect camera test\n");
    if (freenect_init(&f_ctx, NULL) < 0)
    {
        printf("freenect_init() failed\n");
        return 1;
    }
    if (freenect_open_device(f_ctx, &f_dev, 0) < 0)
    {
        printf("Could not open device\n");
        return 1;
    }
    //tilt to zero postion
    freenect_set_tilt_degs(f_dev,tilt_degree);

    freenect_set_depth_callback(f_dev, depth_cb);
    freenect_set_video_callback(f_dev, rgb_cb);

    // create opencv display thread
    res = pthread_create(&cv_thread, NULL, cv_threadfunc, NULL);
    if (res)
    {
        printf("pthread_create failed\n");
        return 1;
    }
    // create tilt thread
    res = pthread_create(&tilt_thread, NULL, tilt_threadfunc, NULL);
    if (res)
    {
        printf("tilt_thread_create failed\n");
        return 1;
    }

    freenect_start_depth(f_dev);
    freenect_start_video(f_dev);

    while(!die && freenect_process_events(f_ctx) >= 0 );


    return 0;

}
