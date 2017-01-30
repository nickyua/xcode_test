//
//  main.cpp
//  opencv_test
//
//  Created by Nikita on 1/30/17.
//  Copyright © 2017 NotaBene. All rights reserved.
//

#include <iostream>
#include <stdio.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

int main(){
    //   freopen("output.txt", "w", stdout);
    VideoCapture cap(0);
    //   Mat img;
    // img = imread("img.png", IMREAD_COLOR);
    namedWindow("Webcam");
    while(true){
         Mat Webcam;
         cap.read(Webcam);
         imshow("Webcam", Webcam);
        waitKey(8);
     }
}

// changes
