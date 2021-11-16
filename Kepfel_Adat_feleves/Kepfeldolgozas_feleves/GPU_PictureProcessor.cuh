#pragma once

#include <vector>
#include <opencv2\opencv.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

using namespace cv;
using namespace std;

class GPU_PictureProcessor
{
public:

	vector<Mat> SplitImage(Mat& picture, int rows, int columns);



private:

};


