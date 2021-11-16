#pragma once

#include <vector>
#include <opencv2\opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

using namespace cv;
using namespace std;

class GPU_PictureAnalyzer
{

public:
	Mat GetAnalyzeCanny1(Mat* img, bool IsCols);

private:

};

