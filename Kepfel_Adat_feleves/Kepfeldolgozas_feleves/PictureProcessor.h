#pragma once

#include <vector>
#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>
#include <algorithm>
#include <random>

using namespace cv;
using namespace std;

class PictureProcessor
{
public:

	PictureProcessor()
	{
		rows = 2;
		columns = 2;
	};

	Mat LoadImage(string imgname);
	vector<Mat> SplitImage(Mat& picture, int rows, int columns);
	void RandomizeImgPieces(vector<Mat>* imgPieces);
	void randomizeImgRotations(vector<Mat>* imgPieces);

	bool GPU_mode = true;

private:
	int rows;
	int columns;

};

