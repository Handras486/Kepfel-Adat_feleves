#include <iostream>
#include <opencv2\opencv.hpp>
#include "PictureProcessor.h"
#include "PictureAnalyzer.h"

using namespace cv;

int main()
{


	PictureProcessor proc;
	Mat picture = proc.LoadImage("macska");

	//csak kijelz�s miatt
	//imshow("picture", picture);
	//waitKey();

	vector<Mat> imgPieces = proc.SplitImage(picture, 4, 4);     

	//csak kijelz�s miatt
	//imshow("imgPiece", imgPieces[0]);

	PictureAnalyzer analyzer(picture.cols, picture.rows);
	vector<PuzzlePiece> puzzlepieces = analyzer.InitializeImgPieces(imgPieces);

	vector<Mat> resultImages = analyzer.AnalyzeImgPieces(puzzlepieces);

	int counter = 0;

	for (auto img : resultImages)
	{
		imshow(format("K�sz k�p" + counter), img);
		counter++;
	}

	waitKey();
}

