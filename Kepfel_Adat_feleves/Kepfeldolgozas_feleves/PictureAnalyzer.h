#pragma once

#include <vector>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

extern struct PuzzlePiece
{
	bool IsAvailable;
	Mat originalPiece;

	//right - 0, down - 1, left - 2, up - 3
	Mat temp;
	vector<Mat> sides = {temp, temp, temp, temp};
};


class PictureAnalyzer
{

public:
	PictureAnalyzer(int cols, int rows)
	{
		ImageCols = cols;
		ImageRows = rows;
	};

	vector<PuzzlePiece> InitializeImgPieces(vector<Mat> imgPieces);
	vector<Mat> AnalyzeImgPieces(vector<PuzzlePiece> puzzlePieces);

private:
	bool AnalyzeImage(Mat temp);
	bool AnalyzeSide(PuzzlePiece* puzzle1, PuzzlePiece* puzzle2);
	void CreatePuzzlePiece(PuzzlePiece* piece);
	void CalculatePuzzleBorders(Mat* piece);
	bool ShapeChecker(Mat* puzzle1side, Mat* puzzle2side);
	bool ColsChecker(Mat* puzzle1side, Mat* puzzle2side);
	bool RowsChecker(Mat* puzzle1side, Mat* puzzle2side);

	int ImageCols;
	int ImageRows;

	bool GPU_mode = true;

	bool analyzeCanny1 = true;
	bool analyzeCanny2 = false;

	Rect rightSide;
	Rect leftSide;
	Rect upSide;
	Rect downSide;
};

