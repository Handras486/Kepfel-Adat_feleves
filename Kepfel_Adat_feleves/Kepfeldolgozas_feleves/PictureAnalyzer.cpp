#include "PictureAnalyzer.h"
#include "GPU_PictureAnalyzer.cuh"

using namespace cv;

const int IMG_SIZE = 10;

vector<PuzzlePiece> PictureAnalyzer::InitializeImgPieces(vector<Mat> imgPieces)
{
	vector<PuzzlePiece> puzzlePieces;

	CalculatePuzzleBorders(&imgPieces.at(0));

	for (auto imgPiece : imgPieces )
	{
		PuzzlePiece temp;
		temp.originalPiece = imgPiece;
		CreatePuzzlePiece(&temp);

		puzzlePieces.push_back(temp);
	}

	return puzzlePieces;
}

vector<Mat> PictureAnalyzer::AnalyzeImgPieces(vector<PuzzlePiece> puzzlePieces)
{
	bool IsComplete = false;
	int i = 0;
	bool IsChangeMade = false;

	while(!IsComplete)
	{

		for (int j = i + 1; j < puzzlePieces.size(); j++)
		{
			if (!puzzlePieces.at(i).IsAvailable)
			{
				break;
			}
			else if (!puzzlePieces.at(j).IsAvailable)
			{
				continue;
			}
			else
			{
				//right-0, left - 1, up - 2, down - 3
				if (AnalyzeSide(&puzzlePieces.at(i), &puzzlePieces.at(j)))
				{
					IsChangeMade = true;
					break;
				}
			}
		}

		i++;

		if (puzzlePieces[0].originalPiece.rows == ImageRows && puzzlePieces[0].originalPiece.cols == ImageCols)
		{
			IsComplete = true;
		}

		if (i == puzzlePieces.size() && IsChangeMade)
		{
			i = 0;
			IsChangeMade = false;
		}
		else if (i == puzzlePieces.size())
		{
			IsComplete = true;
		}

	}

	vector<Mat> resultImages;

	for (auto piece : puzzlePieces)
	{
		if (piece.IsAvailable)
		{
			resultImages.push_back(piece.originalPiece);
		}
	}


	return resultImages;
}

bool PictureAnalyzer::AnalyzeImage(Mat temp)
{
	
	Mat graytemp;
	Mat canny1;
	Mat threshold;
	Mat morph;
	Mat canny2;
	Mat gauss;

	//vizszintes hasonlítás
	if (temp.cols == IMG_SIZE * 2)
	{
		if (analyzeCanny1)
		{
			if (GPU_mode)
			{
				GPU_PictureAnalyzer gpuanalyzer;
				canny1 = gpuanalyzer.GetAnalyzeCanny1(&temp, true);

				int edge = countNonZero(canny1.col(IMG_SIZE)) + countNonZero(canny1.col(IMG_SIZE - 1)) +
					countNonZero(canny1.col(IMG_SIZE - 2)) + countNonZero(canny1.col(IMG_SIZE + 1));
				int surround = countNonZero(canny1.col(IMG_SIZE - 4)) + countNonZero(canny1.col(IMG_SIZE - 3)) +
					countNonZero(canny1.col(IMG_SIZE + 2)) + countNonZero(canny1.col(IMG_SIZE + 3));

				if (edge > temp.rows / 8 && edge > surround)
				{
					return false;
				}
			}
			else
			{
				GaussianBlur(temp, gauss, Size(5, 5), 0);
				Canny(gauss, canny1, 75, 150, 3, true);

				int edge = countNonZero(canny1.col(IMG_SIZE)) + countNonZero(canny1.col(IMG_SIZE - 1)) +
					countNonZero(canny1.col(IMG_SIZE - 2)) + countNonZero(canny1.col(IMG_SIZE + 1));
				int surround = countNonZero(canny1.col(IMG_SIZE - 4)) + countNonZero(canny1.col(IMG_SIZE - 3)) +
					countNonZero(canny1.col(IMG_SIZE + 2)) + countNonZero(canny1.col(IMG_SIZE + 3));

				if (edge > temp.rows / 8 && edge > surround)
				{
					return false;
				}
			}

			//imshow("Canny1vizszintestrue", canny1);
			//imshow("bemenet", temp);
			//waitKey();

			return true;
		}

		if (analyzeCanny2)
		{
			Mat verticalMask = Mat(temp.rows, 1, CV_64F, cv::Scalar(1));

			cvtColor(temp, graytemp, COLOR_BGR2GRAY);
			adaptiveThreshold(graytemp, threshold, 255, THRESH_BINARY, ADAPTIVE_THRESH_GAUSSIAN_C, 17, 0);
			morphologyEx(graytemp, morph, MORPH_OPEN, verticalMask);
			Canny(morph, canny2, 75, 150, 3, true);

			int edge = countNonZero(canny2);

			if (edge > temp.rows / 2)
			{
				return false;
			}

			return true;
		}
	}
	//függõleges hasonlítás
	else if(temp.rows == IMG_SIZE * 2)
	{
		if (analyzeCanny1)
		{
			if (GPU_mode)
			{
				//std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

				GPU_PictureAnalyzer gpuanalyzer;
				canny1 = gpuanalyzer.GetAnalyzeCanny1(&temp, false);

				//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
				//printf("GPU Eltelt ido: %d  ms \n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());

				int edge = countNonZero(canny1.row(IMG_SIZE)) + countNonZero(canny1.row(IMG_SIZE - 1)) +
					countNonZero(canny1.row(IMG_SIZE - 2)) + countNonZero(canny1.row(IMG_SIZE + 1));
				int surround = countNonZero(canny1.row(IMG_SIZE - 4)) + countNonZero(canny1.row(IMG_SIZE - 3)) +
					countNonZero(canny1.row(IMG_SIZE + 2)) + countNonZero(canny1.row(IMG_SIZE + 3));

				if (edge > temp.cols / 8 && edge > surround)
				{
					return false;
				}

			}
			else
			{
				std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

				GaussianBlur(temp, gauss, Size(5, 5), 0);
				Canny(temp, canny1, 75, 150, 3, true);

				std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
				printf("CPU Eltelt ido: %d  ms \n", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());

				int edge = countNonZero(canny1.row(IMG_SIZE)) + countNonZero(canny1.row(IMG_SIZE - 1)) +
					countNonZero(canny1.row(IMG_SIZE - 2)) + countNonZero(canny1.row(IMG_SIZE + 1));
				int surround = countNonZero(canny1.row(IMG_SIZE - 4)) + countNonZero(canny1.row(IMG_SIZE - 3)) +
					countNonZero(canny1.row(IMG_SIZE + 2)) + countNonZero(canny1.row(IMG_SIZE + 3));

				if (edge > temp.cols / 8 && edge > surround)
				{
					return false;
				}
			}

			//imshow("Canny1függölegestrue", canny1);
			//imshow("bemenet", temp);
			//waitKey();


			return true;
		}

		if (analyzeCanny2)
		{
			Mat horizontalMask = Mat(1, temp.cols, CV_64F, cv::Scalar(1));
			cvtColor(temp, graytemp, COLOR_BGR2GRAY);
			adaptiveThreshold(graytemp, threshold, 255, THRESH_BINARY, ADAPTIVE_THRESH_GAUSSIAN_C, 17, 0);
			morphologyEx(threshold, morph, MORPH_OPEN, horizontalMask);
			Canny(morph, canny2, 75, 150, 3, true);

			int edge = countNonZero(canny2);

			if (edge > temp.cols / 2)
			{
				return false;
			}

			return true;

		}
	}

	return false;
}

bool PictureAnalyzer::AnalyzeSide(PuzzlePiece* puzzle1, PuzzlePiece* puzzle2)
{
	////right-0, down-1, left-2, up-3
	Mat temp;

	if (ShapeChecker(&puzzle1->originalPiece, &puzzle2->originalPiece))
	{
		for (int i = 0; i < 4; i++)
		{
			if (i == 0 && puzzle1->originalPiece.rows == puzzle2->originalPiece.rows)
			{
				hconcat(puzzle1->sides[i], puzzle2->sides[2], temp);

				if (ColsChecker(&puzzle1->originalPiece, &temp) && AnalyzeImage(temp))
				{
					hconcat(puzzle1->originalPiece, puzzle2->originalPiece, temp);

					puzzle1->originalPiece = temp;
					CalculatePuzzleBorders(&puzzle1->originalPiece);
					CreatePuzzlePiece(puzzle1);
					puzzle2->IsAvailable = false;

					return true;
				}
			}
			else if (i == 1 && puzzle1->originalPiece.cols == puzzle2->originalPiece.cols)
			{
				vconcat(puzzle1->sides[i], puzzle2->sides[3], temp);

				if (RowsChecker(&puzzle1->originalPiece, &temp) && AnalyzeImage(temp))
				{
					vconcat(puzzle1->originalPiece, puzzle2->originalPiece, temp);

					puzzle1->originalPiece = temp;
					CalculatePuzzleBorders(&puzzle1->originalPiece);
					CreatePuzzlePiece(puzzle1);
					puzzle2->IsAvailable = false;

					return true;
				}
			}
			else if (i == 2 && puzzle1->originalPiece.rows == puzzle2->originalPiece.rows)
			{
				hconcat(puzzle2->sides[0], puzzle1->sides[i], temp);

				if (ColsChecker(&puzzle1->originalPiece, &temp) && AnalyzeImage(temp))
				{
					hconcat(puzzle2->originalPiece, puzzle1->originalPiece, temp);

					puzzle1->originalPiece = temp;
					CalculatePuzzleBorders(&puzzle1->originalPiece);
					CreatePuzzlePiece(puzzle1);
					puzzle2->IsAvailable = false;
					return true;
				}
			}
			else if (i == 3 && puzzle1->originalPiece.cols == puzzle2->originalPiece.cols)
			{
				vconcat(puzzle2->sides[1], puzzle1->sides[i], temp);

				if (RowsChecker(&puzzle1->originalPiece, &temp) && AnalyzeImage(temp))  
				{
					vconcat(puzzle2->originalPiece, puzzle1->originalPiece, temp);

					puzzle1->originalPiece = temp;
					CalculatePuzzleBorders(&puzzle1->originalPiece);
					CreatePuzzlePiece(puzzle1);
					puzzle2->IsAvailable = false;

					return true;
				}
			}
		}
	}

	return false;
}

void PictureAnalyzer::CreatePuzzlePiece(PuzzlePiece* imgPiece)
{
	imgPiece->IsAvailable = true;

	imgPiece->sides[0] = imgPiece->originalPiece(rightSide).clone();
	imgPiece->sides[1] = imgPiece->originalPiece(downSide).clone();
	imgPiece->sides[2] = imgPiece->originalPiece(leftSide).clone();
	imgPiece->sides[3] = imgPiece->originalPiece(upSide).clone();
}

void PictureAnalyzer::CalculatePuzzleBorders(Mat* imgPiece)
{
	rightSide = Rect(imgPiece->cols - (IMG_SIZE), 0, IMG_SIZE, imgPiece->rows);
	leftSide = Rect(0, 0, IMG_SIZE, imgPiece->rows);
	upSide = Rect(0, 0, imgPiece->cols, IMG_SIZE);
	downSide = Rect(0, imgPiece->rows - (IMG_SIZE), imgPiece->cols, IMG_SIZE);
}

bool PictureAnalyzer::ShapeChecker(Mat* puzzle1, Mat* puzzle2)
{
	if ((puzzle1->cols == puzzle2->cols || puzzle1->rows == puzzle2->rows))
	{
		return true;
	}


	return false;
}

bool PictureAnalyzer::ColsChecker(Mat* puzzle1side, Mat* puzzle2side)
{
	if ((puzzle1side->cols + puzzle2side->cols) > ImageCols)
	{
		return false;
	}

	return true;
}

bool PictureAnalyzer::RowsChecker(Mat* puzzle1side, Mat* puzzle2side)
{
	if ((puzzle1side->rows + puzzle2side->rows) > ImageRows)
	{
		return false;
	}

	return true;
}



