#include "PictureProcessor.h"
#include "GPU_PictureProcessor.cuh"

using namespace cv;

Mat PictureProcessor::LoadImage(string imgname)
{
	Mat picture = imread("Pictures/" + imgname + ".png", IMREAD_UNCHANGED);

	if (picture.empty())
	{
		picture = imread("Pictures/" + imgname + ".jpg", IMREAD_UNCHANGED);
	}

	if (picture.empty())
	{
		cerr << "Image not found, loading default image!" << std::endl;
		return imread("Pictures/default.png", IMREAD_UNCHANGED);
	}

	return picture;
}

vector<Mat> PictureProcessor::SplitImage(Mat& img, int columns, int rows)
{
	this->columns = columns;
	this->rows = rows;

	vector<Mat> imgPieces;

	if (!img.data || img.empty())
		cerr << "Problem loading image!" << std::endl;


	//csak kijelzés miatt:
	//cv::Mat maskImg = img.clone();

	if (img.cols % columns == 0 && img.rows % rows == 0)
	{
		if (GPU_mode)
		{
			GPU_PictureProcessor gpuproc;
			imgPieces = gpuproc.SplitImage(img, columns, rows);


		}
		else
		{
			for (int y = 0; y < img.cols; y += img.cols / columns)
			{
				for (int x = 0; x < img.rows; x += img.rows / rows)
				{
					imgPieces.push_back(img(Rect(y, x, (img.cols / columns), (img.rows / rows))).clone());

					//csak kijelzés miatt:
					//rectangle(maskImg, Point(y, x), Point(y + (maskImg.cols / columns) - 1, x + (maskImg.rows / rows) - 1), CV_RGB(255, 0, 0), 1);
					//imshow("Image", maskImg); 
					//waitKey(0); 
				}
			}
		}
	}
	else 
	{
		cerr << "Input row/column size leads to remainders, exiting! (Image size: " << img.cols << " X " << img.rows << " )";
		exit(1);
	}

	//véletlenszerû sorrendben visszaadás:
	RandomizeImgPieces(&imgPieces);


	//véletlenszerû forgatás, jelenleg nem használom
#pragma region Forgatás
	//if (imgPieces[0].cols == imgPieces[0].rows)
	//{
	//	cout << "Image pieces squares!" << endl;
	//	randomizeImgRotations(&imgPieces);
	//}
	//else
	//{
	//	randomizeImgRotations(&imgPieces);
	//	cout << "Image pieces not squares!" << endl;
	//}
#pragma endregion

	return imgPieces;
}

void PictureProcessor::RandomizeImgPieces(vector<Mat>* imgPieces)
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	auto rng = std::default_random_engine{ seed };
	std::shuffle(std::begin(*imgPieces), std::end(*imgPieces), rng);
}

void PictureProcessor::randomizeImgRotations(vector<Mat>* imgPieces)
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator(seed);

	// 0 = 90 fok .. 3 = 360 fok 
	uniform_int_distribution<int> distribution(0, 3);

	for (auto imgPiece : *imgPieces)
	{
		rotate(imgPiece, imgPiece, distribution(generator));
	}
}

