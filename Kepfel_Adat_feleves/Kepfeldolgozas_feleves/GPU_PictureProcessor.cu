#include "GPU_PictureProcessor.cuh"

using namespace cv;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


vector<Mat> GPU_PictureProcessor::SplitImage(Mat& picture, int rows, int columns)
{

    vector<Mat> imgPieces;

    for (int y = 0; y < picture.cols; y += picture.cols / columns)
    {
        for (int x = 0; x < picture.rows; x += picture.rows / rows)
        {
            imgPieces.push_back(picture(Rect(y, x, (picture.cols / columns), (picture.rows / rows))).clone());
        }
    }

	return imgPieces;
}