#include "GPU_PictureAnalyzer.cuh"

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

#define MAX_WIDTH 100

#define BLOCK_SIZE 256

__constant__ double K[MAX_WIDTH];

__global__ void GrayScale(double* blueimg, double* greenimg, double* redimg, double* grayscaleimg, int Width, int Height)
{

    int col = (blockIdx.x * blockDim.x) + threadIdx.x;
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (col < Width && row < Height)
    {
        grayscaleimg[row * Width + col] = redimg[row * Width + col] * 0.3 + greenimg[row * Width + col] * 0.59 + redimg[row * Width + col] * 0.11;
    }

}

__global__ void Gaussian(double* In, double* Out, int kDim, int inWidth, int outWidth, int outHeight) {

    extern __shared__ double loadInGauss[];

    // halo nélküli cellák
    int trueDimX = blockDim.x - (kDim - 1);
    int trueDimY = blockDim.y - (kDim - 1);

    // BlockDim -> trueDim ne menjünk át a halo cellákon
    int col = (blockIdx.x * trueDimX) + threadIdx.x;
    int row = (blockIdx.y * trueDimY) + threadIdx.y;

    if (col < outWidth && row < outHeight) 
    {
        loadInGauss[threadIdx.y * blockDim.x + threadIdx.x] = In[row * inWidth + col];

        __syncthreads();

        if (threadIdx.y < trueDimY && threadIdx.x < trueDimX) 
        { 
            int acc = 0;
            for (int i = 0; i < kDim; ++i)
                for (int j = 0; j < kDim; ++j)
                    acc += loadInGauss[(threadIdx.y + i) * blockDim.x + (threadIdx.x + j)] * K[(i * kDim) + j];
            Out[row * inWidth + col] = acc;
        }
    }
    else
        loadInGauss[threadIdx.y * blockDim.x + threadIdx.x] = 0;
}



void generateGaussian(vector<double>& K, int dim, int radius) {
    double stdev = 1.0;
    double pi = 355.0 / 113.0;
    double constant = 1.0 / (2.0 * pi * pow(stdev, 2));

    for (int i = -radius; i < radius + 1; ++i)
        for (int j = -radius; j < radius + 1; ++j)
            K[(i + radius) * dim + (j + radius)] = constant * (1 / exp((pow(i, 2) + pow(j, 2)) / (2 * pow(stdev, 2))));
}


Mat GPU_PictureAnalyzer::GetAnalyzeCanny1(Mat* img, bool IsCols)
{
    Mat grayscale;

    //Color to GrayScale

    vector<double> host_BlueImg, host_GreenImg, host_RedImg, host_GrayScaleImg;
    double* dev_BlueImg, * dev_GreenImg, * dev_RedImg;
    double* dev_GrayScaleImg;

    Mat channels[3];
    
    cv::split(*img, channels);

    host_BlueImg.assign(channels[0].data, channels[0].data + channels[0].total());
    host_GreenImg.assign(channels[1].data, channels[1].data + channels[1].total());
    host_RedImg.assign(channels[2].data, channels[2].data + channels[2].total());
    host_GrayScaleImg.resize(img->cols * img->rows, 0);

    gpuErrchk(cudaMalloc((void**)&dev_BlueImg, host_BlueImg.size() * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&dev_GreenImg, host_GreenImg.size() * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&dev_RedImg, host_RedImg.size() * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&dev_GrayScaleImg, host_GrayScaleImg.size() * sizeof(double)));

    (cudaMemcpy(dev_BlueImg, host_BlueImg.data(), host_BlueImg.size() * sizeof(double), cudaMemcpyHostToDevice));
    (cudaMemcpy(dev_GreenImg, host_GreenImg.data(), host_GreenImg.size() * sizeof(double), cudaMemcpyHostToDevice));
    (cudaMemcpy(dev_RedImg, host_RedImg.data(), host_RedImg.size() * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_GrayScaleImg, host_GrayScaleImg.data(), host_GrayScaleImg.size() * sizeof(double), cudaMemcpyHostToDevice));

    int blocksize;
    dim3 blockCount;
    dim3 gridCount;

    if (IsCols)
    {
        blocksize = img->cols;
        blockCount = dim3(blocksize, blocksize);
    }
    else if(!IsCols)
    {
        blocksize = img->rows;
        blockCount = dim3(blocksize, blocksize);
    }

    gridCount = dim3(ceil(img->cols / blocksize), ceil(img->rows / blocksize));

    //std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();

    GrayScale <<< gridCount, blockCount, blocksize * blocksize * sizeof(double) >> > (dev_BlueImg, dev_GreenImg, dev_RedImg, dev_GrayScaleImg, img->cols, img->rows);
    
    //std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
    //printf("GPU Eltelt ido (csak kernel - GrayScale): %d  ms \n", std::chrono::duration_cast<std::chrono::microseconds>(end1 - begin1).count());

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(host_GrayScaleImg.data(), dev_GrayScaleImg, host_GrayScaleImg.size() * sizeof(double), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(dev_BlueImg));
    gpuErrchk(cudaFree(dev_GreenImg));
    gpuErrchk(cudaFree(dev_RedImg));
    gpuErrchk(cudaFree(dev_GrayScaleImg));

    vector<int> toInteger(host_GrayScaleImg.begin(), host_GrayScaleImg.end());
    grayscale = Mat(toInteger).reshape(0, img->rows);
    grayscale.convertTo(grayscale, CV_8UC1);

    //imshow("grayscale", grayscale);      // itt kép kimutatása!
    //waitKey();

    //Gaussian

    vector<double> host_KernelImg;
    vector<double> host_InImg,  host_OutImg;

    double* dev_InImg;
    double* dev_OutImg;

    int inCols = grayscale.cols, inRows = grayscale.rows;
    int kDim, kRadius;
    int outCols, outRows;

    int max = 0;
    double threads = 8;

    host_InImg.assign(grayscale.data, grayscale.data + grayscale.total());
    host_OutImg.resize(inCols * inRows, 0);

    //IMG_SIZE = 10 -> Width/Height = 20 jelenleg, max radius 9

    kDim = 3;
    if (IsCols && kDim >= img->cols )
    {
        cerr << "Kernel dimension too big!";
        exit(0);
    }
    else if(kDim >= img->rows)
    {
        cerr << "Kernel dimension too big!";
        exit(0);
    }

    kRadius = floor(kDim / 2);
    host_KernelImg.resize(pow(kDim, 2), 0);

    //copyright internet matematikusok
    generateGaussian(host_KernelImg, kDim, kRadius);

    outCols = inCols; 
    outRows = inRows;

    gpuErrchk(cudaMalloc((void**)&dev_InImg, host_InImg.size() * sizeof(double)));
    gpuErrchk(cudaMalloc((void**)&dev_OutImg, host_OutImg.size() * sizeof(double)));

    gpuErrchk(cudaMemcpy(dev_InImg, host_InImg.data(), host_InImg.size() * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_OutImg, host_OutImg.data(), host_OutImg.size() * sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpyToSymbol(K, host_KernelImg.data(), host_KernelImg.size() * sizeof(double)));


    int halothreads = threads + (kDim - 1); // threadek + halo threadek

    dim3 blockCountGauss(halothreads, halothreads);
    dim3 gridCountGauss(ceil(inCols / threads), ceil(inRows / threads));

    //std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();

    Gaussian <<< gridCountGauss, blockCountGauss, halothreads * halothreads * sizeof(double) >>> (dev_InImg, dev_OutImg, kDim, inCols, outCols, outRows);
    
    //std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    //printf("GPU Eltelt ido (csak kernel - Gauss): %d  ms \n", std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count());
    
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(host_OutImg.data(), dev_OutImg, host_OutImg.size() * sizeof(double), cudaMemcpyDeviceToHost));

    //értékek normalizálása
    for (auto& value : host_OutImg)
        max = (value > max) ? value : max;
    for (auto& value : host_OutImg)                     
        value = (value * 255) / max;


    vector<int> toInt(host_OutImg.begin(), host_OutImg.end()); 
    Mat gauss = Mat(toInt).reshape(0, inRows);
    gauss.convertTo(gauss, CV_8UC1);

    gpuErrchk(cudaFree(dev_InImg));
    gpuErrchk(cudaFree(dev_OutImg));

    //imshow("gauss", gauss);       // itt kép kimutatása!
    //waitKey();

    //Canny


    cuda::GpuMat dst(gauss.rows, gauss.cols, CV_32SC1), src(gauss.rows, gauss.cols, CV_32SC1);
    Mat result_img(gauss.rows, gauss.cols, CV_32SC1);

    // CPU változat
    //Canny(gauss, result_img, 75, 150, 3, true);

    src.upload(gauss);

    cv::Ptr<cv::cuda::CannyEdgeDetector> ptr_canny = cuda::createCannyEdgeDetector(75, 150, 3, true); 

    //std::chrono::steady_clock::time_point begin3 = std::chrono::steady_clock::now();

    ptr_canny->detect(src, dst);

    //std::chrono::steady_clock::time_point end3 = std::chrono::steady_clock::now();
    //printf("GPU Eltelt ido (csak kernel - Canny): %d  ms \n", std::chrono::duration_cast<std::chrono::microseconds>(end3 - begin3).count());

    dst.download(result_img);

    //imshow("canny", result_img);      // itt kép kimutatása!
    //waitKey();

    return result_img;
}