#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <stdio.h>
#define BLUR_SIZE 4
using namespace cv;
__global__
void blur_grayKernel(unsigned char* in, unsigned char* out, int w, int h) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    if (Col < w && Row < h) {
        int pixVal = 0;
        int pixels = 0;
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;
                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    pixVal += in[curRow * w + curCol];
                    pixels++;
                }
            }
        }
        out[Row * w + Col] = (unsigned char)(pixVal / pixels);
    }
}

__global__
void blurKernel(unsigned char* in, unsigned char* out, int w, int h) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    if (Col < w && Row < h) {
        int pixValR = 0, pixValG = 0, pixValB = 0;
        int pixels = 0;
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;
                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    pixValB += in[(curRow * w + curCol) * 3];
                    pixValG += in[(curRow * w + curCol) * 3 + 1];
                    pixValR += in[(curRow * w + curCol) * 3 + 2];
                    pixels++;
                }
            }
        }
        out[(Row * w + Col) * 3] = (unsigned char)(pixValB / pixels);
        out[(Row * w + Col) * 3 + 1] = (unsigned char)(pixValG / pixels);
        out[(Row * w + Col) * 3 + 2] = (unsigned char)(pixValR / pixels);
    }
}
int main()
{
    cv::Mat img = cv::imread("tiger.jpg");
    // Get image dimensions
    int width = img.cols;
    int height = img.rows;
    //Blur
    cv::Mat outputImage(height, width, CV_8UC3);

    unsigned char* d_input, * d_output;
    cudaMalloc((void**)&d_input, width * height * 3 * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, width * height * 3 * sizeof(unsigned char));

    cudaMemcpy(d_input, img.data, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    blurKernel << <gridSize, blockSize >> > (d_input, d_output, width, height);

    cudaMemcpy(outputImage.data, d_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    cv::imshow("Input Image", img);
    cv::imshow("Blurred Image", outputImage);
    cv::imwrite("blur_image.jpg", outputImage);
    cv::waitKey(0);

    return 0;
}
