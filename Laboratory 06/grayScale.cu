#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <stdio.h>
#define CHANNELS 3
__global__
void colorToGreyscaleConversion(unsigned char* Pout, unsigned char* Pin, int width, int height) {
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    if (Col < width && Row < height) {
        // get 1D coordinate for the grayscale image
        int greyOffset = Row * width + Col;
        // one can think of the RGB image having
        // CHANNEL times columns than the grayscale image
        int rgbOffset = greyOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset]; // red value for pixel
        unsigned char g = Pin[rgbOffset + 2]; // green value for pixel
        unsigned char b = Pin[rgbOffset + 3]; // blue value for pixel
        // perform the rescaling and store it
        // We multiply by floating point constants
        Pout[greyOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}


int main()
{
    cv::Mat img = cv::imread("tiger.jpg");
    // Get image dimensions
    int width = img.cols;
    int height = img.rows;

    // Allocate memory for input and output images on the GPU
    unsigned char* d_Pin, * d_Pout;
    cudaMalloc((void**)&d_Pin, width * height * CHANNELS * sizeof(unsigned char));
    cudaMalloc((void**)&d_Pout, width * height * sizeof(unsigned char));

    // Copy input image data to the GPU
    cudaMemcpy(d_Pin, img.data, width * height * CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    colorToGreyscaleConversion << < gridSize, blockSize >> > (d_Pout, d_Pin, width, height);

    // Copy the result back to the host
    unsigned char* h_Pout = new unsigned char[width * height];
    cudaMemcpy(h_Pout, d_Pout, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Create an OpenCV Mat for the output grayscale image
    cv::Mat outputImage(height, width, CV_8UC1, h_Pout);

    // Display the images using OpenCV (optional)
    cv::imshow("Original Image", img);
    cv::imwrite("gray_image.jpg", outputImage);
    cv::imshow("Grayscale Image", outputImage);
    cv::waitKey(0);
    cv::destroyAllWindows();
    // Clean up
    delete[] h_Pout;
    cudaFree(d_Pin);
    cudaFree(d_Pout);
}