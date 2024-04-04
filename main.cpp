#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;
const int MAX_VALUE = 255;

void grayscale(Mat& inputImg, Mat& outputImg) 
{   
    for (int i = 0; i < inputImg.rows; i++) {
        for (int j = 0; j < inputImg.cols; j++) {
            Vec3b pixel = inputImg.at<Vec3b>(i, j);
            int b = pixel[0];
            int g = pixel[1];
            int r = pixel[2];
            uchar grayValue = (r + g + b) / 3;
            outputImg.at<Vec3b>(i, j) = Vec3b(grayValue, grayValue, grayValue);
        }
    }
}

void sepia(Mat& inputImg, Mat& outputImg) 
{
    for (int i = 0; i < inputImg.rows; i++) {
        for (int j = 0; j < inputImg.cols; j++) {
            Vec3b pixel = inputImg.at<Vec3b>(i, j);
            int b = pixel[0];
            int g = pixel[1];
            int r = pixel[2];
            int sepR = (int)(0.393 * b + 0.769 * g + 0.189 * r);
            int sepG = (int)(0.349 * b + 0.686 * g + 0.168 * r);
            int sepB = (int)(0.272 * b + 0.534 * g + 0.131 * r);
            outputImg.at<Vec3b>(i, j) = Vec3b(min(sepB, MAX_VALUE), min(sepG, MAX_VALUE), min(sepR, MAX_VALUE));
        }
    }
}

void negative(Mat& inputImg, Mat& outputImg) 
{
    for (int i = 0; i < inputImg.rows; i++) {
        for (int j = 0; j < inputImg.cols; j++) {
            Vec3b pixel = inputImg.at<Vec3b>(i, j);
            int b = pixel[0];
            int g = pixel[1];
            int r = pixel[2];
            outputImg.at<Vec3b>(i, j) = Vec3b(MAX_VALUE - b, MAX_VALUE - g, MAX_VALUE - r);
        }
    }
}

void contour(Mat& inputImg, Mat& outputImg) 
{
    Mat grayImg;
    cvtColor(inputImg, grayImg, COLOR_BGR2GRAY);
    Mat edges = Mat(inputImg.rows, inputImg.cols, CV_8U);

    for (int i = 1; i < grayImg.rows - 1; i++) {
        for (int j = 1; j < grayImg.cols - 1; j++) {
            float gx = grayImg.at<uchar>(i + 1, j + 1) + 2 * grayImg.at<uchar>(i, j + 1) + grayImg.at<uchar>(i - 1, j + 1) - grayImg.at<uchar>(i + 1, j - 1) - 2 * grayImg.at<uchar>(i, j - 1) - grayImg.at<uchar>(i - 1, j - 1);
            float gy = grayImg.at<uchar>(i + 1, j + 1) + 2 * grayImg.at<uchar>(i + 1, j) + grayImg.at<uchar>(i + 1, j - 1) - grayImg.at<uchar>(i - 1, j - 1) - 2 * grayImg.at<uchar>(i - 1, j) - grayImg.at<uchar>(i - 1, j + 1);
            edges.at<uchar>(i, j) = 255 - sqrt(pow(gx, 2) + pow(gy, 2));
        }
    }
    outputImg = edges.clone();
}


int main() 
{
    Mat image = imread("D:/Camera/color.jpg");
    if (!image.data)
    {
        printf("Error loading image \n"); return -1;
    }

    Mat grayscaleImg = image.clone(); 
    Mat sepiaImg = image.clone(); 
    Mat negativeImg = image.clone(); 
    Mat contourImg = image.clone();

#pragma omp parallel sections num_threads(4)
    {
#pragma omp section
        {
            grayscale(image, grayscaleImg);
        }
#pragma omp section
        {
            sepia(image, sepiaImg);
        }
#pragma omp section
        {
            negative(image, negativeImg);
        }
#pragma omp section
        {
            contour(image, contourImg);
        }
    }

    namedWindow("Input", WINDOW_NORMAL);
    imshow("Input", image);

    namedWindow("Grayscale", WINDOW_NORMAL);
    imshow("Grayscale", grayscaleImg);

    namedWindow("Sepia", WINDOW_NORMAL);
    imshow("Sepia", sepiaImg);

    namedWindow("Negative", WINDOW_NORMAL);
    imshow("Negative", negativeImg);

    namedWindow("Contour", WINDOW_NORMAL);
    imshow("Contour", contourImg);

    waitKey(0);
    destroyAllWindows();
    return 0;
}