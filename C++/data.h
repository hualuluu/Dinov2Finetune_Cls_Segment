#pragma once
#include <opencv2/opencv.hpp>
cv::Mat letterbox(cv::Mat image, cv::Size s, int borderThre, bool center = true);



cv::Mat letterboxOut(cv::Mat mask, cv::Size imageSize, cv::Size inputSize, bool center = true);