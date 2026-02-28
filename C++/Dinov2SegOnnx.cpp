#include <opencv2/opencv.hpp>
#include "data.h"


cv::Mat dataProcessSegOnnx(std::string imagepath, int inputW, int inputH)
{
	cv::Mat image = cv::imread(imagepath);
	cv::Mat imagePad = letterbox(image, cv::Size(inputW, inputH), 114); // 获得pad的图像


	cv::Mat imageRGB, imageResize, imageResizeFloat, imageTrans;
	cv::cvtColor(imagePad, imageRGB, cv::COLOR_BGR2RGB);
	cv::resize(imageRGB, imageResize, cv::Size(inputW, inputH));
	imageResize.convertTo(imageResizeFloat, CV_32FC3, 1.0 / 255.0);

	cv::Scalar mean = cv::Scalar(0.485, 0.456, 0.406);
	cv::Scalar std = cv::Scalar(0.229, 0.224, 0.225);

	// IMAGE - MEAN / STD
	imageResizeFloat = imageResizeFloat - mean;// mean
	// std
	std::vector<cv::Mat> bgrChannels(3);
	cv::split(imageResizeFloat, bgrChannels);
	for (int i = 0; i < 3; i++)
	{
		bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0 / std[i]);
	}

	cv::merge(bgrChannels, imageTrans);

	return imageTrans;
}



int  mainSegOnnx()
{
	// data 
	int inputW = 224, inputH = 224, patchSize = 14, classNum = 2;

	std::string imagepath = "D:/study/Dinov2/data/1_bubble_9.jpg";
	cv::Mat image = cv::imread(imagepath);
	cv::Mat inputImageFloat = dataProcessSegOnnx(imagepath, inputW, inputH);
	cv::Mat _input = cv::dnn::blobFromImage(inputImageFloat);

	// model 
	std::string modelpath = "D:/study/Dinov2/best_seg_finetune_224x224.onnx";
	cv::dnn::Net net = cv::dnn::readNetFromONNX(modelpath);

	// inference	
	net.setInput(_input);
	std::vector<cv::Mat> output_probs;
	std::vector<cv::String> output_layer_names = net.getUnconnectedOutLayersNames();// 获取多输出对应的名称
	net.forward(output_probs, output_layer_names);

	// 3. process outputs
	if (output_probs.size() <= 0)
		return 0;

	float* _output_data_host = (float*)output_probs[0].data; // 1 * * 224 * 224

	
	cv::Mat preds = cv::Mat::zeros(cv::Size(inputW, inputH), CV_8UC1);

	for (int iw = 0; iw < inputW; iw++)
	{
		for (int ih = 0; ih < inputH; ih++)
		{
			int max_id = 0;
			float max_val = _output_data_host[ ih * inputW + iw];

			for (int i = 1; i < classNum; i++)
			{
				float val = _output_data_host[inputH * inputW * i + ih * inputW + iw];

				if (val > max_val)
				{
					max_val = val;
					max_id = i;
				}
			}
			preds.at<uchar>(ih, iw) = max_id * 255;

		}
	}
	
	cv::Mat mask = letterboxOut(preds, image.size(), cv::Size(inputW, inputH));

	return 1;
}





