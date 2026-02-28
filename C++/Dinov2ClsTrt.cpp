#include <opencv2/opencv.hpp>
#include <fstream>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "utils.h"



std::vector<float> dataProcessTrt(std::string imagepath, int inputW, int inputH)
{
	cv::Mat image = cv::imread(imagepath);
	cv::Mat imageRGB, imageResize, imageResizeFloat, imageTrans;
	cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
	cv::resize(imageRGB, imageResize, cv::Size(inputW, inputH));
	imageResize.convertTo(imageResizeFloat, CV_32FC3, 1.0 / 255.0);

	cv::Scalar mean = cv::Scalar(0.485, 0.456, 0.406);
	cv::Scalar std = cv::Scalar(0.229, 0.224, 0.225);

	// IMAGE - MEAN / STD
	imageResizeFloat = imageResizeFloat - mean;// mean

	// 分配 CHW 缓冲区
	std::vector<float> chw(3 * inputW * inputH);
	float* hwc_ptr = (float*)imageResizeFloat.data;
	for (int i = 0; i < inputW * inputH; ++i) {
		for (int c = 0; c < 3; ++c) {
			chw[c * inputW * inputH + i] = hwc_ptr[i * 3 + c] / std[c];
		}
	}
	return chw;
}



void mainClsTrt() 
{
	// data 
	int inputW = 224, inputH = 224;
	int classNum = 5;
	std::string imagepath = "D:/study/Dinov2/data/0366AD2509H03D27.bmp";
	std::string onnxPath = "D:/study/Dinov2/best_cls_finetune_224x224.onnx";
	std::string enginePath = "D:/study/Dinov2/best_cls_finetune_224x224.engine";
	
	// build
	if (!isFileExists(enginePath.c_str()))
	{
		std::cout << "Engine file not exists, building engine..." << std::endl;
		buildModel(onnxPath.c_str(), enginePath.c_str());
	}

	// load model
	TRTLogger logger;

	std::vector<unsigned char> engine_data = load_file(enginePath);
	nvinfer1::IRuntime*  _runtime = nvinfer1::createInferRuntime(logger);
	nvinfer1::ICudaEngine*  _engine = (_runtime)->deserializeCudaEngine(engine_data.data(), engine_data.size());
	if (_engine == nullptr) {
		printf("Deserialize cuda engine failed.\n");
		delete  (_runtime);
		return ;
	}
	int nbIOTensors = _engine->getNbIOTensors();
	if (nbIOTensors != 2 && nbIOTensors != 3) {
		printf("Must be single input, single or two Output, got %d output.\n", nbIOTensors - 1);
		return;
	}

	nvinfer1::IExecutionContext*  _execution_context = _engine->createExecutionContext();
	cudaStream_t _stream = nullptr;
	cudaStreamCreate(&_stream);

	// memory
	float* _output_data_host = nullptr;
	float* _input_data_device = nullptr, * _output_data_device = nullptr;
	long _inputLength = 1 * 3 * inputW * inputH;
	long _outputLength = 1 * classNum;

	cudaMallocHost((void**)&_output_data_host, _outputLength * sizeof(float));
	cudaMalloc((void**)&_input_data_device, _inputLength * sizeof(float));
	cudaMalloc((void**)&_output_data_device, _outputLength * sizeof(float));
	

	// 1. data process 
	std::vector<float>  inputdata = dataProcessTrt(imagepath, inputW, inputH);
	
	// cpu->gpu
	cudaMemcpyAsync(_input_data_device, inputdata.data(), _inputLength * sizeof(float), cudaMemcpyHostToDevice, _stream);

	// 2. runNet
	// 2. 设置输入输出张量地址（TensorRT 10 需要提前绑定）
	const char* input_name = _engine->getIOTensorName(0);      // 假设第一个是输入
	const char* output_name = _engine->getIOTensorName(1);     // 第二个是输出
	_execution_context->setInputShape(input_name, nvinfer1::Dims4{ 1, 3, inputH, inputW });
	_execution_context->setTensorAddress(input_name, _input_data_device);
	_execution_context->setTensorAddress(output_name, _output_data_device);

	bool success = _execution_context->enqueueV3(_stream);
	cudaStreamSynchronize(_stream); // 同步
	// gpu -> cpu
	
	cudaMemcpyAsync(_output_data_host, _output_data_device, _outputLength * sizeof(float), cudaMemcpyDeviceToHost, _stream);
	
	int clss = -1; float max_score = 0.0;
	for (int i = 0; i < classNum; i++)
	{
		if (max_score < _output_data_host[i])
		{
			max_score = _output_data_host[i];
			clss = i;
		}
		std::cout << _output_data_host[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "image's class is: " << clss << std::endl;

	// 释放资源
	cudaStreamDestroy(_stream);
	cudaFree(_input_data_device);
	cudaFree(_output_data_device);
	cudaFreeHost(_output_data_host);
	delete  _execution_context;
	delete _engine;
	delete _runtime;

	return;
}