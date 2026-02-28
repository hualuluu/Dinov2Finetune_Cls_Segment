#pragma once
#include <fstream>
#include "NvInfer.h"
#include "NvOnnxParser.h"

inline const char* severity_string(nvinfer1::ILogger::Severity t);

class TRTLogger : public nvinfer1::ILogger
{
public:
	virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
		if (severity <= Severity::kWARNING) {
			if (severity == Severity::kWARNING) printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
			else if (severity == Severity::kERROR) printf("\031[33m%s: %s\033[0m\n", severity_string(severity), msg);
			else printf("%s: %s\n", severity_string(severity), msg);
		}
	}
};


bool isFileExists(const char* filename);



bool buildModel(const char* onnxPath, const char* enginePath);



// ******************************¼ÓÔØÄ£ÐÍ*****************************
std::vector<unsigned char> load_file(const std::string& file);