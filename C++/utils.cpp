#include "utils.h"



bool isFileExists(const char* filename)
{
	std::ifstream f(filename);
	return f.good();
}



inline const char* severity_string(nvinfer1::ILogger::Severity t) {
	switch (t) {
	case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
	case nvinfer1::ILogger::Severity::kERROR: return "error";
	case nvinfer1::ILogger::Severity::kWARNING: return "warning";
	case nvinfer1::ILogger::Severity::kINFO: return "info";
	case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
	default: return "unknown";
	}
}




bool buildModel(const char* onnxPath, const char* enginePath)
{

	TRTLogger logger;

	// 下面的builder, config, network是基本需要的组件
	// 形象的理解是你需要一个builder去build这个网络，网络自身有结构，这个结构可以有不同的配置
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
	// 创建一个构建配置，指定TensorRT应该如何优化模型，tensorRT生成的模型只能在特定配置下运行
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// 创建网络定义，其中createNetworkV2(1)表示采用显性batch size，新版tensorRT(>=7.0)时，不建议采用0非显性batch size
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

	// onnx parser解析器来解析onnx模型
	auto parser = nvonnxparser::createParser(*network, logger);
	if (!parser->parseFromFile(onnxPath, 1)) {
		printf("Failed to parse classifier.onnx.\n");
		return false;
	}

	// 设置工作区大小
	printf("Workspace Size = %.2f MB\n", (1 << 30) / 1024.0f / 1024.0f);
	config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);

	// Float 
	// config->setFlag(nvinfer1::BuilderFlag::kFP16);

	// 需要通过profile来使得batchsize时动态可变的，这与我们之前导出onnx指定的动态batchsize是对应的
	int maxBatchSize = 1;
	auto profile = builder->createOptimizationProfile();
	auto input_tensor = network->getInput(0);
	auto input_dims = input_tensor->getDimensions();

	// 设置batchsize的最大/最小/最优值
	input_dims.d[0] = 1;
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);

	input_dims.d[0] = maxBatchSize;
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
	config->addOptimizationProfile(profile);

	// 直接构建序列化模型（替换 buildEngineWithConfig + serialize）
	nvinfer1::IHostMemory* model_data = builder->buildSerializedNetwork(*network, *config);
	if (model_data == nullptr) {
		printf("Build engine failed.\n");
		return false;
	}

	// 将序列化数据写入文件
	FILE* f = fopen(enginePath, "wb");
	fwrite(model_data->data(), 1, model_data->size(), f);
	fclose(f);

	// 逆序destory掉指针
	delete   model_data;
	delete  network;
	delete  config;
	delete  builder;

	printf("Build Done.\n");
	return true;
}



// ******************************加载模型*****************************
std::vector<unsigned char> load_file(const std::string& file)
{
	std::ifstream in(file, std::ios::in | std::ios::binary);
	if (!in.is_open()) return {};

	in.seekg(0, std::ios::end);
	size_t length = in.tellg();

	std::vector<uint8_t> data;
	if (length > 0) {
		in.seekg(0, std::ios::beg);
		data.resize(length);

		in.read((char*)&data[0], length);
	}
	in.close();
	return data;
}

