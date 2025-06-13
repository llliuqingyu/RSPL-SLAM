#include "rcf.h"


RCF::RCF()
{
    setReportableSeverity(tensorrt_log::Severity::kINTERNAL_ERROR);
}

bool RCF::build()
{
    auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) {
        std::cerr << "Failed to create IBuilder" << std::endl;
        return false;
    }

    const auto explicit_Batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_Batch));
    if (!network) {
        std::cerr << "Failed to create INetworkDefinition" << std::endl;
        return false;
    }

    auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    parser->parseFromFile(onnx_path_.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    if (!parser) {
        std::cerr << "Failed to create IParser" << std::endl;
        return false;
    }

    auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        std::cerr << "Failed to create IBuilderConfig" << std::endl;
        return false;
    }

    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        std::cerr << "Failed to create IOptimizationProfile" << std::endl;
        return false;
    }
    profile->setDimensions(onnx_input_name_.c_str(),
                           OptProfileSelector::kMIN, Dims4(1, 3, 400, 400));
    profile->setDimensions(onnx_input_name_.c_str(),
                           OptProfileSelector::kOPT, Dims4(1, 3, 1000, 1000));
    profile->setDimensions(onnx_input_name_.c_str(),
                           OptProfileSelector::kMAX, Dims4(1, 3, 2000, 2000));
    config->addOptimizationProfile(profile);

    config->setMaxWorkspaceSize(1_GiB);
    // config->setFlag(nvinfer1::BuilderFlag::kFP16);

    auto profile_stream = makeCudaStream();
    if (!profile_stream) {
        std::cerr << "Failed to create CudaStream" << std::endl;
    }
    config->setProfileStream(*profile_stream);

    TensorRTUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        std::cerr << "Failed to build serialized network" << std::endl;
        return false;
    }

    TensorRTUniquePtr<IRuntime> runtime{createInferRuntime(gLogger.getTRTLogger())};
    if (!runtime) {
        std::cerr << "Failed to create IRuntime" << std::endl;
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>{runtime->deserializeCudaEngine(plan->data(), plan->size(), nullptr)};
    if (!engine_) {
        std::cerr << "Failed to deserialize ICudaEngine" << std::endl;
        return false;
    }
    if (engine_ != nullptr) {
        nvinfer1::IHostMemory *data = engine_->serialize();
        std::ofstream file(engine_path_, std::ios::binary);
        if (!file){
            std::cerr << "Failed to write engine file" << std::endl;
        }
        file.write(reinterpret_cast<const char *>(data->data()), data->size());
        return true;
    }
}

std::vector<u_int> RCF::infer(const cv::Mat &input_image)
{
    if (!context_) {
        context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        assert(context_ != nullptr && "context_ is null!");
    }
    const int input_index = engine_->getBindingIndex(onnx_input_name_.c_str());
    context_->setBindingDimensions(input_index, Dims4(1, 3, input_image.rows, input_image.cols));
    // std::cout << "input_index: " << input_index << std::endl;

    // const int out_230 = engine_->getBindingIndex(output_230_.c_str());
    // std::cout << "out_230 index: " << out_230 << std::endl;

    BufferManager buffers(engine_, 0, context_.get());


    if(!process_input_1(input_image, buffers)) {
        std::cerr << "Failed to process input with 1 channel" << std::endl;
        assert(false && "Failed to process input with 1 channel");
    }
    // if(input_image.channels() == 1) {
    //     if(!process_input_1(input_image, buffers)) {
    //         std::cerr << "Failed to process input with 1 channel" << std::endl;
    //         assert(false && "Failed to process input with 1 channel");
    //     }
    // } else if(input_image.channels() == 3) {
    //     if(!process_input_3(input_image, buffers)) {
    //         std::cerr << "Failed to process input with 3 channels" << std::endl;
    //         assert(false && "Failed to process input with 3 channels");
    //     }
    // } else {
    //     std::cerr << "Unsupported number of channels: " << input_image.channels() << std::endl;
    //     assert(false && "Unsupported number of channels");
    // }

    buffers.copyInputToDevice();

    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    assert(status && "Failed to execute inference");

    buffers.copyOutputToHost();

    std::vector<u_int> result_vector = process_output(buffers, input_image);


    // printBindingInfo(engine_.get());


    return result_vector;
}

std::vector<u_int> RCF::process_output(const BufferManager &buffers, const cv::Mat &input_image) {
    float* infer_output = static_cast<float *>(buffers.getHostBuffer("230"));

    assert(infer_output != nullptr && "output_img is null!");

    std::vector<float> output_vector(input_image.rows * input_image.cols);
    for (int i = 0; i < input_image.rows * input_image.cols; ++i) {
        output_vector[i] = infer_output[i];
    }

    std::vector<u_int> output_img(input_image.rows * input_image.cols, 0);

    for (int i = 0; i < output_vector.size(); ++i) {
        u_int value = static_cast<u_int>(255-output_vector[i]*255);
        output_img[i] = value;
    }

    // std::cout << "image rows: " << input_image.rows << ",  " << "image cols: " << input_image.cols << std::endl;
    // std::cout << "output_vector size: " << output_vector.size() << std::endl;
    // std::cout << infer_output << std::endl;
    // std::cout << "Finish rcf process output!" << std::endl;
    return output_img;
}

bool RCF::process_input_1(const cv::Mat &image, BufferManager &buffers) {
    float* host_data_buffer = static_cast<float *>(buffers.getHostBuffer(onnx_input_name_));
    // if(image.channels() != 1) {
    //     std::cerr << "You use function process_input_1, but the image channls is not 1!" << std::endl;
    //     return false;
    // }

    int h = image.rows;
    int w = image.cols;

    cv::Mat float_img;
    image.convertTo(float_img, CV_32FC1);  // 归一化

    float* trt_input = new float[1 * 3 * h * w];  // Dims4{1, 3, h, w}

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float pixel = float_img.at<float>(y, x);
            trt_input[0 * h * w + y * w + x] = pixel;
            trt_input[1 * h * w + y * w + x] = pixel;
            trt_input[2 * h * w + y * w + x] = pixel;
        }
    }

    for(int i = 0; i < 3 * h * w; ++i) {
        host_data_buffer[i] = trt_input[i]; // 将数据复制到 TensorRT 输入缓冲区
    }
    delete[] trt_input;

    // std::cout << "image channels: " << image.channels() << std::endl;
    return true;
}

bool RCF::process_input_3(const cv::Mat &image, BufferManager &buffers) {
    float* host_data_buffer = static_cast<float *>(buffers.getHostBuffer(onnx_input_name_));
    if(image.channels() != 3) {
        std::cerr << "You use function process_input_3, but the image channls is not 3!" << std::endl;
        return false;
    }
    std::cout << "image channels: " << image.channels() << std::endl;
    int height = image.rows;
    int width = image.cols;
    int channel_size = height * width;
    for (int c = 0; c < 3; ++c) {
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                uchar pixel_value = image.at<cv::Vec3b>(row, col)[c];
                int index = c * channel_size + row * image.cols + col;
                host_data_buffer[index] = static_cast<float>(pixel_value) / 255.0f;
            }
        }
    }
    return true;
}


bool RCF::deserialize_engine(){
    std::ifstream file(engine_path_.c_str(), std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, std::ifstream::end);
        size_t size = file.tellg();
        file.seekg(0, std::ifstream::beg);
        char *model_stream = new char[size];
        file.read(model_stream, size);
        file.close();
        IRuntime *runtime = createInferRuntime(gLogger);
        if (runtime == nullptr) {
            delete[] model_stream;
            return false;
        }
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size));
        delete[] model_stream;
        if (engine_ == nullptr) return false;
        return true;
    }
    return false;
}

void RCF::printBindingInfo(nvinfer1::ICudaEngine* engine) {
    const int num_bindings = engine->getNbBindings();
    for (int i = 0; i < num_bindings; ++i) {
        const char *binding_name = engine->getBindingName(i);
        if (binding_name == nullptr) {
            std::cerr << "Binding name is null for index " << i << std::endl;
            continue;
        }
        std::cout << "Binding name: " << binding_name << std::endl;
        int index = engine->getBindingIndex(binding_name);
        std::cout << "Binding index: " << index << std::endl;
        std::cout << "Binding dimensions: " << engine->getBindingDimensions(index) << std::endl;
        std::cout << "-------------------" << std::endl;
    }
}