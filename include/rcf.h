// #include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>
// #include <logger.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <NvInferPlugin.h>


#include "Thirdparty/TensorRTBuffer/include/buffers.h"

using namespace tensorrt_buffer;
using namespace tensorrt_common;
using namespace tensorrt_log;


class RCF{
public:
    RCF();
    // ~RCF();
    bool build();
    std::vector<u_int> infer(const cv::Mat& input_image);
    bool deserialize_engine();
    bool process_input_1(const cv::Mat &input_image, BufferManager &buffers);
    bool process_input_3(const cv::Mat &input_image, BufferManager &buffers);
    // cv::Mat process_output(const BufferManager &buffers, const cv::Mat &input_image);
    std::vector<u_int> process_output(const BufferManager &buffers, const cv::Mat &input_image);
    void printBindingInfo(nvinfer1::ICudaEngine* engine);

private:
    std::string onnx_input_name_ = "rcf_input";// input size: 1,3,dynamic,dynamic
    std::string onnx_output_name_ = "rcf_output";
    std::string output_226_ = "226";
    std::string output_227_ = "227";
    std::string output_228_ = "228";
    std::string output_229_ = "229";
    std::string output_230_ = "230";


    std::string onnx_path_ = "/home/lqy/airvo_for_github/src/AirVO/output/rcf_onnx.onnx";
    std::string engine_path_ = "/home/lqy/airvo_for_github/src/AirVO/output/rcf.engine";
    // std::string engine_path_ = "/home/lqy/rcf_onnx_tensorrt/build/rcf_fp16.engine";

    // std::string engine_path_ = "rcf.engine";

    // cv::Mat concat_results_ = cv::Mat::zeros(200, 200, CV_8UC1); // Initialize with zeros

    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    // nvinfer1::Dims rcf_input_{};
    // nvinfer1::Dims rcf_output_{};
};
typedef std::shared_ptr<RCF> RCFPtr;