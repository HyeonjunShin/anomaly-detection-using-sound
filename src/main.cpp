#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

// 에러 체크 매크로
#define CHECK(call) { \
    const cudaError_t cudaRet = call; \
    if (cudaRet != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " : " \
                  << cudaGetErrorString(cudaRet) << std::endl; \
        abort(); \
    } \
}

// TensorRT 로거 구현
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

// 엔진 파일 불러오기 및 역직렬화 함수 (이전과 동일)
nvinfer1::ICudaEngine* loadEngine(const std::string& enginePath, nvinfer1::IRuntime* runtime) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error opening engine file: " << enginePath << std::endl;
        return nullptr;
    }
    std::vector<char> engineData;
    file.seekg(0, std::ifstream::end);
    engineData.resize(file.tellg());
    file.seekg(0, std::ifstream::beg);
    file.read(engineData.data(), engineData.size());
    file.close();
    return runtime->deserializeCudaEngine(engineData.data(), engineData.size());
}

int main() {
    Logger gLogger;
    // 런타임 생성
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);

    // .trt 파일 경로 지정
    std::string enginePath = "./output/model.trt";
    nvinfer1::ICudaEngine* engine = loadEngine(enginePath, runtime);
    if (!engine) {
        return -1;
    }

    // 실행 컨텍스트 생성
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        return -1;
    }

    // 입출력 텐서 이름 가져오기 (모델에 맞게 수정 필요)
    // TensorRT 10에서는 num_io_tensors를 사용하여 반복하는 것이 권장됨
    const int numIOTensors = engine->getNbIOTensors();
    if (numIOTensors != 2) {
        std::cerr << "Expected exactly 2 IO tensors (input and output)." << std::endl;
        return -1;
    }

    const char* inputName = engine->getIOTensorName(0);
    const char* outputName = engine->getIOTensorName(1);

    std::cout<< inputName << ", " << outputName << std::endl;

    // 입출력 텐서 정보 가져오기 및 사이즈 계산
    nvinfer1::Dims4 inputDims = nvinfer1::Dims4{1, 1, 100, 64}; // 예시: NCHW, 배치 크기 1
    // 실제 모델의 입력 차원은 getBindingDimensions 등으로 가져와야 합니다.
    // 여기서는 임시 데이터 생성을 위해 예시 차원을 사용합니다.

    // 텐서 크기를 int64_t로 계산
    auto calcSize = [](nvinfer1::Dims dims) -> size_t {
        size_t size = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            size *= dims.d[i];
        }
        return size;
    };

    size_t inputSize = calcSize(inputDims) * sizeof(float);
    size_t outputSize = calcSize(engine->getTensorShape(outputName)) * sizeof(float); // 실제 출력 사이즈 사용

    // GPU 메모리 할당을 위해 포인터 배열 생성
    void* buffers[2]; // 0: input, 1: output
    CHECK(cudaMalloc(&buffers[0], inputSize)); // input buffer
    CHECK(cudaMalloc(&buffers[1], outputSize)); // output buffer

    // 호스트(CPU) 메모리 할당 및 임시 데이터 생성
    std::vector<float> inputData(inputSize / sizeof(float));
    std::vector<float> outputData(outputSize / sizeof(float));

    // 임시 입력 데이터 채우기 (예시)
    std::fill(inputData.begin(), inputData.end(), 1.0f);

    // CUDA 스트림 생성
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // --- 추론 실행 ---

    // 1. 호스트에서 디바이스(GPU)로 입력 데이터 복사
    CHECK(cudaMemcpyAsync(buffers[0], inputData.data(), inputSize, cudaMemcpyHostToDevice, stream));

    // 2. 입력/출력 텐서 주소 설정 (TensorRT 10의 새로운 방식)
    context->setTensorAddress(inputName, buffers[0]);
    context->setTensorAddress(outputName, buffers[1]);

    // 3. 추론 실행
    // enqueueV3 사용 (동기식 또는 비동기식 가능)
    bool success = context->enqueueV3(stream);
    if (!success) {
        std::cerr << "Inference failed." << std::endl;
        return -1;
    }

    // 4. 디바이스에서 호스트(CPU)로 출력 데이터 복사
    CHECK(cudaMemcpyAsync(outputData.data(), buffers[1], outputSize, cudaMemcpyDeviceToHost, stream));

    // 5. 스트림 동기화 (결과를 CPU에서 사용하기 전 완료 대기)
    CHECK(cudaStreamSynchronize(stream));

    // 추론 결과 사용 (outputData 벡터에 결과가 저장됨)
    std::cout << "Inference successful. First output value: " << outputData[0] << std::endl;
    std::cout << "Inference successful. First output value: " << outputData[1] << std::endl;
    std::cout << "Inference successful. First output value: " << outputData[2] << std::endl;

    // --- 리소스 해제 ---
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));
    CHECK(cudaStreamDestroy(stream));
    delete context;
    delete engine;
    delete runtime;

    return 0;
}
