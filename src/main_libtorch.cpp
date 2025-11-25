#include <ATen/ops/abs.h>
#include <ATen/ops/argmax.h>
#include <ATen/ops/log1p.h>
#include <cmath>
#include <cstddef>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <torch/types.h>

struct DataShapes {
    std::vector<int> train;
    std::vector<int> val;
    std::vector<int> test;
};

struct DataTypeInfo {
    std::string data;
    std::string label;
};

struct Metadata {
    int windowSize;
    int hopSize;
    DataShapes shape;
    DataTypeInfo dtype;
    int itemSize;
    std::vector<std::string> proc;
    std::vector<double> mean;
    std::vector<double> std;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DataShapes, train, val, test)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DataTypeInfo, data, label)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Metadata, windowSize, hopSize, shape, dtype, itemSize, proc, mean, std)

Metadata load_metadata(const std::string& json_path) {
    std::ifstream i(json_path);
    if (!i.is_open()) {
        std::cerr << "메타데이터 파일을 열 수 없습니다: " << json_path << std::endl;
        exit(EXIT_FAILURE);
    }
    
    nlohmann::json j;
    i >> j;

    Metadata meta = j.get<Metadata>();
    
    return meta;
}


int main(int argc, const char* argv[]) {
    // std::string modelFile = argv[1];
    // std::string metadataFile = argv[2];
    // std::string dataFile = argv[3];
    // Read metadata.
    Metadata metaData = load_metadata("./data/train3/metadata.json");

    int D1 = metaData.shape.test[0];
    int D2 = metaData.shape.test[1];
    int D3 = metaData.shape.test[2];
    int itemSize = metaData.itemSize;

    size_t dataCount = D1 * D2 * D3;
    size_t labelCount = D1;

    size_t dataBytes = dataCount * itemSize;
    size_t labelBytes = labelCount * itemSize;

    // Read bin file.
    std::ifstream file("./data/train3/test.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open test.bin\n";
        return 1;
    }
    std::vector<double> dataBuf(dataCount);
    std::vector<double> labelBuf(labelCount);
    file.read(reinterpret_cast<char*>(dataBuf.data()), dataBytes);
    file.read(reinterpret_cast<char*>(labelBuf.data()), labelBytes);
    file.close();

    for(size_t i=0; i<metaData.mean.size(); ++i){
        std::cout << metaData.mean[i] << ", ";
    }
    std::cout << std::endl;

    torch::Tensor meanTensor = torch::from_blob(
    metaData.mean.data(), {64}, torch::kFloat64
    ).clone().reshape({1, 1, 64}).to(torch::kFloat32); 

    torch::Tensor stdTensor = torch::from_blob(
    metaData.std.data(), {64}, torch::kFloat64
    ).clone().reshape({1, 1, 64}).to(torch::kFloat32); 


    std::cout << meanTensor << std::endl;
    std::cout << stdTensor << std::endl;

    torch::Tensor dataTensor = torch::from_blob(
        dataBuf.data(), {D1, D2, D3}, torch::kInt64
    ).clone().to(torch::kFloat32);
    torch::Tensor labelTensor = torch::from_blob(
        labelBuf.data(), {D1}, torch::kInt64
    ).clone().to(torch::kFloat32);

    dataTensor = torch::log1p(torch::abs(dataTensor));
    dataTensor = (dataTensor - meanTensor) / stdTensor;

    // std::cout << dataTensor.sizes() << std::endl;
    // std::cout << dataTensor.index({0, 0, 0}).item<double>() << std::endl;
    // std::cout << labelTensor.index({0}).item<double>() << std::endl;
    std::cout << dataTensor[0][0] << std::endl;

    // if (argc != 2) {
    //     std::cerr << "Uesage: ./inference_app <model.pt>\n";
    //     return -1;
    // }

    torch::jit::script::Module module;
    try {
        // torch::jit::load 함수를 사용하여 모델을 역직렬화합니다.
        // module = torch::jit::load(argv[1]);
        module = torch::jit::load("./output/model.pt");
        std::cout << "Success load model\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    module.eval();

    auto device = torch::kCPU;
    dataTensor = dataTensor.to(device);
    module.to(device);
    // torch::Tensor input_tensor = torch::rand({3, 100, 64});

    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(input_tensor);

    torch::Tensor output = module.forward({dataTensor}).toTensor();
    output = torch::argmax(output, -1);
    std::cout << "Output tensor shape: " << output.sizes() << std::endl;
    torch::Tensor correct = (output == labelTensor);
    std::cout << correct.sum() << std::endl;

    // for (int i=0; i<output.sizes()[0]; ++i){
    //     std::cout << "Output: " << output[i][0].item<float>() << ", " << output[i][1].item<float>() << ", " << output[i][2].item<float>() << "\n";
    //     // std::cout << "첫 번째 출력 값: " << output[i][1].item<float>() << std::endl;
    //     // std::cout << "첫 번째 출력 값: " << output[i][2].item<float>() << std::endl;
    // }

    return 0;
}