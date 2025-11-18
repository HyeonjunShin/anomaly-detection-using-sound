from model import FFT2DCNN
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    model = FFT2DCNN(n_classes=3).to(device=device)
    params = torch.load("./output/model.prm", map_location=device)
    model.load_state_dict(params)
    model.eval()

    dummyData = torch.randn(1, 100, 64, device=device)  # batch=32, time=100, freq=64
    torch.onnx.export(
        model=model,
        args=dummyData,
        f="./output/model.onnx",
        input_names=["input"],
        output_names=["output"]
    )


if __name__ == "__main__":
    main()

# trtexec --onnx=output/model.onnx --saveEngine=output/model.trt --fp16