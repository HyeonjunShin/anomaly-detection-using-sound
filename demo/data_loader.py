import torch
from torch.utils.data import DataLoader, TensorDataset
import time

# 임의의 큰 데이터셋
X = torch.randn(50000, 200, 32)
y = torch.randint(0, 2, (50000,))
dataset = TensorDataset(X, y)

def measure_loader_time(pin_memory):
    loader = DataLoader(dataset, batch_size=256, shuffle=True,
                        num_workers=4, pin_memory=pin_memory)
    start = time.time()
    for xb, yb in loader:
        xb = xb.to("cuda", non_blocking=True)
        yb = yb.to("cuda", non_blocking=True)
        _ = xb * 2  # dummy operation
    torch.cuda.synchronize()
    return time.time() - start

t1 = measure_loader_time(pin_memory=False)
t2 = measure_loader_time(pin_memory=True)

print(f"Without pin_memory: {t1:.3f} sec")
print(f"With pin_memory:    {t2:.3f} sec")
print(f"Speedup: {(t1/t2):.2f}x faster")
