import torch
import torchvision
import torchvision.ops as ops

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("PyTorch rodando na GPU:", torch.tensor([1.0]).cuda().device)

if torch.cuda.is_available():
    print("GPU Atual:", torch.cuda.current_device())
    print("Device", torch.cuda.device(0))
    print("Device Name", torch.cuda.get_device_name(0))
else:
    print("Nenhuma GPU configurada")



boxes = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60], [200, 200, 300, 300]], dtype=torch.float32).cuda()
scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32).cuda()
iou_threshold = 0.5

try:
    result = ops.nms(boxes, scores, iou_threshold)
    print("NMS funcionou corretamente na GPU:", result)
except Exception as e:
    print("Erro ao rodar NMS:", e)