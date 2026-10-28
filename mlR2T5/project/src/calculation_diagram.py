import onnxscript
import torch
from models.Mymodel import SPANet

model = SPANet(num_sam=4) 

model.eval()

# 创建一个与模型输入一致的示例输入张量 (dummy input)

dummy_input = torch.randn(1, 3, 256, 256)

output_names = ["clean"] + [f"attn_{i}" for i in range(4)]

# 导出模型
batch_dim = torch.export.Dim("batch_size", min=1) 

torch.onnx.export(
    model,                           # 要导出的模型
    dummy_input,                     # 示例输入
    "project/src/models/calculation_map/spanet.onnx", # 输出的 ONNX 文件名
    input_names=["input"],           # 为输入命名，便于理解
    output_names=output_names,       # 为输出命名（如果你的模型返回tuple，这里可能需要处理）
    opset_version=11,                # ONNX 算子集版本，11或12比较常用
    dynamic_shapes=
    {"x": {0: batch_dim}},           # 使用dynamic_shapes
    dynamo=True,                     # 显式启用新版导出器（PyTorch 2.9+ 默认即为 True）
 
)
print("模型已成功导出为 spanet.onnx")