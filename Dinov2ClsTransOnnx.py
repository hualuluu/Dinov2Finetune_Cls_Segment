"""
转 onnx 代码
不采用之前的 vits.__dict__['vit_small'] 来加载 model结构了,
因为MemEffAttention涉及到 xformers版本问题 导致 转onnx 有问题
所以采用基础的DinoVisionTransformer加载网络结构
"""
import torch, cv2
import onnx
import torch.nn as nn
import onnxruntime as ort
import numpy as np

# 构建模型
class Dinov2ClsModel(nn.Module):
    def __init__(self, dinov2Vits14Model, classNum):
        super().__init__()
        self.transformer = dinov2Vits14Model
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, classNum)
        )

    def forward(self, x):
        features = self.transformer.forward_features(x)
        # 提取 [CLS] token 特征（归一化后的）
        cls_token = features['x_norm_clstoken']  # 形状: [B, hidden_dim]
        x = self.classifier(cls_token)
        return x
 

def export_dinov2_to_onnx(modelpath, dummy_input, onnx_path):

    # 加载ViT-B/14模型
    
    model = torch.load(modelpath, map_location='cpu')
    model.eval()

     # 导出模型
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=True,
        input_names = ["input"],
        output_names = ["output"]
    )
    # 验证导出模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"✓ ONNX模型导出成功: {onnx_path}")
    print(f"✓ 模型输入: {[i.name for i in onnx_model.graph.input]}")
    print(f"✓ 模型输出: {[o.name for o in onnx_model.graph.output]}")
    

def dataProcess (imagepath, inputW, inputH):
    PIXEL_MEANS = (0.485, 0.456, 0.406)    # RGB  format mean and variances
    PIXEL_STDS = (0.229, 0.224, 0.225)

    image = cv2.imread(imagepath)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 先试试简单的 resize 
    image = cv2.resize(imageRGB, (inputW, inputH), interpolation=cv2.INTER_LINEAR)
        
    # 4. 转换为 float32 并缩放到 [0, 1]（等效 ToTensor）
    img_float = image.astype(np.float32) / 255.0

    # 5. 归一化 (Normalize)
    # 将 mean 和 std 转换为 numpy 数组并 reshape 以适配广播
    mean_arr = np.array(PIXEL_MEANS, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(PIXEL_STDS, dtype=np.float32).reshape(1, 1, 3)
    img_norm = (img_float - mean_arr) / std_arr

    # 6. 将 HWC 转换为 CHW（PyTorch 张量格式）
    img_chw = np.transpose(img_norm, (2, 0, 1))   
    img = np.expand_dims(img_chw, axis=0)             # [1, 3, 518, 518]

    return img
    


def predictOnnx(onnx_path, imagepath, inputW, inputH):

    inputData = dataProcess (imagepath, inputW, inputH)
    
    print(inputData)
    # 加载模型
    sess = ort.InferenceSession(onnx_path)

    # 获取输入信息
    for inp in sess.get_inputs():
        print(f"Input name: {inp.name}, shape: {inp.shape}, type: {inp.type}")

    # 获取输出信息
    for out in sess.get_outputs():
        print(f"Output name: {out.name}, shape: {out.shape}, type: {out.type}")

    # 运行推理（输出名称列表可以只取需要的，或传入 None 获取所有输出）
    outputs = sess.run(["output"], {"input": inputData})

    # outputs 是一个列表，按输出名称顺序对应
    clss = outputs[0]   # 形状 [1, num_classes]（分类）或 [1, num_classes, H, W]（分割）
    print(clss)
    pred_class = np.argmax(clss, axis=1).item()
    print(f"Predicted class: {pred_class}")


if __name__ == "__main__":
    # 转onnx 
    # 创建示例输入
    dummy_input = torch.randn(1, 3, 224, 224)  # DINOv2标准输入尺寸

    modelpath = "best_cls_finetune.pth"
    # 执行导出
    onnx_path = "best_cls_finetune.onnx"
    #onnx_model = export_dinov2_to_onnx(modelpath, dummy_input, onnx_path)

    imagepath ="/home/hualulu/code/dinov2/data/battery/val/calss_3/0366AD2509H03D27.bmp"
    predictOnnx(onnx_path, imagepath, 224, 224)
