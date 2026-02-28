"""
不采用之前的 vits.__dict__['vit_small'] 来加载 model结构了,
因为MemEffAttention涉及到 xformers版本问题 导致 转onnx 有问题
所以采用基础的DinoVisionTransformer加载网络结构
"""
import torch, cv2
import onnx
import torch.nn as nn
import onnxruntime as ort
import numpy as np

class SimpleSegHead(nn.Module):

    def __init__(self, in_dim, num_classes, patchSize = 14):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, num_classes, kernel_size=1) # 1 * 1 卷积 改变维度 
        self.upsample = nn.Upsample(scale_factor=patchSize, mode='bilinear', align_corners=False)

    def forward(self, feat_map):
        logits = self.conv(feat_map)          # [B, num_classes, Hp, Wp]
        logits = self.upsample(logits)        # [B, num_classes, H*14, W*14]
        return logits


class ASPPSegHead(nn.Module):
    def __init__(self, in_dim, num_classes, featureDim = 1369, 
        patchSize = 14, hidden_dim=256, atrous_rates=(6, 12, 18)):

        super().__init__()
        self.aspp = nn.ModuleList()
        # 1. 1 * 1 卷积 改变维度
        self.aspp.append(nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        ))

        # 2. 空洞卷积分支
        for rate in atrous_rates:
            self.aspp.append(nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ))

        # 3. 全局平均池化
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

        self.global_avg_pool_upsample = nn.Upsample(size=[featureDim, featureDim], mode='bilinear', align_corners=False)
        
        # 4. 分割 head
        self.seg_fea = nn.Sequential(
            nn.Conv2d(hidden_dim * (len(atrous_rates)+2), hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(hidden_dim, num_classes, 1)
        )
        
        # 5. upsample
        self.upsample = nn.Upsample(scale_factor=patchSize, mode='bilinear', align_corners=False)

    def forward(self, x):
        # x: [B, dim, Hp, Wp]
        aspp_out = []
        # aspp = [conv1*1, atrous6, atrous 8, atrous 12, avg pool + upsample]
        for branch in self.aspp:
            aspp_out.append(branch(x)) # conv1*1, atrous6, atrous 8, atrous 12
            
        # avg pool + upsample
        global_feat = self.global_avg_pool(x)
        global_feat = self.global_avg_pool_upsample(global_feat)
            
        aspp_out.append(global_feat)

        # concat aspp out
        x = torch.cat(aspp_out, dim=1)
        x = self.seg_fea(x)
        x = self.upsample(x)

        return x



# 构建模型
class Dinov2SegModel(nn.Module):
    def __init__(self, dinov2Vits14Model, classNum, num_patches=37, head_type='simple'):
        super().__init__()
        self.transformer = dinov2Vits14Model
        hidden_dim = self.transformer.embed_dim
        patchSize = self.transformer.patch_size

        if head_type == 'simple':
            self.segment = SimpleSegHead(hidden_dim, classNum, patchSize)
        if head_type == 'aspp':
            self.segment = ASPPSegHead(hidden_dim, classNum, num_patches, patchSize)

    def forward(self, x):
        B, C, H, W = x.shape

        features = self.transformer.forward_features(x)
        # 提取 [patch] token 特征（归一化后的）
        patch_tokens = features['x_norm_patchtokens']  # 形状: [B, hidden_dim]
        
        # 重塑为特征图
        Hp = Wp = int(patch_tokens.shape[1] ** 0.5) # 根号 获得特征尺寸
        feat_map = patch_tokens.permute(0, 2, 1).reshape(B, -1, Hp, Wp)# reshape feature map 

        # 分割头
        masks = self.segment(feat_map)                   # [B, num_classes, H, W]

        return masks
    


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
    


def letterbbox(image, new_shape=[224, 224], patchSize=14, fill_value = 114, auto=False, 
                scale_fill=False, center=True):
    
    shape = image.shape[:2]
    # 图像的尺寸
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # 计算需要填充的边界
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, patchSize), np.mod(dh, patchSize)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    
    if center:
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        
    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        if image.ndim == 2:
            image = image[..., None]

    top, bottom = int(round(dh - 0.1)) if center else 0, int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)) if center else 0, int(round(dw + 0.1))
    h, w, c = image.shape
    
    if c == 3:
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(fill_value, fill_value, fill_value))
    else:  # multispectral
        pad_img = np.full((h + top + bottom, w + left + right, c), fill_value=fill_value, dtype=image.dtype)
        pad_img[top : top + h, left : left + w] = image
        image = pad_img

    return image



def dataProcess (imagepath, inputW, inputH, patchSize = 14):
    PIXEL_MEANS = (0.485, 0.456, 0.406)    # RGB  format mean and variances
    PIXEL_STDS = (0.229, 0.224, 0.225)

    image = cv2.imread(imagepath)
    imagePad = letterbbox(image, [inputW, inputH], patchSize) # 获得pad的图像
    imageRGB = cv2.cvtColor(imagePad, cv2.COLOR_BGR2RGB) # RGB

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
    print(outputs)
    # outputs 是一个列表，按输出名称顺序对应
    preds = outputs[0]   # 形状 [1, num_classes]（分类）或 [1, num_classes, H, W]（分割）
    mask_np = np.argmax(preds, axis=1)        # [B, H, W]
    # 2. 如果张量是 CHW，转换为 HWC
    mask_np = np.transpose(mask_np, (1, 2, 0))  # [H, W, C]
    # 3. 如果张量值在 [0,1] 范围，转换为 [0,255] 并转为 uint8
    mask_np = (mask_np * 255).clip(0, 255).astype(np.uint8)
    # 4. 如果是 RGB 图像，OpenCV 需要 BGR 顺序
    mask_bgr = cv2.cvtColor(mask_np, cv2.COLOR_RGB2BGR)

    cv2.imwrite("mask_onnx.png", mask_bgr)

if __name__ == "__main__":
    # 转onnx 
    # 创建示例输入
    dummy_input = torch.randn(1, 3, 224, 224)  # DINOv2标准输入尺寸

    modelpath = "best_seg_aspp.pth"
    # 执行导出
    onnx_path = "best_seg_aspp.onnx"
    onnx_model = export_dinov2_to_onnx(modelpath, dummy_input, onnx_path)

    imagepath = "/home/hualulu/code/dinov2/data/bubble/val/1_bubble_9.jpg"
    predictOnnx(onnx_path, imagepath, 224, 224)
