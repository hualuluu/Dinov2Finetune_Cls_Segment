"""
用于做dinov2 分类的实验代码
test 1: 参考代码 - 微调 backbone + seg , 加载采用 本地的model param , 简单的分割头
test 2: 参考代码 - 微调 backbone + seg , 加载采用 本地的model param , Deeplab Seg

为了后续 转C++ 先采用 Opencv 加载图像
包括：数据加载， 分割头， 测试代码
"""

# 1. 导入并忽略所有警告（可选）
import torch, warnings, os, cv2, random, math, json
warnings.filterwarnings("ignore")
# cuda 设置
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

from sklearn.metrics import classification_report, confusion_matrix
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,  transforms
from dinov2.models import vision_transformer as vits
import torch.optim as optim
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

CUDA_LAUNCH_BLOCKING=1.

# 为 CPU 设置随机种子
torch.manual_seed(42)
# 为当前 GPU 设置种子
torch.cuda.manual_seed(42)
# 如果使用多块 GPU，为所有 GPU 设置种子
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)

PIXEL_MEANS = (0.485, 0.456, 0.406)    # RGB  format mean and variances
PIXEL_STDS = (0.229, 0.224, 0.225)


"""
延用了yolo的letterbox函数
    letterbox:
    满足长宽都要能被32整除，其实直接设置为640,640应该也是可以的
    stride:padImage 长宽的最小公倍数
    new_shape: 图像的新尺寸
    auto:如果为true 则pad后的图像为最小可以整除patchSize的mini矩形，如果为False则图像的大小为640*640
    scale_fill:
    center: 是否以图像的中心点为pad

"""


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


# 构建数据集
class GetSegDatasets(Dataset):
    def __init__(self, dataDir, train = True, inputw=224, inputh=224, patchSize = 14):
        self.imageroot = dataDir
        self.train = train
        self.inputw = inputw
        self.inputh = inputh
        self.patchSize = patchSize
        self.imageData = []
        imagelist = os.listdir(self.imageroot)
        for imagename in imagelist:
            if imagename[-4:] == "json":
                continue
            if imagename.split('.')[0][-3:] == "vis":
                continue
            imagepath = os.path.join(self.imageroot, imagename)
            jsonpath = imagepath.split('.')[0] + ".json"
            self.imageData.append([imagepath, jsonpath])

    def __len__(self):
        return len(self.imageData)

    def getImageTensor(self, imagepath):

        image = cv2.imread(imagepath)
        imagePad = letterbbox(image, [self.inputw, self.inputh], self.patchSize) # 获得pad的图像
        imageRGB = cv2.cvtColor(imagePad, cv2.COLOR_BGR2RGB) # RGB

        # 先试试简单的 resize 
        image = cv2.resize(imageRGB, (self.inputw, self.inputh), interpolation=cv2.INTER_LINEAR)
        # 4. 转换为 float32 并缩放到 [0, 1]（等效 ToTensor）
        img_float = image.astype(np.float32) / 255.0

        # 5. 归一化 (Normalize)
        # 将 mean 和 std 转换为 numpy 数组并 reshape 以适配广播
        mean_arr = np.array(PIXEL_MEANS, dtype=np.float32).reshape(1, 1, 3)
        std_arr = np.array(PIXEL_STDS, dtype=np.float32).reshape(1, 1, 3)
        img_norm = (img_float - mean_arr) / std_arr

        # 6. 将 HWC 转换为 CHW（PyTorch 张量格式）
        img_chw = np.transpose(img_norm, (2, 0, 1))
        imageTensor = torch.from_numpy(img_chw)

        return imageTensor, imagePad
    
    def getMaskTensor(self, jsonpath, imagepath):

        image = cv2.imread(imagepath)
        w, h = image.shape[1], image.shape[0]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        with open(jsonpath, 'r') as f:
            annInfo = json.load(f)
            maskInfos = annInfo["shapes"]

            for maskInfo in maskInfos:
                pts = maskInfo["points"]
                pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

                # 绘制多边形轮廓（绿色线条，粗细 2）
                cv2.fillPoly(mask, [pts], 1)  # 将所有多边形填充为白色

        maskPad = letterbbox(mask, [self.inputw, self.inputh], self.patchSize, 0) # 获得pad的图像
        
        maskchw = np.transpose(maskPad, (2, 0, 1))
        maskTensor = torch.from_numpy(maskchw).long() 
        
        return maskTensor, maskPad

    def __getitem__(self, id):

        # 1. image 
        imagepath = self.imageData[id][0]
        imageTensor, imagePad = self.getImageTensor(imagepath)

        # 2. 读 mask json获得mask图像
        jsonpath = self.imageData[id][1]
        maskTensor, maskPad = self.getMaskTensor(jsonpath,imagepath)

        #print(maskTensor)
        maskPad = cv2.cvtColor(maskPad, cv2.COLOR_GRAY2BGR)
        merge = cv2.addWeighted(imagePad, 0.5, maskPad * 128, 0.5, 0)
        mergepath = imagepath.split('.')[0] + "_vis.jpg"
        cv2.imwrite(mergepath, merge)

        return imageTensor, maskTensor
        


def getDinov2SegDatasets(dataDir, imageSize = 518, batchSize = 64, patchSize = 14):
    valroot = os.path.join(dataDir, "val")
    ValClsDataset = GetSegDatasets(valroot, True, imageSize, imageSize, patchSize)
    valLoader= DataLoader(ValClsDataset, batch_size=batchSize, shuffle=True, num_workers=4)
    print(f"Validation samples: {len(ValClsDataset)}")

    trainroot = os.path.join(dataDir, "train")
    TrainClsDataset = GetSegDatasets(trainroot, True, imageSize, imageSize, patchSize)
    trainLoader = DataLoader(TrainClsDataset, batch_size=batchSize, shuffle=True,  num_workers=4)
    print(f"Training samples: {len(TrainClsDataset)}")
    
    return trainLoader , valLoader

#----------------------Model & Head------------------

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
    

# 比较模型参数
def compare_state_dicts(state_dict1, state_dict2, threshold=1e-6):
    """
    比较两个 state_dict，输出参数发生变化的层及变化幅度。
    
    Args:
        state_dict1 (dict): 第一个 epoch 的权重字典
        state_dict2 (dict): 第二个 epoch 的权重字典
        threshold (float): 判断参数是否“改变”的阈值（基于最大绝对值差异）
    
    Returns:
        dict: 包含变化层信息的字典，键为层名，值为 (L2距离, 最大差异)
    """
    changed_layers = {}
    all_keys = set(state_dict1.keys()) | set(state_dict2.keys())
    
    for key in all_keys:
        if key not in state_dict1:
            print(f"Warning: {key} not found in first state_dict")
            continue
        if key not in state_dict2:
            print(f"Warning: {key} not found in second state_dict")
            continue
        
        param1 = state_dict1[key]
        param2 = state_dict2[key]
        
        # 确保形状相同
        if param1.shape != param2.shape:
            print(f"Shape mismatch for {key}: {param1.shape} vs {param2.shape}")
            continue
        
        # 计算差异
        diff = param1 - param2
        l2_dist = torch.norm(diff).item()
        max_diff = torch.max(torch.abs(diff)).item()
        
        if max_diff > threshold:
            changed_layers[key] = (l2_dist, max_diff)
    
    return changed_layers


# 为了保证loacal模式 和hub模式下一样可以支持任意输入尺寸图像
def adapt_position_encoding(model, new_img_size=518, patch_size=14):
    """
    调整 DINOv2 模型的位置编码以适配新的输入尺寸。
    
    Args:
        model: DINOv2 ViT 模型（应包含 pos_embed 属性）
        new_img_size: 新的正方形图像边长（像素）
        patch_size: 模型的 patch 大小（DINOv2 默认为 14）
    """
    # 获取旧的位置编码
    old_pos_embed = model.pos_embed  # 形状: [1, old_seq_len, dim]
    old_seq_len = old_pos_embed.shape[1]
    embed_dim = old_pos_embed.shape[-1]
    
    # 计算旧的 patch 网格大小
    old_num_patches = (model.patch_embed.img_size[0] // patch_size) ** 2
    # 额外的 token 数量（如 [CLS]、寄存器等）
    num_extra_tokens = old_seq_len - old_num_patches
    
    # 计算新的 patch 网格大小和序列长度
    new_grid_size = new_img_size // patch_size
    new_num_patches = new_grid_size ** 2
    new_seq_len = new_num_patches + num_extra_tokens
    
    if old_seq_len == new_seq_len:
        print("位置编码尺寸已匹配，无需调整。")
        return
    
    print(f"调整位置编码: 从 {old_seq_len} 到 {new_seq_len}")
    
    # 分离固定 token 的位置编码（前 num_extra_tokens 个）
    extra_pos_embed = old_pos_embed[:, :num_extra_tokens, :]  # [1, num_extra_tokens, dim]
    
    # 分离图像块的位置编码
    patch_pos_embed = old_pos_embed[:, num_extra_tokens:, :]  # [1, old_num_patches, dim]
    
    # 将图像块位置编码从 1D 序列重塑为 2D 空间网格
    old_grid_size = int(math.sqrt(old_num_patches))
    # 形状: [1, dim, old_grid_size, old_grid_size]
    patch_pos_embed_2d = patch_pos_embed.reshape(1, old_grid_size, old_grid_size, embed_dim).permute(0, 3, 1, 2)
    
    # 使用双三次插值缩放到新的网格尺寸
    new_patch_pos_embed_2d = F.interpolate(
        patch_pos_embed_2d,
        size=(new_grid_size, new_grid_size),
        mode='bicubic',
        align_corners=False
    )
    
    # 将插值后的 2D 网格重新展平为 1D 序列
    # 形状: [1, new_num_patches, dim]
    new_patch_pos_embed = new_patch_pos_embed_2d.permute(0, 2, 3, 1).reshape(1, new_num_patches, embed_dim)
    
    # 拼接固定 token 和新的图像块位置编码
    new_pos_embed = torch.cat([extra_pos_embed, new_patch_pos_embed], dim=1)
    
    # 替换模型中的位置编码
    model.pos_embed = torch.nn.Parameter(new_pos_embed)
    print("位置编码调整完成。")


# modelFlag = "local" / "hub"
def getDinov2Vits14Model(modelFlag = "local", imageSize = 224, patchSize = 14):

    vit_model= None
    if(modelFlag == "local"):
        
        # vit_model = vits.vit_small(img_size = imageSize, patch_size=patchSize)
        vit_model = vits.__dict__['vit_small'](
            img_size=518,#imageSize,
            patch_size=patchSize,
            num_register_tokens=0,          # 官方模型有4个寄存器
            init_values=1e-5,                # LayerScale初始值
            block_chunks=0,                   # 0表示不分组
        )
        state_dict = torch.load("dinov2_vits14_pretrain.pth", map_location='cpu')
        # 直接加载，使用 strict=False 查看缺失/多余键
        msg = vit_model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", msg.missing_keys)
        print("Unexpected keys:", msg.unexpected_keys)

        # 调整位置编码到目标尺寸 imageSize
        adapt_position_encoding(vit_model, new_img_size=imageSize, patch_size=patchSize)

    elif(modelFlag == "hub"):
        vit_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    if vit_model != None:
        print("vit_small Backbone loaded successfully!")

    return vit_model


# ----------------------- Loss-----------------
# loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: [B, C, H, W], targets: [B, H, W]
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.eye(logits.shape[1], device=logits.device)[targets]  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).contiguous()          # [B, C, H, W]
        
        intersection = (probs * targets_one_hot).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)           # [B, C]
        dice_loss = 1.0 - dice.mean(dim=1)                                           # [B]
        return dice_loss.mean()
    


class DiceCELoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice
    

# 测试流程
def predict(imagepath, modelpath, inputSizeW, inputSizeH, patchSize):
    
    
    modelDinov2Seg = torch.load(modelpath).to(device)
    modelDinov2Seg.eval()
    
    with torch.no_grad():
        image = cv2.imread(imagepath)
        imagePad = letterbbox(image, [inputSizeW, inputSizeH], patchSize) # 获得pad的图像
        imageRGB = cv2.cvtColor(imagePad, cv2.COLOR_BGR2RGB) # RGB

        # 先试试简单的 resize 
        image = cv2.resize(imageRGB, (inputSizeW, inputSizeH), interpolation=cv2.INTER_LINEAR)
        # 4. 转换为 float32 并缩放到 [0, 1]（等效 ToTensor）
        img_float = image.astype(np.float32) / 255.0

        # 5. 归一化 (Normalize)
        # 将 mean 和 std 转换为 numpy 数组并 reshape 以适配广播
        mean_arr = np.array(PIXEL_MEANS, dtype=np.float32).reshape(1, 1, 3)
        std_arr = np.array(PIXEL_STDS, dtype=np.float32).reshape(1, 1, 3)
        img_norm = (img_float - mean_arr) / std_arr

        # 6. 将 HWC 转换为 CHW（PyTorch 张量格式）
        img_chw = np.transpose(img_norm, (2, 0, 1))
        imageTensor = torch.from_numpy(img_chw).unsqueeze(dim=0).to(device)

        preds = modelDinov2Seg(imageTensor)
        mask = torch.argmax(preds, dim=1)        # [B, H, W]
        print(mask.shape)

        # 1. 移动到 CPU 并分离计算图
        mask_np = mask.detach().cpu().numpy()
        # 2. 如果张量是 CHW，转换为 HWC
        mask_np = np.transpose(mask_np, (1, 2, 0))  # [H, W, C]
        # 3. 如果张量值在 [0,1] 范围，转换为 [0,255] 并转为 uint8
        mask_np = (mask_np * 255).clip(0, 255).astype(np.uint8)
        # 4. 如果是 RGB 图像，OpenCV 需要 BGR 顺序
        mask_bgr = cv2.cvtColor(mask_np, cv2.COLOR_RGB2BGR)

        cv2.imwrite("mask.png", mask_bgr)

# 训练流程
def train(trainLoader, modelpath,  epoch = 150, imageSize = 224, patchSize = 14, numClass = 6, 
          vitLoad = "local", head_type = "simple", freeze = False):

    # net
    dinov2_vits14 = getDinov2Vits14Model(vitLoad, imageSize, patchSize)
    num_patches = imageSize // patchSize
    modelDinov2Seg = Dinov2SegModel(dinov2_vits14, numClass, num_patches, head_type).to(device)
    modelDinov2Seg.train()

    # 损失函数的优化器定义
    criterion = DiceCELoss()
    optimizer = None
    if not freeze:
        optimizer = optim.Adam(modelDinov2Seg.parameters(), lr=0.000001)
    else:
        optimizer = optim.Adam(modelDinov2Seg.segment.parameters(), lr=0.001)

    saveModelval = 0.0
    prev_params = None
    for ie in range(epoch):
        
        train_loss = 0
        train_pbar = tqdm(trainLoader, desc=f"Epoch {ie+1}/{epoch} [Train]")
        num = 0

        for image, seg in train_pbar:

            image, target = image.to(device), seg.to(device).squeeze(dim=1)
            optimizer.zero_grad()

            preds = modelDinov2Seg(image)
            trainloss = criterion(preds, target)

            trainloss.backward()
            optimizer.step()

            num+=1.0
            train_loss += trainloss
            train_pbar.set_postfix(loss=f"{trainloss.item():.4f}")

        train_loss /= num
        if (saveModelval < ((1- train_loss))) :
            torch.save(modelDinov2Seg, modelpath)

        current_params = {name: param.data.clone() for name, param in modelDinov2Seg.named_parameters() if param.requires_grad}
    
        if prev_params is not None:
            changed = compare_state_dicts(prev_params, current_params, threshold=1e-6)
            print(f"Epoch {epoch}: {len(changed)} 层参数更新")
            #print(changed)
        
        prev_params = current_params
        


if __name__ == "__main__":

    BatchSize = 16
    epoch = 20
    imageSize = 658
    patchSize = 14
    numClass = 2
    class_names = ['calss_0', 'calss_1', 'calss_2', 'calss_3', 'calss_4']#['calss_0', 'calss_1', 'calss_2', 'calss_3', 'calss_4', 'calss_5']
    modelpath = "best_seg_aspp.pth"
    vitLoad = "local"
    freeze = True
    dataTrans = "myself" # transform myself
    head_type = "aspp"
    dataDir = "/home/hualulu/code/dinov2/data/bubble/"
    
    trainLoader, valLoader = getDinov2SegDatasets(dataDir, imageSize, BatchSize, patchSize)

    #train(trainLoader, modelpath,  epoch, imageSize, patchSize, numClass, vitLoad, head_type, freeze)
    
    imagepath = "/home/hualulu/code/dinov2/data/bubble/val/1_bubble_9.jpg"
    predict(imagepath, modelpath, imageSize, imageSize, patchSize)

