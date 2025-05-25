import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import os

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
    
    def forward(self, x):
        return self.conv_out(self.conv(x))

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
    
    def forward(self, x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        return torch.mul(feat, atten)

class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = self._get_resnet18_backbone()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128)
        self.conv_head16 = ConvBNReLU(128, 128)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, padding=0)
    
    def _get_resnet18_backbone(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        features = []
        for name, module in model.named_children():
            if name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']:
                features.append(module)
        
        backbone = nn.Sequential(*features)
        
        def forward(x):
            feat4 = backbone[:6](x)  # layer1
            feat8 = backbone[6](feat4)  # layer2
            feat16 = backbone[7](feat8)  # layer3
            feat32 = backbone[8](feat16)  # layer4
            return feat8, feat16, feat32
        
        backbone.forward = forward
        return backbone
    
    def forward(self, x):
        feat8, feat16, feat32 = self.backbone(x)
        
        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)
        
        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = nn.functional.interpolate(feat32_sum, size=feat16.size()[2:], mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = nn.functional.interpolate(feat16_sum, size=feat8.size()[2:], mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan//4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_chan//4, out_chan, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        return feat_atten + feat

class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256 + 128, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
    
    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = nn.functional.interpolate(feat_out, size=(H, W), mode='bilinear', align_corners=True)
        feat_out16 = nn.functional.interpolate(feat_out16, size=(H, W), mode='bilinear', align_corners=True)
        feat_out32 = nn.functional.interpolate(feat_out32, size=(H, W), mode='bilinear', align_corners=True)
        
        return feat_out, feat_out16, feat_out32

class HairSegmentator:
    def __init__(self, model_path=None):
        """
        Khởi tạo bộ phân đoạn tóc
        
        Args:
            model_path: đường dẫn đến file model đã huấn luyện (.pth)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BiSeNet(n_classes=19)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Đã tải model từ: {model_path}")
        else:
            print("Cảnh báo: Chưa tải model. Vui lòng tải model bằng load_model()")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Chuẩn hóa ảnh theo ImageNet
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def load_model(self, model_path):
        """Tải model từ file"""
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Đã tải model từ: {model_path}")
        else:
            raise FileNotFoundError(f"Không tìm thấy file model: {model_path}")
    
    def preprocess_image(self, image_path, target_size=512):
        """
        Tiền xử lý ảnh đầu vào
        
        Args:
            image_path: đường dẫn đến ảnh hoặc PIL Image
            target_size: kích thước ảnh sau khi resize
        
        Returns:
            tensor ảnh đã được tiền xử lý
        """
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        # Lưu kích thước gốc
        original_size = image.size
        
        # Resize ảnh
        image = image.resize((target_size, target_size), Image.BILINEAR)
        
        # Chuyển đổi thành tensor
        image_tensor = self.transform(image).unsqueeze(0)
        
        return image_tensor.to(self.device), original_size
    
    def segment_hair(self, image_path, return_confidence=False):
        """
        Phân đoạn tóc từ ảnh
        
        Args:
            image_path: đường dẫn đến ảnh hoặc PIL Image
            return_confidence: có trả về độ tin cậy không
        
        Returns:
            hair_mask: mask tóc (numpy array)
            confidence: độ tin cậy (nếu return_confidence=True)
        """
        # Tiền xử lý ảnh
        image_tensor, original_size = self.preprocess_image(image_path)
        
        # Dự đoán
        with torch.no_grad():
            outputs = self.model(image_tensor)
            prediction = outputs[0].squeeze(0).cpu().numpy()
        
        # Lấy class có xác suất cao nhất
        segmentation = np.argmax(prediction, axis=0)
        
        # Tạo mask tóc (class 17 là tóc trong CelebAMask-HQ)
        hair_mask = (segmentation == 17).astype(np.uint8) * 255
        
        # Resize về kích thước gốc
        hair_mask = cv2.resize(hair_mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        if return_confidence:
            # Tính độ tin cậy trung bình cho vùng tóc
            hair_confidence = np.max(prediction[17]) if np.any(segmentation == 17) else 0.0
            return hair_mask, hair_confidence
        
        return hair_mask
    
    def save_mask(self, mask, output_path):
        """Lưu mask ra file"""
        cv2.imwrite(output_path, mask)
        print(f"Đã lưu mask tóc tại: {output_path}")
    
    def apply_hair_color(self, image_path, hair_mask, color=(255, 0, 0), blend_ratio=0.7):
        """
        Tô màu tóc
        
        Args:
            image_path: đường dẫn đến ảnh gốc
            hair_mask: mask tóc
            color: màu RGB (R, G, B)
            blend_ratio: tỷ lệ pha trộn màu (0-1)
        
        Returns:
            ảnh đã tô màu tóc
        """
        # Đọc ảnh gốc
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
        
        # Đảm bảo mask có cùng kích thước với ảnh
        if hair_mask.shape[:2] != image.shape[:2]:
            hair_mask = cv2.resize(hair_mask, (image.shape[1], image.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Tạo overlay màu
        color_overlay = np.zeros_like(image)
        color_overlay[:] = color[::-1]  # Chuyển RGB sang BGR
        
        # Áp dụng màu lên vùng tóc
        mask_3d = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR) / 255.0
        colored_image = image * (1 - mask_3d * blend_ratio) + color_overlay * mask_3d * blend_ratio
        
        return colored_image.astype(np.uint8)

def main():
    """Hàm demo sử dụng"""
    # Khởi tạo bộ phân đoạn tóc
    segmentator = HairSegmentator()
    
    # Tải model (cần tải model trước)
    model_path = "pretrained/79999_iter.pth"
    if os.path.exists(model_path):
        segmentator.load_model(model_path)
    else:
        print(f"Vui lòng tải model và đặt tại: {model_path}")
        print("Link tải: https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812")
        return
    
    # Đường dẫn ảnh đầu vào
    image_path = "input_image.jpg"  # Thay đổi đường dẫn này
    
    if not os.path.exists(image_path):
        print(f"Không tìm thấy ảnh: {image_path}")
        return
    
    # Phân đoạn tóc
    print("Đang phân đoạn tóc...")
    hair_mask = segmentator.segment_hair(image_path)
    
    # Lưu mask
    segmentator.save_mask(hair_mask, "hair_mask.png")
    
    # Tô màu tóc
    colors = {
        'Đỏ': (255, 0, 0),
        'Xanh lá': (0, 255, 0),
        'Xanh dương': (0, 0, 255),
        'Vàng': (255, 255, 0),
        'Tím': (255, 0, 255),
        'Nâu': (165, 42, 42)
    }
    
    for color_name, color_value in colors.items():
        colored_image = segmentator.apply_hair_color(image_path, hair_mask, color_value)
        output_path = f"hair_colored_{color_name.lower()}.jpg"
        cv2.imwrite(output_path, colored_image)
        print(f"Đã tạo ảnh tóc màu {color_name}: {output_path}")

if __name__ == "__main__":
    main()
