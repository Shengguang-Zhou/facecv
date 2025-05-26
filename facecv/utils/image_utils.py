"""图像预处理工具模块"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List, Union, Dict, Any
from PIL import Image, ImageEnhance
from enum import Enum
import base64
import io

logging.basicConfig(level=logging.INFO)


class ImageFormat(Enum):
    """支持的图像格式"""
    JPEG = "JPEG"
    PNG = "PNG"
    BMP = "BMP"
    WEBP = "WEBP"


class ResizeMethod(Enum):
    """图像缩放方法"""
    BILINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    NEAREST = cv2.INTER_NEAREST
    AREA = cv2.INTER_AREA
    LANCZOS = cv2.INTER_LANCZOS4


class ImageValidator:
    """图像验证器"""
    
    def __init__(self, 
                 max_size: int = 10 * 1024 * 1024,  # 10MB
                 min_resolution: Tuple[int, int] = (64, 64),
                 max_resolution: Tuple[int, int] = (4096, 4096),
                 allowed_formats: List[str] = None):
        """
        初始化图像验证器
        
        Args:
            max_size: 最大文件大小（字节）
            min_resolution: 最小分辨率 (width, height)
            max_resolution: 最大分辨率 (width, height)
            allowed_formats: 允许的格式列表
        """
        self.max_size = max_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.allowed_formats = allowed_formats or [
            ImageFormat.JPEG.value, 
            ImageFormat.PNG.value, 
            ImageFormat.BMP.value,
            ImageFormat.WEBP.value
        ]
    
    def validate_image_data(self, image_data: bytes) -> Dict[str, Any]:
        """
        验证图像数据
        
        Args:
            image_data: 图像二进制数据
            
        Returns:
            验证结果字典
        """
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # 检查文件大小
            if len(image_data) > self.max_size:
                result['errors'].append(f"文件大小超限: {len(image_data)} bytes > {self.max_size} bytes")
                return result
            
            if len(image_data) == 0:
                result['errors'].append("文件为空")
                return result
            
            # 尝试加载图像
            try:
                image = Image.open(io.BytesIO(image_data))
                result['info']['format'] = image.format
                result['info']['size'] = image.size
                result['info']['mode'] = image.mode
            except Exception as e:
                result['errors'].append(f"无法解析图像: {str(e)}")
                return result
            
            # 检查格式
            if image.format not in self.allowed_formats:
                result['errors'].append(f"不支持的格式: {image.format}. 支持的格式: {self.allowed_formats}")
                return result
            
            # 检查分辨率
            width, height = image.size
            min_w, min_h = self.min_resolution
            max_w, max_h = self.max_resolution
            
            if width < min_w or height < min_h:
                result['errors'].append(f"分辨率过低: {width}x{height} < {min_w}x{min_h}")
                return result
            
            if width > max_w or height > max_h:
                result['warnings'].append(f"分辨率过高: {width}x{height} > {max_w}x{max_h}, 建议压缩")
            
            # 检查纵横比
            aspect_ratio = width / height
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                result['warnings'].append(f"异常的纵横比: {aspect_ratio:.2f}")
            
            result['valid'] = True
            result['info']['file_size'] = len(image_data)
            result['info']['aspect_ratio'] = aspect_ratio
            
        except Exception as e:
            result['errors'].append(f"验证过程出错: {str(e)}")
        
        return result
    
    def validate_image_array(self, image: np.ndarray) -> Dict[str, Any]:
        """
        验证numpy图像数组
        
        Args:
            image: numpy图像数组
            
        Returns:
            验证结果字典
        """
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            if not isinstance(image, np.ndarray):
                result['errors'].append("输入不是numpy数组")
                return result
            
            # 检查维度
            if len(image.shape) not in [2, 3]:
                result['errors'].append(f"图像维度错误: {len(image.shape)}. 应该是2或3维")
                return result
            
            # 检查数据类型
            if image.dtype not in [np.uint8, np.float32, np.float64]:
                result['warnings'].append(f"数据类型可能不正确: {image.dtype}")
            
            # 检查分辨率
            if len(image.shape) == 3:
                height, width, channels = image.shape
                result['info']['channels'] = channels
                if channels not in [1, 3, 4]:
                    result['warnings'].append(f"异常的通道数: {channels}")
            else:
                height, width = image.shape
                result['info']['channels'] = 1
            
            result['info']['size'] = (width, height)
            result['info']['dtype'] = str(image.dtype)
            result['info']['shape'] = image.shape
            
            # 检查分辨率限制
            min_w, min_h = self.min_resolution
            max_w, max_h = self.max_resolution
            
            if width < min_w or height < min_h:
                result['errors'].append(f"分辨率过低: {width}x{height} < {min_w}x{min_h}")
                return result
            
            if width > max_w or height > max_h:
                result['warnings'].append(f"分辨率过高: {width}x{height}")
            
            # 检查像素值范围
            if image.dtype == np.uint8:
                if image.min() < 0 or image.max() > 255:
                    result['warnings'].append("uint8图像像素值超出[0,255]范围")
            elif image.dtype in [np.float32, np.float64]:
                if image.min() < 0 or image.max() > 1:
                    result['warnings'].append("float图像像素值可能超出[0,1]范围")
            
            result['valid'] = True
            
        except Exception as e:
            result['errors'].append(f"验证过程出错: {str(e)}")
        
        return result


class ImageProcessor:
    """图像处理器"""
    
    def __init__(self):
        self.validator = ImageValidator()
    
    def load_image(self, image_source: Union[str, bytes, np.ndarray]) -> Optional[np.ndarray]:
        """
        加载图像从多种源
        
        Args:
            image_source: 图像源（文件路径、bytes数据或numpy数组）
            
        Returns:
            numpy图像数组或None
        """
        try:
            if isinstance(image_source, str):
                # 文件路径
                if image_source.startswith('data:image'):
                    # Base64编码的图像
                    return self._load_from_base64(image_source)
                else:
                    # 文件路径
                    return self._load_from_file(image_source)
            
            elif isinstance(image_source, bytes):
                # 二进制数据
                return self._load_from_bytes(image_source)
            
            elif isinstance(image_source, np.ndarray):
                # 已经是numpy数组
                validation = self.validator.validate_image_array(image_source)
                if validation['valid']:
                    return image_source
                else:
                    logging.error(f"图像数组验证失败: {validation['errors']}")
                    return None
            
            else:
                logging.error(f"不支持的图像源类型: {type(image_source)}")
                return None
                
        except Exception as e:
            logging.error(f"加载图像失败: {e}")
            return None
    
    def _load_from_file(self, file_path: str) -> Optional[np.ndarray]:
        """从文件加载图像"""
        try:
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if image is None:
                logging.error(f"无法读取图像文件: {file_path}")
                return None
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logging.error(f"读取文件失败 {file_path}: {e}")
            return None
    
    def _load_from_bytes(self, image_data: bytes) -> Optional[np.ndarray]:
        """从bytes数据加载图像"""
        try:
            # 验证图像数据
            validation = self.validator.validate_image_data(image_data)
            if not validation['valid']:
                logging.error(f"图像数据验证失败: {validation['errors']}")
                return None
            
            # 转换为numpy数组
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logging.error("无法解码图像数据")
                return None
            
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            logging.error(f"从bytes加载图像失败: {e}")
            return None
    
    def _load_from_base64(self, base64_str: str) -> Optional[np.ndarray]:
        """从base64字符串加载图像"""
        try:
            # 提取base64数据
            if 'base64,' in base64_str:
                base64_data = base64_str.split('base64,')[1]
            else:
                base64_data = base64_str
            
            # 解码
            image_data = base64.b64decode(base64_data)
            return self._load_from_bytes(image_data)
            
        except Exception as e:
            logging.error(f"从base64加载图像失败: {e}")
            return None
    
    def resize_image(self, 
                    image: np.ndarray, 
                    target_size: Tuple[int, int],
                    method: ResizeMethod = ResizeMethod.BILINEAR,
                    keep_aspect_ratio: bool = True,
                    pad_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        调整图像大小
        
        Args:
            image: 输入图像
            target_size: 目标尺寸 (width, height)
            method: 缩放方法
            keep_aspect_ratio: 是否保持纵横比
            pad_color: 填充颜色 (R, G, B)
            
        Returns:
            调整后的图像
        """
        try:
            target_width, target_height = target_size
            
            if not keep_aspect_ratio:
                # 直接缩放
                return cv2.resize(image, (target_width, target_height), interpolation=method.value)
            
            # 保持纵横比缩放
            h, w = image.shape[:2]
            aspect_ratio = w / h
            target_aspect_ratio = target_width / target_height
            
            if aspect_ratio > target_aspect_ratio:
                # 图像更宽，以宽度为准
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                # 图像更高，以高度为准
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            
            # 缩放图像
            resized = cv2.resize(image, (new_width, new_height), interpolation=method.value)
            
            # 创建目标尺寸的画布并居中放置
            if len(image.shape) == 3:
                canvas = np.full((target_height, target_width, image.shape[2]), pad_color, dtype=image.dtype)
            else:
                canvas = np.full((target_height, target_width), pad_color[0], dtype=image.dtype)
            
            # 计算居中位置
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            
            # 放置图像
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            
            return canvas
            
        except Exception as e:
            logging.error(f"调整图像大小失败: {e}")
            return image
    
    def normalize_image(self, 
                       image: np.ndarray,
                       target_range: Tuple[float, float] = (0.0, 1.0),
                       mean: Optional[Tuple[float, float, float]] = None,
                       std: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """
        标准化图像
        
        Args:
            image: 输入图像
            target_range: 目标像素值范围
            mean: 减去的均值 (用于模型输入标准化)
            std: 除以的标准差 (用于模型输入标准化)
            
        Returns:
            标准化后的图像
        """
        try:
            # 转换为float类型
            normalized = image.astype(np.float32)
            
            # 缩放到目标范围
            if image.dtype == np.uint8:
                normalized = normalized / 255.0
            
            min_val, max_val = target_range
            if min_val != 0.0 or max_val != 1.0:
                normalized = normalized * (max_val - min_val) + min_val
            
            # 应用均值和标准差标准化（用于深度学习模型）
            if mean is not None:
                mean = np.array(mean).reshape(1, 1, -1)
                normalized = normalized - mean
            
            if std is not None:
                std = np.array(std).reshape(1, 1, -1)
                normalized = normalized / std
            
            return normalized
            
        except Exception as e:
            logging.error(f"标准化图像失败: {e}")
            return image
    
    def enhance_image(self, 
                     image: np.ndarray,
                     brightness: float = 1.0,
                     contrast: float = 1.0,
                     saturation: float = 1.0,
                     sharpness: float = 1.0) -> np.ndarray:
        """
        增强图像质量
        
        Args:
            image: 输入图像
            brightness: 亮度调整因子 (1.0=不变)
            contrast: 对比度调整因子 (1.0=不变)
            saturation: 饱和度调整因子 (1.0=不变)
            sharpness: 锐度调整因子 (1.0=不变)
            
        Returns:
            增强后的图像
        """
        try:
            # 转换为PIL图像进行增强
            pil_image = Image.fromarray(image.astype(np.uint8))
            
            # 亮度调整
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(brightness)
            
            # 对比度调整
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(contrast)
            
            # 饱和度调整
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(saturation)
            
            # 锐度调整
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(sharpness)
            
            return np.array(pil_image)
            
        except Exception as e:
            logging.error(f"增强图像失败: {e}")
            return image
    
    def crop_center(self, image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
        """
        中心裁剪图像
        
        Args:
            image: 输入图像
            crop_size: 裁剪尺寸 (width, height)
            
        Returns:
            裁剪后的图像
        """
        try:
            h, w = image.shape[:2]
            crop_w, crop_h = crop_size
            
            # 确保裁剪尺寸不超过原图像
            crop_w = min(crop_w, w)
            crop_h = min(crop_h, h)
            
            # 计算裁剪起始位置
            start_x = (w - crop_w) // 2
            start_y = (h - crop_h) // 2
            
            # 裁剪
            cropped = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
            
            return cropped
            
        except Exception as e:
            logging.error(f"裁剪图像失败: {e}")
            return image
    
    def to_bytes(self, 
                image: np.ndarray, 
                format: ImageFormat = ImageFormat.JPEG,
                quality: int = 95) -> Optional[bytes]:
        """
        将图像转换为bytes
        
        Args:
            image: 输入图像
            format: 输出格式
            quality: 质量(仅对JPEG有效)
            
        Returns:
            图像bytes数据或None
        """
        try:
            # 确保是uint8类型
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # 转换为PIL图像
            pil_image = Image.fromarray(image)
            
            # 保存到bytes
            buffer = io.BytesIO()
            save_kwargs = {}
            
            if format == ImageFormat.JPEG:
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = True
            
            pil_image.save(buffer, format=format.value, **save_kwargs)
            
            return buffer.getvalue()
            
        except Exception as e:
            logging.error(f"转换图像为bytes失败: {e}")
            return None