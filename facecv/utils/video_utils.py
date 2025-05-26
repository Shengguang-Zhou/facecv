"""视频处理工具模块"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List, Union, Dict, Any, Generator, Callable
from enum import Enum
from pathlib import Path
import time
from datetime import datetime, timedelta
import threading
import queue

logging.basicConfig(level=logging.INFO)


class VideoCodec(Enum):
    """视频编码器"""
    MP4V = 'mp4v'
    XVID = 'XVID'
    H264 = 'H264'
    MJPG = 'MJPG'


class FrameExtractionMethod(Enum):
    """帧提取方法"""
    UNIFORM = "uniform"      # 均匀采样
    KEYFRAME = "keyframe"    # 关键帧
    INTERVAL = "interval"    # 固定间隔
    ADAPTIVE = "adaptive"    # 自适应采样


class VideoInfo:
    """视频信息数据类"""
    
    def __init__(self, video_path: str):
        self.path = video_path
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.frame_count = 0
        self.duration = 0.0
        self.codec = ""
        self.size_mb = 0.0
        self.bitrate = 0
        self.is_valid = False
        
        self._analyze_video()
    
    def _analyze_video(self):
        """分析视频信息"""
        try:
            cap = cv2.VideoCapture(self.path)
            
            if not cap.isOpened():
                logging.error(f"无法打开视频文件: {self.path}")
                return
            
            # 获取视频属性
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if self.fps > 0:
                self.duration = self.frame_count / self.fps
            
            # 获取文件大小
            file_path = Path(self.path)
            if file_path.exists():
                self.size_mb = file_path.stat().st_size / (1024 * 1024)
                if self.duration > 0:
                    self.bitrate = int((self.size_mb * 8 * 1024) / self.duration)
            
            # 获取编码信息
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            self.codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            cap.release()
            self.is_valid = True
            
        except Exception as e:
            logging.error(f"分析视频信息失败: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'path': self.path,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'duration': self.duration,
            'codec': self.codec,
            'size_mb': self.size_mb,
            'bitrate': self.bitrate,
            'is_valid': self.is_valid,
            'resolution': f"{self.width}x{self.height}",
            'aspect_ratio': self.width / self.height if self.height > 0 else 0
        }
    
    def __str__(self):
        return f"VideoInfo({self.width}x{self.height}, {self.fps:.1f}fps, {self.duration:.1f}s, {self.codec})"


class VideoExtractor:
    """视频帧提取器"""
    
    def __init__(self, video_source: Union[str, int]):
        """
        初始化视频提取器
        
        Args:
            video_source: 视频源（文件路径或摄像头ID）
        """
        self.video_source = video_source
        self.cap = None
        self.video_info = None
        
        if isinstance(video_source, str):
            self.video_info = VideoInfo(video_source)
    
    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频源: {self.video_source}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
    
    def extract_frames(self, 
                      method: FrameExtractionMethod = FrameExtractionMethod.UNIFORM,
                      count: int = 10,
                      interval: float = 1.0,
                      start_time: float = 0.0,
                      end_time: Optional[float] = None,
                      target_size: Optional[Tuple[int, int]] = None) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        """
        提取视频帧
        
        Args:
            method: 提取方法
            count: 提取帧数（用于uniform方法）
            interval: 提取间隔秒数（用于interval方法）
            start_time: 开始时间（秒）
            end_time: 结束时间（秒），None表示到视频结尾
            target_size: 目标尺寸 (width, height)
            
        Yields:
            (frame_index, frame, timestamp): 帧索引、帧图像、时间戳
        """
        if not self.cap:
            raise ValueError("视频未打开，请使用with语句")
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            fps = 25.0  # 默认帧率
        
        # 计算起始和结束帧
        start_frame = int(start_time * fps)
        if end_time is not None:
            end_frame = min(int(end_time * fps), total_frames)
        else:
            end_frame = total_frames
        
        # 设置起始位置
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        if method == FrameExtractionMethod.UNIFORM:
            # 均匀采样
            frame_indices = np.linspace(start_frame, end_frame - 1, count, dtype=int)
            
            for frame_idx in frame_indices:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                
                if ret:
                    timestamp = frame_idx / fps
                    
                    if target_size:
                        frame = cv2.resize(frame, target_size)
                    
                    # 转换为RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yield frame_idx, frame_rgb, timestamp
        
        elif method == FrameExtractionMethod.INTERVAL:
            # 固定间隔采样
            interval_frames = int(interval * fps)
            current_frame = start_frame
            
            while current_frame < end_frame:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = self.cap.read()
                
                if ret:
                    timestamp = current_frame / fps
                    
                    if target_size:
                        frame = cv2.resize(frame, target_size)
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yield current_frame, frame_rgb, timestamp
                
                current_frame += interval_frames
        
        elif method == FrameExtractionMethod.ADAPTIVE:
            # 自适应采样（基于场景变化）
            prev_frame = None
            threshold = 30.0  # 场景变化阈值
            
            current_frame = start_frame
            
            while current_frame < end_frame:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # 转换为灰度图进行场景变化检测
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # 计算帧差
                    diff = cv2.absdiff(gray, prev_frame)
                    mean_diff = np.mean(diff)
                    
                    if mean_diff > threshold:
                        # 场景变化较大，提取这一帧
                        timestamp = current_frame / fps
                        
                        if target_size:
                            frame = cv2.resize(frame, target_size)
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        yield current_frame, frame_rgb, timestamp
                
                prev_frame = gray
                current_frame += 1
    
    def extract_single_frame(self, 
                           time_position: float,
                           target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """
        提取指定时间位置的单帧
        
        Args:
            time_position: 时间位置（秒）
            target_size: 目标尺寸 (width, height)
            
        Returns:
            帧图像或None
        """
        if not self.cap:
            raise ValueError("视频未打开，请使用with语句")
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_index = int(time_position * fps)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        
        if ret:
            if target_size:
                frame = cv2.resize(frame, target_size)
            
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return None
    
    def get_video_info(self) -> Optional[VideoInfo]:
        """获取视频信息"""
        return self.video_info


class VideoProcessor:
    """视频处理器"""
    
    def __init__(self):
        self.is_processing = False
        self.stop_flag = threading.Event()
    
    def process_video_stream(self,
                           video_source: Union[str, int],
                           frame_processor: Callable[[np.ndarray, float], Any],
                           fps_limit: int = 30,
                           skip_frames: int = 0,
                           output_queue: Optional[queue.Queue] = None) -> Generator[Tuple[np.ndarray, Any, float], None, None]:
        """
        处理视频流
        
        Args:
            video_source: 视频源
            frame_processor: 帧处理函数 (frame, timestamp) -> result
            fps_limit: FPS限制
            skip_frames: 跳过的帧数
            output_queue: 输出队列（用于异步处理）
            
        Yields:
            (frame, result, timestamp): 处理后的帧、结果、时间戳
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频源: {video_source}")
        
        try:
            self.is_processing = True
            self.stop_flag.clear()
            
            frame_interval = 1.0 / fps_limit if fps_limit > 0 else 0
            last_process_time = 0
            frame_count = 0
            
            while not self.stop_flag.is_set():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                current_time = time.time()
                
                # 跳帧处理
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    continue
                
                # FPS限制
                if frame_interval > 0:
                    time_diff = current_time - last_process_time
                    if time_diff < frame_interval:
                        time.sleep(frame_interval - time_diff)
                        current_time = time.time()
                
                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 处理帧
                timestamp = current_time
                result = frame_processor(frame_rgb, timestamp)
                
                # 输出结果
                output_data = (frame_rgb, result, timestamp)
                
                if output_queue:
                    try:
                        output_queue.put_nowait(output_data)
                    except queue.Full:
                        pass  # 队列满了，跳过这一帧
                else:
                    yield output_data
                
                last_process_time = current_time
                frame_count += 1
        
        finally:
            cap.release()
            self.is_processing = False
    
    def stop_processing(self):
        """停止处理"""
        self.stop_flag.set()
    
    def create_video_from_frames(self,
                                frames: List[np.ndarray],
                                output_path: str,
                                fps: float = 25.0,
                                codec: VideoCodec = VideoCodec.MP4V,
                                quality: int = 95) -> bool:
        """
        从帧序列创建视频
        
        Args:
            frames: 帧列表
            output_path: 输出路径
            fps: 帧率
            codec: 编码器
            quality: 质量（0-100）
            
        Returns:
            是否成功
        """
        if not frames:
            logging.error("帧列表为空")
            return False
        
        try:
            # 获取第一帧的尺寸
            height, width = frames[0].shape[:2]
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*codec.value)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                logging.error(f"无法创建视频文件: {output_path}")
                return False
            
            # 写入帧
            for frame in frames:
                # 确保尺寸一致
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                
                # 转换为BGR
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                out.write(frame_bgr)
            
            out.release()
            logging.info(f"视频创建成功: {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"创建视频失败: {e}")
            return False
    
    def resize_video(self,
                    input_path: str,
                    output_path: str,
                    target_size: Tuple[int, int],
                    keep_aspect_ratio: bool = True) -> bool:
        """
        调整视频尺寸
        
        Args:
            input_path: 输入路径
            output_path: 输出路径
            target_size: 目标尺寸 (width, height)
            keep_aspect_ratio: 是否保持纵横比
            
        Returns:
            是否成功
        """
        try:
            # 获取输入视频信息
            video_info = VideoInfo(input_path)
            if not video_info.is_valid:
                logging.error(f"无效的视频文件: {input_path}")
                return False
            
            cap = cv2.VideoCapture(input_path)
            
            # 计算输出尺寸
            if keep_aspect_ratio:
                target_width, target_height = target_size
                aspect_ratio = video_info.width / video_info.height
                
                if target_width / target_height > aspect_ratio:
                    output_width = int(target_height * aspect_ratio)
                    output_height = target_height
                else:
                    output_width = target_width
                    output_height = int(target_width / aspect_ratio)
            else:
                output_width, output_height = target_size
            
            # 创建输出视频
            fourcc = cv2.VideoWriter_fourcc(*VideoCodec.MP4V.value)
            out = cv2.VideoWriter(output_path, fourcc, video_info.fps, (output_width, output_height))
            
            # 处理每一帧
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 调整尺寸
                resized_frame = cv2.resize(frame, (output_width, output_height))
                out.write(resized_frame)
            
            cap.release()
            out.release()
            
            logging.info(f"视频尺寸调整完成: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"调整视频尺寸失败: {e}")
            return False
    
    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        """
        提取视频音频（需要ffmpeg）
        
        Args:
            video_path: 视频文件路径
            audio_path: 音频输出路径
            
        Returns:
            是否成功
        """
        try:
            import subprocess
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # 不包含视频流
                '-acodec', 'copy',  # 复制音频流
                '-y',  # 覆盖输出文件
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"音频提取成功: {audio_path}")
                return True
            else:
                logging.error(f"音频提取失败: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"提取音频时出错: {e}")
            return False