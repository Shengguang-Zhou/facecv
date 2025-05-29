"""视频流处理模块"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Callable, Dict, List, Optional, Union

import cv2
import numpy as np

from facecv.schemas.face import RecognitionResult

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """流处理配置"""

    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    skip_frames: int = 1  # 每N帧处理一次
    queue_size: int = 2
    show_preview: bool = True
    enable_tracking: bool = True


class VideoStreamProcessor:
    """视频流处理器"""

    def __init__(self, recognizer, config: StreamConfig = None):
        """
        初始化视频流处理器

        Args:
            recognizer: 人脸识别器实例
            config: 流处理配置
        """
        self.recognizer = recognizer
        self.config = config or StreamConfig()
        self.is_running = False
        self.frame_queue = Queue(maxsize=self.config.queue_size)
        self.result_queue = Queue()
        self.stop_event = threading.Event()

    def process_stream(
        self, source: Union[str, int], callback: Optional[Callable[[List[RecognitionResult]], None]] = None
    ) -> List[RecognitionResult]:
        """
        处理视频流

        Args:
            source: 视频源（文件路径、摄像头索引或RTSP URL）
            callback: 识别结果回调函数

        Returns:
            所有识别结果列表
        """
        all_results = []  # Initialize results list first

        try:
            # 打开视频源
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logger.error(f"无法打开视频源: {source}")
                return []

            # 设置摄像头属性
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            logger.info(f"开始处理视频流: {source}")
            logger.info("按 'q' 键退出")

            self.is_running = True

            # 启动推理线程
            inference_thread = threading.Thread(target=self._inference_worker)
            inference_thread.start()

            # FPS 计算
            prev_time = time.time()
            frame_count = 0
            fps = 0
            start_time = time.time()

            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("无法读取视频帧")
                    break

                # 跳帧处理
                frame_count += 1
                if frame_count % self.config.skip_frames != 0:
                    continue

                # 将帧加入队列
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())

                # 获取识别结果
                results = []
                while not self.result_queue.empty():
                    try:
                        result = self.result_queue.get_nowait()
                        results.extend(result)
                        all_results.extend(result)
                    except Empty:
                        break

                # 回调处理
                if callback and results:
                    callback(results)

                # 显示预览
                if self.config.show_preview:
                    display_frame = frame.copy()

                    # 绘制识别结果
                    for result in results:
                        x1, y1, x2, y2 = result.bbox
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        label = f"{result.name}: {result.confidence:.2f}"
                        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # 计算并显示 FPS
                    curr_time = time.time()
                    if curr_time - prev_time > 1.0:
                        fps = frame_count / (curr_time - prev_time)
                        frame_count = 0
                        prev_time = curr_time

                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imshow("Face Recognition", display_frame)

                    # 检查退出键
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("用户按下退出键")
                        break

        except Exception as e:
            logger.error(f"视频流处理错误: {e}")
        finally:
            # 清理资源
            self.is_running = False
            self.stop_event.set()

            # 确保视频捕获对象被释放
            if "cap" in locals() and cap is not None:
                try:
                    cap.release()
                    logger.info("视频捕获对象已释放")
                except Exception as e:
                    logger.error(f"释放视频捕获对象时出错: {e}")

            # 确保所有OpenCV窗口被关闭
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)  # 确保窗口事件被处理
                logger.info("所有OpenCV窗口已关闭")
            except Exception as e:
                logger.error(f"关闭OpenCV窗口时出错: {e}")

            # 等待推理线程结束
            if "inference_thread" in locals() and inference_thread is not None:
                try:
                    inference_thread.join(timeout=2.0)
                    if inference_thread.is_alive():
                        logger.warning("推理线程未能在2秒内结束")
                except Exception as e:
                    logger.error(f"等待推理线程结束时出错: {e}")

            logger.info("视频流处理结束")
            return all_results

    def _inference_worker(self):
        """推理工作线程"""
        while not self.stop_event.is_set():
            try:
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue

                frame = self.frame_queue.get(timeout=0.1)

                # 调整大小以提高处理速度
                if frame.shape[0] > self.config.frame_height or frame.shape[1] > self.config.frame_width:
                    frame = cv2.resize(frame, (self.config.frame_width, self.config.frame_height))

                # 识别人脸
                results = self.recognizer.recognize(frame)

                # 将结果加入队列
                if results:
                    self.result_queue.put(results)

            except Empty:
                continue
            except Exception as e:
                logger.error(f"推理线程错误: {e}")

    async def process_stream_async(
        self, source: Union[str, int], callback: Optional[Callable[[List[RecognitionResult]], None]] = None
    ) -> List[RecognitionResult]:
        """
        异步处理视频流

        Args:
            source: 视频源
            callback: 识别结果回调函数

        Returns:
            所有识别结果列表
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_stream, source, callback)


class VideoStreamManager:
    """Video stream manager for handling multiple concurrent streams"""

    def __init__(self):
        self.streams = {}  # camera_id -> stream_info
        self.results = {}  # camera_id -> results list
        self._lock = threading.Lock()

    def start_stream(self, camera_id: str, source: Union[str, int], process_func: Callable) -> str:
        """Start a new video stream for a specific camera"""

        # Create a thread to process the stream
        def stream_worker():
            try:
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    logger.error(f"Cannot open camera {camera_id} with source {source}")
                    return

                logger.info(f"Started stream for camera {camera_id} with source {source}")

                while camera_id in self.streams:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"Cannot read frame from camera {camera_id}")
                        break

                    # Process frame
                    result = process_func(frame)
                    result["camera_id"] = camera_id  # Add camera_id to result

                    with self._lock:
                        if camera_id not in self.results:
                            self.results[camera_id] = []
                        self.results[camera_id].append(result)

                    # Limit results queue size to prevent memory issues
                    with self._lock:
                        if len(self.results[camera_id]) > 100:
                            self.results[camera_id] = self.results[camera_id][-50:]

                cap.release()
                logger.info(f"Stream ended for camera {camera_id}")

            except Exception as e:
                logger.error(f"Stream worker error for camera {camera_id}: {e}")

        with self._lock:
            # Stop existing stream for this camera if any
            if camera_id in self.streams:
                self.stop_stream(camera_id)

            thread = threading.Thread(target=stream_worker)
            self.streams[camera_id] = {"thread": thread, "source": source, "started_at": time.time()}
            thread.start()

        return camera_id

    def stop_stream(self, camera_id: str) -> bool:
        """Stop a video stream for a specific camera"""
        with self._lock:
            if camera_id in self.streams:
                # Remove from active streams
                stream_info = self.streams.pop(camera_id)
                thread = stream_info["thread"]
                # Thread will stop when it sees camera_id is removed
                thread.join(timeout=1.0)

                # Clean up results
                if camera_id in self.results:
                    del self.results[camera_id]

                logger.info(f"Stopped stream for camera {camera_id}")
                return True
        return False

    def get_results(self, camera_id: str) -> List:
        """Get and clear results for a specific camera stream"""
        with self._lock:
            if camera_id in self.results:
                results = self.results[camera_id]
                self.results[camera_id] = []
                return results
        return []

    def list_active_streams(self) -> List[Dict]:
        """List all active camera streams"""
        with self._lock:
            active_streams = []
            for camera_id, stream_info in self.streams.items():
                active_streams.append(
                    {
                        "camera_id": camera_id,
                        "source": stream_info["source"],
                        "started_at": stream_info["started_at"],
                        "is_alive": stream_info["thread"].is_alive(),
                    }
                )
            return active_streams

    def stop_all_streams(self):
        """Stop all active streams"""
        with self._lock:
            camera_ids = list(self.streams.keys())

        for camera_id in camera_ids:
            self.stop_stream(camera_id)
