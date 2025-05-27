"""System Health Monitoring API Routes"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
import logging
import time
import os
from datetime import datetime
from pydantic import BaseModel

# Try to import system monitoring libraries
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    GPUtil = None

router = APIRouter(prefix="/api/v1/health", tags=["System Health"])
logger = logging.getLogger(__name__)

class SystemMetrics(BaseModel):
    cpu: Dict[str, Any]
    memory: Dict[str, Any] 
    disk: Dict[str, Any]
    gpu: Optional[Dict[str, Any]] = None
    database: Optional[Dict[str, Any]] = None
    models: Optional[Dict[str, Any]] = None

class ComprehensiveHealth(BaseModel):
    healthy: bool
    status: str  # "healthy", "warning", "critical"
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    metrics: SystemMetrics
    timestamp: str

def get_cpu_info():
    """Get CPU information"""
    try:
        if not HAS_PSUTIL:
            return {"error": "psutil not available", "usage": 0.0}
        
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else None
        
        return {
            "usage": cpu_percent,
            "count": cpu_count,
            "frequency_mhz": cpu_freq.current if cpu_freq else None,
            "load_average": load_avg,
            "status": "normal" if cpu_percent < 80 else "high" if cpu_percent < 95 else "critical"
        }
    except Exception as e:
        logger.error(f"Error getting CPU info: {e}")
        return {"error": str(e), "usage": 0.0}

def get_memory_info():
    """Get memory information"""
    try:
        if not HAS_PSUTIL:
            return {"error": "psutil not available", "usage_percent": 0.0}
        
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            "total": memory.total // (1024**3),  # GB
            "used": memory.used // (1024**3),    # GB
            "available": memory.available // (1024**3),  # GB
            "usage_percent": memory.percent,
            "swap_total": swap.total // (1024**3),  # GB
            "swap_used": swap.used // (1024**3),    # GB
            "swap_percent": swap.percent,
            "status": "normal" if memory.percent < 80 else "high" if memory.percent < 95 else "critical"
        }
    except Exception as e:
        logger.error(f"Error getting memory info: {e}")
        return {"error": str(e), "usage_percent": 0.0}

def get_disk_info():
    """Get disk information"""
    try:
        if not HAS_PSUTIL:
            return {"error": "psutil not available", "usage_percent": 0.0}
        
        disk = psutil.disk_usage('/')
        
        return {
            "total": disk.total // (1024**3),  # GB
            "used": disk.used // (1024**3),    # GB  
            "free": disk.free // (1024**3),    # GB
            "usage_percent": (disk.used / disk.total) * 100,
            "status": "normal" if (disk.used / disk.total) < 0.8 else "high" if (disk.used / disk.total) < 0.95 else "critical"
        }
    except Exception as e:
        logger.error(f"Error getting disk info: {e}")
        return {"error": str(e), "usage_percent": 0.0}

def get_gpu_info():
    """Get GPU information with multiple detection methods"""
    # First, try using nvidia-smi command for most reliable detection
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_count = int(result.stdout.strip())
            if gpu_count > 0:
                # Get detailed info via nvidia-smi
                query_cmd = ['nvidia-smi', 
                           '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu',
                           '--format=csv,noheader,nounits']
                detail_result = subprocess.run(query_cmd, capture_output=True, text=True, timeout=5)
                
                if detail_result.returncode == 0:
                    devices = []
                    lines = detail_result.stdout.strip().split('\n')
                    for line in lines:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            devices.append({
                                "id": int(parts[0]),
                                "name": parts[1],
                                "memory_total": int(parts[2]),  # MB
                                "memory_used": int(parts[3]),   # MB
                                "utilization": int(parts[4]),
                                "temperature": int(parts[5])
                            })
                    
                    # Get driver version
                    driver_result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', 
                                                  '--format=csv,noheader'], 
                                                 capture_output=True, text=True, timeout=5)
                    driver_version = driver_result.stdout.strip() if driver_result.returncode == 0 else "Unknown"
                    
                    return {
                        "available": True,
                        "gpu_count": gpu_count,
                        "devices": devices,
                        "total_memory_gb": sum(d["memory_total"] for d in devices) / 1024,
                        "used_memory_gb": sum(d["memory_used"] for d in devices) / 1024,
                        "utilization": sum(d["utilization"] for d in devices) / len(devices),
                        "temperature": sum(d["temperature"] for d in devices) / len(devices),
                        "driver_version": driver_version,
                        "detection_method": "nvidia-smi"
                    }
    except Exception as e:
        logger.debug(f"nvidia-smi detection failed: {e}")
    
    # Fallback to pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        if device_count == 0:
            return {"available": False, "message": "No NVIDIA GPUs found"}
        
        devices = []
        total_memory = 0
        used_memory = 0
        total_utilization = 0
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            device_info = {
                "id": i,
                "name": name,
                "memory_total": mem_info.total // (1024**2),  # MB
                "memory_used": mem_info.used // (1024**2),   # MB
                "utilization": utilization.gpu,
                "temperature": temperature
            }
            devices.append(device_info)
            
            total_memory += mem_info.total
            used_memory += mem_info.used
            total_utilization += utilization.gpu
        
        return {
            "available": True,
            "gpu_count": device_count,
            "devices": devices,
            "total_memory_gb": total_memory // (1024**3),
            "used_memory_gb": used_memory // (1024**3),
            "utilization": total_utilization / device_count,
            "temperature": sum(d["temperature"] for d in devices) / device_count,
            "driver_version": pynvml.nvmlSystemGetDriverVersion().decode('utf-8'),
            "detection_method": "pynvml"
        }
            
        except ImportError:
            # Fallback to GPUtil
            if HAS_GPUTIL:
                gpus = GPUtil.getGPUs()
                if not gpus:
                    return {"available": False, "message": "No GPUs detected"}
                
                devices = []
                for gpu in gpus:
                    devices.append({
                        "id": gpu.id,
                        "name": gpu.name,
                        "memory_total": gpu.memoryTotal,
                        "memory_used": gpu.memoryUsed,
                        "utilization": gpu.load * 100,
                        "temperature": gpu.temperature
                    })
                
                return {
                    "available": True,
                    "gpu_count": len(gpus),
                    "devices": devices,
                    "total_memory_gb": sum(gpu.memoryTotal for gpu in gpus) / 1024,
                    "used_memory_gb": sum(gpu.memoryUsed for gpu in gpus) / 1024,
                    "utilization": sum(gpu.load for gpu in gpus) / len(gpus) * 100,
                    "temperature": sum(gpu.temperature for gpu in gpus) / len(gpus)
                }
            else:
                # Final fallback: check PyTorch CUDA
                try:
                    import torch
                    if torch.cuda.is_available():
                        device_count = torch.cuda.device_count()
                        devices = []
                        for i in range(device_count):
                            devices.append({
                                "id": i,
                                "name": torch.cuda.get_device_name(i),
                                "memory_total": torch.cuda.get_device_properties(i).total_memory // (1024**2),
                                "memory_used": torch.cuda.memory_allocated(i) // (1024**2),
                                "utilization": 0,  # PyTorch doesn't provide utilization
                                "temperature": 0   # PyTorch doesn't provide temperature
                            })
                        return {
                            "available": True,
                            "gpu_count": device_count,
                            "devices": devices,
                            "total_memory_gb": sum(d["memory_total"] for d in devices) / 1024,
                            "used_memory_gb": sum(d["memory_used"] for d in devices) / 1024,
                            "utilization": 0,
                            "temperature": 0,
                            "detection_method": "pytorch",
                            "note": "Limited info available via PyTorch"
                        }
                except ImportError:
                    pass
                
                return {"available": False, "message": "GPU monitoring libraries not available"}
                
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
        return {"available": False, "error": str(e)}

def get_database_info():
    """Get database health information"""
    try:
        from facecv.database import test_database_availability
        
        availability = test_database_availability()
        
        # Check default database
        try:
            from facecv.database import get_default_database
            db = get_default_database()
            face_count = db.get_face_count()
            
            return {
                "status": "healthy",
                "availability": availability,
                "face_count": face_count,
                "default_db": type(db).__name__,
                "last_check": datetime.now().isoformat()
            }
        except Exception as db_error:
            return {
                "status": "error",
                "availability": availability,
                "error": str(db_error),
                "last_check": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        return {
            "status": "error",
            "error": str(e),
            "last_check": datetime.now().isoformat()
        }

@router.get("/comprehensive",
    response_model=ComprehensiveHealth,
    summary="获取综合健康报告",
    description="""获取系统的综合健康状态报告。
    
    包含所有系统组件的状态检查：
    - CPU和内存使用情况
    - 磁盘空间状态
    - GPU状态（如果可用）
    - 数据库连接状态
    - 模型加载状态
    
    **健康状态级别：**
    - healthy: 所有系统正常
    - warning: 有轻微问题但不影响功能
    - critical: 有严重问题需要立即处理
    """)
async def get_comprehensive_health():
    try:
        # Collect all metrics
        cpu_info = get_cpu_info()
        memory_info = get_memory_info()
        disk_info = get_disk_info()
        gpu_info = get_gpu_info()
        database_info = get_database_info()
        
        metrics = SystemMetrics(
            cpu=cpu_info,
            memory=memory_info,
            disk=disk_info,
            gpu=gpu_info,
            database=database_info,
            models={"loaded_count": 0}  # Would check actual loaded models
        )
        
        # Analyze health status
        issues = []
        warnings = []
        recommendations = []
        
        # Check CPU
        if cpu_info.get("usage", 0) > 95:
            issues.append("CPU usage critically high")
        elif cpu_info.get("usage", 0) > 80:
            warnings.append("CPU usage high")
            recommendations.append("Consider scaling up CPU resources")
        
        # Check Memory
        if memory_info.get("usage_percent", 0) > 95:
            issues.append("Memory usage critically high")
        elif memory_info.get("usage_percent", 0) > 80:
            warnings.append("Memory usage high")
            recommendations.append("Consider adding more RAM")
        
        # Check Disk
        if disk_info.get("usage_percent", 0) > 95:
            issues.append("Disk space critically low")
        elif disk_info.get("usage_percent", 0) > 80:
            warnings.append("Disk space running low")
            recommendations.append("Clean up disk space or add storage")
        
        # Check Database
        if database_info.get("status") == "error":
            issues.append("Database connection failed")
            recommendations.append("Check database configuration and connectivity")
        
        # Check GPU (if available)
        if gpu_info.get("available") and gpu_info.get("utilization", 0) > 95:
            warnings.append("GPU utilization very high")
        
        # Determine overall status
        if issues:
            status = "critical"
            healthy = False
        elif warnings:
            status = "warning"
            healthy = True
        else:
            status = "healthy"
            healthy = True
        
        return ComprehensiveHealth(
            healthy=healthy,
            status=status,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            metrics=metrics,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting comprehensive health: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get health status: {str(e)}"
        )

@router.get("/cpu",
    summary="获取CPU健康状态",
    description="""获取CPU的详细使用情况和性能指标。
    
    包括CPU使用率、核心数量、频率和负载平均值。
    """)
async def get_cpu_health():
    try:
        return get_cpu_info()
    except Exception as e:
        logger.error(f"Error getting CPU health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/memory",
    summary="获取内存健康状态",
    description="""获取系统内存和交换空间的使用情况。
    
    包括物理内存和虚拟内存的详细统计信息。
    """)
async def get_memory_health():
    try:
        return get_memory_info()
    except Exception as e:
        logger.error(f"Error getting memory health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/disk",
    summary="获取磁盘健康状态", 
    description="""获取磁盘空间使用情况。
    
    显示总容量、已用空间和可用空间。
    """)
async def get_disk_health():
    try:
        return get_disk_info()
    except Exception as e:
        logger.error(f"Error getting disk health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/gpu",
    summary="获取GPU健康状态",
    description="""获取GPU使用情况和性能指标。
    
    如果系统有GPU，返回GPU使用率、内存、温度等信息。
    如果没有GPU，返回相应说明。
    """)
async def get_gpu_health():
    try:
        return get_gpu_info()
    except Exception as e:
        logger.error(f"Error getting GPU health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/database",
    summary="获取数据库健康状态",
    description="""检查数据库连接和状态。
    
    验证数据库可用性并返回连接统计信息。
    """)
async def get_database_health():
    try:
        db_info = get_database_info()
        
        if db_info.get("status") == "error":
            raise HTTPException(
                status_code=503,
                detail=f"Database health check failed: {db_info.get('error', 'Unknown error')}"
            )
        
        return db_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting database health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance",
    summary="获取性能指标",
    description="""获取系统和服务的性能统计信息。
    
    包括请求处理时间、模型推理性能和系统资源使用趋势。
    """)
async def get_performance_metrics():
    try:
        # Collect performance data
        performance_data = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_usage": get_cpu_info().get("usage", 0),
                "memory_usage": get_memory_info().get("usage_percent", 0),
                "disk_usage": get_disk_info().get("usage_percent", 0)
            },
            "requests": {
                "total_requests": 0,  # Would track actual request counts
                "requests_per_minute": 0,
                "avg_response_time": 0.0,
                "p95_response_time": 0.0,
                "error_rate": 0.0
            },
            "models": {
                "total_inferences": 0,  # Would track actual inference counts
                "avg_inference_time": 0.0,
                "loaded_models": 0
            }
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))