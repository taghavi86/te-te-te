"""
Configuration Management
Loads and validates configuration from config.yaml
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class PoseConfig(BaseModel):
    """Pose estimation configuration."""
    model_type: str = "rtmpose"
    confidence_threshold: float = 0.5
    kp_threshold: float = 0.3
    model_path: str = "models/rtmpose.pth"
    config_path: str = "models/rtmpose_config.py"


class DetectionConfig(BaseModel):
    """Person detection configuration."""
    model_type: str = "rtmdet"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    model_path: str = "models/rtmdet.pth"
    max_persons: int = 5


class TrackingConfig(BaseModel):
    """Tracking configuration."""
    method: str = "bytetrack"
    track_buffer: int = 30
    match_threshold: float = 0.8
    confidence_threshold: float = 0.5


class SmoothingConfig(BaseModel):
    """Temporal smoothing configuration."""
    enabled: bool = True
    method: str = "savitzky_golay"
    window_length: int = 7
    polyorder: int = 2


class DTWConfig(BaseModel):
    """Dynamic Time Warping configuration."""
    enabled: bool = True
    window_size: int = 20
    use_multivariate: bool = True
    normalize: bool = True
    feature_weights: Dict[str, float] = Field(default_factory=lambda: {
        "elbow_angle": 1.3,
        "shoulder_angle": 1.5,
        "hip_rotation": 1.4,
        "wrist_angle": 1.0,
        "knee_angle": 0.9,
        "trunk_angle": 1.2,
        "wrist_velocity": 1.1,
        "elbow_velocity": 1.0,
        "shoulder_velocity": 1.3
    })


class BiomechanicsConfig(BaseModel):
    """Biomechanics analysis configuration."""
    coordinate_system: str = "body_relative"
    scale_reference: str = "shoulder_width"
    handedness: str = "auto"


class LLMConfig(BaseModel):
    """LLM configuration for LM Studio."""
    enabled: bool = True
    endpoint: str = "http://localhost:1234/v1"
    model: str = "qwen3.5-9b"
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: int = 120
    retry_attempts: int = 3
    streaming: bool = True


class GPUConfig(BaseModel):
    """GPU configuration."""
    use_cuda: bool = True
    device: str = "cuda:0"
    allow_tf32: bool = True
    benchmark: bool = True


class CacheConfig(BaseModel):
    """Cache configuration."""
    enabled: bool = True
    directory: str = "cache"
    use_video_hash: bool = True
    hash_algorithm: str = "sha256"


class UIConfig(BaseModel):
    """UI configuration."""
    theme: str = "dark"
    language: str = "en"
    show_fps: bool = True
    auto_save: bool = True
    screenshot_format: str = "png"


class ConfigModel(BaseSettings):
    """Main configuration model."""
    
    model_config = SettingsConfigDict(
        env_prefix='TTANA_',
        env_file='.env',
        extra='ignore'
    )
    
    # Sub-configurations
    pose: PoseConfig = Field(default_factory=PoseConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    dtw: DTWConfig = Field(default_factory=DTWConfig)
    biomechanics: BiomechanicsConfig = Field(default_factory=BiomechanicsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    
    # Direct fields
    database_path: str = "sessions/ttana.db"
    log_level: str = "INFO"
    offline_mode: bool = True


class Config:
    """Configuration manager singleton."""
    
    _instance: Optional['Config'] = None
    _config: Optional[ConfigModel] = None
    
    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> ConfigModel:
        """Load configuration from YAML file."""
        
        if cls._config is not None:
            return cls._config
        
        # Determine config path
        if config_path is None:
            # Try multiple locations
            possible_paths = [
                Path("config.yaml"),
                Path(__file__).parent.parent.parent / "config.yaml",
                Path.home() / ".ttana" / "config.yaml",
            ]
            
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
            
            if config_path is None:
                # Use default config
                cls._config = ConfigModel()
                return cls._config
        
        # Load YAML
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            # Create config model
            cls._config = ConfigModel(**yaml_data)
            
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            print("Using default configuration.")
            cls._config = ConfigModel()
        
        return cls._config
    
    @classmethod
    def get(cls) -> ConfigModel:
        """Get current configuration."""
        if cls._config is None:
            return cls.load()
        return cls._config
    
    @classmethod
    def reload(cls, config_path: Optional[str] = None) -> ConfigModel:
        """Reload configuration from file."""
        cls._config = None
        return cls.load(config_path)
