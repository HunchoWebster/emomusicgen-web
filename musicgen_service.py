import numpy as np
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
import time
import os
import json
import re
from openai import OpenAI
from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import logging
import subprocess
from datetime import datetime
import warnings
from .audio_processor import AudioProcessor
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入MERT补丁
try:
    from mert_patch import patch_mert_config
    # 应用补丁
    patch_mert_config()
except Exception as e:
    print(f"MERT补丁加载失败，可能会影响模型初始化: {str(e)}")

# 忽略特定的警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
# 忽略 xformers 相关的警告
os.environ['XFORMERS_MORE_DETAILS'] = '0'

@dataclass
class MusicGenConfig:
    """音乐生成配置类"""
    model_name: str = 'facebook/musicgen-small'  # 使用small模型作为默认模型
    use_diffusion: bool = False
    duration: int = 10
    top_k: int = 250
    use_sampling: bool = True
    temperature: float = 1.0  # 添加温度参数
    cfg_coef: float = 3.0  # 添加CFG系数
    sample_rate: int = 32000
    output_dir: str = './generated_music'
    # 风格控制参数
    style_params: dict = None

    def __post_init__(self):
        """确保duration参数被正确设置"""
        if not hasattr(self, 'duration') or self.duration is None:
            self.duration = 10
        if self.style_params is None:
            self.style_params = {
                'eval_q': 3,
                'excerpt_length': 3.0,
                'ds_factor': None
            }

# 添加情绪配置映射
EMOTION_PRESET_PARAMS = {
    "anxiety": {
        "model_params": {
            "temperature": 0.9,
            "top_k": 250,
            "cfg_coef": 3.5
        },
        "style_params": {
            "eval_q": 2,
            "excerpt_length": 4.0,
            "ds_factor": 4
        },
        "audio_params": {
            "low_shelf_gain": -2.0,
            "mid_gain": 0.0,
            "high_shelf_gain": -3.0,
            "target_db": -18.0,
            "limiter_threshold": -1.5
        }
    },
    "depression": {
        "model_params": {
            "temperature": 0.85,
            "top_k": 200,
            "cfg_coef": 4.0
        },
        "style_params": {
            "eval_q": 2,
            "excerpt_length": 5.0,
            "ds_factor": 6
        },
        "audio_params": {
            "low_shelf_gain": 1.0,
            "mid_gain": -1.0,
            "high_shelf_gain": -2.0,
            "target_db": -16.0,
            "limiter_threshold": -2.0
        }
    },
    "insomnia": {
        "model_params": {
            "temperature": 0.7,
            "top_k": 150,
            "cfg_coef": 3.0
        },
        "style_params": {
            "eval_q": 3,
            "excerpt_length": 3.0,
            "ds_factor": 4
        },
        "audio_params": {
            "low_shelf_gain": 2.0,
            "mid_gain": -2.0,
            "high_shelf_gain": -4.0,
            "target_db": -20.0,
            "limiter_threshold": -1.0
        }
    },
    "stress": {
        "model_params": {
            "temperature": 0.95,
            "top_k": 220,
            "cfg_coef": 3.8
        },
        "style_params": {
            "eval_q": 2,
            "excerpt_length": 4.0,
            "ds_factor": 5
        },
        "audio_params": {
            "low_shelf_gain": 0.0,
            "mid_gain": -1.0,
            "high_shelf_gain": -2.0,
            "target_db": -18.0,
            "limiter_threshold": -1.5
        }
    },
    "distraction": {
        "model_params": {
            "temperature": 0.8,
            "top_k": 180,
            "cfg_coef": 4.2
        },
        "style_params": {
            "eval_q": 3,
            "excerpt_length": 2.5,
            "ds_factor": 3
        },
        "audio_params": {
            "low_shelf_gain": -1.0,
            "mid_gain": 1.5,
            "high_shelf_gain": -1.0,
            "target_db": -16.0,
            "limiter_threshold": -2.0
        }
    },
    "pain": {
        "model_params": {
            "temperature": 0.75,
            "top_k": 200,
            "cfg_coef": 3.5
        },
        "style_params": {
            "eval_q": 2,
            "excerpt_length": 4.5,
            "ds_factor": 5
        },
        "audio_params": {
            "low_shelf_gain": 1.5,
            "mid_gain": -1.0,
            "high_shelf_gain": -3.0,
            "target_db": -17.0,
            "limiter_threshold": -1.8
        }
    },
    "trauma": {
        "model_params": {
            "temperature": 0.7,
            "top_k": 160,
            "cfg_coef": 4.5
        },
        "style_params": {
            "eval_q": 1,
            "excerpt_length": 5.0,
            "ds_factor": 6
        },
        "audio_params": {
            "low_shelf_gain": 0.5,
            "mid_gain": -0.5,
            "high_shelf_gain": -4.0,
            "target_db": -19.0,
            "limiter_threshold": -1.2
        }
    },
    "calm": {
        "model_params": {
            "temperature": 0.65,
            "top_k": 150,
            "cfg_coef": 3.0
        },
        "style_params": {
            "eval_q": 2,
            "excerpt_length": 4.0,
            "ds_factor": 3
        },
        "audio_params": {
            "low_shelf_gain": 0.0,
            "mid_gain": 0.0,
            "high_shelf_gain": -1.0,
            "target_db": -20.0,
            "limiter_threshold": -1.5
        }
    }
}

class MusicGenService:
    """音乐生成服务类"""
    
    def __init__(self, api_key: Optional[str] = None, config: MusicGenConfig = None, output_dir: str = './generated_music', progress_callback=None):
        """
        初始化音乐生成服务
        
        Args:
            api_key: 豆包API密钥（可选）
            config: MusicGenConfig配置对象
            output_dir: 输出目录
            progress_callback: 进度回调函数
        """
        self.api_key = api_key or os.getenv("ARK_API_KEY")
        self.config = config
        self.output_dir = output_dir
        self.progress_callback = progress_callback
        self.model = None
        self.mbd = None
        self.audio_processor = AudioProcessor(sample_rate=self.config.sample_rate)
        self._setup_logging()
        self.current_emotion_label = None
        
    def _setup_logging(self):
        """设置日志"""
        # 将日志级别改为 INFO 以显示详细信息
        logging.basicConfig(
            level=logging.INFO,  # 从 ERROR 改为 INFO
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 修改其他库的日志级别，但保持主要日志为 INFO
        logging.getLogger('torch').setLevel(logging.ERROR)
        logging.getLogger('audiocraft').setLevel(logging.ERROR)
        logging.getLogger('transformers').setLevel(logging.ERROR)
        
    def initialize(self) -> bool:
        """
        初始化模型和必要组件
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info("正在初始化模型...")
            
            # 尝试不同的模型，如果指定的模型加载失败
            models_to_try = [self.config.model_name]
            
            # 如果请求的是style模型，添加备用模型
            if 'musicgen-small' in self.config.model_name:
                models_to_try.extend(['facebook/musicgen-small', 'facebook/musicgen-medium'])
            
            # 尝试加载每个模型，直到成功
            exception = None
            for model_name in models_to_try:
                try:
                    self.logger.info(f"尝试加载模型: {model_name}")
                    
                    # 在加载模型之前为MERTConfig应用补丁
                    if 'musicgen-small' in model_name:
                        try:
                            from mert_patch import patch_mert_config
                            patch_mert_config()
                        except Exception as patch_err:
                            self.logger.warning(f"MERT补丁应用失败: {str(patch_err)}")
                    
                    # 加载模型
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    self.model = MusicGen.get_pretrained(model_name, device=device)
                    self.logger.info(f"成功加载模型: {model_name}")
                    break
                except Exception as e:
                    self.logger.warning(f"无法加载模型 {model_name}: {str(e)}")
                    exception = e
            
            # 如果所有模型都加载失败
            if self.model is None:
                if exception:
                    raise exception
                else:
                    raise RuntimeError("所有模型加载尝试均失败")
            
            # 如果成功加载了不同的模型，更新配置
            if self.model.name != self.config.model_name:
                self.logger.info(f"使用替代模型: {self.model.name} 而不是请求的 {self.config.model_name}")
                self.config.model_name = self.model.name
            
            # 加载扩散模型（如果需要）
            if self.config.use_diffusion:
                self.mbd = MultiBandDiffusion.get_mbd_musicgen()
            
            # 设置生成参数
            self.model.set_generation_params(
                use_sampling=self.config.use_sampling,
                top_k=self.config.top_k,
                temperature=self.config.temperature,
                duration=self.config.duration,
                cfg_coef=self.config.cfg_coef
            )
            
            # 如果是style模型，设置风格条件器参数
            if 'musicgen-small' in self.config.model_name:
                self._configure_style_conditioner()
            
            # 确保输出目录存在
            Path(self.output_dir).mkdir(exist_ok=True)
            
            self.logger.info("模型初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"模型初始化失败: {str(e)}")
            self.logger.error(f"错误类型: {type(e)}")
            import traceback
            self.logger.error(f"错误堆栈: {traceback.format_exc()}")
            return False
            
    def _configure_style_conditioner(self):
        """配置风格条件器参数"""
        try:
            # 对于musicgen-small模型，直接设置参数
            if 'musicgen-small' in self.config.model_name:
                params = self.config.style_params
                self.model.set_style_conditioner_params(
                    eval_q=params['eval_q'],
                    excerpt_length=params['excerpt_length'],
                    ds_factor=params['ds_factor']
                )
                self.logger.info(f"风格条件器参数已设置: {params}")
                return
                
            # 对于其他模型，检查是否有风格条件器
            # 更安全的检查方式
            if hasattr(self.model.lm, 'condition_provider') and \
               hasattr(self.model.lm.condition_provider, 'conditioners'):
                
                conditioners = self.model.lm.condition_provider.conditioners
                # 检查是否为ModuleDict类型且包含self_wav键
                if hasattr(conditioners, 'keys') and 'self_wav' in conditioners.keys():
                    params = self.config.style_params
                    self.model.set_style_conditioner_params(
                        eval_q=params['eval_q'],
                        excerpt_length=params['excerpt_length'],
                        ds_factor=params['ds_factor']
                    )
                    self.logger.info(f"风格条件器参数已设置: {params}")
                else:
                    self.logger.info(f"当前模型 {self.config.model_name} 的条件器中不包含self_wav，跳过风格参数设置")
            else:
                self.logger.info(f"当前模型 {self.config.model_name} 不支持风格条件器参数设置")
        except Exception as e:
            self.logger.error(f"设置风格条件器参数失败: {str(e)}")
            # 记录更详细的调试信息
            import traceback
            self.logger.debug(f"错误堆栈: {traceback.format_exc()}")
            if hasattr(self.model.lm, 'condition_provider'):
                if hasattr(self.model.lm.condition_provider, 'conditioners'):
                    self.logger.debug(f"条件器类型: {type(self.model.lm.condition_provider.conditioners)}")
                    if hasattr(self.model.lm.condition_provider.conditioners, 'keys'):
                        self.logger.debug(f"条件器键: {self.model.lm.condition_provider.conditioners.keys()}")
    
    def apply_emotion_preset(self, emotion_label: str) -> bool:
        """
        根据情绪标签应用预设参数
        
        Args:
            emotion_label: 情绪标签
            
        Returns:
            bool: 是否成功应用预设
        """
        if emotion_label not in EMOTION_PRESET_PARAMS:
            self.logger.error(f"未找到情绪标签的预设参数: {emotion_label}")
            return False
            
        try:
            preset = EMOTION_PRESET_PARAMS[emotion_label]
            
            # 更新模型参数
            model_params = preset['model_params']
            self.model.set_generation_params(
                use_sampling=self.config.use_sampling,
                top_k=model_params.get('top_k', self.config.top_k),
                temperature=model_params.get('temperature', self.config.temperature),
                duration=self.config.duration,  # 使用配置中的duration
                cfg_coef=model_params.get('cfg_coef', self.config.cfg_coef)
            )
            
            # 安全地设置风格条件器参数（如果存在的话）
            try:
                style_params = preset.get('style_params')
                if style_params and hasattr(self.model, 'set_style_conditioner_params'):
                    # 检查模型是否支持风格参数
                    has_style_conditioner = False
                    
                    # 检查musicgen-small模型
                    if 'musicgen-small' in self.config.model_name:
                        has_style_conditioner = True
                    # 检查其他模型的条件器
                    elif (hasattr(self.model.lm, 'condition_provider') and 
                          hasattr(self.model.lm.condition_provider, 'conditioners')):
                        conditioners = self.model.lm.condition_provider.conditioners
                        if hasattr(conditioners, 'keys') and 'self_wav' in conditioners.keys():
                            has_style_conditioner = True
                    
                    # 如果有风格条件器，则设置参数
                    if has_style_conditioner:
                        self.model.set_style_conditioner_params(
                            eval_q=style_params.get('eval_q', 3),
                            excerpt_length=style_params.get('excerpt_length', 3.0),
                            ds_factor=style_params.get('ds_factor', None)
                        )
                        self.logger.info(f"已应用风格条件器参数: {style_params}")
                    else:
                        self.logger.info(f"当前模型不支持风格条件器参数，跳过设置")
            except Exception as style_err:
                # 如果设置风格参数失败，只记录日志但不影响其他参数的应用
                self.logger.warning(f"应用风格条件器参数失败: {str(style_err)}")
            
            self.current_emotion_label = emotion_label
            self.logger.info(f"已应用情绪标签 '{emotion_label}' 的预设参数")
            return True
            
        except Exception as e:
            self.logger.error(f"应用情绪预设失败: {str(e)}")
            import traceback
            self.logger.error(f"错误堆栈: {traceback.format_exc()}")
            return False
    
    def create_prompt(self, emotion_text: str) -> Optional[dict]:
        """
        Generate music prompt based on emotional description
        
        Args:
            emotion_text: Emotional description text
            
        Returns:
            Optional[dict]: Generated prompt and emotion label
        """
        if not self.api_key:
            self.logger.error("API key not set")
            return None
            
        try:
            self.logger.info(f"开始调用 API，输入文本: {emotion_text}")
            
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://ark.cn-beijing.volces.com/api/v3"
            )
            
            # 记录完整的请求消息
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional music therapy prompt generator. Your task is to analyze the user's emotional description "
                        "and generate two parts:\n"
                        "1. Emotion Label: Identify the primary emotion type from the user's description\n"
                        "2. Music Prompt: Generate a detailed English music generation prompt\n\n"
                        "You must respond with ONLY a valid JSON object in the following format:\n"
                        "{\n"
                        '  "emotion_label": "emotion label",\n'
                        '  "music_prompt": "music generation prompt"\n'
                        "}\n\n"
                        "The emotion label must be exactly one of: anxiety, depression, insomnia, stress, distraction, pain, trauma, calm\n"
                        "The music prompt must:\n"
                        "- Be in English\n"
                        "- Include specific musical elements (rhythm, timbre, melodic features, etc.)\n"
                        "- Target emotional regulation goals\n"
                        "- Be concise and clear\n"
                        "- Must not exceed 130 characters\n\n"
                        "Example valid response:\n"
                        "{\n"
                        '  "emotion_label": "anxiety",\n'
                        '  "music_prompt": "Calm ambient music with gentle piano melodies, soft strings, and slow tempo around 60 BPM"\n'
                        "}"
                    )
                },
                {"role": "user", "content": emotion_text}
            ]
            
            self.logger.info("API 请求消息:")
            self.logger.info(f"System: {messages[0]['content']}")
            self.logger.info(f"User: {messages[1]['content']}")
            
            completion = client.chat.completions.create(
                model="ep-20250220214537-p7622",
                messages=messages
            )
            
            # 记录完整的 API 响应
            self.logger.info("API 完整响应:")
            self.logger.info(f"Model: {completion.model}")
            self.logger.info(f"Created: {completion.created}")
            self.logger.info(f"Usage: {completion.usage}")
            self.logger.info(f"Choices: {len(completion.choices)}")
            
            for i, choice in enumerate(completion.choices):
                self.logger.info(f"Choice {i}:")
                self.logger.info(f"  Index: {choice.index}")
                self.logger.info(f"  Message Role: {choice.message.role}")
                self.logger.info(f"  Message Content: {choice.message.content}")
                self.logger.info(f"  Finish Reason: {choice.finish_reason}")
            
            # 记录原始响应内容，包括所有可能的格式
            response = completion.choices[0].message.content.strip()
            self.logger.info("原始响应内容:")
            self.logger.info("=" * 50)
            self.logger.info(response)
            self.logger.info("=" * 50)
            
            # 尝试不同的响应格式
            self.logger.info("尝试解析响应...")
            
            # 1. 尝试直接解析为 JSON
            try:
                result = json.loads(response)
                self.logger.info("成功直接解析为 JSON")
            except json.JSONDecodeError:
                self.logger.info("直接解析 JSON 失败，尝试其他方法")
                
                # 2. 尝试查找 JSON 对象
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    response = json_match.group(0)
                    self.logger.info("成功提取 JSON 对象")
                    try:
                        result = json.loads(response)
                        self.logger.info("成功解析提取的 JSON 对象")
                    except json.JSONDecodeError as e:
                        self.logger.error(f"解析提取的 JSON 对象失败: {str(e)}")
                        return None
                else:
                    self.logger.warning("未找到 JSON 对象")
                    return None
            
            # 记录解析后的结果
            self.logger.info("解析结果:")
            self.logger.info(f"  Raw Result: {result}")
            
            # Validate required fields
            if 'emotion_label' not in result or 'music_prompt' not in result:
                self.logger.error("Missing required fields in JSON response")
                self.logger.error(f"Available fields: {list(result.keys())}")
                return None
            
            # Validate emotion label
            valid_emotions = {'anxiety', 'depression', 'insomnia', 'stress', 'distraction', 'pain', 'trauma', 'calm'}
            if result['emotion_label'] not in valid_emotions:
                self.logger.error(f"Invalid emotion label: {result['emotion_label']}")
                self.logger.error(f"Valid emotions: {valid_emotions}")
                return None
            
            # Validate music prompt
            if not isinstance(result['music_prompt'], str):
                self.logger.error("Music prompt must be a string")
                self.logger.error(f"Actual type: {type(result['music_prompt'])}")
                return None
            
            prompt_length = len(result['music_prompt'])
            if prompt_length > 150:
                self.logger.warning(f"Music prompt is too long ({prompt_length} chars), truncating to 150 chars")
                result['music_prompt'] = result['music_prompt'][:150]
            
            self.logger.info("最终结果:")
            self.logger.info(f"  Emotion Label: {result['emotion_label']}")
            self.logger.info(f"  Music Prompt: {result['music_prompt']}")
            self.logger.info(f"  Prompt Length: {len(result['music_prompt'])} chars")
            
            # 应用情绪预设参数
            self.apply_emotion_preset(result['emotion_label'])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prompt generation failed: {str(e)}")
            self.logger.error(f"Exception type: {type(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_audio_params_for_emotion(self, emotion_label: str = None) -> dict:
        """
        获取指定情绪标签的音频处理参数
        
        Args:
            emotion_label: 情绪标签，如果为None则使用当前情绪标签
            
        Returns:
            dict: 音频处理参数
        """
        label = emotion_label or self.current_emotion_label
        if not label or label not in EMOTION_PRESET_PARAMS:
            return {
                "low_shelf_gain": 0.0,
                "mid_gain": 0.0,
                "high_shelf_gain": 0.0,
                "target_db": -20.0,
                "limiter_threshold": -1.0
            }
        
        return EMOTION_PRESET_PARAMS[label]['audio_params']
    
    def generate_transition_music(self, from_emotion: str, to_emotion: str, 
                                  from_prompt: str, to_prompt: str, 
                                  transition_duration: int = 10) -> Tuple[bool, Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        生成情绪过渡音乐
        
        Args:
            from_emotion: 起始情绪标签
            to_emotion: 目标情绪标签
            from_prompt: 起始情绪提示词
            to_prompt: 目标情绪提示词
            transition_duration: 过渡音乐时长（秒）
            
        Returns:
            Tuple[bool, Optional[str], Optional[str], Optional[str], Optional[str]]: 
            (是否成功, 生成的音频文件路径, 原始音频文件路径, 起始情绪音频路径, 目标情绪音频路径)
        """
        if not self.model:
            self.logger.error("模型未初始化")
            return False, None, None, None, None
            
        try:
            # 确保transition_duration是整数
            transition_duration = int(transition_duration)
            
            # 计算重叠比例为总时长的25%
            overlap_ratio = 0.25
            overlap_seconds = transition_duration * overlap_ratio
            
            # 计算每段音乐的实际时长，确保最终时长为用户指定的时长
            # 两段生成的音频总时长 = 过渡总时长 + 重叠时长
            total_segments_duration = transition_duration + overlap_seconds
            
            # 计算每段音频时长
            first_segment_duration = int(total_segments_duration * 0.5)
            second_segment_duration = int(total_segments_duration * 0.5)
            
            self.logger.info(f"总时长: {transition_duration}秒, 第一段: {first_segment_duration}秒, 第二段: {second_segment_duration}秒, 重叠时长: {overlap_seconds}秒")
            
            # 生成唯一的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"transition_music_{timestamp}.wav")
            raw_output_file = os.path.join(self.output_dir, f"raw_transition_{timestamp}.wav")
            from_emotion_file = os.path.join(self.output_dir, f"from_emotion_{timestamp}.wav")
            to_emotion_file = os.path.join(self.output_dir, f"to_emotion_{timestamp}.wav")
            
            # 生成起始情绪音乐
            self.apply_emotion_preset(from_emotion)
            from_params = EMOTION_PRESET_PARAMS[from_emotion]['model_params']
            self.model.set_generation_params(
                use_sampling=self.config.use_sampling,
                top_k=from_params.get('top_k', self.config.top_k),
                temperature=from_params.get('temperature', self.config.temperature),
                duration=first_segment_duration,
                cfg_coef=from_params.get('cfg_coef', self.config.cfg_coef)
            )
            
            from_output = self.model.generate(
                descriptions=[from_prompt],
                progress=True
            )
            
            # 生成目标情绪音乐
            self.apply_emotion_preset(to_emotion)
            to_params = EMOTION_PRESET_PARAMS[to_emotion]['model_params']
            self.model.set_generation_params(
                use_sampling=self.config.use_sampling,
                top_k=to_params.get('top_k', self.config.top_k),
                temperature=to_params.get('temperature', self.config.temperature),
                duration=second_segment_duration,
                cfg_coef=to_params.get('cfg_coef', self.config.cfg_coef)
            )
            
            to_output = self.model.generate(
                descriptions=[to_prompt],
                progress=True
            )
            
            # 处理音频数据
            from_audio = from_output[0].cpu().numpy()
            to_audio = to_output[0].cpu().numpy()
            
            # 检查生成的音频是否有效
            self.logger.info(f"第一段音频形状: {from_audio.shape}, 第二段音频形状: {to_audio.shape}")
            self.logger.info(f"第一段音频最大值: {np.max(np.abs(from_audio))}, 第二段音频最大值: {np.max(np.abs(to_audio))}")
            
            # 确保音频数据是2D的 [channels, samples]
            if from_audio.ndim == 3:
                from_audio = from_audio.squeeze(0)
            if to_audio.ndim == 3:
                to_audio = to_audio.squeeze(0)
            
            # 确保通道维度在正确的位置
            if from_audio.ndim == 2 and from_audio.shape[1] < from_audio.shape[0]:
                from_audio = from_audio.T
            if to_audio.ndim == 2 and to_audio.shape[1] < to_audio.shape[0]:
                to_audio = to_audio.T
            
            # 确保音频数据是单声道的
            if from_audio.shape[0] > 1:
                from_audio = from_audio.mean(axis=0, keepdims=True)
            if to_audio.shape[0] > 1:
                to_audio = to_audio.mean(axis=0, keepdims=True)
            
            # 检查第二段音频是否含有有效数据
            if np.max(np.abs(to_audio)) < 0.01:
                self.logger.warning("第二段音频信号强度非常低，可能存在生成问题，重新生成")
                
                # 重新生成第二段音频
                to_output = self.model.generate(
                    descriptions=[to_prompt],
                    progress=True
                )
                to_audio = to_output[0].cpu().numpy()
                
                # 再次确保格式正确
                if to_audio.ndim == 3:
                    to_audio = to_audio.squeeze(0)
                if to_audio.ndim == 2 and to_audio.shape[1] < to_audio.shape[0]:
                    to_audio = to_audio.T
                if to_audio.shape[0] > 1:
                    to_audio = to_audio.mean(axis=0, keepdims=True)
                
                self.logger.info(f"重新生成的第二段音频最大值: {np.max(np.abs(to_audio))}")
            
            # 保存两段原始情绪音频
            # 为WAV写入准备音频数据 - 确保格式正确
            if from_audio.ndim == 2:
                from_audio_for_writing = from_audio.T if from_audio.shape[0] <= 2 else from_audio
            else:
                from_audio_for_writing = from_audio
                
            if to_audio.ndim == 2:
                to_audio_for_writing = to_audio.T if to_audio.shape[0] <= 2 else to_audio
            else:
                to_audio_for_writing = to_audio
            
            # 保存起始情绪和目标情绪的音频
            sf.write(
                str(from_emotion_file),
                from_audio_for_writing.astype(np.float32),
                samplerate=self.config.sample_rate
            )
            sf.write(
                str(to_emotion_file),
                to_audio_for_writing.astype(np.float32),
                samplerate=self.config.sample_rate
            )
            
            # 创建过渡音频 - 使用音频处理器创建平滑过渡
            transition_audio = self.audio_processor.create_transition(
                from_audio, 
                to_audio,
                overlap_seconds=overlap_seconds  # 使用计算的重叠时长
            )
            
            # 检查最终音频的时长（样本数）
            samples_per_second = self.config.sample_rate
            expected_samples = transition_duration * samples_per_second
            actual_samples = transition_audio.shape[1] if transition_audio.ndim == 2 else len(transition_audio)
            
            self.logger.info(f"期望样本数: {expected_samples}, 实际样本数: {actual_samples}")
            
            # 如果生成的音频长度与预期不符，进行调整
            if abs(actual_samples - expected_samples) > samples_per_second:  # 允许1秒的误差
                self.logger.warning(f"音频长度与预期不符，进行调整")
                if actual_samples > expected_samples:
                    # 截断过长的音频
                    if transition_audio.ndim == 2:
                        transition_audio = transition_audio[:, :expected_samples]
                    else:
                        transition_audio = transition_audio[:expected_samples]
                else:
                    # 填充过短的音频
                    padding_length = expected_samples - actual_samples
                    if transition_audio.ndim == 2:
                        transition_audio = np.pad(transition_audio, ((0, 0), (0, padding_length)), mode='constant')
                    else:
                        transition_audio = np.pad(transition_audio, (0, padding_length), mode='constant')
            
            # 为WAV写入准备音频数据 - 确保格式正确
            if transition_audio.ndim == 2:
                # 如果是多通道，转置为 [samples, channels] 格式
                if transition_audio.shape[0] <= 2:  # 通常通道数<=2
                    transition_audio_for_writing = transition_audio.T
                else:
                    # 如果第一维太大，可能已经是 [samples, channels]
                    transition_audio_for_writing = transition_audio
            else:
                # 如果是1D数组，保持不变
                transition_audio_for_writing = transition_audio
            
            # 保存原始音频 - 使用float32格式确保兼容性
            sf.write(
                str(raw_output_file),
                transition_audio_for_writing.astype(np.float32),
                samplerate=self.config.sample_rate
            )
            
            # 应用音频处理 - 使用起始和目标情绪参数的中间值
            from_audio_params = self.get_audio_params_for_emotion(from_emotion)
            to_audio_params = self.get_audio_params_for_emotion(to_emotion)
            
            # 计算中间参数
            transition_params = {
                "low_shelf_gain": (from_audio_params["low_shelf_gain"] + to_audio_params["low_shelf_gain"]) / 2,
                "mid_gain": (from_audio_params["mid_gain"] + to_audio_params["mid_gain"]) / 2,
                "high_shelf_gain": (from_audio_params["high_shelf_gain"] + to_audio_params["high_shelf_gain"]) / 2
            }
            
            target_db = (from_audio_params["target_db"] + to_audio_params["target_db"]) / 2
            limiter_threshold = (from_audio_params["limiter_threshold"] + to_audio_params["limiter_threshold"]) / 2
            
            # 应用处理
            processed_audio = self.audio_processor.process_audio(
                transition_audio,
                normalize=True,
                target_db=target_db,
                eq_params=transition_params
            )
            
            # 应用限幅器
            processed_audio = self.audio_processor.apply_limiter(processed_audio, limiter_threshold)
            
            # 为WAV写入准备处理后的音频数据
            if processed_audio.ndim == 2:
                # 如果是多通道，转置为 [samples, channels] 格式
                if processed_audio.shape[0] <= 2:  # 通常通道数<=2
                    processed_audio_for_writing = processed_audio.T
                else:
                    # 如果第一维太大，可能已经是 [samples, channels]
                    processed_audio_for_writing = processed_audio
            else:
                # 如果是1D数组，保持不变
                processed_audio_for_writing = processed_audio
            
            # 保存处理后的音频文件 - 使用float32格式确保兼容性
            sf.write(
                str(output_file),
                processed_audio_for_writing.astype(np.float32),
                samplerate=self.config.sample_rate
            )
            
            self.logger.info(f"过渡音乐生成成功，保存至: {output_file}")
            return True, output_file, raw_output_file, from_emotion_file, to_emotion_file
            
        except Exception as e:
            self.logger.error(f"过渡音乐生成失败: {str(e)}")
            # 记录更多的调试信息
            if 'transition_audio' in locals():
                self.logger.error(f"transition_audio shape: {transition_audio.shape}, dtype: {transition_audio.dtype}")
            if 'processed_audio' in locals():
                self.logger.error(f"processed_audio shape: {processed_audio.shape}, dtype: {processed_audio.dtype}")
            return False, None, None, None, None
    
    def generate_music(self, prompt: str, eq_params: Optional[Dict[str, float]] = None, target_db: float = -20.0, limiter_threshold: float = -1.0) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        生成音乐
        
        Args:
            prompt: 音乐生成提示词
            eq_params: 均衡器参数字典，包含 low_shelf_gain, mid_gain, high_shelf_gain
            target_db: 目标响度值（dB）
            limiter_threshold: 限幅阈值（dB）
            
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: (是否成功, 生成的音频文件路径, 原始音频文件路径)
        """
        if not self.model:
            self.logger.error("模型未初始化")
            return False, None, None
            
        try:
            # 生成唯一的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"generated_music_{timestamp}.wav")
            raw_output_file = os.path.join(self.output_dir, f"raw_music_{timestamp}.wav")
            
            # 如果未提供EQ参数但有情绪标签，则使用情绪预设的参数
            if eq_params is None and self.current_emotion_label:
                eq_params = self.get_audio_params_for_emotion()
                target_db = eq_params.get('target_db', target_db)
                limiter_threshold = eq_params.get('limiter_threshold', limiter_threshold)
                self.logger.info(f"使用情绪 '{self.current_emotion_label}' 的音频处理参数")
            
            # 使用原始的 MusicGen 模型生成音乐
            output = self.model.generate(
                descriptions=[prompt],
                progress=True,
                return_tokens=True
            )
            
            # 处理音频数据
            audio_data = output[0].cpu().numpy()
            
            if audio_data.ndim == 3:
                audio_data = audio_data.squeeze(0)
            
            if audio_data.ndim == 2 and audio_data.shape[0] < audio_data.shape[1]:
                audio_data = audio_data.T
            
            # 保存原始音频
            sf.write(
                str(raw_output_file),
                audio_data,
                samplerate=self.config.sample_rate
            )
            
            # 应用音频处理
            processed_audio = self.audio_processor.process_audio(
                audio_data,
                normalize=True,
                target_db=target_db,
                eq_params=eq_params
            )
            
            # 应用限幅器
            processed_audio = self.audio_processor.apply_limiter(processed_audio, limiter_threshold)
            
            # 保存处理后的音频文件
            sf.write(
                str(output_file),
                processed_audio,
                samplerate=self.config.sample_rate
            )
            
            self.logger.info(f"音乐生成成功，保存至: {output_file}")
            
            # 处理扩散解码器输出
            if self.config.use_diffusion and self.mbd:
                diffusion_path = os.path.join(self.output_dir, f"generated_music_diffusion_{timestamp}.wav")
                raw_diffusion_path = os.path.join(self.output_dir, f"raw_music_diffusion_{timestamp}.wav")
                out_diffusion = self.mbd.tokens_to_wav(output[1])
                # 保存原始扩散解码器输出
                out_diffusion_np = out_diffusion.cpu().numpy()
                sf.write(
                    str(raw_diffusion_path),
                    out_diffusion_np,
                    samplerate=self.config.sample_rate
                )
                # 对扩散解码器输出也应用相同的音频处理
                out_diffusion = self.audio_processor.process_audio(
                    out_diffusion_np,
                    normalize=True,
                    target_db=-14.0,
                    eq_params=eq_params
                )
                sf.write(
                    str(diffusion_path),
                    out_diffusion,
                    samplerate=self.config.sample_rate
                )
                self.logger.info(f"扩散解码器输出保存至: {diffusion_path}")
            
            return True, output_file, raw_output_file

        except Exception as e:
            self.logger.error(f"音乐生成失败: {str(e)}")
            return False, None, None
            
    def process_emotion(self, emotion_text: str) -> Tuple[bool, Optional[dict], Optional[str]]:
        """
        处理情绪输入并生成音乐的完整流程
        
        Args:
            emotion_text: 情绪描述文本
            
        Returns:
            Tuple[bool, Optional[dict], Optional[str]]: (是否成功, 生成的结果(prompt和emotion_label), 音频文件路径)
        """
        # 生成提示词和情绪标签
        result = self.create_prompt(emotion_text)
        if not result:
            return False, None, None
            
        # 生成音乐
        success, audio_path, raw_audio_path = self.generate_music(result['music_prompt'])
        return success, result, audio_path

def create_service(api_key: Optional[str] = None, model_name: str = 'facebook/musicgen-small', duration: int = 10, use_sampling: bool = True, top_k: int = 250, temperature: float = 1.0, cfg_coef: float = 3.0, output_dir: str = './generated_music', progress_callback=None) -> MusicGenService:
    """
    创建并初始化音乐生成服务的便捷函数
    
    Args:
        api_key: 豆包API密钥（可选）
        model_name: 模型名称
        duration: 生成音乐的时长（秒）
        use_sampling: 是否使用采样
        top_k: top-k采样参数
        temperature: 温度参数
        cfg_coef: CFG系数
        output_dir: 输出目录
        progress_callback: 进度回调函数
        
    Returns:
        MusicGenService: 初始化好的服务实例
    """
    config = MusicGenConfig(
        model_name=model_name,
        duration=duration,
        use_sampling=use_sampling,
        top_k=top_k,
        temperature=temperature,
        cfg_coef=cfg_coef,
        output_dir=output_dir
    )
    service = MusicGenService(api_key, config, output_dir, progress_callback)
    service.initialize()
    return service 