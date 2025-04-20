import os
import sys
import logging
import soundfile as sf
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any

# 将项目根目录添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入所需模块
from webpage.musicgen_service import MusicGenService, create_service, EMOTION_PRESET_PARAMS

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 测试提示词模板
TEST_PROMPTS = {
    "anxiety": [
        "Anxious instrumental music with racing heart beats and scattered flute melodies, creating a tense atmosphere",
        "Fast-paced piano with trembling strings, reflecting nervousness and worry"
    ],
    "depression": [
        "Melancholic piano melody with slow tempo and minor chords, expressing deep sadness",
        "Low atmospheric drone with sparse, distant piano notes that convey emptiness and grief"
    ],
    "insomnia": [
        "Ambient music with subtle electronic textures, quiet but with an underlying restlessness",
        "Minimal compositions with clock-like ticking and gentle synthesizers that evoke late night wakefulness"
    ],
    "stress": [
        "Intense orchestral music with rapid drumbeats and dissonant string sections",
        "Chaotic piano with urgent rhythms and building tension throughout"
    ],
    "calm": [
        "Gentle flowing piano with soft synthesizer pads creating a peaceful atmosphere",
        "Natural ambient sounds mixed with slow guitar arpeggios and gentle wind chimes"
    ],
    "distraction": [
        "Scattered musical fragments with sudden changes in instruments and tempo",
        "Music with multiple competing melodies and rhythms that create a sense of confusion"
    ],
    "pain": [
        "Emotional cello melody with deep, resonant tones expressing physical discomfort",
        "Slow, heavy percussion with dissonant string sections that build and release tension"
    ],
    "ptsd": [
        "Dramatic shifts between quiet ambient sections and sudden intense orchestral bursts",
        "Haunting vocals with distant echoes and unpredictable percussive elements"
    ]
}

# 测试输出目录
TEST_OUTPUT_DIR = Path("static/test_emotion_audio")
ABSOLUTE_OUTPUT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / TEST_OUTPUT_DIR

def ensure_output_dir():
    """确保输出目录存在"""
    ABSOLUTE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {ABSOLUTE_OUTPUT_DIR}")

def prepare_audio_for_writing(audio_data):
    """准备音频数据以便写入文件"""
    # 确保音频数据是有效的numpy数组
    audio_data = np.asarray(audio_data)
    
    # 确保数据类型是float32（soundfile要求）
    audio_data = audio_data.astype(np.float32)
    
    # 处理numpy数组维度
    if audio_data.ndim == 3:
        audio_data = audio_data.squeeze(0)
    
    # 确保通道维度在正确的位置（soundfile期望 [samples, channels]）
    if audio_data.ndim == 2:
        # 如果第一维小于第二维，说明可能是 [channels, samples]，需要转置
        if audio_data.shape[0] < audio_data.shape[1]:
            audio_data = audio_data.T
    
    # 防止静默或无效数据
    if np.max(np.abs(audio_data)) < 1e-10:
        # 生成一个短的静音文件，而不是完全空
        silence = np.zeros((32000,), dtype=np.float32)  # 1秒静音 @32kHz
        silence[0] = 1e-4  # 添加一个微小的脉冲避免格式错误
        return silence
    
    return audio_data

def generate_test_audios(service: MusicGenService): 
    """为每个情绪标签生成测试音频"""
    # 遍历所有情绪标签
    for emotion, prompts in TEST_PROMPTS.items():
        logger.info(f"正在处理情绪标签: {emotion}")
        
        # 为每个情绪生成两个测试用例
        for i, prompt in enumerate(prompts):
            try:
                logger.info(f"生成测试用例 {i+1}: {prompt[:30]}...")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                test_case_id = f"{emotion}_test{i+1}_{timestamp}"
                
                # 1. 完整处理: 使用情绪预设参数和音频处理
                generate_full_processing(service, emotion, prompt, test_case_id)
                
                # 2 & 3. 先生成原始音频，然后一份不处理直接保存，另一份应用音频处理
                # 这样确保两个版本使用完全相同的初始音频
                generate_shared_audio_versions(service, emotion, prompt, test_case_id)
            except Exception as e:
                logger.error(f"处理测试用例时出错: {str(e)}")
                continue

def generate_full_processing(service: MusicGenService, emotion: str, prompt: str, test_id: str):
    """生成完整处理的音频: 使用情绪预设参数和音频处理"""
    try:
        # 应用情绪预设 - 安全地应用预设，即使有错误也继续
        try:
            service.apply_emotion_preset(emotion)
        except Exception as e:
            logger.warning(f"应用情绪预设时出错: {str(e)}")
        
        # 生成音频
        output_file = ABSOLUTE_OUTPUT_DIR / f"{test_id}_full_processing.wav"
        
        # 获取音频处理参数
        audio_params = service.get_audio_params_for_emotion(emotion)
        eq_params = {
            "low_shelf_gain": audio_params["low_shelf_gain"],
            "mid_gain": audio_params["mid_gain"],
            "high_shelf_gain": audio_params["high_shelf_gain"]
        }
        target_db = audio_params["target_db"]
        limiter_threshold = audio_params["limiter_threshold"]
        
        # 使用模型生成音频
        output = service.model.generate(
            descriptions=[prompt],
            progress=True
        )
        
        # 处理音频数据
        audio_data = output[0].cpu().numpy()
        
        # 准备音频数据用于写入
        processed_audio = prepare_audio_for_writing(audio_data)
        
        # 应用音频处理
        try:
            processed_audio = service.audio_processor.process_audio(
                processed_audio,
                normalize=True,
                target_db=target_db,
                eq_params=eq_params
            )
            
            # 应用限幅器
            processed_audio = service.audio_processor.apply_limiter(processed_audio, limiter_threshold)
        except Exception as e:
            logger.warning(f"音频处理失败，使用原始音频: {str(e)}")
        
        # 保存处理后的音频
        sf.write(
            str(output_file),
            processed_audio,
            samplerate=service.config.sample_rate
        )
        
        logger.info(f"完整处理音频已保存: {output_file}")
    except Exception as e:
        logger.error(f"生成完整处理音频失败: {str(e)}")

def generate_shared_audio_versions(service: MusicGenService, emotion: str, prompt: str, test_id: str):
    """生成共享同一个初始音频的两个版本：无处理版本和仅音频处理版本"""
    try:
        # 重置模型参数为默认值
        service.model.set_generation_params(
            use_sampling=service.config.use_sampling,
            top_k=service.config.top_k,
            temperature=service.config.temperature,
            duration=service.config.duration,
            cfg_coef=service.config.cfg_coef
        )
        
        # 定义输出文件路径
        no_processing_file = ABSOLUTE_OUTPUT_DIR / f"{test_id}_no_processing.wav"
        audio_processing_file = ABSOLUTE_OUTPUT_DIR / f"{test_id}_audio_processing_only.wav"
        
        # 使用模型生成音频（只生成一次）
        logger.info(f"生成共享的初始音频...")
        output = service.model.generate(
            descriptions=[prompt],
            progress=True
        )
        
        # 处理音频数据
        audio_data = output[0].cpu().numpy()
        
        # 准备音频数据用于写入
        original_audio = prepare_audio_for_writing(audio_data)
        
        # 1. 保存无处理版本
        sf.write(
            str(no_processing_file),
            original_audio,
            samplerate=service.config.sample_rate
        )
        logger.info(f"无处理音频已保存: {no_processing_file}")
        
        # 2. 应用音频处理并保存
        try:
            # 获取音频处理参数
            audio_params = service.get_audio_params_for_emotion(emotion)
            eq_params = {
                "low_shelf_gain": audio_params["low_shelf_gain"],
                "mid_gain": audio_params["mid_gain"],
                "high_shelf_gain": audio_params["high_shelf_gain"]
            }
            target_db = audio_params["target_db"]
            limiter_threshold = audio_params["limiter_threshold"]
            
            # 复制原始音频以避免修改原始数据
            processed_audio = np.copy(original_audio)
            
            # 应用音频处理
            processed_audio = service.audio_processor.process_audio(
                processed_audio,
                normalize=True,
                target_db=target_db,
                eq_params=eq_params
            )
            
            # 应用限幅器
            processed_audio = service.audio_processor.apply_limiter(processed_audio, limiter_threshold)
            
            # 保存处理后的音频
            sf.write(
                str(audio_processing_file),
                processed_audio,
                samplerate=service.config.sample_rate
            )
            logger.info(f"仅音频处理音频已保存: {audio_processing_file}")
        except Exception as e:
            logger.error(f"音频处理失败: {str(e)}")
    except Exception as e:
        logger.error(f"生成共享音频版本失败: {str(e)}")

def create_index_html():
    """创建一个HTML页面用于查看和比较生成的音频样本"""
    try:
        html_path = ABSOLUTE_OUTPUT_DIR / "index.html"
        
        # 获取所有生成的wav文件
        audio_files = sorted(list(ABSOLUTE_OUTPUT_DIR.glob("*.wav")))
        
        if not audio_files:
            logger.warning("没有找到音频文件，无法创建索引页面")
            return
        
        # 按情绪标签分组
        emotions = {}
        for file in audio_files:
            try:
                parts = file.stem.split('_')
                emotion = parts[0]
                test_case = parts[1]
                processing = '_'.join(parts[3:])
                
                if emotion not in emotions:
                    emotions[emotion] = {}
                
                if test_case not in emotions[emotion]:
                    emotions[emotion][test_case] = []
                    
                emotions[emotion][test_case].append({
                    'file': file.name,
                    'processing': processing
                })
            except IndexError:
                logger.warning(f"跳过格式不符合的文件: {file.name}")
                continue
        
        # 生成HTML内容
        html_content = """
        <!DOCTYPE html>
        <html lang="zh">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>情绪音频测试样本</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #666; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
                h3 { color: #777; margin-top: 20px; }
                .audio-group { margin-bottom: 30px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }
                .processing-method { margin-bottom: 15px; }
                audio { width: 100%; margin-top: 5px; }
                .processing-label { font-weight: bold; margin-bottom: 5px; }
                .test-case { margin-bottom: 20px; padding: 10px; background-color: #eee; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>情绪音频测试样本</h1>
            <p>此页面展示了不同情绪标签下生成的音频样本，包括三种不同的处理方式：</p>
            <ul>
                <li><strong>完整处理</strong>: 使用情绪参数设置模型生成，再应用情绪音频处理</li>
                <li><strong>仅音频处理</strong>: 使用默认模型参数生成，然后应用情绪音频处理</li>
                <li><strong>无处理</strong>: 使用默认模型参数生成，不应用任何音频处理</li>
            </ul>
        """
        
        # 添加每个情绪的测试用例
        for emotion in sorted(emotions.keys()):
            html_content += f'<h2>情绪: {emotion}</h2>\n'
            
            for test_case, samples in emotions[emotion].items():
                html_content += f'<div class="test-case"><h3>测试用例: {test_case}</h3>\n'
                
                # 按处理方式分组
                full_processing = next((s for s in samples if 'full_processing' in s['processing']), None)
                audio_processing = next((s for s in samples if 'audio_processing_only' in s['processing']), None)
                no_processing = next((s for s in samples if 'no_processing' in s['processing']), None)
                
                if full_processing:
                    html_content += f"""
                    <div class="processing-method">
                        <div class="processing-label">完整处理:</div>
                        <audio controls src="{full_processing['file']}"></audio>
                    </div>
                    """
                    
                if audio_processing:
                    html_content += f"""
                    <div class="processing-method">
                        <div class="processing-label">仅音频处理:</div>
                        <audio controls src="{audio_processing['file']}"></audio>
                    </div>
                    """
                    
                if no_processing:
                    html_content += f"""
                    <div class="processing-method">
                        <div class="processing-label">无处理:</div>
                        <audio controls src="{no_processing['file']}"></audio>
                    </div>
                    """
                    
                html_content += '</div>\n'
        
        html_content += """
        </body>
        </html>
        """
        
        # 写入HTML文件
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"索引HTML页面已创建: {html_path}")
    except Exception as e:
        logger.error(f"创建索引HTML页面失败: {str(e)}")

def main():
    """主函数"""
    try:
        # 确保输出目录存在
        ensure_output_dir()
        
        # 创建服务
        service = create_service(
            model_name='facebook/musicgen-small',  # 使用小模型
            duration=3,  # 设置更短的时长以便快速生成
            output_dir=str(ABSOLUTE_OUTPUT_DIR)
        )
        
        if service and service.model:
            logger.info("开始生成测试音频...")
            generate_test_audios(service)
            create_index_html()
            logger.info("测试音频生成完成!")
        else:
            logger.error("服务初始化失败!")
    except Exception as e:
        logger.error(f"执行测试脚本时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 