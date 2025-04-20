import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
import logging
from datetime import datetime
import soundfile as sf
import time
import numpy as np
import glob
import threading
import atexit

# 忽略警告
warnings.filterwarnings('ignore')
# 设置日志级别
logging.getLogger().setLevel(logging.ERROR)

from flask import Flask, render_template, request, jsonify, send_file, Response
from webpage.musicgen_service import create_service, MusicGenConfig, EMOTION_PRESET_PARAMS, MusicGenService
from pathlib import Path
import json
import re

# 创建Flask应用，指定静态文件目录
app = Flask(__name__, static_folder='static', static_url_path='/static')
# 设置 Flask 的日志级别
app.logger.setLevel(logging.ERROR)

os.environ["ARK_API_KEY"] = "9e5d6644-81ee-4360-baf1-db1816b2c344"

# 添加一个全局变量来存储生成进度
generation_progress = {
    'prompt_status': 0,    # 提示词生成进度
    'music_status': 0      # 音乐生成进度
}

# 添加一个全局变量来跟踪当前会话生成的音频文件
current_session_files = set()
# 添加一个锁来保护文件集合的访问
session_files_lock = threading.Lock()

# 全局服务实例和初始化标志
service = None
service_initialized = False

def initialize_service():
    """初始化服务的函数"""
    global service, service_initialized
    try:
        # 检查是否已经初始化
        if service_initialized:
            print("服务已经初始化")
            return
            
        print("开始初始化服务...")
        service = create_service_with_retry(max_retries=2)
        if service is None:
            raise Exception("无法初始化音乐生成服务")
        print(f"服务初始化完成，使用模型: {service.config.model_name}")
        service_initialized = True
            
    except Exception as e:
        print(f"服务初始化失败: {str(e)}")
        print("应用将继续启动，但音乐生成功能可能不可用")
        service = None

# 新增函数：清除临时音频文件
def cleanup_audio_files():
    """清除当前会话生成的所有音频文件"""
    try:
        global current_session_files
        with session_files_lock:
            if not current_session_files:
                return

            print(f"开始清理临时音频文件，数量: {len(current_session_files)}")
            for file_path in current_session_files:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"已删除临时文件: {file_path}")
                    except Exception as e:
                        print(f"删除文件时出错: {file_path}, 错误: {str(e)}")
            current_session_files.clear()
            print("临时音频文件清理完成")
    except Exception as e:
        print(f"清理临时音频文件时出错: {str(e)}")

# 注册应用退出时的清理函数
atexit.register(cleanup_audio_files)

# 创建音乐生成服务实例
def create_service_with_retry(max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"尝试初始化服务 (尝试 {attempt + 1}/{max_retries})...")
            
            # 首先尝试style模型
            service = create_service(
                api_key=os.getenv("ARK_API_KEY"),
                model_name='facebook/musicgen-small',  # 使用small模型
                duration=10,  # 确保时长为10秒
                use_sampling=True,
                top_k=250,
                temperature=1.0,
                cfg_coef=3.0,
                output_dir=os.path.join(os.path.dirname(__file__), 'static', 'generated_music'),
                progress_callback=None
            )
            print("服务创建成功，正在初始化模型...")
            if service.model is not None:
                print(f"模型已加载: {service.model.name}")
                return service
            print(f"初始化尝试 {attempt + 1} 失败 - 模型为空")
            
            # 如果style模型失败，尝试备用模型
            if attempt == 0:  # 只在第一次失败后尝试
                try:
                    print("尝试使用备用模型 'facebook/musicgen-small'...")
                    service = create_service(
                        api_key=os.getenv("ARK_API_KEY"),
                        model_name='facebook/musicgen-small',
                        duration=10,
                        use_sampling=True,
                        top_k=250,
                        temperature=1.0,
                        cfg_coef=3.0,
                        output_dir=os.path.join(os.path.dirname(__file__), 'static', 'generated_music'),
                        progress_callback=None
                    )
                    if service.model is not None:
                        print(f"备用模型加载成功: {service.model.name}")
                        return service
                except Exception as backup_err:
                    print(f"备用模型加载失败: {str(backup_err)}")
                
        except Exception as e:
            print(f"初始化尝试 {attempt + 1} 出错: {str(e)}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")
        
        if attempt < max_retries - 1:
            print(f"等待10秒后重试...")
            time.sleep(10)
    
    # 创建一个最小可用版本
    try:
        print("尝试创建最小可用版本...")
        config = MusicGenConfig(
            model_name='facebook/musicgen-small',
            duration=10,
            use_sampling=True,
            top_k=250,
            temperature=1.0,
            cfg_coef=3.0,
            output_dir=os.path.join(os.path.dirname(__file__), 'static', 'generated_music')
        )
        service = MusicGenService(os.getenv("ARK_API_KEY"), config, 
                                  os.path.join(os.path.dirname(__file__), 'static', 'generated_music'))
        print("最小服务创建成功")
        return service
    except Exception as e:
        print(f"最小服务创建失败: {str(e)}")
        raise Exception(f"服务初始化完全失败：{str(e)}")

# 初始化服务
initialize_service()

@app.before_request
def before_request():
    """在每个请求之前检查服务是否已初始化"""
    global service, service_initialized
    if not service_initialized:
        initialize_service()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transition')
def transition():
    return render_template('transition.html')

@app.route('/progress', methods=['GET'])
def get_progress():
    """获取生成进度"""
    return jsonify(generation_progress)

@app.route('/emotion_presets', methods=['GET'])
def get_emotion_presets():
    """获取情绪预设参数"""
    try:
        # 检查服务是否可用
        if service is None:
            return jsonify({'success': False, 'error': '服务不可用，请联系管理员'})
            
        return jsonify({
            'success': True,
            'presets': EMOTION_PRESET_PARAMS
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'获取情绪预设失败: {str(e)}'
        })

@app.route('/generate', methods=['POST'])
def generate_music():
    try:
        global generation_progress
        # 重置进度
        generation_progress = {
            'prompt_status': 0,
            'music_status': 0
        }
        
        print("收到生成请求")
        emotion_text = request.json.get('emotion_text')
        print(f"情绪文本: {emotion_text}")
        
        if not emotion_text:
            return jsonify({'success': False, 'error': '请输入情绪描述'})

        # 检查服务是否可用
        if service is None:
            return jsonify({'success': False, 'error': '音乐生成服务不可用，请联系管理员'})
            
        # 更新提示词生成进度
        generation_progress['prompt_status'] = 50
        print("开始生成提示词")
        result = service.create_prompt(emotion_text)
        generation_progress['prompt_status'] = 100
        
        if not result:
            return jsonify({'success': False, 'error': '提示词生成失败'})

        print(f"生成的提示词: {result['music_prompt']}")
        print(f"情绪标签: {result['emotion_label']}")

        # 返回提示词和情绪标签
        return jsonify({
            'success': True,
            'prompt': result['music_prompt'],
            'emotion_label': result['emotion_label'],
            'stage': 'prompt'  # 添加阶段标识
        })
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate_music', methods=['POST'])
def generate_music_audio():
    try:
        global generation_progress, current_session_files
        prompt = request.json.get('prompt')
        emotion_label = request.json.get('emotion_label')
        duration = request.json.get('duration', 10)  # 获取时长参数，默认10秒
        eq_params = request.json.get('eq_params', None)
        target_db = request.json.get('target_db', -20.0)
        limiter_threshold = request.json.get('limiter_threshold', -1.0)
        
        if not prompt:
            generation_progress['music_status'] = 0
            return jsonify({'success': False, 'error': '提示词不能为空'})

        # 检查服务是否可用
        if service is None:
            return jsonify({'success': False, 'error': '音乐生成服务不可用，请联系管理员'})

        # 更新服务配置中的duration参数
        service.config.duration = duration

        # 如果提供了情绪标签，则应用情绪预设
        if emotion_label and emotion_label in EMOTION_PRESET_PARAMS:
            service.apply_emotion_preset(emotion_label)
            
            # 如果未提供均衡器参数，则使用情绪预设的参数
            if eq_params is None:
                audio_params = service.get_audio_params_for_emotion(emotion_label)
                eq_params = {
                    'low_shelf_gain': audio_params['low_shelf_gain'],
                    'mid_gain': audio_params['mid_gain'],
                    'high_shelf_gain': audio_params['high_shelf_gain']
                }
                target_db = audio_params['target_db']
                limiter_threshold = audio_params['limiter_threshold']

        # 重置音乐生成进度
        generation_progress['music_status'] = 0
        print("开始生成音乐")
        success, audio_path, raw_audio_path = service.generate_music(
            prompt, 
            eq_params=eq_params,
            target_db=target_db,
            limiter_threshold=limiter_threshold
        )
        
        if not success:
            generation_progress['music_status'] = 0  # 重置进度
            return jsonify({'success': False, 'error': '音乐生成失败'})
            
        generation_progress['music_status'] = 100  # 只在成功时设置100%
        print(f"音乐生成结果: success={success}, path={audio_path}")
            
        audio_filename = os.path.basename(audio_path)
        raw_audio_filename = os.path.basename(raw_audio_path)
        relative_path = f'generated_music/{audio_filename}'
        raw_relative_path = f'generated_music/{raw_audio_filename}'
        
        # 将生成的文件路径添加到当前会话文件集合中
        with session_files_lock:
            current_session_files.add(audio_path)
            current_session_files.add(raw_audio_path)
        
        return jsonify({
            'success': True,
            'audio_path': relative_path,
            'raw_audio_path': raw_relative_path,
            'emotion_label': emotion_label,
            'stage': 'music'  # 添加阶段标识
        })
        
    except Exception as e:
        generation_progress['music_status'] = 0  # 重置进度
        print(f"发生错误: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate_transition', methods=['POST'])
def generate_transition():
    try:
        # 获取参数
        from_emotion = request.json.get('from_emotion')
        to_emotion = request.json.get('to_emotion')
        from_prompt = request.json.get('from_prompt')
        to_prompt = request.json.get('to_prompt')
        transition_duration = request.json.get('duration', 30)
        
        # 验证参数
        if not all([from_emotion, to_emotion, from_prompt, to_prompt]):
            return jsonify({'success': False, 'error': '缺少必要参数'})
        
        if from_emotion not in EMOTION_PRESET_PARAMS or to_emotion not in EMOTION_PRESET_PARAMS:
            return jsonify({'success': False, 'error': '无效的情绪标签'})
        
        # 检查服务是否可用
        if service is None:
            return jsonify({'success': False, 'error': '音乐生成服务不可用，请联系管理员'})
            
        # 重置进度
        generation_progress = {
            'prompt_status': 100,  # 已有提示词，直接设置为100%
            'music_status': 0
        }
        
        print(f"开始生成情绪过渡音乐: {from_emotion} -> {to_emotion}, 提示词: '{from_prompt}' -> '{to_prompt}', 时长: {transition_duration}秒")
        
        # 确保transition_duration是整数
        try:
            transition_duration = int(transition_duration)
            if transition_duration < 5:
                transition_duration = 5
                print(f"过渡时长太短，已调整为最小值5秒")
            elif transition_duration > 120:
                transition_duration = 120
                print(f"过渡时长太长，已调整为最大值120秒")
        except ValueError:
            transition_duration = 30
            print(f"过渡时长格式无效，已设置为默认值30秒")
        
        # 生成过渡音乐
        success, audio_path, raw_audio_path = service.generate_transition_music(
            from_emotion=from_emotion,
            to_emotion=to_emotion,
            from_prompt=from_prompt,
            to_prompt=to_prompt,
            transition_duration=transition_duration
        )
        
        if not success:
            generation_progress['music_status'] = 0  # 重置进度
            error_msg = "过渡音乐生成失败，请尝试使用不同的参数"
            print(error_msg)
            return jsonify({'success': False, 'error': error_msg})
            
        generation_progress['music_status'] = 100  # 只在成功时设置100%
        
        # 验证生成的文件是否存在
        if not os.path.exists(audio_path) or not os.path.exists(raw_audio_path):
            error_msg = "音频文件生成失败，文件不存在"
            print(f"错误: {error_msg}, 音频路径: {audio_path}, 原始音频路径: {raw_audio_path}")
            return jsonify({'success': False, 'error': error_msg})
        
        # 获取相对路径
        audio_filename = os.path.basename(audio_path)
        raw_audio_filename = os.path.basename(raw_audio_path)
        relative_path = f'generated_music/{audio_filename}'
        raw_relative_path = f'generated_music/{raw_audio_filename}'
        
        # 将生成的文件路径添加到当前会话文件集合中
        with session_files_lock:
            current_session_files.add(audio_path)
            current_session_files.add(raw_audio_path)
        
        print(f"过渡音乐生成成功: {relative_path}")
        
        return jsonify({
            'success': True,
            'audio_path': relative_path,
            'raw_audio_path': raw_relative_path,
            'from_emotion': from_emotion,
            'to_emotion': to_emotion,
            'stage': 'transition'
        })
        
    except Exception as e:
        generation_progress['music_status'] = 0  # 重置进度
        error_msg = f"过渡音乐生成错误: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()  # 打印详细堆栈跟踪
        return jsonify({'success': False, 'error': error_msg})

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # 获取原始音频路径和处理参数
        raw_audio_path = request.json.get('raw_audio_path')
        emotion_label = request.json.get('emotion_label')
        eq_params = request.json.get('eq_params')
        target_db = request.json.get('target_db', -20.0)
        limiter_threshold = request.json.get('limiter_threshold', -1.0)
        
        if not raw_audio_path:
            return jsonify({'success': False, 'error': '未找到原始音频'})
        
        # 检查服务是否可用
        if service is None:
            return jsonify({'success': False, 'error': '音频处理服务不可用，请联系管理员'})
            
        # 如果提供了情绪标签但没有均衡器参数，则使用情绪预设的参数    
        if emotion_label and emotion_label in EMOTION_PRESET_PARAMS and not eq_params:
            audio_params = service.get_audio_params_for_emotion(emotion_label)
            eq_params = {
                'low_shelf_gain': audio_params['low_shelf_gain'],
                'mid_gain': audio_params['mid_gain'],
                'high_shelf_gain': audio_params['high_shelf_gain']
            }
            target_db = audio_params['target_db']
            limiter_threshold = audio_params['limiter_threshold']
            
        # 构建完整的文件路径
        full_raw_path = os.path.join(app.static_folder, raw_audio_path)
        
        # 读取原始音频
        audio_data, sample_rate = sf.read(full_raw_path)
        
        # 处理音频
        processed_audio = service.audio_processor.process_audio(
            audio_data,
            normalize=True,
            target_db=target_db,
            eq_params=eq_params
        )
        processed_audio = service.audio_processor.apply_limiter(processed_audio, limiter_threshold)
        
        # 生成新的处理后音频文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_filename = f"processed_music_{timestamp}.wav"
        processed_path = os.path.join(app.static_folder, 'generated_music', processed_filename)
        
        # 保存处理后的音频
        sf.write(processed_path, processed_audio, sample_rate)
        
        # 将生成的文件路径添加到当前会话文件集合中
        with session_files_lock:
            current_session_files.add(processed_path)
        
        return jsonify({
            'success': True,
            'audio_path': f'generated_music/{processed_filename}'
        })
        
    except Exception as e:
        print(f"音频处理错误: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate_transition_to_calm', methods=['POST'])
def generate_transition_to_calm():
    """从当前情绪过渡到平静情绪的音乐生成路由"""
    try:
        # 获取参数
        emotion_text = request.json.get('emotion_text', '')
        transition_duration = request.json.get('duration', 30)
        
        # 验证参数
        if not emotion_text:
            return jsonify({'success': False, 'error': '缺少情绪描述'})
        
        # 检查服务是否可用
        if service is None:
            return jsonify({'success': False, 'error': '音乐生成服务不可用，请联系管理员'})
            
        # 重置进度
        global generation_progress
        generation_progress = {
            'prompt_status': 0,
            'music_status': 0
        }
        
        print(f"开始根据描述生成情绪过渡音乐，情绪描述: '{emotion_text}', 时长: {transition_duration}秒")
        
        # 确保transition_duration是整数
        try:
            transition_duration = int(transition_duration)
            if transition_duration < 5:
                transition_duration = 5
                print(f"过渡时长太短，已调整为最小值5秒")
            elif transition_duration > 120:
                transition_duration = 120
                print(f"过渡时长太长，已调整为最大值120秒")
        except ValueError:
            transition_duration = 30
            print(f"过渡时长格式无效，已设置为默认值30秒")
        
        # 首先使用API生成与情绪描述匹配的音乐提示词和情绪标签
        generation_progress['prompt_status'] = 50
        result = service.create_prompt(emotion_text)
        
        if not result:
            generation_progress['prompt_status'] = 0
            return jsonify({'success': False, 'error': '无法生成与情绪描述匹配的音乐提示词'})
            
        generation_progress['prompt_status'] = 100
        from_emotion = result['emotion_label']
        from_prompt = result['music_prompt']
        
        # 验证情绪标签
        if from_emotion not in EMOTION_PRESET_PARAMS:
            print(f"API返回的情绪标签 '{from_emotion}' 无效，使用默认情绪标签 'anxiety'")
            from_emotion = "anxiety"
        
        # 固定目标情绪为calm (平静)
        to_emotion = "calm"
        
        # 生成与用户情绪音乐相同BPM和key的平静音乐提示词
        calm_prompt = f"Calm peaceful ambient music with the same BPM and key as the emotional music. Gentle piano melodies, soft pads, and soothing textures."
        
        print(f"识别情绪标签: '{from_emotion}'")
        print(f"从情绪提示词: '{from_prompt}'")
        print(f"到平静提示词: '{calm_prompt}'")
        
        # 生成过渡音乐
        generation_progress['music_status'] = 10
        success, audio_path, raw_audio_path, from_emotion_path, to_emotion_path = service.generate_transition_music(
            from_emotion=from_emotion,
            to_emotion=to_emotion,
            from_prompt=from_prompt,
            to_prompt=calm_prompt,
            transition_duration=transition_duration
        )
        
        if not success:
            generation_progress['music_status'] = 0  # 重置进度
            error_msg = "过渡音乐生成失败，请尝试使用不同的参数"
            print(error_msg)
            return jsonify({'success': False, 'error': error_msg})
            
        generation_progress['music_status'] = 100  # 只在成功时设置100%
        
        # 验证生成的文件是否存在
        if not os.path.exists(audio_path) or not os.path.exists(raw_audio_path):
            error_msg = "音频文件生成失败，文件不存在"
            print(f"错误: {error_msg}, 音频路径: {audio_path}, 原始音频路径: {raw_audio_path}")
            return jsonify({'success': False, 'error': error_msg})
        
        # 获取相对路径
        audio_filename = os.path.basename(audio_path)
        raw_audio_filename = os.path.basename(raw_audio_path)
        relative_path = f'generated_music/{audio_filename}'
        raw_relative_path = f'generated_music/{raw_audio_filename}'
        
        # 将生成的文件路径添加到当前会话文件集合中
        with session_files_lock:
            current_session_files.add(audio_path)
            current_session_files.add(raw_audio_path)
            if from_emotion_path:
                current_session_files.add(from_emotion_path)
            if to_emotion_path:
                current_session_files.add(to_emotion_path)
        
        print(f"过渡音乐生成成功: {relative_path}")
        
        return jsonify({
            'success': True,
            'audio_path': relative_path,
            'raw_audio_path': raw_relative_path,
            'from_emotion_path': f'generated_music/{os.path.basename(from_emotion_path)}' if from_emotion_path else None,
            'to_emotion_path': f'generated_music/{os.path.basename(to_emotion_path)}' if to_emotion_path else None,
            'from_emotion': from_emotion,
            'to_emotion': 'calm',
            'stage': 'transition'
        })
        
    except Exception as e:
        generation_progress['music_status'] = 0  # 重置进度
        error_msg = f"过渡音乐生成错误: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': error_msg})

# 添加一个新的路由，用于手动清理临时文件
@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    """手动触发清理临时文件的API"""
    try:
        cleanup_audio_files()
        return jsonify({'success': True, 'message': '临时文件已清理'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # 确保静态文件目录存在
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    generated_music_dir = os.path.join(static_dir, 'generated_music')
    Path(generated_music_dir).mkdir(parents=True, exist_ok=True)
    app.run(debug=True) 