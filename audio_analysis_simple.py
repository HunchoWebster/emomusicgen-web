import os
import re
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
import librosa

# 设置matplotlib显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def find_audio_pairs(audio_dir):
    """查找相同case的no_processing和audio_processing_only音频对"""
    files = os.listdir(audio_dir)
    pattern = re.compile(r'(.+?)_(no_processing|audio_processing_only)\.wav')
    
    # 按照case_id组织文件
    case_files = {}
    for file in files:
        match = pattern.match(file)
        if match:
            case_id = match.group(1)
            file_type = match.group(2)
            if case_id not in case_files:
                case_files[case_id] = {}
            case_files[case_id][file_type] = os.path.join(audio_dir, file)
    
    # 只保留同时包含两种版本的case
    complete_pairs = {}
    for case_id, file_dict in case_files.items():
        if 'no_processing' in file_dict and 'audio_processing_only' in file_dict:
            complete_pairs[case_id] = file_dict
    
    return complete_pairs

def analyze_audio_file(file_path):
    """分析单个音频文件的特性"""
    y, sr = librosa.load(file_path, sr=None)
    
    # 计算RMS能量
    rms = np.sqrt(np.mean(y**2))
    
    # 计算频率区间能量
    n_fft = 2048
    stft = np.abs(librosa.stft(y, n_fft=n_fft))
    
    # 计算梅尔频谱
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    
    # 频率区域大致划分
    # 将频谱划分为低频、中频、高频区域
    freq_bins = stft.shape[0]
    low_freq_bins = int(freq_bins * 0.15)  # 低频区域约占15%
    high_freq_bins = int(freq_bins * 0.7)  # 高频区域从70%开始
    
    # 计算各频率区域的能量
    low_freq_energy = np.mean(np.sum(stft[:low_freq_bins, :], axis=0))
    mid_freq_energy = np.mean(np.sum(stft[low_freq_bins:high_freq_bins, :], axis=0))
    high_freq_energy = np.mean(np.sum(stft[high_freq_bins:, :], axis=0))
    
    return {
        'waveform': y,
        'sample_rate': sr,
        'rms_energy': rms,
        'stft': stft,
        'low_freq_energy': low_freq_energy,
        'mid_freq_energy': mid_freq_energy,
        'high_freq_energy': high_freq_energy
    }

def plot_comparison(case_id, raw_data, processed_data, output_dir):
    """为一对音频创建对比图"""
    plt.figure(figsize=(15, 10))
    
    # 波形对比
    plt.subplot(2, 2, 1)
    plt.plot(raw_data['waveform'], alpha=0.7, label='无处理')
    plt.title(f"波形对比 - {case_id}")
    plt.ylabel('振幅')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(processed_data['waveform'], alpha=0.7, color='r', label='音频处理')
    plt.title("波形对比（处理后）")
    plt.ylabel('振幅')
    plt.legend()
    
    # 频谱图对比
    plt.subplot(2, 2, 3)
    plt.imshow(20 * np.log10(raw_data['stft'] + 1e-10), 
               aspect='auto', origin='lower', 
               extent=[0, len(raw_data['waveform'])/raw_data['sample_rate'], 
                      0, raw_data['sample_rate']/2])
    plt.colorbar(format='%+2.0f dB')
    plt.title("频谱图 - 无处理")
    plt.ylabel('频率 (Hz)')
    plt.xlabel('时间 (秒)')
    
    plt.subplot(2, 2, 4)
    plt.imshow(20 * np.log10(processed_data['stft'] + 1e-10), 
               aspect='auto', origin='lower', 
               extent=[0, len(processed_data['waveform'])/processed_data['sample_rate'], 
                      0, processed_data['sample_rate']/2])
    plt.colorbar(format='%+2.0f dB')
    plt.title("频谱图 - 音频处理")
    plt.ylabel('频率 (Hz)')
    plt.xlabel('时间 (秒)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{case_id}_comparison.png'), dpi=150)
    plt.close()
    
    # 频率能量对比图
    plt.figure(figsize=(10, 6))
    labels = ['低频能量', '中频能量', '高频能量']
    raw_energies = [raw_data['low_freq_energy'], raw_data['mid_freq_energy'], raw_data['high_freq_energy']]
    processed_energies = [processed_data['low_freq_energy'], processed_data['mid_freq_energy'], processed_data['high_freq_energy']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, raw_energies, width, label='无处理')
    plt.bar(x + width/2, processed_energies, width, label='音频处理')
    plt.xlabel('频率范围')
    plt.ylabel('能量')
    plt.title(f'频率能量对比 - {case_id}')
    plt.xticks(x, labels)
    plt.legend()
    
    # 添加数值标签
    for i, v in enumerate(raw_energies):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    for i, v in enumerate(processed_energies):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
        
    # 添加变化百分比
    for i in range(len(labels)):
        change_pct = (processed_energies[i] / raw_energies[i] - 1) * 100
        plt.text(i, max(raw_energies[i], processed_energies[i]) + 0.1, 
                f'变化: {change_pct:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{case_id}_energy_comparison.png'), dpi=150)
    plt.close()

def create_html_index(audio_pairs, output_dir):
    """创建HTML索引页面展示分析结果"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>音频处理效果分析</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
            h1, h2 { color: #333; }
            .case-container { margin-bottom: 40px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
            .comparison-img { max-width: 100%; box-shadow: 0 0 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
            .audio-container { display: flex; gap: 20px; margin-bottom: 20px; }
            .audio-item { flex: 1; }
            .audio-item audio { width: 100%; }
            h3 { margin-top: 10px; margin-bottom: 5px; }
        </style>
    </head>
    <body>
        <h1>音频处理效果分析报告</h1>
        <p>本报告对比分析了原始音频和经过音频处理模块处理后的音频，展示了处理前后的差异。</p>
        
        <h2>各案例详细分析</h2>
    """
    
    # 为每个案例添加详细分析
    for case_id in audio_pairs.keys():
        html_content += f"""
        <div class="case-container">
            <h2>案例: {case_id}</h2>
            
            <div class="audio-container">
                <div class="audio-item">
                    <h3>原始音频</h3>
                    <audio controls src="../test_emotion_audio/{os.path.basename(audio_pairs[case_id]['no_processing'])}"></audio>
                </div>
                <div class="audio-item">
                    <h3>处理后音频</h3>
                    <audio controls src="../test_emotion_audio/{os.path.basename(audio_pairs[case_id]['audio_processing_only'])}"></audio>
                </div>
            </div>
            
            <h3>波形和频谱对比分析</h3>
            <img src="{case_id}_comparison.png" alt="{case_id} 波形频谱对比" class="comparison-img">
            
            <h3>频率能量对比分析</h3>
            <img src="{case_id}_energy_comparison.png" alt="{case_id} 能量对比" class="comparison-img">
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # 写入HTML文件
    with open(os.path.join(output_dir, 'analysis_index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    audio_dir = r"D:\Jupyter\audiocraft\webpage\static\test_emotion_audio"
    output_dir = r"D:\Jupyter\audiocraft\webpage\static\analysis_results"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找音频对
    audio_pairs = find_audio_pairs(audio_dir)
    print(f"找到 {len(audio_pairs)} 对音频文件进行分析")
    
    # 分析结果存储
    analysis_results = {}
    
    # 分析每对音频
    for case_id, file_paths in audio_pairs.items():
        print(f"分析案例: {case_id}")
        
        # 分析原始音频
        raw_data = analyze_audio_file(file_paths['no_processing'])
        
        # 分析处理后音频
        processed_data = analyze_audio_file(file_paths['audio_processing_only'])
        
        # 创建可视化比较
        plot_comparison(case_id, raw_data, processed_data, output_dir)
    
    # 创建HTML索引页面展示结果
    create_html_index(audio_pairs, output_dir)
    print(f"分析完成，结果可以在 {output_dir} 目录查看")

if __name__ == "__main__":
    main() 