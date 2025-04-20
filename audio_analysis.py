import os
import re
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
import pandas as pd

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
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = np.mean(rms)
    
    # 计算频谱质心 (频率中心)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_centroid = np.mean(spectral_centroid)
    
    # 计算频谱带宽
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    avg_bandwidth = np.mean(spectral_bandwidth)
    
    # 计算过零率（音色的一种度量）
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    avg_zcr = np.mean(zero_crossing_rate)
    
    # 计算梅尔频谱
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    
    # 获取低频、中频、高频能量
    low_freq_energy = np.mean(np.sum(mel_spec[:10, :], axis=0))
    mid_freq_energy = np.mean(np.sum(mel_spec[10:80, :], axis=0))
    high_freq_energy = np.mean(np.sum(mel_spec[80:, :], axis=0))
    
    return {
        'waveform': y,
        'sample_rate': sr,
        'rms_energy': avg_rms,
        'spectral_centroid': avg_centroid,
        'spectral_bandwidth': avg_bandwidth,
        'zero_crossing_rate': avg_zcr,
        'mel_spectrogram': mel_spec,
        'low_freq_energy': low_freq_energy,
        'mid_freq_energy': mid_freq_energy,
        'high_freq_energy': high_freq_energy
    }

def plot_comparison(case_id, raw_data, processed_data, output_dir):
    """为一对音频创建对比图"""
    plt.figure(figsize=(15, 12))
    
    # 波形对比
    plt.subplot(3, 2, 1)
    librosa.display.waveshow(raw_data['waveform'], sr=raw_data['sample_rate'], alpha=0.7, label='无处理')
    plt.title(f"波形对比 - {case_id}")
    plt.subplot(3, 2, 2)
    librosa.display.waveshow(processed_data['waveform'], sr=processed_data['sample_rate'], alpha=0.7, color='r', label='音频处理')
    plt.title("波形对比（处理后）")
    
    # 频谱图对比
    plt.subplot(3, 2, 3)
    librosa.display.specshow(librosa.amplitude_to_db(
        np.abs(librosa.stft(raw_data['waveform'])), ref=np.max),
        sr=raw_data['sample_rate'], y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title("频谱图 - 无处理")
    
    plt.subplot(3, 2, 4)
    librosa.display.specshow(librosa.amplitude_to_db(
        np.abs(librosa.stft(processed_data['waveform'])), ref=np.max),
        sr=processed_data['sample_rate'], y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title("频谱图 - 音频处理")
    
    # 能量对比图
    plt.subplot(3, 2, 5)
    labels = ['低频能量', '中频能量', '高频能量']
    raw_energies = [raw_data['low_freq_energy'], raw_data['mid_freq_energy'], raw_data['high_freq_energy']]
    processed_energies = [processed_data['low_freq_energy'], processed_data['mid_freq_energy'], processed_data['high_freq_energy']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, raw_energies, width, label='无处理')
    plt.bar(x + width/2, processed_energies, width, label='音频处理')
    plt.xlabel('频率范围')
    plt.ylabel('能量')
    plt.title('频率能量对比')
    plt.xticks(x, labels)
    plt.legend()
    
    # 添加音频特征数值对比表格
    plt.subplot(3, 2, 6)
    plt.axis('off')
    
    table_data = [
        ['特征', '无处理', '音频处理', '变化百分比'],
        ['RMS能量', f'{raw_data["rms_energy"]:.4f}', f'{processed_data["rms_energy"]:.4f}', 
         f'{(processed_data["rms_energy"]/raw_data["rms_energy"]-1)*100:.1f}%'],
        ['频谱质心', f'{raw_data["spectral_centroid"]:.1f}', f'{processed_data["spectral_centroid"]:.1f}', 
         f'{(processed_data["spectral_centroid"]/raw_data["spectral_centroid"]-1)*100:.1f}%'],
        ['频谱带宽', f'{raw_data["spectral_bandwidth"]:.1f}', f'{processed_data["spectral_bandwidth"]:.1f}', 
         f'{(processed_data["spectral_bandwidth"]/raw_data["spectral_bandwidth"]-1)*100:.1f}%'],
        ['过零率', f'{raw_data["zero_crossing_rate"]:.4f}', f'{processed_data["zero_crossing_rate"]:.4f}', 
         f'{(processed_data["zero_crossing_rate"]/raw_data["zero_crossing_rate"]-1)*100:.1f}%']
    ]
    
    plt.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.25, 0.25, 0.25, 0.25])
    plt.title('音频特征对比')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{case_id}_comparison.png'), dpi=150)
    plt.close()

def create_summary_report(analysis_results, output_dir):
    """创建所有案例的总结报告"""
    summary_data = []
    
    for case_id, result in analysis_results.items():
        summary_data.append({
            '案例ID': case_id,
            'RMS能量变化': (result['processed']['rms_energy'] / result['raw']['rms_energy'] - 1) * 100,
            '频谱质心变化': (result['processed']['spectral_centroid'] / result['raw']['spectral_centroid'] - 1) * 100,
            '频谱带宽变化': (result['processed']['spectral_bandwidth'] / result['raw']['spectral_bandwidth'] - 1) * 100,
            '过零率变化': (result['processed']['zero_crossing_rate'] / result['raw']['zero_crossing_rate'] - 1) * 100,
            '低频能量变化': (result['processed']['low_freq_energy'] / result['raw']['low_freq_energy'] - 1) * 100,
            '中频能量变化': (result['processed']['mid_freq_energy'] / result['raw']['mid_freq_energy'] - 1) * 100,
            '高频能量变化': (result['processed']['high_freq_energy'] / result['raw']['high_freq_energy'] - 1) * 100
        })
    
    df = pd.DataFrame(summary_data)
    
    # 保存CSV报告
    df.to_csv(os.path.join(output_dir, 'audio_processing_summary.csv'), index=False, encoding='utf-8-sig')
    
    # 创建平均变化图表
    plt.figure(figsize=(12, 8))
    
    # 计算每个特征的平均变化
    avg_changes = df.mean(numeric_only=True)
    
    # 创建条形图
    plt.bar(avg_changes.index, avg_changes.values)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('音频处理对各特征的平均影响（%变化）')
    plt.ylabel('百分比变化 (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_changes.png'), dpi=150)
    plt.close()
    
    return df

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
        
        # 存储分析结果
        analysis_results[case_id] = {
            'raw': raw_data,
            'processed': processed_data
        }
        
        # 创建可视化比较
        plot_comparison(case_id, raw_data, processed_data, output_dir)
    
    # 创建总结报告
    summary_df = create_summary_report(analysis_results, output_dir)
    print("分析完成，报告已生成")
    
    # 创建HTML索引页面展示结果
    create_html_index(audio_pairs, output_dir)
    print(f"结果可以在 {output_dir} 目录查看")

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
            .comparison-img { max-width: 100%; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .audio-container { display: flex; gap: 20px; margin-bottom: 20px; }
            .audio-item { flex: 1; }
            .audio-item audio { width: 100%; }
            h3 { margin-top: 10px; margin-bottom: 5px; }
        </style>
    </head>
    <body>
        <h1>音频处理效果分析报告</h1>
        <p>本报告对比分析了原始音频和经过音频处理模块处理后的音频，展示了处理前后的差异。</p>
        
        <h2>整体分析结果</h2>
        <img src="average_changes.png" alt="平均变化图表" class="comparison-img">
        <p>上图展示了音频处理对各个音频特征的平均影响（百分比变化）。</p>
        
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
            
            <h3>详细分析</h3>
            <img src="{case_id}_comparison.png" alt="{case_id} 对比分析" class="comparison-img">
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # 写入HTML文件
    with open(os.path.join(output_dir, 'analysis_index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    main() 