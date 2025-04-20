import numpy as np
import soundfile as sf
from scipy import signal
from typing import Tuple, Optional
import logging
import librosa

class AudioProcessor:
    """音频处理类，提供音量归一化、均衡器和限幅器功能"""
    
    def __init__(self, sample_rate: int = 32000):
        """
        初始化音频处理器
        
        Args:
            sample_rate: 采样率，默认32000Hz
        """
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        
    def normalize_volume(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        对音频进行音量归一化处理
        
        Args:
            audio: 输入音频数据
            target_db: 目标响度值（dB），默认-20 LUFS
            
        Returns:
            np.ndarray: 处理后的音频数据
        """
        try:
            # 计算当前音频的RMS值
            current_rms = np.sqrt(np.mean(audio**2))
            if current_rms == 0:
                return audio
                
            # 计算目标RMS值
            target_rms = 10 ** (target_db / 20)
            
            # 计算增益系数
            gain = target_rms / current_rms
            
            # 应用增益
            normalized_audio = audio * gain
            
            return normalized_audio
            
        except Exception as e:
            self.logger.error(f"音量归一化处理失败: {str(e)}")
            return audio
            
    def apply_eq(self, audio: np.ndarray, 
                 low_shelf_gain: float = 0.0,
                 mid_gain: float = 0.0,
                 high_shelf_gain: float = 0.0,
                 low_freq: float = 100.0,
                 mid_low_freq: float = 1000.0,
                 mid_high_freq: float = 5000.0,
                 high_freq: float = 8000.0) -> np.ndarray:
        """
        应用均衡器处理
        
        Args:
            audio: 输入音频数据
            low_shelf_gain: 低频增益（dB），默认0
            mid_gain: 中频增益（dB），默认0
            high_shelf_gain: 高频增益（dB），默认0
            low_freq: 低频截止频率（Hz），默认100
            mid_low_freq: 中频下限频率（Hz），默认1000
            mid_high_freq: 中频上限频率（Hz），默认5000
            high_freq: 高频截止频率（Hz），默认8000
            
        Returns:
            np.ndarray: 处理后的音频数据
        """
        try:
            processed_audio = audio.copy()
            
            # 检查音频长度是否足够 - 增加最小长度到128
            min_length = 128  # 增加最小长度以确保满足filtfilt需求
            
            # 检查是否需要应用EQ
            apply_eq_needed = (low_shelf_gain != 0 or mid_gain != 0 or high_shelf_gain != 0)
            
            # 如果不需要应用EQ，直接返回原始音频
            if not apply_eq_needed:
                return processed_audio
                
            # 检查音频维度，确保是一维数组
            if processed_audio.ndim > 1:
                # 如果是多通道，处理每个通道
                channels = []
                for channel in range(processed_audio.shape[-1]):
                    channel_data = processed_audio[:, channel]
                    # 对每个通道应用处理
                    if len(channel_data) < min_length:
                        self.logger.warning(f"通道 {channel} 音频长度({len(channel_data)})太短，进行填充至{min_length}个采样点")
                        pad_length = min_length - len(channel_data)
                        channel_data = np.pad(channel_data, (0, pad_length), mode='constant')
                    
                    # 应用滤波器处理通道数据
                    channel_data = self._apply_filters(channel_data, low_shelf_gain, mid_gain, high_shelf_gain,
                                                       low_freq, mid_low_freq, mid_high_freq, high_freq)
                    
                    # 如果之前进行了填充，现在需要移除填充部分
                    if len(channel_data) > len(audio):
                        channel_data = channel_data[:len(audio)]
                        
                    channels.append(channel_data)
                
                # 重新组合多通道
                processed_audio = np.column_stack(channels)
                return processed_audio
            
            # 一维数组处理
            if len(processed_audio) < min_length:
                self.logger.warning(f"音频长度({len(processed_audio)})太短，进行填充至{min_length}个采样点")
                pad_length = min_length - len(processed_audio)
                processed_audio = np.pad(processed_audio, (0, pad_length), mode='constant')
            
            # 应用滤波器
            processed_audio = self._apply_filters(processed_audio, low_shelf_gain, mid_gain, high_shelf_gain,
                                                  low_freq, mid_low_freq, mid_high_freq, high_freq)
            
            # 如果之前进行了填充，现在需要移除填充部分
            if len(processed_audio) > len(audio):
                processed_audio = processed_audio[:len(audio)]
                
            return processed_audio
            
        except Exception as e:
            self.logger.error(f"均衡器处理失败: {str(e)}")
            self.logger.error(f"音频长度: {len(audio)}, 需要大于{min_length}")
            return audio
    
    def _apply_filters(self, audio_data: np.ndarray, 
                       low_shelf_gain: float,
                       mid_gain: float,
                       high_shelf_gain: float,
                       low_freq: float,
                       mid_low_freq: float,
                       mid_high_freq: float,
                       high_freq: float) -> np.ndarray:
        """
        对音频数据应用滤波器
        
        Args:
            audio_data: 单通道音频数据
            low_shelf_gain: 低频增益
            mid_gain: 中频增益
            high_shelf_gain: 高频增益
            low_freq: 低频截止频率
            mid_low_freq: 中频下限频率
            mid_high_freq: 中频上限频率
            high_freq: 高频截止频率
            
        Returns:
            np.ndarray: 应用滤波器后的音频数据
        """
        processed_data = audio_data.copy()
        
        # 设计低架滤波器
        if low_shelf_gain != 0:
            b, a = signal.butter(2, low_freq/(self.sample_rate/2), btype='low')
            low_freq_audio = signal.filtfilt(b, a, processed_data)
            processed_data = processed_data + (low_freq_audio * (10 ** (low_shelf_gain / 20) - 1))
            
        # 设计中频滤波器
        if mid_gain != 0:
            b, a = signal.butter(2, [mid_low_freq/(self.sample_rate/2), mid_high_freq/(self.sample_rate/2)], btype='band')
            mid_freq_audio = signal.filtfilt(b, a, processed_data)
            processed_data = processed_data + (mid_freq_audio * (10 ** (mid_gain / 20) - 1))
            
        # 设计高架滤波器
        if high_shelf_gain != 0:
            b, a = signal.butter(2, high_freq/(self.sample_rate/2), btype='high')
            high_freq_audio = signal.filtfilt(b, a, processed_data)
            processed_data = processed_data + (high_freq_audio * (10 ** (high_shelf_gain / 20) - 1))
            
        return processed_data
            
    def apply_limiter(self, audio: np.ndarray, threshold_db: float = -1.0) -> np.ndarray:
        """
        应用限幅器处理
        
        Args:
            audio: 输入音频数据
            threshold_db: 限幅阈值（dB），默认-1.0
            
        Returns:
            np.ndarray: 处理后的音频数据
        """
        try:
            threshold = 10 ** (threshold_db / 20)
            return np.clip(audio, -threshold, threshold)
        except Exception as e:
            self.logger.error(f"限幅器处理失败: {str(e)}")
            return audio
            
    def process_audio(self, audio: np.ndarray, 
                     normalize: bool = True,
                     target_db: float = -20.0,
                     eq_params: Optional[dict] = None) -> np.ndarray:
        """
        完整的音频处理流程
        
        Args:
            audio: 输入音频数据
            normalize: 是否进行音量归一化
            target_db: 目标响度值（dB）
            eq_params: 均衡器参数字典
            
        Returns:
            np.ndarray: 处理后的音频数据
        """
        try:
            processed_audio = audio.copy()
            
            # 应用音量归一化
            if normalize:
                processed_audio = self.normalize_volume(processed_audio, target_db)
                
            # 应用均衡器
            if eq_params:
                processed_audio = self.apply_eq(
                    processed_audio,
                    low_shelf_gain=eq_params.get('low_shelf_gain', 0.0),
                    mid_gain=eq_params.get('mid_gain', 0.0),
                    high_shelf_gain=eq_params.get('high_shelf_gain', 0.0),
                    low_freq=eq_params.get('low_freq', 100.0),
                    mid_low_freq=eq_params.get('mid_low_freq', 1000.0),
                    mid_high_freq=eq_params.get('mid_high_freq', 5000.0),
                    high_freq=eq_params.get('high_freq', 8000.0)
                )
                
            # 应用限幅器
            processed_audio = self.apply_limiter(processed_audio)
                
            return processed_audio
            
        except Exception as e:
            self.logger.error(f"音频处理失败: {str(e)}")
            return audio
            
    def create_transition(self, from_audio, to_audio, overlap_seconds=2.0):
        """
        创建两段音频之间的平滑过渡
        
        Args:
            from_audio: 起始音频数据
            to_audio: 目标音频数据
            overlap_seconds: 重叠部分的长度（秒）
            
        Returns:
            numpy.ndarray: 过渡后的音频数据
        """
        try:
            # 确保音频数据是numpy数组
            from_audio = np.asarray(from_audio)
            to_audio = np.asarray(to_audio)
            
            # 确保音频数据是2D的 [channels, samples]
            if from_audio.ndim == 1:
                from_audio = from_audio.reshape(1, -1)
            if to_audio.ndim == 1:
                to_audio = to_audio.reshape(1, -1)
                
            # 确保两段音频的通道数相同
            if from_audio.shape[0] != to_audio.shape[0]:
                self.logger.warning(f"音频通道数不匹配: from_audio={from_audio.shape[0]}, to_audio={to_audio.shape[0]}")
                # 如果通道数不同，将多通道音频转换为单通道
                if from_audio.shape[0] > 1:
                    from_audio = from_audio.mean(axis=0, keepdims=True)
                if to_audio.shape[0] > 1:
                    to_audio = to_audio.mean(axis=0, keepdims=True)
            
            # 计算重叠部分的长度（样本数）
            overlap_samples = int(overlap_seconds * self.sample_rate)
            
            # 确保重叠长度不超过音频长度
            overlap_samples = min(overlap_samples, from_audio.shape[1] // 2, to_audio.shape[1] // 2)
            if overlap_samples < 1:
                overlap_samples = 1
                self.logger.warning("重叠长度太短，使用最小值1")
            
            # 使用S型曲线（sin^2）创建淡入淡出曲线，实现更平滑的渐变
            fade_in = np.sin(np.linspace(0, np.pi/2, overlap_samples))**2  # 渐强曲线
            fade_out = np.sin(np.linspace(np.pi/2, 0, overlap_samples))**2  # 渐弱曲线
            
            # 记录淡变曲线信息
            self.logger.info(f"创建过渡: 重叠样本数={overlap_samples}, 重叠时间={overlap_seconds}秒, 使用S型渐变曲线")
            
            # 计算最终音频的长度
            total_length = from_audio.shape[1] + to_audio.shape[1] - overlap_samples
            
            # 创建结果数组
            result = np.zeros((from_audio.shape[0], total_length), dtype=np.float32)
            
            # 规范化音频音量以确保两段音频的音量水平相似
            from_max = np.max(np.abs(from_audio))
            to_max = np.max(np.abs(to_audio))
            
            if from_max > 0 and to_max > 0:
                # 将两段音频调整到相同的平均音量
                avg_max = (from_max + to_max) / 2
                from_audio = from_audio * (avg_max / from_max)
                to_audio = to_audio * (avg_max / to_max)
            
            # 复制第一段音频（除了重叠部分使用渐弱曲线）
            result[:, :from_audio.shape[1]] = from_audio
            for i in range(from_audio.shape[0]):
                result[i, from_audio.shape[1]-overlap_samples:from_audio.shape[1]] *= fade_out
            
            # 复制第二段音频（开始部分使用渐强曲线）
            offset = from_audio.shape[1] - overlap_samples
            result[:, offset:offset+to_audio.shape[1]] += to_audio
            for i in range(to_audio.shape[0]):
                result[i, offset:offset+overlap_samples] *= fade_in
            
            # 平滑重叠区域避免可能的音量峰值
            # 检查并处理可能的音量峰值
            overlap_region = result[:, offset:offset+overlap_samples]
            max_amplitude = np.max(np.abs(overlap_region))
            if max_amplitude > 1.0:
                # 如果重叠区域的音量超过1.0，进行轻微压缩
                scale_factor = 0.95 / max_amplitude
                result[:, offset:offset+overlap_samples] *= scale_factor
                self.logger.info(f"重叠区域音量调整: 最大振幅 {max_amplitude} 调整为 {max_amplitude * scale_factor}")
            
            # 确保输出数据使用float32类型，soundfile库需要此类型
            result = result.astype(np.float32)
            
            # 如果结果是单声道，转换为一维数组
            if result.shape[0] == 1:
                result = result.squeeze(0)
                
            return result
            
        except Exception as e:
            self.logger.error(f"创建音频过渡失败: {str(e)}")
            self.logger.error(f"from_audio shape: {from_audio.shape if hasattr(from_audio, 'shape') else 'unknown'}, dtype: {from_audio.dtype if hasattr(from_audio, 'dtype') else 'unknown'}")
            self.logger.error(f"to_audio shape: {to_audio.shape if hasattr(to_audio, 'shape') else 'unknown'}, dtype: {to_audio.dtype if hasattr(to_audio, 'dtype') else 'unknown'}")
            
            # 在发生错误时，返回起始音频作为回退方案
            return from_audio.astype(np.float32)
        
    def create_smooth_transition(self, from_audio, to_audio, overlap_seconds=2.0):
        """
        创建两段音频之间的平滑过渡，使用更平滑的交叉淡变
        
        Args:
            from_audio: 起始音频数据
            to_audio: 目标音频数据
            overlap_seconds: 重叠部分的长度（秒）
            
        Returns:
            numpy.ndarray: 过渡后的音频数据
        """
        # 确保音频是2D的
        if from_audio.ndim == 1:
            from_audio = from_audio.reshape(1, -1)
        if to_audio.ndim == 1:
            to_audio = to_audio.reshape(1, -1)
            
        # 确保两段音频的通道数相同
        assert from_audio.shape[0] == to_audio.shape[0], "两段音频的通道数必须相同"
        
        # 计算重叠部分的长度（样本数）
        overlap_samples = int(overlap_seconds * self.sample_rate)
        
        # 创建重叠部分的淡入淡出窗口 - 使用余弦窗口实现更平滑的过渡
        fade_in = np.sin(np.linspace(0, np.pi/2, overlap_samples))**2
        fade_out = np.sin(np.linspace(np.pi/2, 0, overlap_samples))**2
        
        # 计算最终音频的长度
        total_length = from_audio.shape[1] + to_audio.shape[1] - overlap_samples
        result = np.zeros((from_audio.shape[0], total_length))
        
        # 复制第一段音频（除了重叠部分用淡出处理）
        result[:, :from_audio.shape[1]] = from_audio
        for i in range(from_audio.shape[0]):
            result[i, from_audio.shape[1]-overlap_samples:from_audio.shape[1]] *= fade_out
        
        # 复制第二段音频（开始部分用淡入处理）
        offset = from_audio.shape[1] - overlap_samples
        result[:, offset:offset+to_audio.shape[1]] += to_audio
        for i in range(to_audio.shape[0]):
            result[i, offset:offset+overlap_samples] *= fade_in
        
        return result
        
    def create_spectral_transition(self, from_audio, to_audio, overlap_seconds=2.0):
        """
        创建两段音频之间的频谱过渡，使用STFT进行平滑过渡
        
        Args:
            from_audio: 起始音频数据
            to_audio: 目标音频数据
            overlap_seconds: 重叠部分的长度（秒）
            
        Returns:
            numpy.ndarray: 过渡后的音频数据
        """
        # 确保音频是单通道的（因为librosa.stft需要单通道）
        if from_audio.ndim > 1:
            from_audio = np.mean(from_audio, axis=0)
        if to_audio.ndim > 1:
            to_audio = np.mean(to_audio, axis=0)
            
        # 计算重叠部分的长度（样本数）
        overlap_samples = int(overlap_seconds * self.sample_rate)
        
        # 提取需要过渡的部分
        from_end = from_audio[-overlap_samples:]
        to_start = to_audio[:overlap_samples]
        
        # 计算短时傅里叶变换
        n_fft = 2048
        hop_length = 512
        
        from_stft = librosa.stft(from_end, n_fft=n_fft, hop_length=hop_length)
        to_stft = librosa.stft(to_start, n_fft=n_fft, hop_length=hop_length)
        
        # 创建线性过渡权重
        num_frames = min(from_stft.shape[1], to_stft.shape[1])
        weights = np.linspace(1, 0, num_frames)
        
        # 线性混合两个STFT
        transition_stft = from_stft[:, :num_frames] * weights + to_stft[:, :num_frames] * (1 - weights)
        
        # 反变换回时域
        transition_audio = librosa.istft(transition_stft, hop_length=hop_length)
        
        # 组合三段音频
        result = np.concatenate([
            from_audio[:-overlap_samples],
            transition_audio,
            to_audio[overlap_samples:]
        ])
        
        # 将结果重新变为原始格式
        if from_audio.ndim > 1 or to_audio.ndim > 1:
            result = result.reshape(1, -1)
            
        return result 