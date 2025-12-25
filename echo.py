import numpy as np
from scipy.io import wavfile


def echo_embed(audio_file, message, d0, d1, frame_amp, frame_size, frame_overlap, use_window=False):
    
    if not 0 < frame_amp <= 1:
        raise ValueError(f"Frame amplitude must be between 0 and 1, got {frame_amp}")
    
    if not 0 <= frame_overlap < 1:
        raise ValueError(f"Frame overlap must be between 0 and 1, got {frame_overlap}")
    
    if d0 <= 0 or d1 <= 0:
        raise ValueError(f"Delays must be positive: d0={d0}, d1={d1}")
    
    if d0 == d1:
        raise ValueError("d0 and d1 must be different to distinguish bits")
    
    if frame_size < max(d0, d1) * 2:
        raise ValueError(f"Frame size ({frame_size}) is too small for delays (d0={d0}, d1={d1})")
        
    sample_rate, audio = wavfile.read(audio_file)
    
    if len(audio.shape) == 1:
        audio = np.column_stack((audio, audio))
        
    audio = audio.astype(np.float32)
    left, right = audio[:, 0], audio[:, 1]
    converted_message = ''.join(format(ord(char), '08b') for char in message) + '00000000'
    num_bits = len(converted_message)
    
    frame_shift = int(frame_size * (1 - frame_overlap))
    frames_available = (len(left) - frame_size) // frame_shift + 1
    
    if num_bits > frames_available:
        raise ValueError("Audio is too short for the message length!")
    
    if use_window:
        window = np.hanning(frame_size)
    else:
        window = np.ones(frame_size)
        
    watermarked_left = np.copy(left)
    watermarked_right = np.copy(right)
    
    for i, bit in enumerate(converted_message):
        delay = d1 if bit == '1' else d0
        frame_shift = int(frame_size * (1 - frame_overlap))
        start = i * frame_shift 
        end = start + frame_size
        
        for j in range(start, end):
            if j + delay < len(watermarked_left):
                watermarked_left[j + delay] += int(frame_amp * left[j])
                watermarked_right[j + delay] += int(frame_amp * right[j])
                
                
    watermarked_audio = np.column_stack((watermarked_left, watermarked_right)).astype(np.int16)
    wavfile.write('watermarked_' + audio_file, sample_rate, watermarked_audio)
    return watermarked_audio


audio_file_name = 'Beginning 2.wav'
message = 'Fix it from the outside'

delay_zero = 100 # when bit = 0, delay by 100 samples
delay_one = 150 # when bit = 1, delay by 150 samples
echo_strength = 0.2 # echo amplitude
frame_length = 8192
frame_overlap = frame_length // 4