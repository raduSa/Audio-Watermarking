from Audio_Watermarking.utils.utils import *
from Audio_Watermarking.algorithms.lsb import embed_lsb, extract_lsb
from Audio_Watermarking.algorithms.echo import embed_echo, extract_echo_nonblind
from Audio_Watermarking.algorithms.spread_spectrum import embed_dct_iss, extract_dct_spread_spectrum
from Audio_Watermarking.algorithms.qim import embed_qim, extract_qim
from Audio_Watermarking.algorithms.qim_dither import embed_qim_dither, extract_qim_dither
from Audio_Watermarking.algorithms.dwt_qim import embed_dwt_qim, extract_dwt_qim
import numpy as np
import matplotlib.pyplot as plt
import os, warnings


if __name__ == "__main__":
    base_dir = 'Audio_Watermarking/sound_files'
    input_audio = os.path.join(base_dir, 'Biome Fest.wav')
    output_audio = os.path.join(base_dir, 'watermarked.wav')
    watermark = 'Fix it from the outside'
    watermark_bits = text_to_bits(watermark)

    warnings.filterwarnings("ignore", category=UserWarning)
    
    fig, axs = plt.subplots(4, figsize=(12, 10))
    
    snrs_lsb = []
    for num_lsbs in range(1, 6):
        embed_lsb(input_audio, output_audio, watermark_bits, num_lsbs=num_lsbs)
        snrs_lsb.append(SNR(input_audio, output_audio))
    
    axs[0].plot(range(1, 6), snrs_lsb, marker='o')
    axs[0].set_title('LSB Watermarking: SNR vs Number of LSBs')
    axs[0].set_xlabel('LSB Number')
    axs[0].set_ylabel('SNR (dB)')
    axs[0].grid()
    
    snrs_echo = []
    for alpha in np.linspace(0.1, 0.5, 5):
        embed_echo(input_audio, output_audio, watermark_bits, alpha=alpha)
        snrs_echo.append(SNR(input_audio, output_audio))
    
    axs[1].plot(np.linspace(0.1, 0.5, 5), snrs_echo, marker='o', color='orange')
    axs[1].set_title('Echo Watermarking: SNR vs Alpha')
    axs[1].set_xlabel('Alpha (Amplitude)')
    axs[1].set_ylabel('SNR (dB)')
    axs[1].grid()
    
    snrs_ss = []
    for alpha in np.linspace(0.1, 0.5, 5):
        embed_dct_iss(input_audio, output_audio, watermark_bits, alpha=alpha)
        snrs_ss.append(SNR(input_audio, output_audio))
    axs[2].plot(np.linspace(0.1, 0.5, 5), snrs_ss, marker='o', color='green')
    axs[2].set_title('Spread Spectrum Watermarking: SNR vs Alpha')
    axs[2].set_xlabel('Alpha (Amplitude)')
    axs[2].set_ylabel('SNR (dB)')
    axs[2].grid()
    
    snrs_qim = []
    for delta in [5, 10, 15, 20, 25]:
        embed_qim(input_audio, output_audio, watermark_bits, delta, frame_size=2048, overlap=512)
        snrs_qim.append(SNR(input_audio, output_audio))
    axs[3].plot([5, 10, 15, 20, 25], snrs_qim, marker='o', color='red')
    axs[3].set_title('QIM Watermarking: SNR vs Delta')
    axs[3].set_xlabel('Delta (Step Size)')
    axs[3].set_ylabel('SNR (dB)')
    axs[3].grid()

    plt.tight_layout()
    plt.savefig('Watermark_SNR_Comparison.pdf')
    plt.show()