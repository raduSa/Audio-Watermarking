from Audio_Watermarking.utils.utils import *
from Audio_Watermarking.algorithms.lsb import embed_lsb, extract_lsb
from Audio_Watermarking.algorithms.echo import embed_echo, extract_echo_blind, extract_echo_nonblind
from Audio_Watermarking.algorithms.spread_spectrum import embed_dct_iss, extract_dct_spread_spectrum
from Audio_Watermarking.algorithms.qim import embed_qim, extract_qim
from Audio_Watermarking.algorithms.qim_dither import embed_qim_dither, extract_qim_dither
from Audio_Watermarking.algorithms.dwt_qim import embed_dwt_qim, extract_dwt_qim
import numpy as np
import matplotlib.pyplot as plt
import os, warnings
from aquatk.metrics.PEAQ import peaq

if __name__ == "__main__":
    base_dir = 'Audio_Watermarking/sound_files/soundtrack'
    watermark = 'I am Ice Cube and this is my Ice Cubical and I am here to ensure your cubes!'
    watermark_bits = text_to_bits(watermark)

    warnings.filterwarnings("ignore", category=UserWarning)
    
    best_peaq = -np.inf
    for alphas in [2.5, 5.0, 10.0]:
        peak_avg = 0
        for file in os.listdir(base_dir):
            if file.startswith('watermarked_ss'):
                continue
            embed_dct_iss(os.path.join(base_dir, file), 
                      os.path.join(base_dir, f'watermarked_ss_{alphas}_{file}'),
                        watermark_bits, alpha=alphas)
            peaq_score = peaq(os.path.join(base_dir, file), 
                                    os.path.join(base_dir, f'watermarked_ss_{alphas}_{file}'))
            peak_avg += peaq_score.odg
            print(peaq_score.odg)
        peak_avg /= len([file for file in os.listdir(base_dir) if not file.startswith('watermarked_ss')])
        print(f"Average PEAQ for alpha {alphas}: {peak_avg}")
        if peak_avg > best_peaq:
            best_peaq = peak_avg
            best_alpha = alphas
        
    print(f"Best alpha: {best_alpha} with PEAQ: {best_peaq}")        