from Audio_Watermarking.algorithms.dct_dwt_svd import *
from Audio_Watermarking.watermarking_tests.attacks import add_noise
from Audio_Watermarking.utils.utils import bit_error_rate
from Audio_Watermarking.watermarking_tests.algorithm_interfaces import DWT_DCT_SVD

base_dir = 'Audio_Watermarking/sound_files'
input_audio = os.path.join(base_dir, 'Biome Fest.wav')
output_audio = os.path.join('Audio_Watermarking/watermarking_tests', 'svd_watermarked.wav')
watermark_bits = np.random.randint(0, 2, 256)
frame_size = 1024
step_size = 150.0

warnings.filterwarnings("ignore", category=UserWarning)

alg = DWT_DCT_SVD()

# embed_dct_dwt_svd(input_audio, output_audio, watermark_bits, step_size=step_size, frame_size=frame_size)
alg.embed(input_audio, output_audio, watermark_bits)
noisy_audio = add_noise(output_audio, snr_db=30)
extracted_bits = alg.extract(noisy_audio)
extracted_bits = np.frombuffer(extracted_bits.encode("ascii"), dtype=np.uint8) - ord('0')
print(watermark_bits ^ extracted_bits)
print(f'BER: {bit_error_rate(watermark_bits, extracted_bits)}')