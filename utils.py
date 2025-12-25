def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        chars.append(chr(int(byte, 2)))
    return ''.join(chars)

def audio_to_bits(samples):
    bits = []
    for sample in samples:
        sample = int(sample)
        unsigned_sample = sample if sample >= 0 else 65536 + sample
        bits.append(format(unsigned_sample, '016b'))
    return ''.join(bits)

def bits_to_audio(bits):
    samples = []
    for i in range(0, len(bits), 16):
        if i + 16 <= len(bits):
            byte = bits[i:i+16]
            sample = int(byte, 2)
            if sample >= 32768:
                sample -= 65536
            samples.append(sample)
    return samples
def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        chars.append(chr(int(byte, 2)))
    return ''.join(chars)

def audio_to_bits(samples):
    bits = []
    for sample in samples:
        sample = int(sample)
        unsigned_sample = sample if sample >= 0 else 65536 + sample
        bits.append(format(unsigned_sample, '016b'))
    return ''.join(bits)

def bits_to_audio(bits):
    samples = []
    for i in range(0, len(bits), 16):
        if i + 16 <= len(bits):
            byte = bits[i:i+16]
            sample = int(byte, 2)
            if sample >= 32768:
                sample -= 65536
            samples.append(sample)
    return samples