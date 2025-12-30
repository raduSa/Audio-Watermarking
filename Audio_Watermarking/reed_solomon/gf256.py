# Primitive polynomial used to generate GF(2^8) from GF(2)
PRIMITIVE_POLY = 0x11d  # x^8 + x^4 + x^3 + x^2 + 1

# EXP: represents the values alpha ^ i, where alpha is a primitive element
# alpha ^ 1 = x, or the symbol 00000010 (which is 2 in decimal)
# alpha ^ 2 = x ^ 2 = 00000010 << 1 = 00000100 (= 4 in decimal)
EXP = [0] * 512
# LOG: LOG[alpha ^ i] = i
LOG = [0] * 256

def init_tables():
    x = 1
    for i in range(255):
        EXP[i] = x
        LOG[x] = i
        x <<= 1
        if x & 0x100: # mod 2^8
            x ^= PRIMITIVE_POLY
    for i in range(255, 512):
        EXP[i] = EXP[i - 255]

init_tables()

def gf_add(a, b):
    return a ^ b # alpha ^ n + alpha ^ n = 0

def gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]] # (alpha ^ k1) * (alpha ^ k2) = alpha ^ (k1 + k2)

def gf_div(a, b):
    if b == 0:
        raise ZeroDivisionError()
    if a == 0:
        return 0
    return EXP[(LOG[a] - LOG[b]) % 255] # (alpha ^ k1) / (alpha ^ k2) = alpha ^ (k1 - k2)   (k1 - k2 can be negative)
