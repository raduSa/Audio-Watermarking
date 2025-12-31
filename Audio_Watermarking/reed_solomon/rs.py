from gf256 import *

# def poly_mul(p, q):
#     r = [0] * (len(p) + len(q) - 1)
#     for i in range(len(p)):
#         for j in range(len(q)):
#             r[i + j] ^= gf_mul(p[i], q[j])
#     return r

# def rs_generator_poly(nsym):
#     g = [1]
#     for i in range(nsym):
#         g = poly_mul(g, [1, EXP[i]])
#     return g

# def rs_encode_msg(msg, nsym):
#     gen = rs_generator_poly(nsym)
#     msg_out = msg + [0] * nsym

#     for i in range(len(msg)):
#         coef = msg_out[i]
#         if coef != 0:
#             for j in range(len(gen)):
#                 msg_out[i + j] ^= gf_mul(gen[j], coef)

#     # original message + parity
#     return msg + msg_out[-nsym:]

# We get the codeword by evaluating the polynomial at alpha ^ (0 through n-1)
# Equivalent do DFT over finite field
def rs_eval_encode(msg, n):    
    k = len(msg)
    codeword = []

    # pm(a) = m_0 + m_1 * a + m_2 * a^2 + ... + m_k-1 * a^(k-1), where the message symboles m = (m_0, m_1, .. m_k-1)
    for i in range(n):
        x = EXP[i]  # alpha^i
        y = 0
        for j in range(k):
            y ^= gf_mul(msg[j], EXP[(i * j) % 255])
        codeword.append(y)

    return codeword


# DECODER

def poly_add(a, b):
    res = [0] * max(len(a), len(b))
    for i in range(len(a)):
        res[i] ^= a[i]
    for i in range(len(b)):
        res[i] ^= b[i]
    return res

def poly_mul(a, b):
    res = [0] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            res[i+j] ^= gf_mul(a[i], b[j])
    return res

def compute_syndromes(received, n, k):
    syndromes = []
    for j in range(1, n - k + 1):
        S = 0
        for i, r in enumerate(received):
            if r != 0:
                S ^= gf_mul(r, EXP[(i * j) % 255])
        syndromes.append(S)
    return syndromes

def solve_linear_system(A, b):
    n = len(b)

    for i in range(n):
        if A[i][i] == 0:
            for j in range(i+1, n):
                if A[j][i] != 0:
                    A[i], A[j] = A[j], A[i]
                    b[i], b[j] = b[j], b[i]
                    break
            else:
                raise ValueError()

        inv = gf_div(1, A[i][i])
        for k in range(i, n):
            A[i][k] = gf_mul(A[i][k], inv)
        b[i] = gf_mul(b[i], inv)

        for j in range(n):
            if j != i and A[j][i] != 0:
                factor = A[j][i]
                for k in range(i, n):
                    A[j][k] ^= gf_mul(factor, A[i][k])
                b[j] ^= gf_mul(factor, b[i])

    return b

def pgz_locator(syndromes):
    t = len(syndromes) // 2
    
    for v in range(t, 0, -1):
        A = []
        b = []
        for i in range(v):
            A.append(syndromes[i:i+v])
            b.append(syndromes[i+v])
        
        try:
            lambdas = solve_linear_system(A, b)
            
            return [1] + lambdas[::-1] 
        except ValueError:            
            pass

    raise ValueError("Uncorrectable errors")

def chien_search(locator, n):
    error_positions = []

    for i in range(n):
        x = EXP[(255 - i) % 255]  # alpha^{-i}
        val = 0
        for j, coef in enumerate(locator):
            if coef != 0:
                val ^= gf_mul(coef, EXP[(LOG[x] * j) % 255])
        if val == 0:
            error_positions.append(i)

    return error_positions

def forney(syndromes, locator, error_positions):
    omega = []
    for i in range(len(syndromes)):
        s = 0
        for j in range(min(i+1, len(locator))):
            s ^= gf_mul(locator[j], syndromes[i-j])
        omega.append(s)

    errors = {}

    for pos in error_positions:
        x_inv = EXP[(255 - pos) % 255]

        num = 0
        for i, w in enumerate(omega):
            if w != 0:
                num ^= gf_mul(w, EXP[(LOG[x_inv] * i) % 255])

        den = 0
        for i in range(1, len(locator), 2):
            den ^= gf_mul(locator[i], EXP[(LOG[x_inv] * (i-1)) % 255])

        errors[pos] = gf_div(num, den)

    return errors

def correct_errors(received, errors):
    corrected = received[:]
    for pos, mag in errors.items():
        corrected[pos] ^= mag
    return corrected

def lagrange_interpolation(xs, ys):
    k = len(xs)
    poly = [0] * k
    
    for i in range(k):
        num = [1]
        den = 1

        for j in range(k):
            if i != j:
                num = poly_mul(num, [xs[j], 1])
                den = gf_mul(den, xs[i] ^ xs[j])

        scale = gf_div(ys[i], den)
        num = [gf_mul(c, scale) for c in num]
        poly = poly_add(poly, num)

    return poly

def rs_decode(received, n, k):
    syndromes = compute_syndromes(received, n, k)
    # print(syndromes)

    if max(syndromes) != 0:
        locator = pgz_locator(syndromes)    
        error_positions = chien_search(locator, n)    
        errors = forney(syndromes, locator, error_positions)
        corrected = correct_errors(received, errors)

        ys = corrected        
    else:
        ys = received

    xs = [EXP[i] for i in range(k)]

    return lagrange_interpolation(xs, ys)

if __name__ == '__main__':
    # --- Test Case ---
    msg = [10, 20, 30, 40, 50]
    print(f'Msg: {msg}')
    encoded = rs_eval_encode(msg, 255) # This must be 255 for the math to work out (considering we are using syndrome decoding)
    
    print(f"Original: {encoded[:len(msg)]}")
    encoded[0] = 11
    encoded[4] = 99
    print(f"Corrupted: {encoded[:len(msg)]}")

    decoded = rs_decode(encoded, 255, 5)
    print(f"decoded:  {decoded}")