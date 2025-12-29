from rs import *

def test_gf_mul_div():
    for a in range(1, 255):
        for b in range(1, 255):            
            assert gf_div(gf_mul(a, b), b) == a

def test_exp_log():
    for i in range(1, 255):
        assert EXP[LOG[i]] == i

def test_rs_eval_encode_consistency():
    msg = [1, 2, 3, 4]
    n = 10
    codeword = rs_eval_encode(msg, n)

    xs = [EXP[i] for i in range(len(msg))]
    ys = codeword[:len(msg)]
    recovered = lagrange_interpolation(xs, ys)    

    assert recovered == msg


if __name__ == '__main__':
    test_gf_mul_div()
    test_exp_log()
    test_rs_eval_encode_consistency()