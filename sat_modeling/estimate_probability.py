#!/usr/bin/env python3
from util import ddt, ddt4
import numpy as np
from IPython import embed


def get_per_sbox_prob(sbox_in, sbox_out, tbox_in, tbox_out, unknown, forget):
    unknown = unknown[:len(sbox_in)]
    forget = forget[:len(sbox_in)]

    indices2_single = (unknown == 0) & (tbox_in == 0)
    indices2_double = (unknown != 0)
    indices3 = forget != 0
    indices4 = (unknown == 0) & (forget == 0)
    assert np.all((indices2_single | indices2_double | indices3 | indices4) == np.full(unknown.shape, True))

    ddt3 = ddt4.ddt3
    ddt2_prob_single = ddt[sbox_in[indices2_single], sbox_out[indices2_single]] / 256
    ddt2_prob_double = (ddt[sbox_in[indices2_double], sbox_out[indices2_double]] / 256)**2
    ddt3_prob = ddt3[sbox_in[indices3], tbox_in[indices3], sbox_out[indices3]] / 256
    ddt4_prob = ddt4[sbox_in[indices4], tbox_in[indices4], sbox_out[indices4], tbox_out[indices4]] / 256

    ddt_prob = np.zeros_like(sbox_in, dtype=float)
    ddt_prob[indices2_single] = ddt2_prob_single
    ddt_prob[indices2_double] = ddt2_prob_double
    ddt_prob[indices3] = ddt3_prob
    ddt_prob[indices4] = ddt4_prob
    return ddt_prob


def get_total_log_prob(sbox_in, sbox_out, tbox_in, tbox_out, unknown, forget):
    ddt_probs = get_per_sbox_prob(sbox_in, sbox_out, tbox_in, tbox_out, unknown, forget)
    return np.sum(np.log2(ddt_probs))


all_log_probs = []
def main(bit_char_file: str):
    with np.load(bit_char_file) as f:
        sbox_in = f.get('sbox_in')
        sbox_out = f.get('sbox_out')
        tbox_in = f.get('tbox_in')
        tbox_out = f.get('tbox_out')
        unknown = f.get('unknown')[:len(sbox_in)]
        forget = f.get('forget')[:len(sbox_in)]
        tweakeys = f.get('tweakeys')

    ddt_probs = get_per_sbox_prob(sbox_in, sbox_out, tbox_in, tbox_out, unknown, forget)
    total_log_prob = np.sum(np.log2(ddt_probs))
    all_log_probs.append(total_log_prob)
    print(f'{bit_char_file}: 2^{total_log_prob:.2f}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bit_char_file', nargs='*', type=str,
                        help='differential characteristic in .npz file format')

    args = parser.parse_args()
    for file in args.bit_char_file:
        if 'coll' in file:
            continue
        main(file)

    all_log_probs = np.array(all_log_probs)
    print(f"range: 2^{all_log_probs.max():.2f} .. 2^{all_log_probs.min()}", end=" ")
    print(f"(mean: 2^{all_log_probs.mean():.2f})")
