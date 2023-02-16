#!/usr/bin/env python3

"""
This script tries to find a collision for RomulusHash based on a bitwise dual
characteristic in numpy's .npz file format.
"""
from z3 import *
from typing import *
from util import *
from read_sol import *
from estimate_probability import get_total_log_prob

from time import process_time
from functools import reduce
from itertools import product, count, takewhile

from binascii import hexlify

from IPython import embed

import numpy as np
import zipfile

from multiprocessing import Process, Queue

contexts = []
interrupted = False

sbox_dnf = DNF.from_indices(16, (sbox.astype(np.uint16) << 0) | (np.arange(256, dtype=np.uint16) << 8))
sbox_cnf = sbox_dnf.to_minimal_cnf()


class CollisionSolver:
    def __init__(self, s: Solver, sbox_in, sbox_out, tbox_in, tbox_out, tweakeys, unknown, forget, prefix: str = ''):
        self.numrounds = len(sbox_in)

        self.s = s
        self._prefix = prefix
        self._nldtool_char = None
        self._char = None

        self._sbox_in = sbox_in
        self._sbox_out = sbox_out

        self._tbox_in = tbox_in
        self._tbox_out = tbox_out

        self._sbox_out_unknown = unknown
        self._sbox_in_unknown = (unknown != 0) & (forget == 0)
        self._forget = forget

        if self._sbox_in_unknown[0, 0, 0]:
            print("[WARN] fixed input difference is set to unknown --> correcting it")
            self._tbox_in[0, 0, 0] = 1
            self._sbox_in_unknown[0, 0, 0] = False

        self._unmasked_tweakey = np.bitwise_xor.reduce(tweakeys, axis=1)
        self._tweakey = [np.bitwise_and(tk, tweakey_mask) for tk in self._unmasked_tweakey]

        self._total_log_prob = None

        self._create_vars()
        self._model_linear_layer()
        self._model_sboxes()

        # add lfsr model
        for lfsr in self.tk2_lfsrs:
            for i, constraint in enumerate(lfsr.get_constraints()):
                s.assert_and_track(constraint, f'{lfsr.name}_{i}')
        for lfsr in self.tk3_lfsrs:
            for i, constraint in enumerate(lfsr.get_constraints()):
                s.assert_and_track(constraint, f'{lfsr.name}_{i}')

    @property
    def total_log_prob(self):
        if self._total_log_prob is not None:
            return self._total_log_prob

        self._total_log_prob = get_total_log_prob(self._sbox_in, self._sbox_out,
                                                  self._tbox_in, self._tbox_out,
                                                  self._sbox_out_unknown, self._forget)
        
        return self._total_log_prob

    def _create_vars(self):
        rnds = self.numrounds
        ctx = self.s.ctx
        self.sbox_input_var = [SkinnyState(f'{self._prefix}sbox_in_{rnd}', ctx=ctx) for rnd in range(rnds)]
        self.sbox_output_var = [SkinnyState(f'{self._prefix}sbox_out_{rnd}', ctx=ctx) for rnd in range(rnds)]
        self.tbox_input_var = [SkinnyState(f'{self._prefix}tbox_in_{rnd}', ctx=ctx) for rnd in range(rnds)]
        self.tbox_output_var = [SkinnyState(f'{self._prefix}tbox_out_{rnd}', ctx=ctx) for rnd in range(rnds)]

        self.pt = self.sbox_input_var[0]

        self.key = SkinnyState('t1', use_bits=True, ctx=ctx)
        self.tk2_lfsrs = [LfsrState(f'tk2_{i}', connection_poly, ctx=ctx) for i in range(16)]
        self.tk3_lfsrs = [LfsrState(f'tk3_{i}', connection_poly, ctx=ctx) for i in range(16)]

        self.round_tweakeys = []
        for rnd in range(rnds):
            key = self.key._vars

            # rows 2, 3 of the base tweakey are updated one round earlier
            # corresponding to indices in range(8, 16)
            tk2 = [self.tk2_lfsrs[i].get_bit_range((rnd + (i in range(8, 16))) // 2) for i in range(16)]

            tk3 = [self.tk3_lfsrs[i].get_bit_range((rnds - 1) // 2 - (rnd + (i in range(8, 16))) // 2) for i in range(16)]

            key = SkinnyState.from_vars(apply_perm(key, tweakey_perm, rnd))
            tk2 = SkinnyState.from_vars(apply_perm(tk2, tweakey_perm, rnd))

            tk3 = SkinnyState.from_vars(apply_perm(tk3, tweakey_perm, rnd))

            self.round_tweakeys.append((key, tk2, tk3))

        self.m1 = self.round_tweakeys[0][1]
        self.m2 = self.round_tweakeys[0][2]

    def _get_tb_out_var_and_deltas(self, rnd):
        variables = np.full((4, 4), None)
        deltas = np.zeros((4, 4), np.uint8)

        known_indices = self._sbox_out_unknown[rnd] == 0
        unknown_indices = self._sbox_out_unknown[rnd] != 0

        variables[known_indices] = self.sbox_output_var[rnd].as_array()[known_indices]
        deltas[known_indices] = self._tbox_out[rnd, known_indices]

        variables[unknown_indices] = self.tbox_output_var[rnd].as_array()[unknown_indices]

        return variables, deltas

    def _model_linear_layer(self):
        for rnd in range(self.numrounds - 1):
            rtks = self.round_tweakeys[rnd]
            in_rcs = expanded_rc[rnd]

            tweakeys = np.zeros((2, 4), object)

            for row, col in product(range(2), range(4)):
                tweakeys[row, col] = simplify(Concat(*rtks[0][row, col])
                                              ^ Concat(*rtks[1][row, col])
                                              ^ Concat(*rtks[2][row, col]))

            sb_mc_input = self.sbox_output_var[rnd].as_array()
            # tb_mc_input = self.tbox_output_var[rnd].as_array()
            tb_mc_input, tb_deltas = self._get_tb_out_var_and_deltas(rnd)

            for row, col in product(range(2), range(4)):
                sb_mc_input[row, col] ^= tweakeys[row, col]
                tb_mc_input[row, col] ^= tweakeys[row, col]

            sb_mc_input = do_shift_rows(sb_mc_input)
            in_rcs = do_shift_rows(in_rcs)
            tb_mc_input = do_shift_rows(tb_mc_input)
            tb_deltas = do_shift_rows(tb_deltas)

            for col in range(4):
                for row in range(4):
                    sb_out_var = self.sbox_input_var[rnd + 1][row, col]
                    tb_out_var = self.tbox_input_var[rnd + 1][row, col]

                    sb_in_vars = sb_mc_input[mixing_mat[row] != 0, col]
                    tb_in_vars = tb_mc_input[mixing_mat[row] != 0, col]
                    tb_delta = int(np.bitwise_xor.reduce(tb_deltas[mixing_mat[row] != 0, col]))

                    constant = int(np.bitwise_xor.reduce(in_rcs[mixing_mat[row] != 0, col]))

                    sb_cond = sb_out_var == reduce(lambda x, y: x ^ y, sb_in_vars) ^ constant
                    tb_cond = tb_out_var == reduce(lambda x, y: x ^ y, tb_in_vars) ^ (constant ^ tb_delta)

                    self.s.assert_and_track(sb_cond, f'lin_sbox_{rnd}_{row}_{col}')
                    if (self._sbox_in_unknown[rnd + 1, row, col]):
                        self.s.assert_and_track(tb_cond, f'lin_tbox_{rnd}_{row}_{col}')

    def _model_single_sbox(self, in_var, out_var, in_delta, out_delta, name: Optional[str] = None, input_delta=0):
        in_vals = get_solution_set(in_delta, out_delta)
        out_vals = sbox[in_vals]
        dnf = DNF.from_indices(16, (out_vals.astype(np.uint16) << 0) | ((in_vals ^ input_delta).astype(np.uint16) << 8))
        #print(f'{in_delta:02x} -> {out_delta:02x}; in_vals: {in_vals!r}')
        cnf = dnf.to_minimal_cnf(['-Decho'])
        #print(f'done')

        variables = [out_var[i] for i in range(8)] + [in_var[i] for i in range(8)]
        # model_dnf(self.s, (in_var, out_var), np.array([in_vals, sbox[in_vals]]).T, name or str(in_var).replace('in_', ''))
        model_cnf(self.s, variables, cnf, name or str(in_var).replace('in_', ''))

    def _model_sboxes(self):
        for rnd in range(self.numrounds):
            for row, col in product(range(4), range(4)):
                # input unknown, output unknwon

                # input known, output unknown

                # input known, output known

                sb_in_var = self.sbox_input_var[rnd][row, col]
                sb_out_var = self.sbox_output_var[rnd][row, col]

                sbox_in = self._sbox_in[rnd, row, col]
                sbox_out = self._sbox_out[rnd, row, col]

                if not self._sbox_out_unknown[rnd, row, col]:
                    # model solution set to ddt4
                    tbox_in = self._tbox_in[rnd, row, col]
                    tbox_out = self._tbox_out[rnd, row, col]

                    in_vals = get_ddt4_solution_set(sbox_in, tbox_in, sbox_out, tbox_out)
                    dnf = DNF.from_indices(16, (sbox[in_vals].astype(np.uint16) << 0) | (in_vals.astype(np.uint16) << 8))
                    cnf = dnf.to_minimal_cnf()

                    variables = [sb_out_var[i] for i in range(8)] + [sb_in_var[i] for i in range(8)]
                    model_cnf(self.s, variables, cnf, f'ddt4_box_{rnd}_{row}_{col}')
                    # model_dnf(self.s, (sb_in_var, sb_out_var), np.array([in_vals, sbox[in_vals]]).T, str(sb_in_var).replace('in_', ''))
                    continue

                # create input var
                tb_delta = 0
                if self._sbox_in_unknown[rnd, row, col]:
                    tb_in_var = self.tbox_input_var[rnd][row, col]
                else:
                    tb_delta = int(self._tbox_in[rnd][row, col])
                    if rnd == row == col == 0:
                        tb_delta = 1
                    assert tb_delta != -1
                    tb_in_var = sb_in_var

                tb_out_var = self.tbox_output_var[rnd][row, col]

                self._model_single_sbox(sb_in_var, sb_out_var, sbox_in, sbox_out, f'sbox_{rnd}_{row*4+col}')
                self._model_single_sbox(tb_in_var, tb_out_var, sbox_in, sbox_out, f'tbox_{rnd}_{row*4+col}', input_delta=tb_delta)

    def constrain_lr(self, lr: bytes):
        constraints = []

        if len(lr) != 32:
            raise ValueError()
        l = lr[:16]
        r = lr[16:]

        ctx = self.s.ctx
        for byte_idx, byte_val in enumerate(l):
            var = self.pt[byte_idx]
            constraints.append(var == BitVecVal(int(byte_val), 8, ctx))

        for byte_idx, byte_val in enumerate(r):
            var = Concat(self.key[byte_idx])
            constraints.append(var == BitVecVal(int(byte_val), 8, ctx))
        return constraints

    def get_rtk_value(self, m: Model, rnd: int):
        return np.array([rtk.get_value(m) for rtk in self.round_tweakeys[rnd]])


def find_collision(bit_char_file: str, start_arg: int, to_arg: int, iv_arg: str, tid: int):
    randomize()

    suffix = '_coll' if iv_arg == 'free' else '_full_coll'
    out_file = bit_char_file.replace('.npz', f'{suffix}.npz')
    log_file = bit_char_file.replace('.npz', f'{suffix}.log')

    with np.load(bit_char_file) as f:
        sbox_in = f.get('sbox_in')[start_arg:to_arg]
        sbox_out = f.get('sbox_out')[start_arg:to_arg]

        tbox_in = f.get('tbox_in')[start_arg:to_arg]
        tbox_out = f.get('tbox_out')[start_arg:to_arg]
        unknown = f.get('unknown')[:len(sbox_in)][start_arg:to_arg]
        forget = f.get('forget')[:len(sbox_in)][start_arg:to_arg]

        tweakeys = f.get('tweakeys')[start_arg:to_arg]

    if args.measure_prefix_filter:
        prefix_filter_effect = measure_prefix(sbox_in, sbox_out, tbox_in, tbox_out, unknown, args.measure_prefix_filter)
        print("prefix filter effect: ")
        print(f"  mean: 2^{np.log2(np.mean(prefix_filter_effect)):.1f}")

    # sanity check characteristic
    for i in range(len(tweakeys) - 1):
        assert np.all(tweakeys[i + 1] == update_tweakey(tweakeys[i]))
        rtk = np.bitwise_xor.reduce(tweakeys[i], axis=0) & tweakey_mask
        assert np.all(sbox_in[i + 1] == do_mix_cols(do_shift_rows(sbox_out[i] ^ rtk)))
        assert np.all((tbox_in[i + 1] == do_mix_cols(do_shift_rows(tbox_out[i]))) | unknown[i + 1])

    model_start = process_time()
    ctx = Context()
    contexts.append(ctx)
    s = Solver(ctx=ctx)
    skinny = CollisionSolver(s, sbox_in, sbox_out, tbox_in, tbox_out, tweakeys, unknown, forget)
    model_end = process_time()
    print(f'creating model took {precisedelta(model_end - model_start)}')

    numrounds = skinny.numrounds
    print(f'total probability for {numrounds} rounds: 2^{skinny.total_log_prob:.1f}')

    start = model_end
    msg_prefix = None
    import tqdm
    bar = tqdm.tqdm()
    for index in takewhile(lambda index: index == 1 or iv_arg == 'random', count(1)):
        if iv_arg == 'free':
            assumptions = []
        elif iv_arg == 'zero':
            assumptions = skinny.constrain_lr(np.zeros((32), dtype=np.uint8))
        elif iv_arg == 'random':
            msg_prefix, lr = generate_prefix(sbox_in, sbox_out, tbox_in, tbox_out, unknown)
            assert romulush_reduce(b'\0' * 32, msg_prefix, numrounds) == lr
            assumptions = skinny.constrain_lr(lr)
        elif iv_arg == 'random_once':
            msg_prefix, lr = generate_prefix(sbox_in, sbox_out, tbox_in, tbox_out, unknown)
            assert romulush_reduce(b'\0' * 32, msg_prefix, numrounds) == lr
            assumptions = skinny.constrain_lr(lr)
            print(f'msg_prefix = {msg_prefix.decode()}\nlr = {lr.hex()}')
        else:
            raise RuntimeError('forgot some value in switch/case')

        this_start = process_time()
        res = s.check(*assumptions)
        end = process_time()
        bar.set_description(f'[{tid:02d}/{index:04d}]  [{precisedelta(end-this_start)} / {precisedelta(end-start)}]\t{res}')

        if res != unsat or interrupted:
            break

    if res != sat:
        if res == unsat:
            with open(log_file, 'a') as f:
                if 'args' in globals():
                    f.write(f'args: {vars(args)}\n\n')
                f.write(f'using characteristic {bit_char_file} for {numrounds} rounds with p=2^{skinny.total_log_prob:.1f}\n')
                f.write(f'UNSAT after {precisedelta(end-start)}\n\n')
        # embed()
        return

    m = s.model()

    # sanity check result
    for i in range(numrounds):
        sbi = skinny.sbox_input_var[i].get_value(m)
        sbo = skinny.sbox_output_var[i].get_value(m)
        assert np.all(sbo == sbox[sbi])
        assert np.all(sbox[sbi] ^ sbox[sbi ^ sbox_in[i]] == sbox_out[i])
        # assert np.all(sbox[sbi] ^ sbox[sbi ^ tbox_in[i]] == tbox_out[i])
        # assert np.all(sbox[sbi ^ sbox_in[i]] ^ sbox[sbi  ^ sbox_in[i] ^ tbox_in[i]] == tbox_out[i])

        if i + 1 in range(numrounds):
            assert np.all(update_tweakey(skinny.get_rtk_value(m, i)) == skinny.get_rtk_value(m, i + 1))
            rtk = np.bitwise_xor.reduce(skinny.get_rtk_value(m, i), axis=0) & tweakey_mask
            rc = expanded_rc[i]
            sbo = skinny.sbox_output_var[i].get_value(m)
            sbi = skinny.sbox_input_var[i + 1].get_value(m)
            assert np.all(sbi == do_mix_cols(do_shift_rows(sbo ^ rtk ^ rc)))

    pt = skinny.pt.get_value(m)
    assert np.all(pt == skinny.sbox_input_var[0].get_value(m))
    key = skinny.key.get_value(m)
    m1 = skinny.m1.get_value(m)
    m2 = skinny.m2.get_value(m)

    # coll = np.array([pt, key, m1, m2])
    pt_delta = np.array([1] + [0] * 15, dtype=np.uint8).reshape(4, 4)

    a = np.zeros([2, numrounds + 1, 4, 4], np.uint8)
    b = np.zeros([2, numrounds + 1, 4, 4], np.uint8)

    a[0], t1 = skinny_verbose(pt, np.array([key, m1, m2]), numrounds)
    a[1], _ = skinny_verbose(pt ^ pt_delta, np.array([key, m1, m2]), numrounds)

    b[0], t2 = skinny_verbose(pt, np.array([key, m1, m2]) ^ tweakeys[0], numrounds)
    b[1], _ = skinny_verbose(pt ^ pt_delta, np.array([key, m1, m2]) ^ tweakeys[0], numrounds)

    a = a.transpose(1, 0, 2, 3)
    b = b.transpose(1, 0, 2, 3)

    coll_tweakeys_1 = np.array([key, m1, m2])
    coll_tweakeys_2 = np.array([key, m1, m2]) ^ tweakeys[0]

    lr = bytes(pt) + bytes(coll_tweakeys_1[0])
    m1 = bytes(coll_tweakeys_1[1:])
    m2 = bytes(coll_tweakeys_2[1:])

    print(f'difference after {numrounds} rounds: ', end='')
    print(hexlify(bytes((a[numrounds] ^ b[numrounds]))).decode())

    if msg_prefix is not None:
        print(f'pre = unhexlify(b"{hexlify(msg_prefix).decode()}")')
    print(f'lr  = unhexlify(b"{hexlify(lr).decode()}")')
    print(f'm1  = unhexlify(b"{hexlify(m1).decode()}")')
    print(f'm2  = unhexlify(b"{hexlify(m2).decode()}")')

    with open(log_file, 'a') as f:
        f.write(f'args: {vars(args)}\n\n')
        f.write(f'using characteristic {bit_char_file} for {numrounds} rounds with p=2^{skinny.total_log_prob:.1f}\n')
        f.write(f'found collision in {precisedelta(end-start)}\n')
        f.write(f'difference after {numrounds} rounds: ')
        f.write(hexlify(bytes((a[numrounds] ^ b[numrounds]))).decode() + '\n')
        if msg_prefix is not None:
            f.write(f'pre:  {hexlify(msg_prefix).decode()}\n')
        f.write(f'L||R: {hexlify(lr).decode()}\n')
        f.write(f'M1:   {hexlify(m1).decode()}\n')
        f.write(f'M2:   {hexlify(m2).decode()}\n\n')

    np.savez(out_file, args=vars(args), numrounds=numrounds, execution_a=a, execution_b=b, lr=lr, m1=m1, m2=m2)
    with zipfile.ZipFile(out_file, mode='a') as zf:
        zf.write(__file__, path.basename(__file__))
        zf.write(bit_char_file)

    assert np.all(a[numrounds] == b[numrounds])
    assert m1 != m2 and romulush_reduce(lr, m1, numrounds) == romulush_reduce(lr, m2, numrounds)
    # embed()

    return msg_prefix, lr, m1, m2


def main(bit_char_file: str):
    numthreads = len(os.sched_getaffinity(0)) if args.iv == 'random' else 1

    q = Queue()

    def run_task(idx: int):
        coll = find_collision(bit_char_file, args.start, args.to, args.iv, idx)
        q.put(coll)

    if numthreads == 1:
        run_task(0)
        return

    threads = [Process(target=run_task, args=(idx,), daemon=True) for idx in range(numthreads)]

    for t in threads:
        t.start()

    try:
        q.get()
    except KeyboardInterrupt:
        pass

    for t in threads:
        t.terminate()
    for t in threads:
        t.join()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bit_char_file', type=str, help='differential characteristic in .npz format')
    parser.add_argument('--track-unsat-core', action='store_true', help='Track unsat core with z3. Will slow down model.')
    parser.add_argument('--iv', choices=['free', 'zero', 'random', 'random_once'], default='free', help='constrain the iv to either zero, a random iv returned by the compression function, or not contain it')
    parser.add_argument('--measure-prefix-filter', nargs='?', default=0, const=1000, help='Measure the effect of filtering prefixes based on first 2 rounds')
    parser.add_argument('--from', type=int, default=None, dest='start', help='start model in round 0..numrounds-1')
    parser.add_argument('--to', type=int, default=None, help='stop model in round 1..numrounds (exclusive)')

    args = parser.parse_args()

    if not args.track_unsat_core:
        # disable assert and track
        def _a_and_t(self, a, b):
            self.assert_exprs(a)
        Solver.assert_and_track = _a_and_t
    else:
        def _raise_exc(self, *args, **kwargs):
            raise NotImplementedError()
        Solver.add = _raise_exc

    main(args.bit_char_file)
