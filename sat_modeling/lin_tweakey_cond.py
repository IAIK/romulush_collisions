#!/usr/bin/env python3

from util import *
import lin_util
from read_sol import TruncatedCharacteristic
import numpy as np
from crypto_solver import BaseIndices

from PIL import Image

from IPython import embed
from dataclasses import dataclass
from itertools import permutations

from sage.all import matrix, GF

def cat(*args):
    return np.concatenate(tuple(args))

@dataclass
class Indices(BaseIndices):
    tkey: np.ndarray
    sbox_in: np.ndarray
    sbox_out: np.ndarray
    numrounds: int

    def __init__(self, numrounds):
        super().__init__()

        self.tkey = self.make_index_array((2, 4, 4, 8))
        self.sbox_in = self.make_index_array((numrounds, 4, 4, 8))
        self.sbox_out = self.make_index_array((numrounds, 4, 4, 8))
        self.numrounds = numrounds

    def describe_idx_array(self, index_array: np.ndarray):
        """
        convenience function to return the underlying array name and unraveled
        index for each linear index in `index_array`.
        """
        variables = {k: v for k, v in vars(self).items() if isinstance(v, np.ndarray)}

        if np.any((index_array < 0) | (index_array >= self.numvars)):
            raise IndexError('index out of bounds')

        res = [None] * np.product(index_array.shape, dtype=int)
        for i, needle in enumerate(index_array.flatten()):
            for k, v in variables.items():
                start, stop = v.flatten()[[0, -1]]
                rng = range(start, stop + 1)

                if needle in rng:
                    idx = np.unravel_index(rng.index(needle), v.shape)
                    res[i] = k + str(list(idx))
                    #res[i] = str(f'{idx[1]}{idx[2]}')
                    break
            else:
                assert False, f"index {needle} not found?"
        return np.array(res, dtype=object).reshape(index_array.shape)

    def describe_lin_expr(self, lin_expr: np.ndarray):
        """
        convenience function to describe (multiple) linear expressions over the
        state given as rows of a matrix.
        """
        if lin_expr.shape[-1] != self.numvars:
            raise ValueError('shape[-1] must match number of variables')

        result = [None] * np.product(lin_expr.shape[:-1], dtype=int)
        for i, expr in enumerate(lin_expr.reshape(-1, self.numvars)):
            indices, = np.where(expr != 0)
            result[i] = ' ^ '.join(self.describe_idx_array(indices))
        return np.array(result, dtype=object).reshape(lin_expr.shape[:-1])


    def select(self, index_array: np.ndarray):
        """
        return a bit mask with the bits set according to index_array according
        to the shape of index_array
        """
        res = np.zeros((np.product(index_array.shape), self.numvars), dtype=np.uint8)
        res[np.arange(res.shape[0]), index_array.flatten()] = 1

        return res.reshape(index_array.shape + (self.numvars,))

@dataclass
class PartialCharacteristics:
    tweakeys: np.ndarray
    relevant_sboxes: np.ndarray
    sbox_inputs: np.ndarray
    sbox_outputs: np.ndarray
    probs: np.ndarray


@dataclass
class LinearConditions:
    char: TruncatedCharacteristic
    indices: Indices

    mat: matrix
    mat_arr: np.ndarray
    kernel: matrix


    tk_mat: np.ndarray
    packed_tk_mat: np.ndarray
    tk_rank: int

    # degrees of freedom (0..16) for each S-box after tweakey difference is fixed
    # indexes by (rnd, row, col)
    sbox_freedom: np.ndarray

    res: Optional[PartialCharacteristics]

    def __init__(self, char: TruncatedCharacteristic):
        self.char = char
        self.indices = Indices(char.numrounds)

        self.mat_arr = build_mat(self.char, self.indices)
        self.mat = as_mat_gf2(self.mat_arr)
        self.kernel = self.mat.right_kernel_matrix()

        self.tk_slice = slice(self.indices.tkey.flatten()[0], self.indices.tkey.flatten()[-1] + 1)
        self.tk_rank = self.kernel[:, self.tk_slice].rank()
        self._init_sbox_freedom()
        self._init_iterate_tweakeys()

        self.res = None
        self.tweakey_count = None


    def _prepare_tweakey_iteration(self):
        pass


    def _init_sbox_freedom(self):
        self.sbox_freedom = np.zeros(self.char.shape, np.uint8)
        for rnd, row, col in zip(*np.nonzero(self.char.sbox)):
            in_idxes = self.indices.sbox_in[rnd, row, col]
            out_idxes = self.indices.sbox_out[rnd, row, col]

            in_mat = self.kernel[self.tk_rank:, list(in_idxes)]
            out_mat = self.kernel[self.tk_rank:, list(out_idxes)]

            self.sbox_freedom[rnd, row, col] = in_mat.rank() + out_mat.rank()

    def _get_best_col_perm(self, relevant_cols) -> list:
        """
        identify column ordering with the most consecutive pivots
        (consecutive pivots allow us to get some S-box transitions without guessing)
        """
        max_free_sboxes = self.tk_rank // 16
        best_free_sboxes = -1
        for start_cols in permutations(range(relevant_cols.shape[0]), max_free_sboxes):
            mat = self.kernel[:self.tk_rank, list(np.array(relevant_cols[list(start_cols)]).flatten())]
            mat.echelonize()
            max_consecutive_pivot = None
            for i, e in enumerate(mat.pivots()):
                if i != e:
                    break
                max_consecutive_pivot = i
            num_free_sboxes = (max_consecutive_pivot + 1) // 16
            if num_free_sboxes == max_free_sboxes:
                best_free_sboxes = num_free_sboxes
                break
            if num_free_sboxes > best_free_sboxes:
                best_free_sboxes = num_free_sboxes

        col_perm = list(start_cols) + list(x for x in range(relevant_cols.shape[0]) if x not in start_cols)
        return col_perm

    def _init_iterate_tweakeys(self):
        """
        return a list of valid tweakey differences for the given truncated characteristic
        """

        #relevant_cols.append(list(out_idxes) + list(in_idxes))
        self.relevant_sboxes = np.where(self.char.sbox & (self.sbox_freedom == 0))
        self.relevant_cols = np.concatenate((self.indices.sbox_out[self.relevant_sboxes], self.indices.sbox_in[self.relevant_sboxes]), axis=-1)
        col_perm = self._get_best_col_perm(self.relevant_cols)
        self.relevant_cols = self.relevant_cols[col_perm]
        self.relevant_sboxes = np.array(self.relevant_sboxes).T[col_perm]


        tk_indices = list(range(self.tk_slice.start, self.tk_slice.stop))
        assert self.tk_slice.step is None and len(tk_indices) == 256
        self.tk_mat = self.kernel[:self.tk_rank, list(self.relevant_cols.flatten()) + tk_indices]
        self.tk_mat.echelonize()
        self.tk_mat = as_numpy(self.tk_mat)
        self.packed_tk_mat = np.packbits(self.tk_mat, axis=1, bitorder='little').astype(np.uint16)
        self.packed_tk_mat = (self.packed_tk_mat[:, 0::2] << 0) | (self.packed_tk_mat[:, 1::2] << 8)


    def count_tweakeys(self) -> int:
        self.tweakey_count = lin_util.count_good_masks(self.packed_tk_mat[:, :-16].copy(), ddt)
        return self.tweakey_count

    def iterate_tweakeys(self) -> PartialCharacteristics:
        if self.res is not None:
            return self.res

        _res = lin_util.find_good_masks(self.packed_tk_mat[:, :-16].copy(), ddt)
        good_masks = _res['mask']
        prob = _res['log_prob']
        bitvecs = (good_masks[:, None] >> np.arange(self.tk_mat.shape[0], dtype=np.uint8)[None, :]) & 1

        tweakeys = ((bitvecs @ self.tk_mat[:, -256:]) % 2).reshape(len(bitvecs), *self.indices.tkey.shape)
        tweakeys = np.packbits(tweakeys, bitorder='little', axis=-1)
        assert tweakeys.shape[-1] == 1
        tweakeys = tweakeys[..., 0]
        assert self.tweakey_count is None or self.tweakey_count == len(tweakeys), f'expected {self.tweakey_count}, got {len(tweakeys)}'
        self.tweakey_count = len(tweakeys)

        sboxes = np.packbits((bitvecs @ self.tk_mat[:, :-256]) % 2, bitorder='little', axis=-1)
        inputs, outputs = sboxes[:, 1::2], sboxes[:, 0::2]

        probs = np.sum(np.log2(ddt[inputs, outputs] / 256), axis=1)
        assert np.all(ddt[inputs, outputs] > 0)
        assert np.allclose(probs, prob)

        self.res = PartialCharacteristics(tweakeys, self.relevant_sboxes, inputs, outputs, probs)
        return self.res

    def __repr__(self):
        res = [
            f'LinearConditions for {self.char.filename}',
            f'2^{self.tk_rank} tweakeys, 2^{self.kernel.rank()} total characteristics',
            f'{len(self.relevant_sboxes)} active S-boxes fixed by tweakey difference',
        ]
        if self.tweakey_count is not None:
            res.append(f'{self.tweakey_count} valid tweakeys')

        return '\n'.join(res)


def _build_linear_layer_constraints(numrounds: int,
                                    indices: Indices) -> np.ndarray:
    constraints = np.zeros((0, 4, 4, 8, indices.numvars), dtype=np.uint8)

    dst_idx = 0
    tk_state = indices.select(indices.tkey)

    for rnd in range(numrounds):
        state = indices.select(indices.sbox_out[rnd])

        # add round tweakey
        #tweakey = np.bitwise_xor.reduce(indices.select(indices.tkey[rnd]), axis=0)
        state[:2] ^= tk_state[0, :2] ^ tk_state[1, :2]
        tk_state = _update_tweakey(tk_state, indices)

        # shift rows
        state = state.reshape(16, 8, indices.numvars)[sr_mapping]

        # mix columns
        state[1] ^= state[2]
        state[2] ^= state[0]
        state[3] ^= state[2]
        state = state[[3, 0, 1, 2]]

        if rnd + 1 in range(numrounds):
            out_state = indices.select(indices.sbox_in[rnd + 1])
            state ^= out_state
        else:
            # this servers as the constraint that the difference after
            # the final linear layer is zero
            ...

        constraints = cat(constraints, state[None, ...])

    return constraints.reshape(-1, indices.numvars)

def _update_tweakey(tk_state: np.ndarray, indices: Indices):
    if tk_state.shape != (2, 4, 4, 8, indices.numvars):
        pass

    tk_state = tk_state.reshape(2, 16, 8, indices.numvars)

    tk_state = tk_state[:, tweakey_perm]

    tmp = tk_state.copy()
    tmp[0, :8, 1:8] = tk_state[0, :8, 0:7]
    tmp[0, :8, 0] = tk_state[0, :8, 7] ^ tk_state[0, :8, 5]
    tmp[1, :8, 0:7] = tk_state[1, :8, 1:8]
    tmp[1, :8, 7] = tk_state[1, :8, 0] ^ tk_state[1, :8, 6]
    tk_state = tmp
    return tk_state.reshape(2, 4, 4, 8, indices.numvars)

def _build_truncated_constraints(char: TruncatedCharacteristic, indices: Indices):
    sbi = indices.select(indices.sbox_in[char.sbox == 0])
    sbo = indices.select(indices.sbox_out[char.sbox == 0])
    tk = np.bitwise_xor.reduce(indices.select(indices.tkey), axis=0)[char.tkey[0] == 0]

    constraints = cat(sbi, sbo, tk)

    return constraints.reshape(-1, indices.numvars)

def build_mat(char: TruncatedCharacteristic, indices: Indices) -> np.ndarray:
    mat_lin_layer = _build_linear_layer_constraints(char.numrounds, indices)
    mat_trunc = _build_truncated_constraints(char, indices)
    return cat(mat_lin_layer, mat_trunc)


def main(char: TruncatedCharacteristic):
    # indices = Indices(char.numrounds)
    # arr = build_mat(char, indices)

    # print(f'{arr.shape=}')
    # m = as_mat_gf2(arr, sparse=False)
    # rkb = m.right_kernel_matrix()

    # print(f'right kernel: {rkb.nrows()}x{rkb.ncols()}')

    # arr = np.array(rkb, dtype=np.uint8)
    # nz_indices = np.any(arr != 0, axis=0)
    # assert np.sum(nz_indices) == 8 * (2 * np.sum(char.sbox) + 2 * np.max(np.sum(char.tkey, axis=(1,2))))
    # arr_nz = arr[:, nz_indices]

    # tk_slice = slice(indices.tkey.flatten()[0], indices.tkey.flatten()[-1] + 1)
    # sbi_slice = slice(indices.sbox_in.flatten()[0], indices.sbox_in.flatten()[-1] + 1)
    # sbo_slice = slice(indices.sbox_out.flatten()[0], indices.sbox_out.flatten()[-1] + 1)

    # tk_mat = as_mat_gf2(np.array(rkb, dtype=np.uint8)[:, tk_slice].copy())
    # tk_mat = tk_mat.submatrix(0, 0, tk_mat.rank(), tk_mat.ncols())

    # print(f'tweakey basis: {tk_mat.nrows()}x{tk_mat.ncols()}')

    lin_cond = LinearConditions(char)
    res = lin_cond.iterate_tweakeys()
    tweakeys, probs = res.tweakeys, res.probs
    embed()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterate-tweakeys', action='store_true', help='count the number of tweakey differences satisfying the s-box transitions')
    parser.add_argument('--count-tweakeys', action='store_true', help='count the number of tweakey differences satisfying the s-box transitions')
    parser.add_argument('byte_char_files', type=str, nargs='*', help='byte-level differential characteristic found by MILP')

    args = parser.parse_args()
    for fname in args.byte_char_files:
        char = TruncatedCharacteristic(fname)
        lin_cond = LinearConditions(char)
        if args.count_tweakeys:
            print(f'{lin_cond.count_tweakeys()=}')
        if args.iterate_tweakeys:
            lin_cond.iterate_tweakeys()
        print(f'{lin_cond!r}')

    #main(char)

