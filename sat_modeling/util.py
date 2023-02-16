from __future__ import annotations
from z3 import *
from typing import *
from read_sol import *
from cnf_util import *

from pycryptosat import Solver as CmsSolver

from functools import reduce, lru_cache
from itertools import product

import numpy as np
from os import path

from _util import *
from skinny import *

from SimpleCache import Cache
cache = Cache()

connection_poly = np.array([0] * 9)
connection_poly[[0, 2, 8]] = 1

sbox = np.array((
  101,  76,   106,  66,   75,   99,   67,   107,  85,   117,  90,   122,  83,   115,  91,   123,
  53,   140,  58,   129,  137,  51,   128,  59,   149,  37,   152,  42,   144,  35,   153,  43,
  229,  204,  232,  193,  201,  224,  192,  233,  213,  245,  216,  248,  208,  240,  217,  249,
  165,  28,   168,  18,   27,   160,  19,   169,  5,    181,  10,   184,  3,    176,  11,   185,
  50,   136,  60,   133,  141,  52,   132,  61,   145,  34,   156,  44,   148,  36,   157,  45,
  98,   74,   108,  69,   77,   100,  68,   109,  82,   114,  92,   124,  84,   116,  93,   125,
  161,  26,   172,  21,   29,   164,  20,   173,  2,    177,  12,   188,  4,    180,  13,   189,
  225,  200,  236,  197,  205,  228,  196,  237,  209,  241,  220,  252,  212,  244,  221,  253,
  54,   142,  56,   130,  139,  48,   131,  57,   150,  38,   154,  40,   147,  32,   155,  41,
  102,  78,   104,  65,   73,   96,   64,   105,  86,   118,  88,   120,  80,   112,  89,   121,
  166,  30,   170,  17,   25,   163,  16,   171,  6,    182,  8,    186,  0,    179,  9,    187,
  230,  206,  234,  194,  203,  227,  195,  235,  214,  246,  218,  250,  211,  243,  219,  251,
  49,   138,  62,   134,  143,  55,   135,  63,   146,  33,   158,  46,   151,  39,   159,  47,
  97,   72,   110,  70,   79,   103,  71,   111,  81,   113,  94,   126,  87,   119,  95,   127,
  162,  24,   174,  22,   31,   167,  23,   175,  1,    178,  14,   190,  7,    183,  15,   191,
  226,  202,  238,  198,  207,  231,  199,  239,  210,  242,  222,  254,  215,  247,  223,  255
), dtype=np.uint8)

round_constants = np.array([
    0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E,
    0x1d, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E, 0x1C, 0x38,
    0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A, 0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04,
    0x09, 0x13, 0x26, 0x0C, 0x19, 0x32, 0x25, 0x0A, 0x15, 0x2A, 0x14, 0x28, 0x10, 0x20
])

sr_mapping = np.array([[ 0,  1,  2,  3],
                       [ 7,  4,  5,  6],
                       [10, 11,  8,  9],
                       [13, 14, 15, 12]])

isr_mapping = np.array([[ 0,  1,  2,  3],
                        [ 5,  6,  7,  4],
                        [10, 11,  8,  9],
                        [15, 12, 13, 14]])

expanded_rc = np.zeros((len(round_constants), 4, 4), np.uint8)
expanded_rc[:, 0, 0] = round_constants & 15
expanded_rc[:, 1, 0] = round_constants >> 4
expanded_rc[:, 2, 0] = 0x2

inv_sbox = np.zeros_like(sbox)
inv_sbox[sbox] = np.arange(len(sbox), dtype=sbox.dtype)

tweakey_mask = np.array([0xFF] * 8 + [0x00] * 8).reshape(4, 4)
tweakey_perm = np.array([9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7])
inv_tweakey_perm = np.empty_like(tweakey_perm)
inv_tweakey_perm[tweakey_perm] = np.arange(len(tweakey_perm), dtype=tweakey_perm.dtype)



mixing_mat = np.array([
    [1, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 1, 1, 0],
    [1, 0, 1, 0],
])

inv_mixing_mat = np.array([
    [0, 1, 0, 0],
    [0, 1, 1, 1],
    [0, 1, 0, 1],
    [1, 0, 0, 1],
])


# hack to allow subscripting BitVec
def _subscript(bvr: BitVecRef, index: Union[int, slice]):
    if isinstance(index, slice):
        return Extract(index.stop - 1, index.start, bvr)
    index = int(index)
    return Extract(index, index, bvr)


BitVecRef.__getitem__ = _subscript


def itershape(shape: Tuple[int, ...]):
    return product(*[range(s) for s in shape])


def update_tweakey(tweakeys: List[np.ndarray]):
    tweakeys = np.array(tweakeys, dtype=np.uint8).reshape(3, 16)

    # permute
    tweakeys = tweakeys[:, tweakey_perm]

    # LSFRs to update TK2 and TK3
    i = slice(8)
    tweakeys[1][i] = (tweakeys[1][i] << 1) + ((tweakeys[1][i] >> 5 & 1) ^ (tweakeys[1][i] >> 7 & 1))
    tweakeys[2][i] = (tweakeys[2][i] >> 1) + (((tweakeys[2][i] & 1) ^ (tweakeys[2][i] >> 6 & 1)) * 128)

    return tweakeys.reshape(3, 4, 4)


def echelon_form_gf2(arr: np.ndarray) -> np.ndarray:
    mat_sage = as_mat_gf2(arr)
    mat_sage.echelonize()
    return as_numpy(mat_sage[:mat_sage.rank()])


def get_ddt():
    ddt_path = path.join(path.dirname(path.realpath(__file__)), "ddt.npy")
    try:
        return np.load(ddt_path)
    except FileNotFoundError:
        pass

    ddt = np.zeros((len(sbox), len(sbox)), dtype=np.int16)

    for in_delta in range(len(sbox)):
        in_val = np.arange(len(sbox), dtype=sbox.dtype)
        out_delta = sbox[in_val] ^ sbox[in_val ^ in_delta]
        out_delta, counts = np.unique(out_delta, return_counts=True)

        ddt[in_delta, out_delta] = counts

    np.save(ddt_path, ddt)
    return ddt


ddt = get_ddt()
ddt4 = DDT4(sbox, 'sbox.npz')


def xddt(alpha: int, beta: int) -> np.ndarray:
    in_vals = np.arange(256)
    in_vals = in_vals[sbox[in_vals] ^ sbox[in_vals ^ alpha] == beta]
    return in_vals


def get_restricted_ddt(in_vals: np.ndarray) -> np.ndarray:
    """
    returns the 256x256 table RDDT, that satisfies
    RDDT[delta, nabla] =  #(S(x) + S(x + delta) == nabla and x + delta in in_vals for all x in in_vals)
    """

    rddt = np.zeros((len(sbox), len(sbox)), dtype=np.int16)

    for in_delta in range(len(sbox)):

        a = in_vals
        b = in_vals ^ in_delta

        _, comm_a, comm_b = np.intersect1d(a, b, return_indices=True)

        comm_a.sort()
        comm_b.sort()

        assert np.all(comm_a == comm_b)

        a = a[comm_a]
        b = b[comm_b]

        out_delta = sbox[a] ^ sbox[b]
        out_delta, counts = np.unique(out_delta, return_counts=True)

        rddt[in_delta, out_delta] = counts

    return rddt

    from IPython import embed
    embed()


class SkinnyState:
    def __init__(self, name: str, use_bits=False, ctx=None, mask=np.full((4, 4), True)):
        if not np.all(mask) and use_bits:
            raise NotImplementedError('mask and use_bits not implemented')

        mask = mask.copy().reshape(-1)
        if use_bits:
            self._vars = [[BitVec(f'{name}_{i}_{j}', 1, ctx=ctx) for j in range(8)] for i in range(16)]
        else:
            self._vars = [BitVec(f'{name}_{i}', 8, ctx=ctx) if mask[i] else 0 for i in range(16)]
        self.name = name
        self._constraint: Optional[np.ndarray] = None

    @staticmethod
    def from_vars(variables: Union[List[BitVec], List[List[Bool]]]):
        if len(variables) != 16:
            raise ValueError(f"expected list of 16 variables, got {len(variables)}")
        for v in variables:
            if len(v) != 8 and not isinstance(v, BitVecRef):
                raise ValueError(f"expected list of 8 Bools or BitVec, got {v!r}")

        res = object.__new__(SkinnyState)
        res._constraint = None
        res.name = None
        res._vars = variables
        return res

    def __getitem__(self, args: Union[int, Tuple[int, int]]):
        try:
            row, col = args
            return self._vars[row * 4 + col]
        except TypeError:
            pass

        return self._vars[args]

    def as_array(self) -> np.ndarray:
        return np.array(self._vars, dtype=object).reshape(4, 4)

    def constrain(self, s: Solver, value):
        self._constraint = value
        for row, col in product(range(4), range(4)):
            var = self[row, col]
            val = value[row][col]

            if isinstance(var, list):
                for i, e in enumerate(reversed(var)):
                    cond = e == bool(val & (1 << i))
                    s.add(cond)
            else:
                cond = var == BitVecVal(val, 8)
                s.add(cond)

    def get_value(self, m: Model, default_value=None):
        result = np.zeros((4, 4), dtype=np.uint8 if default_value is None else type(default_value))
        for row, col in product(range(4), range(4)):
            var = self[row, col]
            val = 0

            if var == 0:
                result[row, col] = 0
                continue

            if isinstance(var, list):
                for i, e in enumerate(reversed(var)):
                    val |= m.eval(e).as_long() << i
            else:
                # val = m.eval(var).as_long()
                eval_var = m.eval(var)
                val = eval_var.as_long() if not eval_var.eq(var) else default_value

            result[row, col] = val

        return result

    def __repr__(self):
        if self.name is not None:
            return f'SkinnyState({self.name!r})'

        return f'SkinnyState({self._vars!r})'


class LfsrState:
    def __init__(self, name: str, connection_poly: List[int], ctx=None):
        """
        connection_poly: list of exponents of the connection polynomial
        """
        assert connection_poly[0] == connection_poly[-1] == 1

        self.ctx = ctx
        self.name = name
        self.vars = dict()
        self.degree = len(connection_poly) - 1
        self.poly = np.array(connection_poly)

        self.constraints_generated = False

    def __repr__(self):
        return f'LfsrState({self.name!r}, {self.poly!r})'

    def get_constraints(self, ref_indices: Optional[range] = None):
        if ref_indices is None:
            start = (max(self.vars.keys()) + min(self.vars.keys()) - self.degree) // 2
            ref_indices = range(start, start + self.degree)

        start = ref_indices.start
        for i in ref_indices:
            self.get_bit(i)

        constraints = []
        self.constraints_generated = True
        for i in sorted(self.vars.keys()):
            if i in ref_indices:
                continue

            offset = i - start
            mask = self.get_bit_mask(offset)
            variables = [self.vars[start + i] for i, e in enumerate(mask) if e]
            total = reduce(lambda a, b: a ^ b, variables)

            constraints.append(self.vars[i] == total)

        return constraints

    def get_bit(self, index: int) -> BitVec:
        if index in self.vars:
            return self.vars[index]

        if self.constraints_generated:
            print("[WARN] generating bit for LFSR after constraints have been generated")

        var = BitVec(f'{self.name}_{index}', 1, ctx=self.ctx)
        self.vars[index] = var
        return var

    def get_bit_range(self, start_idx, numbits=8):
        return [self.get_bit(i) for i in range(start_idx, start_idx + numbits)]

    @lru_cache(maxsize=None)
    def get_bit_mask(self, index: int) -> np.array:
        """
        returns a mask over the indices [0, ..., len(self.poly) - 1] that can
        be used to calculate the bit at index `index`.
        """
        result = np.zeros(self.degree, dtype=int)

        if index in range(self.degree):
            result[index] = 1
            return result

        assert self.poly[0] == self.poly[self.degree] == 1

        if index >= self.degree:
            for i in range(self.degree):
                if self.poly[i] == 1:
                    offset = self.degree - i
                    result ^= self.get_bit_mask(index - offset)
            return result

        if index < 0:
            for i in range(1, self.degree + 1):
                if self.poly[i] == 1:
                    result ^= self.get_bit_mask(index + i)
            return result


def apply_matrix(byte: List[Bool], matrix: np.array, exponent: int):
    matrix = np.linalg.matrix_power(matrix, exponent) % 2

    res = [None] * 8

    for row in range(8):
        total = None
        for col in range(8):
            if not matrix[row, col]:
                continue
            total = Xor(total, byte[col]) if total is not None else byte[col]

        res[row] = total
    return res


def apply_perm(vec: List, perm: List[int], repeats: int):
    for _ in range(repeats):
        new_vec = [None] * len(vec)
        for i, e in enumerate(perm):
            new_vec[i] = vec[e]
        vec = new_vec
    return vec


def do_inv_shift_rows(state):
    return state.flatten()[isr_mapping]


def do_shift_rows(state):
    return state.flatten()[sr_mapping]


def do_inv_mix_cols(state):
    result = np.zeros_like(state)
    for col in range(4):
        result[:, col] = np.bitwise_xor.reduce(inv_mixing_mat * state[:, col], axis=-1)
    return result


def do_mix_cols(state):
    result = np.zeros_like(state)
    for col in range(4):
        result[:, col] = np.bitwise_xor.reduce(mixing_mat * state[:, col], axis=-1)
    return result


def skinny_verbose(pt: np.ndarray, tk: np.ndarray, numrounds: int):
    states = np.zeros((numrounds + 1, 4, 4), np.uint8)
    tweakeys = np.zeros((numrounds, 3, 4, 4), np.uint8)
    states[0] = pt
    tweakeys[0] = tk

    for r in range(numrounds):
        rc = expanded_rc[r]
        states[r + 1] = do_mix_cols(do_shift_rows(sbox[states[r]] ^ (np.bitwise_xor.reduce(tweakeys[r], axis=0) & tweakey_mask) ^ rc))
        if r + 1 in range(numrounds):
            tweakeys[r + 1] = update_tweakey(tweakeys[r])
    return states, tweakeys


def model_cnf(s: Solver, variables: List[BitVec], cnf: CNF, name: Optional[str] = None, extra_var = []):
    true_op = lambda x : x == 1
    false_op = lambda x : x == 0

    variables = [None] + list(variables)

    for i, clause in enumerate(cnf):
        condition = clause.apply_to_variables(variables, true_op, false_op) + extra_var
        if name is not None:
            s.assert_and_track(Or(condition), name + f"___{i}")
        else:
            s.add(Or(condition))


def model_cnf_cms(s: CmsSolver, variables: List[int], cnf: CNF):
    variables = np.array([0] + list(variables), dtype=np.uint32)
    variables[1:] += 1

    translated_cnf = cnf.translate(variables)
    s.add_clauses(np.array(translated_cnf))


def model_dnf(s: Solver, variables: List[BitVec], transitions: List[List[int]], name: Optional[str] = None):
    clauses = [None] * len(transitions)
    assert len(transitions[0]) == len(variables)
    for i, vals in enumerate(transitions):
        clauses[i] = And([var == int(val) for var, val in zip(variables, vals)])

    if name is not None:
        s.assert_and_track(Or(clauses), name)
    else:
        s.add(Or(clauses))


def get_ddt4_solution_set(sbox_in, tbox_in, sbox_out, tbox_out):
    in_vals = np.arange(256, dtype=np.uint8)
    in_vals = in_vals[np.all([sbox[in_vals] ^ sbox[in_vals ^ sbox_in] == sbox_out,
                              sbox[in_vals] ^ sbox[in_vals ^ tbox_in] == tbox_out,
                              sbox[in_vals ^ sbox_in] ^ sbox[in_vals ^ sbox_in ^ tbox_in] == tbox_out], axis=0)]

    assert len(in_vals) == ddt4[sbox_in, tbox_in, sbox_out, tbox_out]
    assert np.all(sbox[in_vals] ^ sbox[in_vals ^ sbox_in] == sbox_out)
    assert np.all(sbox[in_vals] ^ sbox[in_vals ^ tbox_in] == tbox_out)
    assert np.all(sbox[in_vals ^ tbox_in] ^ sbox[in_vals ^ tbox_in ^ sbox_in] == sbox_out)
    assert np.all(sbox[in_vals ^ sbox_in] ^ sbox[in_vals ^ sbox_in ^ tbox_in] == tbox_out)
    return in_vals


def get_solution_set(in_delta, out_delta):
    in_vals = np.arange(256, dtype=np.uint8)
    in_vals = in_vals[sbox[in_vals] ^ sbox[in_vals ^ in_delta] == out_delta]
    return in_vals


def precisedelta(time_range: float):
    res = ''
    days = int(time_range / 86_400)
    hours = int(time_range / 3_600) % 24
    minutes = int(time_range / 60) % 60
    seconds = int(time_range) % 60
    millis = int(time_range * 1_000) % 1000
    # micros = int(time_range * 1_000_000) % 1000
    # nanos = int(time_range * 1_000_000_000) % 1000

    if days > 0:
        res += f'{days} days '
    if hours > 0:
        res += f'{hours} h '
    if minutes > 0:
        res += f'{minutes} m '
    if seconds > 0:
        res += f'{seconds} s '
    if millis > 0:
        res += f'{millis} ms '
    # if micros > 0:
    #      res += f'{micros} us '
    # if nanos > 0:
    #      res += f'{nanos} ns '
    res = res.strip()
    if res == '':
        res = '0 s'
    return res
