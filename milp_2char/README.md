# Dual (cellwise) characteristics for SKINNY-n/3n

## Base model

Find 2 cellwise characteristics (activity patterns) for SKINNY-n/3n:

- $`\delta`$ (state variables `X`, `Y`, round tweakey variable `K`) is the main characteristic between two evaluations of SKINNY as part of two separate compression function evaluations. Model is more or less the same as written by Felix, for `TK2` setting.

- $`\tau`$ (state variable `T`, unknown status `U` where `T` is considered undetermined; active cells that become undetermined at an S-box output have forgotten status `F`). Model for `T` is as above but for `SK` setting.

## Cost function variants

The cost unit 1 corresponds to one active S-box, i.e., weight $`-\log_2(\text{MDP}) = 2`$ for the SKINNY S-box.

- **Naive** bound (`unknownonly=True`, file suffix `u`): ignoring any knowledge of $`\tau`$; each active S-box in $`\delta`$ costs 2.

- **Zero only** bound (`zeroonly=True`, file suffix `z`): like above, but taking into account that when $`\tau`$ is known to be inactive, an S-box in $`\delta`$ costs only 1.

- **Dual** bound (*default*): like above, but part of $`\tau`$ can be determined; it costs 1 extra for each active S-box (whose input and output are determined) in $`\tau`$, though, unless that S-box is also active in $`\delta`$. In other words, an S-box that is active and determined in both $`\delta`$ and $`\tau`$ costs only 1 (as the MDP in the dual DDT is the same as in the normal DDT).

## Solving and Output

Set `roundsrange` (range of round numbers to solve) and option in last line of `skinny_milp.py` and run with the python bin shipped with `gurobipy`. 

The produced `.tex` files can be compiled with `latexmk collision/skinny_rXX.tex`. 
$`\delta`$ is blue, $`\tau`$ is marked in red with `x` for fixed nonzero differences, blank for zero differences, `?` for unknown differences.
Cost model per cell is annotated in green.

## Results

| File                    | Rounds | Naive `u` | Zero `z` | Dual |
|------------------------ |--------|-----------|----------|------|
| `collision/skinny_r1*`  | 1      |         0 |        0 |    0 |
| `collision/skinny_r2*`  | 2      |         0 |        0 |    0 |
| `collision/skinny_r3*`  | 3      |         4 |        2 |    2 |
| `collision/skinny_r4*`  | 4      |         4 |        2 |    2 |
| `collision/skinny_r5*`  | 5      |         4 |        2 |    2 |
| `collision/skinny_r6*`  | 6      |        16 |       11 |   11 |
| `collision/skinny_r7*`  | 7      |        22 |       16 |   16 |
| `collision/skinny_r8*`  | 8      |        34 |       25 |   25 |
| `collision/skinny_r9*`  | 9      |        44 |       33 |   33 |
| `collision/skinny_r10*` | 10     |        54 |       42 |   41 |
| `collision/skinny_r11*` | 11     |        60 |       50 |   46 |
| `collision/skinny_r12*` | 12     |        66 |       59 |   54 |
| `collision/skinny_r13*` | 13     |        78 |       67 |   59 |
| `collision/skinny_r14*` | 14     |        86 |       76 |   69 |
| `collision/skinny_r15*` | 15     |        86 |       77 |   73 |
| `collision/skinny_r16*` | 16     |       106 |       96 |   74 |























