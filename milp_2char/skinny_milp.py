#!/opt/gurobi952/linux64/bin/python3.7

import sys
import re
import gurobipy as gp


### OPERATION HELPERS ##########################################################

def xor_2(in1, in2, out, m):
    m.addConstr(in1 + in2 >= out)
    m.addConstr(in1 + out >= in2)
    m.addConstr(out + in2 >= in1)

def xor_m(in1, in2, in3, in4, out1, out2, out3, out4, m):
    xor_2(in1, in3, out4, m)
    xor_2(out4, in4, out1, m)
    xor_2(in2, in3, out3, m)
    m.addConstr(out2 == in1)

def xor_2_U(in1, in2, out, m):
    # impact of XOR2 on Unknown status
    m.addConstr(in1 <= out)
    m.addConstr(in2 <= out)
    m.addConstr(in1 + in2 >= out)

def xor_ms_U(in1, in2, in3, in4, out1, out2, out3, out4, forget1, forget2, forget3, forget4, m):
    # impact of MC, SB-Forget on Unknown status
    xor_2_U(in1, in3, out4 - forget4, m)
    xor_2_U(out4 - forget4, in4, out1 - forget1, m)
    xor_2_U(in2, in3, out3 - forget3, m)
    m.addConstr(in1 == out2 - forget2)

permutation = [9,15,8,13,10,14,12,11,0,1,2,3,4,5,6,7]
def tk_permutation(K, nr_rounds, m):
    for r in range(nr_rounds):
        for i in i_s:
            for j in j_s:
                pos = i * 4 + j
                new_pos = permutation.index(pos)
                i_new = int(new_pos/4)
                j_new = new_pos%4
                # Do we need constraints here?? 
                m.addConstr(K[i,j,r] == K[i_new,j_new,r+1])

def lane_constraints(LANE, K, nr_rounds, nr_canc, m):
    lane_tk_ids = [[] for i in range(0,16)]
    for i in i_s:
        for j in j_s:
            i_new, j_new = i, j
            for r in range(nr_rounds):
                lane_tk_ids[i * 4 + j].append([i_new,j_new,r])
                m.addConstr(K[i_new, j_new, r] - LANE[i,j] <= 0)
                pos = i_new * 4 + j_new
                new_pos = permutation.index(pos)
                i_new = int(new_pos/4)
                j_new = new_pos%4
    for i in i_s:
        for j in j_s:
            m.addConstr(nr_rounds * LANE[i,j] - sum(K[i_perm,j_perm,r] for i_perm, j_perm, r in lane_tk_ids[i * 4 + j]) <= nr_canc)


### MILP MODEL #################################################################

def skinny_TK_SK(nr_rounds=4, nr_tweaks=2, inputpat="zero", outputpat="zero", dual=True, zeroonly=False, unknownonly=False):
    # create doubly differential model: 0->0 in TK2 (\delta) and 1->any in SK (\tau, with 'unknown' state)
    # all parameters except "nr_rounds" should be kept fixed to default and are just included pro-forma
    # inputpat, outputpat in ["zero", "one", "any"]
    # - zero to zero, delta_tweak=TK2 between 2 compression functions
    #
    # i is x-axis / row, j is y-axis / col, r is round
    r_s = range(nr_rounds+1)
    
    m = gp.Model()

    # Characteristic \delta between two CF calls (need to collide)
    # X --SB,AK--> Y --SR,MC--> X'
    #        K = tweakey (LANE = pattern of initial tweakey, without cancellation)
    X = m.addVars(i_s,j_s,r_s,name='sbox', vtype='B') # sbox activity pattern X[i, j, r] 
    Y = m.addVars(i_s,j_s,r_s,name='rkey', vtype='B') # pattern after additon of the roundtweakkey 
    K = m.addVars(i_s,j_s,r_s,name='tkey', vtype='B') # pattern of the initial tweakey
    # Characteristic \tau between within each CF call
    # T --SB,AK,SR,MC--> T'
    # U = is T unknown?
    # F = is F forgotten here, i.e., it *turns* unknown here?
    if dual:
        T = m.addVars(i_s,j_s,r_s,name='tbox', vtype='B') # sbox activity pattern T[i, j, r] - within compression function (SK)
        U = m.addVars(i_s,j_s,r_s,name='unkn', vtype='B') # unknown pattern U[i, j, r] - corresponding to T
        F = m.addVars(i_s,j_s,r_s,name='forg', vtype='B') # forget pattern F[i, j, r] - corresponding to T

    # Combined Cost of both characteristics (lower bound on independently active S-boxes, 1 = MDP weight, MDP = 2^-2)
    C = m.addVars(i_s,j_s,r_s,name='cost', vtype='I', lb=0, ub=2) # range 0..2
    # cost
    for r in range(nr_rounds):
        for i in i_s:
            for j in j_s:
                m.addConstr(C[i,j,r] >= X[i,j,r])
                if dual: 
                    m.addConstr(C[i,j,r] >= T[i,j,r] - U[i,j,r])
                    m.addConstr(C[i,j,r] >= 2*X[i,j,r] + U[i,j,r] - F[i,j,r] - 1)
    # Tweakey schedule 
    if nr_tweaks > 1:
        LANE = m.addVars(i_s,j_s,name='lane', vtype='B') # indicates if the i-th cell of TK1 or TK2 is active in the initial state 
        if nr_tweaks == 2: 
            nr_canc = 1
        elif nr_tweaks == 3:
            nr_canc = 2
        lane_constraints(LANE, K, nr_rounds, nr_canc, m)
    elif nr_tweaks == 1:
        tk_permutation(K, nr_rounds, m)
    # Round function
    for r in range(nr_rounds): 
        for j in j_s:
            for i in i_s:
                # Sbox
                if dual:
                    m.addConstr(F[i,j,r] <= T[i,j,r]) # only forget if active
                # Tweakey addition
                if i <=1:
                    xor_2(X[i,j,r], K[i,j,r], Y[i,j,r], m)
                else:
                    m.addConstr(Y[i,j,r] == X[i,j,r])
            # Application of the linear layer 
            xor_m(Y[0,j,r], Y[1,(j-1)%4,r], Y[2,(j-2)%4,r], Y[3,(j-3)%4,r], X[0,j,r+1], X[1,j,r+1], X[2,j,r+1], X[3,j,r+1], m)
            if dual:
                xor_m(T[0,j,r], T[1,(j-1)%4,r], T[2,(j-2)%4,r], T[3,(j-3)%4,r], T[0,j,r+1], T[1,j,r+1], T[2,j,r+1], T[3,j,r+1], m)
                xor_ms_U(U[0,j,r], U[1,(j-1)%4,r], U[2,(j-2)%4,r], U[3,(j-3)%4,r], U[0,j,r+1], U[1,j,r+1], U[2,j,r+1], U[3,j,r+1], F[0,j,r+1], F[1,j,r+1], F[2,j,r+1], F[3,j,r+1], m)

    # non trivial solution
    for r in range(nr_rounds):
        m.addConstr(sum(X[i,j,r] + K[i,j,r] for i in i_s for j in j_s) >= 1)

    # restriction: output_diff and input diff
    if inputpat == "zero":
        m.addConstr(sum(X[i,j,0] for i in i_s for j in j_s) == 0)
    if inputpat == "one":
        for i in i_s:
            for j in j_s:
                if i + j == 0:
                    m.addConstr(X[i,j,0] == 1)
                else:
                    m.addConstr(X[i,j,0] == 0)
    if outputpat == "zero":
        m.addConstr(sum(X[i,j,nr_rounds] for i in i_s for j in j_s) == 0)

    if inputpat == "equal" or outputpat == "equal":
        for i in i_s:
            for j in j_s:
                m.addConstr(X[i,j,0] == X[i,j,nr_rounds])

    if dual:
        # inputpat for T == "one":
        for i in i_s:
            for j in j_s:
                if i + j == 0:
                    m.addConstr(T[i,j,0] == 1)
                else:
                    m.addConstr(T[i,j,0] == 0)
                if unknownonly:
                    # U all 1 (initial difference is completely unknown)
                    m.addConstr(U[i,j,0] == 1)
                elif zeroonly:
                    # U[0,0] = 1 (initial active difference is unknown, zero difference is known)
                    if i + j == 0:
                        m.addConstr(U[i,j,0] == 1)
                    else:
                        m.addConstr(U[i,j,0] == 0)
                else:
                    # U = F (initial difference is known)
                    m.addConstr(U[i,j,0] == F[i,j,0])
        # outputpat for T == "any"

    # objective
    m.setObjective(sum(C[i,j,r] for r in range(nr_rounds) for i in i_s for j in j_s), gp.GRB.MINIMIZE)
    return m, X, Y, K, LANE, T, U, F, C


### MAIN #######################################################################

def get_filename(folder, probname):
    return "{}/skinny_r{}".format(folder, probname)

def make_lp_coll(roundsrange, solve=True, zeroonly=False, unknownonly=False):
    dual = True
    tweaks = 2
    inputpat = "zero"
    outputpat = "zero"
    for rounds in roundsrange:
        m, X, Y, K, LANE, T, U, F, C = skinny_TK_SK(rounds, tweaks, inputpat=inputpat, outputpat=outputpat, dual=dual, zeroonly=zeroonly, unknownonly=unknownonly)
        filename = get_filename("collision", str(rounds) + ("z" if zeroonly else "u" if unknownonly else ""))
        m.write(filename + ".lp")
        if solve:
            #m.params.mipfocus = 3
            m.optimize()
            m.write(filename + ".sol")
            print("Bound for {rounds} rounds: {m.objVal}".format(**locals()))

if __name__ == '__main__':
    make_lp_coll(range(1,16+1))
