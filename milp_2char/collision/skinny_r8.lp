\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 rkey[0,0,8] + 0 rkey[0,1,8] + 0 rkey[0,2,8] + 0 rkey[0,3,8]
   + 0 rkey[1,0,8] + 0 rkey[1,1,8] + 0 rkey[1,2,8] + 0 rkey[1,3,8]
   + 0 rkey[2,0,8] + 0 rkey[2,1,8] + 0 rkey[2,2,8] + 0 rkey[2,3,8]
   + 0 rkey[3,0,8] + 0 rkey[3,1,8] + 0 rkey[3,2,8] + 0 rkey[3,3,8]
   + 0 tkey[0,0,8] + 0 tkey[0,1,8] + 0 tkey[0,2,8] + 0 tkey[0,3,8]
   + 0 tkey[1,0,8] + 0 tkey[1,1,8] + 0 tkey[1,2,8] + 0 tkey[1,3,8]
   + 0 tkey[2,0,8] + 0 tkey[2,1,8] + 0 tkey[2,2,8] + 0 tkey[2,3,8]
   + 0 tkey[3,0,8] + 0 tkey[3,1,8] + 0 tkey[3,2,8] + 0 tkey[3,3,8]
   + cost[0,0,0] + cost[0,0,1] + cost[0,0,2] + cost[0,0,3] + cost[0,0,4]
   + cost[0,0,5] + cost[0,0,6] + cost[0,0,7] + 0 cost[0,0,8] + cost[0,1,0]
   + cost[0,1,1] + cost[0,1,2] + cost[0,1,3] + cost[0,1,4] + cost[0,1,5]
   + cost[0,1,6] + cost[0,1,7] + 0 cost[0,1,8] + cost[0,2,0] + cost[0,2,1]
   + cost[0,2,2] + cost[0,2,3] + cost[0,2,4] + cost[0,2,5] + cost[0,2,6]
   + cost[0,2,7] + 0 cost[0,2,8] + cost[0,3,0] + cost[0,3,1] + cost[0,3,2]
   + cost[0,3,3] + cost[0,3,4] + cost[0,3,5] + cost[0,3,6] + cost[0,3,7]
   + 0 cost[0,3,8] + cost[1,0,0] + cost[1,0,1] + cost[1,0,2] + cost[1,0,3]
   + cost[1,0,4] + cost[1,0,5] + cost[1,0,6] + cost[1,0,7] + 0 cost[1,0,8]
   + cost[1,1,0] + cost[1,1,1] + cost[1,1,2] + cost[1,1,3] + cost[1,1,4]
   + cost[1,1,5] + cost[1,1,6] + cost[1,1,7] + 0 cost[1,1,8] + cost[1,2,0]
   + cost[1,2,1] + cost[1,2,2] + cost[1,2,3] + cost[1,2,4] + cost[1,2,5]
   + cost[1,2,6] + cost[1,2,7] + 0 cost[1,2,8] + cost[1,3,0] + cost[1,3,1]
   + cost[1,3,2] + cost[1,3,3] + cost[1,3,4] + cost[1,3,5] + cost[1,3,6]
   + cost[1,3,7] + 0 cost[1,3,8] + cost[2,0,0] + cost[2,0,1] + cost[2,0,2]
   + cost[2,0,3] + cost[2,0,4] + cost[2,0,5] + cost[2,0,6] + cost[2,0,7]
   + 0 cost[2,0,8] + cost[2,1,0] + cost[2,1,1] + cost[2,1,2] + cost[2,1,3]
   + cost[2,1,4] + cost[2,1,5] + cost[2,1,6] + cost[2,1,7] + 0 cost[2,1,8]
   + cost[2,2,0] + cost[2,2,1] + cost[2,2,2] + cost[2,2,3] + cost[2,2,4]
   + cost[2,2,5] + cost[2,2,6] + cost[2,2,7] + 0 cost[2,2,8] + cost[2,3,0]
   + cost[2,3,1] + cost[2,3,2] + cost[2,3,3] + cost[2,3,4] + cost[2,3,5]
   + cost[2,3,6] + cost[2,3,7] + 0 cost[2,3,8] + cost[3,0,0] + cost[3,0,1]
   + cost[3,0,2] + cost[3,0,3] + cost[3,0,4] + cost[3,0,5] + cost[3,0,6]
   + cost[3,0,7] + 0 cost[3,0,8] + cost[3,1,0] + cost[3,1,1] + cost[3,1,2]
   + cost[3,1,3] + cost[3,1,4] + cost[3,1,5] + cost[3,1,6] + cost[3,1,7]
   + 0 cost[3,1,8] + cost[3,2,0] + cost[3,2,1] + cost[3,2,2] + cost[3,2,3]
   + cost[3,2,4] + cost[3,2,5] + cost[3,2,6] + cost[3,2,7] + 0 cost[3,2,8]
   + cost[3,3,0] + cost[3,3,1] + cost[3,3,2] + cost[3,3,3] + cost[3,3,4]
   + cost[3,3,5] + cost[3,3,6] + cost[3,3,7] + 0 cost[3,3,8]
Subject To
 R0: - sbox[0,0,0] + cost[0,0,0] >= 0
 R1: - tbox[0,0,0] + unkn[0,0,0] + cost[0,0,0] >= 0
 R2: - 2 sbox[0,0,0] - unkn[0,0,0] + forg[0,0,0] + cost[0,0,0] >= -1
 R3: - sbox[0,1,0] + cost[0,1,0] >= 0
 R4: - tbox[0,1,0] + unkn[0,1,0] + cost[0,1,0] >= 0
 R5: - 2 sbox[0,1,0] - unkn[0,1,0] + forg[0,1,0] + cost[0,1,0] >= -1
 R6: - sbox[0,2,0] + cost[0,2,0] >= 0
 R7: - tbox[0,2,0] + unkn[0,2,0] + cost[0,2,0] >= 0
 R8: - 2 sbox[0,2,0] - unkn[0,2,0] + forg[0,2,0] + cost[0,2,0] >= -1
 R9: - sbox[0,3,0] + cost[0,3,0] >= 0
 R10: - tbox[0,3,0] + unkn[0,3,0] + cost[0,3,0] >= 0
 R11: - 2 sbox[0,3,0] - unkn[0,3,0] + forg[0,3,0] + cost[0,3,0] >= -1
 R12: - sbox[1,0,0] + cost[1,0,0] >= 0
 R13: - tbox[1,0,0] + unkn[1,0,0] + cost[1,0,0] >= 0
 R14: - 2 sbox[1,0,0] - unkn[1,0,0] + forg[1,0,0] + cost[1,0,0] >= -1
 R15: - sbox[1,1,0] + cost[1,1,0] >= 0
 R16: - tbox[1,1,0] + unkn[1,1,0] + cost[1,1,0] >= 0
 R17: - 2 sbox[1,1,0] - unkn[1,1,0] + forg[1,1,0] + cost[1,1,0] >= -1
 R18: - sbox[1,2,0] + cost[1,2,0] >= 0
 R19: - tbox[1,2,0] + unkn[1,2,0] + cost[1,2,0] >= 0
 R20: - 2 sbox[1,2,0] - unkn[1,2,0] + forg[1,2,0] + cost[1,2,0] >= -1
 R21: - sbox[1,3,0] + cost[1,3,0] >= 0
 R22: - tbox[1,3,0] + unkn[1,3,0] + cost[1,3,0] >= 0
 R23: - 2 sbox[1,3,0] - unkn[1,3,0] + forg[1,3,0] + cost[1,3,0] >= -1
 R24: - sbox[2,0,0] + cost[2,0,0] >= 0
 R25: - tbox[2,0,0] + unkn[2,0,0] + cost[2,0,0] >= 0
 R26: - 2 sbox[2,0,0] - unkn[2,0,0] + forg[2,0,0] + cost[2,0,0] >= -1
 R27: - sbox[2,1,0] + cost[2,1,0] >= 0
 R28: - tbox[2,1,0] + unkn[2,1,0] + cost[2,1,0] >= 0
 R29: - 2 sbox[2,1,0] - unkn[2,1,0] + forg[2,1,0] + cost[2,1,0] >= -1
 R30: - sbox[2,2,0] + cost[2,2,0] >= 0
 R31: - tbox[2,2,0] + unkn[2,2,0] + cost[2,2,0] >= 0
 R32: - 2 sbox[2,2,0] - unkn[2,2,0] + forg[2,2,0] + cost[2,2,0] >= -1
 R33: - sbox[2,3,0] + cost[2,3,0] >= 0
 R34: - tbox[2,3,0] + unkn[2,3,0] + cost[2,3,0] >= 0
 R35: - 2 sbox[2,3,0] - unkn[2,3,0] + forg[2,3,0] + cost[2,3,0] >= -1
 R36: - sbox[3,0,0] + cost[3,0,0] >= 0
 R37: - tbox[3,0,0] + unkn[3,0,0] + cost[3,0,0] >= 0
 R38: - 2 sbox[3,0,0] - unkn[3,0,0] + forg[3,0,0] + cost[3,0,0] >= -1
 R39: - sbox[3,1,0] + cost[3,1,0] >= 0
 R40: - tbox[3,1,0] + unkn[3,1,0] + cost[3,1,0] >= 0
 R41: - 2 sbox[3,1,0] - unkn[3,1,0] + forg[3,1,0] + cost[3,1,0] >= -1
 R42: - sbox[3,2,0] + cost[3,2,0] >= 0
 R43: - tbox[3,2,0] + unkn[3,2,0] + cost[3,2,0] >= 0
 R44: - 2 sbox[3,2,0] - unkn[3,2,0] + forg[3,2,0] + cost[3,2,0] >= -1
 R45: - sbox[3,3,0] + cost[3,3,0] >= 0
 R46: - tbox[3,3,0] + unkn[3,3,0] + cost[3,3,0] >= 0
 R47: - 2 sbox[3,3,0] - unkn[3,3,0] + forg[3,3,0] + cost[3,3,0] >= -1
 R48: - sbox[0,0,1] + cost[0,0,1] >= 0
 R49: - tbox[0,0,1] + unkn[0,0,1] + cost[0,0,1] >= 0
 R50: - 2 sbox[0,0,1] - unkn[0,0,1] + forg[0,0,1] + cost[0,0,1] >= -1
 R51: - sbox[0,1,1] + cost[0,1,1] >= 0
 R52: - tbox[0,1,1] + unkn[0,1,1] + cost[0,1,1] >= 0
 R53: - 2 sbox[0,1,1] - unkn[0,1,1] + forg[0,1,1] + cost[0,1,1] >= -1
 R54: - sbox[0,2,1] + cost[0,2,1] >= 0
 R55: - tbox[0,2,1] + unkn[0,2,1] + cost[0,2,1] >= 0
 R56: - 2 sbox[0,2,1] - unkn[0,2,1] + forg[0,2,1] + cost[0,2,1] >= -1
 R57: - sbox[0,3,1] + cost[0,3,1] >= 0
 R58: - tbox[0,3,1] + unkn[0,3,1] + cost[0,3,1] >= 0
 R59: - 2 sbox[0,3,1] - unkn[0,3,1] + forg[0,3,1] + cost[0,3,1] >= -1
 R60: - sbox[1,0,1] + cost[1,0,1] >= 0
 R61: - tbox[1,0,1] + unkn[1,0,1] + cost[1,0,1] >= 0
 R62: - 2 sbox[1,0,1] - unkn[1,0,1] + forg[1,0,1] + cost[1,0,1] >= -1
 R63: - sbox[1,1,1] + cost[1,1,1] >= 0
 R64: - tbox[1,1,1] + unkn[1,1,1] + cost[1,1,1] >= 0
 R65: - 2 sbox[1,1,1] - unkn[1,1,1] + forg[1,1,1] + cost[1,1,1] >= -1
 R66: - sbox[1,2,1] + cost[1,2,1] >= 0
 R67: - tbox[1,2,1] + unkn[1,2,1] + cost[1,2,1] >= 0
 R68: - 2 sbox[1,2,1] - unkn[1,2,1] + forg[1,2,1] + cost[1,2,1] >= -1
 R69: - sbox[1,3,1] + cost[1,3,1] >= 0
 R70: - tbox[1,3,1] + unkn[1,3,1] + cost[1,3,1] >= 0
 R71: - 2 sbox[1,3,1] - unkn[1,3,1] + forg[1,3,1] + cost[1,3,1] >= -1
 R72: - sbox[2,0,1] + cost[2,0,1] >= 0
 R73: - tbox[2,0,1] + unkn[2,0,1] + cost[2,0,1] >= 0
 R74: - 2 sbox[2,0,1] - unkn[2,0,1] + forg[2,0,1] + cost[2,0,1] >= -1
 R75: - sbox[2,1,1] + cost[2,1,1] >= 0
 R76: - tbox[2,1,1] + unkn[2,1,1] + cost[2,1,1] >= 0
 R77: - 2 sbox[2,1,1] - unkn[2,1,1] + forg[2,1,1] + cost[2,1,1] >= -1
 R78: - sbox[2,2,1] + cost[2,2,1] >= 0
 R79: - tbox[2,2,1] + unkn[2,2,1] + cost[2,2,1] >= 0
 R80: - 2 sbox[2,2,1] - unkn[2,2,1] + forg[2,2,1] + cost[2,2,1] >= -1
 R81: - sbox[2,3,1] + cost[2,3,1] >= 0
 R82: - tbox[2,3,1] + unkn[2,3,1] + cost[2,3,1] >= 0
 R83: - 2 sbox[2,3,1] - unkn[2,3,1] + forg[2,3,1] + cost[2,3,1] >= -1
 R84: - sbox[3,0,1] + cost[3,0,1] >= 0
 R85: - tbox[3,0,1] + unkn[3,0,1] + cost[3,0,1] >= 0
 R86: - 2 sbox[3,0,1] - unkn[3,0,1] + forg[3,0,1] + cost[3,0,1] >= -1
 R87: - sbox[3,1,1] + cost[3,1,1] >= 0
 R88: - tbox[3,1,1] + unkn[3,1,1] + cost[3,1,1] >= 0
 R89: - 2 sbox[3,1,1] - unkn[3,1,1] + forg[3,1,1] + cost[3,1,1] >= -1
 R90: - sbox[3,2,1] + cost[3,2,1] >= 0
 R91: - tbox[3,2,1] + unkn[3,2,1] + cost[3,2,1] >= 0
 R92: - 2 sbox[3,2,1] - unkn[3,2,1] + forg[3,2,1] + cost[3,2,1] >= -1
 R93: - sbox[3,3,1] + cost[3,3,1] >= 0
 R94: - tbox[3,3,1] + unkn[3,3,1] + cost[3,3,1] >= 0
 R95: - 2 sbox[3,3,1] - unkn[3,3,1] + forg[3,3,1] + cost[3,3,1] >= -1
 R96: - sbox[0,0,2] + cost[0,0,2] >= 0
 R97: - tbox[0,0,2] + unkn[0,0,2] + cost[0,0,2] >= 0
 R98: - 2 sbox[0,0,2] - unkn[0,0,2] + forg[0,0,2] + cost[0,0,2] >= -1
 R99: - sbox[0,1,2] + cost[0,1,2] >= 0
 R100: - tbox[0,1,2] + unkn[0,1,2] + cost[0,1,2] >= 0
 R101: - 2 sbox[0,1,2] - unkn[0,1,2] + forg[0,1,2] + cost[0,1,2] >= -1
 R102: - sbox[0,2,2] + cost[0,2,2] >= 0
 R103: - tbox[0,2,2] + unkn[0,2,2] + cost[0,2,2] >= 0
 R104: - 2 sbox[0,2,2] - unkn[0,2,2] + forg[0,2,2] + cost[0,2,2] >= -1
 R105: - sbox[0,3,2] + cost[0,3,2] >= 0
 R106: - tbox[0,3,2] + unkn[0,3,2] + cost[0,3,2] >= 0
 R107: - 2 sbox[0,3,2] - unkn[0,3,2] + forg[0,3,2] + cost[0,3,2] >= -1
 R108: - sbox[1,0,2] + cost[1,0,2] >= 0
 R109: - tbox[1,0,2] + unkn[1,0,2] + cost[1,0,2] >= 0
 R110: - 2 sbox[1,0,2] - unkn[1,0,2] + forg[1,0,2] + cost[1,0,2] >= -1
 R111: - sbox[1,1,2] + cost[1,1,2] >= 0
 R112: - tbox[1,1,2] + unkn[1,1,2] + cost[1,1,2] >= 0
 R113: - 2 sbox[1,1,2] - unkn[1,1,2] + forg[1,1,2] + cost[1,1,2] >= -1
 R114: - sbox[1,2,2] + cost[1,2,2] >= 0
 R115: - tbox[1,2,2] + unkn[1,2,2] + cost[1,2,2] >= 0
 R116: - 2 sbox[1,2,2] - unkn[1,2,2] + forg[1,2,2] + cost[1,2,2] >= -1
 R117: - sbox[1,3,2] + cost[1,3,2] >= 0
 R118: - tbox[1,3,2] + unkn[1,3,2] + cost[1,3,2] >= 0
 R119: - 2 sbox[1,3,2] - unkn[1,3,2] + forg[1,3,2] + cost[1,3,2] >= -1
 R120: - sbox[2,0,2] + cost[2,0,2] >= 0
 R121: - tbox[2,0,2] + unkn[2,0,2] + cost[2,0,2] >= 0
 R122: - 2 sbox[2,0,2] - unkn[2,0,2] + forg[2,0,2] + cost[2,0,2] >= -1
 R123: - sbox[2,1,2] + cost[2,1,2] >= 0
 R124: - tbox[2,1,2] + unkn[2,1,2] + cost[2,1,2] >= 0
 R125: - 2 sbox[2,1,2] - unkn[2,1,2] + forg[2,1,2] + cost[2,1,2] >= -1
 R126: - sbox[2,2,2] + cost[2,2,2] >= 0
 R127: - tbox[2,2,2] + unkn[2,2,2] + cost[2,2,2] >= 0
 R128: - 2 sbox[2,2,2] - unkn[2,2,2] + forg[2,2,2] + cost[2,2,2] >= -1
 R129: - sbox[2,3,2] + cost[2,3,2] >= 0
 R130: - tbox[2,3,2] + unkn[2,3,2] + cost[2,3,2] >= 0
 R131: - 2 sbox[2,3,2] - unkn[2,3,2] + forg[2,3,2] + cost[2,3,2] >= -1
 R132: - sbox[3,0,2] + cost[3,0,2] >= 0
 R133: - tbox[3,0,2] + unkn[3,0,2] + cost[3,0,2] >= 0
 R134: - 2 sbox[3,0,2] - unkn[3,0,2] + forg[3,0,2] + cost[3,0,2] >= -1
 R135: - sbox[3,1,2] + cost[3,1,2] >= 0
 R136: - tbox[3,1,2] + unkn[3,1,2] + cost[3,1,2] >= 0
 R137: - 2 sbox[3,1,2] - unkn[3,1,2] + forg[3,1,2] + cost[3,1,2] >= -1
 R138: - sbox[3,2,2] + cost[3,2,2] >= 0
 R139: - tbox[3,2,2] + unkn[3,2,2] + cost[3,2,2] >= 0
 R140: - 2 sbox[3,2,2] - unkn[3,2,2] + forg[3,2,2] + cost[3,2,2] >= -1
 R141: - sbox[3,3,2] + cost[3,3,2] >= 0
 R142: - tbox[3,3,2] + unkn[3,3,2] + cost[3,3,2] >= 0
 R143: - 2 sbox[3,3,2] - unkn[3,3,2] + forg[3,3,2] + cost[3,3,2] >= -1
 R144: - sbox[0,0,3] + cost[0,0,3] >= 0
 R145: - tbox[0,0,3] + unkn[0,0,3] + cost[0,0,3] >= 0
 R146: - 2 sbox[0,0,3] - unkn[0,0,3] + forg[0,0,3] + cost[0,0,3] >= -1
 R147: - sbox[0,1,3] + cost[0,1,3] >= 0
 R148: - tbox[0,1,3] + unkn[0,1,3] + cost[0,1,3] >= 0
 R149: - 2 sbox[0,1,3] - unkn[0,1,3] + forg[0,1,3] + cost[0,1,3] >= -1
 R150: - sbox[0,2,3] + cost[0,2,3] >= 0
 R151: - tbox[0,2,3] + unkn[0,2,3] + cost[0,2,3] >= 0
 R152: - 2 sbox[0,2,3] - unkn[0,2,3] + forg[0,2,3] + cost[0,2,3] >= -1
 R153: - sbox[0,3,3] + cost[0,3,3] >= 0
 R154: - tbox[0,3,3] + unkn[0,3,3] + cost[0,3,3] >= 0
 R155: - 2 sbox[0,3,3] - unkn[0,3,3] + forg[0,3,3] + cost[0,3,3] >= -1
 R156: - sbox[1,0,3] + cost[1,0,3] >= 0
 R157: - tbox[1,0,3] + unkn[1,0,3] + cost[1,0,3] >= 0
 R158: - 2 sbox[1,0,3] - unkn[1,0,3] + forg[1,0,3] + cost[1,0,3] >= -1
 R159: - sbox[1,1,3] + cost[1,1,3] >= 0
 R160: - tbox[1,1,3] + unkn[1,1,3] + cost[1,1,3] >= 0
 R161: - 2 sbox[1,1,3] - unkn[1,1,3] + forg[1,1,3] + cost[1,1,3] >= -1
 R162: - sbox[1,2,3] + cost[1,2,3] >= 0
 R163: - tbox[1,2,3] + unkn[1,2,3] + cost[1,2,3] >= 0
 R164: - 2 sbox[1,2,3] - unkn[1,2,3] + forg[1,2,3] + cost[1,2,3] >= -1
 R165: - sbox[1,3,3] + cost[1,3,3] >= 0
 R166: - tbox[1,3,3] + unkn[1,3,3] + cost[1,3,3] >= 0
 R167: - 2 sbox[1,3,3] - unkn[1,3,3] + forg[1,3,3] + cost[1,3,3] >= -1
 R168: - sbox[2,0,3] + cost[2,0,3] >= 0
 R169: - tbox[2,0,3] + unkn[2,0,3] + cost[2,0,3] >= 0
 R170: - 2 sbox[2,0,3] - unkn[2,0,3] + forg[2,0,3] + cost[2,0,3] >= -1
 R171: - sbox[2,1,3] + cost[2,1,3] >= 0
 R172: - tbox[2,1,3] + unkn[2,1,3] + cost[2,1,3] >= 0
 R173: - 2 sbox[2,1,3] - unkn[2,1,3] + forg[2,1,3] + cost[2,1,3] >= -1
 R174: - sbox[2,2,3] + cost[2,2,3] >= 0
 R175: - tbox[2,2,3] + unkn[2,2,3] + cost[2,2,3] >= 0
 R176: - 2 sbox[2,2,3] - unkn[2,2,3] + forg[2,2,3] + cost[2,2,3] >= -1
 R177: - sbox[2,3,3] + cost[2,3,3] >= 0
 R178: - tbox[2,3,3] + unkn[2,3,3] + cost[2,3,3] >= 0
 R179: - 2 sbox[2,3,3] - unkn[2,3,3] + forg[2,3,3] + cost[2,3,3] >= -1
 R180: - sbox[3,0,3] + cost[3,0,3] >= 0
 R181: - tbox[3,0,3] + unkn[3,0,3] + cost[3,0,3] >= 0
 R182: - 2 sbox[3,0,3] - unkn[3,0,3] + forg[3,0,3] + cost[3,0,3] >= -1
 R183: - sbox[3,1,3] + cost[3,1,3] >= 0
 R184: - tbox[3,1,3] + unkn[3,1,3] + cost[3,1,3] >= 0
 R185: - 2 sbox[3,1,3] - unkn[3,1,3] + forg[3,1,3] + cost[3,1,3] >= -1
 R186: - sbox[3,2,3] + cost[3,2,3] >= 0
 R187: - tbox[3,2,3] + unkn[3,2,3] + cost[3,2,3] >= 0
 R188: - 2 sbox[3,2,3] - unkn[3,2,3] + forg[3,2,3] + cost[3,2,3] >= -1
 R189: - sbox[3,3,3] + cost[3,3,3] >= 0
 R190: - tbox[3,3,3] + unkn[3,3,3] + cost[3,3,3] >= 0
 R191: - 2 sbox[3,3,3] - unkn[3,3,3] + forg[3,3,3] + cost[3,3,3] >= -1
 R192: - sbox[0,0,4] + cost[0,0,4] >= 0
 R193: - tbox[0,0,4] + unkn[0,0,4] + cost[0,0,4] >= 0
 R194: - 2 sbox[0,0,4] - unkn[0,0,4] + forg[0,0,4] + cost[0,0,4] >= -1
 R195: - sbox[0,1,4] + cost[0,1,4] >= 0
 R196: - tbox[0,1,4] + unkn[0,1,4] + cost[0,1,4] >= 0
 R197: - 2 sbox[0,1,4] - unkn[0,1,4] + forg[0,1,4] + cost[0,1,4] >= -1
 R198: - sbox[0,2,4] + cost[0,2,4] >= 0
 R199: - tbox[0,2,4] + unkn[0,2,4] + cost[0,2,4] >= 0
 R200: - 2 sbox[0,2,4] - unkn[0,2,4] + forg[0,2,4] + cost[0,2,4] >= -1
 R201: - sbox[0,3,4] + cost[0,3,4] >= 0
 R202: - tbox[0,3,4] + unkn[0,3,4] + cost[0,3,4] >= 0
 R203: - 2 sbox[0,3,4] - unkn[0,3,4] + forg[0,3,4] + cost[0,3,4] >= -1
 R204: - sbox[1,0,4] + cost[1,0,4] >= 0
 R205: - tbox[1,0,4] + unkn[1,0,4] + cost[1,0,4] >= 0
 R206: - 2 sbox[1,0,4] - unkn[1,0,4] + forg[1,0,4] + cost[1,0,4] >= -1
 R207: - sbox[1,1,4] + cost[1,1,4] >= 0
 R208: - tbox[1,1,4] + unkn[1,1,4] + cost[1,1,4] >= 0
 R209: - 2 sbox[1,1,4] - unkn[1,1,4] + forg[1,1,4] + cost[1,1,4] >= -1
 R210: - sbox[1,2,4] + cost[1,2,4] >= 0
 R211: - tbox[1,2,4] + unkn[1,2,4] + cost[1,2,4] >= 0
 R212: - 2 sbox[1,2,4] - unkn[1,2,4] + forg[1,2,4] + cost[1,2,4] >= -1
 R213: - sbox[1,3,4] + cost[1,3,4] >= 0
 R214: - tbox[1,3,4] + unkn[1,3,4] + cost[1,3,4] >= 0
 R215: - 2 sbox[1,3,4] - unkn[1,3,4] + forg[1,3,4] + cost[1,3,4] >= -1
 R216: - sbox[2,0,4] + cost[2,0,4] >= 0
 R217: - tbox[2,0,4] + unkn[2,0,4] + cost[2,0,4] >= 0
 R218: - 2 sbox[2,0,4] - unkn[2,0,4] + forg[2,0,4] + cost[2,0,4] >= -1
 R219: - sbox[2,1,4] + cost[2,1,4] >= 0
 R220: - tbox[2,1,4] + unkn[2,1,4] + cost[2,1,4] >= 0
 R221: - 2 sbox[2,1,4] - unkn[2,1,4] + forg[2,1,4] + cost[2,1,4] >= -1
 R222: - sbox[2,2,4] + cost[2,2,4] >= 0
 R223: - tbox[2,2,4] + unkn[2,2,4] + cost[2,2,4] >= 0
 R224: - 2 sbox[2,2,4] - unkn[2,2,4] + forg[2,2,4] + cost[2,2,4] >= -1
 R225: - sbox[2,3,4] + cost[2,3,4] >= 0
 R226: - tbox[2,3,4] + unkn[2,3,4] + cost[2,3,4] >= 0
 R227: - 2 sbox[2,3,4] - unkn[2,3,4] + forg[2,3,4] + cost[2,3,4] >= -1
 R228: - sbox[3,0,4] + cost[3,0,4] >= 0
 R229: - tbox[3,0,4] + unkn[3,0,4] + cost[3,0,4] >= 0
 R230: - 2 sbox[3,0,4] - unkn[3,0,4] + forg[3,0,4] + cost[3,0,4] >= -1
 R231: - sbox[3,1,4] + cost[3,1,4] >= 0
 R232: - tbox[3,1,4] + unkn[3,1,4] + cost[3,1,4] >= 0
 R233: - 2 sbox[3,1,4] - unkn[3,1,4] + forg[3,1,4] + cost[3,1,4] >= -1
 R234: - sbox[3,2,4] + cost[3,2,4] >= 0
 R235: - tbox[3,2,4] + unkn[3,2,4] + cost[3,2,4] >= 0
 R236: - 2 sbox[3,2,4] - unkn[3,2,4] + forg[3,2,4] + cost[3,2,4] >= -1
 R237: - sbox[3,3,4] + cost[3,3,4] >= 0
 R238: - tbox[3,3,4] + unkn[3,3,4] + cost[3,3,4] >= 0
 R239: - 2 sbox[3,3,4] - unkn[3,3,4] + forg[3,3,4] + cost[3,3,4] >= -1
 R240: - sbox[0,0,5] + cost[0,0,5] >= 0
 R241: - tbox[0,0,5] + unkn[0,0,5] + cost[0,0,5] >= 0
 R242: - 2 sbox[0,0,5] - unkn[0,0,5] + forg[0,0,5] + cost[0,0,5] >= -1
 R243: - sbox[0,1,5] + cost[0,1,5] >= 0
 R244: - tbox[0,1,5] + unkn[0,1,5] + cost[0,1,5] >= 0
 R245: - 2 sbox[0,1,5] - unkn[0,1,5] + forg[0,1,5] + cost[0,1,5] >= -1
 R246: - sbox[0,2,5] + cost[0,2,5] >= 0
 R247: - tbox[0,2,5] + unkn[0,2,5] + cost[0,2,5] >= 0
 R248: - 2 sbox[0,2,5] - unkn[0,2,5] + forg[0,2,5] + cost[0,2,5] >= -1
 R249: - sbox[0,3,5] + cost[0,3,5] >= 0
 R250: - tbox[0,3,5] + unkn[0,3,5] + cost[0,3,5] >= 0
 R251: - 2 sbox[0,3,5] - unkn[0,3,5] + forg[0,3,5] + cost[0,3,5] >= -1
 R252: - sbox[1,0,5] + cost[1,0,5] >= 0
 R253: - tbox[1,0,5] + unkn[1,0,5] + cost[1,0,5] >= 0
 R254: - 2 sbox[1,0,5] - unkn[1,0,5] + forg[1,0,5] + cost[1,0,5] >= -1
 R255: - sbox[1,1,5] + cost[1,1,5] >= 0
 R256: - tbox[1,1,5] + unkn[1,1,5] + cost[1,1,5] >= 0
 R257: - 2 sbox[1,1,5] - unkn[1,1,5] + forg[1,1,5] + cost[1,1,5] >= -1
 R258: - sbox[1,2,5] + cost[1,2,5] >= 0
 R259: - tbox[1,2,5] + unkn[1,2,5] + cost[1,2,5] >= 0
 R260: - 2 sbox[1,2,5] - unkn[1,2,5] + forg[1,2,5] + cost[1,2,5] >= -1
 R261: - sbox[1,3,5] + cost[1,3,5] >= 0
 R262: - tbox[1,3,5] + unkn[1,3,5] + cost[1,3,5] >= 0
 R263: - 2 sbox[1,3,5] - unkn[1,3,5] + forg[1,3,5] + cost[1,3,5] >= -1
 R264: - sbox[2,0,5] + cost[2,0,5] >= 0
 R265: - tbox[2,0,5] + unkn[2,0,5] + cost[2,0,5] >= 0
 R266: - 2 sbox[2,0,5] - unkn[2,0,5] + forg[2,0,5] + cost[2,0,5] >= -1
 R267: - sbox[2,1,5] + cost[2,1,5] >= 0
 R268: - tbox[2,1,5] + unkn[2,1,5] + cost[2,1,5] >= 0
 R269: - 2 sbox[2,1,5] - unkn[2,1,5] + forg[2,1,5] + cost[2,1,5] >= -1
 R270: - sbox[2,2,5] + cost[2,2,5] >= 0
 R271: - tbox[2,2,5] + unkn[2,2,5] + cost[2,2,5] >= 0
 R272: - 2 sbox[2,2,5] - unkn[2,2,5] + forg[2,2,5] + cost[2,2,5] >= -1
 R273: - sbox[2,3,5] + cost[2,3,5] >= 0
 R274: - tbox[2,3,5] + unkn[2,3,5] + cost[2,3,5] >= 0
 R275: - 2 sbox[2,3,5] - unkn[2,3,5] + forg[2,3,5] + cost[2,3,5] >= -1
 R276: - sbox[3,0,5] + cost[3,0,5] >= 0
 R277: - tbox[3,0,5] + unkn[3,0,5] + cost[3,0,5] >= 0
 R278: - 2 sbox[3,0,5] - unkn[3,0,5] + forg[3,0,5] + cost[3,0,5] >= -1
 R279: - sbox[3,1,5] + cost[3,1,5] >= 0
 R280: - tbox[3,1,5] + unkn[3,1,5] + cost[3,1,5] >= 0
 R281: - 2 sbox[3,1,5] - unkn[3,1,5] + forg[3,1,5] + cost[3,1,5] >= -1
 R282: - sbox[3,2,5] + cost[3,2,5] >= 0
 R283: - tbox[3,2,5] + unkn[3,2,5] + cost[3,2,5] >= 0
 R284: - 2 sbox[3,2,5] - unkn[3,2,5] + forg[3,2,5] + cost[3,2,5] >= -1
 R285: - sbox[3,3,5] + cost[3,3,5] >= 0
 R286: - tbox[3,3,5] + unkn[3,3,5] + cost[3,3,5] >= 0
 R287: - 2 sbox[3,3,5] - unkn[3,3,5] + forg[3,3,5] + cost[3,3,5] >= -1
 R288: - sbox[0,0,6] + cost[0,0,6] >= 0
 R289: - tbox[0,0,6] + unkn[0,0,6] + cost[0,0,6] >= 0
 R290: - 2 sbox[0,0,6] - unkn[0,0,6] + forg[0,0,6] + cost[0,0,6] >= -1
 R291: - sbox[0,1,6] + cost[0,1,6] >= 0
 R292: - tbox[0,1,6] + unkn[0,1,6] + cost[0,1,6] >= 0
 R293: - 2 sbox[0,1,6] - unkn[0,1,6] + forg[0,1,6] + cost[0,1,6] >= -1
 R294: - sbox[0,2,6] + cost[0,2,6] >= 0
 R295: - tbox[0,2,6] + unkn[0,2,6] + cost[0,2,6] >= 0
 R296: - 2 sbox[0,2,6] - unkn[0,2,6] + forg[0,2,6] + cost[0,2,6] >= -1
 R297: - sbox[0,3,6] + cost[0,3,6] >= 0
 R298: - tbox[0,3,6] + unkn[0,3,6] + cost[0,3,6] >= 0
 R299: - 2 sbox[0,3,6] - unkn[0,3,6] + forg[0,3,6] + cost[0,3,6] >= -1
 R300: - sbox[1,0,6] + cost[1,0,6] >= 0
 R301: - tbox[1,0,6] + unkn[1,0,6] + cost[1,0,6] >= 0
 R302: - 2 sbox[1,0,6] - unkn[1,0,6] + forg[1,0,6] + cost[1,0,6] >= -1
 R303: - sbox[1,1,6] + cost[1,1,6] >= 0
 R304: - tbox[1,1,6] + unkn[1,1,6] + cost[1,1,6] >= 0
 R305: - 2 sbox[1,1,6] - unkn[1,1,6] + forg[1,1,6] + cost[1,1,6] >= -1
 R306: - sbox[1,2,6] + cost[1,2,6] >= 0
 R307: - tbox[1,2,6] + unkn[1,2,6] + cost[1,2,6] >= 0
 R308: - 2 sbox[1,2,6] - unkn[1,2,6] + forg[1,2,6] + cost[1,2,6] >= -1
 R309: - sbox[1,3,6] + cost[1,3,6] >= 0
 R310: - tbox[1,3,6] + unkn[1,3,6] + cost[1,3,6] >= 0
 R311: - 2 sbox[1,3,6] - unkn[1,3,6] + forg[1,3,6] + cost[1,3,6] >= -1
 R312: - sbox[2,0,6] + cost[2,0,6] >= 0
 R313: - tbox[2,0,6] + unkn[2,0,6] + cost[2,0,6] >= 0
 R314: - 2 sbox[2,0,6] - unkn[2,0,6] + forg[2,0,6] + cost[2,0,6] >= -1
 R315: - sbox[2,1,6] + cost[2,1,6] >= 0
 R316: - tbox[2,1,6] + unkn[2,1,6] + cost[2,1,6] >= 0
 R317: - 2 sbox[2,1,6] - unkn[2,1,6] + forg[2,1,6] + cost[2,1,6] >= -1
 R318: - sbox[2,2,6] + cost[2,2,6] >= 0
 R319: - tbox[2,2,6] + unkn[2,2,6] + cost[2,2,6] >= 0
 R320: - 2 sbox[2,2,6] - unkn[2,2,6] + forg[2,2,6] + cost[2,2,6] >= -1
 R321: - sbox[2,3,6] + cost[2,3,6] >= 0
 R322: - tbox[2,3,6] + unkn[2,3,6] + cost[2,3,6] >= 0
 R323: - 2 sbox[2,3,6] - unkn[2,3,6] + forg[2,3,6] + cost[2,3,6] >= -1
 R324: - sbox[3,0,6] + cost[3,0,6] >= 0
 R325: - tbox[3,0,6] + unkn[3,0,6] + cost[3,0,6] >= 0
 R326: - 2 sbox[3,0,6] - unkn[3,0,6] + forg[3,0,6] + cost[3,0,6] >= -1
 R327: - sbox[3,1,6] + cost[3,1,6] >= 0
 R328: - tbox[3,1,6] + unkn[3,1,6] + cost[3,1,6] >= 0
 R329: - 2 sbox[3,1,6] - unkn[3,1,6] + forg[3,1,6] + cost[3,1,6] >= -1
 R330: - sbox[3,2,6] + cost[3,2,6] >= 0
 R331: - tbox[3,2,6] + unkn[3,2,6] + cost[3,2,6] >= 0
 R332: - 2 sbox[3,2,6] - unkn[3,2,6] + forg[3,2,6] + cost[3,2,6] >= -1
 R333: - sbox[3,3,6] + cost[3,3,6] >= 0
 R334: - tbox[3,3,6] + unkn[3,3,6] + cost[3,3,6] >= 0
 R335: - 2 sbox[3,3,6] - unkn[3,3,6] + forg[3,3,6] + cost[3,3,6] >= -1
 R336: - sbox[0,0,7] + cost[0,0,7] >= 0
 R337: - tbox[0,0,7] + unkn[0,0,7] + cost[0,0,7] >= 0
 R338: - 2 sbox[0,0,7] - unkn[0,0,7] + forg[0,0,7] + cost[0,0,7] >= -1
 R339: - sbox[0,1,7] + cost[0,1,7] >= 0
 R340: - tbox[0,1,7] + unkn[0,1,7] + cost[0,1,7] >= 0
 R341: - 2 sbox[0,1,7] - unkn[0,1,7] + forg[0,1,7] + cost[0,1,7] >= -1
 R342: - sbox[0,2,7] + cost[0,2,7] >= 0
 R343: - tbox[0,2,7] + unkn[0,2,7] + cost[0,2,7] >= 0
 R344: - 2 sbox[0,2,7] - unkn[0,2,7] + forg[0,2,7] + cost[0,2,7] >= -1
 R345: - sbox[0,3,7] + cost[0,3,7] >= 0
 R346: - tbox[0,3,7] + unkn[0,3,7] + cost[0,3,7] >= 0
 R347: - 2 sbox[0,3,7] - unkn[0,3,7] + forg[0,3,7] + cost[0,3,7] >= -1
 R348: - sbox[1,0,7] + cost[1,0,7] >= 0
 R349: - tbox[1,0,7] + unkn[1,0,7] + cost[1,0,7] >= 0
 R350: - 2 sbox[1,0,7] - unkn[1,0,7] + forg[1,0,7] + cost[1,0,7] >= -1
 R351: - sbox[1,1,7] + cost[1,1,7] >= 0
 R352: - tbox[1,1,7] + unkn[1,1,7] + cost[1,1,7] >= 0
 R353: - 2 sbox[1,1,7] - unkn[1,1,7] + forg[1,1,7] + cost[1,1,7] >= -1
 R354: - sbox[1,2,7] + cost[1,2,7] >= 0
 R355: - tbox[1,2,7] + unkn[1,2,7] + cost[1,2,7] >= 0
 R356: - 2 sbox[1,2,7] - unkn[1,2,7] + forg[1,2,7] + cost[1,2,7] >= -1
 R357: - sbox[1,3,7] + cost[1,3,7] >= 0
 R358: - tbox[1,3,7] + unkn[1,3,7] + cost[1,3,7] >= 0
 R359: - 2 sbox[1,3,7] - unkn[1,3,7] + forg[1,3,7] + cost[1,3,7] >= -1
 R360: - sbox[2,0,7] + cost[2,0,7] >= 0
 R361: - tbox[2,0,7] + unkn[2,0,7] + cost[2,0,7] >= 0
 R362: - 2 sbox[2,0,7] - unkn[2,0,7] + forg[2,0,7] + cost[2,0,7] >= -1
 R363: - sbox[2,1,7] + cost[2,1,7] >= 0
 R364: - tbox[2,1,7] + unkn[2,1,7] + cost[2,1,7] >= 0
 R365: - 2 sbox[2,1,7] - unkn[2,1,7] + forg[2,1,7] + cost[2,1,7] >= -1
 R366: - sbox[2,2,7] + cost[2,2,7] >= 0
 R367: - tbox[2,2,7] + unkn[2,2,7] + cost[2,2,7] >= 0
 R368: - 2 sbox[2,2,7] - unkn[2,2,7] + forg[2,2,7] + cost[2,2,7] >= -1
 R369: - sbox[2,3,7] + cost[2,3,7] >= 0
 R370: - tbox[2,3,7] + unkn[2,3,7] + cost[2,3,7] >= 0
 R371: - 2 sbox[2,3,7] - unkn[2,3,7] + forg[2,3,7] + cost[2,3,7] >= -1
 R372: - sbox[3,0,7] + cost[3,0,7] >= 0
 R373: - tbox[3,0,7] + unkn[3,0,7] + cost[3,0,7] >= 0
 R374: - 2 sbox[3,0,7] - unkn[3,0,7] + forg[3,0,7] + cost[3,0,7] >= -1
 R375: - sbox[3,1,7] + cost[3,1,7] >= 0
 R376: - tbox[3,1,7] + unkn[3,1,7] + cost[3,1,7] >= 0
 R377: - 2 sbox[3,1,7] - unkn[3,1,7] + forg[3,1,7] + cost[3,1,7] >= -1
 R378: - sbox[3,2,7] + cost[3,2,7] >= 0
 R379: - tbox[3,2,7] + unkn[3,2,7] + cost[3,2,7] >= 0
 R380: - 2 sbox[3,2,7] - unkn[3,2,7] + forg[3,2,7] + cost[3,2,7] >= -1
 R381: - sbox[3,3,7] + cost[3,3,7] >= 0
 R382: - tbox[3,3,7] + unkn[3,3,7] + cost[3,3,7] >= 0
 R383: - 2 sbox[3,3,7] - unkn[3,3,7] + forg[3,3,7] + cost[3,3,7] >= -1
 R384: tkey[0,0,0] - lane[0,0] <= 0
 R385: tkey[2,0,1] - lane[0,0] <= 0
 R386: tkey[0,2,2] - lane[0,0] <= 0
 R387: tkey[2,2,3] - lane[0,0] <= 0
 R388: tkey[1,0,4] - lane[0,0] <= 0
 R389: tkey[3,0,5] - lane[0,0] <= 0
 R390: tkey[1,2,6] - lane[0,0] <= 0
 R391: tkey[3,2,7] - lane[0,0] <= 0
 R392: tkey[0,1,0] - lane[0,1] <= 0
 R393: tkey[2,1,1] - lane[0,1] <= 0
 R394: tkey[0,0,2] - lane[0,1] <= 0
 R395: tkey[2,0,3] - lane[0,1] <= 0
 R396: tkey[0,2,4] - lane[0,1] <= 0
 R397: tkey[2,2,5] - lane[0,1] <= 0
 R398: tkey[1,0,6] - lane[0,1] <= 0
 R399: tkey[3,0,7] - lane[0,1] <= 0
 R400: tkey[0,2,0] - lane[0,2] <= 0
 R401: tkey[2,2,1] - lane[0,2] <= 0
 R402: tkey[1,0,2] - lane[0,2] <= 0
 R403: tkey[3,0,3] - lane[0,2] <= 0
 R404: tkey[1,2,4] - lane[0,2] <= 0
 R405: tkey[3,2,5] - lane[0,2] <= 0
 R406: tkey[1,1,6] - lane[0,2] <= 0
 R407: tkey[3,1,7] - lane[0,2] <= 0
 R408: tkey[0,3,0] - lane[0,3] <= 0
 R409: tkey[2,3,1] - lane[0,3] <= 0
 R410: tkey[1,3,2] - lane[0,3] <= 0
 R411: tkey[3,3,3] - lane[0,3] <= 0
 R412: tkey[0,1,4] - lane[0,3] <= 0
 R413: tkey[2,1,5] - lane[0,3] <= 0
 R414: tkey[0,0,6] - lane[0,3] <= 0
 R415: tkey[2,0,7] - lane[0,3] <= 0
 R416: tkey[1,0,0] - lane[1,0] <= 0
 R417: tkey[3,0,1] - lane[1,0] <= 0
 R418: tkey[1,2,2] - lane[1,0] <= 0
 R419: tkey[3,2,3] - lane[1,0] <= 0
 R420: tkey[1,1,4] - lane[1,0] <= 0
 R421: tkey[3,1,5] - lane[1,0] <= 0
 R422: tkey[0,3,6] - lane[1,0] <= 0
 R423: tkey[2,3,7] - lane[1,0] <= 0
 R424: tkey[1,1,0] - lane[1,1] <= 0
 R425: tkey[3,1,1] - lane[1,1] <= 0
 R426: tkey[0,3,2] - lane[1,1] <= 0
 R427: tkey[2,3,3] - lane[1,1] <= 0
 R428: tkey[1,3,4] - lane[1,1] <= 0
 R429: tkey[3,3,5] - lane[1,1] <= 0
 R430: tkey[0,1,6] - lane[1,1] <= 0
 R431: tkey[2,1,7] - lane[1,1] <= 0
 R432: tkey[1,2,0] - lane[1,2] <= 0
 R433: tkey[3,2,1] - lane[1,2] <= 0
 R434: tkey[1,1,2] - lane[1,2] <= 0
 R435: tkey[3,1,3] - lane[1,2] <= 0
 R436: tkey[0,3,4] - lane[1,2] <= 0
 R437: tkey[2,3,5] - lane[1,2] <= 0
 R438: tkey[1,3,6] - lane[1,2] <= 0
 R439: tkey[3,3,7] - lane[1,2] <= 0
 R440: tkey[1,3,0] - lane[1,3] <= 0
 R441: tkey[3,3,1] - lane[1,3] <= 0
 R442: tkey[0,1,2] - lane[1,3] <= 0
 R443: tkey[2,1,3] - lane[1,3] <= 0
 R444: tkey[0,0,4] - lane[1,3] <= 0
 R445: tkey[2,0,5] - lane[1,3] <= 0
 R446: tkey[0,2,6] - lane[1,3] <= 0
 R447: tkey[2,2,7] - lane[1,3] <= 0
 R448: tkey[2,0,0] - lane[2,0] <= 0
 R449: tkey[0,2,1] - lane[2,0] <= 0
 R450: tkey[2,2,2] - lane[2,0] <= 0
 R451: tkey[1,0,3] - lane[2,0] <= 0
 R452: tkey[3,0,4] - lane[2,0] <= 0
 R453: tkey[1,2,5] - lane[2,0] <= 0
 R454: tkey[3,2,6] - lane[2,0] <= 0
 R455: tkey[1,1,7] - lane[2,0] <= 0
 R456: tkey[2,1,0] - lane[2,1] <= 0
 R457: tkey[0,0,1] - lane[2,1] <= 0
 R458: tkey[2,0,2] - lane[2,1] <= 0
 R459: tkey[0,2,3] - lane[2,1] <= 0
 R460: tkey[2,2,4] - lane[2,1] <= 0
 R461: tkey[1,0,5] - lane[2,1] <= 0
 R462: tkey[3,0,6] - lane[2,1] <= 0
 R463: tkey[1,2,7] - lane[2,1] <= 0
 R464: tkey[2,2,0] - lane[2,2] <= 0
 R465: tkey[1,0,1] - lane[2,2] <= 0
 R466: tkey[3,0,2] - lane[2,2] <= 0
 R467: tkey[1,2,3] - lane[2,2] <= 0
 R468: tkey[3,2,4] - lane[2,2] <= 0
 R469: tkey[1,1,5] - lane[2,2] <= 0
 R470: tkey[3,1,6] - lane[2,2] <= 0
 R471: tkey[0,3,7] - lane[2,2] <= 0
 R472: tkey[2,3,0] - lane[2,3] <= 0
 R473: tkey[1,3,1] - lane[2,3] <= 0
 R474: tkey[3,3,2] - lane[2,3] <= 0
 R475: tkey[0,1,3] - lane[2,3] <= 0
 R476: tkey[2,1,4] - lane[2,3] <= 0
 R477: tkey[0,0,5] - lane[2,3] <= 0
 R478: tkey[2,0,6] - lane[2,3] <= 0
 R479: tkey[0,2,7] - lane[2,3] <= 0
 R480: tkey[3,0,0] - lane[3,0] <= 0
 R481: tkey[1,2,1] - lane[3,0] <= 0
 R482: tkey[3,2,2] - lane[3,0] <= 0
 R483: tkey[1,1,3] - lane[3,0] <= 0
 R484: tkey[3,1,4] - lane[3,0] <= 0
 R485: tkey[0,3,5] - lane[3,0] <= 0
 R486: tkey[2,3,6] - lane[3,0] <= 0
 R487: tkey[1,3,7] - lane[3,0] <= 0
 R488: tkey[3,1,0] - lane[3,1] <= 0
 R489: tkey[0,3,1] - lane[3,1] <= 0
 R490: tkey[2,3,2] - lane[3,1] <= 0
 R491: tkey[1,3,3] - lane[3,1] <= 0
 R492: tkey[3,3,4] - lane[3,1] <= 0
 R493: tkey[0,1,5] - lane[3,1] <= 0
 R494: tkey[2,1,6] - lane[3,1] <= 0
 R495: tkey[0,0,7] - lane[3,1] <= 0
 R496: tkey[3,2,0] - lane[3,2] <= 0
 R497: tkey[1,1,1] - lane[3,2] <= 0
 R498: tkey[3,1,2] - lane[3,2] <= 0
 R499: tkey[0,3,3] - lane[3,2] <= 0
 R500: tkey[2,3,4] - lane[3,2] <= 0
 R501: tkey[1,3,5] - lane[3,2] <= 0
 R502: tkey[3,3,6] - lane[3,2] <= 0
 R503: tkey[0,1,7] - lane[3,2] <= 0
 R504: tkey[3,3,0] - lane[3,3] <= 0
 R505: tkey[0,1,1] - lane[3,3] <= 0
 R506: tkey[2,1,2] - lane[3,3] <= 0
 R507: tkey[0,0,3] - lane[3,3] <= 0
 R508: tkey[2,0,4] - lane[3,3] <= 0
 R509: tkey[0,2,5] - lane[3,3] <= 0
 R510: tkey[2,2,6] - lane[3,3] <= 0
 R511: tkey[1,0,7] - lane[3,3] <= 0
 R512: - tkey[0,0,0] - tkey[0,2,2] - tkey[1,0,4] - tkey[1,2,6]
   - tkey[2,0,1] - tkey[2,2,3] - tkey[3,0,5] - tkey[3,2,7] + 8 lane[0,0]
   <= 1
 R513: - tkey[0,0,2] - tkey[0,1,0] - tkey[0,2,4] - tkey[1,0,6]
   - tkey[2,0,3] - tkey[2,1,1] - tkey[2,2,5] - tkey[3,0,7] + 8 lane[0,1]
   <= 1
 R514: - tkey[0,2,0] - tkey[1,0,2] - tkey[1,1,6] - tkey[1,2,4]
   - tkey[2,2,1] - tkey[3,0,3] - tkey[3,1,7] - tkey[3,2,5] + 8 lane[0,2]
   <= 1
 R515: - tkey[0,0,6] - tkey[0,1,4] - tkey[0,3,0] - tkey[1,3,2]
   - tkey[2,0,7] - tkey[2,1,5] - tkey[2,3,1] - tkey[3,3,3] + 8 lane[0,3]
   <= 1
 R516: - tkey[0,3,6] - tkey[1,0,0] - tkey[1,1,4] - tkey[1,2,2]
   - tkey[2,3,7] - tkey[3,0,1] - tkey[3,1,5] - tkey[3,2,3] + 8 lane[1,0]
   <= 1
 R517: - tkey[0,1,6] - tkey[0,3,2] - tkey[1,1,0] - tkey[1,3,4]
   - tkey[2,1,7] - tkey[2,3,3] - tkey[3,1,1] - tkey[3,3,5] + 8 lane[1,1]
   <= 1
 R518: - tkey[0,3,4] - tkey[1,1,2] - tkey[1,2,0] - tkey[1,3,6]
   - tkey[2,3,5] - tkey[3,1,3] - tkey[3,2,1] - tkey[3,3,7] + 8 lane[1,2]
   <= 1
 R519: - tkey[0,0,4] - tkey[0,1,2] - tkey[0,2,6] - tkey[1,3,0]
   - tkey[2,0,5] - tkey[2,1,3] - tkey[2,2,7] - tkey[3,3,1] + 8 lane[1,3]
   <= 1
 R520: - tkey[0,2,1] - tkey[1,0,3] - tkey[1,1,7] - tkey[1,2,5]
   - tkey[2,0,0] - tkey[2,2,2] - tkey[3,0,4] - tkey[3,2,6] + 8 lane[2,0]
   <= 1
 R521: - tkey[0,0,1] - tkey[0,2,3] - tkey[1,0,5] - tkey[1,2,7]
   - tkey[2,0,2] - tkey[2,1,0] - tkey[2,2,4] - tkey[3,0,6] + 8 lane[2,1]
   <= 1
 R522: - tkey[0,3,7] - tkey[1,0,1] - tkey[1,1,5] - tkey[1,2,3]
   - tkey[2,2,0] - tkey[3,0,2] - tkey[3,1,6] - tkey[3,2,4] + 8 lane[2,2]
   <= 1
 R523: - tkey[0,0,5] - tkey[0,1,3] - tkey[0,2,7] - tkey[1,3,1]
   - tkey[2,0,6] - tkey[2,1,4] - tkey[2,3,0] - tkey[3,3,2] + 8 lane[2,3]
   <= 1
 R524: - tkey[0,3,5] - tkey[1,1,3] - tkey[1,2,1] - tkey[1,3,7]
   - tkey[2,3,6] - tkey[3,0,0] - tkey[3,1,4] - tkey[3,2,2] + 8 lane[3,0]
   <= 1
 R525: - tkey[0,0,7] - tkey[0,1,5] - tkey[0,3,1] - tkey[1,3,3]
   - tkey[2,1,6] - tkey[2,3,2] - tkey[3,1,0] - tkey[3,3,4] + 8 lane[3,1]
   <= 1
 R526: - tkey[0,1,7] - tkey[0,3,3] - tkey[1,1,1] - tkey[1,3,5]
   - tkey[2,3,4] - tkey[3,1,2] - tkey[3,2,0] - tkey[3,3,6] + 8 lane[3,2]
   <= 1
 R527: - tkey[0,0,3] - tkey[0,1,1] - tkey[0,2,5] - tkey[1,0,7]
   - tkey[2,0,4] - tkey[2,1,2] - tkey[2,2,6] - tkey[3,3,0] + 8 lane[3,3]
   <= 1
 R528: - tbox[0,0,0] + forg[0,0,0] <= 0
 R529: sbox[0,0,0] - rkey[0,0,0] + tkey[0,0,0] >= 0
 R530: sbox[0,0,0] + rkey[0,0,0] - tkey[0,0,0] >= 0
 R531: - sbox[0,0,0] + rkey[0,0,0] + tkey[0,0,0] >= 0
 R532: - tbox[1,0,0] + forg[1,0,0] <= 0
 R533: sbox[1,0,0] - rkey[1,0,0] + tkey[1,0,0] >= 0
 R534: sbox[1,0,0] + rkey[1,0,0] - tkey[1,0,0] >= 0
 R535: - sbox[1,0,0] + rkey[1,0,0] + tkey[1,0,0] >= 0
 R536: - tbox[2,0,0] + forg[2,0,0] <= 0
 R537: - sbox[2,0,0] + rkey[2,0,0] = 0
 R538: - tbox[3,0,0] + forg[3,0,0] <= 0
 R539: - sbox[3,0,0] + rkey[3,0,0] = 0
 R540: - sbox[3,0,1] + rkey[0,0,0] + rkey[2,2,0] >= 0
 R541: sbox[3,0,1] + rkey[0,0,0] - rkey[2,2,0] >= 0
 R542: sbox[3,0,1] - rkey[0,0,0] + rkey[2,2,0] >= 0
 R543: - sbox[0,0,1] + sbox[3,0,1] + rkey[3,1,0] >= 0
 R544: sbox[0,0,1] + sbox[3,0,1] - rkey[3,1,0] >= 0
 R545: sbox[0,0,1] - sbox[3,0,1] + rkey[3,1,0] >= 0
 R546: - sbox[2,0,1] + rkey[1,3,0] + rkey[2,2,0] >= 0
 R547: sbox[2,0,1] + rkey[1,3,0] - rkey[2,2,0] >= 0
 R548: sbox[2,0,1] - rkey[1,3,0] + rkey[2,2,0] >= 0
 R549: sbox[1,0,1] - rkey[0,0,0] = 0
 R550: tbox[0,0,0] + tbox[2,2,0] - tbox[3,0,1] >= 0
 R551: tbox[0,0,0] - tbox[2,2,0] + tbox[3,0,1] >= 0
 R552: - tbox[0,0,0] + tbox[2,2,0] + tbox[3,0,1] >= 0
 R553: - tbox[0,0,1] + tbox[3,0,1] + tbox[3,1,0] >= 0
 R554: tbox[0,0,1] + tbox[3,0,1] - tbox[3,1,0] >= 0
 R555: tbox[0,0,1] - tbox[3,0,1] + tbox[3,1,0] >= 0
 R556: tbox[1,3,0] - tbox[2,0,1] + tbox[2,2,0] >= 0
 R557: tbox[1,3,0] + tbox[2,0,1] - tbox[2,2,0] >= 0
 R558: - tbox[1,3,0] + tbox[2,0,1] + tbox[2,2,0] >= 0
 R559: - tbox[0,0,0] + tbox[1,0,1] = 0
 R560: unkn[0,0,0] - unkn[3,0,1] + forg[3,0,1] <= 0
 R561: unkn[2,2,0] - unkn[3,0,1] + forg[3,0,1] <= 0
 R562: unkn[0,0,0] + unkn[2,2,0] - unkn[3,0,1] + forg[3,0,1] >= 0
 R563: - unkn[0,0,1] + unkn[3,0,1] + forg[0,0,1] - forg[3,0,1] <= 0
 R564: - unkn[0,0,1] + unkn[3,1,0] + forg[0,0,1] <= 0
 R565: - unkn[0,0,1] + unkn[3,0,1] + unkn[3,1,0] + forg[0,0,1]
   - forg[3,0,1] >= 0
 R566: unkn[1,3,0] - unkn[2,0,1] + forg[2,0,1] <= 0
 R567: - unkn[2,0,1] + unkn[2,2,0] + forg[2,0,1] <= 0
 R568: unkn[1,3,0] - unkn[2,0,1] + unkn[2,2,0] + forg[2,0,1] >= 0
 R569: unkn[0,0,0] - unkn[1,0,1] + forg[1,0,1] = 0
 R570: - tbox[0,1,0] + forg[0,1,0] <= 0
 R571: sbox[0,1,0] - rkey[0,1,0] + tkey[0,1,0] >= 0
 R572: sbox[0,1,0] + rkey[0,1,0] - tkey[0,1,0] >= 0
 R573: - sbox[0,1,0] + rkey[0,1,0] + tkey[0,1,0] >= 0
 R574: - tbox[1,1,0] + forg[1,1,0] <= 0
 R575: sbox[1,1,0] - rkey[1,1,0] + tkey[1,1,0] >= 0
 R576: sbox[1,1,0] + rkey[1,1,0] - tkey[1,1,0] >= 0
 R577: - sbox[1,1,0] + rkey[1,1,0] + tkey[1,1,0] >= 0
 R578: - tbox[2,1,0] + forg[2,1,0] <= 0
 R579: - sbox[2,1,0] + rkey[2,1,0] = 0
 R580: - tbox[3,1,0] + forg[3,1,0] <= 0
 R581: - sbox[3,1,0] + rkey[3,1,0] = 0
 R582: - sbox[3,1,1] + rkey[0,1,0] + rkey[2,3,0] >= 0
 R583: sbox[3,1,1] + rkey[0,1,0] - rkey[2,3,0] >= 0
 R584: sbox[3,1,1] - rkey[0,1,0] + rkey[2,3,0] >= 0
 R585: - sbox[0,1,1] + sbox[3,1,1] + rkey[3,2,0] >= 0
 R586: sbox[0,1,1] + sbox[3,1,1] - rkey[3,2,0] >= 0
 R587: sbox[0,1,1] - sbox[3,1,1] + rkey[3,2,0] >= 0
 R588: - sbox[2,1,1] + rkey[1,0,0] + rkey[2,3,0] >= 0
 R589: sbox[2,1,1] + rkey[1,0,0] - rkey[2,3,0] >= 0
 R590: sbox[2,1,1] - rkey[1,0,0] + rkey[2,3,0] >= 0
 R591: sbox[1,1,1] - rkey[0,1,0] = 0
 R592: tbox[0,1,0] + tbox[2,3,0] - tbox[3,1,1] >= 0
 R593: tbox[0,1,0] - tbox[2,3,0] + tbox[3,1,1] >= 0
 R594: - tbox[0,1,0] + tbox[2,3,0] + tbox[3,1,1] >= 0
 R595: - tbox[0,1,1] + tbox[3,1,1] + tbox[3,2,0] >= 0
 R596: tbox[0,1,1] + tbox[3,1,1] - tbox[3,2,0] >= 0
 R597: tbox[0,1,1] - tbox[3,1,1] + tbox[3,2,0] >= 0
 R598: tbox[1,0,0] - tbox[2,1,1] + tbox[2,3,0] >= 0
 R599: tbox[1,0,0] + tbox[2,1,1] - tbox[2,3,0] >= 0
 R600: - tbox[1,0,0] + tbox[2,1,1] + tbox[2,3,0] >= 0
 R601: - tbox[0,1,0] + tbox[1,1,1] = 0
 R602: unkn[0,1,0] - unkn[3,1,1] + forg[3,1,1] <= 0
 R603: unkn[2,3,0] - unkn[3,1,1] + forg[3,1,1] <= 0
 R604: unkn[0,1,0] + unkn[2,3,0] - unkn[3,1,1] + forg[3,1,1] >= 0
 R605: - unkn[0,1,1] + unkn[3,1,1] + forg[0,1,1] - forg[3,1,1] <= 0
 R606: - unkn[0,1,1] + unkn[3,2,0] + forg[0,1,1] <= 0
 R607: - unkn[0,1,1] + unkn[3,1,1] + unkn[3,2,0] + forg[0,1,1]
   - forg[3,1,1] >= 0
 R608: unkn[1,0,0] - unkn[2,1,1] + forg[2,1,1] <= 0
 R609: - unkn[2,1,1] + unkn[2,3,0] + forg[2,1,1] <= 0
 R610: unkn[1,0,0] - unkn[2,1,1] + unkn[2,3,0] + forg[2,1,1] >= 0
 R611: unkn[0,1,0] - unkn[1,1,1] + forg[1,1,1] = 0
 R612: - tbox[0,2,0] + forg[0,2,0] <= 0
 R613: sbox[0,2,0] - rkey[0,2,0] + tkey[0,2,0] >= 0
 R614: sbox[0,2,0] + rkey[0,2,0] - tkey[0,2,0] >= 0
 R615: - sbox[0,2,0] + rkey[0,2,0] + tkey[0,2,0] >= 0
 R616: - tbox[1,2,0] + forg[1,2,0] <= 0
 R617: sbox[1,2,0] - rkey[1,2,0] + tkey[1,2,0] >= 0
 R618: sbox[1,2,0] + rkey[1,2,0] - tkey[1,2,0] >= 0
 R619: - sbox[1,2,0] + rkey[1,2,0] + tkey[1,2,0] >= 0
 R620: - tbox[2,2,0] + forg[2,2,0] <= 0
 R621: - sbox[2,2,0] + rkey[2,2,0] = 0
 R622: - tbox[3,2,0] + forg[3,2,0] <= 0
 R623: - sbox[3,2,0] + rkey[3,2,0] = 0
 R624: - sbox[3,2,1] + rkey[0,2,0] + rkey[2,0,0] >= 0
 R625: sbox[3,2,1] + rkey[0,2,0] - rkey[2,0,0] >= 0
 R626: sbox[3,2,1] - rkey[0,2,0] + rkey[2,0,0] >= 0
 R627: - sbox[0,2,1] + sbox[3,2,1] + rkey[3,3,0] >= 0
 R628: sbox[0,2,1] + sbox[3,2,1] - rkey[3,3,0] >= 0
 R629: sbox[0,2,1] - sbox[3,2,1] + rkey[3,3,0] >= 0
 R630: - sbox[2,2,1] + rkey[1,1,0] + rkey[2,0,0] >= 0
 R631: sbox[2,2,1] + rkey[1,1,0] - rkey[2,0,0] >= 0
 R632: sbox[2,2,1] - rkey[1,1,0] + rkey[2,0,0] >= 0
 R633: sbox[1,2,1] - rkey[0,2,0] = 0
 R634: tbox[0,2,0] + tbox[2,0,0] - tbox[3,2,1] >= 0
 R635: tbox[0,2,0] - tbox[2,0,0] + tbox[3,2,1] >= 0
 R636: - tbox[0,2,0] + tbox[2,0,0] + tbox[3,2,1] >= 0
 R637: - tbox[0,2,1] + tbox[3,2,1] + tbox[3,3,0] >= 0
 R638: tbox[0,2,1] + tbox[3,2,1] - tbox[3,3,0] >= 0
 R639: tbox[0,2,1] - tbox[3,2,1] + tbox[3,3,0] >= 0
 R640: tbox[1,1,0] + tbox[2,0,0] - tbox[2,2,1] >= 0
 R641: tbox[1,1,0] - tbox[2,0,0] + tbox[2,2,1] >= 0
 R642: - tbox[1,1,0] + tbox[2,0,0] + tbox[2,2,1] >= 0
 R643: - tbox[0,2,0] + tbox[1,2,1] = 0
 R644: unkn[0,2,0] - unkn[3,2,1] + forg[3,2,1] <= 0
 R645: unkn[2,0,0] - unkn[3,2,1] + forg[3,2,1] <= 0
 R646: unkn[0,2,0] + unkn[2,0,0] - unkn[3,2,1] + forg[3,2,1] >= 0
 R647: - unkn[0,2,1] + unkn[3,2,1] + forg[0,2,1] - forg[3,2,1] <= 0
 R648: - unkn[0,2,1] + unkn[3,3,0] + forg[0,2,1] <= 0
 R649: - unkn[0,2,1] + unkn[3,2,1] + unkn[3,3,0] + forg[0,2,1]
   - forg[3,2,1] >= 0
 R650: unkn[1,1,0] - unkn[2,2,1] + forg[2,2,1] <= 0
 R651: unkn[2,0,0] - unkn[2,2,1] + forg[2,2,1] <= 0
 R652: unkn[1,1,0] + unkn[2,0,0] - unkn[2,2,1] + forg[2,2,1] >= 0
 R653: unkn[0,2,0] - unkn[1,2,1] + forg[1,2,1] = 0
 R654: - tbox[0,3,0] + forg[0,3,0] <= 0
 R655: sbox[0,3,0] - rkey[0,3,0] + tkey[0,3,0] >= 0
 R656: sbox[0,3,0] + rkey[0,3,0] - tkey[0,3,0] >= 0
 R657: - sbox[0,3,0] + rkey[0,3,0] + tkey[0,3,0] >= 0
 R658: - tbox[1,3,0] + forg[1,3,0] <= 0
 R659: sbox[1,3,0] - rkey[1,3,0] + tkey[1,3,0] >= 0
 R660: sbox[1,3,0] + rkey[1,3,0] - tkey[1,3,0] >= 0
 R661: - sbox[1,3,0] + rkey[1,3,0] + tkey[1,3,0] >= 0
 R662: - tbox[2,3,0] + forg[2,3,0] <= 0
 R663: - sbox[2,3,0] + rkey[2,3,0] = 0
 R664: - tbox[3,3,0] + forg[3,3,0] <= 0
 R665: - sbox[3,3,0] + rkey[3,3,0] = 0
 R666: - sbox[3,3,1] + rkey[0,3,0] + rkey[2,1,0] >= 0
 R667: sbox[3,3,1] + rkey[0,3,0] - rkey[2,1,0] >= 0
 R668: sbox[3,3,1] - rkey[0,3,0] + rkey[2,1,0] >= 0
 R669: - sbox[0,3,1] + sbox[3,3,1] + rkey[3,0,0] >= 0
 R670: sbox[0,3,1] + sbox[3,3,1] - rkey[3,0,0] >= 0
 R671: sbox[0,3,1] - sbox[3,3,1] + rkey[3,0,0] >= 0
 R672: - sbox[2,3,1] + rkey[1,2,0] + rkey[2,1,0] >= 0
 R673: sbox[2,3,1] + rkey[1,2,0] - rkey[2,1,0] >= 0
 R674: sbox[2,3,1] - rkey[1,2,0] + rkey[2,1,0] >= 0
 R675: sbox[1,3,1] - rkey[0,3,0] = 0
 R676: tbox[0,3,0] + tbox[2,1,0] - tbox[3,3,1] >= 0
 R677: tbox[0,3,0] - tbox[2,1,0] + tbox[3,3,1] >= 0
 R678: - tbox[0,3,0] + tbox[2,1,0] + tbox[3,3,1] >= 0
 R679: - tbox[0,3,1] + tbox[3,0,0] + tbox[3,3,1] >= 0
 R680: tbox[0,3,1] - tbox[3,0,0] + tbox[3,3,1] >= 0
 R681: tbox[0,3,1] + tbox[3,0,0] - tbox[3,3,1] >= 0
 R682: tbox[1,2,0] + tbox[2,1,0] - tbox[2,3,1] >= 0
 R683: tbox[1,2,0] - tbox[2,1,0] + tbox[2,3,1] >= 0
 R684: - tbox[1,2,0] + tbox[2,1,0] + tbox[2,3,1] >= 0
 R685: - tbox[0,3,0] + tbox[1,3,1] = 0
 R686: unkn[0,3,0] - unkn[3,3,1] + forg[3,3,1] <= 0
 R687: unkn[2,1,0] - unkn[3,3,1] + forg[3,3,1] <= 0
 R688: unkn[0,3,0] + unkn[2,1,0] - unkn[3,3,1] + forg[3,3,1] >= 0
 R689: - unkn[0,3,1] + unkn[3,3,1] + forg[0,3,1] - forg[3,3,1] <= 0
 R690: - unkn[0,3,1] + unkn[3,0,0] + forg[0,3,1] <= 0
 R691: - unkn[0,3,1] + unkn[3,0,0] + unkn[3,3,1] + forg[0,3,1]
   - forg[3,3,1] >= 0
 R692: unkn[1,2,0] - unkn[2,3,1] + forg[2,3,1] <= 0
 R693: unkn[2,1,0] - unkn[2,3,1] + forg[2,3,1] <= 0
 R694: unkn[1,2,0] + unkn[2,1,0] - unkn[2,3,1] + forg[2,3,1] >= 0
 R695: unkn[0,3,0] - unkn[1,3,1] + forg[1,3,1] = 0
 R696: - tbox[0,0,1] + forg[0,0,1] <= 0
 R697: sbox[0,0,1] - rkey[0,0,1] + tkey[0,0,1] >= 0
 R698: sbox[0,0,1] + rkey[0,0,1] - tkey[0,0,1] >= 0
 R699: - sbox[0,0,1] + rkey[0,0,1] + tkey[0,0,1] >= 0
 R700: - tbox[1,0,1] + forg[1,0,1] <= 0
 R701: sbox[1,0,1] - rkey[1,0,1] + tkey[1,0,1] >= 0
 R702: sbox[1,0,1] + rkey[1,0,1] - tkey[1,0,1] >= 0
 R703: - sbox[1,0,1] + rkey[1,0,1] + tkey[1,0,1] >= 0
 R704: - tbox[2,0,1] + forg[2,0,1] <= 0
 R705: - sbox[2,0,1] + rkey[2,0,1] = 0
 R706: - tbox[3,0,1] + forg[3,0,1] <= 0
 R707: - sbox[3,0,1] + rkey[3,0,1] = 0
 R708: - sbox[3,0,2] + rkey[0,0,1] + rkey[2,2,1] >= 0
 R709: sbox[3,0,2] + rkey[0,0,1] - rkey[2,2,1] >= 0
 R710: sbox[3,0,2] - rkey[0,0,1] + rkey[2,2,1] >= 0
 R711: - sbox[0,0,2] + sbox[3,0,2] + rkey[3,1,1] >= 0
 R712: sbox[0,0,2] + sbox[3,0,2] - rkey[3,1,1] >= 0
 R713: sbox[0,0,2] - sbox[3,0,2] + rkey[3,1,1] >= 0
 R714: - sbox[2,0,2] + rkey[1,3,1] + rkey[2,2,1] >= 0
 R715: sbox[2,0,2] + rkey[1,3,1] - rkey[2,2,1] >= 0
 R716: sbox[2,0,2] - rkey[1,3,1] + rkey[2,2,1] >= 0
 R717: sbox[1,0,2] - rkey[0,0,1] = 0
 R718: tbox[0,0,1] + tbox[2,2,1] - tbox[3,0,2] >= 0
 R719: tbox[0,0,1] - tbox[2,2,1] + tbox[3,0,2] >= 0
 R720: - tbox[0,0,1] + tbox[2,2,1] + tbox[3,0,2] >= 0
 R721: - tbox[0,0,2] + tbox[3,0,2] + tbox[3,1,1] >= 0
 R722: tbox[0,0,2] + tbox[3,0,2] - tbox[3,1,1] >= 0
 R723: tbox[0,0,2] - tbox[3,0,2] + tbox[3,1,1] >= 0
 R724: tbox[1,3,1] - tbox[2,0,2] + tbox[2,2,1] >= 0
 R725: tbox[1,3,1] + tbox[2,0,2] - tbox[2,2,1] >= 0
 R726: - tbox[1,3,1] + tbox[2,0,2] + tbox[2,2,1] >= 0
 R727: - tbox[0,0,1] + tbox[1,0,2] = 0
 R728: unkn[0,0,1] - unkn[3,0,2] + forg[3,0,2] <= 0
 R729: unkn[2,2,1] - unkn[3,0,2] + forg[3,0,2] <= 0
 R730: unkn[0,0,1] + unkn[2,2,1] - unkn[3,0,2] + forg[3,0,2] >= 0
 R731: - unkn[0,0,2] + unkn[3,0,2] + forg[0,0,2] - forg[3,0,2] <= 0
 R732: - unkn[0,0,2] + unkn[3,1,1] + forg[0,0,2] <= 0
 R733: - unkn[0,0,2] + unkn[3,0,2] + unkn[3,1,1] + forg[0,0,2]
   - forg[3,0,2] >= 0
 R734: unkn[1,3,1] - unkn[2,0,2] + forg[2,0,2] <= 0
 R735: - unkn[2,0,2] + unkn[2,2,1] + forg[2,0,2] <= 0
 R736: unkn[1,3,1] - unkn[2,0,2] + unkn[2,2,1] + forg[2,0,2] >= 0
 R737: unkn[0,0,1] - unkn[1,0,2] + forg[1,0,2] = 0
 R738: - tbox[0,1,1] + forg[0,1,1] <= 0
 R739: sbox[0,1,1] - rkey[0,1,1] + tkey[0,1,1] >= 0
 R740: sbox[0,1,1] + rkey[0,1,1] - tkey[0,1,1] >= 0
 R741: - sbox[0,1,1] + rkey[0,1,1] + tkey[0,1,1] >= 0
 R742: - tbox[1,1,1] + forg[1,1,1] <= 0
 R743: sbox[1,1,1] - rkey[1,1,1] + tkey[1,1,1] >= 0
 R744: sbox[1,1,1] + rkey[1,1,1] - tkey[1,1,1] >= 0
 R745: - sbox[1,1,1] + rkey[1,1,1] + tkey[1,1,1] >= 0
 R746: - tbox[2,1,1] + forg[2,1,1] <= 0
 R747: - sbox[2,1,1] + rkey[2,1,1] = 0
 R748: - tbox[3,1,1] + forg[3,1,1] <= 0
 R749: - sbox[3,1,1] + rkey[3,1,1] = 0
 R750: - sbox[3,1,2] + rkey[0,1,1] + rkey[2,3,1] >= 0
 R751: sbox[3,1,2] + rkey[0,1,1] - rkey[2,3,1] >= 0
 R752: sbox[3,1,2] - rkey[0,1,1] + rkey[2,3,1] >= 0
 R753: - sbox[0,1,2] + sbox[3,1,2] + rkey[3,2,1] >= 0
 R754: sbox[0,1,2] + sbox[3,1,2] - rkey[3,2,1] >= 0
 R755: sbox[0,1,2] - sbox[3,1,2] + rkey[3,2,1] >= 0
 R756: - sbox[2,1,2] + rkey[1,0,1] + rkey[2,3,1] >= 0
 R757: sbox[2,1,2] + rkey[1,0,1] - rkey[2,3,1] >= 0
 R758: sbox[2,1,2] - rkey[1,0,1] + rkey[2,3,1] >= 0
 R759: sbox[1,1,2] - rkey[0,1,1] = 0
 R760: tbox[0,1,1] + tbox[2,3,1] - tbox[3,1,2] >= 0
 R761: tbox[0,1,1] - tbox[2,3,1] + tbox[3,1,2] >= 0
 R762: - tbox[0,1,1] + tbox[2,3,1] + tbox[3,1,2] >= 0
 R763: - tbox[0,1,2] + tbox[3,1,2] + tbox[3,2,1] >= 0
 R764: tbox[0,1,2] + tbox[3,1,2] - tbox[3,2,1] >= 0
 R765: tbox[0,1,2] - tbox[3,1,2] + tbox[3,2,1] >= 0
 R766: tbox[1,0,1] - tbox[2,1,2] + tbox[2,3,1] >= 0
 R767: tbox[1,0,1] + tbox[2,1,2] - tbox[2,3,1] >= 0
 R768: - tbox[1,0,1] + tbox[2,1,2] + tbox[2,3,1] >= 0
 R769: - tbox[0,1,1] + tbox[1,1,2] = 0
 R770: unkn[0,1,1] - unkn[3,1,2] + forg[3,1,2] <= 0
 R771: unkn[2,3,1] - unkn[3,1,2] + forg[3,1,2] <= 0
 R772: unkn[0,1,1] + unkn[2,3,1] - unkn[3,1,2] + forg[3,1,2] >= 0
 R773: - unkn[0,1,2] + unkn[3,1,2] + forg[0,1,2] - forg[3,1,2] <= 0
 R774: - unkn[0,1,2] + unkn[3,2,1] + forg[0,1,2] <= 0
 R775: - unkn[0,1,2] + unkn[3,1,2] + unkn[3,2,1] + forg[0,1,2]
   - forg[3,1,2] >= 0
 R776: unkn[1,0,1] - unkn[2,1,2] + forg[2,1,2] <= 0
 R777: - unkn[2,1,2] + unkn[2,3,1] + forg[2,1,2] <= 0
 R778: unkn[1,0,1] - unkn[2,1,2] + unkn[2,3,1] + forg[2,1,2] >= 0
 R779: unkn[0,1,1] - unkn[1,1,2] + forg[1,1,2] = 0
 R780: - tbox[0,2,1] + forg[0,2,1] <= 0
 R781: sbox[0,2,1] - rkey[0,2,1] + tkey[0,2,1] >= 0
 R782: sbox[0,2,1] + rkey[0,2,1] - tkey[0,2,1] >= 0
 R783: - sbox[0,2,1] + rkey[0,2,1] + tkey[0,2,1] >= 0
 R784: - tbox[1,2,1] + forg[1,2,1] <= 0
 R785: sbox[1,2,1] - rkey[1,2,1] + tkey[1,2,1] >= 0
 R786: sbox[1,2,1] + rkey[1,2,1] - tkey[1,2,1] >= 0
 R787: - sbox[1,2,1] + rkey[1,2,1] + tkey[1,2,1] >= 0
 R788: - tbox[2,2,1] + forg[2,2,1] <= 0
 R789: - sbox[2,2,1] + rkey[2,2,1] = 0
 R790: - tbox[3,2,1] + forg[3,2,1] <= 0
 R791: - sbox[3,2,1] + rkey[3,2,1] = 0
 R792: - sbox[3,2,2] + rkey[0,2,1] + rkey[2,0,1] >= 0
 R793: sbox[3,2,2] + rkey[0,2,1] - rkey[2,0,1] >= 0
 R794: sbox[3,2,2] - rkey[0,2,1] + rkey[2,0,1] >= 0
 R795: - sbox[0,2,2] + sbox[3,2,2] + rkey[3,3,1] >= 0
 R796: sbox[0,2,2] + sbox[3,2,2] - rkey[3,3,1] >= 0
 R797: sbox[0,2,2] - sbox[3,2,2] + rkey[3,3,1] >= 0
 R798: - sbox[2,2,2] + rkey[1,1,1] + rkey[2,0,1] >= 0
 R799: sbox[2,2,2] + rkey[1,1,1] - rkey[2,0,1] >= 0
 R800: sbox[2,2,2] - rkey[1,1,1] + rkey[2,0,1] >= 0
 R801: sbox[1,2,2] - rkey[0,2,1] = 0
 R802: tbox[0,2,1] + tbox[2,0,1] - tbox[3,2,2] >= 0
 R803: tbox[0,2,1] - tbox[2,0,1] + tbox[3,2,2] >= 0
 R804: - tbox[0,2,1] + tbox[2,0,1] + tbox[3,2,2] >= 0
 R805: - tbox[0,2,2] + tbox[3,2,2] + tbox[3,3,1] >= 0
 R806: tbox[0,2,2] + tbox[3,2,2] - tbox[3,3,1] >= 0
 R807: tbox[0,2,2] - tbox[3,2,2] + tbox[3,3,1] >= 0
 R808: tbox[1,1,1] + tbox[2,0,1] - tbox[2,2,2] >= 0
 R809: tbox[1,1,1] - tbox[2,0,1] + tbox[2,2,2] >= 0
 R810: - tbox[1,1,1] + tbox[2,0,1] + tbox[2,2,2] >= 0
 R811: - tbox[0,2,1] + tbox[1,2,2] = 0
 R812: unkn[0,2,1] - unkn[3,2,2] + forg[3,2,2] <= 0
 R813: unkn[2,0,1] - unkn[3,2,2] + forg[3,2,2] <= 0
 R814: unkn[0,2,1] + unkn[2,0,1] - unkn[3,2,2] + forg[3,2,2] >= 0
 R815: - unkn[0,2,2] + unkn[3,2,2] + forg[0,2,2] - forg[3,2,2] <= 0
 R816: - unkn[0,2,2] + unkn[3,3,1] + forg[0,2,2] <= 0
 R817: - unkn[0,2,2] + unkn[3,2,2] + unkn[3,3,1] + forg[0,2,2]
   - forg[3,2,2] >= 0
 R818: unkn[1,1,1] - unkn[2,2,2] + forg[2,2,2] <= 0
 R819: unkn[2,0,1] - unkn[2,2,2] + forg[2,2,2] <= 0
 R820: unkn[1,1,1] + unkn[2,0,1] - unkn[2,2,2] + forg[2,2,2] >= 0
 R821: unkn[0,2,1] - unkn[1,2,2] + forg[1,2,2] = 0
 R822: - tbox[0,3,1] + forg[0,3,1] <= 0
 R823: sbox[0,3,1] - rkey[0,3,1] + tkey[0,3,1] >= 0
 R824: sbox[0,3,1] + rkey[0,3,1] - tkey[0,3,1] >= 0
 R825: - sbox[0,3,1] + rkey[0,3,1] + tkey[0,3,1] >= 0
 R826: - tbox[1,3,1] + forg[1,3,1] <= 0
 R827: sbox[1,3,1] - rkey[1,3,1] + tkey[1,3,1] >= 0
 R828: sbox[1,3,1] + rkey[1,3,1] - tkey[1,3,1] >= 0
 R829: - sbox[1,3,1] + rkey[1,3,1] + tkey[1,3,1] >= 0
 R830: - tbox[2,3,1] + forg[2,3,1] <= 0
 R831: - sbox[2,3,1] + rkey[2,3,1] = 0
 R832: - tbox[3,3,1] + forg[3,3,1] <= 0
 R833: - sbox[3,3,1] + rkey[3,3,1] = 0
 R834: - sbox[3,3,2] + rkey[0,3,1] + rkey[2,1,1] >= 0
 R835: sbox[3,3,2] + rkey[0,3,1] - rkey[2,1,1] >= 0
 R836: sbox[3,3,2] - rkey[0,3,1] + rkey[2,1,1] >= 0
 R837: - sbox[0,3,2] + sbox[3,3,2] + rkey[3,0,1] >= 0
 R838: sbox[0,3,2] + sbox[3,3,2] - rkey[3,0,1] >= 0
 R839: sbox[0,3,2] - sbox[3,3,2] + rkey[3,0,1] >= 0
 R840: - sbox[2,3,2] + rkey[1,2,1] + rkey[2,1,1] >= 0
 R841: sbox[2,3,2] + rkey[1,2,1] - rkey[2,1,1] >= 0
 R842: sbox[2,3,2] - rkey[1,2,1] + rkey[2,1,1] >= 0
 R843: sbox[1,3,2] - rkey[0,3,1] = 0
 R844: tbox[0,3,1] + tbox[2,1,1] - tbox[3,3,2] >= 0
 R845: tbox[0,3,1] - tbox[2,1,1] + tbox[3,3,2] >= 0
 R846: - tbox[0,3,1] + tbox[2,1,1] + tbox[3,3,2] >= 0
 R847: - tbox[0,3,2] + tbox[3,0,1] + tbox[3,3,2] >= 0
 R848: tbox[0,3,2] - tbox[3,0,1] + tbox[3,3,2] >= 0
 R849: tbox[0,3,2] + tbox[3,0,1] - tbox[3,3,2] >= 0
 R850: tbox[1,2,1] + tbox[2,1,1] - tbox[2,3,2] >= 0
 R851: tbox[1,2,1] - tbox[2,1,1] + tbox[2,3,2] >= 0
 R852: - tbox[1,2,1] + tbox[2,1,1] + tbox[2,3,2] >= 0
 R853: - tbox[0,3,1] + tbox[1,3,2] = 0
 R854: unkn[0,3,1] - unkn[3,3,2] + forg[3,3,2] <= 0
 R855: unkn[2,1,1] - unkn[3,3,2] + forg[3,3,2] <= 0
 R856: unkn[0,3,1] + unkn[2,1,1] - unkn[3,3,2] + forg[3,3,2] >= 0
 R857: - unkn[0,3,2] + unkn[3,3,2] + forg[0,3,2] - forg[3,3,2] <= 0
 R858: - unkn[0,3,2] + unkn[3,0,1] + forg[0,3,2] <= 0
 R859: - unkn[0,3,2] + unkn[3,0,1] + unkn[3,3,2] + forg[0,3,2]
   - forg[3,3,2] >= 0
 R860: unkn[1,2,1] - unkn[2,3,2] + forg[2,3,2] <= 0
 R861: unkn[2,1,1] - unkn[2,3,2] + forg[2,3,2] <= 0
 R862: unkn[1,2,1] + unkn[2,1,1] - unkn[2,3,2] + forg[2,3,2] >= 0
 R863: unkn[0,3,1] - unkn[1,3,2] + forg[1,3,2] = 0
 R864: - tbox[0,0,2] + forg[0,0,2] <= 0
 R865: sbox[0,0,2] - rkey[0,0,2] + tkey[0,0,2] >= 0
 R866: sbox[0,0,2] + rkey[0,0,2] - tkey[0,0,2] >= 0
 R867: - sbox[0,0,2] + rkey[0,0,2] + tkey[0,0,2] >= 0
 R868: - tbox[1,0,2] + forg[1,0,2] <= 0
 R869: sbox[1,0,2] - rkey[1,0,2] + tkey[1,0,2] >= 0
 R870: sbox[1,0,2] + rkey[1,0,2] - tkey[1,0,2] >= 0
 R871: - sbox[1,0,2] + rkey[1,0,2] + tkey[1,0,2] >= 0
 R872: - tbox[2,0,2] + forg[2,0,2] <= 0
 R873: - sbox[2,0,2] + rkey[2,0,2] = 0
 R874: - tbox[3,0,2] + forg[3,0,2] <= 0
 R875: - sbox[3,0,2] + rkey[3,0,2] = 0
 R876: - sbox[3,0,3] + rkey[0,0,2] + rkey[2,2,2] >= 0
 R877: sbox[3,0,3] + rkey[0,0,2] - rkey[2,2,2] >= 0
 R878: sbox[3,0,3] - rkey[0,0,2] + rkey[2,2,2] >= 0
 R879: - sbox[0,0,3] + sbox[3,0,3] + rkey[3,1,2] >= 0
 R880: sbox[0,0,3] + sbox[3,0,3] - rkey[3,1,2] >= 0
 R881: sbox[0,0,3] - sbox[3,0,3] + rkey[3,1,2] >= 0
 R882: - sbox[2,0,3] + rkey[1,3,2] + rkey[2,2,2] >= 0
 R883: sbox[2,0,3] + rkey[1,3,2] - rkey[2,2,2] >= 0
 R884: sbox[2,0,3] - rkey[1,3,2] + rkey[2,2,2] >= 0
 R885: sbox[1,0,3] - rkey[0,0,2] = 0
 R886: tbox[0,0,2] + tbox[2,2,2] - tbox[3,0,3] >= 0
 R887: tbox[0,0,2] - tbox[2,2,2] + tbox[3,0,3] >= 0
 R888: - tbox[0,0,2] + tbox[2,2,2] + tbox[3,0,3] >= 0
 R889: - tbox[0,0,3] + tbox[3,0,3] + tbox[3,1,2] >= 0
 R890: tbox[0,0,3] + tbox[3,0,3] - tbox[3,1,2] >= 0
 R891: tbox[0,0,3] - tbox[3,0,3] + tbox[3,1,2] >= 0
 R892: tbox[1,3,2] - tbox[2,0,3] + tbox[2,2,2] >= 0
 R893: tbox[1,3,2] + tbox[2,0,3] - tbox[2,2,2] >= 0
 R894: - tbox[1,3,2] + tbox[2,0,3] + tbox[2,2,2] >= 0
 R895: - tbox[0,0,2] + tbox[1,0,3] = 0
 R896: unkn[0,0,2] - unkn[3,0,3] + forg[3,0,3] <= 0
 R897: unkn[2,2,2] - unkn[3,0,3] + forg[3,0,3] <= 0
 R898: unkn[0,0,2] + unkn[2,2,2] - unkn[3,0,3] + forg[3,0,3] >= 0
 R899: - unkn[0,0,3] + unkn[3,0,3] + forg[0,0,3] - forg[3,0,3] <= 0
 R900: - unkn[0,0,3] + unkn[3,1,2] + forg[0,0,3] <= 0
 R901: - unkn[0,0,3] + unkn[3,0,3] + unkn[3,1,2] + forg[0,0,3]
   - forg[3,0,3] >= 0
 R902: unkn[1,3,2] - unkn[2,0,3] + forg[2,0,3] <= 0
 R903: - unkn[2,0,3] + unkn[2,2,2] + forg[2,0,3] <= 0
 R904: unkn[1,3,2] - unkn[2,0,3] + unkn[2,2,2] + forg[2,0,3] >= 0
 R905: unkn[0,0,2] - unkn[1,0,3] + forg[1,0,3] = 0
 R906: - tbox[0,1,2] + forg[0,1,2] <= 0
 R907: sbox[0,1,2] - rkey[0,1,2] + tkey[0,1,2] >= 0
 R908: sbox[0,1,2] + rkey[0,1,2] - tkey[0,1,2] >= 0
 R909: - sbox[0,1,2] + rkey[0,1,2] + tkey[0,1,2] >= 0
 R910: - tbox[1,1,2] + forg[1,1,2] <= 0
 R911: sbox[1,1,2] - rkey[1,1,2] + tkey[1,1,2] >= 0
 R912: sbox[1,1,2] + rkey[1,1,2] - tkey[1,1,2] >= 0
 R913: - sbox[1,1,2] + rkey[1,1,2] + tkey[1,1,2] >= 0
 R914: - tbox[2,1,2] + forg[2,1,2] <= 0
 R915: - sbox[2,1,2] + rkey[2,1,2] = 0
 R916: - tbox[3,1,2] + forg[3,1,2] <= 0
 R917: - sbox[3,1,2] + rkey[3,1,2] = 0
 R918: - sbox[3,1,3] + rkey[0,1,2] + rkey[2,3,2] >= 0
 R919: sbox[3,1,3] + rkey[0,1,2] - rkey[2,3,2] >= 0
 R920: sbox[3,1,3] - rkey[0,1,2] + rkey[2,3,2] >= 0
 R921: - sbox[0,1,3] + sbox[3,1,3] + rkey[3,2,2] >= 0
 R922: sbox[0,1,3] + sbox[3,1,3] - rkey[3,2,2] >= 0
 R923: sbox[0,1,3] - sbox[3,1,3] + rkey[3,2,2] >= 0
 R924: - sbox[2,1,3] + rkey[1,0,2] + rkey[2,3,2] >= 0
 R925: sbox[2,1,3] + rkey[1,0,2] - rkey[2,3,2] >= 0
 R926: sbox[2,1,3] - rkey[1,0,2] + rkey[2,3,2] >= 0
 R927: sbox[1,1,3] - rkey[0,1,2] = 0
 R928: tbox[0,1,2] + tbox[2,3,2] - tbox[3,1,3] >= 0
 R929: tbox[0,1,2] - tbox[2,3,2] + tbox[3,1,3] >= 0
 R930: - tbox[0,1,2] + tbox[2,3,2] + tbox[3,1,3] >= 0
 R931: - tbox[0,1,3] + tbox[3,1,3] + tbox[3,2,2] >= 0
 R932: tbox[0,1,3] + tbox[3,1,3] - tbox[3,2,2] >= 0
 R933: tbox[0,1,3] - tbox[3,1,3] + tbox[3,2,2] >= 0
 R934: tbox[1,0,2] - tbox[2,1,3] + tbox[2,3,2] >= 0
 R935: tbox[1,0,2] + tbox[2,1,3] - tbox[2,3,2] >= 0
 R936: - tbox[1,0,2] + tbox[2,1,3] + tbox[2,3,2] >= 0
 R937: - tbox[0,1,2] + tbox[1,1,3] = 0
 R938: unkn[0,1,2] - unkn[3,1,3] + forg[3,1,3] <= 0
 R939: unkn[2,3,2] - unkn[3,1,3] + forg[3,1,3] <= 0
 R940: unkn[0,1,2] + unkn[2,3,2] - unkn[3,1,3] + forg[3,1,3] >= 0
 R941: - unkn[0,1,3] + unkn[3,1,3] + forg[0,1,3] - forg[3,1,3] <= 0
 R942: - unkn[0,1,3] + unkn[3,2,2] + forg[0,1,3] <= 0
 R943: - unkn[0,1,3] + unkn[3,1,3] + unkn[3,2,2] + forg[0,1,3]
   - forg[3,1,3] >= 0
 R944: unkn[1,0,2] - unkn[2,1,3] + forg[2,1,3] <= 0
 R945: - unkn[2,1,3] + unkn[2,3,2] + forg[2,1,3] <= 0
 R946: unkn[1,0,2] - unkn[2,1,3] + unkn[2,3,2] + forg[2,1,3] >= 0
 R947: unkn[0,1,2] - unkn[1,1,3] + forg[1,1,3] = 0
 R948: - tbox[0,2,2] + forg[0,2,2] <= 0
 R949: sbox[0,2,2] - rkey[0,2,2] + tkey[0,2,2] >= 0
 R950: sbox[0,2,2] + rkey[0,2,2] - tkey[0,2,2] >= 0
 R951: - sbox[0,2,2] + rkey[0,2,2] + tkey[0,2,2] >= 0
 R952: - tbox[1,2,2] + forg[1,2,2] <= 0
 R953: sbox[1,2,2] - rkey[1,2,2] + tkey[1,2,2] >= 0
 R954: sbox[1,2,2] + rkey[1,2,2] - tkey[1,2,2] >= 0
 R955: - sbox[1,2,2] + rkey[1,2,2] + tkey[1,2,2] >= 0
 R956: - tbox[2,2,2] + forg[2,2,2] <= 0
 R957: - sbox[2,2,2] + rkey[2,2,2] = 0
 R958: - tbox[3,2,2] + forg[3,2,2] <= 0
 R959: - sbox[3,2,2] + rkey[3,2,2] = 0
 R960: - sbox[3,2,3] + rkey[0,2,2] + rkey[2,0,2] >= 0
 R961: sbox[3,2,3] + rkey[0,2,2] - rkey[2,0,2] >= 0
 R962: sbox[3,2,3] - rkey[0,2,2] + rkey[2,0,2] >= 0
 R963: - sbox[0,2,3] + sbox[3,2,3] + rkey[3,3,2] >= 0
 R964: sbox[0,2,3] + sbox[3,2,3] - rkey[3,3,2] >= 0
 R965: sbox[0,2,3] - sbox[3,2,3] + rkey[3,3,2] >= 0
 R966: - sbox[2,2,3] + rkey[1,1,2] + rkey[2,0,2] >= 0
 R967: sbox[2,2,3] + rkey[1,1,2] - rkey[2,0,2] >= 0
 R968: sbox[2,2,3] - rkey[1,1,2] + rkey[2,0,2] >= 0
 R969: sbox[1,2,3] - rkey[0,2,2] = 0
 R970: tbox[0,2,2] + tbox[2,0,2] - tbox[3,2,3] >= 0
 R971: tbox[0,2,2] - tbox[2,0,2] + tbox[3,2,3] >= 0
 R972: - tbox[0,2,2] + tbox[2,0,2] + tbox[3,2,3] >= 0
 R973: - tbox[0,2,3] + tbox[3,2,3] + tbox[3,3,2] >= 0
 R974: tbox[0,2,3] + tbox[3,2,3] - tbox[3,3,2] >= 0
 R975: tbox[0,2,3] - tbox[3,2,3] + tbox[3,3,2] >= 0
 R976: tbox[1,1,2] + tbox[2,0,2] - tbox[2,2,3] >= 0
 R977: tbox[1,1,2] - tbox[2,0,2] + tbox[2,2,3] >= 0
 R978: - tbox[1,1,2] + tbox[2,0,2] + tbox[2,2,3] >= 0
 R979: - tbox[0,2,2] + tbox[1,2,3] = 0
 R980: unkn[0,2,2] - unkn[3,2,3] + forg[3,2,3] <= 0
 R981: unkn[2,0,2] - unkn[3,2,3] + forg[3,2,3] <= 0
 R982: unkn[0,2,2] + unkn[2,0,2] - unkn[3,2,3] + forg[3,2,3] >= 0
 R983: - unkn[0,2,3] + unkn[3,2,3] + forg[0,2,3] - forg[3,2,3] <= 0
 R984: - unkn[0,2,3] + unkn[3,3,2] + forg[0,2,3] <= 0
 R985: - unkn[0,2,3] + unkn[3,2,3] + unkn[3,3,2] + forg[0,2,3]
   - forg[3,2,3] >= 0
 R986: unkn[1,1,2] - unkn[2,2,3] + forg[2,2,3] <= 0
 R987: unkn[2,0,2] - unkn[2,2,3] + forg[2,2,3] <= 0
 R988: unkn[1,1,2] + unkn[2,0,2] - unkn[2,2,3] + forg[2,2,3] >= 0
 R989: unkn[0,2,2] - unkn[1,2,3] + forg[1,2,3] = 0
 R990: - tbox[0,3,2] + forg[0,3,2] <= 0
 R991: sbox[0,3,2] - rkey[0,3,2] + tkey[0,3,2] >= 0
 R992: sbox[0,3,2] + rkey[0,3,2] - tkey[0,3,2] >= 0
 R993: - sbox[0,3,2] + rkey[0,3,2] + tkey[0,3,2] >= 0
 R994: - tbox[1,3,2] + forg[1,3,2] <= 0
 R995: sbox[1,3,2] - rkey[1,3,2] + tkey[1,3,2] >= 0
 R996: sbox[1,3,2] + rkey[1,3,2] - tkey[1,3,2] >= 0
 R997: - sbox[1,3,2] + rkey[1,3,2] + tkey[1,3,2] >= 0
 R998: - tbox[2,3,2] + forg[2,3,2] <= 0
 R999: - sbox[2,3,2] + rkey[2,3,2] = 0
 R1000: - tbox[3,3,2] + forg[3,3,2] <= 0
 R1001: - sbox[3,3,2] + rkey[3,3,2] = 0
 R1002: - sbox[3,3,3] + rkey[0,3,2] + rkey[2,1,2] >= 0
 R1003: sbox[3,3,3] + rkey[0,3,2] - rkey[2,1,2] >= 0
 R1004: sbox[3,3,3] - rkey[0,3,2] + rkey[2,1,2] >= 0
 R1005: - sbox[0,3,3] + sbox[3,3,3] + rkey[3,0,2] >= 0
 R1006: sbox[0,3,3] + sbox[3,3,3] - rkey[3,0,2] >= 0
 R1007: sbox[0,3,3] - sbox[3,3,3] + rkey[3,0,2] >= 0
 R1008: - sbox[2,3,3] + rkey[1,2,2] + rkey[2,1,2] >= 0
 R1009: sbox[2,3,3] + rkey[1,2,2] - rkey[2,1,2] >= 0
 R1010: sbox[2,3,3] - rkey[1,2,2] + rkey[2,1,2] >= 0
 R1011: sbox[1,3,3] - rkey[0,3,2] = 0
 R1012: tbox[0,3,2] + tbox[2,1,2] - tbox[3,3,3] >= 0
 R1013: tbox[0,3,2] - tbox[2,1,2] + tbox[3,3,3] >= 0
 R1014: - tbox[0,3,2] + tbox[2,1,2] + tbox[3,3,3] >= 0
 R1015: - tbox[0,3,3] + tbox[3,0,2] + tbox[3,3,3] >= 0
 R1016: tbox[0,3,3] - tbox[3,0,2] + tbox[3,3,3] >= 0
 R1017: tbox[0,3,3] + tbox[3,0,2] - tbox[3,3,3] >= 0
 R1018: tbox[1,2,2] + tbox[2,1,2] - tbox[2,3,3] >= 0
 R1019: tbox[1,2,2] - tbox[2,1,2] + tbox[2,3,3] >= 0
 R1020: - tbox[1,2,2] + tbox[2,1,2] + tbox[2,3,3] >= 0
 R1021: - tbox[0,3,2] + tbox[1,3,3] = 0
 R1022: unkn[0,3,2] - unkn[3,3,3] + forg[3,3,3] <= 0
 R1023: unkn[2,1,2] - unkn[3,3,3] + forg[3,3,3] <= 0
 R1024: unkn[0,3,2] + unkn[2,1,2] - unkn[3,3,3] + forg[3,3,3] >= 0
 R1025: - unkn[0,3,3] + unkn[3,3,3] + forg[0,3,3] - forg[3,3,3] <= 0
 R1026: - unkn[0,3,3] + unkn[3,0,2] + forg[0,3,3] <= 0
 R1027: - unkn[0,3,3] + unkn[3,0,2] + unkn[3,3,3] + forg[0,3,3]
   - forg[3,3,3] >= 0
 R1028: unkn[1,2,2] - unkn[2,3,3] + forg[2,3,3] <= 0
 R1029: unkn[2,1,2] - unkn[2,3,3] + forg[2,3,3] <= 0
 R1030: unkn[1,2,2] + unkn[2,1,2] - unkn[2,3,3] + forg[2,3,3] >= 0
 R1031: unkn[0,3,2] - unkn[1,3,3] + forg[1,3,3] = 0
 R1032: - tbox[0,0,3] + forg[0,0,3] <= 0
 R1033: sbox[0,0,3] - rkey[0,0,3] + tkey[0,0,3] >= 0
 R1034: sbox[0,0,3] + rkey[0,0,3] - tkey[0,0,3] >= 0
 R1035: - sbox[0,0,3] + rkey[0,0,3] + tkey[0,0,3] >= 0
 R1036: - tbox[1,0,3] + forg[1,0,3] <= 0
 R1037: sbox[1,0,3] - rkey[1,0,3] + tkey[1,0,3] >= 0
 R1038: sbox[1,0,3] + rkey[1,0,3] - tkey[1,0,3] >= 0
 R1039: - sbox[1,0,3] + rkey[1,0,3] + tkey[1,0,3] >= 0
 R1040: - tbox[2,0,3] + forg[2,0,3] <= 0
 R1041: - sbox[2,0,3] + rkey[2,0,3] = 0
 R1042: - tbox[3,0,3] + forg[3,0,3] <= 0
 R1043: - sbox[3,0,3] + rkey[3,0,3] = 0
 R1044: - sbox[3,0,4] + rkey[0,0,3] + rkey[2,2,3] >= 0
 R1045: sbox[3,0,4] + rkey[0,0,3] - rkey[2,2,3] >= 0
 R1046: sbox[3,0,4] - rkey[0,0,3] + rkey[2,2,3] >= 0
 R1047: - sbox[0,0,4] + sbox[3,0,4] + rkey[3,1,3] >= 0
 R1048: sbox[0,0,4] + sbox[3,0,4] - rkey[3,1,3] >= 0
 R1049: sbox[0,0,4] - sbox[3,0,4] + rkey[3,1,3] >= 0
 R1050: - sbox[2,0,4] + rkey[1,3,3] + rkey[2,2,3] >= 0
 R1051: sbox[2,0,4] + rkey[1,3,3] - rkey[2,2,3] >= 0
 R1052: sbox[2,0,4] - rkey[1,3,3] + rkey[2,2,3] >= 0
 R1053: sbox[1,0,4] - rkey[0,0,3] = 0
 R1054: tbox[0,0,3] + tbox[2,2,3] - tbox[3,0,4] >= 0
 R1055: tbox[0,0,3] - tbox[2,2,3] + tbox[3,0,4] >= 0
 R1056: - tbox[0,0,3] + tbox[2,2,3] + tbox[3,0,4] >= 0
 R1057: - tbox[0,0,4] + tbox[3,0,4] + tbox[3,1,3] >= 0
 R1058: tbox[0,0,4] + tbox[3,0,4] - tbox[3,1,3] >= 0
 R1059: tbox[0,0,4] - tbox[3,0,4] + tbox[3,1,3] >= 0
 R1060: tbox[1,3,3] - tbox[2,0,4] + tbox[2,2,3] >= 0
 R1061: tbox[1,3,3] + tbox[2,0,4] - tbox[2,2,3] >= 0
 R1062: - tbox[1,3,3] + tbox[2,0,4] + tbox[2,2,3] >= 0
 R1063: - tbox[0,0,3] + tbox[1,0,4] = 0
 R1064: unkn[0,0,3] - unkn[3,0,4] + forg[3,0,4] <= 0
 R1065: unkn[2,2,3] - unkn[3,0,4] + forg[3,0,4] <= 0
 R1066: unkn[0,0,3] + unkn[2,2,3] - unkn[3,0,4] + forg[3,0,4] >= 0
 R1067: - unkn[0,0,4] + unkn[3,0,4] + forg[0,0,4] - forg[3,0,4] <= 0
 R1068: - unkn[0,0,4] + unkn[3,1,3] + forg[0,0,4] <= 0
 R1069: - unkn[0,0,4] + unkn[3,0,4] + unkn[3,1,3] + forg[0,0,4]
   - forg[3,0,4] >= 0
 R1070: unkn[1,3,3] - unkn[2,0,4] + forg[2,0,4] <= 0
 R1071: - unkn[2,0,4] + unkn[2,2,3] + forg[2,0,4] <= 0
 R1072: unkn[1,3,3] - unkn[2,0,4] + unkn[2,2,3] + forg[2,0,4] >= 0
 R1073: unkn[0,0,3] - unkn[1,0,4] + forg[1,0,4] = 0
 R1074: - tbox[0,1,3] + forg[0,1,3] <= 0
 R1075: sbox[0,1,3] - rkey[0,1,3] + tkey[0,1,3] >= 0
 R1076: sbox[0,1,3] + rkey[0,1,3] - tkey[0,1,3] >= 0
 R1077: - sbox[0,1,3] + rkey[0,1,3] + tkey[0,1,3] >= 0
 R1078: - tbox[1,1,3] + forg[1,1,3] <= 0
 R1079: sbox[1,1,3] - rkey[1,1,3] + tkey[1,1,3] >= 0
 R1080: sbox[1,1,3] + rkey[1,1,3] - tkey[1,1,3] >= 0
 R1081: - sbox[1,1,3] + rkey[1,1,3] + tkey[1,1,3] >= 0
 R1082: - tbox[2,1,3] + forg[2,1,3] <= 0
 R1083: - sbox[2,1,3] + rkey[2,1,3] = 0
 R1084: - tbox[3,1,3] + forg[3,1,3] <= 0
 R1085: - sbox[3,1,3] + rkey[3,1,3] = 0
 R1086: - sbox[3,1,4] + rkey[0,1,3] + rkey[2,3,3] >= 0
 R1087: sbox[3,1,4] + rkey[0,1,3] - rkey[2,3,3] >= 0
 R1088: sbox[3,1,4] - rkey[0,1,3] + rkey[2,3,3] >= 0
 R1089: - sbox[0,1,4] + sbox[3,1,4] + rkey[3,2,3] >= 0
 R1090: sbox[0,1,4] + sbox[3,1,4] - rkey[3,2,3] >= 0
 R1091: sbox[0,1,4] - sbox[3,1,4] + rkey[3,2,3] >= 0
 R1092: - sbox[2,1,4] + rkey[1,0,3] + rkey[2,3,3] >= 0
 R1093: sbox[2,1,4] + rkey[1,0,3] - rkey[2,3,3] >= 0
 R1094: sbox[2,1,4] - rkey[1,0,3] + rkey[2,3,3] >= 0
 R1095: sbox[1,1,4] - rkey[0,1,3] = 0
 R1096: tbox[0,1,3] + tbox[2,3,3] - tbox[3,1,4] >= 0
 R1097: tbox[0,1,3] - tbox[2,3,3] + tbox[3,1,4] >= 0
 R1098: - tbox[0,1,3] + tbox[2,3,3] + tbox[3,1,4] >= 0
 R1099: - tbox[0,1,4] + tbox[3,1,4] + tbox[3,2,3] >= 0
 R1100: tbox[0,1,4] + tbox[3,1,4] - tbox[3,2,3] >= 0
 R1101: tbox[0,1,4] - tbox[3,1,4] + tbox[3,2,3] >= 0
 R1102: tbox[1,0,3] - tbox[2,1,4] + tbox[2,3,3] >= 0
 R1103: tbox[1,0,3] + tbox[2,1,4] - tbox[2,3,3] >= 0
 R1104: - tbox[1,0,3] + tbox[2,1,4] + tbox[2,3,3] >= 0
 R1105: - tbox[0,1,3] + tbox[1,1,4] = 0
 R1106: unkn[0,1,3] - unkn[3,1,4] + forg[3,1,4] <= 0
 R1107: unkn[2,3,3] - unkn[3,1,4] + forg[3,1,4] <= 0
 R1108: unkn[0,1,3] + unkn[2,3,3] - unkn[3,1,4] + forg[3,1,4] >= 0
 R1109: - unkn[0,1,4] + unkn[3,1,4] + forg[0,1,4] - forg[3,1,4] <= 0
 R1110: - unkn[0,1,4] + unkn[3,2,3] + forg[0,1,4] <= 0
 R1111: - unkn[0,1,4] + unkn[3,1,4] + unkn[3,2,3] + forg[0,1,4]
   - forg[3,1,4] >= 0
 R1112: unkn[1,0,3] - unkn[2,1,4] + forg[2,1,4] <= 0
 R1113: - unkn[2,1,4] + unkn[2,3,3] + forg[2,1,4] <= 0
 R1114: unkn[1,0,3] - unkn[2,1,4] + unkn[2,3,3] + forg[2,1,4] >= 0
 R1115: unkn[0,1,3] - unkn[1,1,4] + forg[1,1,4] = 0
 R1116: - tbox[0,2,3] + forg[0,2,3] <= 0
 R1117: sbox[0,2,3] - rkey[0,2,3] + tkey[0,2,3] >= 0
 R1118: sbox[0,2,3] + rkey[0,2,3] - tkey[0,2,3] >= 0
 R1119: - sbox[0,2,3] + rkey[0,2,3] + tkey[0,2,3] >= 0
 R1120: - tbox[1,2,3] + forg[1,2,3] <= 0
 R1121: sbox[1,2,3] - rkey[1,2,3] + tkey[1,2,3] >= 0
 R1122: sbox[1,2,3] + rkey[1,2,3] - tkey[1,2,3] >= 0
 R1123: - sbox[1,2,3] + rkey[1,2,3] + tkey[1,2,3] >= 0
 R1124: - tbox[2,2,3] + forg[2,2,3] <= 0
 R1125: - sbox[2,2,3] + rkey[2,2,3] = 0
 R1126: - tbox[3,2,3] + forg[3,2,3] <= 0
 R1127: - sbox[3,2,3] + rkey[3,2,3] = 0
 R1128: - sbox[3,2,4] + rkey[0,2,3] + rkey[2,0,3] >= 0
 R1129: sbox[3,2,4] + rkey[0,2,3] - rkey[2,0,3] >= 0
 R1130: sbox[3,2,4] - rkey[0,2,3] + rkey[2,0,3] >= 0
 R1131: - sbox[0,2,4] + sbox[3,2,4] + rkey[3,3,3] >= 0
 R1132: sbox[0,2,4] + sbox[3,2,4] - rkey[3,3,3] >= 0
 R1133: sbox[0,2,4] - sbox[3,2,4] + rkey[3,3,3] >= 0
 R1134: - sbox[2,2,4] + rkey[1,1,3] + rkey[2,0,3] >= 0
 R1135: sbox[2,2,4] + rkey[1,1,3] - rkey[2,0,3] >= 0
 R1136: sbox[2,2,4] - rkey[1,1,3] + rkey[2,0,3] >= 0
 R1137: sbox[1,2,4] - rkey[0,2,3] = 0
 R1138: tbox[0,2,3] + tbox[2,0,3] - tbox[3,2,4] >= 0
 R1139: tbox[0,2,3] - tbox[2,0,3] + tbox[3,2,4] >= 0
 R1140: - tbox[0,2,3] + tbox[2,0,3] + tbox[3,2,4] >= 0
 R1141: - tbox[0,2,4] + tbox[3,2,4] + tbox[3,3,3] >= 0
 R1142: tbox[0,2,4] + tbox[3,2,4] - tbox[3,3,3] >= 0
 R1143: tbox[0,2,4] - tbox[3,2,4] + tbox[3,3,3] >= 0
 R1144: tbox[1,1,3] + tbox[2,0,3] - tbox[2,2,4] >= 0
 R1145: tbox[1,1,3] - tbox[2,0,3] + tbox[2,2,4] >= 0
 R1146: - tbox[1,1,3] + tbox[2,0,3] + tbox[2,2,4] >= 0
 R1147: - tbox[0,2,3] + tbox[1,2,4] = 0
 R1148: unkn[0,2,3] - unkn[3,2,4] + forg[3,2,4] <= 0
 R1149: unkn[2,0,3] - unkn[3,2,4] + forg[3,2,4] <= 0
 R1150: unkn[0,2,3] + unkn[2,0,3] - unkn[3,2,4] + forg[3,2,4] >= 0
 R1151: - unkn[0,2,4] + unkn[3,2,4] + forg[0,2,4] - forg[3,2,4] <= 0
 R1152: - unkn[0,2,4] + unkn[3,3,3] + forg[0,2,4] <= 0
 R1153: - unkn[0,2,4] + unkn[3,2,4] + unkn[3,3,3] + forg[0,2,4]
   - forg[3,2,4] >= 0
 R1154: unkn[1,1,3] - unkn[2,2,4] + forg[2,2,4] <= 0
 R1155: unkn[2,0,3] - unkn[2,2,4] + forg[2,2,4] <= 0
 R1156: unkn[1,1,3] + unkn[2,0,3] - unkn[2,2,4] + forg[2,2,4] >= 0
 R1157: unkn[0,2,3] - unkn[1,2,4] + forg[1,2,4] = 0
 R1158: - tbox[0,3,3] + forg[0,3,3] <= 0
 R1159: sbox[0,3,3] - rkey[0,3,3] + tkey[0,3,3] >= 0
 R1160: sbox[0,3,3] + rkey[0,3,3] - tkey[0,3,3] >= 0
 R1161: - sbox[0,3,3] + rkey[0,3,3] + tkey[0,3,3] >= 0
 R1162: - tbox[1,3,3] + forg[1,3,3] <= 0
 R1163: sbox[1,3,3] - rkey[1,3,3] + tkey[1,3,3] >= 0
 R1164: sbox[1,3,3] + rkey[1,3,3] - tkey[1,3,3] >= 0
 R1165: - sbox[1,3,3] + rkey[1,3,3] + tkey[1,3,3] >= 0
 R1166: - tbox[2,3,3] + forg[2,3,3] <= 0
 R1167: - sbox[2,3,3] + rkey[2,3,3] = 0
 R1168: - tbox[3,3,3] + forg[3,3,3] <= 0
 R1169: - sbox[3,3,3] + rkey[3,3,3] = 0
 R1170: - sbox[3,3,4] + rkey[0,3,3] + rkey[2,1,3] >= 0
 R1171: sbox[3,3,4] + rkey[0,3,3] - rkey[2,1,3] >= 0
 R1172: sbox[3,3,4] - rkey[0,3,3] + rkey[2,1,3] >= 0
 R1173: - sbox[0,3,4] + sbox[3,3,4] + rkey[3,0,3] >= 0
 R1174: sbox[0,3,4] + sbox[3,3,4] - rkey[3,0,3] >= 0
 R1175: sbox[0,3,4] - sbox[3,3,4] + rkey[3,0,3] >= 0
 R1176: - sbox[2,3,4] + rkey[1,2,3] + rkey[2,1,3] >= 0
 R1177: sbox[2,3,4] + rkey[1,2,3] - rkey[2,1,3] >= 0
 R1178: sbox[2,3,4] - rkey[1,2,3] + rkey[2,1,3] >= 0
 R1179: sbox[1,3,4] - rkey[0,3,3] = 0
 R1180: tbox[0,3,3] + tbox[2,1,3] - tbox[3,3,4] >= 0
 R1181: tbox[0,3,3] - tbox[2,1,3] + tbox[3,3,4] >= 0
 R1182: - tbox[0,3,3] + tbox[2,1,3] + tbox[3,3,4] >= 0
 R1183: - tbox[0,3,4] + tbox[3,0,3] + tbox[3,3,4] >= 0
 R1184: tbox[0,3,4] - tbox[3,0,3] + tbox[3,3,4] >= 0
 R1185: tbox[0,3,4] + tbox[3,0,3] - tbox[3,3,4] >= 0
 R1186: tbox[1,2,3] + tbox[2,1,3] - tbox[2,3,4] >= 0
 R1187: tbox[1,2,3] - tbox[2,1,3] + tbox[2,3,4] >= 0
 R1188: - tbox[1,2,3] + tbox[2,1,3] + tbox[2,3,4] >= 0
 R1189: - tbox[0,3,3] + tbox[1,3,4] = 0
 R1190: unkn[0,3,3] - unkn[3,3,4] + forg[3,3,4] <= 0
 R1191: unkn[2,1,3] - unkn[3,3,4] + forg[3,3,4] <= 0
 R1192: unkn[0,3,3] + unkn[2,1,3] - unkn[3,3,4] + forg[3,3,4] >= 0
 R1193: - unkn[0,3,4] + unkn[3,3,4] + forg[0,3,4] - forg[3,3,4] <= 0
 R1194: - unkn[0,3,4] + unkn[3,0,3] + forg[0,3,4] <= 0
 R1195: - unkn[0,3,4] + unkn[3,0,3] + unkn[3,3,4] + forg[0,3,4]
   - forg[3,3,4] >= 0
 R1196: unkn[1,2,3] - unkn[2,3,4] + forg[2,3,4] <= 0
 R1197: unkn[2,1,3] - unkn[2,3,4] + forg[2,3,4] <= 0
 R1198: unkn[1,2,3] + unkn[2,1,3] - unkn[2,3,4] + forg[2,3,4] >= 0
 R1199: unkn[0,3,3] - unkn[1,3,4] + forg[1,3,4] = 0
 R1200: - tbox[0,0,4] + forg[0,0,4] <= 0
 R1201: sbox[0,0,4] - rkey[0,0,4] + tkey[0,0,4] >= 0
 R1202: sbox[0,0,4] + rkey[0,0,4] - tkey[0,0,4] >= 0
 R1203: - sbox[0,0,4] + rkey[0,0,4] + tkey[0,0,4] >= 0
 R1204: - tbox[1,0,4] + forg[1,0,4] <= 0
 R1205: sbox[1,0,4] - rkey[1,0,4] + tkey[1,0,4] >= 0
 R1206: sbox[1,0,4] + rkey[1,0,4] - tkey[1,0,4] >= 0
 R1207: - sbox[1,0,4] + rkey[1,0,4] + tkey[1,0,4] >= 0
 R1208: - tbox[2,0,4] + forg[2,0,4] <= 0
 R1209: - sbox[2,0,4] + rkey[2,0,4] = 0
 R1210: - tbox[3,0,4] + forg[3,0,4] <= 0
 R1211: - sbox[3,0,4] + rkey[3,0,4] = 0
 R1212: - sbox[3,0,5] + rkey[0,0,4] + rkey[2,2,4] >= 0
 R1213: sbox[3,0,5] + rkey[0,0,4] - rkey[2,2,4] >= 0
 R1214: sbox[3,0,5] - rkey[0,0,4] + rkey[2,2,4] >= 0
 R1215: - sbox[0,0,5] + sbox[3,0,5] + rkey[3,1,4] >= 0
 R1216: sbox[0,0,5] + sbox[3,0,5] - rkey[3,1,4] >= 0
 R1217: sbox[0,0,5] - sbox[3,0,5] + rkey[3,1,4] >= 0
 R1218: - sbox[2,0,5] + rkey[1,3,4] + rkey[2,2,4] >= 0
 R1219: sbox[2,0,5] + rkey[1,3,4] - rkey[2,2,4] >= 0
 R1220: sbox[2,0,5] - rkey[1,3,4] + rkey[2,2,4] >= 0
 R1221: sbox[1,0,5] - rkey[0,0,4] = 0
 R1222: tbox[0,0,4] + tbox[2,2,4] - tbox[3,0,5] >= 0
 R1223: tbox[0,0,4] - tbox[2,2,4] + tbox[3,0,5] >= 0
 R1224: - tbox[0,0,4] + tbox[2,2,4] + tbox[3,0,5] >= 0
 R1225: - tbox[0,0,5] + tbox[3,0,5] + tbox[3,1,4] >= 0
 R1226: tbox[0,0,5] + tbox[3,0,5] - tbox[3,1,4] >= 0
 R1227: tbox[0,0,5] - tbox[3,0,5] + tbox[3,1,4] >= 0
 R1228: tbox[1,3,4] - tbox[2,0,5] + tbox[2,2,4] >= 0
 R1229: tbox[1,3,4] + tbox[2,0,5] - tbox[2,2,4] >= 0
 R1230: - tbox[1,3,4] + tbox[2,0,5] + tbox[2,2,4] >= 0
 R1231: - tbox[0,0,4] + tbox[1,0,5] = 0
 R1232: unkn[0,0,4] - unkn[3,0,5] + forg[3,0,5] <= 0
 R1233: unkn[2,2,4] - unkn[3,0,5] + forg[3,0,5] <= 0
 R1234: unkn[0,0,4] + unkn[2,2,4] - unkn[3,0,5] + forg[3,0,5] >= 0
 R1235: - unkn[0,0,5] + unkn[3,0,5] + forg[0,0,5] - forg[3,0,5] <= 0
 R1236: - unkn[0,0,5] + unkn[3,1,4] + forg[0,0,5] <= 0
 R1237: - unkn[0,0,5] + unkn[3,0,5] + unkn[3,1,4] + forg[0,0,5]
   - forg[3,0,5] >= 0
 R1238: unkn[1,3,4] - unkn[2,0,5] + forg[2,0,5] <= 0
 R1239: - unkn[2,0,5] + unkn[2,2,4] + forg[2,0,5] <= 0
 R1240: unkn[1,3,4] - unkn[2,0,5] + unkn[2,2,4] + forg[2,0,5] >= 0
 R1241: unkn[0,0,4] - unkn[1,0,5] + forg[1,0,5] = 0
 R1242: - tbox[0,1,4] + forg[0,1,4] <= 0
 R1243: sbox[0,1,4] - rkey[0,1,4] + tkey[0,1,4] >= 0
 R1244: sbox[0,1,4] + rkey[0,1,4] - tkey[0,1,4] >= 0
 R1245: - sbox[0,1,4] + rkey[0,1,4] + tkey[0,1,4] >= 0
 R1246: - tbox[1,1,4] + forg[1,1,4] <= 0
 R1247: sbox[1,1,4] - rkey[1,1,4] + tkey[1,1,4] >= 0
 R1248: sbox[1,1,4] + rkey[1,1,4] - tkey[1,1,4] >= 0
 R1249: - sbox[1,1,4] + rkey[1,1,4] + tkey[1,1,4] >= 0
 R1250: - tbox[2,1,4] + forg[2,1,4] <= 0
 R1251: - sbox[2,1,4] + rkey[2,1,4] = 0
 R1252: - tbox[3,1,4] + forg[3,1,4] <= 0
 R1253: - sbox[3,1,4] + rkey[3,1,4] = 0
 R1254: - sbox[3,1,5] + rkey[0,1,4] + rkey[2,3,4] >= 0
 R1255: sbox[3,1,5] + rkey[0,1,4] - rkey[2,3,4] >= 0
 R1256: sbox[3,1,5] - rkey[0,1,4] + rkey[2,3,4] >= 0
 R1257: - sbox[0,1,5] + sbox[3,1,5] + rkey[3,2,4] >= 0
 R1258: sbox[0,1,5] + sbox[3,1,5] - rkey[3,2,4] >= 0
 R1259: sbox[0,1,5] - sbox[3,1,5] + rkey[3,2,4] >= 0
 R1260: - sbox[2,1,5] + rkey[1,0,4] + rkey[2,3,4] >= 0
 R1261: sbox[2,1,5] + rkey[1,0,4] - rkey[2,3,4] >= 0
 R1262: sbox[2,1,5] - rkey[1,0,4] + rkey[2,3,4] >= 0
 R1263: sbox[1,1,5] - rkey[0,1,4] = 0
 R1264: tbox[0,1,4] + tbox[2,3,4] - tbox[3,1,5] >= 0
 R1265: tbox[0,1,4] - tbox[2,3,4] + tbox[3,1,5] >= 0
 R1266: - tbox[0,1,4] + tbox[2,3,4] + tbox[3,1,5] >= 0
 R1267: - tbox[0,1,5] + tbox[3,1,5] + tbox[3,2,4] >= 0
 R1268: tbox[0,1,5] + tbox[3,1,5] - tbox[3,2,4] >= 0
 R1269: tbox[0,1,5] - tbox[3,1,5] + tbox[3,2,4] >= 0
 R1270: tbox[1,0,4] - tbox[2,1,5] + tbox[2,3,4] >= 0
 R1271: tbox[1,0,4] + tbox[2,1,5] - tbox[2,3,4] >= 0
 R1272: - tbox[1,0,4] + tbox[2,1,5] + tbox[2,3,4] >= 0
 R1273: - tbox[0,1,4] + tbox[1,1,5] = 0
 R1274: unkn[0,1,4] - unkn[3,1,5] + forg[3,1,5] <= 0
 R1275: unkn[2,3,4] - unkn[3,1,5] + forg[3,1,5] <= 0
 R1276: unkn[0,1,4] + unkn[2,3,4] - unkn[3,1,5] + forg[3,1,5] >= 0
 R1277: - unkn[0,1,5] + unkn[3,1,5] + forg[0,1,5] - forg[3,1,5] <= 0
 R1278: - unkn[0,1,5] + unkn[3,2,4] + forg[0,1,5] <= 0
 R1279: - unkn[0,1,5] + unkn[3,1,5] + unkn[3,2,4] + forg[0,1,5]
   - forg[3,1,5] >= 0
 R1280: unkn[1,0,4] - unkn[2,1,5] + forg[2,1,5] <= 0
 R1281: - unkn[2,1,5] + unkn[2,3,4] + forg[2,1,5] <= 0
 R1282: unkn[1,0,4] - unkn[2,1,5] + unkn[2,3,4] + forg[2,1,5] >= 0
 R1283: unkn[0,1,4] - unkn[1,1,5] + forg[1,1,5] = 0
 R1284: - tbox[0,2,4] + forg[0,2,4] <= 0
 R1285: sbox[0,2,4] - rkey[0,2,4] + tkey[0,2,4] >= 0
 R1286: sbox[0,2,4] + rkey[0,2,4] - tkey[0,2,4] >= 0
 R1287: - sbox[0,2,4] + rkey[0,2,4] + tkey[0,2,4] >= 0
 R1288: - tbox[1,2,4] + forg[1,2,4] <= 0
 R1289: sbox[1,2,4] - rkey[1,2,4] + tkey[1,2,4] >= 0
 R1290: sbox[1,2,4] + rkey[1,2,4] - tkey[1,2,4] >= 0
 R1291: - sbox[1,2,4] + rkey[1,2,4] + tkey[1,2,4] >= 0
 R1292: - tbox[2,2,4] + forg[2,2,4] <= 0
 R1293: - sbox[2,2,4] + rkey[2,2,4] = 0
 R1294: - tbox[3,2,4] + forg[3,2,4] <= 0
 R1295: - sbox[3,2,4] + rkey[3,2,4] = 0
 R1296: - sbox[3,2,5] + rkey[0,2,4] + rkey[2,0,4] >= 0
 R1297: sbox[3,2,5] + rkey[0,2,4] - rkey[2,0,4] >= 0
 R1298: sbox[3,2,5] - rkey[0,2,4] + rkey[2,0,4] >= 0
 R1299: - sbox[0,2,5] + sbox[3,2,5] + rkey[3,3,4] >= 0
 R1300: sbox[0,2,5] + sbox[3,2,5] - rkey[3,3,4] >= 0
 R1301: sbox[0,2,5] - sbox[3,2,5] + rkey[3,3,4] >= 0
 R1302: - sbox[2,2,5] + rkey[1,1,4] + rkey[2,0,4] >= 0
 R1303: sbox[2,2,5] + rkey[1,1,4] - rkey[2,0,4] >= 0
 R1304: sbox[2,2,5] - rkey[1,1,4] + rkey[2,0,4] >= 0
 R1305: sbox[1,2,5] - rkey[0,2,4] = 0
 R1306: tbox[0,2,4] + tbox[2,0,4] - tbox[3,2,5] >= 0
 R1307: tbox[0,2,4] - tbox[2,0,4] + tbox[3,2,5] >= 0
 R1308: - tbox[0,2,4] + tbox[2,0,4] + tbox[3,2,5] >= 0
 R1309: - tbox[0,2,5] + tbox[3,2,5] + tbox[3,3,4] >= 0
 R1310: tbox[0,2,5] + tbox[3,2,5] - tbox[3,3,4] >= 0
 R1311: tbox[0,2,5] - tbox[3,2,5] + tbox[3,3,4] >= 0
 R1312: tbox[1,1,4] + tbox[2,0,4] - tbox[2,2,5] >= 0
 R1313: tbox[1,1,4] - tbox[2,0,4] + tbox[2,2,5] >= 0
 R1314: - tbox[1,1,4] + tbox[2,0,4] + tbox[2,2,5] >= 0
 R1315: - tbox[0,2,4] + tbox[1,2,5] = 0
 R1316: unkn[0,2,4] - unkn[3,2,5] + forg[3,2,5] <= 0
 R1317: unkn[2,0,4] - unkn[3,2,5] + forg[3,2,5] <= 0
 R1318: unkn[0,2,4] + unkn[2,0,4] - unkn[3,2,5] + forg[3,2,5] >= 0
 R1319: - unkn[0,2,5] + unkn[3,2,5] + forg[0,2,5] - forg[3,2,5] <= 0
 R1320: - unkn[0,2,5] + unkn[3,3,4] + forg[0,2,5] <= 0
 R1321: - unkn[0,2,5] + unkn[3,2,5] + unkn[3,3,4] + forg[0,2,5]
   - forg[3,2,5] >= 0
 R1322: unkn[1,1,4] - unkn[2,2,5] + forg[2,2,5] <= 0
 R1323: unkn[2,0,4] - unkn[2,2,5] + forg[2,2,5] <= 0
 R1324: unkn[1,1,4] + unkn[2,0,4] - unkn[2,2,5] + forg[2,2,5] >= 0
 R1325: unkn[0,2,4] - unkn[1,2,5] + forg[1,2,5] = 0
 R1326: - tbox[0,3,4] + forg[0,3,4] <= 0
 R1327: sbox[0,3,4] - rkey[0,3,4] + tkey[0,3,4] >= 0
 R1328: sbox[0,3,4] + rkey[0,3,4] - tkey[0,3,4] >= 0
 R1329: - sbox[0,3,4] + rkey[0,3,4] + tkey[0,3,4] >= 0
 R1330: - tbox[1,3,4] + forg[1,3,4] <= 0
 R1331: sbox[1,3,4] - rkey[1,3,4] + tkey[1,3,4] >= 0
 R1332: sbox[1,3,4] + rkey[1,3,4] - tkey[1,3,4] >= 0
 R1333: - sbox[1,3,4] + rkey[1,3,4] + tkey[1,3,4] >= 0
 R1334: - tbox[2,3,4] + forg[2,3,4] <= 0
 R1335: - sbox[2,3,4] + rkey[2,3,4] = 0
 R1336: - tbox[3,3,4] + forg[3,3,4] <= 0
 R1337: - sbox[3,3,4] + rkey[3,3,4] = 0
 R1338: - sbox[3,3,5] + rkey[0,3,4] + rkey[2,1,4] >= 0
 R1339: sbox[3,3,5] + rkey[0,3,4] - rkey[2,1,4] >= 0
 R1340: sbox[3,3,5] - rkey[0,3,4] + rkey[2,1,4] >= 0
 R1341: - sbox[0,3,5] + sbox[3,3,5] + rkey[3,0,4] >= 0
 R1342: sbox[0,3,5] + sbox[3,3,5] - rkey[3,0,4] >= 0
 R1343: sbox[0,3,5] - sbox[3,3,5] + rkey[3,0,4] >= 0
 R1344: - sbox[2,3,5] + rkey[1,2,4] + rkey[2,1,4] >= 0
 R1345: sbox[2,3,5] + rkey[1,2,4] - rkey[2,1,4] >= 0
 R1346: sbox[2,3,5] - rkey[1,2,4] + rkey[2,1,4] >= 0
 R1347: sbox[1,3,5] - rkey[0,3,4] = 0
 R1348: tbox[0,3,4] + tbox[2,1,4] - tbox[3,3,5] >= 0
 R1349: tbox[0,3,4] - tbox[2,1,4] + tbox[3,3,5] >= 0
 R1350: - tbox[0,3,4] + tbox[2,1,4] + tbox[3,3,5] >= 0
 R1351: - tbox[0,3,5] + tbox[3,0,4] + tbox[3,3,5] >= 0
 R1352: tbox[0,3,5] - tbox[3,0,4] + tbox[3,3,5] >= 0
 R1353: tbox[0,3,5] + tbox[3,0,4] - tbox[3,3,5] >= 0
 R1354: tbox[1,2,4] + tbox[2,1,4] - tbox[2,3,5] >= 0
 R1355: tbox[1,2,4] - tbox[2,1,4] + tbox[2,3,5] >= 0
 R1356: - tbox[1,2,4] + tbox[2,1,4] + tbox[2,3,5] >= 0
 R1357: - tbox[0,3,4] + tbox[1,3,5] = 0
 R1358: unkn[0,3,4] - unkn[3,3,5] + forg[3,3,5] <= 0
 R1359: unkn[2,1,4] - unkn[3,3,5] + forg[3,3,5] <= 0
 R1360: unkn[0,3,4] + unkn[2,1,4] - unkn[3,3,5] + forg[3,3,5] >= 0
 R1361: - unkn[0,3,5] + unkn[3,3,5] + forg[0,3,5] - forg[3,3,5] <= 0
 R1362: - unkn[0,3,5] + unkn[3,0,4] + forg[0,3,5] <= 0
 R1363: - unkn[0,3,5] + unkn[3,0,4] + unkn[3,3,5] + forg[0,3,5]
   - forg[3,3,5] >= 0
 R1364: unkn[1,2,4] - unkn[2,3,5] + forg[2,3,5] <= 0
 R1365: unkn[2,1,4] - unkn[2,3,5] + forg[2,3,5] <= 0
 R1366: unkn[1,2,4] + unkn[2,1,4] - unkn[2,3,5] + forg[2,3,5] >= 0
 R1367: unkn[0,3,4] - unkn[1,3,5] + forg[1,3,5] = 0
 R1368: - tbox[0,0,5] + forg[0,0,5] <= 0
 R1369: sbox[0,0,5] - rkey[0,0,5] + tkey[0,0,5] >= 0
 R1370: sbox[0,0,5] + rkey[0,0,5] - tkey[0,0,5] >= 0
 R1371: - sbox[0,0,5] + rkey[0,0,5] + tkey[0,0,5] >= 0
 R1372: - tbox[1,0,5] + forg[1,0,5] <= 0
 R1373: sbox[1,0,5] - rkey[1,0,5] + tkey[1,0,5] >= 0
 R1374: sbox[1,0,5] + rkey[1,0,5] - tkey[1,0,5] >= 0
 R1375: - sbox[1,0,5] + rkey[1,0,5] + tkey[1,0,5] >= 0
 R1376: - tbox[2,0,5] + forg[2,0,5] <= 0
 R1377: - sbox[2,0,5] + rkey[2,0,5] = 0
 R1378: - tbox[3,0,5] + forg[3,0,5] <= 0
 R1379: - sbox[3,0,5] + rkey[3,0,5] = 0
 R1380: - sbox[3,0,6] + rkey[0,0,5] + rkey[2,2,5] >= 0
 R1381: sbox[3,0,6] + rkey[0,0,5] - rkey[2,2,5] >= 0
 R1382: sbox[3,0,6] - rkey[0,0,5] + rkey[2,2,5] >= 0
 R1383: - sbox[0,0,6] + sbox[3,0,6] + rkey[3,1,5] >= 0
 R1384: sbox[0,0,6] + sbox[3,0,6] - rkey[3,1,5] >= 0
 R1385: sbox[0,0,6] - sbox[3,0,6] + rkey[3,1,5] >= 0
 R1386: - sbox[2,0,6] + rkey[1,3,5] + rkey[2,2,5] >= 0
 R1387: sbox[2,0,6] + rkey[1,3,5] - rkey[2,2,5] >= 0
 R1388: sbox[2,0,6] - rkey[1,3,5] + rkey[2,2,5] >= 0
 R1389: sbox[1,0,6] - rkey[0,0,5] = 0
 R1390: tbox[0,0,5] + tbox[2,2,5] - tbox[3,0,6] >= 0
 R1391: tbox[0,0,5] - tbox[2,2,5] + tbox[3,0,6] >= 0
 R1392: - tbox[0,0,5] + tbox[2,2,5] + tbox[3,0,6] >= 0
 R1393: - tbox[0,0,6] + tbox[3,0,6] + tbox[3,1,5] >= 0
 R1394: tbox[0,0,6] + tbox[3,0,6] - tbox[3,1,5] >= 0
 R1395: tbox[0,0,6] - tbox[3,0,6] + tbox[3,1,5] >= 0
 R1396: tbox[1,3,5] - tbox[2,0,6] + tbox[2,2,5] >= 0
 R1397: tbox[1,3,5] + tbox[2,0,6] - tbox[2,2,5] >= 0
 R1398: - tbox[1,3,5] + tbox[2,0,6] + tbox[2,2,5] >= 0
 R1399: - tbox[0,0,5] + tbox[1,0,6] = 0
 R1400: unkn[0,0,5] - unkn[3,0,6] + forg[3,0,6] <= 0
 R1401: unkn[2,2,5] - unkn[3,0,6] + forg[3,0,6] <= 0
 R1402: unkn[0,0,5] + unkn[2,2,5] - unkn[3,0,6] + forg[3,0,6] >= 0
 R1403: - unkn[0,0,6] + unkn[3,0,6] + forg[0,0,6] - forg[3,0,6] <= 0
 R1404: - unkn[0,0,6] + unkn[3,1,5] + forg[0,0,6] <= 0
 R1405: - unkn[0,0,6] + unkn[3,0,6] + unkn[3,1,5] + forg[0,0,6]
   - forg[3,0,6] >= 0
 R1406: unkn[1,3,5] - unkn[2,0,6] + forg[2,0,6] <= 0
 R1407: - unkn[2,0,6] + unkn[2,2,5] + forg[2,0,6] <= 0
 R1408: unkn[1,3,5] - unkn[2,0,6] + unkn[2,2,5] + forg[2,0,6] >= 0
 R1409: unkn[0,0,5] - unkn[1,0,6] + forg[1,0,6] = 0
 R1410: - tbox[0,1,5] + forg[0,1,5] <= 0
 R1411: sbox[0,1,5] - rkey[0,1,5] + tkey[0,1,5] >= 0
 R1412: sbox[0,1,5] + rkey[0,1,5] - tkey[0,1,5] >= 0
 R1413: - sbox[0,1,5] + rkey[0,1,5] + tkey[0,1,5] >= 0
 R1414: - tbox[1,1,5] + forg[1,1,5] <= 0
 R1415: sbox[1,1,5] - rkey[1,1,5] + tkey[1,1,5] >= 0
 R1416: sbox[1,1,5] + rkey[1,1,5] - tkey[1,1,5] >= 0
 R1417: - sbox[1,1,5] + rkey[1,1,5] + tkey[1,1,5] >= 0
 R1418: - tbox[2,1,5] + forg[2,1,5] <= 0
 R1419: - sbox[2,1,5] + rkey[2,1,5] = 0
 R1420: - tbox[3,1,5] + forg[3,1,5] <= 0
 R1421: - sbox[3,1,5] + rkey[3,1,5] = 0
 R1422: - sbox[3,1,6] + rkey[0,1,5] + rkey[2,3,5] >= 0
 R1423: sbox[3,1,6] + rkey[0,1,5] - rkey[2,3,5] >= 0
 R1424: sbox[3,1,6] - rkey[0,1,5] + rkey[2,3,5] >= 0
 R1425: - sbox[0,1,6] + sbox[3,1,6] + rkey[3,2,5] >= 0
 R1426: sbox[0,1,6] + sbox[3,1,6] - rkey[3,2,5] >= 0
 R1427: sbox[0,1,6] - sbox[3,1,6] + rkey[3,2,5] >= 0
 R1428: - sbox[2,1,6] + rkey[1,0,5] + rkey[2,3,5] >= 0
 R1429: sbox[2,1,6] + rkey[1,0,5] - rkey[2,3,5] >= 0
 R1430: sbox[2,1,6] - rkey[1,0,5] + rkey[2,3,5] >= 0
 R1431: sbox[1,1,6] - rkey[0,1,5] = 0
 R1432: tbox[0,1,5] + tbox[2,3,5] - tbox[3,1,6] >= 0
 R1433: tbox[0,1,5] - tbox[2,3,5] + tbox[3,1,6] >= 0
 R1434: - tbox[0,1,5] + tbox[2,3,5] + tbox[3,1,6] >= 0
 R1435: - tbox[0,1,6] + tbox[3,1,6] + tbox[3,2,5] >= 0
 R1436: tbox[0,1,6] + tbox[3,1,6] - tbox[3,2,5] >= 0
 R1437: tbox[0,1,6] - tbox[3,1,6] + tbox[3,2,5] >= 0
 R1438: tbox[1,0,5] - tbox[2,1,6] + tbox[2,3,5] >= 0
 R1439: tbox[1,0,5] + tbox[2,1,6] - tbox[2,3,5] >= 0
 R1440: - tbox[1,0,5] + tbox[2,1,6] + tbox[2,3,5] >= 0
 R1441: - tbox[0,1,5] + tbox[1,1,6] = 0
 R1442: unkn[0,1,5] - unkn[3,1,6] + forg[3,1,6] <= 0
 R1443: unkn[2,3,5] - unkn[3,1,6] + forg[3,1,6] <= 0
 R1444: unkn[0,1,5] + unkn[2,3,5] - unkn[3,1,6] + forg[3,1,6] >= 0
 R1445: - unkn[0,1,6] + unkn[3,1,6] + forg[0,1,6] - forg[3,1,6] <= 0
 R1446: - unkn[0,1,6] + unkn[3,2,5] + forg[0,1,6] <= 0
 R1447: - unkn[0,1,6] + unkn[3,1,6] + unkn[3,2,5] + forg[0,1,6]
   - forg[3,1,6] >= 0
 R1448: unkn[1,0,5] - unkn[2,1,6] + forg[2,1,6] <= 0
 R1449: - unkn[2,1,6] + unkn[2,3,5] + forg[2,1,6] <= 0
 R1450: unkn[1,0,5] - unkn[2,1,6] + unkn[2,3,5] + forg[2,1,6] >= 0
 R1451: unkn[0,1,5] - unkn[1,1,6] + forg[1,1,6] = 0
 R1452: - tbox[0,2,5] + forg[0,2,5] <= 0
 R1453: sbox[0,2,5] - rkey[0,2,5] + tkey[0,2,5] >= 0
 R1454: sbox[0,2,5] + rkey[0,2,5] - tkey[0,2,5] >= 0
 R1455: - sbox[0,2,5] + rkey[0,2,5] + tkey[0,2,5] >= 0
 R1456: - tbox[1,2,5] + forg[1,2,5] <= 0
 R1457: sbox[1,2,5] - rkey[1,2,5] + tkey[1,2,5] >= 0
 R1458: sbox[1,2,5] + rkey[1,2,5] - tkey[1,2,5] >= 0
 R1459: - sbox[1,2,5] + rkey[1,2,5] + tkey[1,2,5] >= 0
 R1460: - tbox[2,2,5] + forg[2,2,5] <= 0
 R1461: - sbox[2,2,5] + rkey[2,2,5] = 0
 R1462: - tbox[3,2,5] + forg[3,2,5] <= 0
 R1463: - sbox[3,2,5] + rkey[3,2,5] = 0
 R1464: - sbox[3,2,6] + rkey[0,2,5] + rkey[2,0,5] >= 0
 R1465: sbox[3,2,6] + rkey[0,2,5] - rkey[2,0,5] >= 0
 R1466: sbox[3,2,6] - rkey[0,2,5] + rkey[2,0,5] >= 0
 R1467: - sbox[0,2,6] + sbox[3,2,6] + rkey[3,3,5] >= 0
 R1468: sbox[0,2,6] + sbox[3,2,6] - rkey[3,3,5] >= 0
 R1469: sbox[0,2,6] - sbox[3,2,6] + rkey[3,3,5] >= 0
 R1470: - sbox[2,2,6] + rkey[1,1,5] + rkey[2,0,5] >= 0
 R1471: sbox[2,2,6] + rkey[1,1,5] - rkey[2,0,5] >= 0
 R1472: sbox[2,2,6] - rkey[1,1,5] + rkey[2,0,5] >= 0
 R1473: sbox[1,2,6] - rkey[0,2,5] = 0
 R1474: tbox[0,2,5] + tbox[2,0,5] - tbox[3,2,6] >= 0
 R1475: tbox[0,2,5] - tbox[2,0,5] + tbox[3,2,6] >= 0
 R1476: - tbox[0,2,5] + tbox[2,0,5] + tbox[3,2,6] >= 0
 R1477: - tbox[0,2,6] + tbox[3,2,6] + tbox[3,3,5] >= 0
 R1478: tbox[0,2,6] + tbox[3,2,6] - tbox[3,3,5] >= 0
 R1479: tbox[0,2,6] - tbox[3,2,6] + tbox[3,3,5] >= 0
 R1480: tbox[1,1,5] + tbox[2,0,5] - tbox[2,2,6] >= 0
 R1481: tbox[1,1,5] - tbox[2,0,5] + tbox[2,2,6] >= 0
 R1482: - tbox[1,1,5] + tbox[2,0,5] + tbox[2,2,6] >= 0
 R1483: - tbox[0,2,5] + tbox[1,2,6] = 0
 R1484: unkn[0,2,5] - unkn[3,2,6] + forg[3,2,6] <= 0
 R1485: unkn[2,0,5] - unkn[3,2,6] + forg[3,2,6] <= 0
 R1486: unkn[0,2,5] + unkn[2,0,5] - unkn[3,2,6] + forg[3,2,6] >= 0
 R1487: - unkn[0,2,6] + unkn[3,2,6] + forg[0,2,6] - forg[3,2,6] <= 0
 R1488: - unkn[0,2,6] + unkn[3,3,5] + forg[0,2,6] <= 0
 R1489: - unkn[0,2,6] + unkn[3,2,6] + unkn[3,3,5] + forg[0,2,6]
   - forg[3,2,6] >= 0
 R1490: unkn[1,1,5] - unkn[2,2,6] + forg[2,2,6] <= 0
 R1491: unkn[2,0,5] - unkn[2,2,6] + forg[2,2,6] <= 0
 R1492: unkn[1,1,5] + unkn[2,0,5] - unkn[2,2,6] + forg[2,2,6] >= 0
 R1493: unkn[0,2,5] - unkn[1,2,6] + forg[1,2,6] = 0
 R1494: - tbox[0,3,5] + forg[0,3,5] <= 0
 R1495: sbox[0,3,5] - rkey[0,3,5] + tkey[0,3,5] >= 0
 R1496: sbox[0,3,5] + rkey[0,3,5] - tkey[0,3,5] >= 0
 R1497: - sbox[0,3,5] + rkey[0,3,5] + tkey[0,3,5] >= 0
 R1498: - tbox[1,3,5] + forg[1,3,5] <= 0
 R1499: sbox[1,3,5] - rkey[1,3,5] + tkey[1,3,5] >= 0
 R1500: sbox[1,3,5] + rkey[1,3,5] - tkey[1,3,5] >= 0
 R1501: - sbox[1,3,5] + rkey[1,3,5] + tkey[1,3,5] >= 0
 R1502: - tbox[2,3,5] + forg[2,3,5] <= 0
 R1503: - sbox[2,3,5] + rkey[2,3,5] = 0
 R1504: - tbox[3,3,5] + forg[3,3,5] <= 0
 R1505: - sbox[3,3,5] + rkey[3,3,5] = 0
 R1506: - sbox[3,3,6] + rkey[0,3,5] + rkey[2,1,5] >= 0
 R1507: sbox[3,3,6] + rkey[0,3,5] - rkey[2,1,5] >= 0
 R1508: sbox[3,3,6] - rkey[0,3,5] + rkey[2,1,5] >= 0
 R1509: - sbox[0,3,6] + sbox[3,3,6] + rkey[3,0,5] >= 0
 R1510: sbox[0,3,6] + sbox[3,3,6] - rkey[3,0,5] >= 0
 R1511: sbox[0,3,6] - sbox[3,3,6] + rkey[3,0,5] >= 0
 R1512: - sbox[2,3,6] + rkey[1,2,5] + rkey[2,1,5] >= 0
 R1513: sbox[2,3,6] + rkey[1,2,5] - rkey[2,1,5] >= 0
 R1514: sbox[2,3,6] - rkey[1,2,5] + rkey[2,1,5] >= 0
 R1515: sbox[1,3,6] - rkey[0,3,5] = 0
 R1516: tbox[0,3,5] + tbox[2,1,5] - tbox[3,3,6] >= 0
 R1517: tbox[0,3,5] - tbox[2,1,5] + tbox[3,3,6] >= 0
 R1518: - tbox[0,3,5] + tbox[2,1,5] + tbox[3,3,6] >= 0
 R1519: - tbox[0,3,6] + tbox[3,0,5] + tbox[3,3,6] >= 0
 R1520: tbox[0,3,6] - tbox[3,0,5] + tbox[3,3,6] >= 0
 R1521: tbox[0,3,6] + tbox[3,0,5] - tbox[3,3,6] >= 0
 R1522: tbox[1,2,5] + tbox[2,1,5] - tbox[2,3,6] >= 0
 R1523: tbox[1,2,5] - tbox[2,1,5] + tbox[2,3,6] >= 0
 R1524: - tbox[1,2,5] + tbox[2,1,5] + tbox[2,3,6] >= 0
 R1525: - tbox[0,3,5] + tbox[1,3,6] = 0
 R1526: unkn[0,3,5] - unkn[3,3,6] + forg[3,3,6] <= 0
 R1527: unkn[2,1,5] - unkn[3,3,6] + forg[3,3,6] <= 0
 R1528: unkn[0,3,5] + unkn[2,1,5] - unkn[3,3,6] + forg[3,3,6] >= 0
 R1529: - unkn[0,3,6] + unkn[3,3,6] + forg[0,3,6] - forg[3,3,6] <= 0
 R1530: - unkn[0,3,6] + unkn[3,0,5] + forg[0,3,6] <= 0
 R1531: - unkn[0,3,6] + unkn[3,0,5] + unkn[3,3,6] + forg[0,3,6]
   - forg[3,3,6] >= 0
 R1532: unkn[1,2,5] - unkn[2,3,6] + forg[2,3,6] <= 0
 R1533: unkn[2,1,5] - unkn[2,3,6] + forg[2,3,6] <= 0
 R1534: unkn[1,2,5] + unkn[2,1,5] - unkn[2,3,6] + forg[2,3,6] >= 0
 R1535: unkn[0,3,5] - unkn[1,3,6] + forg[1,3,6] = 0
 R1536: - tbox[0,0,6] + forg[0,0,6] <= 0
 R1537: sbox[0,0,6] - rkey[0,0,6] + tkey[0,0,6] >= 0
 R1538: sbox[0,0,6] + rkey[0,0,6] - tkey[0,0,6] >= 0
 R1539: - sbox[0,0,6] + rkey[0,0,6] + tkey[0,0,6] >= 0
 R1540: - tbox[1,0,6] + forg[1,0,6] <= 0
 R1541: sbox[1,0,6] - rkey[1,0,6] + tkey[1,0,6] >= 0
 R1542: sbox[1,0,6] + rkey[1,0,6] - tkey[1,0,6] >= 0
 R1543: - sbox[1,0,6] + rkey[1,0,6] + tkey[1,0,6] >= 0
 R1544: - tbox[2,0,6] + forg[2,0,6] <= 0
 R1545: - sbox[2,0,6] + rkey[2,0,6] = 0
 R1546: - tbox[3,0,6] + forg[3,0,6] <= 0
 R1547: - sbox[3,0,6] + rkey[3,0,6] = 0
 R1548: - sbox[3,0,7] + rkey[0,0,6] + rkey[2,2,6] >= 0
 R1549: sbox[3,0,7] + rkey[0,0,6] - rkey[2,2,6] >= 0
 R1550: sbox[3,0,7] - rkey[0,0,6] + rkey[2,2,6] >= 0
 R1551: - sbox[0,0,7] + sbox[3,0,7] + rkey[3,1,6] >= 0
 R1552: sbox[0,0,7] + sbox[3,0,7] - rkey[3,1,6] >= 0
 R1553: sbox[0,0,7] - sbox[3,0,7] + rkey[3,1,6] >= 0
 R1554: - sbox[2,0,7] + rkey[1,3,6] + rkey[2,2,6] >= 0
 R1555: sbox[2,0,7] + rkey[1,3,6] - rkey[2,2,6] >= 0
 R1556: sbox[2,0,7] - rkey[1,3,6] + rkey[2,2,6] >= 0
 R1557: sbox[1,0,7] - rkey[0,0,6] = 0
 R1558: tbox[0,0,6] + tbox[2,2,6] - tbox[3,0,7] >= 0
 R1559: tbox[0,0,6] - tbox[2,2,6] + tbox[3,0,7] >= 0
 R1560: - tbox[0,0,6] + tbox[2,2,6] + tbox[3,0,7] >= 0
 R1561: - tbox[0,0,7] + tbox[3,0,7] + tbox[3,1,6] >= 0
 R1562: tbox[0,0,7] + tbox[3,0,7] - tbox[3,1,6] >= 0
 R1563: tbox[0,0,7] - tbox[3,0,7] + tbox[3,1,6] >= 0
 R1564: tbox[1,3,6] - tbox[2,0,7] + tbox[2,2,6] >= 0
 R1565: tbox[1,3,6] + tbox[2,0,7] - tbox[2,2,6] >= 0
 R1566: - tbox[1,3,6] + tbox[2,0,7] + tbox[2,2,6] >= 0
 R1567: - tbox[0,0,6] + tbox[1,0,7] = 0
 R1568: unkn[0,0,6] - unkn[3,0,7] + forg[3,0,7] <= 0
 R1569: unkn[2,2,6] - unkn[3,0,7] + forg[3,0,7] <= 0
 R1570: unkn[0,0,6] + unkn[2,2,6] - unkn[3,0,7] + forg[3,0,7] >= 0
 R1571: - unkn[0,0,7] + unkn[3,0,7] + forg[0,0,7] - forg[3,0,7] <= 0
 R1572: - unkn[0,0,7] + unkn[3,1,6] + forg[0,0,7] <= 0
 R1573: - unkn[0,0,7] + unkn[3,0,7] + unkn[3,1,6] + forg[0,0,7]
   - forg[3,0,7] >= 0
 R1574: unkn[1,3,6] - unkn[2,0,7] + forg[2,0,7] <= 0
 R1575: - unkn[2,0,7] + unkn[2,2,6] + forg[2,0,7] <= 0
 R1576: unkn[1,3,6] - unkn[2,0,7] + unkn[2,2,6] + forg[2,0,7] >= 0
 R1577: unkn[0,0,6] - unkn[1,0,7] + forg[1,0,7] = 0
 R1578: - tbox[0,1,6] + forg[0,1,6] <= 0
 R1579: sbox[0,1,6] - rkey[0,1,6] + tkey[0,1,6] >= 0
 R1580: sbox[0,1,6] + rkey[0,1,6] - tkey[0,1,6] >= 0
 R1581: - sbox[0,1,6] + rkey[0,1,6] + tkey[0,1,6] >= 0
 R1582: - tbox[1,1,6] + forg[1,1,6] <= 0
 R1583: sbox[1,1,6] - rkey[1,1,6] + tkey[1,1,6] >= 0
 R1584: sbox[1,1,6] + rkey[1,1,6] - tkey[1,1,6] >= 0
 R1585: - sbox[1,1,6] + rkey[1,1,6] + tkey[1,1,6] >= 0
 R1586: - tbox[2,1,6] + forg[2,1,6] <= 0
 R1587: - sbox[2,1,6] + rkey[2,1,6] = 0
 R1588: - tbox[3,1,6] + forg[3,1,6] <= 0
 R1589: - sbox[3,1,6] + rkey[3,1,6] = 0
 R1590: - sbox[3,1,7] + rkey[0,1,6] + rkey[2,3,6] >= 0
 R1591: sbox[3,1,7] + rkey[0,1,6] - rkey[2,3,6] >= 0
 R1592: sbox[3,1,7] - rkey[0,1,6] + rkey[2,3,6] >= 0
 R1593: - sbox[0,1,7] + sbox[3,1,7] + rkey[3,2,6] >= 0
 R1594: sbox[0,1,7] + sbox[3,1,7] - rkey[3,2,6] >= 0
 R1595: sbox[0,1,7] - sbox[3,1,7] + rkey[3,2,6] >= 0
 R1596: - sbox[2,1,7] + rkey[1,0,6] + rkey[2,3,6] >= 0
 R1597: sbox[2,1,7] + rkey[1,0,6] - rkey[2,3,6] >= 0
 R1598: sbox[2,1,7] - rkey[1,0,6] + rkey[2,3,6] >= 0
 R1599: sbox[1,1,7] - rkey[0,1,6] = 0
 R1600: tbox[0,1,6] + tbox[2,3,6] - tbox[3,1,7] >= 0
 R1601: tbox[0,1,6] - tbox[2,3,6] + tbox[3,1,7] >= 0
 R1602: - tbox[0,1,6] + tbox[2,3,6] + tbox[3,1,7] >= 0
 R1603: - tbox[0,1,7] + tbox[3,1,7] + tbox[3,2,6] >= 0
 R1604: tbox[0,1,7] + tbox[3,1,7] - tbox[3,2,6] >= 0
 R1605: tbox[0,1,7] - tbox[3,1,7] + tbox[3,2,6] >= 0
 R1606: tbox[1,0,6] - tbox[2,1,7] + tbox[2,3,6] >= 0
 R1607: tbox[1,0,6] + tbox[2,1,7] - tbox[2,3,6] >= 0
 R1608: - tbox[1,0,6] + tbox[2,1,7] + tbox[2,3,6] >= 0
 R1609: - tbox[0,1,6] + tbox[1,1,7] = 0
 R1610: unkn[0,1,6] - unkn[3,1,7] + forg[3,1,7] <= 0
 R1611: unkn[2,3,6] - unkn[3,1,7] + forg[3,1,7] <= 0
 R1612: unkn[0,1,6] + unkn[2,3,6] - unkn[3,1,7] + forg[3,1,7] >= 0
 R1613: - unkn[0,1,7] + unkn[3,1,7] + forg[0,1,7] - forg[3,1,7] <= 0
 R1614: - unkn[0,1,7] + unkn[3,2,6] + forg[0,1,7] <= 0
 R1615: - unkn[0,1,7] + unkn[3,1,7] + unkn[3,2,6] + forg[0,1,7]
   - forg[3,1,7] >= 0
 R1616: unkn[1,0,6] - unkn[2,1,7] + forg[2,1,7] <= 0
 R1617: - unkn[2,1,7] + unkn[2,3,6] + forg[2,1,7] <= 0
 R1618: unkn[1,0,6] - unkn[2,1,7] + unkn[2,3,6] + forg[2,1,7] >= 0
 R1619: unkn[0,1,6] - unkn[1,1,7] + forg[1,1,7] = 0
 R1620: - tbox[0,2,6] + forg[0,2,6] <= 0
 R1621: sbox[0,2,6] - rkey[0,2,6] + tkey[0,2,6] >= 0
 R1622: sbox[0,2,6] + rkey[0,2,6] - tkey[0,2,6] >= 0
 R1623: - sbox[0,2,6] + rkey[0,2,6] + tkey[0,2,6] >= 0
 R1624: - tbox[1,2,6] + forg[1,2,6] <= 0
 R1625: sbox[1,2,6] - rkey[1,2,6] + tkey[1,2,6] >= 0
 R1626: sbox[1,2,6] + rkey[1,2,6] - tkey[1,2,6] >= 0
 R1627: - sbox[1,2,6] + rkey[1,2,6] + tkey[1,2,6] >= 0
 R1628: - tbox[2,2,6] + forg[2,2,6] <= 0
 R1629: - sbox[2,2,6] + rkey[2,2,6] = 0
 R1630: - tbox[3,2,6] + forg[3,2,6] <= 0
 R1631: - sbox[3,2,6] + rkey[3,2,6] = 0
 R1632: - sbox[3,2,7] + rkey[0,2,6] + rkey[2,0,6] >= 0
 R1633: sbox[3,2,7] + rkey[0,2,6] - rkey[2,0,6] >= 0
 R1634: sbox[3,2,7] - rkey[0,2,6] + rkey[2,0,6] >= 0
 R1635: - sbox[0,2,7] + sbox[3,2,7] + rkey[3,3,6] >= 0
 R1636: sbox[0,2,7] + sbox[3,2,7] - rkey[3,3,6] >= 0
 R1637: sbox[0,2,7] - sbox[3,2,7] + rkey[3,3,6] >= 0
 R1638: - sbox[2,2,7] + rkey[1,1,6] + rkey[2,0,6] >= 0
 R1639: sbox[2,2,7] + rkey[1,1,6] - rkey[2,0,6] >= 0
 R1640: sbox[2,2,7] - rkey[1,1,6] + rkey[2,0,6] >= 0
 R1641: sbox[1,2,7] - rkey[0,2,6] = 0
 R1642: tbox[0,2,6] + tbox[2,0,6] - tbox[3,2,7] >= 0
 R1643: tbox[0,2,6] - tbox[2,0,6] + tbox[3,2,7] >= 0
 R1644: - tbox[0,2,6] + tbox[2,0,6] + tbox[3,2,7] >= 0
 R1645: - tbox[0,2,7] + tbox[3,2,7] + tbox[3,3,6] >= 0
 R1646: tbox[0,2,7] + tbox[3,2,7] - tbox[3,3,6] >= 0
 R1647: tbox[0,2,7] - tbox[3,2,7] + tbox[3,3,6] >= 0
 R1648: tbox[1,1,6] + tbox[2,0,6] - tbox[2,2,7] >= 0
 R1649: tbox[1,1,6] - tbox[2,0,6] + tbox[2,2,7] >= 0
 R1650: - tbox[1,1,6] + tbox[2,0,6] + tbox[2,2,7] >= 0
 R1651: - tbox[0,2,6] + tbox[1,2,7] = 0
 R1652: unkn[0,2,6] - unkn[3,2,7] + forg[3,2,7] <= 0
 R1653: unkn[2,0,6] - unkn[3,2,7] + forg[3,2,7] <= 0
 R1654: unkn[0,2,6] + unkn[2,0,6] - unkn[3,2,7] + forg[3,2,7] >= 0
 R1655: - unkn[0,2,7] + unkn[3,2,7] + forg[0,2,7] - forg[3,2,7] <= 0
 R1656: - unkn[0,2,7] + unkn[3,3,6] + forg[0,2,7] <= 0
 R1657: - unkn[0,2,7] + unkn[3,2,7] + unkn[3,3,6] + forg[0,2,7]
   - forg[3,2,7] >= 0
 R1658: unkn[1,1,6] - unkn[2,2,7] + forg[2,2,7] <= 0
 R1659: unkn[2,0,6] - unkn[2,2,7] + forg[2,2,7] <= 0
 R1660: unkn[1,1,6] + unkn[2,0,6] - unkn[2,2,7] + forg[2,2,7] >= 0
 R1661: unkn[0,2,6] - unkn[1,2,7] + forg[1,2,7] = 0
 R1662: - tbox[0,3,6] + forg[0,3,6] <= 0
 R1663: sbox[0,3,6] - rkey[0,3,6] + tkey[0,3,6] >= 0
 R1664: sbox[0,3,6] + rkey[0,3,6] - tkey[0,3,6] >= 0
 R1665: - sbox[0,3,6] + rkey[0,3,6] + tkey[0,3,6] >= 0
 R1666: - tbox[1,3,6] + forg[1,3,6] <= 0
 R1667: sbox[1,3,6] - rkey[1,3,6] + tkey[1,3,6] >= 0
 R1668: sbox[1,3,6] + rkey[1,3,6] - tkey[1,3,6] >= 0
 R1669: - sbox[1,3,6] + rkey[1,3,6] + tkey[1,3,6] >= 0
 R1670: - tbox[2,3,6] + forg[2,3,6] <= 0
 R1671: - sbox[2,3,6] + rkey[2,3,6] = 0
 R1672: - tbox[3,3,6] + forg[3,3,6] <= 0
 R1673: - sbox[3,3,6] + rkey[3,3,6] = 0
 R1674: - sbox[3,3,7] + rkey[0,3,6] + rkey[2,1,6] >= 0
 R1675: sbox[3,3,7] + rkey[0,3,6] - rkey[2,1,6] >= 0
 R1676: sbox[3,3,7] - rkey[0,3,6] + rkey[2,1,6] >= 0
 R1677: - sbox[0,3,7] + sbox[3,3,7] + rkey[3,0,6] >= 0
 R1678: sbox[0,3,7] + sbox[3,3,7] - rkey[3,0,6] >= 0
 R1679: sbox[0,3,7] - sbox[3,3,7] + rkey[3,0,6] >= 0
 R1680: - sbox[2,3,7] + rkey[1,2,6] + rkey[2,1,6] >= 0
 R1681: sbox[2,3,7] + rkey[1,2,6] - rkey[2,1,6] >= 0
 R1682: sbox[2,3,7] - rkey[1,2,6] + rkey[2,1,6] >= 0
 R1683: sbox[1,3,7] - rkey[0,3,6] = 0
 R1684: tbox[0,3,6] + tbox[2,1,6] - tbox[3,3,7] >= 0
 R1685: tbox[0,3,6] - tbox[2,1,6] + tbox[3,3,7] >= 0
 R1686: - tbox[0,3,6] + tbox[2,1,6] + tbox[3,3,7] >= 0
 R1687: - tbox[0,3,7] + tbox[3,0,6] + tbox[3,3,7] >= 0
 R1688: tbox[0,3,7] - tbox[3,0,6] + tbox[3,3,7] >= 0
 R1689: tbox[0,3,7] + tbox[3,0,6] - tbox[3,3,7] >= 0
 R1690: tbox[1,2,6] + tbox[2,1,6] - tbox[2,3,7] >= 0
 R1691: tbox[1,2,6] - tbox[2,1,6] + tbox[2,3,7] >= 0
 R1692: - tbox[1,2,6] + tbox[2,1,6] + tbox[2,3,7] >= 0
 R1693: - tbox[0,3,6] + tbox[1,3,7] = 0
 R1694: unkn[0,3,6] - unkn[3,3,7] + forg[3,3,7] <= 0
 R1695: unkn[2,1,6] - unkn[3,3,7] + forg[3,3,7] <= 0
 R1696: unkn[0,3,6] + unkn[2,1,6] - unkn[3,3,7] + forg[3,3,7] >= 0
 R1697: - unkn[0,3,7] + unkn[3,3,7] + forg[0,3,7] - forg[3,3,7] <= 0
 R1698: - unkn[0,3,7] + unkn[3,0,6] + forg[0,3,7] <= 0
 R1699: - unkn[0,3,7] + unkn[3,0,6] + unkn[3,3,7] + forg[0,3,7]
   - forg[3,3,7] >= 0
 R1700: unkn[1,2,6] - unkn[2,3,7] + forg[2,3,7] <= 0
 R1701: unkn[2,1,6] - unkn[2,3,7] + forg[2,3,7] <= 0
 R1702: unkn[1,2,6] + unkn[2,1,6] - unkn[2,3,7] + forg[2,3,7] >= 0
 R1703: unkn[0,3,6] - unkn[1,3,7] + forg[1,3,7] = 0
 R1704: - tbox[0,0,7] + forg[0,0,7] <= 0
 R1705: sbox[0,0,7] - rkey[0,0,7] + tkey[0,0,7] >= 0
 R1706: sbox[0,0,7] + rkey[0,0,7] - tkey[0,0,7] >= 0
 R1707: - sbox[0,0,7] + rkey[0,0,7] + tkey[0,0,7] >= 0
 R1708: - tbox[1,0,7] + forg[1,0,7] <= 0
 R1709: sbox[1,0,7] - rkey[1,0,7] + tkey[1,0,7] >= 0
 R1710: sbox[1,0,7] + rkey[1,0,7] - tkey[1,0,7] >= 0
 R1711: - sbox[1,0,7] + rkey[1,0,7] + tkey[1,0,7] >= 0
 R1712: - tbox[2,0,7] + forg[2,0,7] <= 0
 R1713: - sbox[2,0,7] + rkey[2,0,7] = 0
 R1714: - tbox[3,0,7] + forg[3,0,7] <= 0
 R1715: - sbox[3,0,7] + rkey[3,0,7] = 0
 R1716: - sbox[3,0,8] + rkey[0,0,7] + rkey[2,2,7] >= 0
 R1717: sbox[3,0,8] + rkey[0,0,7] - rkey[2,2,7] >= 0
 R1718: sbox[3,0,8] - rkey[0,0,7] + rkey[2,2,7] >= 0
 R1719: - sbox[0,0,8] + sbox[3,0,8] + rkey[3,1,7] >= 0
 R1720: sbox[0,0,8] + sbox[3,0,8] - rkey[3,1,7] >= 0
 R1721: sbox[0,0,8] - sbox[3,0,8] + rkey[3,1,7] >= 0
 R1722: - sbox[2,0,8] + rkey[1,3,7] + rkey[2,2,7] >= 0
 R1723: sbox[2,0,8] + rkey[1,3,7] - rkey[2,2,7] >= 0
 R1724: sbox[2,0,8] - rkey[1,3,7] + rkey[2,2,7] >= 0
 R1725: sbox[1,0,8] - rkey[0,0,7] = 0
 R1726: tbox[0,0,7] + tbox[2,2,7] - tbox[3,0,8] >= 0
 R1727: tbox[0,0,7] - tbox[2,2,7] + tbox[3,0,8] >= 0
 R1728: - tbox[0,0,7] + tbox[2,2,7] + tbox[3,0,8] >= 0
 R1729: - tbox[0,0,8] + tbox[3,0,8] + tbox[3,1,7] >= 0
 R1730: tbox[0,0,8] + tbox[3,0,8] - tbox[3,1,7] >= 0
 R1731: tbox[0,0,8] - tbox[3,0,8] + tbox[3,1,7] >= 0
 R1732: tbox[1,3,7] - tbox[2,0,8] + tbox[2,2,7] >= 0
 R1733: tbox[1,3,7] + tbox[2,0,8] - tbox[2,2,7] >= 0
 R1734: - tbox[1,3,7] + tbox[2,0,8] + tbox[2,2,7] >= 0
 R1735: - tbox[0,0,7] + tbox[1,0,8] = 0
 R1736: unkn[0,0,7] - unkn[3,0,8] + forg[3,0,8] <= 0
 R1737: unkn[2,2,7] - unkn[3,0,8] + forg[3,0,8] <= 0
 R1738: unkn[0,0,7] + unkn[2,2,7] - unkn[3,0,8] + forg[3,0,8] >= 0
 R1739: - unkn[0,0,8] + unkn[3,0,8] + forg[0,0,8] - forg[3,0,8] <= 0
 R1740: - unkn[0,0,8] + unkn[3,1,7] + forg[0,0,8] <= 0
 R1741: - unkn[0,0,8] + unkn[3,0,8] + unkn[3,1,7] + forg[0,0,8]
   - forg[3,0,8] >= 0
 R1742: unkn[1,3,7] - unkn[2,0,8] + forg[2,0,8] <= 0
 R1743: - unkn[2,0,8] + unkn[2,2,7] + forg[2,0,8] <= 0
 R1744: unkn[1,3,7] - unkn[2,0,8] + unkn[2,2,7] + forg[2,0,8] >= 0
 R1745: unkn[0,0,7] - unkn[1,0,8] + forg[1,0,8] = 0
 R1746: - tbox[0,1,7] + forg[0,1,7] <= 0
 R1747: sbox[0,1,7] - rkey[0,1,7] + tkey[0,1,7] >= 0
 R1748: sbox[0,1,7] + rkey[0,1,7] - tkey[0,1,7] >= 0
 R1749: - sbox[0,1,7] + rkey[0,1,7] + tkey[0,1,7] >= 0
 R1750: - tbox[1,1,7] + forg[1,1,7] <= 0
 R1751: sbox[1,1,7] - rkey[1,1,7] + tkey[1,1,7] >= 0
 R1752: sbox[1,1,7] + rkey[1,1,7] - tkey[1,1,7] >= 0
 R1753: - sbox[1,1,7] + rkey[1,1,7] + tkey[1,1,7] >= 0
 R1754: - tbox[2,1,7] + forg[2,1,7] <= 0
 R1755: - sbox[2,1,7] + rkey[2,1,7] = 0
 R1756: - tbox[3,1,7] + forg[3,1,7] <= 0
 R1757: - sbox[3,1,7] + rkey[3,1,7] = 0
 R1758: - sbox[3,1,8] + rkey[0,1,7] + rkey[2,3,7] >= 0
 R1759: sbox[3,1,8] + rkey[0,1,7] - rkey[2,3,7] >= 0
 R1760: sbox[3,1,8] - rkey[0,1,7] + rkey[2,3,7] >= 0
 R1761: - sbox[0,1,8] + sbox[3,1,8] + rkey[3,2,7] >= 0
 R1762: sbox[0,1,8] + sbox[3,1,8] - rkey[3,2,7] >= 0
 R1763: sbox[0,1,8] - sbox[3,1,8] + rkey[3,2,7] >= 0
 R1764: - sbox[2,1,8] + rkey[1,0,7] + rkey[2,3,7] >= 0
 R1765: sbox[2,1,8] + rkey[1,0,7] - rkey[2,3,7] >= 0
 R1766: sbox[2,1,8] - rkey[1,0,7] + rkey[2,3,7] >= 0
 R1767: sbox[1,1,8] - rkey[0,1,7] = 0
 R1768: tbox[0,1,7] + tbox[2,3,7] - tbox[3,1,8] >= 0
 R1769: tbox[0,1,7] - tbox[2,3,7] + tbox[3,1,8] >= 0
 R1770: - tbox[0,1,7] + tbox[2,3,7] + tbox[3,1,8] >= 0
 R1771: - tbox[0,1,8] + tbox[3,1,8] + tbox[3,2,7] >= 0
 R1772: tbox[0,1,8] + tbox[3,1,8] - tbox[3,2,7] >= 0
 R1773: tbox[0,1,8] - tbox[3,1,8] + tbox[3,2,7] >= 0
 R1774: tbox[1,0,7] - tbox[2,1,8] + tbox[2,3,7] >= 0
 R1775: tbox[1,0,7] + tbox[2,1,8] - tbox[2,3,7] >= 0
 R1776: - tbox[1,0,7] + tbox[2,1,8] + tbox[2,3,7] >= 0
 R1777: - tbox[0,1,7] + tbox[1,1,8] = 0
 R1778: unkn[0,1,7] - unkn[3,1,8] + forg[3,1,8] <= 0
 R1779: unkn[2,3,7] - unkn[3,1,8] + forg[3,1,8] <= 0
 R1780: unkn[0,1,7] + unkn[2,3,7] - unkn[3,1,8] + forg[3,1,8] >= 0
 R1781: - unkn[0,1,8] + unkn[3,1,8] + forg[0,1,8] - forg[3,1,8] <= 0
 R1782: - unkn[0,1,8] + unkn[3,2,7] + forg[0,1,8] <= 0
 R1783: - unkn[0,1,8] + unkn[3,1,8] + unkn[3,2,7] + forg[0,1,8]
   - forg[3,1,8] >= 0
 R1784: unkn[1,0,7] - unkn[2,1,8] + forg[2,1,8] <= 0
 R1785: - unkn[2,1,8] + unkn[2,3,7] + forg[2,1,8] <= 0
 R1786: unkn[1,0,7] - unkn[2,1,8] + unkn[2,3,7] + forg[2,1,8] >= 0
 R1787: unkn[0,1,7] - unkn[1,1,8] + forg[1,1,8] = 0
 R1788: - tbox[0,2,7] + forg[0,2,7] <= 0
 R1789: sbox[0,2,7] - rkey[0,2,7] + tkey[0,2,7] >= 0
 R1790: sbox[0,2,7] + rkey[0,2,7] - tkey[0,2,7] >= 0
 R1791: - sbox[0,2,7] + rkey[0,2,7] + tkey[0,2,7] >= 0
 R1792: - tbox[1,2,7] + forg[1,2,7] <= 0
 R1793: sbox[1,2,7] - rkey[1,2,7] + tkey[1,2,7] >= 0
 R1794: sbox[1,2,7] + rkey[1,2,7] - tkey[1,2,7] >= 0
 R1795: - sbox[1,2,7] + rkey[1,2,7] + tkey[1,2,7] >= 0
 R1796: - tbox[2,2,7] + forg[2,2,7] <= 0
 R1797: - sbox[2,2,7] + rkey[2,2,7] = 0
 R1798: - tbox[3,2,7] + forg[3,2,7] <= 0
 R1799: - sbox[3,2,7] + rkey[3,2,7] = 0
 R1800: - sbox[3,2,8] + rkey[0,2,7] + rkey[2,0,7] >= 0
 R1801: sbox[3,2,8] + rkey[0,2,7] - rkey[2,0,7] >= 0
 R1802: sbox[3,2,8] - rkey[0,2,7] + rkey[2,0,7] >= 0
 R1803: - sbox[0,2,8] + sbox[3,2,8] + rkey[3,3,7] >= 0
 R1804: sbox[0,2,8] + sbox[3,2,8] - rkey[3,3,7] >= 0
 R1805: sbox[0,2,8] - sbox[3,2,8] + rkey[3,3,7] >= 0
 R1806: - sbox[2,2,8] + rkey[1,1,7] + rkey[2,0,7] >= 0
 R1807: sbox[2,2,8] + rkey[1,1,7] - rkey[2,0,7] >= 0
 R1808: sbox[2,2,8] - rkey[1,1,7] + rkey[2,0,7] >= 0
 R1809: sbox[1,2,8] - rkey[0,2,7] = 0
 R1810: tbox[0,2,7] + tbox[2,0,7] - tbox[3,2,8] >= 0
 R1811: tbox[0,2,7] - tbox[2,0,7] + tbox[3,2,8] >= 0
 R1812: - tbox[0,2,7] + tbox[2,0,7] + tbox[3,2,8] >= 0
 R1813: - tbox[0,2,8] + tbox[3,2,8] + tbox[3,3,7] >= 0
 R1814: tbox[0,2,8] + tbox[3,2,8] - tbox[3,3,7] >= 0
 R1815: tbox[0,2,8] - tbox[3,2,8] + tbox[3,3,7] >= 0
 R1816: tbox[1,1,7] + tbox[2,0,7] - tbox[2,2,8] >= 0
 R1817: tbox[1,1,7] - tbox[2,0,7] + tbox[2,2,8] >= 0
 R1818: - tbox[1,1,7] + tbox[2,0,7] + tbox[2,2,8] >= 0
 R1819: - tbox[0,2,7] + tbox[1,2,8] = 0
 R1820: unkn[0,2,7] - unkn[3,2,8] + forg[3,2,8] <= 0
 R1821: unkn[2,0,7] - unkn[3,2,8] + forg[3,2,8] <= 0
 R1822: unkn[0,2,7] + unkn[2,0,7] - unkn[3,2,8] + forg[3,2,8] >= 0
 R1823: - unkn[0,2,8] + unkn[3,2,8] + forg[0,2,8] - forg[3,2,8] <= 0
 R1824: - unkn[0,2,8] + unkn[3,3,7] + forg[0,2,8] <= 0
 R1825: - unkn[0,2,8] + unkn[3,2,8] + unkn[3,3,7] + forg[0,2,8]
   - forg[3,2,8] >= 0
 R1826: unkn[1,1,7] - unkn[2,2,8] + forg[2,2,8] <= 0
 R1827: unkn[2,0,7] - unkn[2,2,8] + forg[2,2,8] <= 0
 R1828: unkn[1,1,7] + unkn[2,0,7] - unkn[2,2,8] + forg[2,2,8] >= 0
 R1829: unkn[0,2,7] - unkn[1,2,8] + forg[1,2,8] = 0
 R1830: - tbox[0,3,7] + forg[0,3,7] <= 0
 R1831: sbox[0,3,7] - rkey[0,3,7] + tkey[0,3,7] >= 0
 R1832: sbox[0,3,7] + rkey[0,3,7] - tkey[0,3,7] >= 0
 R1833: - sbox[0,3,7] + rkey[0,3,7] + tkey[0,3,7] >= 0
 R1834: - tbox[1,3,7] + forg[1,3,7] <= 0
 R1835: sbox[1,3,7] - rkey[1,3,7] + tkey[1,3,7] >= 0
 R1836: sbox[1,3,7] + rkey[1,3,7] - tkey[1,3,7] >= 0
 R1837: - sbox[1,3,7] + rkey[1,3,7] + tkey[1,3,7] >= 0
 R1838: - tbox[2,3,7] + forg[2,3,7] <= 0
 R1839: - sbox[2,3,7] + rkey[2,3,7] = 0
 R1840: - tbox[3,3,7] + forg[3,3,7] <= 0
 R1841: - sbox[3,3,7] + rkey[3,3,7] = 0
 R1842: - sbox[3,3,8] + rkey[0,3,7] + rkey[2,1,7] >= 0
 R1843: sbox[3,3,8] + rkey[0,3,7] - rkey[2,1,7] >= 0
 R1844: sbox[3,3,8] - rkey[0,3,7] + rkey[2,1,7] >= 0
 R1845: - sbox[0,3,8] + sbox[3,3,8] + rkey[3,0,7] >= 0
 R1846: sbox[0,3,8] + sbox[3,3,8] - rkey[3,0,7] >= 0
 R1847: sbox[0,3,8] - sbox[3,3,8] + rkey[3,0,7] >= 0
 R1848: - sbox[2,3,8] + rkey[1,2,7] + rkey[2,1,7] >= 0
 R1849: sbox[2,3,8] + rkey[1,2,7] - rkey[2,1,7] >= 0
 R1850: sbox[2,3,8] - rkey[1,2,7] + rkey[2,1,7] >= 0
 R1851: sbox[1,3,8] - rkey[0,3,7] = 0
 R1852: tbox[0,3,7] + tbox[2,1,7] - tbox[3,3,8] >= 0
 R1853: tbox[0,3,7] - tbox[2,1,7] + tbox[3,3,8] >= 0
 R1854: - tbox[0,3,7] + tbox[2,1,7] + tbox[3,3,8] >= 0
 R1855: - tbox[0,3,8] + tbox[3,0,7] + tbox[3,3,8] >= 0
 R1856: tbox[0,3,8] - tbox[3,0,7] + tbox[3,3,8] >= 0
 R1857: tbox[0,3,8] + tbox[3,0,7] - tbox[3,3,8] >= 0
 R1858: tbox[1,2,7] + tbox[2,1,7] - tbox[2,3,8] >= 0
 R1859: tbox[1,2,7] - tbox[2,1,7] + tbox[2,3,8] >= 0
 R1860: - tbox[1,2,7] + tbox[2,1,7] + tbox[2,3,8] >= 0
 R1861: - tbox[0,3,7] + tbox[1,3,8] = 0
 R1862: unkn[0,3,7] - unkn[3,3,8] + forg[3,3,8] <= 0
 R1863: unkn[2,1,7] - unkn[3,3,8] + forg[3,3,8] <= 0
 R1864: unkn[0,3,7] + unkn[2,1,7] - unkn[3,3,8] + forg[3,3,8] >= 0
 R1865: - unkn[0,3,8] + unkn[3,3,8] + forg[0,3,8] - forg[3,3,8] <= 0
 R1866: - unkn[0,3,8] + unkn[3,0,7] + forg[0,3,8] <= 0
 R1867: - unkn[0,3,8] + unkn[3,0,7] + unkn[3,3,8] + forg[0,3,8]
   - forg[3,3,8] >= 0
 R1868: unkn[1,2,7] - unkn[2,3,8] + forg[2,3,8] <= 0
 R1869: unkn[2,1,7] - unkn[2,3,8] + forg[2,3,8] <= 0
 R1870: unkn[1,2,7] + unkn[2,1,7] - unkn[2,3,8] + forg[2,3,8] >= 0
 R1871: unkn[0,3,7] - unkn[1,3,8] + forg[1,3,8] = 0
 R1872: sbox[0,0,0] + sbox[0,1,0] + sbox[0,2,0] + sbox[0,3,0] + sbox[1,0,0]
   + sbox[1,1,0] + sbox[1,2,0] + sbox[1,3,0] + sbox[2,0,0] + sbox[2,1,0]
   + sbox[2,2,0] + sbox[2,3,0] + sbox[3,0,0] + sbox[3,1,0] + sbox[3,2,0]
   + sbox[3,3,0] + tkey[0,0,0] + tkey[0,1,0] + tkey[0,2,0] + tkey[0,3,0]
   + tkey[1,0,0] + tkey[1,1,0] + tkey[1,2,0] + tkey[1,3,0] + tkey[2,0,0]
   + tkey[2,1,0] + tkey[2,2,0] + tkey[2,3,0] + tkey[3,0,0] + tkey[3,1,0]
   + tkey[3,2,0] + tkey[3,3,0] >= 1
 R1873: sbox[0,0,1] + sbox[0,1,1] + sbox[0,2,1] + sbox[0,3,1] + sbox[1,0,1]
   + sbox[1,1,1] + sbox[1,2,1] + sbox[1,3,1] + sbox[2,0,1] + sbox[2,1,1]
   + sbox[2,2,1] + sbox[2,3,1] + sbox[3,0,1] + sbox[3,1,1] + sbox[3,2,1]
   + sbox[3,3,1] + tkey[0,0,1] + tkey[0,1,1] + tkey[0,2,1] + tkey[0,3,1]
   + tkey[1,0,1] + tkey[1,1,1] + tkey[1,2,1] + tkey[1,3,1] + tkey[2,0,1]
   + tkey[2,1,1] + tkey[2,2,1] + tkey[2,3,1] + tkey[3,0,1] + tkey[3,1,1]
   + tkey[3,2,1] + tkey[3,3,1] >= 1
 R1874: sbox[0,0,2] + sbox[0,1,2] + sbox[0,2,2] + sbox[0,3,2] + sbox[1,0,2]
   + sbox[1,1,2] + sbox[1,2,2] + sbox[1,3,2] + sbox[2,0,2] + sbox[2,1,2]
   + sbox[2,2,2] + sbox[2,3,2] + sbox[3,0,2] + sbox[3,1,2] + sbox[3,2,2]
   + sbox[3,3,2] + tkey[0,0,2] + tkey[0,1,2] + tkey[0,2,2] + tkey[0,3,2]
   + tkey[1,0,2] + tkey[1,1,2] + tkey[1,2,2] + tkey[1,3,2] + tkey[2,0,2]
   + tkey[2,1,2] + tkey[2,2,2] + tkey[2,3,2] + tkey[3,0,2] + tkey[3,1,2]
   + tkey[3,2,2] + tkey[3,3,2] >= 1
 R1875: sbox[0,0,3] + sbox[0,1,3] + sbox[0,2,3] + sbox[0,3,3] + sbox[1,0,3]
   + sbox[1,1,3] + sbox[1,2,3] + sbox[1,3,3] + sbox[2,0,3] + sbox[2,1,3]
   + sbox[2,2,3] + sbox[2,3,3] + sbox[3,0,3] + sbox[3,1,3] + sbox[3,2,3]
   + sbox[3,3,3] + tkey[0,0,3] + tkey[0,1,3] + tkey[0,2,3] + tkey[0,3,3]
   + tkey[1,0,3] + tkey[1,1,3] + tkey[1,2,3] + tkey[1,3,3] + tkey[2,0,3]
   + tkey[2,1,3] + tkey[2,2,3] + tkey[2,3,3] + tkey[3,0,3] + tkey[3,1,3]
   + tkey[3,2,3] + tkey[3,3,3] >= 1
 R1876: sbox[0,0,4] + sbox[0,1,4] + sbox[0,2,4] + sbox[0,3,4] + sbox[1,0,4]
   + sbox[1,1,4] + sbox[1,2,4] + sbox[1,3,4] + sbox[2,0,4] + sbox[2,1,4]
   + sbox[2,2,4] + sbox[2,3,4] + sbox[3,0,4] + sbox[3,1,4] + sbox[3,2,4]
   + sbox[3,3,4] + tkey[0,0,4] + tkey[0,1,4] + tkey[0,2,4] + tkey[0,3,4]
   + tkey[1,0,4] + tkey[1,1,4] + tkey[1,2,4] + tkey[1,3,4] + tkey[2,0,4]
   + tkey[2,1,4] + tkey[2,2,4] + tkey[2,3,4] + tkey[3,0,4] + tkey[3,1,4]
   + tkey[3,2,4] + tkey[3,3,4] >= 1
 R1877: sbox[0,0,5] + sbox[0,1,5] + sbox[0,2,5] + sbox[0,3,5] + sbox[1,0,5]
   + sbox[1,1,5] + sbox[1,2,5] + sbox[1,3,5] + sbox[2,0,5] + sbox[2,1,5]
   + sbox[2,2,5] + sbox[2,3,5] + sbox[3,0,5] + sbox[3,1,5] + sbox[3,2,5]
   + sbox[3,3,5] + tkey[0,0,5] + tkey[0,1,5] + tkey[0,2,5] + tkey[0,3,5]
   + tkey[1,0,5] + tkey[1,1,5] + tkey[1,2,5] + tkey[1,3,5] + tkey[2,0,5]
   + tkey[2,1,5] + tkey[2,2,5] + tkey[2,3,5] + tkey[3,0,5] + tkey[3,1,5]
   + tkey[3,2,5] + tkey[3,3,5] >= 1
 R1878: sbox[0,0,6] + sbox[0,1,6] + sbox[0,2,6] + sbox[0,3,6] + sbox[1,0,6]
   + sbox[1,1,6] + sbox[1,2,6] + sbox[1,3,6] + sbox[2,0,6] + sbox[2,1,6]
   + sbox[2,2,6] + sbox[2,3,6] + sbox[3,0,6] + sbox[3,1,6] + sbox[3,2,6]
   + sbox[3,3,6] + tkey[0,0,6] + tkey[0,1,6] + tkey[0,2,6] + tkey[0,3,6]
   + tkey[1,0,6] + tkey[1,1,6] + tkey[1,2,6] + tkey[1,3,6] + tkey[2,0,6]
   + tkey[2,1,6] + tkey[2,2,6] + tkey[2,3,6] + tkey[3,0,6] + tkey[3,1,6]
   + tkey[3,2,6] + tkey[3,3,6] >= 1
 R1879: sbox[0,0,7] + sbox[0,1,7] + sbox[0,2,7] + sbox[0,3,7] + sbox[1,0,7]
   + sbox[1,1,7] + sbox[1,2,7] + sbox[1,3,7] + sbox[2,0,7] + sbox[2,1,7]
   + sbox[2,2,7] + sbox[2,3,7] + sbox[3,0,7] + sbox[3,1,7] + sbox[3,2,7]
   + sbox[3,3,7] + tkey[0,0,7] + tkey[0,1,7] + tkey[0,2,7] + tkey[0,3,7]
   + tkey[1,0,7] + tkey[1,1,7] + tkey[1,2,7] + tkey[1,3,7] + tkey[2,0,7]
   + tkey[2,1,7] + tkey[2,2,7] + tkey[2,3,7] + tkey[3,0,7] + tkey[3,1,7]
   + tkey[3,2,7] + tkey[3,3,7] >= 1
 R1880: sbox[0,0,0] + sbox[0,1,0] + sbox[0,2,0] + sbox[0,3,0] + sbox[1,0,0]
   + sbox[1,1,0] + sbox[1,2,0] + sbox[1,3,0] + sbox[2,0,0] + sbox[2,1,0]
   + sbox[2,2,0] + sbox[2,3,0] + sbox[3,0,0] + sbox[3,1,0] + sbox[3,2,0]
   + sbox[3,3,0] = 0
 R1881: sbox[0,0,8] + sbox[0,1,8] + sbox[0,2,8] + sbox[0,3,8] + sbox[1,0,8]
   + sbox[1,1,8] + sbox[1,2,8] + sbox[1,3,8] + sbox[2,0,8] + sbox[2,1,8]
   + sbox[2,2,8] + sbox[2,3,8] + sbox[3,0,8] + sbox[3,1,8] + sbox[3,2,8]
   + sbox[3,3,8] = 0
 R1882: tbox[0,0,0] = 1
 R1883: unkn[0,0,0] - forg[0,0,0] = 0
 R1884: tbox[0,1,0] = 0
 R1885: unkn[0,1,0] - forg[0,1,0] = 0
 R1886: tbox[0,2,0] = 0
 R1887: unkn[0,2,0] - forg[0,2,0] = 0
 R1888: tbox[0,3,0] = 0
 R1889: unkn[0,3,0] - forg[0,3,0] = 0
 R1890: tbox[1,0,0] = 0
 R1891: unkn[1,0,0] - forg[1,0,0] = 0
 R1892: tbox[1,1,0] = 0
 R1893: unkn[1,1,0] - forg[1,1,0] = 0
 R1894: tbox[1,2,0] = 0
 R1895: unkn[1,2,0] - forg[1,2,0] = 0
 R1896: tbox[1,3,0] = 0
 R1897: unkn[1,3,0] - forg[1,3,0] = 0
 R1898: tbox[2,0,0] = 0
 R1899: unkn[2,0,0] - forg[2,0,0] = 0
 R1900: tbox[2,1,0] = 0
 R1901: unkn[2,1,0] - forg[2,1,0] = 0
 R1902: tbox[2,2,0] = 0
 R1903: unkn[2,2,0] - forg[2,2,0] = 0
 R1904: tbox[2,3,0] = 0
 R1905: unkn[2,3,0] - forg[2,3,0] = 0
 R1906: tbox[3,0,0] = 0
 R1907: unkn[3,0,0] - forg[3,0,0] = 0
 R1908: tbox[3,1,0] = 0
 R1909: unkn[3,1,0] - forg[3,1,0] = 0
 R1910: tbox[3,2,0] = 0
 R1911: unkn[3,2,0] - forg[3,2,0] = 0
 R1912: tbox[3,3,0] = 0
 R1913: unkn[3,3,0] - forg[3,3,0] = 0
Bounds
 cost[0,0,0] <= 2
 cost[0,0,1] <= 2
 cost[0,0,2] <= 2
 cost[0,0,3] <= 2
 cost[0,0,4] <= 2
 cost[0,0,5] <= 2
 cost[0,0,6] <= 2
 cost[0,0,7] <= 2
 cost[0,0,8] <= 2
 cost[0,1,0] <= 2
 cost[0,1,1] <= 2
 cost[0,1,2] <= 2
 cost[0,1,3] <= 2
 cost[0,1,4] <= 2
 cost[0,1,5] <= 2
 cost[0,1,6] <= 2
 cost[0,1,7] <= 2
 cost[0,1,8] <= 2
 cost[0,2,0] <= 2
 cost[0,2,1] <= 2
 cost[0,2,2] <= 2
 cost[0,2,3] <= 2
 cost[0,2,4] <= 2
 cost[0,2,5] <= 2
 cost[0,2,6] <= 2
 cost[0,2,7] <= 2
 cost[0,2,8] <= 2
 cost[0,3,0] <= 2
 cost[0,3,1] <= 2
 cost[0,3,2] <= 2
 cost[0,3,3] <= 2
 cost[0,3,4] <= 2
 cost[0,3,5] <= 2
 cost[0,3,6] <= 2
 cost[0,3,7] <= 2
 cost[0,3,8] <= 2
 cost[1,0,0] <= 2
 cost[1,0,1] <= 2
 cost[1,0,2] <= 2
 cost[1,0,3] <= 2
 cost[1,0,4] <= 2
 cost[1,0,5] <= 2
 cost[1,0,6] <= 2
 cost[1,0,7] <= 2
 cost[1,0,8] <= 2
 cost[1,1,0] <= 2
 cost[1,1,1] <= 2
 cost[1,1,2] <= 2
 cost[1,1,3] <= 2
 cost[1,1,4] <= 2
 cost[1,1,5] <= 2
 cost[1,1,6] <= 2
 cost[1,1,7] <= 2
 cost[1,1,8] <= 2
 cost[1,2,0] <= 2
 cost[1,2,1] <= 2
 cost[1,2,2] <= 2
 cost[1,2,3] <= 2
 cost[1,2,4] <= 2
 cost[1,2,5] <= 2
 cost[1,2,6] <= 2
 cost[1,2,7] <= 2
 cost[1,2,8] <= 2
 cost[1,3,0] <= 2
 cost[1,3,1] <= 2
 cost[1,3,2] <= 2
 cost[1,3,3] <= 2
 cost[1,3,4] <= 2
 cost[1,3,5] <= 2
 cost[1,3,6] <= 2
 cost[1,3,7] <= 2
 cost[1,3,8] <= 2
 cost[2,0,0] <= 2
 cost[2,0,1] <= 2
 cost[2,0,2] <= 2
 cost[2,0,3] <= 2
 cost[2,0,4] <= 2
 cost[2,0,5] <= 2
 cost[2,0,6] <= 2
 cost[2,0,7] <= 2
 cost[2,0,8] <= 2
 cost[2,1,0] <= 2
 cost[2,1,1] <= 2
 cost[2,1,2] <= 2
 cost[2,1,3] <= 2
 cost[2,1,4] <= 2
 cost[2,1,5] <= 2
 cost[2,1,6] <= 2
 cost[2,1,7] <= 2
 cost[2,1,8] <= 2
 cost[2,2,0] <= 2
 cost[2,2,1] <= 2
 cost[2,2,2] <= 2
 cost[2,2,3] <= 2
 cost[2,2,4] <= 2
 cost[2,2,5] <= 2
 cost[2,2,6] <= 2
 cost[2,2,7] <= 2
 cost[2,2,8] <= 2
 cost[2,3,0] <= 2
 cost[2,3,1] <= 2
 cost[2,3,2] <= 2
 cost[2,3,3] <= 2
 cost[2,3,4] <= 2
 cost[2,3,5] <= 2
 cost[2,3,6] <= 2
 cost[2,3,7] <= 2
 cost[2,3,8] <= 2
 cost[3,0,0] <= 2
 cost[3,0,1] <= 2
 cost[3,0,2] <= 2
 cost[3,0,3] <= 2
 cost[3,0,4] <= 2
 cost[3,0,5] <= 2
 cost[3,0,6] <= 2
 cost[3,0,7] <= 2
 cost[3,0,8] <= 2
 cost[3,1,0] <= 2
 cost[3,1,1] <= 2
 cost[3,1,2] <= 2
 cost[3,1,3] <= 2
 cost[3,1,4] <= 2
 cost[3,1,5] <= 2
 cost[3,1,6] <= 2
 cost[3,1,7] <= 2
 cost[3,1,8] <= 2
 cost[3,2,0] <= 2
 cost[3,2,1] <= 2
 cost[3,2,2] <= 2
 cost[3,2,3] <= 2
 cost[3,2,4] <= 2
 cost[3,2,5] <= 2
 cost[3,2,6] <= 2
 cost[3,2,7] <= 2
 cost[3,2,8] <= 2
 cost[3,3,0] <= 2
 cost[3,3,1] <= 2
 cost[3,3,2] <= 2
 cost[3,3,3] <= 2
 cost[3,3,4] <= 2
 cost[3,3,5] <= 2
 cost[3,3,6] <= 2
 cost[3,3,7] <= 2
 cost[3,3,8] <= 2
Binaries
 sbox[0,0,0] sbox[0,0,1] sbox[0,0,2] sbox[0,0,3] sbox[0,0,4] sbox[0,0,5]
 sbox[0,0,6] sbox[0,0,7] sbox[0,0,8] sbox[0,1,0] sbox[0,1,1] sbox[0,1,2]
 sbox[0,1,3] sbox[0,1,4] sbox[0,1,5] sbox[0,1,6] sbox[0,1,7] sbox[0,1,8]
 sbox[0,2,0] sbox[0,2,1] sbox[0,2,2] sbox[0,2,3] sbox[0,2,4] sbox[0,2,5]
 sbox[0,2,6] sbox[0,2,7] sbox[0,2,8] sbox[0,3,0] sbox[0,3,1] sbox[0,3,2]
 sbox[0,3,3] sbox[0,3,4] sbox[0,3,5] sbox[0,3,6] sbox[0,3,7] sbox[0,3,8]
 sbox[1,0,0] sbox[1,0,1] sbox[1,0,2] sbox[1,0,3] sbox[1,0,4] sbox[1,0,5]
 sbox[1,0,6] sbox[1,0,7] sbox[1,0,8] sbox[1,1,0] sbox[1,1,1] sbox[1,1,2]
 sbox[1,1,3] sbox[1,1,4] sbox[1,1,5] sbox[1,1,6] sbox[1,1,7] sbox[1,1,8]
 sbox[1,2,0] sbox[1,2,1] sbox[1,2,2] sbox[1,2,3] sbox[1,2,4] sbox[1,2,5]
 sbox[1,2,6] sbox[1,2,7] sbox[1,2,8] sbox[1,3,0] sbox[1,3,1] sbox[1,3,2]
 sbox[1,3,3] sbox[1,3,4] sbox[1,3,5] sbox[1,3,6] sbox[1,3,7] sbox[1,3,8]
 sbox[2,0,0] sbox[2,0,1] sbox[2,0,2] sbox[2,0,3] sbox[2,0,4] sbox[2,0,5]
 sbox[2,0,6] sbox[2,0,7] sbox[2,0,8] sbox[2,1,0] sbox[2,1,1] sbox[2,1,2]
 sbox[2,1,3] sbox[2,1,4] sbox[2,1,5] sbox[2,1,6] sbox[2,1,7] sbox[2,1,8]
 sbox[2,2,0] sbox[2,2,1] sbox[2,2,2] sbox[2,2,3] sbox[2,2,4] sbox[2,2,5]
 sbox[2,2,6] sbox[2,2,7] sbox[2,2,8] sbox[2,3,0] sbox[2,3,1] sbox[2,3,2]
 sbox[2,3,3] sbox[2,3,4] sbox[2,3,5] sbox[2,3,6] sbox[2,3,7] sbox[2,3,8]
 sbox[3,0,0] sbox[3,0,1] sbox[3,0,2] sbox[3,0,3] sbox[3,0,4] sbox[3,0,5]
 sbox[3,0,6] sbox[3,0,7] sbox[3,0,8] sbox[3,1,0] sbox[3,1,1] sbox[3,1,2]
 sbox[3,1,3] sbox[3,1,4] sbox[3,1,5] sbox[3,1,6] sbox[3,1,7] sbox[3,1,8]
 sbox[3,2,0] sbox[3,2,1] sbox[3,2,2] sbox[3,2,3] sbox[3,2,4] sbox[3,2,5]
 sbox[3,2,6] sbox[3,2,7] sbox[3,2,8] sbox[3,3,0] sbox[3,3,1] sbox[3,3,2]
 sbox[3,3,3] sbox[3,3,4] sbox[3,3,5] sbox[3,3,6] sbox[3,3,7] sbox[3,3,8]
 rkey[0,0,0] rkey[0,0,1] rkey[0,0,2] rkey[0,0,3] rkey[0,0,4] rkey[0,0,5]
 rkey[0,0,6] rkey[0,0,7] rkey[0,0,8] rkey[0,1,0] rkey[0,1,1] rkey[0,1,2]
 rkey[0,1,3] rkey[0,1,4] rkey[0,1,5] rkey[0,1,6] rkey[0,1,7] rkey[0,1,8]
 rkey[0,2,0] rkey[0,2,1] rkey[0,2,2] rkey[0,2,3] rkey[0,2,4] rkey[0,2,5]
 rkey[0,2,6] rkey[0,2,7] rkey[0,2,8] rkey[0,3,0] rkey[0,3,1] rkey[0,3,2]
 rkey[0,3,3] rkey[0,3,4] rkey[0,3,5] rkey[0,3,6] rkey[0,3,7] rkey[0,3,8]
 rkey[1,0,0] rkey[1,0,1] rkey[1,0,2] rkey[1,0,3] rkey[1,0,4] rkey[1,0,5]
 rkey[1,0,6] rkey[1,0,7] rkey[1,0,8] rkey[1,1,0] rkey[1,1,1] rkey[1,1,2]
 rkey[1,1,3] rkey[1,1,4] rkey[1,1,5] rkey[1,1,6] rkey[1,1,7] rkey[1,1,8]
 rkey[1,2,0] rkey[1,2,1] rkey[1,2,2] rkey[1,2,3] rkey[1,2,4] rkey[1,2,5]
 rkey[1,2,6] rkey[1,2,7] rkey[1,2,8] rkey[1,3,0] rkey[1,3,1] rkey[1,3,2]
 rkey[1,3,3] rkey[1,3,4] rkey[1,3,5] rkey[1,3,6] rkey[1,3,7] rkey[1,3,8]
 rkey[2,0,0] rkey[2,0,1] rkey[2,0,2] rkey[2,0,3] rkey[2,0,4] rkey[2,0,5]
 rkey[2,0,6] rkey[2,0,7] rkey[2,0,8] rkey[2,1,0] rkey[2,1,1] rkey[2,1,2]
 rkey[2,1,3] rkey[2,1,4] rkey[2,1,5] rkey[2,1,6] rkey[2,1,7] rkey[2,1,8]
 rkey[2,2,0] rkey[2,2,1] rkey[2,2,2] rkey[2,2,3] rkey[2,2,4] rkey[2,2,5]
 rkey[2,2,6] rkey[2,2,7] rkey[2,2,8] rkey[2,3,0] rkey[2,3,1] rkey[2,3,2]
 rkey[2,3,3] rkey[2,3,4] rkey[2,3,5] rkey[2,3,6] rkey[2,3,7] rkey[2,3,8]
 rkey[3,0,0] rkey[3,0,1] rkey[3,0,2] rkey[3,0,3] rkey[3,0,4] rkey[3,0,5]
 rkey[3,0,6] rkey[3,0,7] rkey[3,0,8] rkey[3,1,0] rkey[3,1,1] rkey[3,1,2]
 rkey[3,1,3] rkey[3,1,4] rkey[3,1,5] rkey[3,1,6] rkey[3,1,7] rkey[3,1,8]
 rkey[3,2,0] rkey[3,2,1] rkey[3,2,2] rkey[3,2,3] rkey[3,2,4] rkey[3,2,5]
 rkey[3,2,6] rkey[3,2,7] rkey[3,2,8] rkey[3,3,0] rkey[3,3,1] rkey[3,3,2]
 rkey[3,3,3] rkey[3,3,4] rkey[3,3,5] rkey[3,3,6] rkey[3,3,7] rkey[3,3,8]
 tkey[0,0,0] tkey[0,0,1] tkey[0,0,2] tkey[0,0,3] tkey[0,0,4] tkey[0,0,5]
 tkey[0,0,6] tkey[0,0,7] tkey[0,0,8] tkey[0,1,0] tkey[0,1,1] tkey[0,1,2]
 tkey[0,1,3] tkey[0,1,4] tkey[0,1,5] tkey[0,1,6] tkey[0,1,7] tkey[0,1,8]
 tkey[0,2,0] tkey[0,2,1] tkey[0,2,2] tkey[0,2,3] tkey[0,2,4] tkey[0,2,5]
 tkey[0,2,6] tkey[0,2,7] tkey[0,2,8] tkey[0,3,0] tkey[0,3,1] tkey[0,3,2]
 tkey[0,3,3] tkey[0,3,4] tkey[0,3,5] tkey[0,3,6] tkey[0,3,7] tkey[0,3,8]
 tkey[1,0,0] tkey[1,0,1] tkey[1,0,2] tkey[1,0,3] tkey[1,0,4] tkey[1,0,5]
 tkey[1,0,6] tkey[1,0,7] tkey[1,0,8] tkey[1,1,0] tkey[1,1,1] tkey[1,1,2]
 tkey[1,1,3] tkey[1,1,4] tkey[1,1,5] tkey[1,1,6] tkey[1,1,7] tkey[1,1,8]
 tkey[1,2,0] tkey[1,2,1] tkey[1,2,2] tkey[1,2,3] tkey[1,2,4] tkey[1,2,5]
 tkey[1,2,6] tkey[1,2,7] tkey[1,2,8] tkey[1,3,0] tkey[1,3,1] tkey[1,3,2]
 tkey[1,3,3] tkey[1,3,4] tkey[1,3,5] tkey[1,3,6] tkey[1,3,7] tkey[1,3,8]
 tkey[2,0,0] tkey[2,0,1] tkey[2,0,2] tkey[2,0,3] tkey[2,0,4] tkey[2,0,5]
 tkey[2,0,6] tkey[2,0,7] tkey[2,0,8] tkey[2,1,0] tkey[2,1,1] tkey[2,1,2]
 tkey[2,1,3] tkey[2,1,4] tkey[2,1,5] tkey[2,1,6] tkey[2,1,7] tkey[2,1,8]
 tkey[2,2,0] tkey[2,2,1] tkey[2,2,2] tkey[2,2,3] tkey[2,2,4] tkey[2,2,5]
 tkey[2,2,6] tkey[2,2,7] tkey[2,2,8] tkey[2,3,0] tkey[2,3,1] tkey[2,3,2]
 tkey[2,3,3] tkey[2,3,4] tkey[2,3,5] tkey[2,3,6] tkey[2,3,7] tkey[2,3,8]
 tkey[3,0,0] tkey[3,0,1] tkey[3,0,2] tkey[3,0,3] tkey[3,0,4] tkey[3,0,5]
 tkey[3,0,6] tkey[3,0,7] tkey[3,0,8] tkey[3,1,0] tkey[3,1,1] tkey[3,1,2]
 tkey[3,1,3] tkey[3,1,4] tkey[3,1,5] tkey[3,1,6] tkey[3,1,7] tkey[3,1,8]
 tkey[3,2,0] tkey[3,2,1] tkey[3,2,2] tkey[3,2,3] tkey[3,2,4] tkey[3,2,5]
 tkey[3,2,6] tkey[3,2,7] tkey[3,2,8] tkey[3,3,0] tkey[3,3,1] tkey[3,3,2]
 tkey[3,3,3] tkey[3,3,4] tkey[3,3,5] tkey[3,3,6] tkey[3,3,7] tkey[3,3,8]
 tbox[0,0,0] tbox[0,0,1] tbox[0,0,2] tbox[0,0,3] tbox[0,0,4] tbox[0,0,5]
 tbox[0,0,6] tbox[0,0,7] tbox[0,0,8] tbox[0,1,0] tbox[0,1,1] tbox[0,1,2]
 tbox[0,1,3] tbox[0,1,4] tbox[0,1,5] tbox[0,1,6] tbox[0,1,7] tbox[0,1,8]
 tbox[0,2,0] tbox[0,2,1] tbox[0,2,2] tbox[0,2,3] tbox[0,2,4] tbox[0,2,5]
 tbox[0,2,6] tbox[0,2,7] tbox[0,2,8] tbox[0,3,0] tbox[0,3,1] tbox[0,3,2]
 tbox[0,3,3] tbox[0,3,4] tbox[0,3,5] tbox[0,3,6] tbox[0,3,7] tbox[0,3,8]
 tbox[1,0,0] tbox[1,0,1] tbox[1,0,2] tbox[1,0,3] tbox[1,0,4] tbox[1,0,5]
 tbox[1,0,6] tbox[1,0,7] tbox[1,0,8] tbox[1,1,0] tbox[1,1,1] tbox[1,1,2]
 tbox[1,1,3] tbox[1,1,4] tbox[1,1,5] tbox[1,1,6] tbox[1,1,7] tbox[1,1,8]
 tbox[1,2,0] tbox[1,2,1] tbox[1,2,2] tbox[1,2,3] tbox[1,2,4] tbox[1,2,5]
 tbox[1,2,6] tbox[1,2,7] tbox[1,2,8] tbox[1,3,0] tbox[1,3,1] tbox[1,3,2]
 tbox[1,3,3] tbox[1,3,4] tbox[1,3,5] tbox[1,3,6] tbox[1,3,7] tbox[1,3,8]
 tbox[2,0,0] tbox[2,0,1] tbox[2,0,2] tbox[2,0,3] tbox[2,0,4] tbox[2,0,5]
 tbox[2,0,6] tbox[2,0,7] tbox[2,0,8] tbox[2,1,0] tbox[2,1,1] tbox[2,1,2]
 tbox[2,1,3] tbox[2,1,4] tbox[2,1,5] tbox[2,1,6] tbox[2,1,7] tbox[2,1,8]
 tbox[2,2,0] tbox[2,2,1] tbox[2,2,2] tbox[2,2,3] tbox[2,2,4] tbox[2,2,5]
 tbox[2,2,6] tbox[2,2,7] tbox[2,2,8] tbox[2,3,0] tbox[2,3,1] tbox[2,3,2]
 tbox[2,3,3] tbox[2,3,4] tbox[2,3,5] tbox[2,3,6] tbox[2,3,7] tbox[2,3,8]
 tbox[3,0,0] tbox[3,0,1] tbox[3,0,2] tbox[3,0,3] tbox[3,0,4] tbox[3,0,5]
 tbox[3,0,6] tbox[3,0,7] tbox[3,0,8] tbox[3,1,0] tbox[3,1,1] tbox[3,1,2]
 tbox[3,1,3] tbox[3,1,4] tbox[3,1,5] tbox[3,1,6] tbox[3,1,7] tbox[3,1,8]
 tbox[3,2,0] tbox[3,2,1] tbox[3,2,2] tbox[3,2,3] tbox[3,2,4] tbox[3,2,5]
 tbox[3,2,6] tbox[3,2,7] tbox[3,2,8] tbox[3,3,0] tbox[3,3,1] tbox[3,3,2]
 tbox[3,3,3] tbox[3,3,4] tbox[3,3,5] tbox[3,3,6] tbox[3,3,7] tbox[3,3,8]
 unkn[0,0,0] unkn[0,0,1] unkn[0,0,2] unkn[0,0,3] unkn[0,0,4] unkn[0,0,5]
 unkn[0,0,6] unkn[0,0,7] unkn[0,0,8] unkn[0,1,0] unkn[0,1,1] unkn[0,1,2]
 unkn[0,1,3] unkn[0,1,4] unkn[0,1,5] unkn[0,1,6] unkn[0,1,7] unkn[0,1,8]
 unkn[0,2,0] unkn[0,2,1] unkn[0,2,2] unkn[0,2,3] unkn[0,2,4] unkn[0,2,5]
 unkn[0,2,6] unkn[0,2,7] unkn[0,2,8] unkn[0,3,0] unkn[0,3,1] unkn[0,3,2]
 unkn[0,3,3] unkn[0,3,4] unkn[0,3,5] unkn[0,3,6] unkn[0,3,7] unkn[0,3,8]
 unkn[1,0,0] unkn[1,0,1] unkn[1,0,2] unkn[1,0,3] unkn[1,0,4] unkn[1,0,5]
 unkn[1,0,6] unkn[1,0,7] unkn[1,0,8] unkn[1,1,0] unkn[1,1,1] unkn[1,1,2]
 unkn[1,1,3] unkn[1,1,4] unkn[1,1,5] unkn[1,1,6] unkn[1,1,7] unkn[1,1,8]
 unkn[1,2,0] unkn[1,2,1] unkn[1,2,2] unkn[1,2,3] unkn[1,2,4] unkn[1,2,5]
 unkn[1,2,6] unkn[1,2,7] unkn[1,2,8] unkn[1,3,0] unkn[1,3,1] unkn[1,3,2]
 unkn[1,3,3] unkn[1,3,4] unkn[1,3,5] unkn[1,3,6] unkn[1,3,7] unkn[1,3,8]
 unkn[2,0,0] unkn[2,0,1] unkn[2,0,2] unkn[2,0,3] unkn[2,0,4] unkn[2,0,5]
 unkn[2,0,6] unkn[2,0,7] unkn[2,0,8] unkn[2,1,0] unkn[2,1,1] unkn[2,1,2]
 unkn[2,1,3] unkn[2,1,4] unkn[2,1,5] unkn[2,1,6] unkn[2,1,7] unkn[2,1,8]
 unkn[2,2,0] unkn[2,2,1] unkn[2,2,2] unkn[2,2,3] unkn[2,2,4] unkn[2,2,5]
 unkn[2,2,6] unkn[2,2,7] unkn[2,2,8] unkn[2,3,0] unkn[2,3,1] unkn[2,3,2]
 unkn[2,3,3] unkn[2,3,4] unkn[2,3,5] unkn[2,3,6] unkn[2,3,7] unkn[2,3,8]
 unkn[3,0,0] unkn[3,0,1] unkn[3,0,2] unkn[3,0,3] unkn[3,0,4] unkn[3,0,5]
 unkn[3,0,6] unkn[3,0,7] unkn[3,0,8] unkn[3,1,0] unkn[3,1,1] unkn[3,1,2]
 unkn[3,1,3] unkn[3,1,4] unkn[3,1,5] unkn[3,1,6] unkn[3,1,7] unkn[3,1,8]
 unkn[3,2,0] unkn[3,2,1] unkn[3,2,2] unkn[3,2,3] unkn[3,2,4] unkn[3,2,5]
 unkn[3,2,6] unkn[3,2,7] unkn[3,2,8] unkn[3,3,0] unkn[3,3,1] unkn[3,3,2]
 unkn[3,3,3] unkn[3,3,4] unkn[3,3,5] unkn[3,3,6] unkn[3,3,7] unkn[3,3,8]
 forg[0,0,0] forg[0,0,1] forg[0,0,2] forg[0,0,3] forg[0,0,4] forg[0,0,5]
 forg[0,0,6] forg[0,0,7] forg[0,0,8] forg[0,1,0] forg[0,1,1] forg[0,1,2]
 forg[0,1,3] forg[0,1,4] forg[0,1,5] forg[0,1,6] forg[0,1,7] forg[0,1,8]
 forg[0,2,0] forg[0,2,1] forg[0,2,2] forg[0,2,3] forg[0,2,4] forg[0,2,5]
 forg[0,2,6] forg[0,2,7] forg[0,2,8] forg[0,3,0] forg[0,3,1] forg[0,3,2]
 forg[0,3,3] forg[0,3,4] forg[0,3,5] forg[0,3,6] forg[0,3,7] forg[0,3,8]
 forg[1,0,0] forg[1,0,1] forg[1,0,2] forg[1,0,3] forg[1,0,4] forg[1,0,5]
 forg[1,0,6] forg[1,0,7] forg[1,0,8] forg[1,1,0] forg[1,1,1] forg[1,1,2]
 forg[1,1,3] forg[1,1,4] forg[1,1,5] forg[1,1,6] forg[1,1,7] forg[1,1,8]
 forg[1,2,0] forg[1,2,1] forg[1,2,2] forg[1,2,3] forg[1,2,4] forg[1,2,5]
 forg[1,2,6] forg[1,2,7] forg[1,2,8] forg[1,3,0] forg[1,3,1] forg[1,3,2]
 forg[1,3,3] forg[1,3,4] forg[1,3,5] forg[1,3,6] forg[1,3,7] forg[1,3,8]
 forg[2,0,0] forg[2,0,1] forg[2,0,2] forg[2,0,3] forg[2,0,4] forg[2,0,5]
 forg[2,0,6] forg[2,0,7] forg[2,0,8] forg[2,1,0] forg[2,1,1] forg[2,1,2]
 forg[2,1,3] forg[2,1,4] forg[2,1,5] forg[2,1,6] forg[2,1,7] forg[2,1,8]
 forg[2,2,0] forg[2,2,1] forg[2,2,2] forg[2,2,3] forg[2,2,4] forg[2,2,5]
 forg[2,2,6] forg[2,2,7] forg[2,2,8] forg[2,3,0] forg[2,3,1] forg[2,3,2]
 forg[2,3,3] forg[2,3,4] forg[2,3,5] forg[2,3,6] forg[2,3,7] forg[2,3,8]
 forg[3,0,0] forg[3,0,1] forg[3,0,2] forg[3,0,3] forg[3,0,4] forg[3,0,5]
 forg[3,0,6] forg[3,0,7] forg[3,0,8] forg[3,1,0] forg[3,1,1] forg[3,1,2]
 forg[3,1,3] forg[3,1,4] forg[3,1,5] forg[3,1,6] forg[3,1,7] forg[3,1,8]
 forg[3,2,0] forg[3,2,1] forg[3,2,2] forg[3,2,3] forg[3,2,4] forg[3,2,5]
 forg[3,2,6] forg[3,2,7] forg[3,2,8] forg[3,3,0] forg[3,3,1] forg[3,3,2]
 forg[3,3,3] forg[3,3,4] forg[3,3,5] forg[3,3,6] forg[3,3,7] forg[3,3,8]
 lane[0,0] lane[0,1] lane[0,2] lane[0,3] lane[1,0] lane[1,1] lane[1,2]
 lane[1,3] lane[2,0] lane[2,1] lane[2,2] lane[2,3] lane[3,0] lane[3,1]
 lane[3,2] lane[3,3]
Generals
 cost[0,0,0] cost[0,0,1] cost[0,0,2] cost[0,0,3] cost[0,0,4] cost[0,0,5]
 cost[0,0,6] cost[0,0,7] cost[0,0,8] cost[0,1,0] cost[0,1,1] cost[0,1,2]
 cost[0,1,3] cost[0,1,4] cost[0,1,5] cost[0,1,6] cost[0,1,7] cost[0,1,8]
 cost[0,2,0] cost[0,2,1] cost[0,2,2] cost[0,2,3] cost[0,2,4] cost[0,2,5]
 cost[0,2,6] cost[0,2,7] cost[0,2,8] cost[0,3,0] cost[0,3,1] cost[0,3,2]
 cost[0,3,3] cost[0,3,4] cost[0,3,5] cost[0,3,6] cost[0,3,7] cost[0,3,8]
 cost[1,0,0] cost[1,0,1] cost[1,0,2] cost[1,0,3] cost[1,0,4] cost[1,0,5]
 cost[1,0,6] cost[1,0,7] cost[1,0,8] cost[1,1,0] cost[1,1,1] cost[1,1,2]
 cost[1,1,3] cost[1,1,4] cost[1,1,5] cost[1,1,6] cost[1,1,7] cost[1,1,8]
 cost[1,2,0] cost[1,2,1] cost[1,2,2] cost[1,2,3] cost[1,2,4] cost[1,2,5]
 cost[1,2,6] cost[1,2,7] cost[1,2,8] cost[1,3,0] cost[1,3,1] cost[1,3,2]
 cost[1,3,3] cost[1,3,4] cost[1,3,5] cost[1,3,6] cost[1,3,7] cost[1,3,8]
 cost[2,0,0] cost[2,0,1] cost[2,0,2] cost[2,0,3] cost[2,0,4] cost[2,0,5]
 cost[2,0,6] cost[2,0,7] cost[2,0,8] cost[2,1,0] cost[2,1,1] cost[2,1,2]
 cost[2,1,3] cost[2,1,4] cost[2,1,5] cost[2,1,6] cost[2,1,7] cost[2,1,8]
 cost[2,2,0] cost[2,2,1] cost[2,2,2] cost[2,2,3] cost[2,2,4] cost[2,2,5]
 cost[2,2,6] cost[2,2,7] cost[2,2,8] cost[2,3,0] cost[2,3,1] cost[2,3,2]
 cost[2,3,3] cost[2,3,4] cost[2,3,5] cost[2,3,6] cost[2,3,7] cost[2,3,8]
 cost[3,0,0] cost[3,0,1] cost[3,0,2] cost[3,0,3] cost[3,0,4] cost[3,0,5]
 cost[3,0,6] cost[3,0,7] cost[3,0,8] cost[3,1,0] cost[3,1,1] cost[3,1,2]
 cost[3,1,3] cost[3,1,4] cost[3,1,5] cost[3,1,6] cost[3,1,7] cost[3,1,8]
 cost[3,2,0] cost[3,2,1] cost[3,2,2] cost[3,2,3] cost[3,2,4] cost[3,2,5]
 cost[3,2,6] cost[3,2,7] cost[3,2,8] cost[3,3,0] cost[3,3,1] cost[3,3,2]
 cost[3,3,3] cost[3,3,4] cost[3,3,5] cost[3,3,6] cost[3,3,7] cost[3,3,8]
End
