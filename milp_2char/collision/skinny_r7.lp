\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 rkey[0,0,7] + 0 rkey[0,1,7] + 0 rkey[0,2,7] + 0 rkey[0,3,7]
   + 0 rkey[1,0,7] + 0 rkey[1,1,7] + 0 rkey[1,2,7] + 0 rkey[1,3,7]
   + 0 rkey[2,0,7] + 0 rkey[2,1,7] + 0 rkey[2,2,7] + 0 rkey[2,3,7]
   + 0 rkey[3,0,7] + 0 rkey[3,1,7] + 0 rkey[3,2,7] + 0 rkey[3,3,7]
   + 0 tkey[0,0,7] + 0 tkey[0,1,7] + 0 tkey[0,2,7] + 0 tkey[0,3,7]
   + 0 tkey[1,0,7] + 0 tkey[1,1,7] + 0 tkey[1,2,7] + 0 tkey[1,3,7]
   + 0 tkey[2,0,7] + 0 tkey[2,1,7] + 0 tkey[2,2,7] + 0 tkey[2,3,7]
   + 0 tkey[3,0,7] + 0 tkey[3,1,7] + 0 tkey[3,2,7] + 0 tkey[3,3,7]
   + cost[0,0,0] + cost[0,0,1] + cost[0,0,2] + cost[0,0,3] + cost[0,0,4]
   + cost[0,0,5] + cost[0,0,6] + 0 cost[0,0,7] + cost[0,1,0] + cost[0,1,1]
   + cost[0,1,2] + cost[0,1,3] + cost[0,1,4] + cost[0,1,5] + cost[0,1,6]
   + 0 cost[0,1,7] + cost[0,2,0] + cost[0,2,1] + cost[0,2,2] + cost[0,2,3]
   + cost[0,2,4] + cost[0,2,5] + cost[0,2,6] + 0 cost[0,2,7] + cost[0,3,0]
   + cost[0,3,1] + cost[0,3,2] + cost[0,3,3] + cost[0,3,4] + cost[0,3,5]
   + cost[0,3,6] + 0 cost[0,3,7] + cost[1,0,0] + cost[1,0,1] + cost[1,0,2]
   + cost[1,0,3] + cost[1,0,4] + cost[1,0,5] + cost[1,0,6] + 0 cost[1,0,7]
   + cost[1,1,0] + cost[1,1,1] + cost[1,1,2] + cost[1,1,3] + cost[1,1,4]
   + cost[1,1,5] + cost[1,1,6] + 0 cost[1,1,7] + cost[1,2,0] + cost[1,2,1]
   + cost[1,2,2] + cost[1,2,3] + cost[1,2,4] + cost[1,2,5] + cost[1,2,6]
   + 0 cost[1,2,7] + cost[1,3,0] + cost[1,3,1] + cost[1,3,2] + cost[1,3,3]
   + cost[1,3,4] + cost[1,3,5] + cost[1,3,6] + 0 cost[1,3,7] + cost[2,0,0]
   + cost[2,0,1] + cost[2,0,2] + cost[2,0,3] + cost[2,0,4] + cost[2,0,5]
   + cost[2,0,6] + 0 cost[2,0,7] + cost[2,1,0] + cost[2,1,1] + cost[2,1,2]
   + cost[2,1,3] + cost[2,1,4] + cost[2,1,5] + cost[2,1,6] + 0 cost[2,1,7]
   + cost[2,2,0] + cost[2,2,1] + cost[2,2,2] + cost[2,2,3] + cost[2,2,4]
   + cost[2,2,5] + cost[2,2,6] + 0 cost[2,2,7] + cost[2,3,0] + cost[2,3,1]
   + cost[2,3,2] + cost[2,3,3] + cost[2,3,4] + cost[2,3,5] + cost[2,3,6]
   + 0 cost[2,3,7] + cost[3,0,0] + cost[3,0,1] + cost[3,0,2] + cost[3,0,3]
   + cost[3,0,4] + cost[3,0,5] + cost[3,0,6] + 0 cost[3,0,7] + cost[3,1,0]
   + cost[3,1,1] + cost[3,1,2] + cost[3,1,3] + cost[3,1,4] + cost[3,1,5]
   + cost[3,1,6] + 0 cost[3,1,7] + cost[3,2,0] + cost[3,2,1] + cost[3,2,2]
   + cost[3,2,3] + cost[3,2,4] + cost[3,2,5] + cost[3,2,6] + 0 cost[3,2,7]
   + cost[3,3,0] + cost[3,3,1] + cost[3,3,2] + cost[3,3,3] + cost[3,3,4]
   + cost[3,3,5] + cost[3,3,6] + 0 cost[3,3,7]
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
 R336: tkey[0,0,0] - lane[0,0] <= 0
 R337: tkey[2,0,1] - lane[0,0] <= 0
 R338: tkey[0,2,2] - lane[0,0] <= 0
 R339: tkey[2,2,3] - lane[0,0] <= 0
 R340: tkey[1,0,4] - lane[0,0] <= 0
 R341: tkey[3,0,5] - lane[0,0] <= 0
 R342: tkey[1,2,6] - lane[0,0] <= 0
 R343: tkey[0,1,0] - lane[0,1] <= 0
 R344: tkey[2,1,1] - lane[0,1] <= 0
 R345: tkey[0,0,2] - lane[0,1] <= 0
 R346: tkey[2,0,3] - lane[0,1] <= 0
 R347: tkey[0,2,4] - lane[0,1] <= 0
 R348: tkey[2,2,5] - lane[0,1] <= 0
 R349: tkey[1,0,6] - lane[0,1] <= 0
 R350: tkey[0,2,0] - lane[0,2] <= 0
 R351: tkey[2,2,1] - lane[0,2] <= 0
 R352: tkey[1,0,2] - lane[0,2] <= 0
 R353: tkey[3,0,3] - lane[0,2] <= 0
 R354: tkey[1,2,4] - lane[0,2] <= 0
 R355: tkey[3,2,5] - lane[0,2] <= 0
 R356: tkey[1,1,6] - lane[0,2] <= 0
 R357: tkey[0,3,0] - lane[0,3] <= 0
 R358: tkey[2,3,1] - lane[0,3] <= 0
 R359: tkey[1,3,2] - lane[0,3] <= 0
 R360: tkey[3,3,3] - lane[0,3] <= 0
 R361: tkey[0,1,4] - lane[0,3] <= 0
 R362: tkey[2,1,5] - lane[0,3] <= 0
 R363: tkey[0,0,6] - lane[0,3] <= 0
 R364: tkey[1,0,0] - lane[1,0] <= 0
 R365: tkey[3,0,1] - lane[1,0] <= 0
 R366: tkey[1,2,2] - lane[1,0] <= 0
 R367: tkey[3,2,3] - lane[1,0] <= 0
 R368: tkey[1,1,4] - lane[1,0] <= 0
 R369: tkey[3,1,5] - lane[1,0] <= 0
 R370: tkey[0,3,6] - lane[1,0] <= 0
 R371: tkey[1,1,0] - lane[1,1] <= 0
 R372: tkey[3,1,1] - lane[1,1] <= 0
 R373: tkey[0,3,2] - lane[1,1] <= 0
 R374: tkey[2,3,3] - lane[1,1] <= 0
 R375: tkey[1,3,4] - lane[1,1] <= 0
 R376: tkey[3,3,5] - lane[1,1] <= 0
 R377: tkey[0,1,6] - lane[1,1] <= 0
 R378: tkey[1,2,0] - lane[1,2] <= 0
 R379: tkey[3,2,1] - lane[1,2] <= 0
 R380: tkey[1,1,2] - lane[1,2] <= 0
 R381: tkey[3,1,3] - lane[1,2] <= 0
 R382: tkey[0,3,4] - lane[1,2] <= 0
 R383: tkey[2,3,5] - lane[1,2] <= 0
 R384: tkey[1,3,6] - lane[1,2] <= 0
 R385: tkey[1,3,0] - lane[1,3] <= 0
 R386: tkey[3,3,1] - lane[1,3] <= 0
 R387: tkey[0,1,2] - lane[1,3] <= 0
 R388: tkey[2,1,3] - lane[1,3] <= 0
 R389: tkey[0,0,4] - lane[1,3] <= 0
 R390: tkey[2,0,5] - lane[1,3] <= 0
 R391: tkey[0,2,6] - lane[1,3] <= 0
 R392: tkey[2,0,0] - lane[2,0] <= 0
 R393: tkey[0,2,1] - lane[2,0] <= 0
 R394: tkey[2,2,2] - lane[2,0] <= 0
 R395: tkey[1,0,3] - lane[2,0] <= 0
 R396: tkey[3,0,4] - lane[2,0] <= 0
 R397: tkey[1,2,5] - lane[2,0] <= 0
 R398: tkey[3,2,6] - lane[2,0] <= 0
 R399: tkey[2,1,0] - lane[2,1] <= 0
 R400: tkey[0,0,1] - lane[2,1] <= 0
 R401: tkey[2,0,2] - lane[2,1] <= 0
 R402: tkey[0,2,3] - lane[2,1] <= 0
 R403: tkey[2,2,4] - lane[2,1] <= 0
 R404: tkey[1,0,5] - lane[2,1] <= 0
 R405: tkey[3,0,6] - lane[2,1] <= 0
 R406: tkey[2,2,0] - lane[2,2] <= 0
 R407: tkey[1,0,1] - lane[2,2] <= 0
 R408: tkey[3,0,2] - lane[2,2] <= 0
 R409: tkey[1,2,3] - lane[2,2] <= 0
 R410: tkey[3,2,4] - lane[2,2] <= 0
 R411: tkey[1,1,5] - lane[2,2] <= 0
 R412: tkey[3,1,6] - lane[2,2] <= 0
 R413: tkey[2,3,0] - lane[2,3] <= 0
 R414: tkey[1,3,1] - lane[2,3] <= 0
 R415: tkey[3,3,2] - lane[2,3] <= 0
 R416: tkey[0,1,3] - lane[2,3] <= 0
 R417: tkey[2,1,4] - lane[2,3] <= 0
 R418: tkey[0,0,5] - lane[2,3] <= 0
 R419: tkey[2,0,6] - lane[2,3] <= 0
 R420: tkey[3,0,0] - lane[3,0] <= 0
 R421: tkey[1,2,1] - lane[3,0] <= 0
 R422: tkey[3,2,2] - lane[3,0] <= 0
 R423: tkey[1,1,3] - lane[3,0] <= 0
 R424: tkey[3,1,4] - lane[3,0] <= 0
 R425: tkey[0,3,5] - lane[3,0] <= 0
 R426: tkey[2,3,6] - lane[3,0] <= 0
 R427: tkey[3,1,0] - lane[3,1] <= 0
 R428: tkey[0,3,1] - lane[3,1] <= 0
 R429: tkey[2,3,2] - lane[3,1] <= 0
 R430: tkey[1,3,3] - lane[3,1] <= 0
 R431: tkey[3,3,4] - lane[3,1] <= 0
 R432: tkey[0,1,5] - lane[3,1] <= 0
 R433: tkey[2,1,6] - lane[3,1] <= 0
 R434: tkey[3,2,0] - lane[3,2] <= 0
 R435: tkey[1,1,1] - lane[3,2] <= 0
 R436: tkey[3,1,2] - lane[3,2] <= 0
 R437: tkey[0,3,3] - lane[3,2] <= 0
 R438: tkey[2,3,4] - lane[3,2] <= 0
 R439: tkey[1,3,5] - lane[3,2] <= 0
 R440: tkey[3,3,6] - lane[3,2] <= 0
 R441: tkey[3,3,0] - lane[3,3] <= 0
 R442: tkey[0,1,1] - lane[3,3] <= 0
 R443: tkey[2,1,2] - lane[3,3] <= 0
 R444: tkey[0,0,3] - lane[3,3] <= 0
 R445: tkey[2,0,4] - lane[3,3] <= 0
 R446: tkey[0,2,5] - lane[3,3] <= 0
 R447: tkey[2,2,6] - lane[3,3] <= 0
 R448: - tkey[0,0,0] - tkey[0,2,2] - tkey[1,0,4] - tkey[1,2,6]
   - tkey[2,0,1] - tkey[2,2,3] - tkey[3,0,5] + 7 lane[0,0] <= 1
 R449: - tkey[0,0,2] - tkey[0,1,0] - tkey[0,2,4] - tkey[1,0,6]
   - tkey[2,0,3] - tkey[2,1,1] - tkey[2,2,5] + 7 lane[0,1] <= 1
 R450: - tkey[0,2,0] - tkey[1,0,2] - tkey[1,1,6] - tkey[1,2,4]
   - tkey[2,2,1] - tkey[3,0,3] - tkey[3,2,5] + 7 lane[0,2] <= 1
 R451: - tkey[0,0,6] - tkey[0,1,4] - tkey[0,3,0] - tkey[1,3,2]
   - tkey[2,1,5] - tkey[2,3,1] - tkey[3,3,3] + 7 lane[0,3] <= 1
 R452: - tkey[0,3,6] - tkey[1,0,0] - tkey[1,1,4] - tkey[1,2,2]
   - tkey[3,0,1] - tkey[3,1,5] - tkey[3,2,3] + 7 lane[1,0] <= 1
 R453: - tkey[0,1,6] - tkey[0,3,2] - tkey[1,1,0] - tkey[1,3,4]
   - tkey[2,3,3] - tkey[3,1,1] - tkey[3,3,5] + 7 lane[1,1] <= 1
 R454: - tkey[0,3,4] - tkey[1,1,2] - tkey[1,2,0] - tkey[1,3,6]
   - tkey[2,3,5] - tkey[3,1,3] - tkey[3,2,1] + 7 lane[1,2] <= 1
 R455: - tkey[0,0,4] - tkey[0,1,2] - tkey[0,2,6] - tkey[1,3,0]
   - tkey[2,0,5] - tkey[2,1,3] - tkey[3,3,1] + 7 lane[1,3] <= 1
 R456: - tkey[0,2,1] - tkey[1,0,3] - tkey[1,2,5] - tkey[2,0,0]
   - tkey[2,2,2] - tkey[3,0,4] - tkey[3,2,6] + 7 lane[2,0] <= 1
 R457: - tkey[0,0,1] - tkey[0,2,3] - tkey[1,0,5] - tkey[2,0,2]
   - tkey[2,1,0] - tkey[2,2,4] - tkey[3,0,6] + 7 lane[2,1] <= 1
 R458: - tkey[1,0,1] - tkey[1,1,5] - tkey[1,2,3] - tkey[2,2,0]
   - tkey[3,0,2] - tkey[3,1,6] - tkey[3,2,4] + 7 lane[2,2] <= 1
 R459: - tkey[0,0,5] - tkey[0,1,3] - tkey[1,3,1] - tkey[2,0,6]
   - tkey[2,1,4] - tkey[2,3,0] - tkey[3,3,2] + 7 lane[2,3] <= 1
 R460: - tkey[0,3,5] - tkey[1,1,3] - tkey[1,2,1] - tkey[2,3,6]
   - tkey[3,0,0] - tkey[3,1,4] - tkey[3,2,2] + 7 lane[3,0] <= 1
 R461: - tkey[0,1,5] - tkey[0,3,1] - tkey[1,3,3] - tkey[2,1,6]
   - tkey[2,3,2] - tkey[3,1,0] - tkey[3,3,4] + 7 lane[3,1] <= 1
 R462: - tkey[0,3,3] - tkey[1,1,1] - tkey[1,3,5] - tkey[2,3,4]
   - tkey[3,1,2] - tkey[3,2,0] - tkey[3,3,6] + 7 lane[3,2] <= 1
 R463: - tkey[0,0,3] - tkey[0,1,1] - tkey[0,2,5] - tkey[2,0,4]
   - tkey[2,1,2] - tkey[2,2,6] - tkey[3,3,0] + 7 lane[3,3] <= 1
 R464: - tbox[0,0,0] + forg[0,0,0] <= 0
 R465: sbox[0,0,0] - rkey[0,0,0] + tkey[0,0,0] >= 0
 R466: sbox[0,0,0] + rkey[0,0,0] - tkey[0,0,0] >= 0
 R467: - sbox[0,0,0] + rkey[0,0,0] + tkey[0,0,0] >= 0
 R468: - tbox[1,0,0] + forg[1,0,0] <= 0
 R469: sbox[1,0,0] - rkey[1,0,0] + tkey[1,0,0] >= 0
 R470: sbox[1,0,0] + rkey[1,0,0] - tkey[1,0,0] >= 0
 R471: - sbox[1,0,0] + rkey[1,0,0] + tkey[1,0,0] >= 0
 R472: - tbox[2,0,0] + forg[2,0,0] <= 0
 R473: - sbox[2,0,0] + rkey[2,0,0] = 0
 R474: - tbox[3,0,0] + forg[3,0,0] <= 0
 R475: - sbox[3,0,0] + rkey[3,0,0] = 0
 R476: - sbox[3,0,1] + rkey[0,0,0] + rkey[2,2,0] >= 0
 R477: sbox[3,0,1] + rkey[0,0,0] - rkey[2,2,0] >= 0
 R478: sbox[3,0,1] - rkey[0,0,0] + rkey[2,2,0] >= 0
 R479: - sbox[0,0,1] + sbox[3,0,1] + rkey[3,1,0] >= 0
 R480: sbox[0,0,1] + sbox[3,0,1] - rkey[3,1,0] >= 0
 R481: sbox[0,0,1] - sbox[3,0,1] + rkey[3,1,0] >= 0
 R482: - sbox[2,0,1] + rkey[1,3,0] + rkey[2,2,0] >= 0
 R483: sbox[2,0,1] + rkey[1,3,0] - rkey[2,2,0] >= 0
 R484: sbox[2,0,1] - rkey[1,3,0] + rkey[2,2,0] >= 0
 R485: sbox[1,0,1] - rkey[0,0,0] = 0
 R486: tbox[0,0,0] + tbox[2,2,0] - tbox[3,0,1] >= 0
 R487: tbox[0,0,0] - tbox[2,2,0] + tbox[3,0,1] >= 0
 R488: - tbox[0,0,0] + tbox[2,2,0] + tbox[3,0,1] >= 0
 R489: - tbox[0,0,1] + tbox[3,0,1] + tbox[3,1,0] >= 0
 R490: tbox[0,0,1] + tbox[3,0,1] - tbox[3,1,0] >= 0
 R491: tbox[0,0,1] - tbox[3,0,1] + tbox[3,1,0] >= 0
 R492: tbox[1,3,0] - tbox[2,0,1] + tbox[2,2,0] >= 0
 R493: tbox[1,3,0] + tbox[2,0,1] - tbox[2,2,0] >= 0
 R494: - tbox[1,3,0] + tbox[2,0,1] + tbox[2,2,0] >= 0
 R495: - tbox[0,0,0] + tbox[1,0,1] = 0
 R496: unkn[0,0,0] - unkn[3,0,1] + forg[3,0,1] <= 0
 R497: unkn[2,2,0] - unkn[3,0,1] + forg[3,0,1] <= 0
 R498: unkn[0,0,0] + unkn[2,2,0] - unkn[3,0,1] + forg[3,0,1] >= 0
 R499: - unkn[0,0,1] + unkn[3,0,1] + forg[0,0,1] - forg[3,0,1] <= 0
 R500: - unkn[0,0,1] + unkn[3,1,0] + forg[0,0,1] <= 0
 R501: - unkn[0,0,1] + unkn[3,0,1] + unkn[3,1,0] + forg[0,0,1]
   - forg[3,0,1] >= 0
 R502: unkn[1,3,0] - unkn[2,0,1] + forg[2,0,1] <= 0
 R503: - unkn[2,0,1] + unkn[2,2,0] + forg[2,0,1] <= 0
 R504: unkn[1,3,0] - unkn[2,0,1] + unkn[2,2,0] + forg[2,0,1] >= 0
 R505: unkn[0,0,0] - unkn[1,0,1] + forg[1,0,1] = 0
 R506: - tbox[0,1,0] + forg[0,1,0] <= 0
 R507: sbox[0,1,0] - rkey[0,1,0] + tkey[0,1,0] >= 0
 R508: sbox[0,1,0] + rkey[0,1,0] - tkey[0,1,0] >= 0
 R509: - sbox[0,1,0] + rkey[0,1,0] + tkey[0,1,0] >= 0
 R510: - tbox[1,1,0] + forg[1,1,0] <= 0
 R511: sbox[1,1,0] - rkey[1,1,0] + tkey[1,1,0] >= 0
 R512: sbox[1,1,0] + rkey[1,1,0] - tkey[1,1,0] >= 0
 R513: - sbox[1,1,0] + rkey[1,1,0] + tkey[1,1,0] >= 0
 R514: - tbox[2,1,0] + forg[2,1,0] <= 0
 R515: - sbox[2,1,0] + rkey[2,1,0] = 0
 R516: - tbox[3,1,0] + forg[3,1,0] <= 0
 R517: - sbox[3,1,0] + rkey[3,1,0] = 0
 R518: - sbox[3,1,1] + rkey[0,1,0] + rkey[2,3,0] >= 0
 R519: sbox[3,1,1] + rkey[0,1,0] - rkey[2,3,0] >= 0
 R520: sbox[3,1,1] - rkey[0,1,0] + rkey[2,3,0] >= 0
 R521: - sbox[0,1,1] + sbox[3,1,1] + rkey[3,2,0] >= 0
 R522: sbox[0,1,1] + sbox[3,1,1] - rkey[3,2,0] >= 0
 R523: sbox[0,1,1] - sbox[3,1,1] + rkey[3,2,0] >= 0
 R524: - sbox[2,1,1] + rkey[1,0,0] + rkey[2,3,0] >= 0
 R525: sbox[2,1,1] + rkey[1,0,0] - rkey[2,3,0] >= 0
 R526: sbox[2,1,1] - rkey[1,0,0] + rkey[2,3,0] >= 0
 R527: sbox[1,1,1] - rkey[0,1,0] = 0
 R528: tbox[0,1,0] + tbox[2,3,0] - tbox[3,1,1] >= 0
 R529: tbox[0,1,0] - tbox[2,3,0] + tbox[3,1,1] >= 0
 R530: - tbox[0,1,0] + tbox[2,3,0] + tbox[3,1,1] >= 0
 R531: - tbox[0,1,1] + tbox[3,1,1] + tbox[3,2,0] >= 0
 R532: tbox[0,1,1] + tbox[3,1,1] - tbox[3,2,0] >= 0
 R533: tbox[0,1,1] - tbox[3,1,1] + tbox[3,2,0] >= 0
 R534: tbox[1,0,0] - tbox[2,1,1] + tbox[2,3,0] >= 0
 R535: tbox[1,0,0] + tbox[2,1,1] - tbox[2,3,0] >= 0
 R536: - tbox[1,0,0] + tbox[2,1,1] + tbox[2,3,0] >= 0
 R537: - tbox[0,1,0] + tbox[1,1,1] = 0
 R538: unkn[0,1,0] - unkn[3,1,1] + forg[3,1,1] <= 0
 R539: unkn[2,3,0] - unkn[3,1,1] + forg[3,1,1] <= 0
 R540: unkn[0,1,0] + unkn[2,3,0] - unkn[3,1,1] + forg[3,1,1] >= 0
 R541: - unkn[0,1,1] + unkn[3,1,1] + forg[0,1,1] - forg[3,1,1] <= 0
 R542: - unkn[0,1,1] + unkn[3,2,0] + forg[0,1,1] <= 0
 R543: - unkn[0,1,1] + unkn[3,1,1] + unkn[3,2,0] + forg[0,1,1]
   - forg[3,1,1] >= 0
 R544: unkn[1,0,0] - unkn[2,1,1] + forg[2,1,1] <= 0
 R545: - unkn[2,1,1] + unkn[2,3,0] + forg[2,1,1] <= 0
 R546: unkn[1,0,0] - unkn[2,1,1] + unkn[2,3,0] + forg[2,1,1] >= 0
 R547: unkn[0,1,0] - unkn[1,1,1] + forg[1,1,1] = 0
 R548: - tbox[0,2,0] + forg[0,2,0] <= 0
 R549: sbox[0,2,0] - rkey[0,2,0] + tkey[0,2,0] >= 0
 R550: sbox[0,2,0] + rkey[0,2,0] - tkey[0,2,0] >= 0
 R551: - sbox[0,2,0] + rkey[0,2,0] + tkey[0,2,0] >= 0
 R552: - tbox[1,2,0] + forg[1,2,0] <= 0
 R553: sbox[1,2,0] - rkey[1,2,0] + tkey[1,2,0] >= 0
 R554: sbox[1,2,0] + rkey[1,2,0] - tkey[1,2,0] >= 0
 R555: - sbox[1,2,0] + rkey[1,2,0] + tkey[1,2,0] >= 0
 R556: - tbox[2,2,0] + forg[2,2,0] <= 0
 R557: - sbox[2,2,0] + rkey[2,2,0] = 0
 R558: - tbox[3,2,0] + forg[3,2,0] <= 0
 R559: - sbox[3,2,0] + rkey[3,2,0] = 0
 R560: - sbox[3,2,1] + rkey[0,2,0] + rkey[2,0,0] >= 0
 R561: sbox[3,2,1] + rkey[0,2,0] - rkey[2,0,0] >= 0
 R562: sbox[3,2,1] - rkey[0,2,0] + rkey[2,0,0] >= 0
 R563: - sbox[0,2,1] + sbox[3,2,1] + rkey[3,3,0] >= 0
 R564: sbox[0,2,1] + sbox[3,2,1] - rkey[3,3,0] >= 0
 R565: sbox[0,2,1] - sbox[3,2,1] + rkey[3,3,0] >= 0
 R566: - sbox[2,2,1] + rkey[1,1,0] + rkey[2,0,0] >= 0
 R567: sbox[2,2,1] + rkey[1,1,0] - rkey[2,0,0] >= 0
 R568: sbox[2,2,1] - rkey[1,1,0] + rkey[2,0,0] >= 0
 R569: sbox[1,2,1] - rkey[0,2,0] = 0
 R570: tbox[0,2,0] + tbox[2,0,0] - tbox[3,2,1] >= 0
 R571: tbox[0,2,0] - tbox[2,0,0] + tbox[3,2,1] >= 0
 R572: - tbox[0,2,0] + tbox[2,0,0] + tbox[3,2,1] >= 0
 R573: - tbox[0,2,1] + tbox[3,2,1] + tbox[3,3,0] >= 0
 R574: tbox[0,2,1] + tbox[3,2,1] - tbox[3,3,0] >= 0
 R575: tbox[0,2,1] - tbox[3,2,1] + tbox[3,3,0] >= 0
 R576: tbox[1,1,0] + tbox[2,0,0] - tbox[2,2,1] >= 0
 R577: tbox[1,1,0] - tbox[2,0,0] + tbox[2,2,1] >= 0
 R578: - tbox[1,1,0] + tbox[2,0,0] + tbox[2,2,1] >= 0
 R579: - tbox[0,2,0] + tbox[1,2,1] = 0
 R580: unkn[0,2,0] - unkn[3,2,1] + forg[3,2,1] <= 0
 R581: unkn[2,0,0] - unkn[3,2,1] + forg[3,2,1] <= 0
 R582: unkn[0,2,0] + unkn[2,0,0] - unkn[3,2,1] + forg[3,2,1] >= 0
 R583: - unkn[0,2,1] + unkn[3,2,1] + forg[0,2,1] - forg[3,2,1] <= 0
 R584: - unkn[0,2,1] + unkn[3,3,0] + forg[0,2,1] <= 0
 R585: - unkn[0,2,1] + unkn[3,2,1] + unkn[3,3,0] + forg[0,2,1]
   - forg[3,2,1] >= 0
 R586: unkn[1,1,0] - unkn[2,2,1] + forg[2,2,1] <= 0
 R587: unkn[2,0,0] - unkn[2,2,1] + forg[2,2,1] <= 0
 R588: unkn[1,1,0] + unkn[2,0,0] - unkn[2,2,1] + forg[2,2,1] >= 0
 R589: unkn[0,2,0] - unkn[1,2,1] + forg[1,2,1] = 0
 R590: - tbox[0,3,0] + forg[0,3,0] <= 0
 R591: sbox[0,3,0] - rkey[0,3,0] + tkey[0,3,0] >= 0
 R592: sbox[0,3,0] + rkey[0,3,0] - tkey[0,3,0] >= 0
 R593: - sbox[0,3,0] + rkey[0,3,0] + tkey[0,3,0] >= 0
 R594: - tbox[1,3,0] + forg[1,3,0] <= 0
 R595: sbox[1,3,0] - rkey[1,3,0] + tkey[1,3,0] >= 0
 R596: sbox[1,3,0] + rkey[1,3,0] - tkey[1,3,0] >= 0
 R597: - sbox[1,3,0] + rkey[1,3,0] + tkey[1,3,0] >= 0
 R598: - tbox[2,3,0] + forg[2,3,0] <= 0
 R599: - sbox[2,3,0] + rkey[2,3,0] = 0
 R600: - tbox[3,3,0] + forg[3,3,0] <= 0
 R601: - sbox[3,3,0] + rkey[3,3,0] = 0
 R602: - sbox[3,3,1] + rkey[0,3,0] + rkey[2,1,0] >= 0
 R603: sbox[3,3,1] + rkey[0,3,0] - rkey[2,1,0] >= 0
 R604: sbox[3,3,1] - rkey[0,3,0] + rkey[2,1,0] >= 0
 R605: - sbox[0,3,1] + sbox[3,3,1] + rkey[3,0,0] >= 0
 R606: sbox[0,3,1] + sbox[3,3,1] - rkey[3,0,0] >= 0
 R607: sbox[0,3,1] - sbox[3,3,1] + rkey[3,0,0] >= 0
 R608: - sbox[2,3,1] + rkey[1,2,0] + rkey[2,1,0] >= 0
 R609: sbox[2,3,1] + rkey[1,2,0] - rkey[2,1,0] >= 0
 R610: sbox[2,3,1] - rkey[1,2,0] + rkey[2,1,0] >= 0
 R611: sbox[1,3,1] - rkey[0,3,0] = 0
 R612: tbox[0,3,0] + tbox[2,1,0] - tbox[3,3,1] >= 0
 R613: tbox[0,3,0] - tbox[2,1,0] + tbox[3,3,1] >= 0
 R614: - tbox[0,3,0] + tbox[2,1,0] + tbox[3,3,1] >= 0
 R615: - tbox[0,3,1] + tbox[3,0,0] + tbox[3,3,1] >= 0
 R616: tbox[0,3,1] - tbox[3,0,0] + tbox[3,3,1] >= 0
 R617: tbox[0,3,1] + tbox[3,0,0] - tbox[3,3,1] >= 0
 R618: tbox[1,2,0] + tbox[2,1,0] - tbox[2,3,1] >= 0
 R619: tbox[1,2,0] - tbox[2,1,0] + tbox[2,3,1] >= 0
 R620: - tbox[1,2,0] + tbox[2,1,0] + tbox[2,3,1] >= 0
 R621: - tbox[0,3,0] + tbox[1,3,1] = 0
 R622: unkn[0,3,0] - unkn[3,3,1] + forg[3,3,1] <= 0
 R623: unkn[2,1,0] - unkn[3,3,1] + forg[3,3,1] <= 0
 R624: unkn[0,3,0] + unkn[2,1,0] - unkn[3,3,1] + forg[3,3,1] >= 0
 R625: - unkn[0,3,1] + unkn[3,3,1] + forg[0,3,1] - forg[3,3,1] <= 0
 R626: - unkn[0,3,1] + unkn[3,0,0] + forg[0,3,1] <= 0
 R627: - unkn[0,3,1] + unkn[3,0,0] + unkn[3,3,1] + forg[0,3,1]
   - forg[3,3,1] >= 0
 R628: unkn[1,2,0] - unkn[2,3,1] + forg[2,3,1] <= 0
 R629: unkn[2,1,0] - unkn[2,3,1] + forg[2,3,1] <= 0
 R630: unkn[1,2,0] + unkn[2,1,0] - unkn[2,3,1] + forg[2,3,1] >= 0
 R631: unkn[0,3,0] - unkn[1,3,1] + forg[1,3,1] = 0
 R632: - tbox[0,0,1] + forg[0,0,1] <= 0
 R633: sbox[0,0,1] - rkey[0,0,1] + tkey[0,0,1] >= 0
 R634: sbox[0,0,1] + rkey[0,0,1] - tkey[0,0,1] >= 0
 R635: - sbox[0,0,1] + rkey[0,0,1] + tkey[0,0,1] >= 0
 R636: - tbox[1,0,1] + forg[1,0,1] <= 0
 R637: sbox[1,0,1] - rkey[1,0,1] + tkey[1,0,1] >= 0
 R638: sbox[1,0,1] + rkey[1,0,1] - tkey[1,0,1] >= 0
 R639: - sbox[1,0,1] + rkey[1,0,1] + tkey[1,0,1] >= 0
 R640: - tbox[2,0,1] + forg[2,0,1] <= 0
 R641: - sbox[2,0,1] + rkey[2,0,1] = 0
 R642: - tbox[3,0,1] + forg[3,0,1] <= 0
 R643: - sbox[3,0,1] + rkey[3,0,1] = 0
 R644: - sbox[3,0,2] + rkey[0,0,1] + rkey[2,2,1] >= 0
 R645: sbox[3,0,2] + rkey[0,0,1] - rkey[2,2,1] >= 0
 R646: sbox[3,0,2] - rkey[0,0,1] + rkey[2,2,1] >= 0
 R647: - sbox[0,0,2] + sbox[3,0,2] + rkey[3,1,1] >= 0
 R648: sbox[0,0,2] + sbox[3,0,2] - rkey[3,1,1] >= 0
 R649: sbox[0,0,2] - sbox[3,0,2] + rkey[3,1,1] >= 0
 R650: - sbox[2,0,2] + rkey[1,3,1] + rkey[2,2,1] >= 0
 R651: sbox[2,0,2] + rkey[1,3,1] - rkey[2,2,1] >= 0
 R652: sbox[2,0,2] - rkey[1,3,1] + rkey[2,2,1] >= 0
 R653: sbox[1,0,2] - rkey[0,0,1] = 0
 R654: tbox[0,0,1] + tbox[2,2,1] - tbox[3,0,2] >= 0
 R655: tbox[0,0,1] - tbox[2,2,1] + tbox[3,0,2] >= 0
 R656: - tbox[0,0,1] + tbox[2,2,1] + tbox[3,0,2] >= 0
 R657: - tbox[0,0,2] + tbox[3,0,2] + tbox[3,1,1] >= 0
 R658: tbox[0,0,2] + tbox[3,0,2] - tbox[3,1,1] >= 0
 R659: tbox[0,0,2] - tbox[3,0,2] + tbox[3,1,1] >= 0
 R660: tbox[1,3,1] - tbox[2,0,2] + tbox[2,2,1] >= 0
 R661: tbox[1,3,1] + tbox[2,0,2] - tbox[2,2,1] >= 0
 R662: - tbox[1,3,1] + tbox[2,0,2] + tbox[2,2,1] >= 0
 R663: - tbox[0,0,1] + tbox[1,0,2] = 0
 R664: unkn[0,0,1] - unkn[3,0,2] + forg[3,0,2] <= 0
 R665: unkn[2,2,1] - unkn[3,0,2] + forg[3,0,2] <= 0
 R666: unkn[0,0,1] + unkn[2,2,1] - unkn[3,0,2] + forg[3,0,2] >= 0
 R667: - unkn[0,0,2] + unkn[3,0,2] + forg[0,0,2] - forg[3,0,2] <= 0
 R668: - unkn[0,0,2] + unkn[3,1,1] + forg[0,0,2] <= 0
 R669: - unkn[0,0,2] + unkn[3,0,2] + unkn[3,1,1] + forg[0,0,2]
   - forg[3,0,2] >= 0
 R670: unkn[1,3,1] - unkn[2,0,2] + forg[2,0,2] <= 0
 R671: - unkn[2,0,2] + unkn[2,2,1] + forg[2,0,2] <= 0
 R672: unkn[1,3,1] - unkn[2,0,2] + unkn[2,2,1] + forg[2,0,2] >= 0
 R673: unkn[0,0,1] - unkn[1,0,2] + forg[1,0,2] = 0
 R674: - tbox[0,1,1] + forg[0,1,1] <= 0
 R675: sbox[0,1,1] - rkey[0,1,1] + tkey[0,1,1] >= 0
 R676: sbox[0,1,1] + rkey[0,1,1] - tkey[0,1,1] >= 0
 R677: - sbox[0,1,1] + rkey[0,1,1] + tkey[0,1,1] >= 0
 R678: - tbox[1,1,1] + forg[1,1,1] <= 0
 R679: sbox[1,1,1] - rkey[1,1,1] + tkey[1,1,1] >= 0
 R680: sbox[1,1,1] + rkey[1,1,1] - tkey[1,1,1] >= 0
 R681: - sbox[1,1,1] + rkey[1,1,1] + tkey[1,1,1] >= 0
 R682: - tbox[2,1,1] + forg[2,1,1] <= 0
 R683: - sbox[2,1,1] + rkey[2,1,1] = 0
 R684: - tbox[3,1,1] + forg[3,1,1] <= 0
 R685: - sbox[3,1,1] + rkey[3,1,1] = 0
 R686: - sbox[3,1,2] + rkey[0,1,1] + rkey[2,3,1] >= 0
 R687: sbox[3,1,2] + rkey[0,1,1] - rkey[2,3,1] >= 0
 R688: sbox[3,1,2] - rkey[0,1,1] + rkey[2,3,1] >= 0
 R689: - sbox[0,1,2] + sbox[3,1,2] + rkey[3,2,1] >= 0
 R690: sbox[0,1,2] + sbox[3,1,2] - rkey[3,2,1] >= 0
 R691: sbox[0,1,2] - sbox[3,1,2] + rkey[3,2,1] >= 0
 R692: - sbox[2,1,2] + rkey[1,0,1] + rkey[2,3,1] >= 0
 R693: sbox[2,1,2] + rkey[1,0,1] - rkey[2,3,1] >= 0
 R694: sbox[2,1,2] - rkey[1,0,1] + rkey[2,3,1] >= 0
 R695: sbox[1,1,2] - rkey[0,1,1] = 0
 R696: tbox[0,1,1] + tbox[2,3,1] - tbox[3,1,2] >= 0
 R697: tbox[0,1,1] - tbox[2,3,1] + tbox[3,1,2] >= 0
 R698: - tbox[0,1,1] + tbox[2,3,1] + tbox[3,1,2] >= 0
 R699: - tbox[0,1,2] + tbox[3,1,2] + tbox[3,2,1] >= 0
 R700: tbox[0,1,2] + tbox[3,1,2] - tbox[3,2,1] >= 0
 R701: tbox[0,1,2] - tbox[3,1,2] + tbox[3,2,1] >= 0
 R702: tbox[1,0,1] - tbox[2,1,2] + tbox[2,3,1] >= 0
 R703: tbox[1,0,1] + tbox[2,1,2] - tbox[2,3,1] >= 0
 R704: - tbox[1,0,1] + tbox[2,1,2] + tbox[2,3,1] >= 0
 R705: - tbox[0,1,1] + tbox[1,1,2] = 0
 R706: unkn[0,1,1] - unkn[3,1,2] + forg[3,1,2] <= 0
 R707: unkn[2,3,1] - unkn[3,1,2] + forg[3,1,2] <= 0
 R708: unkn[0,1,1] + unkn[2,3,1] - unkn[3,1,2] + forg[3,1,2] >= 0
 R709: - unkn[0,1,2] + unkn[3,1,2] + forg[0,1,2] - forg[3,1,2] <= 0
 R710: - unkn[0,1,2] + unkn[3,2,1] + forg[0,1,2] <= 0
 R711: - unkn[0,1,2] + unkn[3,1,2] + unkn[3,2,1] + forg[0,1,2]
   - forg[3,1,2] >= 0
 R712: unkn[1,0,1] - unkn[2,1,2] + forg[2,1,2] <= 0
 R713: - unkn[2,1,2] + unkn[2,3,1] + forg[2,1,2] <= 0
 R714: unkn[1,0,1] - unkn[2,1,2] + unkn[2,3,1] + forg[2,1,2] >= 0
 R715: unkn[0,1,1] - unkn[1,1,2] + forg[1,1,2] = 0
 R716: - tbox[0,2,1] + forg[0,2,1] <= 0
 R717: sbox[0,2,1] - rkey[0,2,1] + tkey[0,2,1] >= 0
 R718: sbox[0,2,1] + rkey[0,2,1] - tkey[0,2,1] >= 0
 R719: - sbox[0,2,1] + rkey[0,2,1] + tkey[0,2,1] >= 0
 R720: - tbox[1,2,1] + forg[1,2,1] <= 0
 R721: sbox[1,2,1] - rkey[1,2,1] + tkey[1,2,1] >= 0
 R722: sbox[1,2,1] + rkey[1,2,1] - tkey[1,2,1] >= 0
 R723: - sbox[1,2,1] + rkey[1,2,1] + tkey[1,2,1] >= 0
 R724: - tbox[2,2,1] + forg[2,2,1] <= 0
 R725: - sbox[2,2,1] + rkey[2,2,1] = 0
 R726: - tbox[3,2,1] + forg[3,2,1] <= 0
 R727: - sbox[3,2,1] + rkey[3,2,1] = 0
 R728: - sbox[3,2,2] + rkey[0,2,1] + rkey[2,0,1] >= 0
 R729: sbox[3,2,2] + rkey[0,2,1] - rkey[2,0,1] >= 0
 R730: sbox[3,2,2] - rkey[0,2,1] + rkey[2,0,1] >= 0
 R731: - sbox[0,2,2] + sbox[3,2,2] + rkey[3,3,1] >= 0
 R732: sbox[0,2,2] + sbox[3,2,2] - rkey[3,3,1] >= 0
 R733: sbox[0,2,2] - sbox[3,2,2] + rkey[3,3,1] >= 0
 R734: - sbox[2,2,2] + rkey[1,1,1] + rkey[2,0,1] >= 0
 R735: sbox[2,2,2] + rkey[1,1,1] - rkey[2,0,1] >= 0
 R736: sbox[2,2,2] - rkey[1,1,1] + rkey[2,0,1] >= 0
 R737: sbox[1,2,2] - rkey[0,2,1] = 0
 R738: tbox[0,2,1] + tbox[2,0,1] - tbox[3,2,2] >= 0
 R739: tbox[0,2,1] - tbox[2,0,1] + tbox[3,2,2] >= 0
 R740: - tbox[0,2,1] + tbox[2,0,1] + tbox[3,2,2] >= 0
 R741: - tbox[0,2,2] + tbox[3,2,2] + tbox[3,3,1] >= 0
 R742: tbox[0,2,2] + tbox[3,2,2] - tbox[3,3,1] >= 0
 R743: tbox[0,2,2] - tbox[3,2,2] + tbox[3,3,1] >= 0
 R744: tbox[1,1,1] + tbox[2,0,1] - tbox[2,2,2] >= 0
 R745: tbox[1,1,1] - tbox[2,0,1] + tbox[2,2,2] >= 0
 R746: - tbox[1,1,1] + tbox[2,0,1] + tbox[2,2,2] >= 0
 R747: - tbox[0,2,1] + tbox[1,2,2] = 0
 R748: unkn[0,2,1] - unkn[3,2,2] + forg[3,2,2] <= 0
 R749: unkn[2,0,1] - unkn[3,2,2] + forg[3,2,2] <= 0
 R750: unkn[0,2,1] + unkn[2,0,1] - unkn[3,2,2] + forg[3,2,2] >= 0
 R751: - unkn[0,2,2] + unkn[3,2,2] + forg[0,2,2] - forg[3,2,2] <= 0
 R752: - unkn[0,2,2] + unkn[3,3,1] + forg[0,2,2] <= 0
 R753: - unkn[0,2,2] + unkn[3,2,2] + unkn[3,3,1] + forg[0,2,2]
   - forg[3,2,2] >= 0
 R754: unkn[1,1,1] - unkn[2,2,2] + forg[2,2,2] <= 0
 R755: unkn[2,0,1] - unkn[2,2,2] + forg[2,2,2] <= 0
 R756: unkn[1,1,1] + unkn[2,0,1] - unkn[2,2,2] + forg[2,2,2] >= 0
 R757: unkn[0,2,1] - unkn[1,2,2] + forg[1,2,2] = 0
 R758: - tbox[0,3,1] + forg[0,3,1] <= 0
 R759: sbox[0,3,1] - rkey[0,3,1] + tkey[0,3,1] >= 0
 R760: sbox[0,3,1] + rkey[0,3,1] - tkey[0,3,1] >= 0
 R761: - sbox[0,3,1] + rkey[0,3,1] + tkey[0,3,1] >= 0
 R762: - tbox[1,3,1] + forg[1,3,1] <= 0
 R763: sbox[1,3,1] - rkey[1,3,1] + tkey[1,3,1] >= 0
 R764: sbox[1,3,1] + rkey[1,3,1] - tkey[1,3,1] >= 0
 R765: - sbox[1,3,1] + rkey[1,3,1] + tkey[1,3,1] >= 0
 R766: - tbox[2,3,1] + forg[2,3,1] <= 0
 R767: - sbox[2,3,1] + rkey[2,3,1] = 0
 R768: - tbox[3,3,1] + forg[3,3,1] <= 0
 R769: - sbox[3,3,1] + rkey[3,3,1] = 0
 R770: - sbox[3,3,2] + rkey[0,3,1] + rkey[2,1,1] >= 0
 R771: sbox[3,3,2] + rkey[0,3,1] - rkey[2,1,1] >= 0
 R772: sbox[3,3,2] - rkey[0,3,1] + rkey[2,1,1] >= 0
 R773: - sbox[0,3,2] + sbox[3,3,2] + rkey[3,0,1] >= 0
 R774: sbox[0,3,2] + sbox[3,3,2] - rkey[3,0,1] >= 0
 R775: sbox[0,3,2] - sbox[3,3,2] + rkey[3,0,1] >= 0
 R776: - sbox[2,3,2] + rkey[1,2,1] + rkey[2,1,1] >= 0
 R777: sbox[2,3,2] + rkey[1,2,1] - rkey[2,1,1] >= 0
 R778: sbox[2,3,2] - rkey[1,2,1] + rkey[2,1,1] >= 0
 R779: sbox[1,3,2] - rkey[0,3,1] = 0
 R780: tbox[0,3,1] + tbox[2,1,1] - tbox[3,3,2] >= 0
 R781: tbox[0,3,1] - tbox[2,1,1] + tbox[3,3,2] >= 0
 R782: - tbox[0,3,1] + tbox[2,1,1] + tbox[3,3,2] >= 0
 R783: - tbox[0,3,2] + tbox[3,0,1] + tbox[3,3,2] >= 0
 R784: tbox[0,3,2] - tbox[3,0,1] + tbox[3,3,2] >= 0
 R785: tbox[0,3,2] + tbox[3,0,1] - tbox[3,3,2] >= 0
 R786: tbox[1,2,1] + tbox[2,1,1] - tbox[2,3,2] >= 0
 R787: tbox[1,2,1] - tbox[2,1,1] + tbox[2,3,2] >= 0
 R788: - tbox[1,2,1] + tbox[2,1,1] + tbox[2,3,2] >= 0
 R789: - tbox[0,3,1] + tbox[1,3,2] = 0
 R790: unkn[0,3,1] - unkn[3,3,2] + forg[3,3,2] <= 0
 R791: unkn[2,1,1] - unkn[3,3,2] + forg[3,3,2] <= 0
 R792: unkn[0,3,1] + unkn[2,1,1] - unkn[3,3,2] + forg[3,3,2] >= 0
 R793: - unkn[0,3,2] + unkn[3,3,2] + forg[0,3,2] - forg[3,3,2] <= 0
 R794: - unkn[0,3,2] + unkn[3,0,1] + forg[0,3,2] <= 0
 R795: - unkn[0,3,2] + unkn[3,0,1] + unkn[3,3,2] + forg[0,3,2]
   - forg[3,3,2] >= 0
 R796: unkn[1,2,1] - unkn[2,3,2] + forg[2,3,2] <= 0
 R797: unkn[2,1,1] - unkn[2,3,2] + forg[2,3,2] <= 0
 R798: unkn[1,2,1] + unkn[2,1,1] - unkn[2,3,2] + forg[2,3,2] >= 0
 R799: unkn[0,3,1] - unkn[1,3,2] + forg[1,3,2] = 0
 R800: - tbox[0,0,2] + forg[0,0,2] <= 0
 R801: sbox[0,0,2] - rkey[0,0,2] + tkey[0,0,2] >= 0
 R802: sbox[0,0,2] + rkey[0,0,2] - tkey[0,0,2] >= 0
 R803: - sbox[0,0,2] + rkey[0,0,2] + tkey[0,0,2] >= 0
 R804: - tbox[1,0,2] + forg[1,0,2] <= 0
 R805: sbox[1,0,2] - rkey[1,0,2] + tkey[1,0,2] >= 0
 R806: sbox[1,0,2] + rkey[1,0,2] - tkey[1,0,2] >= 0
 R807: - sbox[1,0,2] + rkey[1,0,2] + tkey[1,0,2] >= 0
 R808: - tbox[2,0,2] + forg[2,0,2] <= 0
 R809: - sbox[2,0,2] + rkey[2,0,2] = 0
 R810: - tbox[3,0,2] + forg[3,0,2] <= 0
 R811: - sbox[3,0,2] + rkey[3,0,2] = 0
 R812: - sbox[3,0,3] + rkey[0,0,2] + rkey[2,2,2] >= 0
 R813: sbox[3,0,3] + rkey[0,0,2] - rkey[2,2,2] >= 0
 R814: sbox[3,0,3] - rkey[0,0,2] + rkey[2,2,2] >= 0
 R815: - sbox[0,0,3] + sbox[3,0,3] + rkey[3,1,2] >= 0
 R816: sbox[0,0,3] + sbox[3,0,3] - rkey[3,1,2] >= 0
 R817: sbox[0,0,3] - sbox[3,0,3] + rkey[3,1,2] >= 0
 R818: - sbox[2,0,3] + rkey[1,3,2] + rkey[2,2,2] >= 0
 R819: sbox[2,0,3] + rkey[1,3,2] - rkey[2,2,2] >= 0
 R820: sbox[2,0,3] - rkey[1,3,2] + rkey[2,2,2] >= 0
 R821: sbox[1,0,3] - rkey[0,0,2] = 0
 R822: tbox[0,0,2] + tbox[2,2,2] - tbox[3,0,3] >= 0
 R823: tbox[0,0,2] - tbox[2,2,2] + tbox[3,0,3] >= 0
 R824: - tbox[0,0,2] + tbox[2,2,2] + tbox[3,0,3] >= 0
 R825: - tbox[0,0,3] + tbox[3,0,3] + tbox[3,1,2] >= 0
 R826: tbox[0,0,3] + tbox[3,0,3] - tbox[3,1,2] >= 0
 R827: tbox[0,0,3] - tbox[3,0,3] + tbox[3,1,2] >= 0
 R828: tbox[1,3,2] - tbox[2,0,3] + tbox[2,2,2] >= 0
 R829: tbox[1,3,2] + tbox[2,0,3] - tbox[2,2,2] >= 0
 R830: - tbox[1,3,2] + tbox[2,0,3] + tbox[2,2,2] >= 0
 R831: - tbox[0,0,2] + tbox[1,0,3] = 0
 R832: unkn[0,0,2] - unkn[3,0,3] + forg[3,0,3] <= 0
 R833: unkn[2,2,2] - unkn[3,0,3] + forg[3,0,3] <= 0
 R834: unkn[0,0,2] + unkn[2,2,2] - unkn[3,0,3] + forg[3,0,3] >= 0
 R835: - unkn[0,0,3] + unkn[3,0,3] + forg[0,0,3] - forg[3,0,3] <= 0
 R836: - unkn[0,0,3] + unkn[3,1,2] + forg[0,0,3] <= 0
 R837: - unkn[0,0,3] + unkn[3,0,3] + unkn[3,1,2] + forg[0,0,3]
   - forg[3,0,3] >= 0
 R838: unkn[1,3,2] - unkn[2,0,3] + forg[2,0,3] <= 0
 R839: - unkn[2,0,3] + unkn[2,2,2] + forg[2,0,3] <= 0
 R840: unkn[1,3,2] - unkn[2,0,3] + unkn[2,2,2] + forg[2,0,3] >= 0
 R841: unkn[0,0,2] - unkn[1,0,3] + forg[1,0,3] = 0
 R842: - tbox[0,1,2] + forg[0,1,2] <= 0
 R843: sbox[0,1,2] - rkey[0,1,2] + tkey[0,1,2] >= 0
 R844: sbox[0,1,2] + rkey[0,1,2] - tkey[0,1,2] >= 0
 R845: - sbox[0,1,2] + rkey[0,1,2] + tkey[0,1,2] >= 0
 R846: - tbox[1,1,2] + forg[1,1,2] <= 0
 R847: sbox[1,1,2] - rkey[1,1,2] + tkey[1,1,2] >= 0
 R848: sbox[1,1,2] + rkey[1,1,2] - tkey[1,1,2] >= 0
 R849: - sbox[1,1,2] + rkey[1,1,2] + tkey[1,1,2] >= 0
 R850: - tbox[2,1,2] + forg[2,1,2] <= 0
 R851: - sbox[2,1,2] + rkey[2,1,2] = 0
 R852: - tbox[3,1,2] + forg[3,1,2] <= 0
 R853: - sbox[3,1,2] + rkey[3,1,2] = 0
 R854: - sbox[3,1,3] + rkey[0,1,2] + rkey[2,3,2] >= 0
 R855: sbox[3,1,3] + rkey[0,1,2] - rkey[2,3,2] >= 0
 R856: sbox[3,1,3] - rkey[0,1,2] + rkey[2,3,2] >= 0
 R857: - sbox[0,1,3] + sbox[3,1,3] + rkey[3,2,2] >= 0
 R858: sbox[0,1,3] + sbox[3,1,3] - rkey[3,2,2] >= 0
 R859: sbox[0,1,3] - sbox[3,1,3] + rkey[3,2,2] >= 0
 R860: - sbox[2,1,3] + rkey[1,0,2] + rkey[2,3,2] >= 0
 R861: sbox[2,1,3] + rkey[1,0,2] - rkey[2,3,2] >= 0
 R862: sbox[2,1,3] - rkey[1,0,2] + rkey[2,3,2] >= 0
 R863: sbox[1,1,3] - rkey[0,1,2] = 0
 R864: tbox[0,1,2] + tbox[2,3,2] - tbox[3,1,3] >= 0
 R865: tbox[0,1,2] - tbox[2,3,2] + tbox[3,1,3] >= 0
 R866: - tbox[0,1,2] + tbox[2,3,2] + tbox[3,1,3] >= 0
 R867: - tbox[0,1,3] + tbox[3,1,3] + tbox[3,2,2] >= 0
 R868: tbox[0,1,3] + tbox[3,1,3] - tbox[3,2,2] >= 0
 R869: tbox[0,1,3] - tbox[3,1,3] + tbox[3,2,2] >= 0
 R870: tbox[1,0,2] - tbox[2,1,3] + tbox[2,3,2] >= 0
 R871: tbox[1,0,2] + tbox[2,1,3] - tbox[2,3,2] >= 0
 R872: - tbox[1,0,2] + tbox[2,1,3] + tbox[2,3,2] >= 0
 R873: - tbox[0,1,2] + tbox[1,1,3] = 0
 R874: unkn[0,1,2] - unkn[3,1,3] + forg[3,1,3] <= 0
 R875: unkn[2,3,2] - unkn[3,1,3] + forg[3,1,3] <= 0
 R876: unkn[0,1,2] + unkn[2,3,2] - unkn[3,1,3] + forg[3,1,3] >= 0
 R877: - unkn[0,1,3] + unkn[3,1,3] + forg[0,1,3] - forg[3,1,3] <= 0
 R878: - unkn[0,1,3] + unkn[3,2,2] + forg[0,1,3] <= 0
 R879: - unkn[0,1,3] + unkn[3,1,3] + unkn[3,2,2] + forg[0,1,3]
   - forg[3,1,3] >= 0
 R880: unkn[1,0,2] - unkn[2,1,3] + forg[2,1,3] <= 0
 R881: - unkn[2,1,3] + unkn[2,3,2] + forg[2,1,3] <= 0
 R882: unkn[1,0,2] - unkn[2,1,3] + unkn[2,3,2] + forg[2,1,3] >= 0
 R883: unkn[0,1,2] - unkn[1,1,3] + forg[1,1,3] = 0
 R884: - tbox[0,2,2] + forg[0,2,2] <= 0
 R885: sbox[0,2,2] - rkey[0,2,2] + tkey[0,2,2] >= 0
 R886: sbox[0,2,2] + rkey[0,2,2] - tkey[0,2,2] >= 0
 R887: - sbox[0,2,2] + rkey[0,2,2] + tkey[0,2,2] >= 0
 R888: - tbox[1,2,2] + forg[1,2,2] <= 0
 R889: sbox[1,2,2] - rkey[1,2,2] + tkey[1,2,2] >= 0
 R890: sbox[1,2,2] + rkey[1,2,2] - tkey[1,2,2] >= 0
 R891: - sbox[1,2,2] + rkey[1,2,2] + tkey[1,2,2] >= 0
 R892: - tbox[2,2,2] + forg[2,2,2] <= 0
 R893: - sbox[2,2,2] + rkey[2,2,2] = 0
 R894: - tbox[3,2,2] + forg[3,2,2] <= 0
 R895: - sbox[3,2,2] + rkey[3,2,2] = 0
 R896: - sbox[3,2,3] + rkey[0,2,2] + rkey[2,0,2] >= 0
 R897: sbox[3,2,3] + rkey[0,2,2] - rkey[2,0,2] >= 0
 R898: sbox[3,2,3] - rkey[0,2,2] + rkey[2,0,2] >= 0
 R899: - sbox[0,2,3] + sbox[3,2,3] + rkey[3,3,2] >= 0
 R900: sbox[0,2,3] + sbox[3,2,3] - rkey[3,3,2] >= 0
 R901: sbox[0,2,3] - sbox[3,2,3] + rkey[3,3,2] >= 0
 R902: - sbox[2,2,3] + rkey[1,1,2] + rkey[2,0,2] >= 0
 R903: sbox[2,2,3] + rkey[1,1,2] - rkey[2,0,2] >= 0
 R904: sbox[2,2,3] - rkey[1,1,2] + rkey[2,0,2] >= 0
 R905: sbox[1,2,3] - rkey[0,2,2] = 0
 R906: tbox[0,2,2] + tbox[2,0,2] - tbox[3,2,3] >= 0
 R907: tbox[0,2,2] - tbox[2,0,2] + tbox[3,2,3] >= 0
 R908: - tbox[0,2,2] + tbox[2,0,2] + tbox[3,2,3] >= 0
 R909: - tbox[0,2,3] + tbox[3,2,3] + tbox[3,3,2] >= 0
 R910: tbox[0,2,3] + tbox[3,2,3] - tbox[3,3,2] >= 0
 R911: tbox[0,2,3] - tbox[3,2,3] + tbox[3,3,2] >= 0
 R912: tbox[1,1,2] + tbox[2,0,2] - tbox[2,2,3] >= 0
 R913: tbox[1,1,2] - tbox[2,0,2] + tbox[2,2,3] >= 0
 R914: - tbox[1,1,2] + tbox[2,0,2] + tbox[2,2,3] >= 0
 R915: - tbox[0,2,2] + tbox[1,2,3] = 0
 R916: unkn[0,2,2] - unkn[3,2,3] + forg[3,2,3] <= 0
 R917: unkn[2,0,2] - unkn[3,2,3] + forg[3,2,3] <= 0
 R918: unkn[0,2,2] + unkn[2,0,2] - unkn[3,2,3] + forg[3,2,3] >= 0
 R919: - unkn[0,2,3] + unkn[3,2,3] + forg[0,2,3] - forg[3,2,3] <= 0
 R920: - unkn[0,2,3] + unkn[3,3,2] + forg[0,2,3] <= 0
 R921: - unkn[0,2,3] + unkn[3,2,3] + unkn[3,3,2] + forg[0,2,3]
   - forg[3,2,3] >= 0
 R922: unkn[1,1,2] - unkn[2,2,3] + forg[2,2,3] <= 0
 R923: unkn[2,0,2] - unkn[2,2,3] + forg[2,2,3] <= 0
 R924: unkn[1,1,2] + unkn[2,0,2] - unkn[2,2,3] + forg[2,2,3] >= 0
 R925: unkn[0,2,2] - unkn[1,2,3] + forg[1,2,3] = 0
 R926: - tbox[0,3,2] + forg[0,3,2] <= 0
 R927: sbox[0,3,2] - rkey[0,3,2] + tkey[0,3,2] >= 0
 R928: sbox[0,3,2] + rkey[0,3,2] - tkey[0,3,2] >= 0
 R929: - sbox[0,3,2] + rkey[0,3,2] + tkey[0,3,2] >= 0
 R930: - tbox[1,3,2] + forg[1,3,2] <= 0
 R931: sbox[1,3,2] - rkey[1,3,2] + tkey[1,3,2] >= 0
 R932: sbox[1,3,2] + rkey[1,3,2] - tkey[1,3,2] >= 0
 R933: - sbox[1,3,2] + rkey[1,3,2] + tkey[1,3,2] >= 0
 R934: - tbox[2,3,2] + forg[2,3,2] <= 0
 R935: - sbox[2,3,2] + rkey[2,3,2] = 0
 R936: - tbox[3,3,2] + forg[3,3,2] <= 0
 R937: - sbox[3,3,2] + rkey[3,3,2] = 0
 R938: - sbox[3,3,3] + rkey[0,3,2] + rkey[2,1,2] >= 0
 R939: sbox[3,3,3] + rkey[0,3,2] - rkey[2,1,2] >= 0
 R940: sbox[3,3,3] - rkey[0,3,2] + rkey[2,1,2] >= 0
 R941: - sbox[0,3,3] + sbox[3,3,3] + rkey[3,0,2] >= 0
 R942: sbox[0,3,3] + sbox[3,3,3] - rkey[3,0,2] >= 0
 R943: sbox[0,3,3] - sbox[3,3,3] + rkey[3,0,2] >= 0
 R944: - sbox[2,3,3] + rkey[1,2,2] + rkey[2,1,2] >= 0
 R945: sbox[2,3,3] + rkey[1,2,2] - rkey[2,1,2] >= 0
 R946: sbox[2,3,3] - rkey[1,2,2] + rkey[2,1,2] >= 0
 R947: sbox[1,3,3] - rkey[0,3,2] = 0
 R948: tbox[0,3,2] + tbox[2,1,2] - tbox[3,3,3] >= 0
 R949: tbox[0,3,2] - tbox[2,1,2] + tbox[3,3,3] >= 0
 R950: - tbox[0,3,2] + tbox[2,1,2] + tbox[3,3,3] >= 0
 R951: - tbox[0,3,3] + tbox[3,0,2] + tbox[3,3,3] >= 0
 R952: tbox[0,3,3] - tbox[3,0,2] + tbox[3,3,3] >= 0
 R953: tbox[0,3,3] + tbox[3,0,2] - tbox[3,3,3] >= 0
 R954: tbox[1,2,2] + tbox[2,1,2] - tbox[2,3,3] >= 0
 R955: tbox[1,2,2] - tbox[2,1,2] + tbox[2,3,3] >= 0
 R956: - tbox[1,2,2] + tbox[2,1,2] + tbox[2,3,3] >= 0
 R957: - tbox[0,3,2] + tbox[1,3,3] = 0
 R958: unkn[0,3,2] - unkn[3,3,3] + forg[3,3,3] <= 0
 R959: unkn[2,1,2] - unkn[3,3,3] + forg[3,3,3] <= 0
 R960: unkn[0,3,2] + unkn[2,1,2] - unkn[3,3,3] + forg[3,3,3] >= 0
 R961: - unkn[0,3,3] + unkn[3,3,3] + forg[0,3,3] - forg[3,3,3] <= 0
 R962: - unkn[0,3,3] + unkn[3,0,2] + forg[0,3,3] <= 0
 R963: - unkn[0,3,3] + unkn[3,0,2] + unkn[3,3,3] + forg[0,3,3]
   - forg[3,3,3] >= 0
 R964: unkn[1,2,2] - unkn[2,3,3] + forg[2,3,3] <= 0
 R965: unkn[2,1,2] - unkn[2,3,3] + forg[2,3,3] <= 0
 R966: unkn[1,2,2] + unkn[2,1,2] - unkn[2,3,3] + forg[2,3,3] >= 0
 R967: unkn[0,3,2] - unkn[1,3,3] + forg[1,3,3] = 0
 R968: - tbox[0,0,3] + forg[0,0,3] <= 0
 R969: sbox[0,0,3] - rkey[0,0,3] + tkey[0,0,3] >= 0
 R970: sbox[0,0,3] + rkey[0,0,3] - tkey[0,0,3] >= 0
 R971: - sbox[0,0,3] + rkey[0,0,3] + tkey[0,0,3] >= 0
 R972: - tbox[1,0,3] + forg[1,0,3] <= 0
 R973: sbox[1,0,3] - rkey[1,0,3] + tkey[1,0,3] >= 0
 R974: sbox[1,0,3] + rkey[1,0,3] - tkey[1,0,3] >= 0
 R975: - sbox[1,0,3] + rkey[1,0,3] + tkey[1,0,3] >= 0
 R976: - tbox[2,0,3] + forg[2,0,3] <= 0
 R977: - sbox[2,0,3] + rkey[2,0,3] = 0
 R978: - tbox[3,0,3] + forg[3,0,3] <= 0
 R979: - sbox[3,0,3] + rkey[3,0,3] = 0
 R980: - sbox[3,0,4] + rkey[0,0,3] + rkey[2,2,3] >= 0
 R981: sbox[3,0,4] + rkey[0,0,3] - rkey[2,2,3] >= 0
 R982: sbox[3,0,4] - rkey[0,0,3] + rkey[2,2,3] >= 0
 R983: - sbox[0,0,4] + sbox[3,0,4] + rkey[3,1,3] >= 0
 R984: sbox[0,0,4] + sbox[3,0,4] - rkey[3,1,3] >= 0
 R985: sbox[0,0,4] - sbox[3,0,4] + rkey[3,1,3] >= 0
 R986: - sbox[2,0,4] + rkey[1,3,3] + rkey[2,2,3] >= 0
 R987: sbox[2,0,4] + rkey[1,3,3] - rkey[2,2,3] >= 0
 R988: sbox[2,0,4] - rkey[1,3,3] + rkey[2,2,3] >= 0
 R989: sbox[1,0,4] - rkey[0,0,3] = 0
 R990: tbox[0,0,3] + tbox[2,2,3] - tbox[3,0,4] >= 0
 R991: tbox[0,0,3] - tbox[2,2,3] + tbox[3,0,4] >= 0
 R992: - tbox[0,0,3] + tbox[2,2,3] + tbox[3,0,4] >= 0
 R993: - tbox[0,0,4] + tbox[3,0,4] + tbox[3,1,3] >= 0
 R994: tbox[0,0,4] + tbox[3,0,4] - tbox[3,1,3] >= 0
 R995: tbox[0,0,4] - tbox[3,0,4] + tbox[3,1,3] >= 0
 R996: tbox[1,3,3] - tbox[2,0,4] + tbox[2,2,3] >= 0
 R997: tbox[1,3,3] + tbox[2,0,4] - tbox[2,2,3] >= 0
 R998: - tbox[1,3,3] + tbox[2,0,4] + tbox[2,2,3] >= 0
 R999: - tbox[0,0,3] + tbox[1,0,4] = 0
 R1000: unkn[0,0,3] - unkn[3,0,4] + forg[3,0,4] <= 0
 R1001: unkn[2,2,3] - unkn[3,0,4] + forg[3,0,4] <= 0
 R1002: unkn[0,0,3] + unkn[2,2,3] - unkn[3,0,4] + forg[3,0,4] >= 0
 R1003: - unkn[0,0,4] + unkn[3,0,4] + forg[0,0,4] - forg[3,0,4] <= 0
 R1004: - unkn[0,0,4] + unkn[3,1,3] + forg[0,0,4] <= 0
 R1005: - unkn[0,0,4] + unkn[3,0,4] + unkn[3,1,3] + forg[0,0,4]
   - forg[3,0,4] >= 0
 R1006: unkn[1,3,3] - unkn[2,0,4] + forg[2,0,4] <= 0
 R1007: - unkn[2,0,4] + unkn[2,2,3] + forg[2,0,4] <= 0
 R1008: unkn[1,3,3] - unkn[2,0,4] + unkn[2,2,3] + forg[2,0,4] >= 0
 R1009: unkn[0,0,3] - unkn[1,0,4] + forg[1,0,4] = 0
 R1010: - tbox[0,1,3] + forg[0,1,3] <= 0
 R1011: sbox[0,1,3] - rkey[0,1,3] + tkey[0,1,3] >= 0
 R1012: sbox[0,1,3] + rkey[0,1,3] - tkey[0,1,3] >= 0
 R1013: - sbox[0,1,3] + rkey[0,1,3] + tkey[0,1,3] >= 0
 R1014: - tbox[1,1,3] + forg[1,1,3] <= 0
 R1015: sbox[1,1,3] - rkey[1,1,3] + tkey[1,1,3] >= 0
 R1016: sbox[1,1,3] + rkey[1,1,3] - tkey[1,1,3] >= 0
 R1017: - sbox[1,1,3] + rkey[1,1,3] + tkey[1,1,3] >= 0
 R1018: - tbox[2,1,3] + forg[2,1,3] <= 0
 R1019: - sbox[2,1,3] + rkey[2,1,3] = 0
 R1020: - tbox[3,1,3] + forg[3,1,3] <= 0
 R1021: - sbox[3,1,3] + rkey[3,1,3] = 0
 R1022: - sbox[3,1,4] + rkey[0,1,3] + rkey[2,3,3] >= 0
 R1023: sbox[3,1,4] + rkey[0,1,3] - rkey[2,3,3] >= 0
 R1024: sbox[3,1,4] - rkey[0,1,3] + rkey[2,3,3] >= 0
 R1025: - sbox[0,1,4] + sbox[3,1,4] + rkey[3,2,3] >= 0
 R1026: sbox[0,1,4] + sbox[3,1,4] - rkey[3,2,3] >= 0
 R1027: sbox[0,1,4] - sbox[3,1,4] + rkey[3,2,3] >= 0
 R1028: - sbox[2,1,4] + rkey[1,0,3] + rkey[2,3,3] >= 0
 R1029: sbox[2,1,4] + rkey[1,0,3] - rkey[2,3,3] >= 0
 R1030: sbox[2,1,4] - rkey[1,0,3] + rkey[2,3,3] >= 0
 R1031: sbox[1,1,4] - rkey[0,1,3] = 0
 R1032: tbox[0,1,3] + tbox[2,3,3] - tbox[3,1,4] >= 0
 R1033: tbox[0,1,3] - tbox[2,3,3] + tbox[3,1,4] >= 0
 R1034: - tbox[0,1,3] + tbox[2,3,3] + tbox[3,1,4] >= 0
 R1035: - tbox[0,1,4] + tbox[3,1,4] + tbox[3,2,3] >= 0
 R1036: tbox[0,1,4] + tbox[3,1,4] - tbox[3,2,3] >= 0
 R1037: tbox[0,1,4] - tbox[3,1,4] + tbox[3,2,3] >= 0
 R1038: tbox[1,0,3] - tbox[2,1,4] + tbox[2,3,3] >= 0
 R1039: tbox[1,0,3] + tbox[2,1,4] - tbox[2,3,3] >= 0
 R1040: - tbox[1,0,3] + tbox[2,1,4] + tbox[2,3,3] >= 0
 R1041: - tbox[0,1,3] + tbox[1,1,4] = 0
 R1042: unkn[0,1,3] - unkn[3,1,4] + forg[3,1,4] <= 0
 R1043: unkn[2,3,3] - unkn[3,1,4] + forg[3,1,4] <= 0
 R1044: unkn[0,1,3] + unkn[2,3,3] - unkn[3,1,4] + forg[3,1,4] >= 0
 R1045: - unkn[0,1,4] + unkn[3,1,4] + forg[0,1,4] - forg[3,1,4] <= 0
 R1046: - unkn[0,1,4] + unkn[3,2,3] + forg[0,1,4] <= 0
 R1047: - unkn[0,1,4] + unkn[3,1,4] + unkn[3,2,3] + forg[0,1,4]
   - forg[3,1,4] >= 0
 R1048: unkn[1,0,3] - unkn[2,1,4] + forg[2,1,4] <= 0
 R1049: - unkn[2,1,4] + unkn[2,3,3] + forg[2,1,4] <= 0
 R1050: unkn[1,0,3] - unkn[2,1,4] + unkn[2,3,3] + forg[2,1,4] >= 0
 R1051: unkn[0,1,3] - unkn[1,1,4] + forg[1,1,4] = 0
 R1052: - tbox[0,2,3] + forg[0,2,3] <= 0
 R1053: sbox[0,2,3] - rkey[0,2,3] + tkey[0,2,3] >= 0
 R1054: sbox[0,2,3] + rkey[0,2,3] - tkey[0,2,3] >= 0
 R1055: - sbox[0,2,3] + rkey[0,2,3] + tkey[0,2,3] >= 0
 R1056: - tbox[1,2,3] + forg[1,2,3] <= 0
 R1057: sbox[1,2,3] - rkey[1,2,3] + tkey[1,2,3] >= 0
 R1058: sbox[1,2,3] + rkey[1,2,3] - tkey[1,2,3] >= 0
 R1059: - sbox[1,2,3] + rkey[1,2,3] + tkey[1,2,3] >= 0
 R1060: - tbox[2,2,3] + forg[2,2,3] <= 0
 R1061: - sbox[2,2,3] + rkey[2,2,3] = 0
 R1062: - tbox[3,2,3] + forg[3,2,3] <= 0
 R1063: - sbox[3,2,3] + rkey[3,2,3] = 0
 R1064: - sbox[3,2,4] + rkey[0,2,3] + rkey[2,0,3] >= 0
 R1065: sbox[3,2,4] + rkey[0,2,3] - rkey[2,0,3] >= 0
 R1066: sbox[3,2,4] - rkey[0,2,3] + rkey[2,0,3] >= 0
 R1067: - sbox[0,2,4] + sbox[3,2,4] + rkey[3,3,3] >= 0
 R1068: sbox[0,2,4] + sbox[3,2,4] - rkey[3,3,3] >= 0
 R1069: sbox[0,2,4] - sbox[3,2,4] + rkey[3,3,3] >= 0
 R1070: - sbox[2,2,4] + rkey[1,1,3] + rkey[2,0,3] >= 0
 R1071: sbox[2,2,4] + rkey[1,1,3] - rkey[2,0,3] >= 0
 R1072: sbox[2,2,4] - rkey[1,1,3] + rkey[2,0,3] >= 0
 R1073: sbox[1,2,4] - rkey[0,2,3] = 0
 R1074: tbox[0,2,3] + tbox[2,0,3] - tbox[3,2,4] >= 0
 R1075: tbox[0,2,3] - tbox[2,0,3] + tbox[3,2,4] >= 0
 R1076: - tbox[0,2,3] + tbox[2,0,3] + tbox[3,2,4] >= 0
 R1077: - tbox[0,2,4] + tbox[3,2,4] + tbox[3,3,3] >= 0
 R1078: tbox[0,2,4] + tbox[3,2,4] - tbox[3,3,3] >= 0
 R1079: tbox[0,2,4] - tbox[3,2,4] + tbox[3,3,3] >= 0
 R1080: tbox[1,1,3] + tbox[2,0,3] - tbox[2,2,4] >= 0
 R1081: tbox[1,1,3] - tbox[2,0,3] + tbox[2,2,4] >= 0
 R1082: - tbox[1,1,3] + tbox[2,0,3] + tbox[2,2,4] >= 0
 R1083: - tbox[0,2,3] + tbox[1,2,4] = 0
 R1084: unkn[0,2,3] - unkn[3,2,4] + forg[3,2,4] <= 0
 R1085: unkn[2,0,3] - unkn[3,2,4] + forg[3,2,4] <= 0
 R1086: unkn[0,2,3] + unkn[2,0,3] - unkn[3,2,4] + forg[3,2,4] >= 0
 R1087: - unkn[0,2,4] + unkn[3,2,4] + forg[0,2,4] - forg[3,2,4] <= 0
 R1088: - unkn[0,2,4] + unkn[3,3,3] + forg[0,2,4] <= 0
 R1089: - unkn[0,2,4] + unkn[3,2,4] + unkn[3,3,3] + forg[0,2,4]
   - forg[3,2,4] >= 0
 R1090: unkn[1,1,3] - unkn[2,2,4] + forg[2,2,4] <= 0
 R1091: unkn[2,0,3] - unkn[2,2,4] + forg[2,2,4] <= 0
 R1092: unkn[1,1,3] + unkn[2,0,3] - unkn[2,2,4] + forg[2,2,4] >= 0
 R1093: unkn[0,2,3] - unkn[1,2,4] + forg[1,2,4] = 0
 R1094: - tbox[0,3,3] + forg[0,3,3] <= 0
 R1095: sbox[0,3,3] - rkey[0,3,3] + tkey[0,3,3] >= 0
 R1096: sbox[0,3,3] + rkey[0,3,3] - tkey[0,3,3] >= 0
 R1097: - sbox[0,3,3] + rkey[0,3,3] + tkey[0,3,3] >= 0
 R1098: - tbox[1,3,3] + forg[1,3,3] <= 0
 R1099: sbox[1,3,3] - rkey[1,3,3] + tkey[1,3,3] >= 0
 R1100: sbox[1,3,3] + rkey[1,3,3] - tkey[1,3,3] >= 0
 R1101: - sbox[1,3,3] + rkey[1,3,3] + tkey[1,3,3] >= 0
 R1102: - tbox[2,3,3] + forg[2,3,3] <= 0
 R1103: - sbox[2,3,3] + rkey[2,3,3] = 0
 R1104: - tbox[3,3,3] + forg[3,3,3] <= 0
 R1105: - sbox[3,3,3] + rkey[3,3,3] = 0
 R1106: - sbox[3,3,4] + rkey[0,3,3] + rkey[2,1,3] >= 0
 R1107: sbox[3,3,4] + rkey[0,3,3] - rkey[2,1,3] >= 0
 R1108: sbox[3,3,4] - rkey[0,3,3] + rkey[2,1,3] >= 0
 R1109: - sbox[0,3,4] + sbox[3,3,4] + rkey[3,0,3] >= 0
 R1110: sbox[0,3,4] + sbox[3,3,4] - rkey[3,0,3] >= 0
 R1111: sbox[0,3,4] - sbox[3,3,4] + rkey[3,0,3] >= 0
 R1112: - sbox[2,3,4] + rkey[1,2,3] + rkey[2,1,3] >= 0
 R1113: sbox[2,3,4] + rkey[1,2,3] - rkey[2,1,3] >= 0
 R1114: sbox[2,3,4] - rkey[1,2,3] + rkey[2,1,3] >= 0
 R1115: sbox[1,3,4] - rkey[0,3,3] = 0
 R1116: tbox[0,3,3] + tbox[2,1,3] - tbox[3,3,4] >= 0
 R1117: tbox[0,3,3] - tbox[2,1,3] + tbox[3,3,4] >= 0
 R1118: - tbox[0,3,3] + tbox[2,1,3] + tbox[3,3,4] >= 0
 R1119: - tbox[0,3,4] + tbox[3,0,3] + tbox[3,3,4] >= 0
 R1120: tbox[0,3,4] - tbox[3,0,3] + tbox[3,3,4] >= 0
 R1121: tbox[0,3,4] + tbox[3,0,3] - tbox[3,3,4] >= 0
 R1122: tbox[1,2,3] + tbox[2,1,3] - tbox[2,3,4] >= 0
 R1123: tbox[1,2,3] - tbox[2,1,3] + tbox[2,3,4] >= 0
 R1124: - tbox[1,2,3] + tbox[2,1,3] + tbox[2,3,4] >= 0
 R1125: - tbox[0,3,3] + tbox[1,3,4] = 0
 R1126: unkn[0,3,3] - unkn[3,3,4] + forg[3,3,4] <= 0
 R1127: unkn[2,1,3] - unkn[3,3,4] + forg[3,3,4] <= 0
 R1128: unkn[0,3,3] + unkn[2,1,3] - unkn[3,3,4] + forg[3,3,4] >= 0
 R1129: - unkn[0,3,4] + unkn[3,3,4] + forg[0,3,4] - forg[3,3,4] <= 0
 R1130: - unkn[0,3,4] + unkn[3,0,3] + forg[0,3,4] <= 0
 R1131: - unkn[0,3,4] + unkn[3,0,3] + unkn[3,3,4] + forg[0,3,4]
   - forg[3,3,4] >= 0
 R1132: unkn[1,2,3] - unkn[2,3,4] + forg[2,3,4] <= 0
 R1133: unkn[2,1,3] - unkn[2,3,4] + forg[2,3,4] <= 0
 R1134: unkn[1,2,3] + unkn[2,1,3] - unkn[2,3,4] + forg[2,3,4] >= 0
 R1135: unkn[0,3,3] - unkn[1,3,4] + forg[1,3,4] = 0
 R1136: - tbox[0,0,4] + forg[0,0,4] <= 0
 R1137: sbox[0,0,4] - rkey[0,0,4] + tkey[0,0,4] >= 0
 R1138: sbox[0,0,4] + rkey[0,0,4] - tkey[0,0,4] >= 0
 R1139: - sbox[0,0,4] + rkey[0,0,4] + tkey[0,0,4] >= 0
 R1140: - tbox[1,0,4] + forg[1,0,4] <= 0
 R1141: sbox[1,0,4] - rkey[1,0,4] + tkey[1,0,4] >= 0
 R1142: sbox[1,0,4] + rkey[1,0,4] - tkey[1,0,4] >= 0
 R1143: - sbox[1,0,4] + rkey[1,0,4] + tkey[1,0,4] >= 0
 R1144: - tbox[2,0,4] + forg[2,0,4] <= 0
 R1145: - sbox[2,0,4] + rkey[2,0,4] = 0
 R1146: - tbox[3,0,4] + forg[3,0,4] <= 0
 R1147: - sbox[3,0,4] + rkey[3,0,4] = 0
 R1148: - sbox[3,0,5] + rkey[0,0,4] + rkey[2,2,4] >= 0
 R1149: sbox[3,0,5] + rkey[0,0,4] - rkey[2,2,4] >= 0
 R1150: sbox[3,0,5] - rkey[0,0,4] + rkey[2,2,4] >= 0
 R1151: - sbox[0,0,5] + sbox[3,0,5] + rkey[3,1,4] >= 0
 R1152: sbox[0,0,5] + sbox[3,0,5] - rkey[3,1,4] >= 0
 R1153: sbox[0,0,5] - sbox[3,0,5] + rkey[3,1,4] >= 0
 R1154: - sbox[2,0,5] + rkey[1,3,4] + rkey[2,2,4] >= 0
 R1155: sbox[2,0,5] + rkey[1,3,4] - rkey[2,2,4] >= 0
 R1156: sbox[2,0,5] - rkey[1,3,4] + rkey[2,2,4] >= 0
 R1157: sbox[1,0,5] - rkey[0,0,4] = 0
 R1158: tbox[0,0,4] + tbox[2,2,4] - tbox[3,0,5] >= 0
 R1159: tbox[0,0,4] - tbox[2,2,4] + tbox[3,0,5] >= 0
 R1160: - tbox[0,0,4] + tbox[2,2,4] + tbox[3,0,5] >= 0
 R1161: - tbox[0,0,5] + tbox[3,0,5] + tbox[3,1,4] >= 0
 R1162: tbox[0,0,5] + tbox[3,0,5] - tbox[3,1,4] >= 0
 R1163: tbox[0,0,5] - tbox[3,0,5] + tbox[3,1,4] >= 0
 R1164: tbox[1,3,4] - tbox[2,0,5] + tbox[2,2,4] >= 0
 R1165: tbox[1,3,4] + tbox[2,0,5] - tbox[2,2,4] >= 0
 R1166: - tbox[1,3,4] + tbox[2,0,5] + tbox[2,2,4] >= 0
 R1167: - tbox[0,0,4] + tbox[1,0,5] = 0
 R1168: unkn[0,0,4] - unkn[3,0,5] + forg[3,0,5] <= 0
 R1169: unkn[2,2,4] - unkn[3,0,5] + forg[3,0,5] <= 0
 R1170: unkn[0,0,4] + unkn[2,2,4] - unkn[3,0,5] + forg[3,0,5] >= 0
 R1171: - unkn[0,0,5] + unkn[3,0,5] + forg[0,0,5] - forg[3,0,5] <= 0
 R1172: - unkn[0,0,5] + unkn[3,1,4] + forg[0,0,5] <= 0
 R1173: - unkn[0,0,5] + unkn[3,0,5] + unkn[3,1,4] + forg[0,0,5]
   - forg[3,0,5] >= 0
 R1174: unkn[1,3,4] - unkn[2,0,5] + forg[2,0,5] <= 0
 R1175: - unkn[2,0,5] + unkn[2,2,4] + forg[2,0,5] <= 0
 R1176: unkn[1,3,4] - unkn[2,0,5] + unkn[2,2,4] + forg[2,0,5] >= 0
 R1177: unkn[0,0,4] - unkn[1,0,5] + forg[1,0,5] = 0
 R1178: - tbox[0,1,4] + forg[0,1,4] <= 0
 R1179: sbox[0,1,4] - rkey[0,1,4] + tkey[0,1,4] >= 0
 R1180: sbox[0,1,4] + rkey[0,1,4] - tkey[0,1,4] >= 0
 R1181: - sbox[0,1,4] + rkey[0,1,4] + tkey[0,1,4] >= 0
 R1182: - tbox[1,1,4] + forg[1,1,4] <= 0
 R1183: sbox[1,1,4] - rkey[1,1,4] + tkey[1,1,4] >= 0
 R1184: sbox[1,1,4] + rkey[1,1,4] - tkey[1,1,4] >= 0
 R1185: - sbox[1,1,4] + rkey[1,1,4] + tkey[1,1,4] >= 0
 R1186: - tbox[2,1,4] + forg[2,1,4] <= 0
 R1187: - sbox[2,1,4] + rkey[2,1,4] = 0
 R1188: - tbox[3,1,4] + forg[3,1,4] <= 0
 R1189: - sbox[3,1,4] + rkey[3,1,4] = 0
 R1190: - sbox[3,1,5] + rkey[0,1,4] + rkey[2,3,4] >= 0
 R1191: sbox[3,1,5] + rkey[0,1,4] - rkey[2,3,4] >= 0
 R1192: sbox[3,1,5] - rkey[0,1,4] + rkey[2,3,4] >= 0
 R1193: - sbox[0,1,5] + sbox[3,1,5] + rkey[3,2,4] >= 0
 R1194: sbox[0,1,5] + sbox[3,1,5] - rkey[3,2,4] >= 0
 R1195: sbox[0,1,5] - sbox[3,1,5] + rkey[3,2,4] >= 0
 R1196: - sbox[2,1,5] + rkey[1,0,4] + rkey[2,3,4] >= 0
 R1197: sbox[2,1,5] + rkey[1,0,4] - rkey[2,3,4] >= 0
 R1198: sbox[2,1,5] - rkey[1,0,4] + rkey[2,3,4] >= 0
 R1199: sbox[1,1,5] - rkey[0,1,4] = 0
 R1200: tbox[0,1,4] + tbox[2,3,4] - tbox[3,1,5] >= 0
 R1201: tbox[0,1,4] - tbox[2,3,4] + tbox[3,1,5] >= 0
 R1202: - tbox[0,1,4] + tbox[2,3,4] + tbox[3,1,5] >= 0
 R1203: - tbox[0,1,5] + tbox[3,1,5] + tbox[3,2,4] >= 0
 R1204: tbox[0,1,5] + tbox[3,1,5] - tbox[3,2,4] >= 0
 R1205: tbox[0,1,5] - tbox[3,1,5] + tbox[3,2,4] >= 0
 R1206: tbox[1,0,4] - tbox[2,1,5] + tbox[2,3,4] >= 0
 R1207: tbox[1,0,4] + tbox[2,1,5] - tbox[2,3,4] >= 0
 R1208: - tbox[1,0,4] + tbox[2,1,5] + tbox[2,3,4] >= 0
 R1209: - tbox[0,1,4] + tbox[1,1,5] = 0
 R1210: unkn[0,1,4] - unkn[3,1,5] + forg[3,1,5] <= 0
 R1211: unkn[2,3,4] - unkn[3,1,5] + forg[3,1,5] <= 0
 R1212: unkn[0,1,4] + unkn[2,3,4] - unkn[3,1,5] + forg[3,1,5] >= 0
 R1213: - unkn[0,1,5] + unkn[3,1,5] + forg[0,1,5] - forg[3,1,5] <= 0
 R1214: - unkn[0,1,5] + unkn[3,2,4] + forg[0,1,5] <= 0
 R1215: - unkn[0,1,5] + unkn[3,1,5] + unkn[3,2,4] + forg[0,1,5]
   - forg[3,1,5] >= 0
 R1216: unkn[1,0,4] - unkn[2,1,5] + forg[2,1,5] <= 0
 R1217: - unkn[2,1,5] + unkn[2,3,4] + forg[2,1,5] <= 0
 R1218: unkn[1,0,4] - unkn[2,1,5] + unkn[2,3,4] + forg[2,1,5] >= 0
 R1219: unkn[0,1,4] - unkn[1,1,5] + forg[1,1,5] = 0
 R1220: - tbox[0,2,4] + forg[0,2,4] <= 0
 R1221: sbox[0,2,4] - rkey[0,2,4] + tkey[0,2,4] >= 0
 R1222: sbox[0,2,4] + rkey[0,2,4] - tkey[0,2,4] >= 0
 R1223: - sbox[0,2,4] + rkey[0,2,4] + tkey[0,2,4] >= 0
 R1224: - tbox[1,2,4] + forg[1,2,4] <= 0
 R1225: sbox[1,2,4] - rkey[1,2,4] + tkey[1,2,4] >= 0
 R1226: sbox[1,2,4] + rkey[1,2,4] - tkey[1,2,4] >= 0
 R1227: - sbox[1,2,4] + rkey[1,2,4] + tkey[1,2,4] >= 0
 R1228: - tbox[2,2,4] + forg[2,2,4] <= 0
 R1229: - sbox[2,2,4] + rkey[2,2,4] = 0
 R1230: - tbox[3,2,4] + forg[3,2,4] <= 0
 R1231: - sbox[3,2,4] + rkey[3,2,4] = 0
 R1232: - sbox[3,2,5] + rkey[0,2,4] + rkey[2,0,4] >= 0
 R1233: sbox[3,2,5] + rkey[0,2,4] - rkey[2,0,4] >= 0
 R1234: sbox[3,2,5] - rkey[0,2,4] + rkey[2,0,4] >= 0
 R1235: - sbox[0,2,5] + sbox[3,2,5] + rkey[3,3,4] >= 0
 R1236: sbox[0,2,5] + sbox[3,2,5] - rkey[3,3,4] >= 0
 R1237: sbox[0,2,5] - sbox[3,2,5] + rkey[3,3,4] >= 0
 R1238: - sbox[2,2,5] + rkey[1,1,4] + rkey[2,0,4] >= 0
 R1239: sbox[2,2,5] + rkey[1,1,4] - rkey[2,0,4] >= 0
 R1240: sbox[2,2,5] - rkey[1,1,4] + rkey[2,0,4] >= 0
 R1241: sbox[1,2,5] - rkey[0,2,4] = 0
 R1242: tbox[0,2,4] + tbox[2,0,4] - tbox[3,2,5] >= 0
 R1243: tbox[0,2,4] - tbox[2,0,4] + tbox[3,2,5] >= 0
 R1244: - tbox[0,2,4] + tbox[2,0,4] + tbox[3,2,5] >= 0
 R1245: - tbox[0,2,5] + tbox[3,2,5] + tbox[3,3,4] >= 0
 R1246: tbox[0,2,5] + tbox[3,2,5] - tbox[3,3,4] >= 0
 R1247: tbox[0,2,5] - tbox[3,2,5] + tbox[3,3,4] >= 0
 R1248: tbox[1,1,4] + tbox[2,0,4] - tbox[2,2,5] >= 0
 R1249: tbox[1,1,4] - tbox[2,0,4] + tbox[2,2,5] >= 0
 R1250: - tbox[1,1,4] + tbox[2,0,4] + tbox[2,2,5] >= 0
 R1251: - tbox[0,2,4] + tbox[1,2,5] = 0
 R1252: unkn[0,2,4] - unkn[3,2,5] + forg[3,2,5] <= 0
 R1253: unkn[2,0,4] - unkn[3,2,5] + forg[3,2,5] <= 0
 R1254: unkn[0,2,4] + unkn[2,0,4] - unkn[3,2,5] + forg[3,2,5] >= 0
 R1255: - unkn[0,2,5] + unkn[3,2,5] + forg[0,2,5] - forg[3,2,5] <= 0
 R1256: - unkn[0,2,5] + unkn[3,3,4] + forg[0,2,5] <= 0
 R1257: - unkn[0,2,5] + unkn[3,2,5] + unkn[3,3,4] + forg[0,2,5]
   - forg[3,2,5] >= 0
 R1258: unkn[1,1,4] - unkn[2,2,5] + forg[2,2,5] <= 0
 R1259: unkn[2,0,4] - unkn[2,2,5] + forg[2,2,5] <= 0
 R1260: unkn[1,1,4] + unkn[2,0,4] - unkn[2,2,5] + forg[2,2,5] >= 0
 R1261: unkn[0,2,4] - unkn[1,2,5] + forg[1,2,5] = 0
 R1262: - tbox[0,3,4] + forg[0,3,4] <= 0
 R1263: sbox[0,3,4] - rkey[0,3,4] + tkey[0,3,4] >= 0
 R1264: sbox[0,3,4] + rkey[0,3,4] - tkey[0,3,4] >= 0
 R1265: - sbox[0,3,4] + rkey[0,3,4] + tkey[0,3,4] >= 0
 R1266: - tbox[1,3,4] + forg[1,3,4] <= 0
 R1267: sbox[1,3,4] - rkey[1,3,4] + tkey[1,3,4] >= 0
 R1268: sbox[1,3,4] + rkey[1,3,4] - tkey[1,3,4] >= 0
 R1269: - sbox[1,3,4] + rkey[1,3,4] + tkey[1,3,4] >= 0
 R1270: - tbox[2,3,4] + forg[2,3,4] <= 0
 R1271: - sbox[2,3,4] + rkey[2,3,4] = 0
 R1272: - tbox[3,3,4] + forg[3,3,4] <= 0
 R1273: - sbox[3,3,4] + rkey[3,3,4] = 0
 R1274: - sbox[3,3,5] + rkey[0,3,4] + rkey[2,1,4] >= 0
 R1275: sbox[3,3,5] + rkey[0,3,4] - rkey[2,1,4] >= 0
 R1276: sbox[3,3,5] - rkey[0,3,4] + rkey[2,1,4] >= 0
 R1277: - sbox[0,3,5] + sbox[3,3,5] + rkey[3,0,4] >= 0
 R1278: sbox[0,3,5] + sbox[3,3,5] - rkey[3,0,4] >= 0
 R1279: sbox[0,3,5] - sbox[3,3,5] + rkey[3,0,4] >= 0
 R1280: - sbox[2,3,5] + rkey[1,2,4] + rkey[2,1,4] >= 0
 R1281: sbox[2,3,5] + rkey[1,2,4] - rkey[2,1,4] >= 0
 R1282: sbox[2,3,5] - rkey[1,2,4] + rkey[2,1,4] >= 0
 R1283: sbox[1,3,5] - rkey[0,3,4] = 0
 R1284: tbox[0,3,4] + tbox[2,1,4] - tbox[3,3,5] >= 0
 R1285: tbox[0,3,4] - tbox[2,1,4] + tbox[3,3,5] >= 0
 R1286: - tbox[0,3,4] + tbox[2,1,4] + tbox[3,3,5] >= 0
 R1287: - tbox[0,3,5] + tbox[3,0,4] + tbox[3,3,5] >= 0
 R1288: tbox[0,3,5] - tbox[3,0,4] + tbox[3,3,5] >= 0
 R1289: tbox[0,3,5] + tbox[3,0,4] - tbox[3,3,5] >= 0
 R1290: tbox[1,2,4] + tbox[2,1,4] - tbox[2,3,5] >= 0
 R1291: tbox[1,2,4] - tbox[2,1,4] + tbox[2,3,5] >= 0
 R1292: - tbox[1,2,4] + tbox[2,1,4] + tbox[2,3,5] >= 0
 R1293: - tbox[0,3,4] + tbox[1,3,5] = 0
 R1294: unkn[0,3,4] - unkn[3,3,5] + forg[3,3,5] <= 0
 R1295: unkn[2,1,4] - unkn[3,3,5] + forg[3,3,5] <= 0
 R1296: unkn[0,3,4] + unkn[2,1,4] - unkn[3,3,5] + forg[3,3,5] >= 0
 R1297: - unkn[0,3,5] + unkn[3,3,5] + forg[0,3,5] - forg[3,3,5] <= 0
 R1298: - unkn[0,3,5] + unkn[3,0,4] + forg[0,3,5] <= 0
 R1299: - unkn[0,3,5] + unkn[3,0,4] + unkn[3,3,5] + forg[0,3,5]
   - forg[3,3,5] >= 0
 R1300: unkn[1,2,4] - unkn[2,3,5] + forg[2,3,5] <= 0
 R1301: unkn[2,1,4] - unkn[2,3,5] + forg[2,3,5] <= 0
 R1302: unkn[1,2,4] + unkn[2,1,4] - unkn[2,3,5] + forg[2,3,5] >= 0
 R1303: unkn[0,3,4] - unkn[1,3,5] + forg[1,3,5] = 0
 R1304: - tbox[0,0,5] + forg[0,0,5] <= 0
 R1305: sbox[0,0,5] - rkey[0,0,5] + tkey[0,0,5] >= 0
 R1306: sbox[0,0,5] + rkey[0,0,5] - tkey[0,0,5] >= 0
 R1307: - sbox[0,0,5] + rkey[0,0,5] + tkey[0,0,5] >= 0
 R1308: - tbox[1,0,5] + forg[1,0,5] <= 0
 R1309: sbox[1,0,5] - rkey[1,0,5] + tkey[1,0,5] >= 0
 R1310: sbox[1,0,5] + rkey[1,0,5] - tkey[1,0,5] >= 0
 R1311: - sbox[1,0,5] + rkey[1,0,5] + tkey[1,0,5] >= 0
 R1312: - tbox[2,0,5] + forg[2,0,5] <= 0
 R1313: - sbox[2,0,5] + rkey[2,0,5] = 0
 R1314: - tbox[3,0,5] + forg[3,0,5] <= 0
 R1315: - sbox[3,0,5] + rkey[3,0,5] = 0
 R1316: - sbox[3,0,6] + rkey[0,0,5] + rkey[2,2,5] >= 0
 R1317: sbox[3,0,6] + rkey[0,0,5] - rkey[2,2,5] >= 0
 R1318: sbox[3,0,6] - rkey[0,0,5] + rkey[2,2,5] >= 0
 R1319: - sbox[0,0,6] + sbox[3,0,6] + rkey[3,1,5] >= 0
 R1320: sbox[0,0,6] + sbox[3,0,6] - rkey[3,1,5] >= 0
 R1321: sbox[0,0,6] - sbox[3,0,6] + rkey[3,1,5] >= 0
 R1322: - sbox[2,0,6] + rkey[1,3,5] + rkey[2,2,5] >= 0
 R1323: sbox[2,0,6] + rkey[1,3,5] - rkey[2,2,5] >= 0
 R1324: sbox[2,0,6] - rkey[1,3,5] + rkey[2,2,5] >= 0
 R1325: sbox[1,0,6] - rkey[0,0,5] = 0
 R1326: tbox[0,0,5] + tbox[2,2,5] - tbox[3,0,6] >= 0
 R1327: tbox[0,0,5] - tbox[2,2,5] + tbox[3,0,6] >= 0
 R1328: - tbox[0,0,5] + tbox[2,2,5] + tbox[3,0,6] >= 0
 R1329: - tbox[0,0,6] + tbox[3,0,6] + tbox[3,1,5] >= 0
 R1330: tbox[0,0,6] + tbox[3,0,6] - tbox[3,1,5] >= 0
 R1331: tbox[0,0,6] - tbox[3,0,6] + tbox[3,1,5] >= 0
 R1332: tbox[1,3,5] - tbox[2,0,6] + tbox[2,2,5] >= 0
 R1333: tbox[1,3,5] + tbox[2,0,6] - tbox[2,2,5] >= 0
 R1334: - tbox[1,3,5] + tbox[2,0,6] + tbox[2,2,5] >= 0
 R1335: - tbox[0,0,5] + tbox[1,0,6] = 0
 R1336: unkn[0,0,5] - unkn[3,0,6] + forg[3,0,6] <= 0
 R1337: unkn[2,2,5] - unkn[3,0,6] + forg[3,0,6] <= 0
 R1338: unkn[0,0,5] + unkn[2,2,5] - unkn[3,0,6] + forg[3,0,6] >= 0
 R1339: - unkn[0,0,6] + unkn[3,0,6] + forg[0,0,6] - forg[3,0,6] <= 0
 R1340: - unkn[0,0,6] + unkn[3,1,5] + forg[0,0,6] <= 0
 R1341: - unkn[0,0,6] + unkn[3,0,6] + unkn[3,1,5] + forg[0,0,6]
   - forg[3,0,6] >= 0
 R1342: unkn[1,3,5] - unkn[2,0,6] + forg[2,0,6] <= 0
 R1343: - unkn[2,0,6] + unkn[2,2,5] + forg[2,0,6] <= 0
 R1344: unkn[1,3,5] - unkn[2,0,6] + unkn[2,2,5] + forg[2,0,6] >= 0
 R1345: unkn[0,0,5] - unkn[1,0,6] + forg[1,0,6] = 0
 R1346: - tbox[0,1,5] + forg[0,1,5] <= 0
 R1347: sbox[0,1,5] - rkey[0,1,5] + tkey[0,1,5] >= 0
 R1348: sbox[0,1,5] + rkey[0,1,5] - tkey[0,1,5] >= 0
 R1349: - sbox[0,1,5] + rkey[0,1,5] + tkey[0,1,5] >= 0
 R1350: - tbox[1,1,5] + forg[1,1,5] <= 0
 R1351: sbox[1,1,5] - rkey[1,1,5] + tkey[1,1,5] >= 0
 R1352: sbox[1,1,5] + rkey[1,1,5] - tkey[1,1,5] >= 0
 R1353: - sbox[1,1,5] + rkey[1,1,5] + tkey[1,1,5] >= 0
 R1354: - tbox[2,1,5] + forg[2,1,5] <= 0
 R1355: - sbox[2,1,5] + rkey[2,1,5] = 0
 R1356: - tbox[3,1,5] + forg[3,1,5] <= 0
 R1357: - sbox[3,1,5] + rkey[3,1,5] = 0
 R1358: - sbox[3,1,6] + rkey[0,1,5] + rkey[2,3,5] >= 0
 R1359: sbox[3,1,6] + rkey[0,1,5] - rkey[2,3,5] >= 0
 R1360: sbox[3,1,6] - rkey[0,1,5] + rkey[2,3,5] >= 0
 R1361: - sbox[0,1,6] + sbox[3,1,6] + rkey[3,2,5] >= 0
 R1362: sbox[0,1,6] + sbox[3,1,6] - rkey[3,2,5] >= 0
 R1363: sbox[0,1,6] - sbox[3,1,6] + rkey[3,2,5] >= 0
 R1364: - sbox[2,1,6] + rkey[1,0,5] + rkey[2,3,5] >= 0
 R1365: sbox[2,1,6] + rkey[1,0,5] - rkey[2,3,5] >= 0
 R1366: sbox[2,1,6] - rkey[1,0,5] + rkey[2,3,5] >= 0
 R1367: sbox[1,1,6] - rkey[0,1,5] = 0
 R1368: tbox[0,1,5] + tbox[2,3,5] - tbox[3,1,6] >= 0
 R1369: tbox[0,1,5] - tbox[2,3,5] + tbox[3,1,6] >= 0
 R1370: - tbox[0,1,5] + tbox[2,3,5] + tbox[3,1,6] >= 0
 R1371: - tbox[0,1,6] + tbox[3,1,6] + tbox[3,2,5] >= 0
 R1372: tbox[0,1,6] + tbox[3,1,6] - tbox[3,2,5] >= 0
 R1373: tbox[0,1,6] - tbox[3,1,6] + tbox[3,2,5] >= 0
 R1374: tbox[1,0,5] - tbox[2,1,6] + tbox[2,3,5] >= 0
 R1375: tbox[1,0,5] + tbox[2,1,6] - tbox[2,3,5] >= 0
 R1376: - tbox[1,0,5] + tbox[2,1,6] + tbox[2,3,5] >= 0
 R1377: - tbox[0,1,5] + tbox[1,1,6] = 0
 R1378: unkn[0,1,5] - unkn[3,1,6] + forg[3,1,6] <= 0
 R1379: unkn[2,3,5] - unkn[3,1,6] + forg[3,1,6] <= 0
 R1380: unkn[0,1,5] + unkn[2,3,5] - unkn[3,1,6] + forg[3,1,6] >= 0
 R1381: - unkn[0,1,6] + unkn[3,1,6] + forg[0,1,6] - forg[3,1,6] <= 0
 R1382: - unkn[0,1,6] + unkn[3,2,5] + forg[0,1,6] <= 0
 R1383: - unkn[0,1,6] + unkn[3,1,6] + unkn[3,2,5] + forg[0,1,6]
   - forg[3,1,6] >= 0
 R1384: unkn[1,0,5] - unkn[2,1,6] + forg[2,1,6] <= 0
 R1385: - unkn[2,1,6] + unkn[2,3,5] + forg[2,1,6] <= 0
 R1386: unkn[1,0,5] - unkn[2,1,6] + unkn[2,3,5] + forg[2,1,6] >= 0
 R1387: unkn[0,1,5] - unkn[1,1,6] + forg[1,1,6] = 0
 R1388: - tbox[0,2,5] + forg[0,2,5] <= 0
 R1389: sbox[0,2,5] - rkey[0,2,5] + tkey[0,2,5] >= 0
 R1390: sbox[0,2,5] + rkey[0,2,5] - tkey[0,2,5] >= 0
 R1391: - sbox[0,2,5] + rkey[0,2,5] + tkey[0,2,5] >= 0
 R1392: - tbox[1,2,5] + forg[1,2,5] <= 0
 R1393: sbox[1,2,5] - rkey[1,2,5] + tkey[1,2,5] >= 0
 R1394: sbox[1,2,5] + rkey[1,2,5] - tkey[1,2,5] >= 0
 R1395: - sbox[1,2,5] + rkey[1,2,5] + tkey[1,2,5] >= 0
 R1396: - tbox[2,2,5] + forg[2,2,5] <= 0
 R1397: - sbox[2,2,5] + rkey[2,2,5] = 0
 R1398: - tbox[3,2,5] + forg[3,2,5] <= 0
 R1399: - sbox[3,2,5] + rkey[3,2,5] = 0
 R1400: - sbox[3,2,6] + rkey[0,2,5] + rkey[2,0,5] >= 0
 R1401: sbox[3,2,6] + rkey[0,2,5] - rkey[2,0,5] >= 0
 R1402: sbox[3,2,6] - rkey[0,2,5] + rkey[2,0,5] >= 0
 R1403: - sbox[0,2,6] + sbox[3,2,6] + rkey[3,3,5] >= 0
 R1404: sbox[0,2,6] + sbox[3,2,6] - rkey[3,3,5] >= 0
 R1405: sbox[0,2,6] - sbox[3,2,6] + rkey[3,3,5] >= 0
 R1406: - sbox[2,2,6] + rkey[1,1,5] + rkey[2,0,5] >= 0
 R1407: sbox[2,2,6] + rkey[1,1,5] - rkey[2,0,5] >= 0
 R1408: sbox[2,2,6] - rkey[1,1,5] + rkey[2,0,5] >= 0
 R1409: sbox[1,2,6] - rkey[0,2,5] = 0
 R1410: tbox[0,2,5] + tbox[2,0,5] - tbox[3,2,6] >= 0
 R1411: tbox[0,2,5] - tbox[2,0,5] + tbox[3,2,6] >= 0
 R1412: - tbox[0,2,5] + tbox[2,0,5] + tbox[3,2,6] >= 0
 R1413: - tbox[0,2,6] + tbox[3,2,6] + tbox[3,3,5] >= 0
 R1414: tbox[0,2,6] + tbox[3,2,6] - tbox[3,3,5] >= 0
 R1415: tbox[0,2,6] - tbox[3,2,6] + tbox[3,3,5] >= 0
 R1416: tbox[1,1,5] + tbox[2,0,5] - tbox[2,2,6] >= 0
 R1417: tbox[1,1,5] - tbox[2,0,5] + tbox[2,2,6] >= 0
 R1418: - tbox[1,1,5] + tbox[2,0,5] + tbox[2,2,6] >= 0
 R1419: - tbox[0,2,5] + tbox[1,2,6] = 0
 R1420: unkn[0,2,5] - unkn[3,2,6] + forg[3,2,6] <= 0
 R1421: unkn[2,0,5] - unkn[3,2,6] + forg[3,2,6] <= 0
 R1422: unkn[0,2,5] + unkn[2,0,5] - unkn[3,2,6] + forg[3,2,6] >= 0
 R1423: - unkn[0,2,6] + unkn[3,2,6] + forg[0,2,6] - forg[3,2,6] <= 0
 R1424: - unkn[0,2,6] + unkn[3,3,5] + forg[0,2,6] <= 0
 R1425: - unkn[0,2,6] + unkn[3,2,6] + unkn[3,3,5] + forg[0,2,6]
   - forg[3,2,6] >= 0
 R1426: unkn[1,1,5] - unkn[2,2,6] + forg[2,2,6] <= 0
 R1427: unkn[2,0,5] - unkn[2,2,6] + forg[2,2,6] <= 0
 R1428: unkn[1,1,5] + unkn[2,0,5] - unkn[2,2,6] + forg[2,2,6] >= 0
 R1429: unkn[0,2,5] - unkn[1,2,6] + forg[1,2,6] = 0
 R1430: - tbox[0,3,5] + forg[0,3,5] <= 0
 R1431: sbox[0,3,5] - rkey[0,3,5] + tkey[0,3,5] >= 0
 R1432: sbox[0,3,5] + rkey[0,3,5] - tkey[0,3,5] >= 0
 R1433: - sbox[0,3,5] + rkey[0,3,5] + tkey[0,3,5] >= 0
 R1434: - tbox[1,3,5] + forg[1,3,5] <= 0
 R1435: sbox[1,3,5] - rkey[1,3,5] + tkey[1,3,5] >= 0
 R1436: sbox[1,3,5] + rkey[1,3,5] - tkey[1,3,5] >= 0
 R1437: - sbox[1,3,5] + rkey[1,3,5] + tkey[1,3,5] >= 0
 R1438: - tbox[2,3,5] + forg[2,3,5] <= 0
 R1439: - sbox[2,3,5] + rkey[2,3,5] = 0
 R1440: - tbox[3,3,5] + forg[3,3,5] <= 0
 R1441: - sbox[3,3,5] + rkey[3,3,5] = 0
 R1442: - sbox[3,3,6] + rkey[0,3,5] + rkey[2,1,5] >= 0
 R1443: sbox[3,3,6] + rkey[0,3,5] - rkey[2,1,5] >= 0
 R1444: sbox[3,3,6] - rkey[0,3,5] + rkey[2,1,5] >= 0
 R1445: - sbox[0,3,6] + sbox[3,3,6] + rkey[3,0,5] >= 0
 R1446: sbox[0,3,6] + sbox[3,3,6] - rkey[3,0,5] >= 0
 R1447: sbox[0,3,6] - sbox[3,3,6] + rkey[3,0,5] >= 0
 R1448: - sbox[2,3,6] + rkey[1,2,5] + rkey[2,1,5] >= 0
 R1449: sbox[2,3,6] + rkey[1,2,5] - rkey[2,1,5] >= 0
 R1450: sbox[2,3,6] - rkey[1,2,5] + rkey[2,1,5] >= 0
 R1451: sbox[1,3,6] - rkey[0,3,5] = 0
 R1452: tbox[0,3,5] + tbox[2,1,5] - tbox[3,3,6] >= 0
 R1453: tbox[0,3,5] - tbox[2,1,5] + tbox[3,3,6] >= 0
 R1454: - tbox[0,3,5] + tbox[2,1,5] + tbox[3,3,6] >= 0
 R1455: - tbox[0,3,6] + tbox[3,0,5] + tbox[3,3,6] >= 0
 R1456: tbox[0,3,6] - tbox[3,0,5] + tbox[3,3,6] >= 0
 R1457: tbox[0,3,6] + tbox[3,0,5] - tbox[3,3,6] >= 0
 R1458: tbox[1,2,5] + tbox[2,1,5] - tbox[2,3,6] >= 0
 R1459: tbox[1,2,5] - tbox[2,1,5] + tbox[2,3,6] >= 0
 R1460: - tbox[1,2,5] + tbox[2,1,5] + tbox[2,3,6] >= 0
 R1461: - tbox[0,3,5] + tbox[1,3,6] = 0
 R1462: unkn[0,3,5] - unkn[3,3,6] + forg[3,3,6] <= 0
 R1463: unkn[2,1,5] - unkn[3,3,6] + forg[3,3,6] <= 0
 R1464: unkn[0,3,5] + unkn[2,1,5] - unkn[3,3,6] + forg[3,3,6] >= 0
 R1465: - unkn[0,3,6] + unkn[3,3,6] + forg[0,3,6] - forg[3,3,6] <= 0
 R1466: - unkn[0,3,6] + unkn[3,0,5] + forg[0,3,6] <= 0
 R1467: - unkn[0,3,6] + unkn[3,0,5] + unkn[3,3,6] + forg[0,3,6]
   - forg[3,3,6] >= 0
 R1468: unkn[1,2,5] - unkn[2,3,6] + forg[2,3,6] <= 0
 R1469: unkn[2,1,5] - unkn[2,3,6] + forg[2,3,6] <= 0
 R1470: unkn[1,2,5] + unkn[2,1,5] - unkn[2,3,6] + forg[2,3,6] >= 0
 R1471: unkn[0,3,5] - unkn[1,3,6] + forg[1,3,6] = 0
 R1472: - tbox[0,0,6] + forg[0,0,6] <= 0
 R1473: sbox[0,0,6] - rkey[0,0,6] + tkey[0,0,6] >= 0
 R1474: sbox[0,0,6] + rkey[0,0,6] - tkey[0,0,6] >= 0
 R1475: - sbox[0,0,6] + rkey[0,0,6] + tkey[0,0,6] >= 0
 R1476: - tbox[1,0,6] + forg[1,0,6] <= 0
 R1477: sbox[1,0,6] - rkey[1,0,6] + tkey[1,0,6] >= 0
 R1478: sbox[1,0,6] + rkey[1,0,6] - tkey[1,0,6] >= 0
 R1479: - sbox[1,0,6] + rkey[1,0,6] + tkey[1,0,6] >= 0
 R1480: - tbox[2,0,6] + forg[2,0,6] <= 0
 R1481: - sbox[2,0,6] + rkey[2,0,6] = 0
 R1482: - tbox[3,0,6] + forg[3,0,6] <= 0
 R1483: - sbox[3,0,6] + rkey[3,0,6] = 0
 R1484: - sbox[3,0,7] + rkey[0,0,6] + rkey[2,2,6] >= 0
 R1485: sbox[3,0,7] + rkey[0,0,6] - rkey[2,2,6] >= 0
 R1486: sbox[3,0,7] - rkey[0,0,6] + rkey[2,2,6] >= 0
 R1487: - sbox[0,0,7] + sbox[3,0,7] + rkey[3,1,6] >= 0
 R1488: sbox[0,0,7] + sbox[3,0,7] - rkey[3,1,6] >= 0
 R1489: sbox[0,0,7] - sbox[3,0,7] + rkey[3,1,6] >= 0
 R1490: - sbox[2,0,7] + rkey[1,3,6] + rkey[2,2,6] >= 0
 R1491: sbox[2,0,7] + rkey[1,3,6] - rkey[2,2,6] >= 0
 R1492: sbox[2,0,7] - rkey[1,3,6] + rkey[2,2,6] >= 0
 R1493: sbox[1,0,7] - rkey[0,0,6] = 0
 R1494: tbox[0,0,6] + tbox[2,2,6] - tbox[3,0,7] >= 0
 R1495: tbox[0,0,6] - tbox[2,2,6] + tbox[3,0,7] >= 0
 R1496: - tbox[0,0,6] + tbox[2,2,6] + tbox[3,0,7] >= 0
 R1497: - tbox[0,0,7] + tbox[3,0,7] + tbox[3,1,6] >= 0
 R1498: tbox[0,0,7] + tbox[3,0,7] - tbox[3,1,6] >= 0
 R1499: tbox[0,0,7] - tbox[3,0,7] + tbox[3,1,6] >= 0
 R1500: tbox[1,3,6] - tbox[2,0,7] + tbox[2,2,6] >= 0
 R1501: tbox[1,3,6] + tbox[2,0,7] - tbox[2,2,6] >= 0
 R1502: - tbox[1,3,6] + tbox[2,0,7] + tbox[2,2,6] >= 0
 R1503: - tbox[0,0,6] + tbox[1,0,7] = 0
 R1504: unkn[0,0,6] - unkn[3,0,7] + forg[3,0,7] <= 0
 R1505: unkn[2,2,6] - unkn[3,0,7] + forg[3,0,7] <= 0
 R1506: unkn[0,0,6] + unkn[2,2,6] - unkn[3,0,7] + forg[3,0,7] >= 0
 R1507: - unkn[0,0,7] + unkn[3,0,7] + forg[0,0,7] - forg[3,0,7] <= 0
 R1508: - unkn[0,0,7] + unkn[3,1,6] + forg[0,0,7] <= 0
 R1509: - unkn[0,0,7] + unkn[3,0,7] + unkn[3,1,6] + forg[0,0,7]
   - forg[3,0,7] >= 0
 R1510: unkn[1,3,6] - unkn[2,0,7] + forg[2,0,7] <= 0
 R1511: - unkn[2,0,7] + unkn[2,2,6] + forg[2,0,7] <= 0
 R1512: unkn[1,3,6] - unkn[2,0,7] + unkn[2,2,6] + forg[2,0,7] >= 0
 R1513: unkn[0,0,6] - unkn[1,0,7] + forg[1,0,7] = 0
 R1514: - tbox[0,1,6] + forg[0,1,6] <= 0
 R1515: sbox[0,1,6] - rkey[0,1,6] + tkey[0,1,6] >= 0
 R1516: sbox[0,1,6] + rkey[0,1,6] - tkey[0,1,6] >= 0
 R1517: - sbox[0,1,6] + rkey[0,1,6] + tkey[0,1,6] >= 0
 R1518: - tbox[1,1,6] + forg[1,1,6] <= 0
 R1519: sbox[1,1,6] - rkey[1,1,6] + tkey[1,1,6] >= 0
 R1520: sbox[1,1,6] + rkey[1,1,6] - tkey[1,1,6] >= 0
 R1521: - sbox[1,1,6] + rkey[1,1,6] + tkey[1,1,6] >= 0
 R1522: - tbox[2,1,6] + forg[2,1,6] <= 0
 R1523: - sbox[2,1,6] + rkey[2,1,6] = 0
 R1524: - tbox[3,1,6] + forg[3,1,6] <= 0
 R1525: - sbox[3,1,6] + rkey[3,1,6] = 0
 R1526: - sbox[3,1,7] + rkey[0,1,6] + rkey[2,3,6] >= 0
 R1527: sbox[3,1,7] + rkey[0,1,6] - rkey[2,3,6] >= 0
 R1528: sbox[3,1,7] - rkey[0,1,6] + rkey[2,3,6] >= 0
 R1529: - sbox[0,1,7] + sbox[3,1,7] + rkey[3,2,6] >= 0
 R1530: sbox[0,1,7] + sbox[3,1,7] - rkey[3,2,6] >= 0
 R1531: sbox[0,1,7] - sbox[3,1,7] + rkey[3,2,6] >= 0
 R1532: - sbox[2,1,7] + rkey[1,0,6] + rkey[2,3,6] >= 0
 R1533: sbox[2,1,7] + rkey[1,0,6] - rkey[2,3,6] >= 0
 R1534: sbox[2,1,7] - rkey[1,0,6] + rkey[2,3,6] >= 0
 R1535: sbox[1,1,7] - rkey[0,1,6] = 0
 R1536: tbox[0,1,6] + tbox[2,3,6] - tbox[3,1,7] >= 0
 R1537: tbox[0,1,6] - tbox[2,3,6] + tbox[3,1,7] >= 0
 R1538: - tbox[0,1,6] + tbox[2,3,6] + tbox[3,1,7] >= 0
 R1539: - tbox[0,1,7] + tbox[3,1,7] + tbox[3,2,6] >= 0
 R1540: tbox[0,1,7] + tbox[3,1,7] - tbox[3,2,6] >= 0
 R1541: tbox[0,1,7] - tbox[3,1,7] + tbox[3,2,6] >= 0
 R1542: tbox[1,0,6] - tbox[2,1,7] + tbox[2,3,6] >= 0
 R1543: tbox[1,0,6] + tbox[2,1,7] - tbox[2,3,6] >= 0
 R1544: - tbox[1,0,6] + tbox[2,1,7] + tbox[2,3,6] >= 0
 R1545: - tbox[0,1,6] + tbox[1,1,7] = 0
 R1546: unkn[0,1,6] - unkn[3,1,7] + forg[3,1,7] <= 0
 R1547: unkn[2,3,6] - unkn[3,1,7] + forg[3,1,7] <= 0
 R1548: unkn[0,1,6] + unkn[2,3,6] - unkn[3,1,7] + forg[3,1,7] >= 0
 R1549: - unkn[0,1,7] + unkn[3,1,7] + forg[0,1,7] - forg[3,1,7] <= 0
 R1550: - unkn[0,1,7] + unkn[3,2,6] + forg[0,1,7] <= 0
 R1551: - unkn[0,1,7] + unkn[3,1,7] + unkn[3,2,6] + forg[0,1,7]
   - forg[3,1,7] >= 0
 R1552: unkn[1,0,6] - unkn[2,1,7] + forg[2,1,7] <= 0
 R1553: - unkn[2,1,7] + unkn[2,3,6] + forg[2,1,7] <= 0
 R1554: unkn[1,0,6] - unkn[2,1,7] + unkn[2,3,6] + forg[2,1,7] >= 0
 R1555: unkn[0,1,6] - unkn[1,1,7] + forg[1,1,7] = 0
 R1556: - tbox[0,2,6] + forg[0,2,6] <= 0
 R1557: sbox[0,2,6] - rkey[0,2,6] + tkey[0,2,6] >= 0
 R1558: sbox[0,2,6] + rkey[0,2,6] - tkey[0,2,6] >= 0
 R1559: - sbox[0,2,6] + rkey[0,2,6] + tkey[0,2,6] >= 0
 R1560: - tbox[1,2,6] + forg[1,2,6] <= 0
 R1561: sbox[1,2,6] - rkey[1,2,6] + tkey[1,2,6] >= 0
 R1562: sbox[1,2,6] + rkey[1,2,6] - tkey[1,2,6] >= 0
 R1563: - sbox[1,2,6] + rkey[1,2,6] + tkey[1,2,6] >= 0
 R1564: - tbox[2,2,6] + forg[2,2,6] <= 0
 R1565: - sbox[2,2,6] + rkey[2,2,6] = 0
 R1566: - tbox[3,2,6] + forg[3,2,6] <= 0
 R1567: - sbox[3,2,6] + rkey[3,2,6] = 0
 R1568: - sbox[3,2,7] + rkey[0,2,6] + rkey[2,0,6] >= 0
 R1569: sbox[3,2,7] + rkey[0,2,6] - rkey[2,0,6] >= 0
 R1570: sbox[3,2,7] - rkey[0,2,6] + rkey[2,0,6] >= 0
 R1571: - sbox[0,2,7] + sbox[3,2,7] + rkey[3,3,6] >= 0
 R1572: sbox[0,2,7] + sbox[3,2,7] - rkey[3,3,6] >= 0
 R1573: sbox[0,2,7] - sbox[3,2,7] + rkey[3,3,6] >= 0
 R1574: - sbox[2,2,7] + rkey[1,1,6] + rkey[2,0,6] >= 0
 R1575: sbox[2,2,7] + rkey[1,1,6] - rkey[2,0,6] >= 0
 R1576: sbox[2,2,7] - rkey[1,1,6] + rkey[2,0,6] >= 0
 R1577: sbox[1,2,7] - rkey[0,2,6] = 0
 R1578: tbox[0,2,6] + tbox[2,0,6] - tbox[3,2,7] >= 0
 R1579: tbox[0,2,6] - tbox[2,0,6] + tbox[3,2,7] >= 0
 R1580: - tbox[0,2,6] + tbox[2,0,6] + tbox[3,2,7] >= 0
 R1581: - tbox[0,2,7] + tbox[3,2,7] + tbox[3,3,6] >= 0
 R1582: tbox[0,2,7] + tbox[3,2,7] - tbox[3,3,6] >= 0
 R1583: tbox[0,2,7] - tbox[3,2,7] + tbox[3,3,6] >= 0
 R1584: tbox[1,1,6] + tbox[2,0,6] - tbox[2,2,7] >= 0
 R1585: tbox[1,1,6] - tbox[2,0,6] + tbox[2,2,7] >= 0
 R1586: - tbox[1,1,6] + tbox[2,0,6] + tbox[2,2,7] >= 0
 R1587: - tbox[0,2,6] + tbox[1,2,7] = 0
 R1588: unkn[0,2,6] - unkn[3,2,7] + forg[3,2,7] <= 0
 R1589: unkn[2,0,6] - unkn[3,2,7] + forg[3,2,7] <= 0
 R1590: unkn[0,2,6] + unkn[2,0,6] - unkn[3,2,7] + forg[3,2,7] >= 0
 R1591: - unkn[0,2,7] + unkn[3,2,7] + forg[0,2,7] - forg[3,2,7] <= 0
 R1592: - unkn[0,2,7] + unkn[3,3,6] + forg[0,2,7] <= 0
 R1593: - unkn[0,2,7] + unkn[3,2,7] + unkn[3,3,6] + forg[0,2,7]
   - forg[3,2,7] >= 0
 R1594: unkn[1,1,6] - unkn[2,2,7] + forg[2,2,7] <= 0
 R1595: unkn[2,0,6] - unkn[2,2,7] + forg[2,2,7] <= 0
 R1596: unkn[1,1,6] + unkn[2,0,6] - unkn[2,2,7] + forg[2,2,7] >= 0
 R1597: unkn[0,2,6] - unkn[1,2,7] + forg[1,2,7] = 0
 R1598: - tbox[0,3,6] + forg[0,3,6] <= 0
 R1599: sbox[0,3,6] - rkey[0,3,6] + tkey[0,3,6] >= 0
 R1600: sbox[0,3,6] + rkey[0,3,6] - tkey[0,3,6] >= 0
 R1601: - sbox[0,3,6] + rkey[0,3,6] + tkey[0,3,6] >= 0
 R1602: - tbox[1,3,6] + forg[1,3,6] <= 0
 R1603: sbox[1,3,6] - rkey[1,3,6] + tkey[1,3,6] >= 0
 R1604: sbox[1,3,6] + rkey[1,3,6] - tkey[1,3,6] >= 0
 R1605: - sbox[1,3,6] + rkey[1,3,6] + tkey[1,3,6] >= 0
 R1606: - tbox[2,3,6] + forg[2,3,6] <= 0
 R1607: - sbox[2,3,6] + rkey[2,3,6] = 0
 R1608: - tbox[3,3,6] + forg[3,3,6] <= 0
 R1609: - sbox[3,3,6] + rkey[3,3,6] = 0
 R1610: - sbox[3,3,7] + rkey[0,3,6] + rkey[2,1,6] >= 0
 R1611: sbox[3,3,7] + rkey[0,3,6] - rkey[2,1,6] >= 0
 R1612: sbox[3,3,7] - rkey[0,3,6] + rkey[2,1,6] >= 0
 R1613: - sbox[0,3,7] + sbox[3,3,7] + rkey[3,0,6] >= 0
 R1614: sbox[0,3,7] + sbox[3,3,7] - rkey[3,0,6] >= 0
 R1615: sbox[0,3,7] - sbox[3,3,7] + rkey[3,0,6] >= 0
 R1616: - sbox[2,3,7] + rkey[1,2,6] + rkey[2,1,6] >= 0
 R1617: sbox[2,3,7] + rkey[1,2,6] - rkey[2,1,6] >= 0
 R1618: sbox[2,3,7] - rkey[1,2,6] + rkey[2,1,6] >= 0
 R1619: sbox[1,3,7] - rkey[0,3,6] = 0
 R1620: tbox[0,3,6] + tbox[2,1,6] - tbox[3,3,7] >= 0
 R1621: tbox[0,3,6] - tbox[2,1,6] + tbox[3,3,7] >= 0
 R1622: - tbox[0,3,6] + tbox[2,1,6] + tbox[3,3,7] >= 0
 R1623: - tbox[0,3,7] + tbox[3,0,6] + tbox[3,3,7] >= 0
 R1624: tbox[0,3,7] - tbox[3,0,6] + tbox[3,3,7] >= 0
 R1625: tbox[0,3,7] + tbox[3,0,6] - tbox[3,3,7] >= 0
 R1626: tbox[1,2,6] + tbox[2,1,6] - tbox[2,3,7] >= 0
 R1627: tbox[1,2,6] - tbox[2,1,6] + tbox[2,3,7] >= 0
 R1628: - tbox[1,2,6] + tbox[2,1,6] + tbox[2,3,7] >= 0
 R1629: - tbox[0,3,6] + tbox[1,3,7] = 0
 R1630: unkn[0,3,6] - unkn[3,3,7] + forg[3,3,7] <= 0
 R1631: unkn[2,1,6] - unkn[3,3,7] + forg[3,3,7] <= 0
 R1632: unkn[0,3,6] + unkn[2,1,6] - unkn[3,3,7] + forg[3,3,7] >= 0
 R1633: - unkn[0,3,7] + unkn[3,3,7] + forg[0,3,7] - forg[3,3,7] <= 0
 R1634: - unkn[0,3,7] + unkn[3,0,6] + forg[0,3,7] <= 0
 R1635: - unkn[0,3,7] + unkn[3,0,6] + unkn[3,3,7] + forg[0,3,7]
   - forg[3,3,7] >= 0
 R1636: unkn[1,2,6] - unkn[2,3,7] + forg[2,3,7] <= 0
 R1637: unkn[2,1,6] - unkn[2,3,7] + forg[2,3,7] <= 0
 R1638: unkn[1,2,6] + unkn[2,1,6] - unkn[2,3,7] + forg[2,3,7] >= 0
 R1639: unkn[0,3,6] - unkn[1,3,7] + forg[1,3,7] = 0
 R1640: sbox[0,0,0] + sbox[0,1,0] + sbox[0,2,0] + sbox[0,3,0] + sbox[1,0,0]
   + sbox[1,1,0] + sbox[1,2,0] + sbox[1,3,0] + sbox[2,0,0] + sbox[2,1,0]
   + sbox[2,2,0] + sbox[2,3,0] + sbox[3,0,0] + sbox[3,1,0] + sbox[3,2,0]
   + sbox[3,3,0] + tkey[0,0,0] + tkey[0,1,0] + tkey[0,2,0] + tkey[0,3,0]
   + tkey[1,0,0] + tkey[1,1,0] + tkey[1,2,0] + tkey[1,3,0] + tkey[2,0,0]
   + tkey[2,1,0] + tkey[2,2,0] + tkey[2,3,0] + tkey[3,0,0] + tkey[3,1,0]
   + tkey[3,2,0] + tkey[3,3,0] >= 1
 R1641: sbox[0,0,1] + sbox[0,1,1] + sbox[0,2,1] + sbox[0,3,1] + sbox[1,0,1]
   + sbox[1,1,1] + sbox[1,2,1] + sbox[1,3,1] + sbox[2,0,1] + sbox[2,1,1]
   + sbox[2,2,1] + sbox[2,3,1] + sbox[3,0,1] + sbox[3,1,1] + sbox[3,2,1]
   + sbox[3,3,1] + tkey[0,0,1] + tkey[0,1,1] + tkey[0,2,1] + tkey[0,3,1]
   + tkey[1,0,1] + tkey[1,1,1] + tkey[1,2,1] + tkey[1,3,1] + tkey[2,0,1]
   + tkey[2,1,1] + tkey[2,2,1] + tkey[2,3,1] + tkey[3,0,1] + tkey[3,1,1]
   + tkey[3,2,1] + tkey[3,3,1] >= 1
 R1642: sbox[0,0,2] + sbox[0,1,2] + sbox[0,2,2] + sbox[0,3,2] + sbox[1,0,2]
   + sbox[1,1,2] + sbox[1,2,2] + sbox[1,3,2] + sbox[2,0,2] + sbox[2,1,2]
   + sbox[2,2,2] + sbox[2,3,2] + sbox[3,0,2] + sbox[3,1,2] + sbox[3,2,2]
   + sbox[3,3,2] + tkey[0,0,2] + tkey[0,1,2] + tkey[0,2,2] + tkey[0,3,2]
   + tkey[1,0,2] + tkey[1,1,2] + tkey[1,2,2] + tkey[1,3,2] + tkey[2,0,2]
   + tkey[2,1,2] + tkey[2,2,2] + tkey[2,3,2] + tkey[3,0,2] + tkey[3,1,2]
   + tkey[3,2,2] + tkey[3,3,2] >= 1
 R1643: sbox[0,0,3] + sbox[0,1,3] + sbox[0,2,3] + sbox[0,3,3] + sbox[1,0,3]
   + sbox[1,1,3] + sbox[1,2,3] + sbox[1,3,3] + sbox[2,0,3] + sbox[2,1,3]
   + sbox[2,2,3] + sbox[2,3,3] + sbox[3,0,3] + sbox[3,1,3] + sbox[3,2,3]
   + sbox[3,3,3] + tkey[0,0,3] + tkey[0,1,3] + tkey[0,2,3] + tkey[0,3,3]
   + tkey[1,0,3] + tkey[1,1,3] + tkey[1,2,3] + tkey[1,3,3] + tkey[2,0,3]
   + tkey[2,1,3] + tkey[2,2,3] + tkey[2,3,3] + tkey[3,0,3] + tkey[3,1,3]
   + tkey[3,2,3] + tkey[3,3,3] >= 1
 R1644: sbox[0,0,4] + sbox[0,1,4] + sbox[0,2,4] + sbox[0,3,4] + sbox[1,0,4]
   + sbox[1,1,4] + sbox[1,2,4] + sbox[1,3,4] + sbox[2,0,4] + sbox[2,1,4]
   + sbox[2,2,4] + sbox[2,3,4] + sbox[3,0,4] + sbox[3,1,4] + sbox[3,2,4]
   + sbox[3,3,4] + tkey[0,0,4] + tkey[0,1,4] + tkey[0,2,4] + tkey[0,3,4]
   + tkey[1,0,4] + tkey[1,1,4] + tkey[1,2,4] + tkey[1,3,4] + tkey[2,0,4]
   + tkey[2,1,4] + tkey[2,2,4] + tkey[2,3,4] + tkey[3,0,4] + tkey[3,1,4]
   + tkey[3,2,4] + tkey[3,3,4] >= 1
 R1645: sbox[0,0,5] + sbox[0,1,5] + sbox[0,2,5] + sbox[0,3,5] + sbox[1,0,5]
   + sbox[1,1,5] + sbox[1,2,5] + sbox[1,3,5] + sbox[2,0,5] + sbox[2,1,5]
   + sbox[2,2,5] + sbox[2,3,5] + sbox[3,0,5] + sbox[3,1,5] + sbox[3,2,5]
   + sbox[3,3,5] + tkey[0,0,5] + tkey[0,1,5] + tkey[0,2,5] + tkey[0,3,5]
   + tkey[1,0,5] + tkey[1,1,5] + tkey[1,2,5] + tkey[1,3,5] + tkey[2,0,5]
   + tkey[2,1,5] + tkey[2,2,5] + tkey[2,3,5] + tkey[3,0,5] + tkey[3,1,5]
   + tkey[3,2,5] + tkey[3,3,5] >= 1
 R1646: sbox[0,0,6] + sbox[0,1,6] + sbox[0,2,6] + sbox[0,3,6] + sbox[1,0,6]
   + sbox[1,1,6] + sbox[1,2,6] + sbox[1,3,6] + sbox[2,0,6] + sbox[2,1,6]
   + sbox[2,2,6] + sbox[2,3,6] + sbox[3,0,6] + sbox[3,1,6] + sbox[3,2,6]
   + sbox[3,3,6] + tkey[0,0,6] + tkey[0,1,6] + tkey[0,2,6] + tkey[0,3,6]
   + tkey[1,0,6] + tkey[1,1,6] + tkey[1,2,6] + tkey[1,3,6] + tkey[2,0,6]
   + tkey[2,1,6] + tkey[2,2,6] + tkey[2,3,6] + tkey[3,0,6] + tkey[3,1,6]
   + tkey[3,2,6] + tkey[3,3,6] >= 1
 R1647: sbox[0,0,0] + sbox[0,1,0] + sbox[0,2,0] + sbox[0,3,0] + sbox[1,0,0]
   + sbox[1,1,0] + sbox[1,2,0] + sbox[1,3,0] + sbox[2,0,0] + sbox[2,1,0]
   + sbox[2,2,0] + sbox[2,3,0] + sbox[3,0,0] + sbox[3,1,0] + sbox[3,2,0]
   + sbox[3,3,0] = 0
 R1648: sbox[0,0,7] + sbox[0,1,7] + sbox[0,2,7] + sbox[0,3,7] + sbox[1,0,7]
   + sbox[1,1,7] + sbox[1,2,7] + sbox[1,3,7] + sbox[2,0,7] + sbox[2,1,7]
   + sbox[2,2,7] + sbox[2,3,7] + sbox[3,0,7] + sbox[3,1,7] + sbox[3,2,7]
   + sbox[3,3,7] = 0
 R1649: tbox[0,0,0] = 1
 R1650: unkn[0,0,0] - forg[0,0,0] = 0
 R1651: tbox[0,1,0] = 0
 R1652: unkn[0,1,0] - forg[0,1,0] = 0
 R1653: tbox[0,2,0] = 0
 R1654: unkn[0,2,0] - forg[0,2,0] = 0
 R1655: tbox[0,3,0] = 0
 R1656: unkn[0,3,0] - forg[0,3,0] = 0
 R1657: tbox[1,0,0] = 0
 R1658: unkn[1,0,0] - forg[1,0,0] = 0
 R1659: tbox[1,1,0] = 0
 R1660: unkn[1,1,0] - forg[1,1,0] = 0
 R1661: tbox[1,2,0] = 0
 R1662: unkn[1,2,0] - forg[1,2,0] = 0
 R1663: tbox[1,3,0] = 0
 R1664: unkn[1,3,0] - forg[1,3,0] = 0
 R1665: tbox[2,0,0] = 0
 R1666: unkn[2,0,0] - forg[2,0,0] = 0
 R1667: tbox[2,1,0] = 0
 R1668: unkn[2,1,0] - forg[2,1,0] = 0
 R1669: tbox[2,2,0] = 0
 R1670: unkn[2,2,0] - forg[2,2,0] = 0
 R1671: tbox[2,3,0] = 0
 R1672: unkn[2,3,0] - forg[2,3,0] = 0
 R1673: tbox[3,0,0] = 0
 R1674: unkn[3,0,0] - forg[3,0,0] = 0
 R1675: tbox[3,1,0] = 0
 R1676: unkn[3,1,0] - forg[3,1,0] = 0
 R1677: tbox[3,2,0] = 0
 R1678: unkn[3,2,0] - forg[3,2,0] = 0
 R1679: tbox[3,3,0] = 0
 R1680: unkn[3,3,0] - forg[3,3,0] = 0
Bounds
 cost[0,0,0] <= 2
 cost[0,0,1] <= 2
 cost[0,0,2] <= 2
 cost[0,0,3] <= 2
 cost[0,0,4] <= 2
 cost[0,0,5] <= 2
 cost[0,0,6] <= 2
 cost[0,0,7] <= 2
 cost[0,1,0] <= 2
 cost[0,1,1] <= 2
 cost[0,1,2] <= 2
 cost[0,1,3] <= 2
 cost[0,1,4] <= 2
 cost[0,1,5] <= 2
 cost[0,1,6] <= 2
 cost[0,1,7] <= 2
 cost[0,2,0] <= 2
 cost[0,2,1] <= 2
 cost[0,2,2] <= 2
 cost[0,2,3] <= 2
 cost[0,2,4] <= 2
 cost[0,2,5] <= 2
 cost[0,2,6] <= 2
 cost[0,2,7] <= 2
 cost[0,3,0] <= 2
 cost[0,3,1] <= 2
 cost[0,3,2] <= 2
 cost[0,3,3] <= 2
 cost[0,3,4] <= 2
 cost[0,3,5] <= 2
 cost[0,3,6] <= 2
 cost[0,3,7] <= 2
 cost[1,0,0] <= 2
 cost[1,0,1] <= 2
 cost[1,0,2] <= 2
 cost[1,0,3] <= 2
 cost[1,0,4] <= 2
 cost[1,0,5] <= 2
 cost[1,0,6] <= 2
 cost[1,0,7] <= 2
 cost[1,1,0] <= 2
 cost[1,1,1] <= 2
 cost[1,1,2] <= 2
 cost[1,1,3] <= 2
 cost[1,1,4] <= 2
 cost[1,1,5] <= 2
 cost[1,1,6] <= 2
 cost[1,1,7] <= 2
 cost[1,2,0] <= 2
 cost[1,2,1] <= 2
 cost[1,2,2] <= 2
 cost[1,2,3] <= 2
 cost[1,2,4] <= 2
 cost[1,2,5] <= 2
 cost[1,2,6] <= 2
 cost[1,2,7] <= 2
 cost[1,3,0] <= 2
 cost[1,3,1] <= 2
 cost[1,3,2] <= 2
 cost[1,3,3] <= 2
 cost[1,3,4] <= 2
 cost[1,3,5] <= 2
 cost[1,3,6] <= 2
 cost[1,3,7] <= 2
 cost[2,0,0] <= 2
 cost[2,0,1] <= 2
 cost[2,0,2] <= 2
 cost[2,0,3] <= 2
 cost[2,0,4] <= 2
 cost[2,0,5] <= 2
 cost[2,0,6] <= 2
 cost[2,0,7] <= 2
 cost[2,1,0] <= 2
 cost[2,1,1] <= 2
 cost[2,1,2] <= 2
 cost[2,1,3] <= 2
 cost[2,1,4] <= 2
 cost[2,1,5] <= 2
 cost[2,1,6] <= 2
 cost[2,1,7] <= 2
 cost[2,2,0] <= 2
 cost[2,2,1] <= 2
 cost[2,2,2] <= 2
 cost[2,2,3] <= 2
 cost[2,2,4] <= 2
 cost[2,2,5] <= 2
 cost[2,2,6] <= 2
 cost[2,2,7] <= 2
 cost[2,3,0] <= 2
 cost[2,3,1] <= 2
 cost[2,3,2] <= 2
 cost[2,3,3] <= 2
 cost[2,3,4] <= 2
 cost[2,3,5] <= 2
 cost[2,3,6] <= 2
 cost[2,3,7] <= 2
 cost[3,0,0] <= 2
 cost[3,0,1] <= 2
 cost[3,0,2] <= 2
 cost[3,0,3] <= 2
 cost[3,0,4] <= 2
 cost[3,0,5] <= 2
 cost[3,0,6] <= 2
 cost[3,0,7] <= 2
 cost[3,1,0] <= 2
 cost[3,1,1] <= 2
 cost[3,1,2] <= 2
 cost[3,1,3] <= 2
 cost[3,1,4] <= 2
 cost[3,1,5] <= 2
 cost[3,1,6] <= 2
 cost[3,1,7] <= 2
 cost[3,2,0] <= 2
 cost[3,2,1] <= 2
 cost[3,2,2] <= 2
 cost[3,2,3] <= 2
 cost[3,2,4] <= 2
 cost[3,2,5] <= 2
 cost[3,2,6] <= 2
 cost[3,2,7] <= 2
 cost[3,3,0] <= 2
 cost[3,3,1] <= 2
 cost[3,3,2] <= 2
 cost[3,3,3] <= 2
 cost[3,3,4] <= 2
 cost[3,3,5] <= 2
 cost[3,3,6] <= 2
 cost[3,3,7] <= 2
Binaries
 sbox[0,0,0] sbox[0,0,1] sbox[0,0,2] sbox[0,0,3] sbox[0,0,4] sbox[0,0,5]
 sbox[0,0,6] sbox[0,0,7] sbox[0,1,0] sbox[0,1,1] sbox[0,1,2] sbox[0,1,3]
 sbox[0,1,4] sbox[0,1,5] sbox[0,1,6] sbox[0,1,7] sbox[0,2,0] sbox[0,2,1]
 sbox[0,2,2] sbox[0,2,3] sbox[0,2,4] sbox[0,2,5] sbox[0,2,6] sbox[0,2,7]
 sbox[0,3,0] sbox[0,3,1] sbox[0,3,2] sbox[0,3,3] sbox[0,3,4] sbox[0,3,5]
 sbox[0,3,6] sbox[0,3,7] sbox[1,0,0] sbox[1,0,1] sbox[1,0,2] sbox[1,0,3]
 sbox[1,0,4] sbox[1,0,5] sbox[1,0,6] sbox[1,0,7] sbox[1,1,0] sbox[1,1,1]
 sbox[1,1,2] sbox[1,1,3] sbox[1,1,4] sbox[1,1,5] sbox[1,1,6] sbox[1,1,7]
 sbox[1,2,0] sbox[1,2,1] sbox[1,2,2] sbox[1,2,3] sbox[1,2,4] sbox[1,2,5]
 sbox[1,2,6] sbox[1,2,7] sbox[1,3,0] sbox[1,3,1] sbox[1,3,2] sbox[1,3,3]
 sbox[1,3,4] sbox[1,3,5] sbox[1,3,6] sbox[1,3,7] sbox[2,0,0] sbox[2,0,1]
 sbox[2,0,2] sbox[2,0,3] sbox[2,0,4] sbox[2,0,5] sbox[2,0,6] sbox[2,0,7]
 sbox[2,1,0] sbox[2,1,1] sbox[2,1,2] sbox[2,1,3] sbox[2,1,4] sbox[2,1,5]
 sbox[2,1,6] sbox[2,1,7] sbox[2,2,0] sbox[2,2,1] sbox[2,2,2] sbox[2,2,3]
 sbox[2,2,4] sbox[2,2,5] sbox[2,2,6] sbox[2,2,7] sbox[2,3,0] sbox[2,3,1]
 sbox[2,3,2] sbox[2,3,3] sbox[2,3,4] sbox[2,3,5] sbox[2,3,6] sbox[2,3,7]
 sbox[3,0,0] sbox[3,0,1] sbox[3,0,2] sbox[3,0,3] sbox[3,0,4] sbox[3,0,5]
 sbox[3,0,6] sbox[3,0,7] sbox[3,1,0] sbox[3,1,1] sbox[3,1,2] sbox[3,1,3]
 sbox[3,1,4] sbox[3,1,5] sbox[3,1,6] sbox[3,1,7] sbox[3,2,0] sbox[3,2,1]
 sbox[3,2,2] sbox[3,2,3] sbox[3,2,4] sbox[3,2,5] sbox[3,2,6] sbox[3,2,7]
 sbox[3,3,0] sbox[3,3,1] sbox[3,3,2] sbox[3,3,3] sbox[3,3,4] sbox[3,3,5]
 sbox[3,3,6] sbox[3,3,7] rkey[0,0,0] rkey[0,0,1] rkey[0,0,2] rkey[0,0,3]
 rkey[0,0,4] rkey[0,0,5] rkey[0,0,6] rkey[0,0,7] rkey[0,1,0] rkey[0,1,1]
 rkey[0,1,2] rkey[0,1,3] rkey[0,1,4] rkey[0,1,5] rkey[0,1,6] rkey[0,1,7]
 rkey[0,2,0] rkey[0,2,1] rkey[0,2,2] rkey[0,2,3] rkey[0,2,4] rkey[0,2,5]
 rkey[0,2,6] rkey[0,2,7] rkey[0,3,0] rkey[0,3,1] rkey[0,3,2] rkey[0,3,3]
 rkey[0,3,4] rkey[0,3,5] rkey[0,3,6] rkey[0,3,7] rkey[1,0,0] rkey[1,0,1]
 rkey[1,0,2] rkey[1,0,3] rkey[1,0,4] rkey[1,0,5] rkey[1,0,6] rkey[1,0,7]
 rkey[1,1,0] rkey[1,1,1] rkey[1,1,2] rkey[1,1,3] rkey[1,1,4] rkey[1,1,5]
 rkey[1,1,6] rkey[1,1,7] rkey[1,2,0] rkey[1,2,1] rkey[1,2,2] rkey[1,2,3]
 rkey[1,2,4] rkey[1,2,5] rkey[1,2,6] rkey[1,2,7] rkey[1,3,0] rkey[1,3,1]
 rkey[1,3,2] rkey[1,3,3] rkey[1,3,4] rkey[1,3,5] rkey[1,3,6] rkey[1,3,7]
 rkey[2,0,0] rkey[2,0,1] rkey[2,0,2] rkey[2,0,3] rkey[2,0,4] rkey[2,0,5]
 rkey[2,0,6] rkey[2,0,7] rkey[2,1,0] rkey[2,1,1] rkey[2,1,2] rkey[2,1,3]
 rkey[2,1,4] rkey[2,1,5] rkey[2,1,6] rkey[2,1,7] rkey[2,2,0] rkey[2,2,1]
 rkey[2,2,2] rkey[2,2,3] rkey[2,2,4] rkey[2,2,5] rkey[2,2,6] rkey[2,2,7]
 rkey[2,3,0] rkey[2,3,1] rkey[2,3,2] rkey[2,3,3] rkey[2,3,4] rkey[2,3,5]
 rkey[2,3,6] rkey[2,3,7] rkey[3,0,0] rkey[3,0,1] rkey[3,0,2] rkey[3,0,3]
 rkey[3,0,4] rkey[3,0,5] rkey[3,0,6] rkey[3,0,7] rkey[3,1,0] rkey[3,1,1]
 rkey[3,1,2] rkey[3,1,3] rkey[3,1,4] rkey[3,1,5] rkey[3,1,6] rkey[3,1,7]
 rkey[3,2,0] rkey[3,2,1] rkey[3,2,2] rkey[3,2,3] rkey[3,2,4] rkey[3,2,5]
 rkey[3,2,6] rkey[3,2,7] rkey[3,3,0] rkey[3,3,1] rkey[3,3,2] rkey[3,3,3]
 rkey[3,3,4] rkey[3,3,5] rkey[3,3,6] rkey[3,3,7] tkey[0,0,0] tkey[0,0,1]
 tkey[0,0,2] tkey[0,0,3] tkey[0,0,4] tkey[0,0,5] tkey[0,0,6] tkey[0,0,7]
 tkey[0,1,0] tkey[0,1,1] tkey[0,1,2] tkey[0,1,3] tkey[0,1,4] tkey[0,1,5]
 tkey[0,1,6] tkey[0,1,7] tkey[0,2,0] tkey[0,2,1] tkey[0,2,2] tkey[0,2,3]
 tkey[0,2,4] tkey[0,2,5] tkey[0,2,6] tkey[0,2,7] tkey[0,3,0] tkey[0,3,1]
 tkey[0,3,2] tkey[0,3,3] tkey[0,3,4] tkey[0,3,5] tkey[0,3,6] tkey[0,3,7]
 tkey[1,0,0] tkey[1,0,1] tkey[1,0,2] tkey[1,0,3] tkey[1,0,4] tkey[1,0,5]
 tkey[1,0,6] tkey[1,0,7] tkey[1,1,0] tkey[1,1,1] tkey[1,1,2] tkey[1,1,3]
 tkey[1,1,4] tkey[1,1,5] tkey[1,1,6] tkey[1,1,7] tkey[1,2,0] tkey[1,2,1]
 tkey[1,2,2] tkey[1,2,3] tkey[1,2,4] tkey[1,2,5] tkey[1,2,6] tkey[1,2,7]
 tkey[1,3,0] tkey[1,3,1] tkey[1,3,2] tkey[1,3,3] tkey[1,3,4] tkey[1,3,5]
 tkey[1,3,6] tkey[1,3,7] tkey[2,0,0] tkey[2,0,1] tkey[2,0,2] tkey[2,0,3]
 tkey[2,0,4] tkey[2,0,5] tkey[2,0,6] tkey[2,0,7] tkey[2,1,0] tkey[2,1,1]
 tkey[2,1,2] tkey[2,1,3] tkey[2,1,4] tkey[2,1,5] tkey[2,1,6] tkey[2,1,7]
 tkey[2,2,0] tkey[2,2,1] tkey[2,2,2] tkey[2,2,3] tkey[2,2,4] tkey[2,2,5]
 tkey[2,2,6] tkey[2,2,7] tkey[2,3,0] tkey[2,3,1] tkey[2,3,2] tkey[2,3,3]
 tkey[2,3,4] tkey[2,3,5] tkey[2,3,6] tkey[2,3,7] tkey[3,0,0] tkey[3,0,1]
 tkey[3,0,2] tkey[3,0,3] tkey[3,0,4] tkey[3,0,5] tkey[3,0,6] tkey[3,0,7]
 tkey[3,1,0] tkey[3,1,1] tkey[3,1,2] tkey[3,1,3] tkey[3,1,4] tkey[3,1,5]
 tkey[3,1,6] tkey[3,1,7] tkey[3,2,0] tkey[3,2,1] tkey[3,2,2] tkey[3,2,3]
 tkey[3,2,4] tkey[3,2,5] tkey[3,2,6] tkey[3,2,7] tkey[3,3,0] tkey[3,3,1]
 tkey[3,3,2] tkey[3,3,3] tkey[3,3,4] tkey[3,3,5] tkey[3,3,6] tkey[3,3,7]
 tbox[0,0,0] tbox[0,0,1] tbox[0,0,2] tbox[0,0,3] tbox[0,0,4] tbox[0,0,5]
 tbox[0,0,6] tbox[0,0,7] tbox[0,1,0] tbox[0,1,1] tbox[0,1,2] tbox[0,1,3]
 tbox[0,1,4] tbox[0,1,5] tbox[0,1,6] tbox[0,1,7] tbox[0,2,0] tbox[0,2,1]
 tbox[0,2,2] tbox[0,2,3] tbox[0,2,4] tbox[0,2,5] tbox[0,2,6] tbox[0,2,7]
 tbox[0,3,0] tbox[0,3,1] tbox[0,3,2] tbox[0,3,3] tbox[0,3,4] tbox[0,3,5]
 tbox[0,3,6] tbox[0,3,7] tbox[1,0,0] tbox[1,0,1] tbox[1,0,2] tbox[1,0,3]
 tbox[1,0,4] tbox[1,0,5] tbox[1,0,6] tbox[1,0,7] tbox[1,1,0] tbox[1,1,1]
 tbox[1,1,2] tbox[1,1,3] tbox[1,1,4] tbox[1,1,5] tbox[1,1,6] tbox[1,1,7]
 tbox[1,2,0] tbox[1,2,1] tbox[1,2,2] tbox[1,2,3] tbox[1,2,4] tbox[1,2,5]
 tbox[1,2,6] tbox[1,2,7] tbox[1,3,0] tbox[1,3,1] tbox[1,3,2] tbox[1,3,3]
 tbox[1,3,4] tbox[1,3,5] tbox[1,3,6] tbox[1,3,7] tbox[2,0,0] tbox[2,0,1]
 tbox[2,0,2] tbox[2,0,3] tbox[2,0,4] tbox[2,0,5] tbox[2,0,6] tbox[2,0,7]
 tbox[2,1,0] tbox[2,1,1] tbox[2,1,2] tbox[2,1,3] tbox[2,1,4] tbox[2,1,5]
 tbox[2,1,6] tbox[2,1,7] tbox[2,2,0] tbox[2,2,1] tbox[2,2,2] tbox[2,2,3]
 tbox[2,2,4] tbox[2,2,5] tbox[2,2,6] tbox[2,2,7] tbox[2,3,0] tbox[2,3,1]
 tbox[2,3,2] tbox[2,3,3] tbox[2,3,4] tbox[2,3,5] tbox[2,3,6] tbox[2,3,7]
 tbox[3,0,0] tbox[3,0,1] tbox[3,0,2] tbox[3,0,3] tbox[3,0,4] tbox[3,0,5]
 tbox[3,0,6] tbox[3,0,7] tbox[3,1,0] tbox[3,1,1] tbox[3,1,2] tbox[3,1,3]
 tbox[3,1,4] tbox[3,1,5] tbox[3,1,6] tbox[3,1,7] tbox[3,2,0] tbox[3,2,1]
 tbox[3,2,2] tbox[3,2,3] tbox[3,2,4] tbox[3,2,5] tbox[3,2,6] tbox[3,2,7]
 tbox[3,3,0] tbox[3,3,1] tbox[3,3,2] tbox[3,3,3] tbox[3,3,4] tbox[3,3,5]
 tbox[3,3,6] tbox[3,3,7] unkn[0,0,0] unkn[0,0,1] unkn[0,0,2] unkn[0,0,3]
 unkn[0,0,4] unkn[0,0,5] unkn[0,0,6] unkn[0,0,7] unkn[0,1,0] unkn[0,1,1]
 unkn[0,1,2] unkn[0,1,3] unkn[0,1,4] unkn[0,1,5] unkn[0,1,6] unkn[0,1,7]
 unkn[0,2,0] unkn[0,2,1] unkn[0,2,2] unkn[0,2,3] unkn[0,2,4] unkn[0,2,5]
 unkn[0,2,6] unkn[0,2,7] unkn[0,3,0] unkn[0,3,1] unkn[0,3,2] unkn[0,3,3]
 unkn[0,3,4] unkn[0,3,5] unkn[0,3,6] unkn[0,3,7] unkn[1,0,0] unkn[1,0,1]
 unkn[1,0,2] unkn[1,0,3] unkn[1,0,4] unkn[1,0,5] unkn[1,0,6] unkn[1,0,7]
 unkn[1,1,0] unkn[1,1,1] unkn[1,1,2] unkn[1,1,3] unkn[1,1,4] unkn[1,1,5]
 unkn[1,1,6] unkn[1,1,7] unkn[1,2,0] unkn[1,2,1] unkn[1,2,2] unkn[1,2,3]
 unkn[1,2,4] unkn[1,2,5] unkn[1,2,6] unkn[1,2,7] unkn[1,3,0] unkn[1,3,1]
 unkn[1,3,2] unkn[1,3,3] unkn[1,3,4] unkn[1,3,5] unkn[1,3,6] unkn[1,3,7]
 unkn[2,0,0] unkn[2,0,1] unkn[2,0,2] unkn[2,0,3] unkn[2,0,4] unkn[2,0,5]
 unkn[2,0,6] unkn[2,0,7] unkn[2,1,0] unkn[2,1,1] unkn[2,1,2] unkn[2,1,3]
 unkn[2,1,4] unkn[2,1,5] unkn[2,1,6] unkn[2,1,7] unkn[2,2,0] unkn[2,2,1]
 unkn[2,2,2] unkn[2,2,3] unkn[2,2,4] unkn[2,2,5] unkn[2,2,6] unkn[2,2,7]
 unkn[2,3,0] unkn[2,3,1] unkn[2,3,2] unkn[2,3,3] unkn[2,3,4] unkn[2,3,5]
 unkn[2,3,6] unkn[2,3,7] unkn[3,0,0] unkn[3,0,1] unkn[3,0,2] unkn[3,0,3]
 unkn[3,0,4] unkn[3,0,5] unkn[3,0,6] unkn[3,0,7] unkn[3,1,0] unkn[3,1,1]
 unkn[3,1,2] unkn[3,1,3] unkn[3,1,4] unkn[3,1,5] unkn[3,1,6] unkn[3,1,7]
 unkn[3,2,0] unkn[3,2,1] unkn[3,2,2] unkn[3,2,3] unkn[3,2,4] unkn[3,2,5]
 unkn[3,2,6] unkn[3,2,7] unkn[3,3,0] unkn[3,3,1] unkn[3,3,2] unkn[3,3,3]
 unkn[3,3,4] unkn[3,3,5] unkn[3,3,6] unkn[3,3,7] forg[0,0,0] forg[0,0,1]
 forg[0,0,2] forg[0,0,3] forg[0,0,4] forg[0,0,5] forg[0,0,6] forg[0,0,7]
 forg[0,1,0] forg[0,1,1] forg[0,1,2] forg[0,1,3] forg[0,1,4] forg[0,1,5]
 forg[0,1,6] forg[0,1,7] forg[0,2,0] forg[0,2,1] forg[0,2,2] forg[0,2,3]
 forg[0,2,4] forg[0,2,5] forg[0,2,6] forg[0,2,7] forg[0,3,0] forg[0,3,1]
 forg[0,3,2] forg[0,3,3] forg[0,3,4] forg[0,3,5] forg[0,3,6] forg[0,3,7]
 forg[1,0,0] forg[1,0,1] forg[1,0,2] forg[1,0,3] forg[1,0,4] forg[1,0,5]
 forg[1,0,6] forg[1,0,7] forg[1,1,0] forg[1,1,1] forg[1,1,2] forg[1,1,3]
 forg[1,1,4] forg[1,1,5] forg[1,1,6] forg[1,1,7] forg[1,2,0] forg[1,2,1]
 forg[1,2,2] forg[1,2,3] forg[1,2,4] forg[1,2,5] forg[1,2,6] forg[1,2,7]
 forg[1,3,0] forg[1,3,1] forg[1,3,2] forg[1,3,3] forg[1,3,4] forg[1,3,5]
 forg[1,3,6] forg[1,3,7] forg[2,0,0] forg[2,0,1] forg[2,0,2] forg[2,0,3]
 forg[2,0,4] forg[2,0,5] forg[2,0,6] forg[2,0,7] forg[2,1,0] forg[2,1,1]
 forg[2,1,2] forg[2,1,3] forg[2,1,4] forg[2,1,5] forg[2,1,6] forg[2,1,7]
 forg[2,2,0] forg[2,2,1] forg[2,2,2] forg[2,2,3] forg[2,2,4] forg[2,2,5]
 forg[2,2,6] forg[2,2,7] forg[2,3,0] forg[2,3,1] forg[2,3,2] forg[2,3,3]
 forg[2,3,4] forg[2,3,5] forg[2,3,6] forg[2,3,7] forg[3,0,0] forg[3,0,1]
 forg[3,0,2] forg[3,0,3] forg[3,0,4] forg[3,0,5] forg[3,0,6] forg[3,0,7]
 forg[3,1,0] forg[3,1,1] forg[3,1,2] forg[3,1,3] forg[3,1,4] forg[3,1,5]
 forg[3,1,6] forg[3,1,7] forg[3,2,0] forg[3,2,1] forg[3,2,2] forg[3,2,3]
 forg[3,2,4] forg[3,2,5] forg[3,2,6] forg[3,2,7] forg[3,3,0] forg[3,3,1]
 forg[3,3,2] forg[3,3,3] forg[3,3,4] forg[3,3,5] forg[3,3,6] forg[3,3,7]
 lane[0,0] lane[0,1] lane[0,2] lane[0,3] lane[1,0] lane[1,1] lane[1,2]
 lane[1,3] lane[2,0] lane[2,1] lane[2,2] lane[2,3] lane[3,0] lane[3,1]
 lane[3,2] lane[3,3]
Generals
 cost[0,0,0] cost[0,0,1] cost[0,0,2] cost[0,0,3] cost[0,0,4] cost[0,0,5]
 cost[0,0,6] cost[0,0,7] cost[0,1,0] cost[0,1,1] cost[0,1,2] cost[0,1,3]
 cost[0,1,4] cost[0,1,5] cost[0,1,6] cost[0,1,7] cost[0,2,0] cost[0,2,1]
 cost[0,2,2] cost[0,2,3] cost[0,2,4] cost[0,2,5] cost[0,2,6] cost[0,2,7]
 cost[0,3,0] cost[0,3,1] cost[0,3,2] cost[0,3,3] cost[0,3,4] cost[0,3,5]
 cost[0,3,6] cost[0,3,7] cost[1,0,0] cost[1,0,1] cost[1,0,2] cost[1,0,3]
 cost[1,0,4] cost[1,0,5] cost[1,0,6] cost[1,0,7] cost[1,1,0] cost[1,1,1]
 cost[1,1,2] cost[1,1,3] cost[1,1,4] cost[1,1,5] cost[1,1,6] cost[1,1,7]
 cost[1,2,0] cost[1,2,1] cost[1,2,2] cost[1,2,3] cost[1,2,4] cost[1,2,5]
 cost[1,2,6] cost[1,2,7] cost[1,3,0] cost[1,3,1] cost[1,3,2] cost[1,3,3]
 cost[1,3,4] cost[1,3,5] cost[1,3,6] cost[1,3,7] cost[2,0,0] cost[2,0,1]
 cost[2,0,2] cost[2,0,3] cost[2,0,4] cost[2,0,5] cost[2,0,6] cost[2,0,7]
 cost[2,1,0] cost[2,1,1] cost[2,1,2] cost[2,1,3] cost[2,1,4] cost[2,1,5]
 cost[2,1,6] cost[2,1,7] cost[2,2,0] cost[2,2,1] cost[2,2,2] cost[2,2,3]
 cost[2,2,4] cost[2,2,5] cost[2,2,6] cost[2,2,7] cost[2,3,0] cost[2,3,1]
 cost[2,3,2] cost[2,3,3] cost[2,3,4] cost[2,3,5] cost[2,3,6] cost[2,3,7]
 cost[3,0,0] cost[3,0,1] cost[3,0,2] cost[3,0,3] cost[3,0,4] cost[3,0,5]
 cost[3,0,6] cost[3,0,7] cost[3,1,0] cost[3,1,1] cost[3,1,2] cost[3,1,3]
 cost[3,1,4] cost[3,1,5] cost[3,1,6] cost[3,1,7] cost[3,2,0] cost[3,2,1]
 cost[3,2,2] cost[3,2,3] cost[3,2,4] cost[3,2,5] cost[3,2,6] cost[3,2,7]
 cost[3,3,0] cost[3,3,1] cost[3,3,2] cost[3,3,3] cost[3,3,4] cost[3,3,5]
 cost[3,3,6] cost[3,3,7]
End
