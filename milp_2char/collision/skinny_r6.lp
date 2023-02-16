\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 rkey[0,0,6] + 0 rkey[0,1,6] + 0 rkey[0,2,6] + 0 rkey[0,3,6]
   + 0 rkey[1,0,6] + 0 rkey[1,1,6] + 0 rkey[1,2,6] + 0 rkey[1,3,6]
   + 0 rkey[2,0,6] + 0 rkey[2,1,6] + 0 rkey[2,2,6] + 0 rkey[2,3,6]
   + 0 rkey[3,0,6] + 0 rkey[3,1,6] + 0 rkey[3,2,6] + 0 rkey[3,3,6]
   + 0 tkey[0,0,6] + 0 tkey[0,1,6] + 0 tkey[0,2,6] + 0 tkey[0,3,6]
   + 0 tkey[1,0,6] + 0 tkey[1,1,6] + 0 tkey[1,2,6] + 0 tkey[1,3,6]
   + 0 tkey[2,0,6] + 0 tkey[2,1,6] + 0 tkey[2,2,6] + 0 tkey[2,3,6]
   + 0 tkey[3,0,6] + 0 tkey[3,1,6] + 0 tkey[3,2,6] + 0 tkey[3,3,6]
   + cost[0,0,0] + cost[0,0,1] + cost[0,0,2] + cost[0,0,3] + cost[0,0,4]
   + cost[0,0,5] + 0 cost[0,0,6] + cost[0,1,0] + cost[0,1,1] + cost[0,1,2]
   + cost[0,1,3] + cost[0,1,4] + cost[0,1,5] + 0 cost[0,1,6] + cost[0,2,0]
   + cost[0,2,1] + cost[0,2,2] + cost[0,2,3] + cost[0,2,4] + cost[0,2,5]
   + 0 cost[0,2,6] + cost[0,3,0] + cost[0,3,1] + cost[0,3,2] + cost[0,3,3]
   + cost[0,3,4] + cost[0,3,5] + 0 cost[0,3,6] + cost[1,0,0] + cost[1,0,1]
   + cost[1,0,2] + cost[1,0,3] + cost[1,0,4] + cost[1,0,5] + 0 cost[1,0,6]
   + cost[1,1,0] + cost[1,1,1] + cost[1,1,2] + cost[1,1,3] + cost[1,1,4]
   + cost[1,1,5] + 0 cost[1,1,6] + cost[1,2,0] + cost[1,2,1] + cost[1,2,2]
   + cost[1,2,3] + cost[1,2,4] + cost[1,2,5] + 0 cost[1,2,6] + cost[1,3,0]
   + cost[1,3,1] + cost[1,3,2] + cost[1,3,3] + cost[1,3,4] + cost[1,3,5]
   + 0 cost[1,3,6] + cost[2,0,0] + cost[2,0,1] + cost[2,0,2] + cost[2,0,3]
   + cost[2,0,4] + cost[2,0,5] + 0 cost[2,0,6] + cost[2,1,0] + cost[2,1,1]
   + cost[2,1,2] + cost[2,1,3] + cost[2,1,4] + cost[2,1,5] + 0 cost[2,1,6]
   + cost[2,2,0] + cost[2,2,1] + cost[2,2,2] + cost[2,2,3] + cost[2,2,4]
   + cost[2,2,5] + 0 cost[2,2,6] + cost[2,3,0] + cost[2,3,1] + cost[2,3,2]
   + cost[2,3,3] + cost[2,3,4] + cost[2,3,5] + 0 cost[2,3,6] + cost[3,0,0]
   + cost[3,0,1] + cost[3,0,2] + cost[3,0,3] + cost[3,0,4] + cost[3,0,5]
   + 0 cost[3,0,6] + cost[3,1,0] + cost[3,1,1] + cost[3,1,2] + cost[3,1,3]
   + cost[3,1,4] + cost[3,1,5] + 0 cost[3,1,6] + cost[3,2,0] + cost[3,2,1]
   + cost[3,2,2] + cost[3,2,3] + cost[3,2,4] + cost[3,2,5] + 0 cost[3,2,6]
   + cost[3,3,0] + cost[3,3,1] + cost[3,3,2] + cost[3,3,3] + cost[3,3,4]
   + cost[3,3,5] + 0 cost[3,3,6]
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
 R288: tkey[0,0,0] - lane[0,0] <= 0
 R289: tkey[2,0,1] - lane[0,0] <= 0
 R290: tkey[0,2,2] - lane[0,0] <= 0
 R291: tkey[2,2,3] - lane[0,0] <= 0
 R292: tkey[1,0,4] - lane[0,0] <= 0
 R293: tkey[3,0,5] - lane[0,0] <= 0
 R294: tkey[0,1,0] - lane[0,1] <= 0
 R295: tkey[2,1,1] - lane[0,1] <= 0
 R296: tkey[0,0,2] - lane[0,1] <= 0
 R297: tkey[2,0,3] - lane[0,1] <= 0
 R298: tkey[0,2,4] - lane[0,1] <= 0
 R299: tkey[2,2,5] - lane[0,1] <= 0
 R300: tkey[0,2,0] - lane[0,2] <= 0
 R301: tkey[2,2,1] - lane[0,2] <= 0
 R302: tkey[1,0,2] - lane[0,2] <= 0
 R303: tkey[3,0,3] - lane[0,2] <= 0
 R304: tkey[1,2,4] - lane[0,2] <= 0
 R305: tkey[3,2,5] - lane[0,2] <= 0
 R306: tkey[0,3,0] - lane[0,3] <= 0
 R307: tkey[2,3,1] - lane[0,3] <= 0
 R308: tkey[1,3,2] - lane[0,3] <= 0
 R309: tkey[3,3,3] - lane[0,3] <= 0
 R310: tkey[0,1,4] - lane[0,3] <= 0
 R311: tkey[2,1,5] - lane[0,3] <= 0
 R312: tkey[1,0,0] - lane[1,0] <= 0
 R313: tkey[3,0,1] - lane[1,0] <= 0
 R314: tkey[1,2,2] - lane[1,0] <= 0
 R315: tkey[3,2,3] - lane[1,0] <= 0
 R316: tkey[1,1,4] - lane[1,0] <= 0
 R317: tkey[3,1,5] - lane[1,0] <= 0
 R318: tkey[1,1,0] - lane[1,1] <= 0
 R319: tkey[3,1,1] - lane[1,1] <= 0
 R320: tkey[0,3,2] - lane[1,1] <= 0
 R321: tkey[2,3,3] - lane[1,1] <= 0
 R322: tkey[1,3,4] - lane[1,1] <= 0
 R323: tkey[3,3,5] - lane[1,1] <= 0
 R324: tkey[1,2,0] - lane[1,2] <= 0
 R325: tkey[3,2,1] - lane[1,2] <= 0
 R326: tkey[1,1,2] - lane[1,2] <= 0
 R327: tkey[3,1,3] - lane[1,2] <= 0
 R328: tkey[0,3,4] - lane[1,2] <= 0
 R329: tkey[2,3,5] - lane[1,2] <= 0
 R330: tkey[1,3,0] - lane[1,3] <= 0
 R331: tkey[3,3,1] - lane[1,3] <= 0
 R332: tkey[0,1,2] - lane[1,3] <= 0
 R333: tkey[2,1,3] - lane[1,3] <= 0
 R334: tkey[0,0,4] - lane[1,3] <= 0
 R335: tkey[2,0,5] - lane[1,3] <= 0
 R336: tkey[2,0,0] - lane[2,0] <= 0
 R337: tkey[0,2,1] - lane[2,0] <= 0
 R338: tkey[2,2,2] - lane[2,0] <= 0
 R339: tkey[1,0,3] - lane[2,0] <= 0
 R340: tkey[3,0,4] - lane[2,0] <= 0
 R341: tkey[1,2,5] - lane[2,0] <= 0
 R342: tkey[2,1,0] - lane[2,1] <= 0
 R343: tkey[0,0,1] - lane[2,1] <= 0
 R344: tkey[2,0,2] - lane[2,1] <= 0
 R345: tkey[0,2,3] - lane[2,1] <= 0
 R346: tkey[2,2,4] - lane[2,1] <= 0
 R347: tkey[1,0,5] - lane[2,1] <= 0
 R348: tkey[2,2,0] - lane[2,2] <= 0
 R349: tkey[1,0,1] - lane[2,2] <= 0
 R350: tkey[3,0,2] - lane[2,2] <= 0
 R351: tkey[1,2,3] - lane[2,2] <= 0
 R352: tkey[3,2,4] - lane[2,2] <= 0
 R353: tkey[1,1,5] - lane[2,2] <= 0
 R354: tkey[2,3,0] - lane[2,3] <= 0
 R355: tkey[1,3,1] - lane[2,3] <= 0
 R356: tkey[3,3,2] - lane[2,3] <= 0
 R357: tkey[0,1,3] - lane[2,3] <= 0
 R358: tkey[2,1,4] - lane[2,3] <= 0
 R359: tkey[0,0,5] - lane[2,3] <= 0
 R360: tkey[3,0,0] - lane[3,0] <= 0
 R361: tkey[1,2,1] - lane[3,0] <= 0
 R362: tkey[3,2,2] - lane[3,0] <= 0
 R363: tkey[1,1,3] - lane[3,0] <= 0
 R364: tkey[3,1,4] - lane[3,0] <= 0
 R365: tkey[0,3,5] - lane[3,0] <= 0
 R366: tkey[3,1,0] - lane[3,1] <= 0
 R367: tkey[0,3,1] - lane[3,1] <= 0
 R368: tkey[2,3,2] - lane[3,1] <= 0
 R369: tkey[1,3,3] - lane[3,1] <= 0
 R370: tkey[3,3,4] - lane[3,1] <= 0
 R371: tkey[0,1,5] - lane[3,1] <= 0
 R372: tkey[3,2,0] - lane[3,2] <= 0
 R373: tkey[1,1,1] - lane[3,2] <= 0
 R374: tkey[3,1,2] - lane[3,2] <= 0
 R375: tkey[0,3,3] - lane[3,2] <= 0
 R376: tkey[2,3,4] - lane[3,2] <= 0
 R377: tkey[1,3,5] - lane[3,2] <= 0
 R378: tkey[3,3,0] - lane[3,3] <= 0
 R379: tkey[0,1,1] - lane[3,3] <= 0
 R380: tkey[2,1,2] - lane[3,3] <= 0
 R381: tkey[0,0,3] - lane[3,3] <= 0
 R382: tkey[2,0,4] - lane[3,3] <= 0
 R383: tkey[0,2,5] - lane[3,3] <= 0
 R384: - tkey[0,0,0] - tkey[0,2,2] - tkey[1,0,4] - tkey[2,0,1]
   - tkey[2,2,3] - tkey[3,0,5] + 6 lane[0,0] <= 1
 R385: - tkey[0,0,2] - tkey[0,1,0] - tkey[0,2,4] - tkey[2,0,3]
   - tkey[2,1,1] - tkey[2,2,5] + 6 lane[0,1] <= 1
 R386: - tkey[0,2,0] - tkey[1,0,2] - tkey[1,2,4] - tkey[2,2,1]
   - tkey[3,0,3] - tkey[3,2,5] + 6 lane[0,2] <= 1
 R387: - tkey[0,1,4] - tkey[0,3,0] - tkey[1,3,2] - tkey[2,1,5]
   - tkey[2,3,1] - tkey[3,3,3] + 6 lane[0,3] <= 1
 R388: - tkey[1,0,0] - tkey[1,1,4] - tkey[1,2,2] - tkey[3,0,1]
   - tkey[3,1,5] - tkey[3,2,3] + 6 lane[1,0] <= 1
 R389: - tkey[0,3,2] - tkey[1,1,0] - tkey[1,3,4] - tkey[2,3,3]
   - tkey[3,1,1] - tkey[3,3,5] + 6 lane[1,1] <= 1
 R390: - tkey[0,3,4] - tkey[1,1,2] - tkey[1,2,0] - tkey[2,3,5]
   - tkey[3,1,3] - tkey[3,2,1] + 6 lane[1,2] <= 1
 R391: - tkey[0,0,4] - tkey[0,1,2] - tkey[1,3,0] - tkey[2,0,5]
   - tkey[2,1,3] - tkey[3,3,1] + 6 lane[1,3] <= 1
 R392: - tkey[0,2,1] - tkey[1,0,3] - tkey[1,2,5] - tkey[2,0,0]
   - tkey[2,2,2] - tkey[3,0,4] + 6 lane[2,0] <= 1
 R393: - tkey[0,0,1] - tkey[0,2,3] - tkey[1,0,5] - tkey[2,0,2]
   - tkey[2,1,0] - tkey[2,2,4] + 6 lane[2,1] <= 1
 R394: - tkey[1,0,1] - tkey[1,1,5] - tkey[1,2,3] - tkey[2,2,0]
   - tkey[3,0,2] - tkey[3,2,4] + 6 lane[2,2] <= 1
 R395: - tkey[0,0,5] - tkey[0,1,3] - tkey[1,3,1] - tkey[2,1,4]
   - tkey[2,3,0] - tkey[3,3,2] + 6 lane[2,3] <= 1
 R396: - tkey[0,3,5] - tkey[1,1,3] - tkey[1,2,1] - tkey[3,0,0]
   - tkey[3,1,4] - tkey[3,2,2] + 6 lane[3,0] <= 1
 R397: - tkey[0,1,5] - tkey[0,3,1] - tkey[1,3,3] - tkey[2,3,2]
   - tkey[3,1,0] - tkey[3,3,4] + 6 lane[3,1] <= 1
 R398: - tkey[0,3,3] - tkey[1,1,1] - tkey[1,3,5] - tkey[2,3,4]
   - tkey[3,1,2] - tkey[3,2,0] + 6 lane[3,2] <= 1
 R399: - tkey[0,0,3] - tkey[0,1,1] - tkey[0,2,5] - tkey[2,0,4]
   - tkey[2,1,2] - tkey[3,3,0] + 6 lane[3,3] <= 1
 R400: - tbox[0,0,0] + forg[0,0,0] <= 0
 R401: sbox[0,0,0] - rkey[0,0,0] + tkey[0,0,0] >= 0
 R402: sbox[0,0,0] + rkey[0,0,0] - tkey[0,0,0] >= 0
 R403: - sbox[0,0,0] + rkey[0,0,0] + tkey[0,0,0] >= 0
 R404: - tbox[1,0,0] + forg[1,0,0] <= 0
 R405: sbox[1,0,0] - rkey[1,0,0] + tkey[1,0,0] >= 0
 R406: sbox[1,0,0] + rkey[1,0,0] - tkey[1,0,0] >= 0
 R407: - sbox[1,0,0] + rkey[1,0,0] + tkey[1,0,0] >= 0
 R408: - tbox[2,0,0] + forg[2,0,0] <= 0
 R409: - sbox[2,0,0] + rkey[2,0,0] = 0
 R410: - tbox[3,0,0] + forg[3,0,0] <= 0
 R411: - sbox[3,0,0] + rkey[3,0,0] = 0
 R412: - sbox[3,0,1] + rkey[0,0,0] + rkey[2,2,0] >= 0
 R413: sbox[3,0,1] + rkey[0,0,0] - rkey[2,2,0] >= 0
 R414: sbox[3,0,1] - rkey[0,0,0] + rkey[2,2,0] >= 0
 R415: - sbox[0,0,1] + sbox[3,0,1] + rkey[3,1,0] >= 0
 R416: sbox[0,0,1] + sbox[3,0,1] - rkey[3,1,0] >= 0
 R417: sbox[0,0,1] - sbox[3,0,1] + rkey[3,1,0] >= 0
 R418: - sbox[2,0,1] + rkey[1,3,0] + rkey[2,2,0] >= 0
 R419: sbox[2,0,1] + rkey[1,3,0] - rkey[2,2,0] >= 0
 R420: sbox[2,0,1] - rkey[1,3,0] + rkey[2,2,0] >= 0
 R421: sbox[1,0,1] - rkey[0,0,0] = 0
 R422: tbox[0,0,0] + tbox[2,2,0] - tbox[3,0,1] >= 0
 R423: tbox[0,0,0] - tbox[2,2,0] + tbox[3,0,1] >= 0
 R424: - tbox[0,0,0] + tbox[2,2,0] + tbox[3,0,1] >= 0
 R425: - tbox[0,0,1] + tbox[3,0,1] + tbox[3,1,0] >= 0
 R426: tbox[0,0,1] + tbox[3,0,1] - tbox[3,1,0] >= 0
 R427: tbox[0,0,1] - tbox[3,0,1] + tbox[3,1,0] >= 0
 R428: tbox[1,3,0] - tbox[2,0,1] + tbox[2,2,0] >= 0
 R429: tbox[1,3,0] + tbox[2,0,1] - tbox[2,2,0] >= 0
 R430: - tbox[1,3,0] + tbox[2,0,1] + tbox[2,2,0] >= 0
 R431: - tbox[0,0,0] + tbox[1,0,1] = 0
 R432: unkn[0,0,0] - unkn[3,0,1] + forg[3,0,1] <= 0
 R433: unkn[2,2,0] - unkn[3,0,1] + forg[3,0,1] <= 0
 R434: unkn[0,0,0] + unkn[2,2,0] - unkn[3,0,1] + forg[3,0,1] >= 0
 R435: - unkn[0,0,1] + unkn[3,0,1] + forg[0,0,1] - forg[3,0,1] <= 0
 R436: - unkn[0,0,1] + unkn[3,1,0] + forg[0,0,1] <= 0
 R437: - unkn[0,0,1] + unkn[3,0,1] + unkn[3,1,0] + forg[0,0,1]
   - forg[3,0,1] >= 0
 R438: unkn[1,3,0] - unkn[2,0,1] + forg[2,0,1] <= 0
 R439: - unkn[2,0,1] + unkn[2,2,0] + forg[2,0,1] <= 0
 R440: unkn[1,3,0] - unkn[2,0,1] + unkn[2,2,0] + forg[2,0,1] >= 0
 R441: unkn[0,0,0] - unkn[1,0,1] + forg[1,0,1] = 0
 R442: - tbox[0,1,0] + forg[0,1,0] <= 0
 R443: sbox[0,1,0] - rkey[0,1,0] + tkey[0,1,0] >= 0
 R444: sbox[0,1,0] + rkey[0,1,0] - tkey[0,1,0] >= 0
 R445: - sbox[0,1,0] + rkey[0,1,0] + tkey[0,1,0] >= 0
 R446: - tbox[1,1,0] + forg[1,1,0] <= 0
 R447: sbox[1,1,0] - rkey[1,1,0] + tkey[1,1,0] >= 0
 R448: sbox[1,1,0] + rkey[1,1,0] - tkey[1,1,0] >= 0
 R449: - sbox[1,1,0] + rkey[1,1,0] + tkey[1,1,0] >= 0
 R450: - tbox[2,1,0] + forg[2,1,0] <= 0
 R451: - sbox[2,1,0] + rkey[2,1,0] = 0
 R452: - tbox[3,1,0] + forg[3,1,0] <= 0
 R453: - sbox[3,1,0] + rkey[3,1,0] = 0
 R454: - sbox[3,1,1] + rkey[0,1,0] + rkey[2,3,0] >= 0
 R455: sbox[3,1,1] + rkey[0,1,0] - rkey[2,3,0] >= 0
 R456: sbox[3,1,1] - rkey[0,1,0] + rkey[2,3,0] >= 0
 R457: - sbox[0,1,1] + sbox[3,1,1] + rkey[3,2,0] >= 0
 R458: sbox[0,1,1] + sbox[3,1,1] - rkey[3,2,0] >= 0
 R459: sbox[0,1,1] - sbox[3,1,1] + rkey[3,2,0] >= 0
 R460: - sbox[2,1,1] + rkey[1,0,0] + rkey[2,3,0] >= 0
 R461: sbox[2,1,1] + rkey[1,0,0] - rkey[2,3,0] >= 0
 R462: sbox[2,1,1] - rkey[1,0,0] + rkey[2,3,0] >= 0
 R463: sbox[1,1,1] - rkey[0,1,0] = 0
 R464: tbox[0,1,0] + tbox[2,3,0] - tbox[3,1,1] >= 0
 R465: tbox[0,1,0] - tbox[2,3,0] + tbox[3,1,1] >= 0
 R466: - tbox[0,1,0] + tbox[2,3,0] + tbox[3,1,1] >= 0
 R467: - tbox[0,1,1] + tbox[3,1,1] + tbox[3,2,0] >= 0
 R468: tbox[0,1,1] + tbox[3,1,1] - tbox[3,2,0] >= 0
 R469: tbox[0,1,1] - tbox[3,1,1] + tbox[3,2,0] >= 0
 R470: tbox[1,0,0] - tbox[2,1,1] + tbox[2,3,0] >= 0
 R471: tbox[1,0,0] + tbox[2,1,1] - tbox[2,3,0] >= 0
 R472: - tbox[1,0,0] + tbox[2,1,1] + tbox[2,3,0] >= 0
 R473: - tbox[0,1,0] + tbox[1,1,1] = 0
 R474: unkn[0,1,0] - unkn[3,1,1] + forg[3,1,1] <= 0
 R475: unkn[2,3,0] - unkn[3,1,1] + forg[3,1,1] <= 0
 R476: unkn[0,1,0] + unkn[2,3,0] - unkn[3,1,1] + forg[3,1,1] >= 0
 R477: - unkn[0,1,1] + unkn[3,1,1] + forg[0,1,1] - forg[3,1,1] <= 0
 R478: - unkn[0,1,1] + unkn[3,2,0] + forg[0,1,1] <= 0
 R479: - unkn[0,1,1] + unkn[3,1,1] + unkn[3,2,0] + forg[0,1,1]
   - forg[3,1,1] >= 0
 R480: unkn[1,0,0] - unkn[2,1,1] + forg[2,1,1] <= 0
 R481: - unkn[2,1,1] + unkn[2,3,0] + forg[2,1,1] <= 0
 R482: unkn[1,0,0] - unkn[2,1,1] + unkn[2,3,0] + forg[2,1,1] >= 0
 R483: unkn[0,1,0] - unkn[1,1,1] + forg[1,1,1] = 0
 R484: - tbox[0,2,0] + forg[0,2,0] <= 0
 R485: sbox[0,2,0] - rkey[0,2,0] + tkey[0,2,0] >= 0
 R486: sbox[0,2,0] + rkey[0,2,0] - tkey[0,2,0] >= 0
 R487: - sbox[0,2,0] + rkey[0,2,0] + tkey[0,2,0] >= 0
 R488: - tbox[1,2,0] + forg[1,2,0] <= 0
 R489: sbox[1,2,0] - rkey[1,2,0] + tkey[1,2,0] >= 0
 R490: sbox[1,2,0] + rkey[1,2,0] - tkey[1,2,0] >= 0
 R491: - sbox[1,2,0] + rkey[1,2,0] + tkey[1,2,0] >= 0
 R492: - tbox[2,2,0] + forg[2,2,0] <= 0
 R493: - sbox[2,2,0] + rkey[2,2,0] = 0
 R494: - tbox[3,2,0] + forg[3,2,0] <= 0
 R495: - sbox[3,2,0] + rkey[3,2,0] = 0
 R496: - sbox[3,2,1] + rkey[0,2,0] + rkey[2,0,0] >= 0
 R497: sbox[3,2,1] + rkey[0,2,0] - rkey[2,0,0] >= 0
 R498: sbox[3,2,1] - rkey[0,2,0] + rkey[2,0,0] >= 0
 R499: - sbox[0,2,1] + sbox[3,2,1] + rkey[3,3,0] >= 0
 R500: sbox[0,2,1] + sbox[3,2,1] - rkey[3,3,0] >= 0
 R501: sbox[0,2,1] - sbox[3,2,1] + rkey[3,3,0] >= 0
 R502: - sbox[2,2,1] + rkey[1,1,0] + rkey[2,0,0] >= 0
 R503: sbox[2,2,1] + rkey[1,1,0] - rkey[2,0,0] >= 0
 R504: sbox[2,2,1] - rkey[1,1,0] + rkey[2,0,0] >= 0
 R505: sbox[1,2,1] - rkey[0,2,0] = 0
 R506: tbox[0,2,0] + tbox[2,0,0] - tbox[3,2,1] >= 0
 R507: tbox[0,2,0] - tbox[2,0,0] + tbox[3,2,1] >= 0
 R508: - tbox[0,2,0] + tbox[2,0,0] + tbox[3,2,1] >= 0
 R509: - tbox[0,2,1] + tbox[3,2,1] + tbox[3,3,0] >= 0
 R510: tbox[0,2,1] + tbox[3,2,1] - tbox[3,3,0] >= 0
 R511: tbox[0,2,1] - tbox[3,2,1] + tbox[3,3,0] >= 0
 R512: tbox[1,1,0] + tbox[2,0,0] - tbox[2,2,1] >= 0
 R513: tbox[1,1,0] - tbox[2,0,0] + tbox[2,2,1] >= 0
 R514: - tbox[1,1,0] + tbox[2,0,0] + tbox[2,2,1] >= 0
 R515: - tbox[0,2,0] + tbox[1,2,1] = 0
 R516: unkn[0,2,0] - unkn[3,2,1] + forg[3,2,1] <= 0
 R517: unkn[2,0,0] - unkn[3,2,1] + forg[3,2,1] <= 0
 R518: unkn[0,2,0] + unkn[2,0,0] - unkn[3,2,1] + forg[3,2,1] >= 0
 R519: - unkn[0,2,1] + unkn[3,2,1] + forg[0,2,1] - forg[3,2,1] <= 0
 R520: - unkn[0,2,1] + unkn[3,3,0] + forg[0,2,1] <= 0
 R521: - unkn[0,2,1] + unkn[3,2,1] + unkn[3,3,0] + forg[0,2,1]
   - forg[3,2,1] >= 0
 R522: unkn[1,1,0] - unkn[2,2,1] + forg[2,2,1] <= 0
 R523: unkn[2,0,0] - unkn[2,2,1] + forg[2,2,1] <= 0
 R524: unkn[1,1,0] + unkn[2,0,0] - unkn[2,2,1] + forg[2,2,1] >= 0
 R525: unkn[0,2,0] - unkn[1,2,1] + forg[1,2,1] = 0
 R526: - tbox[0,3,0] + forg[0,3,0] <= 0
 R527: sbox[0,3,0] - rkey[0,3,0] + tkey[0,3,0] >= 0
 R528: sbox[0,3,0] + rkey[0,3,0] - tkey[0,3,0] >= 0
 R529: - sbox[0,3,0] + rkey[0,3,0] + tkey[0,3,0] >= 0
 R530: - tbox[1,3,0] + forg[1,3,0] <= 0
 R531: sbox[1,3,0] - rkey[1,3,0] + tkey[1,3,0] >= 0
 R532: sbox[1,3,0] + rkey[1,3,0] - tkey[1,3,0] >= 0
 R533: - sbox[1,3,0] + rkey[1,3,0] + tkey[1,3,0] >= 0
 R534: - tbox[2,3,0] + forg[2,3,0] <= 0
 R535: - sbox[2,3,0] + rkey[2,3,0] = 0
 R536: - tbox[3,3,0] + forg[3,3,0] <= 0
 R537: - sbox[3,3,0] + rkey[3,3,0] = 0
 R538: - sbox[3,3,1] + rkey[0,3,0] + rkey[2,1,0] >= 0
 R539: sbox[3,3,1] + rkey[0,3,0] - rkey[2,1,0] >= 0
 R540: sbox[3,3,1] - rkey[0,3,0] + rkey[2,1,0] >= 0
 R541: - sbox[0,3,1] + sbox[3,3,1] + rkey[3,0,0] >= 0
 R542: sbox[0,3,1] + sbox[3,3,1] - rkey[3,0,0] >= 0
 R543: sbox[0,3,1] - sbox[3,3,1] + rkey[3,0,0] >= 0
 R544: - sbox[2,3,1] + rkey[1,2,0] + rkey[2,1,0] >= 0
 R545: sbox[2,3,1] + rkey[1,2,0] - rkey[2,1,0] >= 0
 R546: sbox[2,3,1] - rkey[1,2,0] + rkey[2,1,0] >= 0
 R547: sbox[1,3,1] - rkey[0,3,0] = 0
 R548: tbox[0,3,0] + tbox[2,1,0] - tbox[3,3,1] >= 0
 R549: tbox[0,3,0] - tbox[2,1,0] + tbox[3,3,1] >= 0
 R550: - tbox[0,3,0] + tbox[2,1,0] + tbox[3,3,1] >= 0
 R551: - tbox[0,3,1] + tbox[3,0,0] + tbox[3,3,1] >= 0
 R552: tbox[0,3,1] - tbox[3,0,0] + tbox[3,3,1] >= 0
 R553: tbox[0,3,1] + tbox[3,0,0] - tbox[3,3,1] >= 0
 R554: tbox[1,2,0] + tbox[2,1,0] - tbox[2,3,1] >= 0
 R555: tbox[1,2,0] - tbox[2,1,0] + tbox[2,3,1] >= 0
 R556: - tbox[1,2,0] + tbox[2,1,0] + tbox[2,3,1] >= 0
 R557: - tbox[0,3,0] + tbox[1,3,1] = 0
 R558: unkn[0,3,0] - unkn[3,3,1] + forg[3,3,1] <= 0
 R559: unkn[2,1,0] - unkn[3,3,1] + forg[3,3,1] <= 0
 R560: unkn[0,3,0] + unkn[2,1,0] - unkn[3,3,1] + forg[3,3,1] >= 0
 R561: - unkn[0,3,1] + unkn[3,3,1] + forg[0,3,1] - forg[3,3,1] <= 0
 R562: - unkn[0,3,1] + unkn[3,0,0] + forg[0,3,1] <= 0
 R563: - unkn[0,3,1] + unkn[3,0,0] + unkn[3,3,1] + forg[0,3,1]
   - forg[3,3,1] >= 0
 R564: unkn[1,2,0] - unkn[2,3,1] + forg[2,3,1] <= 0
 R565: unkn[2,1,0] - unkn[2,3,1] + forg[2,3,1] <= 0
 R566: unkn[1,2,0] + unkn[2,1,0] - unkn[2,3,1] + forg[2,3,1] >= 0
 R567: unkn[0,3,0] - unkn[1,3,1] + forg[1,3,1] = 0
 R568: - tbox[0,0,1] + forg[0,0,1] <= 0
 R569: sbox[0,0,1] - rkey[0,0,1] + tkey[0,0,1] >= 0
 R570: sbox[0,0,1] + rkey[0,0,1] - tkey[0,0,1] >= 0
 R571: - sbox[0,0,1] + rkey[0,0,1] + tkey[0,0,1] >= 0
 R572: - tbox[1,0,1] + forg[1,0,1] <= 0
 R573: sbox[1,0,1] - rkey[1,0,1] + tkey[1,0,1] >= 0
 R574: sbox[1,0,1] + rkey[1,0,1] - tkey[1,0,1] >= 0
 R575: - sbox[1,0,1] + rkey[1,0,1] + tkey[1,0,1] >= 0
 R576: - tbox[2,0,1] + forg[2,0,1] <= 0
 R577: - sbox[2,0,1] + rkey[2,0,1] = 0
 R578: - tbox[3,0,1] + forg[3,0,1] <= 0
 R579: - sbox[3,0,1] + rkey[3,0,1] = 0
 R580: - sbox[3,0,2] + rkey[0,0,1] + rkey[2,2,1] >= 0
 R581: sbox[3,0,2] + rkey[0,0,1] - rkey[2,2,1] >= 0
 R582: sbox[3,0,2] - rkey[0,0,1] + rkey[2,2,1] >= 0
 R583: - sbox[0,0,2] + sbox[3,0,2] + rkey[3,1,1] >= 0
 R584: sbox[0,0,2] + sbox[3,0,2] - rkey[3,1,1] >= 0
 R585: sbox[0,0,2] - sbox[3,0,2] + rkey[3,1,1] >= 0
 R586: - sbox[2,0,2] + rkey[1,3,1] + rkey[2,2,1] >= 0
 R587: sbox[2,0,2] + rkey[1,3,1] - rkey[2,2,1] >= 0
 R588: sbox[2,0,2] - rkey[1,3,1] + rkey[2,2,1] >= 0
 R589: sbox[1,0,2] - rkey[0,0,1] = 0
 R590: tbox[0,0,1] + tbox[2,2,1] - tbox[3,0,2] >= 0
 R591: tbox[0,0,1] - tbox[2,2,1] + tbox[3,0,2] >= 0
 R592: - tbox[0,0,1] + tbox[2,2,1] + tbox[3,0,2] >= 0
 R593: - tbox[0,0,2] + tbox[3,0,2] + tbox[3,1,1] >= 0
 R594: tbox[0,0,2] + tbox[3,0,2] - tbox[3,1,1] >= 0
 R595: tbox[0,0,2] - tbox[3,0,2] + tbox[3,1,1] >= 0
 R596: tbox[1,3,1] - tbox[2,0,2] + tbox[2,2,1] >= 0
 R597: tbox[1,3,1] + tbox[2,0,2] - tbox[2,2,1] >= 0
 R598: - tbox[1,3,1] + tbox[2,0,2] + tbox[2,2,1] >= 0
 R599: - tbox[0,0,1] + tbox[1,0,2] = 0
 R600: unkn[0,0,1] - unkn[3,0,2] + forg[3,0,2] <= 0
 R601: unkn[2,2,1] - unkn[3,0,2] + forg[3,0,2] <= 0
 R602: unkn[0,0,1] + unkn[2,2,1] - unkn[3,0,2] + forg[3,0,2] >= 0
 R603: - unkn[0,0,2] + unkn[3,0,2] + forg[0,0,2] - forg[3,0,2] <= 0
 R604: - unkn[0,0,2] + unkn[3,1,1] + forg[0,0,2] <= 0
 R605: - unkn[0,0,2] + unkn[3,0,2] + unkn[3,1,1] + forg[0,0,2]
   - forg[3,0,2] >= 0
 R606: unkn[1,3,1] - unkn[2,0,2] + forg[2,0,2] <= 0
 R607: - unkn[2,0,2] + unkn[2,2,1] + forg[2,0,2] <= 0
 R608: unkn[1,3,1] - unkn[2,0,2] + unkn[2,2,1] + forg[2,0,2] >= 0
 R609: unkn[0,0,1] - unkn[1,0,2] + forg[1,0,2] = 0
 R610: - tbox[0,1,1] + forg[0,1,1] <= 0
 R611: sbox[0,1,1] - rkey[0,1,1] + tkey[0,1,1] >= 0
 R612: sbox[0,1,1] + rkey[0,1,1] - tkey[0,1,1] >= 0
 R613: - sbox[0,1,1] + rkey[0,1,1] + tkey[0,1,1] >= 0
 R614: - tbox[1,1,1] + forg[1,1,1] <= 0
 R615: sbox[1,1,1] - rkey[1,1,1] + tkey[1,1,1] >= 0
 R616: sbox[1,1,1] + rkey[1,1,1] - tkey[1,1,1] >= 0
 R617: - sbox[1,1,1] + rkey[1,1,1] + tkey[1,1,1] >= 0
 R618: - tbox[2,1,1] + forg[2,1,1] <= 0
 R619: - sbox[2,1,1] + rkey[2,1,1] = 0
 R620: - tbox[3,1,1] + forg[3,1,1] <= 0
 R621: - sbox[3,1,1] + rkey[3,1,1] = 0
 R622: - sbox[3,1,2] + rkey[0,1,1] + rkey[2,3,1] >= 0
 R623: sbox[3,1,2] + rkey[0,1,1] - rkey[2,3,1] >= 0
 R624: sbox[3,1,2] - rkey[0,1,1] + rkey[2,3,1] >= 0
 R625: - sbox[0,1,2] + sbox[3,1,2] + rkey[3,2,1] >= 0
 R626: sbox[0,1,2] + sbox[3,1,2] - rkey[3,2,1] >= 0
 R627: sbox[0,1,2] - sbox[3,1,2] + rkey[3,2,1] >= 0
 R628: - sbox[2,1,2] + rkey[1,0,1] + rkey[2,3,1] >= 0
 R629: sbox[2,1,2] + rkey[1,0,1] - rkey[2,3,1] >= 0
 R630: sbox[2,1,2] - rkey[1,0,1] + rkey[2,3,1] >= 0
 R631: sbox[1,1,2] - rkey[0,1,1] = 0
 R632: tbox[0,1,1] + tbox[2,3,1] - tbox[3,1,2] >= 0
 R633: tbox[0,1,1] - tbox[2,3,1] + tbox[3,1,2] >= 0
 R634: - tbox[0,1,1] + tbox[2,3,1] + tbox[3,1,2] >= 0
 R635: - tbox[0,1,2] + tbox[3,1,2] + tbox[3,2,1] >= 0
 R636: tbox[0,1,2] + tbox[3,1,2] - tbox[3,2,1] >= 0
 R637: tbox[0,1,2] - tbox[3,1,2] + tbox[3,2,1] >= 0
 R638: tbox[1,0,1] - tbox[2,1,2] + tbox[2,3,1] >= 0
 R639: tbox[1,0,1] + tbox[2,1,2] - tbox[2,3,1] >= 0
 R640: - tbox[1,0,1] + tbox[2,1,2] + tbox[2,3,1] >= 0
 R641: - tbox[0,1,1] + tbox[1,1,2] = 0
 R642: unkn[0,1,1] - unkn[3,1,2] + forg[3,1,2] <= 0
 R643: unkn[2,3,1] - unkn[3,1,2] + forg[3,1,2] <= 0
 R644: unkn[0,1,1] + unkn[2,3,1] - unkn[3,1,2] + forg[3,1,2] >= 0
 R645: - unkn[0,1,2] + unkn[3,1,2] + forg[0,1,2] - forg[3,1,2] <= 0
 R646: - unkn[0,1,2] + unkn[3,2,1] + forg[0,1,2] <= 0
 R647: - unkn[0,1,2] + unkn[3,1,2] + unkn[3,2,1] + forg[0,1,2]
   - forg[3,1,2] >= 0
 R648: unkn[1,0,1] - unkn[2,1,2] + forg[2,1,2] <= 0
 R649: - unkn[2,1,2] + unkn[2,3,1] + forg[2,1,2] <= 0
 R650: unkn[1,0,1] - unkn[2,1,2] + unkn[2,3,1] + forg[2,1,2] >= 0
 R651: unkn[0,1,1] - unkn[1,1,2] + forg[1,1,2] = 0
 R652: - tbox[0,2,1] + forg[0,2,1] <= 0
 R653: sbox[0,2,1] - rkey[0,2,1] + tkey[0,2,1] >= 0
 R654: sbox[0,2,1] + rkey[0,2,1] - tkey[0,2,1] >= 0
 R655: - sbox[0,2,1] + rkey[0,2,1] + tkey[0,2,1] >= 0
 R656: - tbox[1,2,1] + forg[1,2,1] <= 0
 R657: sbox[1,2,1] - rkey[1,2,1] + tkey[1,2,1] >= 0
 R658: sbox[1,2,1] + rkey[1,2,1] - tkey[1,2,1] >= 0
 R659: - sbox[1,2,1] + rkey[1,2,1] + tkey[1,2,1] >= 0
 R660: - tbox[2,2,1] + forg[2,2,1] <= 0
 R661: - sbox[2,2,1] + rkey[2,2,1] = 0
 R662: - tbox[3,2,1] + forg[3,2,1] <= 0
 R663: - sbox[3,2,1] + rkey[3,2,1] = 0
 R664: - sbox[3,2,2] + rkey[0,2,1] + rkey[2,0,1] >= 0
 R665: sbox[3,2,2] + rkey[0,2,1] - rkey[2,0,1] >= 0
 R666: sbox[3,2,2] - rkey[0,2,1] + rkey[2,0,1] >= 0
 R667: - sbox[0,2,2] + sbox[3,2,2] + rkey[3,3,1] >= 0
 R668: sbox[0,2,2] + sbox[3,2,2] - rkey[3,3,1] >= 0
 R669: sbox[0,2,2] - sbox[3,2,2] + rkey[3,3,1] >= 0
 R670: - sbox[2,2,2] + rkey[1,1,1] + rkey[2,0,1] >= 0
 R671: sbox[2,2,2] + rkey[1,1,1] - rkey[2,0,1] >= 0
 R672: sbox[2,2,2] - rkey[1,1,1] + rkey[2,0,1] >= 0
 R673: sbox[1,2,2] - rkey[0,2,1] = 0
 R674: tbox[0,2,1] + tbox[2,0,1] - tbox[3,2,2] >= 0
 R675: tbox[0,2,1] - tbox[2,0,1] + tbox[3,2,2] >= 0
 R676: - tbox[0,2,1] + tbox[2,0,1] + tbox[3,2,2] >= 0
 R677: - tbox[0,2,2] + tbox[3,2,2] + tbox[3,3,1] >= 0
 R678: tbox[0,2,2] + tbox[3,2,2] - tbox[3,3,1] >= 0
 R679: tbox[0,2,2] - tbox[3,2,2] + tbox[3,3,1] >= 0
 R680: tbox[1,1,1] + tbox[2,0,1] - tbox[2,2,2] >= 0
 R681: tbox[1,1,1] - tbox[2,0,1] + tbox[2,2,2] >= 0
 R682: - tbox[1,1,1] + tbox[2,0,1] + tbox[2,2,2] >= 0
 R683: - tbox[0,2,1] + tbox[1,2,2] = 0
 R684: unkn[0,2,1] - unkn[3,2,2] + forg[3,2,2] <= 0
 R685: unkn[2,0,1] - unkn[3,2,2] + forg[3,2,2] <= 0
 R686: unkn[0,2,1] + unkn[2,0,1] - unkn[3,2,2] + forg[3,2,2] >= 0
 R687: - unkn[0,2,2] + unkn[3,2,2] + forg[0,2,2] - forg[3,2,2] <= 0
 R688: - unkn[0,2,2] + unkn[3,3,1] + forg[0,2,2] <= 0
 R689: - unkn[0,2,2] + unkn[3,2,2] + unkn[3,3,1] + forg[0,2,2]
   - forg[3,2,2] >= 0
 R690: unkn[1,1,1] - unkn[2,2,2] + forg[2,2,2] <= 0
 R691: unkn[2,0,1] - unkn[2,2,2] + forg[2,2,2] <= 0
 R692: unkn[1,1,1] + unkn[2,0,1] - unkn[2,2,2] + forg[2,2,2] >= 0
 R693: unkn[0,2,1] - unkn[1,2,2] + forg[1,2,2] = 0
 R694: - tbox[0,3,1] + forg[0,3,1] <= 0
 R695: sbox[0,3,1] - rkey[0,3,1] + tkey[0,3,1] >= 0
 R696: sbox[0,3,1] + rkey[0,3,1] - tkey[0,3,1] >= 0
 R697: - sbox[0,3,1] + rkey[0,3,1] + tkey[0,3,1] >= 0
 R698: - tbox[1,3,1] + forg[1,3,1] <= 0
 R699: sbox[1,3,1] - rkey[1,3,1] + tkey[1,3,1] >= 0
 R700: sbox[1,3,1] + rkey[1,3,1] - tkey[1,3,1] >= 0
 R701: - sbox[1,3,1] + rkey[1,3,1] + tkey[1,3,1] >= 0
 R702: - tbox[2,3,1] + forg[2,3,1] <= 0
 R703: - sbox[2,3,1] + rkey[2,3,1] = 0
 R704: - tbox[3,3,1] + forg[3,3,1] <= 0
 R705: - sbox[3,3,1] + rkey[3,3,1] = 0
 R706: - sbox[3,3,2] + rkey[0,3,1] + rkey[2,1,1] >= 0
 R707: sbox[3,3,2] + rkey[0,3,1] - rkey[2,1,1] >= 0
 R708: sbox[3,3,2] - rkey[0,3,1] + rkey[2,1,1] >= 0
 R709: - sbox[0,3,2] + sbox[3,3,2] + rkey[3,0,1] >= 0
 R710: sbox[0,3,2] + sbox[3,3,2] - rkey[3,0,1] >= 0
 R711: sbox[0,3,2] - sbox[3,3,2] + rkey[3,0,1] >= 0
 R712: - sbox[2,3,2] + rkey[1,2,1] + rkey[2,1,1] >= 0
 R713: sbox[2,3,2] + rkey[1,2,1] - rkey[2,1,1] >= 0
 R714: sbox[2,3,2] - rkey[1,2,1] + rkey[2,1,1] >= 0
 R715: sbox[1,3,2] - rkey[0,3,1] = 0
 R716: tbox[0,3,1] + tbox[2,1,1] - tbox[3,3,2] >= 0
 R717: tbox[0,3,1] - tbox[2,1,1] + tbox[3,3,2] >= 0
 R718: - tbox[0,3,1] + tbox[2,1,1] + tbox[3,3,2] >= 0
 R719: - tbox[0,3,2] + tbox[3,0,1] + tbox[3,3,2] >= 0
 R720: tbox[0,3,2] - tbox[3,0,1] + tbox[3,3,2] >= 0
 R721: tbox[0,3,2] + tbox[3,0,1] - tbox[3,3,2] >= 0
 R722: tbox[1,2,1] + tbox[2,1,1] - tbox[2,3,2] >= 0
 R723: tbox[1,2,1] - tbox[2,1,1] + tbox[2,3,2] >= 0
 R724: - tbox[1,2,1] + tbox[2,1,1] + tbox[2,3,2] >= 0
 R725: - tbox[0,3,1] + tbox[1,3,2] = 0
 R726: unkn[0,3,1] - unkn[3,3,2] + forg[3,3,2] <= 0
 R727: unkn[2,1,1] - unkn[3,3,2] + forg[3,3,2] <= 0
 R728: unkn[0,3,1] + unkn[2,1,1] - unkn[3,3,2] + forg[3,3,2] >= 0
 R729: - unkn[0,3,2] + unkn[3,3,2] + forg[0,3,2] - forg[3,3,2] <= 0
 R730: - unkn[0,3,2] + unkn[3,0,1] + forg[0,3,2] <= 0
 R731: - unkn[0,3,2] + unkn[3,0,1] + unkn[3,3,2] + forg[0,3,2]
   - forg[3,3,2] >= 0
 R732: unkn[1,2,1] - unkn[2,3,2] + forg[2,3,2] <= 0
 R733: unkn[2,1,1] - unkn[2,3,2] + forg[2,3,2] <= 0
 R734: unkn[1,2,1] + unkn[2,1,1] - unkn[2,3,2] + forg[2,3,2] >= 0
 R735: unkn[0,3,1] - unkn[1,3,2] + forg[1,3,2] = 0
 R736: - tbox[0,0,2] + forg[0,0,2] <= 0
 R737: sbox[0,0,2] - rkey[0,0,2] + tkey[0,0,2] >= 0
 R738: sbox[0,0,2] + rkey[0,0,2] - tkey[0,0,2] >= 0
 R739: - sbox[0,0,2] + rkey[0,0,2] + tkey[0,0,2] >= 0
 R740: - tbox[1,0,2] + forg[1,0,2] <= 0
 R741: sbox[1,0,2] - rkey[1,0,2] + tkey[1,0,2] >= 0
 R742: sbox[1,0,2] + rkey[1,0,2] - tkey[1,0,2] >= 0
 R743: - sbox[1,0,2] + rkey[1,0,2] + tkey[1,0,2] >= 0
 R744: - tbox[2,0,2] + forg[2,0,2] <= 0
 R745: - sbox[2,0,2] + rkey[2,0,2] = 0
 R746: - tbox[3,0,2] + forg[3,0,2] <= 0
 R747: - sbox[3,0,2] + rkey[3,0,2] = 0
 R748: - sbox[3,0,3] + rkey[0,0,2] + rkey[2,2,2] >= 0
 R749: sbox[3,0,3] + rkey[0,0,2] - rkey[2,2,2] >= 0
 R750: sbox[3,0,3] - rkey[0,0,2] + rkey[2,2,2] >= 0
 R751: - sbox[0,0,3] + sbox[3,0,3] + rkey[3,1,2] >= 0
 R752: sbox[0,0,3] + sbox[3,0,3] - rkey[3,1,2] >= 0
 R753: sbox[0,0,3] - sbox[3,0,3] + rkey[3,1,2] >= 0
 R754: - sbox[2,0,3] + rkey[1,3,2] + rkey[2,2,2] >= 0
 R755: sbox[2,0,3] + rkey[1,3,2] - rkey[2,2,2] >= 0
 R756: sbox[2,0,3] - rkey[1,3,2] + rkey[2,2,2] >= 0
 R757: sbox[1,0,3] - rkey[0,0,2] = 0
 R758: tbox[0,0,2] + tbox[2,2,2] - tbox[3,0,3] >= 0
 R759: tbox[0,0,2] - tbox[2,2,2] + tbox[3,0,3] >= 0
 R760: - tbox[0,0,2] + tbox[2,2,2] + tbox[3,0,3] >= 0
 R761: - tbox[0,0,3] + tbox[3,0,3] + tbox[3,1,2] >= 0
 R762: tbox[0,0,3] + tbox[3,0,3] - tbox[3,1,2] >= 0
 R763: tbox[0,0,3] - tbox[3,0,3] + tbox[3,1,2] >= 0
 R764: tbox[1,3,2] - tbox[2,0,3] + tbox[2,2,2] >= 0
 R765: tbox[1,3,2] + tbox[2,0,3] - tbox[2,2,2] >= 0
 R766: - tbox[1,3,2] + tbox[2,0,3] + tbox[2,2,2] >= 0
 R767: - tbox[0,0,2] + tbox[1,0,3] = 0
 R768: unkn[0,0,2] - unkn[3,0,3] + forg[3,0,3] <= 0
 R769: unkn[2,2,2] - unkn[3,0,3] + forg[3,0,3] <= 0
 R770: unkn[0,0,2] + unkn[2,2,2] - unkn[3,0,3] + forg[3,0,3] >= 0
 R771: - unkn[0,0,3] + unkn[3,0,3] + forg[0,0,3] - forg[3,0,3] <= 0
 R772: - unkn[0,0,3] + unkn[3,1,2] + forg[0,0,3] <= 0
 R773: - unkn[0,0,3] + unkn[3,0,3] + unkn[3,1,2] + forg[0,0,3]
   - forg[3,0,3] >= 0
 R774: unkn[1,3,2] - unkn[2,0,3] + forg[2,0,3] <= 0
 R775: - unkn[2,0,3] + unkn[2,2,2] + forg[2,0,3] <= 0
 R776: unkn[1,3,2] - unkn[2,0,3] + unkn[2,2,2] + forg[2,0,3] >= 0
 R777: unkn[0,0,2] - unkn[1,0,3] + forg[1,0,3] = 0
 R778: - tbox[0,1,2] + forg[0,1,2] <= 0
 R779: sbox[0,1,2] - rkey[0,1,2] + tkey[0,1,2] >= 0
 R780: sbox[0,1,2] + rkey[0,1,2] - tkey[0,1,2] >= 0
 R781: - sbox[0,1,2] + rkey[0,1,2] + tkey[0,1,2] >= 0
 R782: - tbox[1,1,2] + forg[1,1,2] <= 0
 R783: sbox[1,1,2] - rkey[1,1,2] + tkey[1,1,2] >= 0
 R784: sbox[1,1,2] + rkey[1,1,2] - tkey[1,1,2] >= 0
 R785: - sbox[1,1,2] + rkey[1,1,2] + tkey[1,1,2] >= 0
 R786: - tbox[2,1,2] + forg[2,1,2] <= 0
 R787: - sbox[2,1,2] + rkey[2,1,2] = 0
 R788: - tbox[3,1,2] + forg[3,1,2] <= 0
 R789: - sbox[3,1,2] + rkey[3,1,2] = 0
 R790: - sbox[3,1,3] + rkey[0,1,2] + rkey[2,3,2] >= 0
 R791: sbox[3,1,3] + rkey[0,1,2] - rkey[2,3,2] >= 0
 R792: sbox[3,1,3] - rkey[0,1,2] + rkey[2,3,2] >= 0
 R793: - sbox[0,1,3] + sbox[3,1,3] + rkey[3,2,2] >= 0
 R794: sbox[0,1,3] + sbox[3,1,3] - rkey[3,2,2] >= 0
 R795: sbox[0,1,3] - sbox[3,1,3] + rkey[3,2,2] >= 0
 R796: - sbox[2,1,3] + rkey[1,0,2] + rkey[2,3,2] >= 0
 R797: sbox[2,1,3] + rkey[1,0,2] - rkey[2,3,2] >= 0
 R798: sbox[2,1,3] - rkey[1,0,2] + rkey[2,3,2] >= 0
 R799: sbox[1,1,3] - rkey[0,1,2] = 0
 R800: tbox[0,1,2] + tbox[2,3,2] - tbox[3,1,3] >= 0
 R801: tbox[0,1,2] - tbox[2,3,2] + tbox[3,1,3] >= 0
 R802: - tbox[0,1,2] + tbox[2,3,2] + tbox[3,1,3] >= 0
 R803: - tbox[0,1,3] + tbox[3,1,3] + tbox[3,2,2] >= 0
 R804: tbox[0,1,3] + tbox[3,1,3] - tbox[3,2,2] >= 0
 R805: tbox[0,1,3] - tbox[3,1,3] + tbox[3,2,2] >= 0
 R806: tbox[1,0,2] - tbox[2,1,3] + tbox[2,3,2] >= 0
 R807: tbox[1,0,2] + tbox[2,1,3] - tbox[2,3,2] >= 0
 R808: - tbox[1,0,2] + tbox[2,1,3] + tbox[2,3,2] >= 0
 R809: - tbox[0,1,2] + tbox[1,1,3] = 0
 R810: unkn[0,1,2] - unkn[3,1,3] + forg[3,1,3] <= 0
 R811: unkn[2,3,2] - unkn[3,1,3] + forg[3,1,3] <= 0
 R812: unkn[0,1,2] + unkn[2,3,2] - unkn[3,1,3] + forg[3,1,3] >= 0
 R813: - unkn[0,1,3] + unkn[3,1,3] + forg[0,1,3] - forg[3,1,3] <= 0
 R814: - unkn[0,1,3] + unkn[3,2,2] + forg[0,1,3] <= 0
 R815: - unkn[0,1,3] + unkn[3,1,3] + unkn[3,2,2] + forg[0,1,3]
   - forg[3,1,3] >= 0
 R816: unkn[1,0,2] - unkn[2,1,3] + forg[2,1,3] <= 0
 R817: - unkn[2,1,3] + unkn[2,3,2] + forg[2,1,3] <= 0
 R818: unkn[1,0,2] - unkn[2,1,3] + unkn[2,3,2] + forg[2,1,3] >= 0
 R819: unkn[0,1,2] - unkn[1,1,3] + forg[1,1,3] = 0
 R820: - tbox[0,2,2] + forg[0,2,2] <= 0
 R821: sbox[0,2,2] - rkey[0,2,2] + tkey[0,2,2] >= 0
 R822: sbox[0,2,2] + rkey[0,2,2] - tkey[0,2,2] >= 0
 R823: - sbox[0,2,2] + rkey[0,2,2] + tkey[0,2,2] >= 0
 R824: - tbox[1,2,2] + forg[1,2,2] <= 0
 R825: sbox[1,2,2] - rkey[1,2,2] + tkey[1,2,2] >= 0
 R826: sbox[1,2,2] + rkey[1,2,2] - tkey[1,2,2] >= 0
 R827: - sbox[1,2,2] + rkey[1,2,2] + tkey[1,2,2] >= 0
 R828: - tbox[2,2,2] + forg[2,2,2] <= 0
 R829: - sbox[2,2,2] + rkey[2,2,2] = 0
 R830: - tbox[3,2,2] + forg[3,2,2] <= 0
 R831: - sbox[3,2,2] + rkey[3,2,2] = 0
 R832: - sbox[3,2,3] + rkey[0,2,2] + rkey[2,0,2] >= 0
 R833: sbox[3,2,3] + rkey[0,2,2] - rkey[2,0,2] >= 0
 R834: sbox[3,2,3] - rkey[0,2,2] + rkey[2,0,2] >= 0
 R835: - sbox[0,2,3] + sbox[3,2,3] + rkey[3,3,2] >= 0
 R836: sbox[0,2,3] + sbox[3,2,3] - rkey[3,3,2] >= 0
 R837: sbox[0,2,3] - sbox[3,2,3] + rkey[3,3,2] >= 0
 R838: - sbox[2,2,3] + rkey[1,1,2] + rkey[2,0,2] >= 0
 R839: sbox[2,2,3] + rkey[1,1,2] - rkey[2,0,2] >= 0
 R840: sbox[2,2,3] - rkey[1,1,2] + rkey[2,0,2] >= 0
 R841: sbox[1,2,3] - rkey[0,2,2] = 0
 R842: tbox[0,2,2] + tbox[2,0,2] - tbox[3,2,3] >= 0
 R843: tbox[0,2,2] - tbox[2,0,2] + tbox[3,2,3] >= 0
 R844: - tbox[0,2,2] + tbox[2,0,2] + tbox[3,2,3] >= 0
 R845: - tbox[0,2,3] + tbox[3,2,3] + tbox[3,3,2] >= 0
 R846: tbox[0,2,3] + tbox[3,2,3] - tbox[3,3,2] >= 0
 R847: tbox[0,2,3] - tbox[3,2,3] + tbox[3,3,2] >= 0
 R848: tbox[1,1,2] + tbox[2,0,2] - tbox[2,2,3] >= 0
 R849: tbox[1,1,2] - tbox[2,0,2] + tbox[2,2,3] >= 0
 R850: - tbox[1,1,2] + tbox[2,0,2] + tbox[2,2,3] >= 0
 R851: - tbox[0,2,2] + tbox[1,2,3] = 0
 R852: unkn[0,2,2] - unkn[3,2,3] + forg[3,2,3] <= 0
 R853: unkn[2,0,2] - unkn[3,2,3] + forg[3,2,3] <= 0
 R854: unkn[0,2,2] + unkn[2,0,2] - unkn[3,2,3] + forg[3,2,3] >= 0
 R855: - unkn[0,2,3] + unkn[3,2,3] + forg[0,2,3] - forg[3,2,3] <= 0
 R856: - unkn[0,2,3] + unkn[3,3,2] + forg[0,2,3] <= 0
 R857: - unkn[0,2,3] + unkn[3,2,3] + unkn[3,3,2] + forg[0,2,3]
   - forg[3,2,3] >= 0
 R858: unkn[1,1,2] - unkn[2,2,3] + forg[2,2,3] <= 0
 R859: unkn[2,0,2] - unkn[2,2,3] + forg[2,2,3] <= 0
 R860: unkn[1,1,2] + unkn[2,0,2] - unkn[2,2,3] + forg[2,2,3] >= 0
 R861: unkn[0,2,2] - unkn[1,2,3] + forg[1,2,3] = 0
 R862: - tbox[0,3,2] + forg[0,3,2] <= 0
 R863: sbox[0,3,2] - rkey[0,3,2] + tkey[0,3,2] >= 0
 R864: sbox[0,3,2] + rkey[0,3,2] - tkey[0,3,2] >= 0
 R865: - sbox[0,3,2] + rkey[0,3,2] + tkey[0,3,2] >= 0
 R866: - tbox[1,3,2] + forg[1,3,2] <= 0
 R867: sbox[1,3,2] - rkey[1,3,2] + tkey[1,3,2] >= 0
 R868: sbox[1,3,2] + rkey[1,3,2] - tkey[1,3,2] >= 0
 R869: - sbox[1,3,2] + rkey[1,3,2] + tkey[1,3,2] >= 0
 R870: - tbox[2,3,2] + forg[2,3,2] <= 0
 R871: - sbox[2,3,2] + rkey[2,3,2] = 0
 R872: - tbox[3,3,2] + forg[3,3,2] <= 0
 R873: - sbox[3,3,2] + rkey[3,3,2] = 0
 R874: - sbox[3,3,3] + rkey[0,3,2] + rkey[2,1,2] >= 0
 R875: sbox[3,3,3] + rkey[0,3,2] - rkey[2,1,2] >= 0
 R876: sbox[3,3,3] - rkey[0,3,2] + rkey[2,1,2] >= 0
 R877: - sbox[0,3,3] + sbox[3,3,3] + rkey[3,0,2] >= 0
 R878: sbox[0,3,3] + sbox[3,3,3] - rkey[3,0,2] >= 0
 R879: sbox[0,3,3] - sbox[3,3,3] + rkey[3,0,2] >= 0
 R880: - sbox[2,3,3] + rkey[1,2,2] + rkey[2,1,2] >= 0
 R881: sbox[2,3,3] + rkey[1,2,2] - rkey[2,1,2] >= 0
 R882: sbox[2,3,3] - rkey[1,2,2] + rkey[2,1,2] >= 0
 R883: sbox[1,3,3] - rkey[0,3,2] = 0
 R884: tbox[0,3,2] + tbox[2,1,2] - tbox[3,3,3] >= 0
 R885: tbox[0,3,2] - tbox[2,1,2] + tbox[3,3,3] >= 0
 R886: - tbox[0,3,2] + tbox[2,1,2] + tbox[3,3,3] >= 0
 R887: - tbox[0,3,3] + tbox[3,0,2] + tbox[3,3,3] >= 0
 R888: tbox[0,3,3] - tbox[3,0,2] + tbox[3,3,3] >= 0
 R889: tbox[0,3,3] + tbox[3,0,2] - tbox[3,3,3] >= 0
 R890: tbox[1,2,2] + tbox[2,1,2] - tbox[2,3,3] >= 0
 R891: tbox[1,2,2] - tbox[2,1,2] + tbox[2,3,3] >= 0
 R892: - tbox[1,2,2] + tbox[2,1,2] + tbox[2,3,3] >= 0
 R893: - tbox[0,3,2] + tbox[1,3,3] = 0
 R894: unkn[0,3,2] - unkn[3,3,3] + forg[3,3,3] <= 0
 R895: unkn[2,1,2] - unkn[3,3,3] + forg[3,3,3] <= 0
 R896: unkn[0,3,2] + unkn[2,1,2] - unkn[3,3,3] + forg[3,3,3] >= 0
 R897: - unkn[0,3,3] + unkn[3,3,3] + forg[0,3,3] - forg[3,3,3] <= 0
 R898: - unkn[0,3,3] + unkn[3,0,2] + forg[0,3,3] <= 0
 R899: - unkn[0,3,3] + unkn[3,0,2] + unkn[3,3,3] + forg[0,3,3]
   - forg[3,3,3] >= 0
 R900: unkn[1,2,2] - unkn[2,3,3] + forg[2,3,3] <= 0
 R901: unkn[2,1,2] - unkn[2,3,3] + forg[2,3,3] <= 0
 R902: unkn[1,2,2] + unkn[2,1,2] - unkn[2,3,3] + forg[2,3,3] >= 0
 R903: unkn[0,3,2] - unkn[1,3,3] + forg[1,3,3] = 0
 R904: - tbox[0,0,3] + forg[0,0,3] <= 0
 R905: sbox[0,0,3] - rkey[0,0,3] + tkey[0,0,3] >= 0
 R906: sbox[0,0,3] + rkey[0,0,3] - tkey[0,0,3] >= 0
 R907: - sbox[0,0,3] + rkey[0,0,3] + tkey[0,0,3] >= 0
 R908: - tbox[1,0,3] + forg[1,0,3] <= 0
 R909: sbox[1,0,3] - rkey[1,0,3] + tkey[1,0,3] >= 0
 R910: sbox[1,0,3] + rkey[1,0,3] - tkey[1,0,3] >= 0
 R911: - sbox[1,0,3] + rkey[1,0,3] + tkey[1,0,3] >= 0
 R912: - tbox[2,0,3] + forg[2,0,3] <= 0
 R913: - sbox[2,0,3] + rkey[2,0,3] = 0
 R914: - tbox[3,0,3] + forg[3,0,3] <= 0
 R915: - sbox[3,0,3] + rkey[3,0,3] = 0
 R916: - sbox[3,0,4] + rkey[0,0,3] + rkey[2,2,3] >= 0
 R917: sbox[3,0,4] + rkey[0,0,3] - rkey[2,2,3] >= 0
 R918: sbox[3,0,4] - rkey[0,0,3] + rkey[2,2,3] >= 0
 R919: - sbox[0,0,4] + sbox[3,0,4] + rkey[3,1,3] >= 0
 R920: sbox[0,0,4] + sbox[3,0,4] - rkey[3,1,3] >= 0
 R921: sbox[0,0,4] - sbox[3,0,4] + rkey[3,1,3] >= 0
 R922: - sbox[2,0,4] + rkey[1,3,3] + rkey[2,2,3] >= 0
 R923: sbox[2,0,4] + rkey[1,3,3] - rkey[2,2,3] >= 0
 R924: sbox[2,0,4] - rkey[1,3,3] + rkey[2,2,3] >= 0
 R925: sbox[1,0,4] - rkey[0,0,3] = 0
 R926: tbox[0,0,3] + tbox[2,2,3] - tbox[3,0,4] >= 0
 R927: tbox[0,0,3] - tbox[2,2,3] + tbox[3,0,4] >= 0
 R928: - tbox[0,0,3] + tbox[2,2,3] + tbox[3,0,4] >= 0
 R929: - tbox[0,0,4] + tbox[3,0,4] + tbox[3,1,3] >= 0
 R930: tbox[0,0,4] + tbox[3,0,4] - tbox[3,1,3] >= 0
 R931: tbox[0,0,4] - tbox[3,0,4] + tbox[3,1,3] >= 0
 R932: tbox[1,3,3] - tbox[2,0,4] + tbox[2,2,3] >= 0
 R933: tbox[1,3,3] + tbox[2,0,4] - tbox[2,2,3] >= 0
 R934: - tbox[1,3,3] + tbox[2,0,4] + tbox[2,2,3] >= 0
 R935: - tbox[0,0,3] + tbox[1,0,4] = 0
 R936: unkn[0,0,3] - unkn[3,0,4] + forg[3,0,4] <= 0
 R937: unkn[2,2,3] - unkn[3,0,4] + forg[3,0,4] <= 0
 R938: unkn[0,0,3] + unkn[2,2,3] - unkn[3,0,4] + forg[3,0,4] >= 0
 R939: - unkn[0,0,4] + unkn[3,0,4] + forg[0,0,4] - forg[3,0,4] <= 0
 R940: - unkn[0,0,4] + unkn[3,1,3] + forg[0,0,4] <= 0
 R941: - unkn[0,0,4] + unkn[3,0,4] + unkn[3,1,3] + forg[0,0,4]
   - forg[3,0,4] >= 0
 R942: unkn[1,3,3] - unkn[2,0,4] + forg[2,0,4] <= 0
 R943: - unkn[2,0,4] + unkn[2,2,3] + forg[2,0,4] <= 0
 R944: unkn[1,3,3] - unkn[2,0,4] + unkn[2,2,3] + forg[2,0,4] >= 0
 R945: unkn[0,0,3] - unkn[1,0,4] + forg[1,0,4] = 0
 R946: - tbox[0,1,3] + forg[0,1,3] <= 0
 R947: sbox[0,1,3] - rkey[0,1,3] + tkey[0,1,3] >= 0
 R948: sbox[0,1,3] + rkey[0,1,3] - tkey[0,1,3] >= 0
 R949: - sbox[0,1,3] + rkey[0,1,3] + tkey[0,1,3] >= 0
 R950: - tbox[1,1,3] + forg[1,1,3] <= 0
 R951: sbox[1,1,3] - rkey[1,1,3] + tkey[1,1,3] >= 0
 R952: sbox[1,1,3] + rkey[1,1,3] - tkey[1,1,3] >= 0
 R953: - sbox[1,1,3] + rkey[1,1,3] + tkey[1,1,3] >= 0
 R954: - tbox[2,1,3] + forg[2,1,3] <= 0
 R955: - sbox[2,1,3] + rkey[2,1,3] = 0
 R956: - tbox[3,1,3] + forg[3,1,3] <= 0
 R957: - sbox[3,1,3] + rkey[3,1,3] = 0
 R958: - sbox[3,1,4] + rkey[0,1,3] + rkey[2,3,3] >= 0
 R959: sbox[3,1,4] + rkey[0,1,3] - rkey[2,3,3] >= 0
 R960: sbox[3,1,4] - rkey[0,1,3] + rkey[2,3,3] >= 0
 R961: - sbox[0,1,4] + sbox[3,1,4] + rkey[3,2,3] >= 0
 R962: sbox[0,1,4] + sbox[3,1,4] - rkey[3,2,3] >= 0
 R963: sbox[0,1,4] - sbox[3,1,4] + rkey[3,2,3] >= 0
 R964: - sbox[2,1,4] + rkey[1,0,3] + rkey[2,3,3] >= 0
 R965: sbox[2,1,4] + rkey[1,0,3] - rkey[2,3,3] >= 0
 R966: sbox[2,1,4] - rkey[1,0,3] + rkey[2,3,3] >= 0
 R967: sbox[1,1,4] - rkey[0,1,3] = 0
 R968: tbox[0,1,3] + tbox[2,3,3] - tbox[3,1,4] >= 0
 R969: tbox[0,1,3] - tbox[2,3,3] + tbox[3,1,4] >= 0
 R970: - tbox[0,1,3] + tbox[2,3,3] + tbox[3,1,4] >= 0
 R971: - tbox[0,1,4] + tbox[3,1,4] + tbox[3,2,3] >= 0
 R972: tbox[0,1,4] + tbox[3,1,4] - tbox[3,2,3] >= 0
 R973: tbox[0,1,4] - tbox[3,1,4] + tbox[3,2,3] >= 0
 R974: tbox[1,0,3] - tbox[2,1,4] + tbox[2,3,3] >= 0
 R975: tbox[1,0,3] + tbox[2,1,4] - tbox[2,3,3] >= 0
 R976: - tbox[1,0,3] + tbox[2,1,4] + tbox[2,3,3] >= 0
 R977: - tbox[0,1,3] + tbox[1,1,4] = 0
 R978: unkn[0,1,3] - unkn[3,1,4] + forg[3,1,4] <= 0
 R979: unkn[2,3,3] - unkn[3,1,4] + forg[3,1,4] <= 0
 R980: unkn[0,1,3] + unkn[2,3,3] - unkn[3,1,4] + forg[3,1,4] >= 0
 R981: - unkn[0,1,4] + unkn[3,1,4] + forg[0,1,4] - forg[3,1,4] <= 0
 R982: - unkn[0,1,4] + unkn[3,2,3] + forg[0,1,4] <= 0
 R983: - unkn[0,1,4] + unkn[3,1,4] + unkn[3,2,3] + forg[0,1,4]
   - forg[3,1,4] >= 0
 R984: unkn[1,0,3] - unkn[2,1,4] + forg[2,1,4] <= 0
 R985: - unkn[2,1,4] + unkn[2,3,3] + forg[2,1,4] <= 0
 R986: unkn[1,0,3] - unkn[2,1,4] + unkn[2,3,3] + forg[2,1,4] >= 0
 R987: unkn[0,1,3] - unkn[1,1,4] + forg[1,1,4] = 0
 R988: - tbox[0,2,3] + forg[0,2,3] <= 0
 R989: sbox[0,2,3] - rkey[0,2,3] + tkey[0,2,3] >= 0
 R990: sbox[0,2,3] + rkey[0,2,3] - tkey[0,2,3] >= 0
 R991: - sbox[0,2,3] + rkey[0,2,3] + tkey[0,2,3] >= 0
 R992: - tbox[1,2,3] + forg[1,2,3] <= 0
 R993: sbox[1,2,3] - rkey[1,2,3] + tkey[1,2,3] >= 0
 R994: sbox[1,2,3] + rkey[1,2,3] - tkey[1,2,3] >= 0
 R995: - sbox[1,2,3] + rkey[1,2,3] + tkey[1,2,3] >= 0
 R996: - tbox[2,2,3] + forg[2,2,3] <= 0
 R997: - sbox[2,2,3] + rkey[2,2,3] = 0
 R998: - tbox[3,2,3] + forg[3,2,3] <= 0
 R999: - sbox[3,2,3] + rkey[3,2,3] = 0
 R1000: - sbox[3,2,4] + rkey[0,2,3] + rkey[2,0,3] >= 0
 R1001: sbox[3,2,4] + rkey[0,2,3] - rkey[2,0,3] >= 0
 R1002: sbox[3,2,4] - rkey[0,2,3] + rkey[2,0,3] >= 0
 R1003: - sbox[0,2,4] + sbox[3,2,4] + rkey[3,3,3] >= 0
 R1004: sbox[0,2,4] + sbox[3,2,4] - rkey[3,3,3] >= 0
 R1005: sbox[0,2,4] - sbox[3,2,4] + rkey[3,3,3] >= 0
 R1006: - sbox[2,2,4] + rkey[1,1,3] + rkey[2,0,3] >= 0
 R1007: sbox[2,2,4] + rkey[1,1,3] - rkey[2,0,3] >= 0
 R1008: sbox[2,2,4] - rkey[1,1,3] + rkey[2,0,3] >= 0
 R1009: sbox[1,2,4] - rkey[0,2,3] = 0
 R1010: tbox[0,2,3] + tbox[2,0,3] - tbox[3,2,4] >= 0
 R1011: tbox[0,2,3] - tbox[2,0,3] + tbox[3,2,4] >= 0
 R1012: - tbox[0,2,3] + tbox[2,0,3] + tbox[3,2,4] >= 0
 R1013: - tbox[0,2,4] + tbox[3,2,4] + tbox[3,3,3] >= 0
 R1014: tbox[0,2,4] + tbox[3,2,4] - tbox[3,3,3] >= 0
 R1015: tbox[0,2,4] - tbox[3,2,4] + tbox[3,3,3] >= 0
 R1016: tbox[1,1,3] + tbox[2,0,3] - tbox[2,2,4] >= 0
 R1017: tbox[1,1,3] - tbox[2,0,3] + tbox[2,2,4] >= 0
 R1018: - tbox[1,1,3] + tbox[2,0,3] + tbox[2,2,4] >= 0
 R1019: - tbox[0,2,3] + tbox[1,2,4] = 0
 R1020: unkn[0,2,3] - unkn[3,2,4] + forg[3,2,4] <= 0
 R1021: unkn[2,0,3] - unkn[3,2,4] + forg[3,2,4] <= 0
 R1022: unkn[0,2,3] + unkn[2,0,3] - unkn[3,2,4] + forg[3,2,4] >= 0
 R1023: - unkn[0,2,4] + unkn[3,2,4] + forg[0,2,4] - forg[3,2,4] <= 0
 R1024: - unkn[0,2,4] + unkn[3,3,3] + forg[0,2,4] <= 0
 R1025: - unkn[0,2,4] + unkn[3,2,4] + unkn[3,3,3] + forg[0,2,4]
   - forg[3,2,4] >= 0
 R1026: unkn[1,1,3] - unkn[2,2,4] + forg[2,2,4] <= 0
 R1027: unkn[2,0,3] - unkn[2,2,4] + forg[2,2,4] <= 0
 R1028: unkn[1,1,3] + unkn[2,0,3] - unkn[2,2,4] + forg[2,2,4] >= 0
 R1029: unkn[0,2,3] - unkn[1,2,4] + forg[1,2,4] = 0
 R1030: - tbox[0,3,3] + forg[0,3,3] <= 0
 R1031: sbox[0,3,3] - rkey[0,3,3] + tkey[0,3,3] >= 0
 R1032: sbox[0,3,3] + rkey[0,3,3] - tkey[0,3,3] >= 0
 R1033: - sbox[0,3,3] + rkey[0,3,3] + tkey[0,3,3] >= 0
 R1034: - tbox[1,3,3] + forg[1,3,3] <= 0
 R1035: sbox[1,3,3] - rkey[1,3,3] + tkey[1,3,3] >= 0
 R1036: sbox[1,3,3] + rkey[1,3,3] - tkey[1,3,3] >= 0
 R1037: - sbox[1,3,3] + rkey[1,3,3] + tkey[1,3,3] >= 0
 R1038: - tbox[2,3,3] + forg[2,3,3] <= 0
 R1039: - sbox[2,3,3] + rkey[2,3,3] = 0
 R1040: - tbox[3,3,3] + forg[3,3,3] <= 0
 R1041: - sbox[3,3,3] + rkey[3,3,3] = 0
 R1042: - sbox[3,3,4] + rkey[0,3,3] + rkey[2,1,3] >= 0
 R1043: sbox[3,3,4] + rkey[0,3,3] - rkey[2,1,3] >= 0
 R1044: sbox[3,3,4] - rkey[0,3,3] + rkey[2,1,3] >= 0
 R1045: - sbox[0,3,4] + sbox[3,3,4] + rkey[3,0,3] >= 0
 R1046: sbox[0,3,4] + sbox[3,3,4] - rkey[3,0,3] >= 0
 R1047: sbox[0,3,4] - sbox[3,3,4] + rkey[3,0,3] >= 0
 R1048: - sbox[2,3,4] + rkey[1,2,3] + rkey[2,1,3] >= 0
 R1049: sbox[2,3,4] + rkey[1,2,3] - rkey[2,1,3] >= 0
 R1050: sbox[2,3,4] - rkey[1,2,3] + rkey[2,1,3] >= 0
 R1051: sbox[1,3,4] - rkey[0,3,3] = 0
 R1052: tbox[0,3,3] + tbox[2,1,3] - tbox[3,3,4] >= 0
 R1053: tbox[0,3,3] - tbox[2,1,3] + tbox[3,3,4] >= 0
 R1054: - tbox[0,3,3] + tbox[2,1,3] + tbox[3,3,4] >= 0
 R1055: - tbox[0,3,4] + tbox[3,0,3] + tbox[3,3,4] >= 0
 R1056: tbox[0,3,4] - tbox[3,0,3] + tbox[3,3,4] >= 0
 R1057: tbox[0,3,4] + tbox[3,0,3] - tbox[3,3,4] >= 0
 R1058: tbox[1,2,3] + tbox[2,1,3] - tbox[2,3,4] >= 0
 R1059: tbox[1,2,3] - tbox[2,1,3] + tbox[2,3,4] >= 0
 R1060: - tbox[1,2,3] + tbox[2,1,3] + tbox[2,3,4] >= 0
 R1061: - tbox[0,3,3] + tbox[1,3,4] = 0
 R1062: unkn[0,3,3] - unkn[3,3,4] + forg[3,3,4] <= 0
 R1063: unkn[2,1,3] - unkn[3,3,4] + forg[3,3,4] <= 0
 R1064: unkn[0,3,3] + unkn[2,1,3] - unkn[3,3,4] + forg[3,3,4] >= 0
 R1065: - unkn[0,3,4] + unkn[3,3,4] + forg[0,3,4] - forg[3,3,4] <= 0
 R1066: - unkn[0,3,4] + unkn[3,0,3] + forg[0,3,4] <= 0
 R1067: - unkn[0,3,4] + unkn[3,0,3] + unkn[3,3,4] + forg[0,3,4]
   - forg[3,3,4] >= 0
 R1068: unkn[1,2,3] - unkn[2,3,4] + forg[2,3,4] <= 0
 R1069: unkn[2,1,3] - unkn[2,3,4] + forg[2,3,4] <= 0
 R1070: unkn[1,2,3] + unkn[2,1,3] - unkn[2,3,4] + forg[2,3,4] >= 0
 R1071: unkn[0,3,3] - unkn[1,3,4] + forg[1,3,4] = 0
 R1072: - tbox[0,0,4] + forg[0,0,4] <= 0
 R1073: sbox[0,0,4] - rkey[0,0,4] + tkey[0,0,4] >= 0
 R1074: sbox[0,0,4] + rkey[0,0,4] - tkey[0,0,4] >= 0
 R1075: - sbox[0,0,4] + rkey[0,0,4] + tkey[0,0,4] >= 0
 R1076: - tbox[1,0,4] + forg[1,0,4] <= 0
 R1077: sbox[1,0,4] - rkey[1,0,4] + tkey[1,0,4] >= 0
 R1078: sbox[1,0,4] + rkey[1,0,4] - tkey[1,0,4] >= 0
 R1079: - sbox[1,0,4] + rkey[1,0,4] + tkey[1,0,4] >= 0
 R1080: - tbox[2,0,4] + forg[2,0,4] <= 0
 R1081: - sbox[2,0,4] + rkey[2,0,4] = 0
 R1082: - tbox[3,0,4] + forg[3,0,4] <= 0
 R1083: - sbox[3,0,4] + rkey[3,0,4] = 0
 R1084: - sbox[3,0,5] + rkey[0,0,4] + rkey[2,2,4] >= 0
 R1085: sbox[3,0,5] + rkey[0,0,4] - rkey[2,2,4] >= 0
 R1086: sbox[3,0,5] - rkey[0,0,4] + rkey[2,2,4] >= 0
 R1087: - sbox[0,0,5] + sbox[3,0,5] + rkey[3,1,4] >= 0
 R1088: sbox[0,0,5] + sbox[3,0,5] - rkey[3,1,4] >= 0
 R1089: sbox[0,0,5] - sbox[3,0,5] + rkey[3,1,4] >= 0
 R1090: - sbox[2,0,5] + rkey[1,3,4] + rkey[2,2,4] >= 0
 R1091: sbox[2,0,5] + rkey[1,3,4] - rkey[2,2,4] >= 0
 R1092: sbox[2,0,5] - rkey[1,3,4] + rkey[2,2,4] >= 0
 R1093: sbox[1,0,5] - rkey[0,0,4] = 0
 R1094: tbox[0,0,4] + tbox[2,2,4] - tbox[3,0,5] >= 0
 R1095: tbox[0,0,4] - tbox[2,2,4] + tbox[3,0,5] >= 0
 R1096: - tbox[0,0,4] + tbox[2,2,4] + tbox[3,0,5] >= 0
 R1097: - tbox[0,0,5] + tbox[3,0,5] + tbox[3,1,4] >= 0
 R1098: tbox[0,0,5] + tbox[3,0,5] - tbox[3,1,4] >= 0
 R1099: tbox[0,0,5] - tbox[3,0,5] + tbox[3,1,4] >= 0
 R1100: tbox[1,3,4] - tbox[2,0,5] + tbox[2,2,4] >= 0
 R1101: tbox[1,3,4] + tbox[2,0,5] - tbox[2,2,4] >= 0
 R1102: - tbox[1,3,4] + tbox[2,0,5] + tbox[2,2,4] >= 0
 R1103: - tbox[0,0,4] + tbox[1,0,5] = 0
 R1104: unkn[0,0,4] - unkn[3,0,5] + forg[3,0,5] <= 0
 R1105: unkn[2,2,4] - unkn[3,0,5] + forg[3,0,5] <= 0
 R1106: unkn[0,0,4] + unkn[2,2,4] - unkn[3,0,5] + forg[3,0,5] >= 0
 R1107: - unkn[0,0,5] + unkn[3,0,5] + forg[0,0,5] - forg[3,0,5] <= 0
 R1108: - unkn[0,0,5] + unkn[3,1,4] + forg[0,0,5] <= 0
 R1109: - unkn[0,0,5] + unkn[3,0,5] + unkn[3,1,4] + forg[0,0,5]
   - forg[3,0,5] >= 0
 R1110: unkn[1,3,4] - unkn[2,0,5] + forg[2,0,5] <= 0
 R1111: - unkn[2,0,5] + unkn[2,2,4] + forg[2,0,5] <= 0
 R1112: unkn[1,3,4] - unkn[2,0,5] + unkn[2,2,4] + forg[2,0,5] >= 0
 R1113: unkn[0,0,4] - unkn[1,0,5] + forg[1,0,5] = 0
 R1114: - tbox[0,1,4] + forg[0,1,4] <= 0
 R1115: sbox[0,1,4] - rkey[0,1,4] + tkey[0,1,4] >= 0
 R1116: sbox[0,1,4] + rkey[0,1,4] - tkey[0,1,4] >= 0
 R1117: - sbox[0,1,4] + rkey[0,1,4] + tkey[0,1,4] >= 0
 R1118: - tbox[1,1,4] + forg[1,1,4] <= 0
 R1119: sbox[1,1,4] - rkey[1,1,4] + tkey[1,1,4] >= 0
 R1120: sbox[1,1,4] + rkey[1,1,4] - tkey[1,1,4] >= 0
 R1121: - sbox[1,1,4] + rkey[1,1,4] + tkey[1,1,4] >= 0
 R1122: - tbox[2,1,4] + forg[2,1,4] <= 0
 R1123: - sbox[2,1,4] + rkey[2,1,4] = 0
 R1124: - tbox[3,1,4] + forg[3,1,4] <= 0
 R1125: - sbox[3,1,4] + rkey[3,1,4] = 0
 R1126: - sbox[3,1,5] + rkey[0,1,4] + rkey[2,3,4] >= 0
 R1127: sbox[3,1,5] + rkey[0,1,4] - rkey[2,3,4] >= 0
 R1128: sbox[3,1,5] - rkey[0,1,4] + rkey[2,3,4] >= 0
 R1129: - sbox[0,1,5] + sbox[3,1,5] + rkey[3,2,4] >= 0
 R1130: sbox[0,1,5] + sbox[3,1,5] - rkey[3,2,4] >= 0
 R1131: sbox[0,1,5] - sbox[3,1,5] + rkey[3,2,4] >= 0
 R1132: - sbox[2,1,5] + rkey[1,0,4] + rkey[2,3,4] >= 0
 R1133: sbox[2,1,5] + rkey[1,0,4] - rkey[2,3,4] >= 0
 R1134: sbox[2,1,5] - rkey[1,0,4] + rkey[2,3,4] >= 0
 R1135: sbox[1,1,5] - rkey[0,1,4] = 0
 R1136: tbox[0,1,4] + tbox[2,3,4] - tbox[3,1,5] >= 0
 R1137: tbox[0,1,4] - tbox[2,3,4] + tbox[3,1,5] >= 0
 R1138: - tbox[0,1,4] + tbox[2,3,4] + tbox[3,1,5] >= 0
 R1139: - tbox[0,1,5] + tbox[3,1,5] + tbox[3,2,4] >= 0
 R1140: tbox[0,1,5] + tbox[3,1,5] - tbox[3,2,4] >= 0
 R1141: tbox[0,1,5] - tbox[3,1,5] + tbox[3,2,4] >= 0
 R1142: tbox[1,0,4] - tbox[2,1,5] + tbox[2,3,4] >= 0
 R1143: tbox[1,0,4] + tbox[2,1,5] - tbox[2,3,4] >= 0
 R1144: - tbox[1,0,4] + tbox[2,1,5] + tbox[2,3,4] >= 0
 R1145: - tbox[0,1,4] + tbox[1,1,5] = 0
 R1146: unkn[0,1,4] - unkn[3,1,5] + forg[3,1,5] <= 0
 R1147: unkn[2,3,4] - unkn[3,1,5] + forg[3,1,5] <= 0
 R1148: unkn[0,1,4] + unkn[2,3,4] - unkn[3,1,5] + forg[3,1,5] >= 0
 R1149: - unkn[0,1,5] + unkn[3,1,5] + forg[0,1,5] - forg[3,1,5] <= 0
 R1150: - unkn[0,1,5] + unkn[3,2,4] + forg[0,1,5] <= 0
 R1151: - unkn[0,1,5] + unkn[3,1,5] + unkn[3,2,4] + forg[0,1,5]
   - forg[3,1,5] >= 0
 R1152: unkn[1,0,4] - unkn[2,1,5] + forg[2,1,5] <= 0
 R1153: - unkn[2,1,5] + unkn[2,3,4] + forg[2,1,5] <= 0
 R1154: unkn[1,0,4] - unkn[2,1,5] + unkn[2,3,4] + forg[2,1,5] >= 0
 R1155: unkn[0,1,4] - unkn[1,1,5] + forg[1,1,5] = 0
 R1156: - tbox[0,2,4] + forg[0,2,4] <= 0
 R1157: sbox[0,2,4] - rkey[0,2,4] + tkey[0,2,4] >= 0
 R1158: sbox[0,2,4] + rkey[0,2,4] - tkey[0,2,4] >= 0
 R1159: - sbox[0,2,4] + rkey[0,2,4] + tkey[0,2,4] >= 0
 R1160: - tbox[1,2,4] + forg[1,2,4] <= 0
 R1161: sbox[1,2,4] - rkey[1,2,4] + tkey[1,2,4] >= 0
 R1162: sbox[1,2,4] + rkey[1,2,4] - tkey[1,2,4] >= 0
 R1163: - sbox[1,2,4] + rkey[1,2,4] + tkey[1,2,4] >= 0
 R1164: - tbox[2,2,4] + forg[2,2,4] <= 0
 R1165: - sbox[2,2,4] + rkey[2,2,4] = 0
 R1166: - tbox[3,2,4] + forg[3,2,4] <= 0
 R1167: - sbox[3,2,4] + rkey[3,2,4] = 0
 R1168: - sbox[3,2,5] + rkey[0,2,4] + rkey[2,0,4] >= 0
 R1169: sbox[3,2,5] + rkey[0,2,4] - rkey[2,0,4] >= 0
 R1170: sbox[3,2,5] - rkey[0,2,4] + rkey[2,0,4] >= 0
 R1171: - sbox[0,2,5] + sbox[3,2,5] + rkey[3,3,4] >= 0
 R1172: sbox[0,2,5] + sbox[3,2,5] - rkey[3,3,4] >= 0
 R1173: sbox[0,2,5] - sbox[3,2,5] + rkey[3,3,4] >= 0
 R1174: - sbox[2,2,5] + rkey[1,1,4] + rkey[2,0,4] >= 0
 R1175: sbox[2,2,5] + rkey[1,1,4] - rkey[2,0,4] >= 0
 R1176: sbox[2,2,5] - rkey[1,1,4] + rkey[2,0,4] >= 0
 R1177: sbox[1,2,5] - rkey[0,2,4] = 0
 R1178: tbox[0,2,4] + tbox[2,0,4] - tbox[3,2,5] >= 0
 R1179: tbox[0,2,4] - tbox[2,0,4] + tbox[3,2,5] >= 0
 R1180: - tbox[0,2,4] + tbox[2,0,4] + tbox[3,2,5] >= 0
 R1181: - tbox[0,2,5] + tbox[3,2,5] + tbox[3,3,4] >= 0
 R1182: tbox[0,2,5] + tbox[3,2,5] - tbox[3,3,4] >= 0
 R1183: tbox[0,2,5] - tbox[3,2,5] + tbox[3,3,4] >= 0
 R1184: tbox[1,1,4] + tbox[2,0,4] - tbox[2,2,5] >= 0
 R1185: tbox[1,1,4] - tbox[2,0,4] + tbox[2,2,5] >= 0
 R1186: - tbox[1,1,4] + tbox[2,0,4] + tbox[2,2,5] >= 0
 R1187: - tbox[0,2,4] + tbox[1,2,5] = 0
 R1188: unkn[0,2,4] - unkn[3,2,5] + forg[3,2,5] <= 0
 R1189: unkn[2,0,4] - unkn[3,2,5] + forg[3,2,5] <= 0
 R1190: unkn[0,2,4] + unkn[2,0,4] - unkn[3,2,5] + forg[3,2,5] >= 0
 R1191: - unkn[0,2,5] + unkn[3,2,5] + forg[0,2,5] - forg[3,2,5] <= 0
 R1192: - unkn[0,2,5] + unkn[3,3,4] + forg[0,2,5] <= 0
 R1193: - unkn[0,2,5] + unkn[3,2,5] + unkn[3,3,4] + forg[0,2,5]
   - forg[3,2,5] >= 0
 R1194: unkn[1,1,4] - unkn[2,2,5] + forg[2,2,5] <= 0
 R1195: unkn[2,0,4] - unkn[2,2,5] + forg[2,2,5] <= 0
 R1196: unkn[1,1,4] + unkn[2,0,4] - unkn[2,2,5] + forg[2,2,5] >= 0
 R1197: unkn[0,2,4] - unkn[1,2,5] + forg[1,2,5] = 0
 R1198: - tbox[0,3,4] + forg[0,3,4] <= 0
 R1199: sbox[0,3,4] - rkey[0,3,4] + tkey[0,3,4] >= 0
 R1200: sbox[0,3,4] + rkey[0,3,4] - tkey[0,3,4] >= 0
 R1201: - sbox[0,3,4] + rkey[0,3,4] + tkey[0,3,4] >= 0
 R1202: - tbox[1,3,4] + forg[1,3,4] <= 0
 R1203: sbox[1,3,4] - rkey[1,3,4] + tkey[1,3,4] >= 0
 R1204: sbox[1,3,4] + rkey[1,3,4] - tkey[1,3,4] >= 0
 R1205: - sbox[1,3,4] + rkey[1,3,4] + tkey[1,3,4] >= 0
 R1206: - tbox[2,3,4] + forg[2,3,4] <= 0
 R1207: - sbox[2,3,4] + rkey[2,3,4] = 0
 R1208: - tbox[3,3,4] + forg[3,3,4] <= 0
 R1209: - sbox[3,3,4] + rkey[3,3,4] = 0
 R1210: - sbox[3,3,5] + rkey[0,3,4] + rkey[2,1,4] >= 0
 R1211: sbox[3,3,5] + rkey[0,3,4] - rkey[2,1,4] >= 0
 R1212: sbox[3,3,5] - rkey[0,3,4] + rkey[2,1,4] >= 0
 R1213: - sbox[0,3,5] + sbox[3,3,5] + rkey[3,0,4] >= 0
 R1214: sbox[0,3,5] + sbox[3,3,5] - rkey[3,0,4] >= 0
 R1215: sbox[0,3,5] - sbox[3,3,5] + rkey[3,0,4] >= 0
 R1216: - sbox[2,3,5] + rkey[1,2,4] + rkey[2,1,4] >= 0
 R1217: sbox[2,3,5] + rkey[1,2,4] - rkey[2,1,4] >= 0
 R1218: sbox[2,3,5] - rkey[1,2,4] + rkey[2,1,4] >= 0
 R1219: sbox[1,3,5] - rkey[0,3,4] = 0
 R1220: tbox[0,3,4] + tbox[2,1,4] - tbox[3,3,5] >= 0
 R1221: tbox[0,3,4] - tbox[2,1,4] + tbox[3,3,5] >= 0
 R1222: - tbox[0,3,4] + tbox[2,1,4] + tbox[3,3,5] >= 0
 R1223: - tbox[0,3,5] + tbox[3,0,4] + tbox[3,3,5] >= 0
 R1224: tbox[0,3,5] - tbox[3,0,4] + tbox[3,3,5] >= 0
 R1225: tbox[0,3,5] + tbox[3,0,4] - tbox[3,3,5] >= 0
 R1226: tbox[1,2,4] + tbox[2,1,4] - tbox[2,3,5] >= 0
 R1227: tbox[1,2,4] - tbox[2,1,4] + tbox[2,3,5] >= 0
 R1228: - tbox[1,2,4] + tbox[2,1,4] + tbox[2,3,5] >= 0
 R1229: - tbox[0,3,4] + tbox[1,3,5] = 0
 R1230: unkn[0,3,4] - unkn[3,3,5] + forg[3,3,5] <= 0
 R1231: unkn[2,1,4] - unkn[3,3,5] + forg[3,3,5] <= 0
 R1232: unkn[0,3,4] + unkn[2,1,4] - unkn[3,3,5] + forg[3,3,5] >= 0
 R1233: - unkn[0,3,5] + unkn[3,3,5] + forg[0,3,5] - forg[3,3,5] <= 0
 R1234: - unkn[0,3,5] + unkn[3,0,4] + forg[0,3,5] <= 0
 R1235: - unkn[0,3,5] + unkn[3,0,4] + unkn[3,3,5] + forg[0,3,5]
   - forg[3,3,5] >= 0
 R1236: unkn[1,2,4] - unkn[2,3,5] + forg[2,3,5] <= 0
 R1237: unkn[2,1,4] - unkn[2,3,5] + forg[2,3,5] <= 0
 R1238: unkn[1,2,4] + unkn[2,1,4] - unkn[2,3,5] + forg[2,3,5] >= 0
 R1239: unkn[0,3,4] - unkn[1,3,5] + forg[1,3,5] = 0
 R1240: - tbox[0,0,5] + forg[0,0,5] <= 0
 R1241: sbox[0,0,5] - rkey[0,0,5] + tkey[0,0,5] >= 0
 R1242: sbox[0,0,5] + rkey[0,0,5] - tkey[0,0,5] >= 0
 R1243: - sbox[0,0,5] + rkey[0,0,5] + tkey[0,0,5] >= 0
 R1244: - tbox[1,0,5] + forg[1,0,5] <= 0
 R1245: sbox[1,0,5] - rkey[1,0,5] + tkey[1,0,5] >= 0
 R1246: sbox[1,0,5] + rkey[1,0,5] - tkey[1,0,5] >= 0
 R1247: - sbox[1,0,5] + rkey[1,0,5] + tkey[1,0,5] >= 0
 R1248: - tbox[2,0,5] + forg[2,0,5] <= 0
 R1249: - sbox[2,0,5] + rkey[2,0,5] = 0
 R1250: - tbox[3,0,5] + forg[3,0,5] <= 0
 R1251: - sbox[3,0,5] + rkey[3,0,5] = 0
 R1252: - sbox[3,0,6] + rkey[0,0,5] + rkey[2,2,5] >= 0
 R1253: sbox[3,0,6] + rkey[0,0,5] - rkey[2,2,5] >= 0
 R1254: sbox[3,0,6] - rkey[0,0,5] + rkey[2,2,5] >= 0
 R1255: - sbox[0,0,6] + sbox[3,0,6] + rkey[3,1,5] >= 0
 R1256: sbox[0,0,6] + sbox[3,0,6] - rkey[3,1,5] >= 0
 R1257: sbox[0,0,6] - sbox[3,0,6] + rkey[3,1,5] >= 0
 R1258: - sbox[2,0,6] + rkey[1,3,5] + rkey[2,2,5] >= 0
 R1259: sbox[2,0,6] + rkey[1,3,5] - rkey[2,2,5] >= 0
 R1260: sbox[2,0,6] - rkey[1,3,5] + rkey[2,2,5] >= 0
 R1261: sbox[1,0,6] - rkey[0,0,5] = 0
 R1262: tbox[0,0,5] + tbox[2,2,5] - tbox[3,0,6] >= 0
 R1263: tbox[0,0,5] - tbox[2,2,5] + tbox[3,0,6] >= 0
 R1264: - tbox[0,0,5] + tbox[2,2,5] + tbox[3,0,6] >= 0
 R1265: - tbox[0,0,6] + tbox[3,0,6] + tbox[3,1,5] >= 0
 R1266: tbox[0,0,6] + tbox[3,0,6] - tbox[3,1,5] >= 0
 R1267: tbox[0,0,6] - tbox[3,0,6] + tbox[3,1,5] >= 0
 R1268: tbox[1,3,5] - tbox[2,0,6] + tbox[2,2,5] >= 0
 R1269: tbox[1,3,5] + tbox[2,0,6] - tbox[2,2,5] >= 0
 R1270: - tbox[1,3,5] + tbox[2,0,6] + tbox[2,2,5] >= 0
 R1271: - tbox[0,0,5] + tbox[1,0,6] = 0
 R1272: unkn[0,0,5] - unkn[3,0,6] + forg[3,0,6] <= 0
 R1273: unkn[2,2,5] - unkn[3,0,6] + forg[3,0,6] <= 0
 R1274: unkn[0,0,5] + unkn[2,2,5] - unkn[3,0,6] + forg[3,0,6] >= 0
 R1275: - unkn[0,0,6] + unkn[3,0,6] + forg[0,0,6] - forg[3,0,6] <= 0
 R1276: - unkn[0,0,6] + unkn[3,1,5] + forg[0,0,6] <= 0
 R1277: - unkn[0,0,6] + unkn[3,0,6] + unkn[3,1,5] + forg[0,0,6]
   - forg[3,0,6] >= 0
 R1278: unkn[1,3,5] - unkn[2,0,6] + forg[2,0,6] <= 0
 R1279: - unkn[2,0,6] + unkn[2,2,5] + forg[2,0,6] <= 0
 R1280: unkn[1,3,5] - unkn[2,0,6] + unkn[2,2,5] + forg[2,0,6] >= 0
 R1281: unkn[0,0,5] - unkn[1,0,6] + forg[1,0,6] = 0
 R1282: - tbox[0,1,5] + forg[0,1,5] <= 0
 R1283: sbox[0,1,5] - rkey[0,1,5] + tkey[0,1,5] >= 0
 R1284: sbox[0,1,5] + rkey[0,1,5] - tkey[0,1,5] >= 0
 R1285: - sbox[0,1,5] + rkey[0,1,5] + tkey[0,1,5] >= 0
 R1286: - tbox[1,1,5] + forg[1,1,5] <= 0
 R1287: sbox[1,1,5] - rkey[1,1,5] + tkey[1,1,5] >= 0
 R1288: sbox[1,1,5] + rkey[1,1,5] - tkey[1,1,5] >= 0
 R1289: - sbox[1,1,5] + rkey[1,1,5] + tkey[1,1,5] >= 0
 R1290: - tbox[2,1,5] + forg[2,1,5] <= 0
 R1291: - sbox[2,1,5] + rkey[2,1,5] = 0
 R1292: - tbox[3,1,5] + forg[3,1,5] <= 0
 R1293: - sbox[3,1,5] + rkey[3,1,5] = 0
 R1294: - sbox[3,1,6] + rkey[0,1,5] + rkey[2,3,5] >= 0
 R1295: sbox[3,1,6] + rkey[0,1,5] - rkey[2,3,5] >= 0
 R1296: sbox[3,1,6] - rkey[0,1,5] + rkey[2,3,5] >= 0
 R1297: - sbox[0,1,6] + sbox[3,1,6] + rkey[3,2,5] >= 0
 R1298: sbox[0,1,6] + sbox[3,1,6] - rkey[3,2,5] >= 0
 R1299: sbox[0,1,6] - sbox[3,1,6] + rkey[3,2,5] >= 0
 R1300: - sbox[2,1,6] + rkey[1,0,5] + rkey[2,3,5] >= 0
 R1301: sbox[2,1,6] + rkey[1,0,5] - rkey[2,3,5] >= 0
 R1302: sbox[2,1,6] - rkey[1,0,5] + rkey[2,3,5] >= 0
 R1303: sbox[1,1,6] - rkey[0,1,5] = 0
 R1304: tbox[0,1,5] + tbox[2,3,5] - tbox[3,1,6] >= 0
 R1305: tbox[0,1,5] - tbox[2,3,5] + tbox[3,1,6] >= 0
 R1306: - tbox[0,1,5] + tbox[2,3,5] + tbox[3,1,6] >= 0
 R1307: - tbox[0,1,6] + tbox[3,1,6] + tbox[3,2,5] >= 0
 R1308: tbox[0,1,6] + tbox[3,1,6] - tbox[3,2,5] >= 0
 R1309: tbox[0,1,6] - tbox[3,1,6] + tbox[3,2,5] >= 0
 R1310: tbox[1,0,5] - tbox[2,1,6] + tbox[2,3,5] >= 0
 R1311: tbox[1,0,5] + tbox[2,1,6] - tbox[2,3,5] >= 0
 R1312: - tbox[1,0,5] + tbox[2,1,6] + tbox[2,3,5] >= 0
 R1313: - tbox[0,1,5] + tbox[1,1,6] = 0
 R1314: unkn[0,1,5] - unkn[3,1,6] + forg[3,1,6] <= 0
 R1315: unkn[2,3,5] - unkn[3,1,6] + forg[3,1,6] <= 0
 R1316: unkn[0,1,5] + unkn[2,3,5] - unkn[3,1,6] + forg[3,1,6] >= 0
 R1317: - unkn[0,1,6] + unkn[3,1,6] + forg[0,1,6] - forg[3,1,6] <= 0
 R1318: - unkn[0,1,6] + unkn[3,2,5] + forg[0,1,6] <= 0
 R1319: - unkn[0,1,6] + unkn[3,1,6] + unkn[3,2,5] + forg[0,1,6]
   - forg[3,1,6] >= 0
 R1320: unkn[1,0,5] - unkn[2,1,6] + forg[2,1,6] <= 0
 R1321: - unkn[2,1,6] + unkn[2,3,5] + forg[2,1,6] <= 0
 R1322: unkn[1,0,5] - unkn[2,1,6] + unkn[2,3,5] + forg[2,1,6] >= 0
 R1323: unkn[0,1,5] - unkn[1,1,6] + forg[1,1,6] = 0
 R1324: - tbox[0,2,5] + forg[0,2,5] <= 0
 R1325: sbox[0,2,5] - rkey[0,2,5] + tkey[0,2,5] >= 0
 R1326: sbox[0,2,5] + rkey[0,2,5] - tkey[0,2,5] >= 0
 R1327: - sbox[0,2,5] + rkey[0,2,5] + tkey[0,2,5] >= 0
 R1328: - tbox[1,2,5] + forg[1,2,5] <= 0
 R1329: sbox[1,2,5] - rkey[1,2,5] + tkey[1,2,5] >= 0
 R1330: sbox[1,2,5] + rkey[1,2,5] - tkey[1,2,5] >= 0
 R1331: - sbox[1,2,5] + rkey[1,2,5] + tkey[1,2,5] >= 0
 R1332: - tbox[2,2,5] + forg[2,2,5] <= 0
 R1333: - sbox[2,2,5] + rkey[2,2,5] = 0
 R1334: - tbox[3,2,5] + forg[3,2,5] <= 0
 R1335: - sbox[3,2,5] + rkey[3,2,5] = 0
 R1336: - sbox[3,2,6] + rkey[0,2,5] + rkey[2,0,5] >= 0
 R1337: sbox[3,2,6] + rkey[0,2,5] - rkey[2,0,5] >= 0
 R1338: sbox[3,2,6] - rkey[0,2,5] + rkey[2,0,5] >= 0
 R1339: - sbox[0,2,6] + sbox[3,2,6] + rkey[3,3,5] >= 0
 R1340: sbox[0,2,6] + sbox[3,2,6] - rkey[3,3,5] >= 0
 R1341: sbox[0,2,6] - sbox[3,2,6] + rkey[3,3,5] >= 0
 R1342: - sbox[2,2,6] + rkey[1,1,5] + rkey[2,0,5] >= 0
 R1343: sbox[2,2,6] + rkey[1,1,5] - rkey[2,0,5] >= 0
 R1344: sbox[2,2,6] - rkey[1,1,5] + rkey[2,0,5] >= 0
 R1345: sbox[1,2,6] - rkey[0,2,5] = 0
 R1346: tbox[0,2,5] + tbox[2,0,5] - tbox[3,2,6] >= 0
 R1347: tbox[0,2,5] - tbox[2,0,5] + tbox[3,2,6] >= 0
 R1348: - tbox[0,2,5] + tbox[2,0,5] + tbox[3,2,6] >= 0
 R1349: - tbox[0,2,6] + tbox[3,2,6] + tbox[3,3,5] >= 0
 R1350: tbox[0,2,6] + tbox[3,2,6] - tbox[3,3,5] >= 0
 R1351: tbox[0,2,6] - tbox[3,2,6] + tbox[3,3,5] >= 0
 R1352: tbox[1,1,5] + tbox[2,0,5] - tbox[2,2,6] >= 0
 R1353: tbox[1,1,5] - tbox[2,0,5] + tbox[2,2,6] >= 0
 R1354: - tbox[1,1,5] + tbox[2,0,5] + tbox[2,2,6] >= 0
 R1355: - tbox[0,2,5] + tbox[1,2,6] = 0
 R1356: unkn[0,2,5] - unkn[3,2,6] + forg[3,2,6] <= 0
 R1357: unkn[2,0,5] - unkn[3,2,6] + forg[3,2,6] <= 0
 R1358: unkn[0,2,5] + unkn[2,0,5] - unkn[3,2,6] + forg[3,2,6] >= 0
 R1359: - unkn[0,2,6] + unkn[3,2,6] + forg[0,2,6] - forg[3,2,6] <= 0
 R1360: - unkn[0,2,6] + unkn[3,3,5] + forg[0,2,6] <= 0
 R1361: - unkn[0,2,6] + unkn[3,2,6] + unkn[3,3,5] + forg[0,2,6]
   - forg[3,2,6] >= 0
 R1362: unkn[1,1,5] - unkn[2,2,6] + forg[2,2,6] <= 0
 R1363: unkn[2,0,5] - unkn[2,2,6] + forg[2,2,6] <= 0
 R1364: unkn[1,1,5] + unkn[2,0,5] - unkn[2,2,6] + forg[2,2,6] >= 0
 R1365: unkn[0,2,5] - unkn[1,2,6] + forg[1,2,6] = 0
 R1366: - tbox[0,3,5] + forg[0,3,5] <= 0
 R1367: sbox[0,3,5] - rkey[0,3,5] + tkey[0,3,5] >= 0
 R1368: sbox[0,3,5] + rkey[0,3,5] - tkey[0,3,5] >= 0
 R1369: - sbox[0,3,5] + rkey[0,3,5] + tkey[0,3,5] >= 0
 R1370: - tbox[1,3,5] + forg[1,3,5] <= 0
 R1371: sbox[1,3,5] - rkey[1,3,5] + tkey[1,3,5] >= 0
 R1372: sbox[1,3,5] + rkey[1,3,5] - tkey[1,3,5] >= 0
 R1373: - sbox[1,3,5] + rkey[1,3,5] + tkey[1,3,5] >= 0
 R1374: - tbox[2,3,5] + forg[2,3,5] <= 0
 R1375: - sbox[2,3,5] + rkey[2,3,5] = 0
 R1376: - tbox[3,3,5] + forg[3,3,5] <= 0
 R1377: - sbox[3,3,5] + rkey[3,3,5] = 0
 R1378: - sbox[3,3,6] + rkey[0,3,5] + rkey[2,1,5] >= 0
 R1379: sbox[3,3,6] + rkey[0,3,5] - rkey[2,1,5] >= 0
 R1380: sbox[3,3,6] - rkey[0,3,5] + rkey[2,1,5] >= 0
 R1381: - sbox[0,3,6] + sbox[3,3,6] + rkey[3,0,5] >= 0
 R1382: sbox[0,3,6] + sbox[3,3,6] - rkey[3,0,5] >= 0
 R1383: sbox[0,3,6] - sbox[3,3,6] + rkey[3,0,5] >= 0
 R1384: - sbox[2,3,6] + rkey[1,2,5] + rkey[2,1,5] >= 0
 R1385: sbox[2,3,6] + rkey[1,2,5] - rkey[2,1,5] >= 0
 R1386: sbox[2,3,6] - rkey[1,2,5] + rkey[2,1,5] >= 0
 R1387: sbox[1,3,6] - rkey[0,3,5] = 0
 R1388: tbox[0,3,5] + tbox[2,1,5] - tbox[3,3,6] >= 0
 R1389: tbox[0,3,5] - tbox[2,1,5] + tbox[3,3,6] >= 0
 R1390: - tbox[0,3,5] + tbox[2,1,5] + tbox[3,3,6] >= 0
 R1391: - tbox[0,3,6] + tbox[3,0,5] + tbox[3,3,6] >= 0
 R1392: tbox[0,3,6] - tbox[3,0,5] + tbox[3,3,6] >= 0
 R1393: tbox[0,3,6] + tbox[3,0,5] - tbox[3,3,6] >= 0
 R1394: tbox[1,2,5] + tbox[2,1,5] - tbox[2,3,6] >= 0
 R1395: tbox[1,2,5] - tbox[2,1,5] + tbox[2,3,6] >= 0
 R1396: - tbox[1,2,5] + tbox[2,1,5] + tbox[2,3,6] >= 0
 R1397: - tbox[0,3,5] + tbox[1,3,6] = 0
 R1398: unkn[0,3,5] - unkn[3,3,6] + forg[3,3,6] <= 0
 R1399: unkn[2,1,5] - unkn[3,3,6] + forg[3,3,6] <= 0
 R1400: unkn[0,3,5] + unkn[2,1,5] - unkn[3,3,6] + forg[3,3,6] >= 0
 R1401: - unkn[0,3,6] + unkn[3,3,6] + forg[0,3,6] - forg[3,3,6] <= 0
 R1402: - unkn[0,3,6] + unkn[3,0,5] + forg[0,3,6] <= 0
 R1403: - unkn[0,3,6] + unkn[3,0,5] + unkn[3,3,6] + forg[0,3,6]
   - forg[3,3,6] >= 0
 R1404: unkn[1,2,5] - unkn[2,3,6] + forg[2,3,6] <= 0
 R1405: unkn[2,1,5] - unkn[2,3,6] + forg[2,3,6] <= 0
 R1406: unkn[1,2,5] + unkn[2,1,5] - unkn[2,3,6] + forg[2,3,6] >= 0
 R1407: unkn[0,3,5] - unkn[1,3,6] + forg[1,3,6] = 0
 R1408: sbox[0,0,0] + sbox[0,1,0] + sbox[0,2,0] + sbox[0,3,0] + sbox[1,0,0]
   + sbox[1,1,0] + sbox[1,2,0] + sbox[1,3,0] + sbox[2,0,0] + sbox[2,1,0]
   + sbox[2,2,0] + sbox[2,3,0] + sbox[3,0,0] + sbox[3,1,0] + sbox[3,2,0]
   + sbox[3,3,0] + tkey[0,0,0] + tkey[0,1,0] + tkey[0,2,0] + tkey[0,3,0]
   + tkey[1,0,0] + tkey[1,1,0] + tkey[1,2,0] + tkey[1,3,0] + tkey[2,0,0]
   + tkey[2,1,0] + tkey[2,2,0] + tkey[2,3,0] + tkey[3,0,0] + tkey[3,1,0]
   + tkey[3,2,0] + tkey[3,3,0] >= 1
 R1409: sbox[0,0,1] + sbox[0,1,1] + sbox[0,2,1] + sbox[0,3,1] + sbox[1,0,1]
   + sbox[1,1,1] + sbox[1,2,1] + sbox[1,3,1] + sbox[2,0,1] + sbox[2,1,1]
   + sbox[2,2,1] + sbox[2,3,1] + sbox[3,0,1] + sbox[3,1,1] + sbox[3,2,1]
   + sbox[3,3,1] + tkey[0,0,1] + tkey[0,1,1] + tkey[0,2,1] + tkey[0,3,1]
   + tkey[1,0,1] + tkey[1,1,1] + tkey[1,2,1] + tkey[1,3,1] + tkey[2,0,1]
   + tkey[2,1,1] + tkey[2,2,1] + tkey[2,3,1] + tkey[3,0,1] + tkey[3,1,1]
   + tkey[3,2,1] + tkey[3,3,1] >= 1
 R1410: sbox[0,0,2] + sbox[0,1,2] + sbox[0,2,2] + sbox[0,3,2] + sbox[1,0,2]
   + sbox[1,1,2] + sbox[1,2,2] + sbox[1,3,2] + sbox[2,0,2] + sbox[2,1,2]
   + sbox[2,2,2] + sbox[2,3,2] + sbox[3,0,2] + sbox[3,1,2] + sbox[3,2,2]
   + sbox[3,3,2] + tkey[0,0,2] + tkey[0,1,2] + tkey[0,2,2] + tkey[0,3,2]
   + tkey[1,0,2] + tkey[1,1,2] + tkey[1,2,2] + tkey[1,3,2] + tkey[2,0,2]
   + tkey[2,1,2] + tkey[2,2,2] + tkey[2,3,2] + tkey[3,0,2] + tkey[3,1,2]
   + tkey[3,2,2] + tkey[3,3,2] >= 1
 R1411: sbox[0,0,3] + sbox[0,1,3] + sbox[0,2,3] + sbox[0,3,3] + sbox[1,0,3]
   + sbox[1,1,3] + sbox[1,2,3] + sbox[1,3,3] + sbox[2,0,3] + sbox[2,1,3]
   + sbox[2,2,3] + sbox[2,3,3] + sbox[3,0,3] + sbox[3,1,3] + sbox[3,2,3]
   + sbox[3,3,3] + tkey[0,0,3] + tkey[0,1,3] + tkey[0,2,3] + tkey[0,3,3]
   + tkey[1,0,3] + tkey[1,1,3] + tkey[1,2,3] + tkey[1,3,3] + tkey[2,0,3]
   + tkey[2,1,3] + tkey[2,2,3] + tkey[2,3,3] + tkey[3,0,3] + tkey[3,1,3]
   + tkey[3,2,3] + tkey[3,3,3] >= 1
 R1412: sbox[0,0,4] + sbox[0,1,4] + sbox[0,2,4] + sbox[0,3,4] + sbox[1,0,4]
   + sbox[1,1,4] + sbox[1,2,4] + sbox[1,3,4] + sbox[2,0,4] + sbox[2,1,4]
   + sbox[2,2,4] + sbox[2,3,4] + sbox[3,0,4] + sbox[3,1,4] + sbox[3,2,4]
   + sbox[3,3,4] + tkey[0,0,4] + tkey[0,1,4] + tkey[0,2,4] + tkey[0,3,4]
   + tkey[1,0,4] + tkey[1,1,4] + tkey[1,2,4] + tkey[1,3,4] + tkey[2,0,4]
   + tkey[2,1,4] + tkey[2,2,4] + tkey[2,3,4] + tkey[3,0,4] + tkey[3,1,4]
   + tkey[3,2,4] + tkey[3,3,4] >= 1
 R1413: sbox[0,0,5] + sbox[0,1,5] + sbox[0,2,5] + sbox[0,3,5] + sbox[1,0,5]
   + sbox[1,1,5] + sbox[1,2,5] + sbox[1,3,5] + sbox[2,0,5] + sbox[2,1,5]
   + sbox[2,2,5] + sbox[2,3,5] + sbox[3,0,5] + sbox[3,1,5] + sbox[3,2,5]
   + sbox[3,3,5] + tkey[0,0,5] + tkey[0,1,5] + tkey[0,2,5] + tkey[0,3,5]
   + tkey[1,0,5] + tkey[1,1,5] + tkey[1,2,5] + tkey[1,3,5] + tkey[2,0,5]
   + tkey[2,1,5] + tkey[2,2,5] + tkey[2,3,5] + tkey[3,0,5] + tkey[3,1,5]
   + tkey[3,2,5] + tkey[3,3,5] >= 1
 R1414: sbox[0,0,0] + sbox[0,1,0] + sbox[0,2,0] + sbox[0,3,0] + sbox[1,0,0]
   + sbox[1,1,0] + sbox[1,2,0] + sbox[1,3,0] + sbox[2,0,0] + sbox[2,1,0]
   + sbox[2,2,0] + sbox[2,3,0] + sbox[3,0,0] + sbox[3,1,0] + sbox[3,2,0]
   + sbox[3,3,0] = 0
 R1415: sbox[0,0,6] + sbox[0,1,6] + sbox[0,2,6] + sbox[0,3,6] + sbox[1,0,6]
   + sbox[1,1,6] + sbox[1,2,6] + sbox[1,3,6] + sbox[2,0,6] + sbox[2,1,6]
   + sbox[2,2,6] + sbox[2,3,6] + sbox[3,0,6] + sbox[3,1,6] + sbox[3,2,6]
   + sbox[3,3,6] = 0
 R1416: tbox[0,0,0] = 1
 R1417: unkn[0,0,0] - forg[0,0,0] = 0
 R1418: tbox[0,1,0] = 0
 R1419: unkn[0,1,0] - forg[0,1,0] = 0
 R1420: tbox[0,2,0] = 0
 R1421: unkn[0,2,0] - forg[0,2,0] = 0
 R1422: tbox[0,3,0] = 0
 R1423: unkn[0,3,0] - forg[0,3,0] = 0
 R1424: tbox[1,0,0] = 0
 R1425: unkn[1,0,0] - forg[1,0,0] = 0
 R1426: tbox[1,1,0] = 0
 R1427: unkn[1,1,0] - forg[1,1,0] = 0
 R1428: tbox[1,2,0] = 0
 R1429: unkn[1,2,0] - forg[1,2,0] = 0
 R1430: tbox[1,3,0] = 0
 R1431: unkn[1,3,0] - forg[1,3,0] = 0
 R1432: tbox[2,0,0] = 0
 R1433: unkn[2,0,0] - forg[2,0,0] = 0
 R1434: tbox[2,1,0] = 0
 R1435: unkn[2,1,0] - forg[2,1,0] = 0
 R1436: tbox[2,2,0] = 0
 R1437: unkn[2,2,0] - forg[2,2,0] = 0
 R1438: tbox[2,3,0] = 0
 R1439: unkn[2,3,0] - forg[2,3,0] = 0
 R1440: tbox[3,0,0] = 0
 R1441: unkn[3,0,0] - forg[3,0,0] = 0
 R1442: tbox[3,1,0] = 0
 R1443: unkn[3,1,0] - forg[3,1,0] = 0
 R1444: tbox[3,2,0] = 0
 R1445: unkn[3,2,0] - forg[3,2,0] = 0
 R1446: tbox[3,3,0] = 0
 R1447: unkn[3,3,0] - forg[3,3,0] = 0
Bounds
 cost[0,0,0] <= 2
 cost[0,0,1] <= 2
 cost[0,0,2] <= 2
 cost[0,0,3] <= 2
 cost[0,0,4] <= 2
 cost[0,0,5] <= 2
 cost[0,0,6] <= 2
 cost[0,1,0] <= 2
 cost[0,1,1] <= 2
 cost[0,1,2] <= 2
 cost[0,1,3] <= 2
 cost[0,1,4] <= 2
 cost[0,1,5] <= 2
 cost[0,1,6] <= 2
 cost[0,2,0] <= 2
 cost[0,2,1] <= 2
 cost[0,2,2] <= 2
 cost[0,2,3] <= 2
 cost[0,2,4] <= 2
 cost[0,2,5] <= 2
 cost[0,2,6] <= 2
 cost[0,3,0] <= 2
 cost[0,3,1] <= 2
 cost[0,3,2] <= 2
 cost[0,3,3] <= 2
 cost[0,3,4] <= 2
 cost[0,3,5] <= 2
 cost[0,3,6] <= 2
 cost[1,0,0] <= 2
 cost[1,0,1] <= 2
 cost[1,0,2] <= 2
 cost[1,0,3] <= 2
 cost[1,0,4] <= 2
 cost[1,0,5] <= 2
 cost[1,0,6] <= 2
 cost[1,1,0] <= 2
 cost[1,1,1] <= 2
 cost[1,1,2] <= 2
 cost[1,1,3] <= 2
 cost[1,1,4] <= 2
 cost[1,1,5] <= 2
 cost[1,1,6] <= 2
 cost[1,2,0] <= 2
 cost[1,2,1] <= 2
 cost[1,2,2] <= 2
 cost[1,2,3] <= 2
 cost[1,2,4] <= 2
 cost[1,2,5] <= 2
 cost[1,2,6] <= 2
 cost[1,3,0] <= 2
 cost[1,3,1] <= 2
 cost[1,3,2] <= 2
 cost[1,3,3] <= 2
 cost[1,3,4] <= 2
 cost[1,3,5] <= 2
 cost[1,3,6] <= 2
 cost[2,0,0] <= 2
 cost[2,0,1] <= 2
 cost[2,0,2] <= 2
 cost[2,0,3] <= 2
 cost[2,0,4] <= 2
 cost[2,0,5] <= 2
 cost[2,0,6] <= 2
 cost[2,1,0] <= 2
 cost[2,1,1] <= 2
 cost[2,1,2] <= 2
 cost[2,1,3] <= 2
 cost[2,1,4] <= 2
 cost[2,1,5] <= 2
 cost[2,1,6] <= 2
 cost[2,2,0] <= 2
 cost[2,2,1] <= 2
 cost[2,2,2] <= 2
 cost[2,2,3] <= 2
 cost[2,2,4] <= 2
 cost[2,2,5] <= 2
 cost[2,2,6] <= 2
 cost[2,3,0] <= 2
 cost[2,3,1] <= 2
 cost[2,3,2] <= 2
 cost[2,3,3] <= 2
 cost[2,3,4] <= 2
 cost[2,3,5] <= 2
 cost[2,3,6] <= 2
 cost[3,0,0] <= 2
 cost[3,0,1] <= 2
 cost[3,0,2] <= 2
 cost[3,0,3] <= 2
 cost[3,0,4] <= 2
 cost[3,0,5] <= 2
 cost[3,0,6] <= 2
 cost[3,1,0] <= 2
 cost[3,1,1] <= 2
 cost[3,1,2] <= 2
 cost[3,1,3] <= 2
 cost[3,1,4] <= 2
 cost[3,1,5] <= 2
 cost[3,1,6] <= 2
 cost[3,2,0] <= 2
 cost[3,2,1] <= 2
 cost[3,2,2] <= 2
 cost[3,2,3] <= 2
 cost[3,2,4] <= 2
 cost[3,2,5] <= 2
 cost[3,2,6] <= 2
 cost[3,3,0] <= 2
 cost[3,3,1] <= 2
 cost[3,3,2] <= 2
 cost[3,3,3] <= 2
 cost[3,3,4] <= 2
 cost[3,3,5] <= 2
 cost[3,3,6] <= 2
Binaries
 sbox[0,0,0] sbox[0,0,1] sbox[0,0,2] sbox[0,0,3] sbox[0,0,4] sbox[0,0,5]
 sbox[0,0,6] sbox[0,1,0] sbox[0,1,1] sbox[0,1,2] sbox[0,1,3] sbox[0,1,4]
 sbox[0,1,5] sbox[0,1,6] sbox[0,2,0] sbox[0,2,1] sbox[0,2,2] sbox[0,2,3]
 sbox[0,2,4] sbox[0,2,5] sbox[0,2,6] sbox[0,3,0] sbox[0,3,1] sbox[0,3,2]
 sbox[0,3,3] sbox[0,3,4] sbox[0,3,5] sbox[0,3,6] sbox[1,0,0] sbox[1,0,1]
 sbox[1,0,2] sbox[1,0,3] sbox[1,0,4] sbox[1,0,5] sbox[1,0,6] sbox[1,1,0]
 sbox[1,1,1] sbox[1,1,2] sbox[1,1,3] sbox[1,1,4] sbox[1,1,5] sbox[1,1,6]
 sbox[1,2,0] sbox[1,2,1] sbox[1,2,2] sbox[1,2,3] sbox[1,2,4] sbox[1,2,5]
 sbox[1,2,6] sbox[1,3,0] sbox[1,3,1] sbox[1,3,2] sbox[1,3,3] sbox[1,3,4]
 sbox[1,3,5] sbox[1,3,6] sbox[2,0,0] sbox[2,0,1] sbox[2,0,2] sbox[2,0,3]
 sbox[2,0,4] sbox[2,0,5] sbox[2,0,6] sbox[2,1,0] sbox[2,1,1] sbox[2,1,2]
 sbox[2,1,3] sbox[2,1,4] sbox[2,1,5] sbox[2,1,6] sbox[2,2,0] sbox[2,2,1]
 sbox[2,2,2] sbox[2,2,3] sbox[2,2,4] sbox[2,2,5] sbox[2,2,6] sbox[2,3,0]
 sbox[2,3,1] sbox[2,3,2] sbox[2,3,3] sbox[2,3,4] sbox[2,3,5] sbox[2,3,6]
 sbox[3,0,0] sbox[3,0,1] sbox[3,0,2] sbox[3,0,3] sbox[3,0,4] sbox[3,0,5]
 sbox[3,0,6] sbox[3,1,0] sbox[3,1,1] sbox[3,1,2] sbox[3,1,3] sbox[3,1,4]
 sbox[3,1,5] sbox[3,1,6] sbox[3,2,0] sbox[3,2,1] sbox[3,2,2] sbox[3,2,3]
 sbox[3,2,4] sbox[3,2,5] sbox[3,2,6] sbox[3,3,0] sbox[3,3,1] sbox[3,3,2]
 sbox[3,3,3] sbox[3,3,4] sbox[3,3,5] sbox[3,3,6] rkey[0,0,0] rkey[0,0,1]
 rkey[0,0,2] rkey[0,0,3] rkey[0,0,4] rkey[0,0,5] rkey[0,0,6] rkey[0,1,0]
 rkey[0,1,1] rkey[0,1,2] rkey[0,1,3] rkey[0,1,4] rkey[0,1,5] rkey[0,1,6]
 rkey[0,2,0] rkey[0,2,1] rkey[0,2,2] rkey[0,2,3] rkey[0,2,4] rkey[0,2,5]
 rkey[0,2,6] rkey[0,3,0] rkey[0,3,1] rkey[0,3,2] rkey[0,3,3] rkey[0,3,4]
 rkey[0,3,5] rkey[0,3,6] rkey[1,0,0] rkey[1,0,1] rkey[1,0,2] rkey[1,0,3]
 rkey[1,0,4] rkey[1,0,5] rkey[1,0,6] rkey[1,1,0] rkey[1,1,1] rkey[1,1,2]
 rkey[1,1,3] rkey[1,1,4] rkey[1,1,5] rkey[1,1,6] rkey[1,2,0] rkey[1,2,1]
 rkey[1,2,2] rkey[1,2,3] rkey[1,2,4] rkey[1,2,5] rkey[1,2,6] rkey[1,3,0]
 rkey[1,3,1] rkey[1,3,2] rkey[1,3,3] rkey[1,3,4] rkey[1,3,5] rkey[1,3,6]
 rkey[2,0,0] rkey[2,0,1] rkey[2,0,2] rkey[2,0,3] rkey[2,0,4] rkey[2,0,5]
 rkey[2,0,6] rkey[2,1,0] rkey[2,1,1] rkey[2,1,2] rkey[2,1,3] rkey[2,1,4]
 rkey[2,1,5] rkey[2,1,6] rkey[2,2,0] rkey[2,2,1] rkey[2,2,2] rkey[2,2,3]
 rkey[2,2,4] rkey[2,2,5] rkey[2,2,6] rkey[2,3,0] rkey[2,3,1] rkey[2,3,2]
 rkey[2,3,3] rkey[2,3,4] rkey[2,3,5] rkey[2,3,6] rkey[3,0,0] rkey[3,0,1]
 rkey[3,0,2] rkey[3,0,3] rkey[3,0,4] rkey[3,0,5] rkey[3,0,6] rkey[3,1,0]
 rkey[3,1,1] rkey[3,1,2] rkey[3,1,3] rkey[3,1,4] rkey[3,1,5] rkey[3,1,6]
 rkey[3,2,0] rkey[3,2,1] rkey[3,2,2] rkey[3,2,3] rkey[3,2,4] rkey[3,2,5]
 rkey[3,2,6] rkey[3,3,0] rkey[3,3,1] rkey[3,3,2] rkey[3,3,3] rkey[3,3,4]
 rkey[3,3,5] rkey[3,3,6] tkey[0,0,0] tkey[0,0,1] tkey[0,0,2] tkey[0,0,3]
 tkey[0,0,4] tkey[0,0,5] tkey[0,0,6] tkey[0,1,0] tkey[0,1,1] tkey[0,1,2]
 tkey[0,1,3] tkey[0,1,4] tkey[0,1,5] tkey[0,1,6] tkey[0,2,0] tkey[0,2,1]
 tkey[0,2,2] tkey[0,2,3] tkey[0,2,4] tkey[0,2,5] tkey[0,2,6] tkey[0,3,0]
 tkey[0,3,1] tkey[0,3,2] tkey[0,3,3] tkey[0,3,4] tkey[0,3,5] tkey[0,3,6]
 tkey[1,0,0] tkey[1,0,1] tkey[1,0,2] tkey[1,0,3] tkey[1,0,4] tkey[1,0,5]
 tkey[1,0,6] tkey[1,1,0] tkey[1,1,1] tkey[1,1,2] tkey[1,1,3] tkey[1,1,4]
 tkey[1,1,5] tkey[1,1,6] tkey[1,2,0] tkey[1,2,1] tkey[1,2,2] tkey[1,2,3]
 tkey[1,2,4] tkey[1,2,5] tkey[1,2,6] tkey[1,3,0] tkey[1,3,1] tkey[1,3,2]
 tkey[1,3,3] tkey[1,3,4] tkey[1,3,5] tkey[1,3,6] tkey[2,0,0] tkey[2,0,1]
 tkey[2,0,2] tkey[2,0,3] tkey[2,0,4] tkey[2,0,5] tkey[2,0,6] tkey[2,1,0]
 tkey[2,1,1] tkey[2,1,2] tkey[2,1,3] tkey[2,1,4] tkey[2,1,5] tkey[2,1,6]
 tkey[2,2,0] tkey[2,2,1] tkey[2,2,2] tkey[2,2,3] tkey[2,2,4] tkey[2,2,5]
 tkey[2,2,6] tkey[2,3,0] tkey[2,3,1] tkey[2,3,2] tkey[2,3,3] tkey[2,3,4]
 tkey[2,3,5] tkey[2,3,6] tkey[3,0,0] tkey[3,0,1] tkey[3,0,2] tkey[3,0,3]
 tkey[3,0,4] tkey[3,0,5] tkey[3,0,6] tkey[3,1,0] tkey[3,1,1] tkey[3,1,2]
 tkey[3,1,3] tkey[3,1,4] tkey[3,1,5] tkey[3,1,6] tkey[3,2,0] tkey[3,2,1]
 tkey[3,2,2] tkey[3,2,3] tkey[3,2,4] tkey[3,2,5] tkey[3,2,6] tkey[3,3,0]
 tkey[3,3,1] tkey[3,3,2] tkey[3,3,3] tkey[3,3,4] tkey[3,3,5] tkey[3,3,6]
 tbox[0,0,0] tbox[0,0,1] tbox[0,0,2] tbox[0,0,3] tbox[0,0,4] tbox[0,0,5]
 tbox[0,0,6] tbox[0,1,0] tbox[0,1,1] tbox[0,1,2] tbox[0,1,3] tbox[0,1,4]
 tbox[0,1,5] tbox[0,1,6] tbox[0,2,0] tbox[0,2,1] tbox[0,2,2] tbox[0,2,3]
 tbox[0,2,4] tbox[0,2,5] tbox[0,2,6] tbox[0,3,0] tbox[0,3,1] tbox[0,3,2]
 tbox[0,3,3] tbox[0,3,4] tbox[0,3,5] tbox[0,3,6] tbox[1,0,0] tbox[1,0,1]
 tbox[1,0,2] tbox[1,0,3] tbox[1,0,4] tbox[1,0,5] tbox[1,0,6] tbox[1,1,0]
 tbox[1,1,1] tbox[1,1,2] tbox[1,1,3] tbox[1,1,4] tbox[1,1,5] tbox[1,1,6]
 tbox[1,2,0] tbox[1,2,1] tbox[1,2,2] tbox[1,2,3] tbox[1,2,4] tbox[1,2,5]
 tbox[1,2,6] tbox[1,3,0] tbox[1,3,1] tbox[1,3,2] tbox[1,3,3] tbox[1,3,4]
 tbox[1,3,5] tbox[1,3,6] tbox[2,0,0] tbox[2,0,1] tbox[2,0,2] tbox[2,0,3]
 tbox[2,0,4] tbox[2,0,5] tbox[2,0,6] tbox[2,1,0] tbox[2,1,1] tbox[2,1,2]
 tbox[2,1,3] tbox[2,1,4] tbox[2,1,5] tbox[2,1,6] tbox[2,2,0] tbox[2,2,1]
 tbox[2,2,2] tbox[2,2,3] tbox[2,2,4] tbox[2,2,5] tbox[2,2,6] tbox[2,3,0]
 tbox[2,3,1] tbox[2,3,2] tbox[2,3,3] tbox[2,3,4] tbox[2,3,5] tbox[2,3,6]
 tbox[3,0,0] tbox[3,0,1] tbox[3,0,2] tbox[3,0,3] tbox[3,0,4] tbox[3,0,5]
 tbox[3,0,6] tbox[3,1,0] tbox[3,1,1] tbox[3,1,2] tbox[3,1,3] tbox[3,1,4]
 tbox[3,1,5] tbox[3,1,6] tbox[3,2,0] tbox[3,2,1] tbox[3,2,2] tbox[3,2,3]
 tbox[3,2,4] tbox[3,2,5] tbox[3,2,6] tbox[3,3,0] tbox[3,3,1] tbox[3,3,2]
 tbox[3,3,3] tbox[3,3,4] tbox[3,3,5] tbox[3,3,6] unkn[0,0,0] unkn[0,0,1]
 unkn[0,0,2] unkn[0,0,3] unkn[0,0,4] unkn[0,0,5] unkn[0,0,6] unkn[0,1,0]
 unkn[0,1,1] unkn[0,1,2] unkn[0,1,3] unkn[0,1,4] unkn[0,1,5] unkn[0,1,6]
 unkn[0,2,0] unkn[0,2,1] unkn[0,2,2] unkn[0,2,3] unkn[0,2,4] unkn[0,2,5]
 unkn[0,2,6] unkn[0,3,0] unkn[0,3,1] unkn[0,3,2] unkn[0,3,3] unkn[0,3,4]
 unkn[0,3,5] unkn[0,3,6] unkn[1,0,0] unkn[1,0,1] unkn[1,0,2] unkn[1,0,3]
 unkn[1,0,4] unkn[1,0,5] unkn[1,0,6] unkn[1,1,0] unkn[1,1,1] unkn[1,1,2]
 unkn[1,1,3] unkn[1,1,4] unkn[1,1,5] unkn[1,1,6] unkn[1,2,0] unkn[1,2,1]
 unkn[1,2,2] unkn[1,2,3] unkn[1,2,4] unkn[1,2,5] unkn[1,2,6] unkn[1,3,0]
 unkn[1,3,1] unkn[1,3,2] unkn[1,3,3] unkn[1,3,4] unkn[1,3,5] unkn[1,3,6]
 unkn[2,0,0] unkn[2,0,1] unkn[2,0,2] unkn[2,0,3] unkn[2,0,4] unkn[2,0,5]
 unkn[2,0,6] unkn[2,1,0] unkn[2,1,1] unkn[2,1,2] unkn[2,1,3] unkn[2,1,4]
 unkn[2,1,5] unkn[2,1,6] unkn[2,2,0] unkn[2,2,1] unkn[2,2,2] unkn[2,2,3]
 unkn[2,2,4] unkn[2,2,5] unkn[2,2,6] unkn[2,3,0] unkn[2,3,1] unkn[2,3,2]
 unkn[2,3,3] unkn[2,3,4] unkn[2,3,5] unkn[2,3,6] unkn[3,0,0] unkn[3,0,1]
 unkn[3,0,2] unkn[3,0,3] unkn[3,0,4] unkn[3,0,5] unkn[3,0,6] unkn[3,1,0]
 unkn[3,1,1] unkn[3,1,2] unkn[3,1,3] unkn[3,1,4] unkn[3,1,5] unkn[3,1,6]
 unkn[3,2,0] unkn[3,2,1] unkn[3,2,2] unkn[3,2,3] unkn[3,2,4] unkn[3,2,5]
 unkn[3,2,6] unkn[3,3,0] unkn[3,3,1] unkn[3,3,2] unkn[3,3,3] unkn[3,3,4]
 unkn[3,3,5] unkn[3,3,6] forg[0,0,0] forg[0,0,1] forg[0,0,2] forg[0,0,3]
 forg[0,0,4] forg[0,0,5] forg[0,0,6] forg[0,1,0] forg[0,1,1] forg[0,1,2]
 forg[0,1,3] forg[0,1,4] forg[0,1,5] forg[0,1,6] forg[0,2,0] forg[0,2,1]
 forg[0,2,2] forg[0,2,3] forg[0,2,4] forg[0,2,5] forg[0,2,6] forg[0,3,0]
 forg[0,3,1] forg[0,3,2] forg[0,3,3] forg[0,3,4] forg[0,3,5] forg[0,3,6]
 forg[1,0,0] forg[1,0,1] forg[1,0,2] forg[1,0,3] forg[1,0,4] forg[1,0,5]
 forg[1,0,6] forg[1,1,0] forg[1,1,1] forg[1,1,2] forg[1,1,3] forg[1,1,4]
 forg[1,1,5] forg[1,1,6] forg[1,2,0] forg[1,2,1] forg[1,2,2] forg[1,2,3]
 forg[1,2,4] forg[1,2,5] forg[1,2,6] forg[1,3,0] forg[1,3,1] forg[1,3,2]
 forg[1,3,3] forg[1,3,4] forg[1,3,5] forg[1,3,6] forg[2,0,0] forg[2,0,1]
 forg[2,0,2] forg[2,0,3] forg[2,0,4] forg[2,0,5] forg[2,0,6] forg[2,1,0]
 forg[2,1,1] forg[2,1,2] forg[2,1,3] forg[2,1,4] forg[2,1,5] forg[2,1,6]
 forg[2,2,0] forg[2,2,1] forg[2,2,2] forg[2,2,3] forg[2,2,4] forg[2,2,5]
 forg[2,2,6] forg[2,3,0] forg[2,3,1] forg[2,3,2] forg[2,3,3] forg[2,3,4]
 forg[2,3,5] forg[2,3,6] forg[3,0,0] forg[3,0,1] forg[3,0,2] forg[3,0,3]
 forg[3,0,4] forg[3,0,5] forg[3,0,6] forg[3,1,0] forg[3,1,1] forg[3,1,2]
 forg[3,1,3] forg[3,1,4] forg[3,1,5] forg[3,1,6] forg[3,2,0] forg[3,2,1]
 forg[3,2,2] forg[3,2,3] forg[3,2,4] forg[3,2,5] forg[3,2,6] forg[3,3,0]
 forg[3,3,1] forg[3,3,2] forg[3,3,3] forg[3,3,4] forg[3,3,5] forg[3,3,6]
 lane[0,0] lane[0,1] lane[0,2] lane[0,3] lane[1,0] lane[1,1] lane[1,2]
 lane[1,3] lane[2,0] lane[2,1] lane[2,2] lane[2,3] lane[3,0] lane[3,1]
 lane[3,2] lane[3,3]
Generals
 cost[0,0,0] cost[0,0,1] cost[0,0,2] cost[0,0,3] cost[0,0,4] cost[0,0,5]
 cost[0,0,6] cost[0,1,0] cost[0,1,1] cost[0,1,2] cost[0,1,3] cost[0,1,4]
 cost[0,1,5] cost[0,1,6] cost[0,2,0] cost[0,2,1] cost[0,2,2] cost[0,2,3]
 cost[0,2,4] cost[0,2,5] cost[0,2,6] cost[0,3,0] cost[0,3,1] cost[0,3,2]
 cost[0,3,3] cost[0,3,4] cost[0,3,5] cost[0,3,6] cost[1,0,0] cost[1,0,1]
 cost[1,0,2] cost[1,0,3] cost[1,0,4] cost[1,0,5] cost[1,0,6] cost[1,1,0]
 cost[1,1,1] cost[1,1,2] cost[1,1,3] cost[1,1,4] cost[1,1,5] cost[1,1,6]
 cost[1,2,0] cost[1,2,1] cost[1,2,2] cost[1,2,3] cost[1,2,4] cost[1,2,5]
 cost[1,2,6] cost[1,3,0] cost[1,3,1] cost[1,3,2] cost[1,3,3] cost[1,3,4]
 cost[1,3,5] cost[1,3,6] cost[2,0,0] cost[2,0,1] cost[2,0,2] cost[2,0,3]
 cost[2,0,4] cost[2,0,5] cost[2,0,6] cost[2,1,0] cost[2,1,1] cost[2,1,2]
 cost[2,1,3] cost[2,1,4] cost[2,1,5] cost[2,1,6] cost[2,2,0] cost[2,2,1]
 cost[2,2,2] cost[2,2,3] cost[2,2,4] cost[2,2,5] cost[2,2,6] cost[2,3,0]
 cost[2,3,1] cost[2,3,2] cost[2,3,3] cost[2,3,4] cost[2,3,5] cost[2,3,6]
 cost[3,0,0] cost[3,0,1] cost[3,0,2] cost[3,0,3] cost[3,0,4] cost[3,0,5]
 cost[3,0,6] cost[3,1,0] cost[3,1,1] cost[3,1,2] cost[3,1,3] cost[3,1,4]
 cost[3,1,5] cost[3,1,6] cost[3,2,0] cost[3,2,1] cost[3,2,2] cost[3,2,3]
 cost[3,2,4] cost[3,2,5] cost[3,2,6] cost[3,3,0] cost[3,3,1] cost[3,3,2]
 cost[3,3,3] cost[3,3,4] cost[3,3,5] cost[3,3,6]
End
