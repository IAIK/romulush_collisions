\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 rkey[0,0,5] + 0 rkey[0,1,5] + 0 rkey[0,2,5] + 0 rkey[0,3,5]
   + 0 rkey[1,0,5] + 0 rkey[1,1,5] + 0 rkey[1,2,5] + 0 rkey[1,3,5]
   + 0 rkey[2,0,5] + 0 rkey[2,1,5] + 0 rkey[2,2,5] + 0 rkey[2,3,5]
   + 0 rkey[3,0,5] + 0 rkey[3,1,5] + 0 rkey[3,2,5] + 0 rkey[3,3,5]
   + 0 tkey[0,0,5] + 0 tkey[0,1,5] + 0 tkey[0,2,5] + 0 tkey[0,3,5]
   + 0 tkey[1,0,5] + 0 tkey[1,1,5] + 0 tkey[1,2,5] + 0 tkey[1,3,5]
   + 0 tkey[2,0,5] + 0 tkey[2,1,5] + 0 tkey[2,2,5] + 0 tkey[2,3,5]
   + 0 tkey[3,0,5] + 0 tkey[3,1,5] + 0 tkey[3,2,5] + 0 tkey[3,3,5]
   + cost[0,0,0] + cost[0,0,1] + cost[0,0,2] + cost[0,0,3] + cost[0,0,4]
   + 0 cost[0,0,5] + cost[0,1,0] + cost[0,1,1] + cost[0,1,2] + cost[0,1,3]
   + cost[0,1,4] + 0 cost[0,1,5] + cost[0,2,0] + cost[0,2,1] + cost[0,2,2]
   + cost[0,2,3] + cost[0,2,4] + 0 cost[0,2,5] + cost[0,3,0] + cost[0,3,1]
   + cost[0,3,2] + cost[0,3,3] + cost[0,3,4] + 0 cost[0,3,5] + cost[1,0,0]
   + cost[1,0,1] + cost[1,0,2] + cost[1,0,3] + cost[1,0,4] + 0 cost[1,0,5]
   + cost[1,1,0] + cost[1,1,1] + cost[1,1,2] + cost[1,1,3] + cost[1,1,4]
   + 0 cost[1,1,5] + cost[1,2,0] + cost[1,2,1] + cost[1,2,2] + cost[1,2,3]
   + cost[1,2,4] + 0 cost[1,2,5] + cost[1,3,0] + cost[1,3,1] + cost[1,3,2]
   + cost[1,3,3] + cost[1,3,4] + 0 cost[1,3,5] + cost[2,0,0] + cost[2,0,1]
   + cost[2,0,2] + cost[2,0,3] + cost[2,0,4] + 0 cost[2,0,5] + cost[2,1,0]
   + cost[2,1,1] + cost[2,1,2] + cost[2,1,3] + cost[2,1,4] + 0 cost[2,1,5]
   + cost[2,2,0] + cost[2,2,1] + cost[2,2,2] + cost[2,2,3] + cost[2,2,4]
   + 0 cost[2,2,5] + cost[2,3,0] + cost[2,3,1] + cost[2,3,2] + cost[2,3,3]
   + cost[2,3,4] + 0 cost[2,3,5] + cost[3,0,0] + cost[3,0,1] + cost[3,0,2]
   + cost[3,0,3] + cost[3,0,4] + 0 cost[3,0,5] + cost[3,1,0] + cost[3,1,1]
   + cost[3,1,2] + cost[3,1,3] + cost[3,1,4] + 0 cost[3,1,5] + cost[3,2,0]
   + cost[3,2,1] + cost[3,2,2] + cost[3,2,3] + cost[3,2,4] + 0 cost[3,2,5]
   + cost[3,3,0] + cost[3,3,1] + cost[3,3,2] + cost[3,3,3] + cost[3,3,4]
   + 0 cost[3,3,5]
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
 R240: tkey[0,0,0] - lane[0,0] <= 0
 R241: tkey[2,0,1] - lane[0,0] <= 0
 R242: tkey[0,2,2] - lane[0,0] <= 0
 R243: tkey[2,2,3] - lane[0,0] <= 0
 R244: tkey[1,0,4] - lane[0,0] <= 0
 R245: tkey[0,1,0] - lane[0,1] <= 0
 R246: tkey[2,1,1] - lane[0,1] <= 0
 R247: tkey[0,0,2] - lane[0,1] <= 0
 R248: tkey[2,0,3] - lane[0,1] <= 0
 R249: tkey[0,2,4] - lane[0,1] <= 0
 R250: tkey[0,2,0] - lane[0,2] <= 0
 R251: tkey[2,2,1] - lane[0,2] <= 0
 R252: tkey[1,0,2] - lane[0,2] <= 0
 R253: tkey[3,0,3] - lane[0,2] <= 0
 R254: tkey[1,2,4] - lane[0,2] <= 0
 R255: tkey[0,3,0] - lane[0,3] <= 0
 R256: tkey[2,3,1] - lane[0,3] <= 0
 R257: tkey[1,3,2] - lane[0,3] <= 0
 R258: tkey[3,3,3] - lane[0,3] <= 0
 R259: tkey[0,1,4] - lane[0,3] <= 0
 R260: tkey[1,0,0] - lane[1,0] <= 0
 R261: tkey[3,0,1] - lane[1,0] <= 0
 R262: tkey[1,2,2] - lane[1,0] <= 0
 R263: tkey[3,2,3] - lane[1,0] <= 0
 R264: tkey[1,1,4] - lane[1,0] <= 0
 R265: tkey[1,1,0] - lane[1,1] <= 0
 R266: tkey[3,1,1] - lane[1,1] <= 0
 R267: tkey[0,3,2] - lane[1,1] <= 0
 R268: tkey[2,3,3] - lane[1,1] <= 0
 R269: tkey[1,3,4] - lane[1,1] <= 0
 R270: tkey[1,2,0] - lane[1,2] <= 0
 R271: tkey[3,2,1] - lane[1,2] <= 0
 R272: tkey[1,1,2] - lane[1,2] <= 0
 R273: tkey[3,1,3] - lane[1,2] <= 0
 R274: tkey[0,3,4] - lane[1,2] <= 0
 R275: tkey[1,3,0] - lane[1,3] <= 0
 R276: tkey[3,3,1] - lane[1,3] <= 0
 R277: tkey[0,1,2] - lane[1,3] <= 0
 R278: tkey[2,1,3] - lane[1,3] <= 0
 R279: tkey[0,0,4] - lane[1,3] <= 0
 R280: tkey[2,0,0] - lane[2,0] <= 0
 R281: tkey[0,2,1] - lane[2,0] <= 0
 R282: tkey[2,2,2] - lane[2,0] <= 0
 R283: tkey[1,0,3] - lane[2,0] <= 0
 R284: tkey[3,0,4] - lane[2,0] <= 0
 R285: tkey[2,1,0] - lane[2,1] <= 0
 R286: tkey[0,0,1] - lane[2,1] <= 0
 R287: tkey[2,0,2] - lane[2,1] <= 0
 R288: tkey[0,2,3] - lane[2,1] <= 0
 R289: tkey[2,2,4] - lane[2,1] <= 0
 R290: tkey[2,2,0] - lane[2,2] <= 0
 R291: tkey[1,0,1] - lane[2,2] <= 0
 R292: tkey[3,0,2] - lane[2,2] <= 0
 R293: tkey[1,2,3] - lane[2,2] <= 0
 R294: tkey[3,2,4] - lane[2,2] <= 0
 R295: tkey[2,3,0] - lane[2,3] <= 0
 R296: tkey[1,3,1] - lane[2,3] <= 0
 R297: tkey[3,3,2] - lane[2,3] <= 0
 R298: tkey[0,1,3] - lane[2,3] <= 0
 R299: tkey[2,1,4] - lane[2,3] <= 0
 R300: tkey[3,0,0] - lane[3,0] <= 0
 R301: tkey[1,2,1] - lane[3,0] <= 0
 R302: tkey[3,2,2] - lane[3,0] <= 0
 R303: tkey[1,1,3] - lane[3,0] <= 0
 R304: tkey[3,1,4] - lane[3,0] <= 0
 R305: tkey[3,1,0] - lane[3,1] <= 0
 R306: tkey[0,3,1] - lane[3,1] <= 0
 R307: tkey[2,3,2] - lane[3,1] <= 0
 R308: tkey[1,3,3] - lane[3,1] <= 0
 R309: tkey[3,3,4] - lane[3,1] <= 0
 R310: tkey[3,2,0] - lane[3,2] <= 0
 R311: tkey[1,1,1] - lane[3,2] <= 0
 R312: tkey[3,1,2] - lane[3,2] <= 0
 R313: tkey[0,3,3] - lane[3,2] <= 0
 R314: tkey[2,3,4] - lane[3,2] <= 0
 R315: tkey[3,3,0] - lane[3,3] <= 0
 R316: tkey[0,1,1] - lane[3,3] <= 0
 R317: tkey[2,1,2] - lane[3,3] <= 0
 R318: tkey[0,0,3] - lane[3,3] <= 0
 R319: tkey[2,0,4] - lane[3,3] <= 0
 R320: - tkey[0,0,0] - tkey[0,2,2] - tkey[1,0,4] - tkey[2,0,1]
   - tkey[2,2,3] + 5 lane[0,0] <= 1
 R321: - tkey[0,0,2] - tkey[0,1,0] - tkey[0,2,4] - tkey[2,0,3]
   - tkey[2,1,1] + 5 lane[0,1] <= 1
 R322: - tkey[0,2,0] - tkey[1,0,2] - tkey[1,2,4] - tkey[2,2,1]
   - tkey[3,0,3] + 5 lane[0,2] <= 1
 R323: - tkey[0,1,4] - tkey[0,3,0] - tkey[1,3,2] - tkey[2,3,1]
   - tkey[3,3,3] + 5 lane[0,3] <= 1
 R324: - tkey[1,0,0] - tkey[1,1,4] - tkey[1,2,2] - tkey[3,0,1]
   - tkey[3,2,3] + 5 lane[1,0] <= 1
 R325: - tkey[0,3,2] - tkey[1,1,0] - tkey[1,3,4] - tkey[2,3,3]
   - tkey[3,1,1] + 5 lane[1,1] <= 1
 R326: - tkey[0,3,4] - tkey[1,1,2] - tkey[1,2,0] - tkey[3,1,3]
   - tkey[3,2,1] + 5 lane[1,2] <= 1
 R327: - tkey[0,0,4] - tkey[0,1,2] - tkey[1,3,0] - tkey[2,1,3]
   - tkey[3,3,1] + 5 lane[1,3] <= 1
 R328: - tkey[0,2,1] - tkey[1,0,3] - tkey[2,0,0] - tkey[2,2,2]
   - tkey[3,0,4] + 5 lane[2,0] <= 1
 R329: - tkey[0,0,1] - tkey[0,2,3] - tkey[2,0,2] - tkey[2,1,0]
   - tkey[2,2,4] + 5 lane[2,1] <= 1
 R330: - tkey[1,0,1] - tkey[1,2,3] - tkey[2,2,0] - tkey[3,0,2]
   - tkey[3,2,4] + 5 lane[2,2] <= 1
 R331: - tkey[0,1,3] - tkey[1,3,1] - tkey[2,1,4] - tkey[2,3,0]
   - tkey[3,3,2] + 5 lane[2,3] <= 1
 R332: - tkey[1,1,3] - tkey[1,2,1] - tkey[3,0,0] - tkey[3,1,4]
   - tkey[3,2,2] + 5 lane[3,0] <= 1
 R333: - tkey[0,3,1] - tkey[1,3,3] - tkey[2,3,2] - tkey[3,1,0]
   - tkey[3,3,4] + 5 lane[3,1] <= 1
 R334: - tkey[0,3,3] - tkey[1,1,1] - tkey[2,3,4] - tkey[3,1,2]
   - tkey[3,2,0] + 5 lane[3,2] <= 1
 R335: - tkey[0,0,3] - tkey[0,1,1] - tkey[2,0,4] - tkey[2,1,2]
   - tkey[3,3,0] + 5 lane[3,3] <= 1
 R336: - tbox[0,0,0] + forg[0,0,0] <= 0
 R337: sbox[0,0,0] - rkey[0,0,0] + tkey[0,0,0] >= 0
 R338: sbox[0,0,0] + rkey[0,0,0] - tkey[0,0,0] >= 0
 R339: - sbox[0,0,0] + rkey[0,0,0] + tkey[0,0,0] >= 0
 R340: - tbox[1,0,0] + forg[1,0,0] <= 0
 R341: sbox[1,0,0] - rkey[1,0,0] + tkey[1,0,0] >= 0
 R342: sbox[1,0,0] + rkey[1,0,0] - tkey[1,0,0] >= 0
 R343: - sbox[1,0,0] + rkey[1,0,0] + tkey[1,0,0] >= 0
 R344: - tbox[2,0,0] + forg[2,0,0] <= 0
 R345: - sbox[2,0,0] + rkey[2,0,0] = 0
 R346: - tbox[3,0,0] + forg[3,0,0] <= 0
 R347: - sbox[3,0,0] + rkey[3,0,0] = 0
 R348: - sbox[3,0,1] + rkey[0,0,0] + rkey[2,2,0] >= 0
 R349: sbox[3,0,1] + rkey[0,0,0] - rkey[2,2,0] >= 0
 R350: sbox[3,0,1] - rkey[0,0,0] + rkey[2,2,0] >= 0
 R351: - sbox[0,0,1] + sbox[3,0,1] + rkey[3,1,0] >= 0
 R352: sbox[0,0,1] + sbox[3,0,1] - rkey[3,1,0] >= 0
 R353: sbox[0,0,1] - sbox[3,0,1] + rkey[3,1,0] >= 0
 R354: - sbox[2,0,1] + rkey[1,3,0] + rkey[2,2,0] >= 0
 R355: sbox[2,0,1] + rkey[1,3,0] - rkey[2,2,0] >= 0
 R356: sbox[2,0,1] - rkey[1,3,0] + rkey[2,2,0] >= 0
 R357: sbox[1,0,1] - rkey[0,0,0] = 0
 R358: tbox[0,0,0] + tbox[2,2,0] - tbox[3,0,1] >= 0
 R359: tbox[0,0,0] - tbox[2,2,0] + tbox[3,0,1] >= 0
 R360: - tbox[0,0,0] + tbox[2,2,0] + tbox[3,0,1] >= 0
 R361: - tbox[0,0,1] + tbox[3,0,1] + tbox[3,1,0] >= 0
 R362: tbox[0,0,1] + tbox[3,0,1] - tbox[3,1,0] >= 0
 R363: tbox[0,0,1] - tbox[3,0,1] + tbox[3,1,0] >= 0
 R364: tbox[1,3,0] - tbox[2,0,1] + tbox[2,2,0] >= 0
 R365: tbox[1,3,0] + tbox[2,0,1] - tbox[2,2,0] >= 0
 R366: - tbox[1,3,0] + tbox[2,0,1] + tbox[2,2,0] >= 0
 R367: - tbox[0,0,0] + tbox[1,0,1] = 0
 R368: unkn[0,0,0] - unkn[3,0,1] + forg[3,0,1] <= 0
 R369: unkn[2,2,0] - unkn[3,0,1] + forg[3,0,1] <= 0
 R370: unkn[0,0,0] + unkn[2,2,0] - unkn[3,0,1] + forg[3,0,1] >= 0
 R371: - unkn[0,0,1] + unkn[3,0,1] + forg[0,0,1] - forg[3,0,1] <= 0
 R372: - unkn[0,0,1] + unkn[3,1,0] + forg[0,0,1] <= 0
 R373: - unkn[0,0,1] + unkn[3,0,1] + unkn[3,1,0] + forg[0,0,1]
   - forg[3,0,1] >= 0
 R374: unkn[1,3,0] - unkn[2,0,1] + forg[2,0,1] <= 0
 R375: - unkn[2,0,1] + unkn[2,2,0] + forg[2,0,1] <= 0
 R376: unkn[1,3,0] - unkn[2,0,1] + unkn[2,2,0] + forg[2,0,1] >= 0
 R377: unkn[0,0,0] - unkn[1,0,1] + forg[1,0,1] = 0
 R378: - tbox[0,1,0] + forg[0,1,0] <= 0
 R379: sbox[0,1,0] - rkey[0,1,0] + tkey[0,1,0] >= 0
 R380: sbox[0,1,0] + rkey[0,1,0] - tkey[0,1,0] >= 0
 R381: - sbox[0,1,0] + rkey[0,1,0] + tkey[0,1,0] >= 0
 R382: - tbox[1,1,0] + forg[1,1,0] <= 0
 R383: sbox[1,1,0] - rkey[1,1,0] + tkey[1,1,0] >= 0
 R384: sbox[1,1,0] + rkey[1,1,0] - tkey[1,1,0] >= 0
 R385: - sbox[1,1,0] + rkey[1,1,0] + tkey[1,1,0] >= 0
 R386: - tbox[2,1,0] + forg[2,1,0] <= 0
 R387: - sbox[2,1,0] + rkey[2,1,0] = 0
 R388: - tbox[3,1,0] + forg[3,1,0] <= 0
 R389: - sbox[3,1,0] + rkey[3,1,0] = 0
 R390: - sbox[3,1,1] + rkey[0,1,0] + rkey[2,3,0] >= 0
 R391: sbox[3,1,1] + rkey[0,1,0] - rkey[2,3,0] >= 0
 R392: sbox[3,1,1] - rkey[0,1,0] + rkey[2,3,0] >= 0
 R393: - sbox[0,1,1] + sbox[3,1,1] + rkey[3,2,0] >= 0
 R394: sbox[0,1,1] + sbox[3,1,1] - rkey[3,2,0] >= 0
 R395: sbox[0,1,1] - sbox[3,1,1] + rkey[3,2,0] >= 0
 R396: - sbox[2,1,1] + rkey[1,0,0] + rkey[2,3,0] >= 0
 R397: sbox[2,1,1] + rkey[1,0,0] - rkey[2,3,0] >= 0
 R398: sbox[2,1,1] - rkey[1,0,0] + rkey[2,3,0] >= 0
 R399: sbox[1,1,1] - rkey[0,1,0] = 0
 R400: tbox[0,1,0] + tbox[2,3,0] - tbox[3,1,1] >= 0
 R401: tbox[0,1,0] - tbox[2,3,0] + tbox[3,1,1] >= 0
 R402: - tbox[0,1,0] + tbox[2,3,0] + tbox[3,1,1] >= 0
 R403: - tbox[0,1,1] + tbox[3,1,1] + tbox[3,2,0] >= 0
 R404: tbox[0,1,1] + tbox[3,1,1] - tbox[3,2,0] >= 0
 R405: tbox[0,1,1] - tbox[3,1,1] + tbox[3,2,0] >= 0
 R406: tbox[1,0,0] - tbox[2,1,1] + tbox[2,3,0] >= 0
 R407: tbox[1,0,0] + tbox[2,1,1] - tbox[2,3,0] >= 0
 R408: - tbox[1,0,0] + tbox[2,1,1] + tbox[2,3,0] >= 0
 R409: - tbox[0,1,0] + tbox[1,1,1] = 0
 R410: unkn[0,1,0] - unkn[3,1,1] + forg[3,1,1] <= 0
 R411: unkn[2,3,0] - unkn[3,1,1] + forg[3,1,1] <= 0
 R412: unkn[0,1,0] + unkn[2,3,0] - unkn[3,1,1] + forg[3,1,1] >= 0
 R413: - unkn[0,1,1] + unkn[3,1,1] + forg[0,1,1] - forg[3,1,1] <= 0
 R414: - unkn[0,1,1] + unkn[3,2,0] + forg[0,1,1] <= 0
 R415: - unkn[0,1,1] + unkn[3,1,1] + unkn[3,2,0] + forg[0,1,1]
   - forg[3,1,1] >= 0
 R416: unkn[1,0,0] - unkn[2,1,1] + forg[2,1,1] <= 0
 R417: - unkn[2,1,1] + unkn[2,3,0] + forg[2,1,1] <= 0
 R418: unkn[1,0,0] - unkn[2,1,1] + unkn[2,3,0] + forg[2,1,1] >= 0
 R419: unkn[0,1,0] - unkn[1,1,1] + forg[1,1,1] = 0
 R420: - tbox[0,2,0] + forg[0,2,0] <= 0
 R421: sbox[0,2,0] - rkey[0,2,0] + tkey[0,2,0] >= 0
 R422: sbox[0,2,0] + rkey[0,2,0] - tkey[0,2,0] >= 0
 R423: - sbox[0,2,0] + rkey[0,2,0] + tkey[0,2,0] >= 0
 R424: - tbox[1,2,0] + forg[1,2,0] <= 0
 R425: sbox[1,2,0] - rkey[1,2,0] + tkey[1,2,0] >= 0
 R426: sbox[1,2,0] + rkey[1,2,0] - tkey[1,2,0] >= 0
 R427: - sbox[1,2,0] + rkey[1,2,0] + tkey[1,2,0] >= 0
 R428: - tbox[2,2,0] + forg[2,2,0] <= 0
 R429: - sbox[2,2,0] + rkey[2,2,0] = 0
 R430: - tbox[3,2,0] + forg[3,2,0] <= 0
 R431: - sbox[3,2,0] + rkey[3,2,0] = 0
 R432: - sbox[3,2,1] + rkey[0,2,0] + rkey[2,0,0] >= 0
 R433: sbox[3,2,1] + rkey[0,2,0] - rkey[2,0,0] >= 0
 R434: sbox[3,2,1] - rkey[0,2,0] + rkey[2,0,0] >= 0
 R435: - sbox[0,2,1] + sbox[3,2,1] + rkey[3,3,0] >= 0
 R436: sbox[0,2,1] + sbox[3,2,1] - rkey[3,3,0] >= 0
 R437: sbox[0,2,1] - sbox[3,2,1] + rkey[3,3,0] >= 0
 R438: - sbox[2,2,1] + rkey[1,1,0] + rkey[2,0,0] >= 0
 R439: sbox[2,2,1] + rkey[1,1,0] - rkey[2,0,0] >= 0
 R440: sbox[2,2,1] - rkey[1,1,0] + rkey[2,0,0] >= 0
 R441: sbox[1,2,1] - rkey[0,2,0] = 0
 R442: tbox[0,2,0] + tbox[2,0,0] - tbox[3,2,1] >= 0
 R443: tbox[0,2,0] - tbox[2,0,0] + tbox[3,2,1] >= 0
 R444: - tbox[0,2,0] + tbox[2,0,0] + tbox[3,2,1] >= 0
 R445: - tbox[0,2,1] + tbox[3,2,1] + tbox[3,3,0] >= 0
 R446: tbox[0,2,1] + tbox[3,2,1] - tbox[3,3,0] >= 0
 R447: tbox[0,2,1] - tbox[3,2,1] + tbox[3,3,0] >= 0
 R448: tbox[1,1,0] + tbox[2,0,0] - tbox[2,2,1] >= 0
 R449: tbox[1,1,0] - tbox[2,0,0] + tbox[2,2,1] >= 0
 R450: - tbox[1,1,0] + tbox[2,0,0] + tbox[2,2,1] >= 0
 R451: - tbox[0,2,0] + tbox[1,2,1] = 0
 R452: unkn[0,2,0] - unkn[3,2,1] + forg[3,2,1] <= 0
 R453: unkn[2,0,0] - unkn[3,2,1] + forg[3,2,1] <= 0
 R454: unkn[0,2,0] + unkn[2,0,0] - unkn[3,2,1] + forg[3,2,1] >= 0
 R455: - unkn[0,2,1] + unkn[3,2,1] + forg[0,2,1] - forg[3,2,1] <= 0
 R456: - unkn[0,2,1] + unkn[3,3,0] + forg[0,2,1] <= 0
 R457: - unkn[0,2,1] + unkn[3,2,1] + unkn[3,3,0] + forg[0,2,1]
   - forg[3,2,1] >= 0
 R458: unkn[1,1,0] - unkn[2,2,1] + forg[2,2,1] <= 0
 R459: unkn[2,0,0] - unkn[2,2,1] + forg[2,2,1] <= 0
 R460: unkn[1,1,0] + unkn[2,0,0] - unkn[2,2,1] + forg[2,2,1] >= 0
 R461: unkn[0,2,0] - unkn[1,2,1] + forg[1,2,1] = 0
 R462: - tbox[0,3,0] + forg[0,3,0] <= 0
 R463: sbox[0,3,0] - rkey[0,3,0] + tkey[0,3,0] >= 0
 R464: sbox[0,3,0] + rkey[0,3,0] - tkey[0,3,0] >= 0
 R465: - sbox[0,3,0] + rkey[0,3,0] + tkey[0,3,0] >= 0
 R466: - tbox[1,3,0] + forg[1,3,0] <= 0
 R467: sbox[1,3,0] - rkey[1,3,0] + tkey[1,3,0] >= 0
 R468: sbox[1,3,0] + rkey[1,3,0] - tkey[1,3,0] >= 0
 R469: - sbox[1,3,0] + rkey[1,3,0] + tkey[1,3,0] >= 0
 R470: - tbox[2,3,0] + forg[2,3,0] <= 0
 R471: - sbox[2,3,0] + rkey[2,3,0] = 0
 R472: - tbox[3,3,0] + forg[3,3,0] <= 0
 R473: - sbox[3,3,0] + rkey[3,3,0] = 0
 R474: - sbox[3,3,1] + rkey[0,3,0] + rkey[2,1,0] >= 0
 R475: sbox[3,3,1] + rkey[0,3,0] - rkey[2,1,0] >= 0
 R476: sbox[3,3,1] - rkey[0,3,0] + rkey[2,1,0] >= 0
 R477: - sbox[0,3,1] + sbox[3,3,1] + rkey[3,0,0] >= 0
 R478: sbox[0,3,1] + sbox[3,3,1] - rkey[3,0,0] >= 0
 R479: sbox[0,3,1] - sbox[3,3,1] + rkey[3,0,0] >= 0
 R480: - sbox[2,3,1] + rkey[1,2,0] + rkey[2,1,0] >= 0
 R481: sbox[2,3,1] + rkey[1,2,0] - rkey[2,1,0] >= 0
 R482: sbox[2,3,1] - rkey[1,2,0] + rkey[2,1,0] >= 0
 R483: sbox[1,3,1] - rkey[0,3,0] = 0
 R484: tbox[0,3,0] + tbox[2,1,0] - tbox[3,3,1] >= 0
 R485: tbox[0,3,0] - tbox[2,1,0] + tbox[3,3,1] >= 0
 R486: - tbox[0,3,0] + tbox[2,1,0] + tbox[3,3,1] >= 0
 R487: - tbox[0,3,1] + tbox[3,0,0] + tbox[3,3,1] >= 0
 R488: tbox[0,3,1] - tbox[3,0,0] + tbox[3,3,1] >= 0
 R489: tbox[0,3,1] + tbox[3,0,0] - tbox[3,3,1] >= 0
 R490: tbox[1,2,0] + tbox[2,1,0] - tbox[2,3,1] >= 0
 R491: tbox[1,2,0] - tbox[2,1,0] + tbox[2,3,1] >= 0
 R492: - tbox[1,2,0] + tbox[2,1,0] + tbox[2,3,1] >= 0
 R493: - tbox[0,3,0] + tbox[1,3,1] = 0
 R494: unkn[0,3,0] - unkn[3,3,1] + forg[3,3,1] <= 0
 R495: unkn[2,1,0] - unkn[3,3,1] + forg[3,3,1] <= 0
 R496: unkn[0,3,0] + unkn[2,1,0] - unkn[3,3,1] + forg[3,3,1] >= 0
 R497: - unkn[0,3,1] + unkn[3,3,1] + forg[0,3,1] - forg[3,3,1] <= 0
 R498: - unkn[0,3,1] + unkn[3,0,0] + forg[0,3,1] <= 0
 R499: - unkn[0,3,1] + unkn[3,0,0] + unkn[3,3,1] + forg[0,3,1]
   - forg[3,3,1] >= 0
 R500: unkn[1,2,0] - unkn[2,3,1] + forg[2,3,1] <= 0
 R501: unkn[2,1,0] - unkn[2,3,1] + forg[2,3,1] <= 0
 R502: unkn[1,2,0] + unkn[2,1,0] - unkn[2,3,1] + forg[2,3,1] >= 0
 R503: unkn[0,3,0] - unkn[1,3,1] + forg[1,3,1] = 0
 R504: - tbox[0,0,1] + forg[0,0,1] <= 0
 R505: sbox[0,0,1] - rkey[0,0,1] + tkey[0,0,1] >= 0
 R506: sbox[0,0,1] + rkey[0,0,1] - tkey[0,0,1] >= 0
 R507: - sbox[0,0,1] + rkey[0,0,1] + tkey[0,0,1] >= 0
 R508: - tbox[1,0,1] + forg[1,0,1] <= 0
 R509: sbox[1,0,1] - rkey[1,0,1] + tkey[1,0,1] >= 0
 R510: sbox[1,0,1] + rkey[1,0,1] - tkey[1,0,1] >= 0
 R511: - sbox[1,0,1] + rkey[1,0,1] + tkey[1,0,1] >= 0
 R512: - tbox[2,0,1] + forg[2,0,1] <= 0
 R513: - sbox[2,0,1] + rkey[2,0,1] = 0
 R514: - tbox[3,0,1] + forg[3,0,1] <= 0
 R515: - sbox[3,0,1] + rkey[3,0,1] = 0
 R516: - sbox[3,0,2] + rkey[0,0,1] + rkey[2,2,1] >= 0
 R517: sbox[3,0,2] + rkey[0,0,1] - rkey[2,2,1] >= 0
 R518: sbox[3,0,2] - rkey[0,0,1] + rkey[2,2,1] >= 0
 R519: - sbox[0,0,2] + sbox[3,0,2] + rkey[3,1,1] >= 0
 R520: sbox[0,0,2] + sbox[3,0,2] - rkey[3,1,1] >= 0
 R521: sbox[0,0,2] - sbox[3,0,2] + rkey[3,1,1] >= 0
 R522: - sbox[2,0,2] + rkey[1,3,1] + rkey[2,2,1] >= 0
 R523: sbox[2,0,2] + rkey[1,3,1] - rkey[2,2,1] >= 0
 R524: sbox[2,0,2] - rkey[1,3,1] + rkey[2,2,1] >= 0
 R525: sbox[1,0,2] - rkey[0,0,1] = 0
 R526: tbox[0,0,1] + tbox[2,2,1] - tbox[3,0,2] >= 0
 R527: tbox[0,0,1] - tbox[2,2,1] + tbox[3,0,2] >= 0
 R528: - tbox[0,0,1] + tbox[2,2,1] + tbox[3,0,2] >= 0
 R529: - tbox[0,0,2] + tbox[3,0,2] + tbox[3,1,1] >= 0
 R530: tbox[0,0,2] + tbox[3,0,2] - tbox[3,1,1] >= 0
 R531: tbox[0,0,2] - tbox[3,0,2] + tbox[3,1,1] >= 0
 R532: tbox[1,3,1] - tbox[2,0,2] + tbox[2,2,1] >= 0
 R533: tbox[1,3,1] + tbox[2,0,2] - tbox[2,2,1] >= 0
 R534: - tbox[1,3,1] + tbox[2,0,2] + tbox[2,2,1] >= 0
 R535: - tbox[0,0,1] + tbox[1,0,2] = 0
 R536: unkn[0,0,1] - unkn[3,0,2] + forg[3,0,2] <= 0
 R537: unkn[2,2,1] - unkn[3,0,2] + forg[3,0,2] <= 0
 R538: unkn[0,0,1] + unkn[2,2,1] - unkn[3,0,2] + forg[3,0,2] >= 0
 R539: - unkn[0,0,2] + unkn[3,0,2] + forg[0,0,2] - forg[3,0,2] <= 0
 R540: - unkn[0,0,2] + unkn[3,1,1] + forg[0,0,2] <= 0
 R541: - unkn[0,0,2] + unkn[3,0,2] + unkn[3,1,1] + forg[0,0,2]
   - forg[3,0,2] >= 0
 R542: unkn[1,3,1] - unkn[2,0,2] + forg[2,0,2] <= 0
 R543: - unkn[2,0,2] + unkn[2,2,1] + forg[2,0,2] <= 0
 R544: unkn[1,3,1] - unkn[2,0,2] + unkn[2,2,1] + forg[2,0,2] >= 0
 R545: unkn[0,0,1] - unkn[1,0,2] + forg[1,0,2] = 0
 R546: - tbox[0,1,1] + forg[0,1,1] <= 0
 R547: sbox[0,1,1] - rkey[0,1,1] + tkey[0,1,1] >= 0
 R548: sbox[0,1,1] + rkey[0,1,1] - tkey[0,1,1] >= 0
 R549: - sbox[0,1,1] + rkey[0,1,1] + tkey[0,1,1] >= 0
 R550: - tbox[1,1,1] + forg[1,1,1] <= 0
 R551: sbox[1,1,1] - rkey[1,1,1] + tkey[1,1,1] >= 0
 R552: sbox[1,1,1] + rkey[1,1,1] - tkey[1,1,1] >= 0
 R553: - sbox[1,1,1] + rkey[1,1,1] + tkey[1,1,1] >= 0
 R554: - tbox[2,1,1] + forg[2,1,1] <= 0
 R555: - sbox[2,1,1] + rkey[2,1,1] = 0
 R556: - tbox[3,1,1] + forg[3,1,1] <= 0
 R557: - sbox[3,1,1] + rkey[3,1,1] = 0
 R558: - sbox[3,1,2] + rkey[0,1,1] + rkey[2,3,1] >= 0
 R559: sbox[3,1,2] + rkey[0,1,1] - rkey[2,3,1] >= 0
 R560: sbox[3,1,2] - rkey[0,1,1] + rkey[2,3,1] >= 0
 R561: - sbox[0,1,2] + sbox[3,1,2] + rkey[3,2,1] >= 0
 R562: sbox[0,1,2] + sbox[3,1,2] - rkey[3,2,1] >= 0
 R563: sbox[0,1,2] - sbox[3,1,2] + rkey[3,2,1] >= 0
 R564: - sbox[2,1,2] + rkey[1,0,1] + rkey[2,3,1] >= 0
 R565: sbox[2,1,2] + rkey[1,0,1] - rkey[2,3,1] >= 0
 R566: sbox[2,1,2] - rkey[1,0,1] + rkey[2,3,1] >= 0
 R567: sbox[1,1,2] - rkey[0,1,1] = 0
 R568: tbox[0,1,1] + tbox[2,3,1] - tbox[3,1,2] >= 0
 R569: tbox[0,1,1] - tbox[2,3,1] + tbox[3,1,2] >= 0
 R570: - tbox[0,1,1] + tbox[2,3,1] + tbox[3,1,2] >= 0
 R571: - tbox[0,1,2] + tbox[3,1,2] + tbox[3,2,1] >= 0
 R572: tbox[0,1,2] + tbox[3,1,2] - tbox[3,2,1] >= 0
 R573: tbox[0,1,2] - tbox[3,1,2] + tbox[3,2,1] >= 0
 R574: tbox[1,0,1] - tbox[2,1,2] + tbox[2,3,1] >= 0
 R575: tbox[1,0,1] + tbox[2,1,2] - tbox[2,3,1] >= 0
 R576: - tbox[1,0,1] + tbox[2,1,2] + tbox[2,3,1] >= 0
 R577: - tbox[0,1,1] + tbox[1,1,2] = 0
 R578: unkn[0,1,1] - unkn[3,1,2] + forg[3,1,2] <= 0
 R579: unkn[2,3,1] - unkn[3,1,2] + forg[3,1,2] <= 0
 R580: unkn[0,1,1] + unkn[2,3,1] - unkn[3,1,2] + forg[3,1,2] >= 0
 R581: - unkn[0,1,2] + unkn[3,1,2] + forg[0,1,2] - forg[3,1,2] <= 0
 R582: - unkn[0,1,2] + unkn[3,2,1] + forg[0,1,2] <= 0
 R583: - unkn[0,1,2] + unkn[3,1,2] + unkn[3,2,1] + forg[0,1,2]
   - forg[3,1,2] >= 0
 R584: unkn[1,0,1] - unkn[2,1,2] + forg[2,1,2] <= 0
 R585: - unkn[2,1,2] + unkn[2,3,1] + forg[2,1,2] <= 0
 R586: unkn[1,0,1] - unkn[2,1,2] + unkn[2,3,1] + forg[2,1,2] >= 0
 R587: unkn[0,1,1] - unkn[1,1,2] + forg[1,1,2] = 0
 R588: - tbox[0,2,1] + forg[0,2,1] <= 0
 R589: sbox[0,2,1] - rkey[0,2,1] + tkey[0,2,1] >= 0
 R590: sbox[0,2,1] + rkey[0,2,1] - tkey[0,2,1] >= 0
 R591: - sbox[0,2,1] + rkey[0,2,1] + tkey[0,2,1] >= 0
 R592: - tbox[1,2,1] + forg[1,2,1] <= 0
 R593: sbox[1,2,1] - rkey[1,2,1] + tkey[1,2,1] >= 0
 R594: sbox[1,2,1] + rkey[1,2,1] - tkey[1,2,1] >= 0
 R595: - sbox[1,2,1] + rkey[1,2,1] + tkey[1,2,1] >= 0
 R596: - tbox[2,2,1] + forg[2,2,1] <= 0
 R597: - sbox[2,2,1] + rkey[2,2,1] = 0
 R598: - tbox[3,2,1] + forg[3,2,1] <= 0
 R599: - sbox[3,2,1] + rkey[3,2,1] = 0
 R600: - sbox[3,2,2] + rkey[0,2,1] + rkey[2,0,1] >= 0
 R601: sbox[3,2,2] + rkey[0,2,1] - rkey[2,0,1] >= 0
 R602: sbox[3,2,2] - rkey[0,2,1] + rkey[2,0,1] >= 0
 R603: - sbox[0,2,2] + sbox[3,2,2] + rkey[3,3,1] >= 0
 R604: sbox[0,2,2] + sbox[3,2,2] - rkey[3,3,1] >= 0
 R605: sbox[0,2,2] - sbox[3,2,2] + rkey[3,3,1] >= 0
 R606: - sbox[2,2,2] + rkey[1,1,1] + rkey[2,0,1] >= 0
 R607: sbox[2,2,2] + rkey[1,1,1] - rkey[2,0,1] >= 0
 R608: sbox[2,2,2] - rkey[1,1,1] + rkey[2,0,1] >= 0
 R609: sbox[1,2,2] - rkey[0,2,1] = 0
 R610: tbox[0,2,1] + tbox[2,0,1] - tbox[3,2,2] >= 0
 R611: tbox[0,2,1] - tbox[2,0,1] + tbox[3,2,2] >= 0
 R612: - tbox[0,2,1] + tbox[2,0,1] + tbox[3,2,2] >= 0
 R613: - tbox[0,2,2] + tbox[3,2,2] + tbox[3,3,1] >= 0
 R614: tbox[0,2,2] + tbox[3,2,2] - tbox[3,3,1] >= 0
 R615: tbox[0,2,2] - tbox[3,2,2] + tbox[3,3,1] >= 0
 R616: tbox[1,1,1] + tbox[2,0,1] - tbox[2,2,2] >= 0
 R617: tbox[1,1,1] - tbox[2,0,1] + tbox[2,2,2] >= 0
 R618: - tbox[1,1,1] + tbox[2,0,1] + tbox[2,2,2] >= 0
 R619: - tbox[0,2,1] + tbox[1,2,2] = 0
 R620: unkn[0,2,1] - unkn[3,2,2] + forg[3,2,2] <= 0
 R621: unkn[2,0,1] - unkn[3,2,2] + forg[3,2,2] <= 0
 R622: unkn[0,2,1] + unkn[2,0,1] - unkn[3,2,2] + forg[3,2,2] >= 0
 R623: - unkn[0,2,2] + unkn[3,2,2] + forg[0,2,2] - forg[3,2,2] <= 0
 R624: - unkn[0,2,2] + unkn[3,3,1] + forg[0,2,2] <= 0
 R625: - unkn[0,2,2] + unkn[3,2,2] + unkn[3,3,1] + forg[0,2,2]
   - forg[3,2,2] >= 0
 R626: unkn[1,1,1] - unkn[2,2,2] + forg[2,2,2] <= 0
 R627: unkn[2,0,1] - unkn[2,2,2] + forg[2,2,2] <= 0
 R628: unkn[1,1,1] + unkn[2,0,1] - unkn[2,2,2] + forg[2,2,2] >= 0
 R629: unkn[0,2,1] - unkn[1,2,2] + forg[1,2,2] = 0
 R630: - tbox[0,3,1] + forg[0,3,1] <= 0
 R631: sbox[0,3,1] - rkey[0,3,1] + tkey[0,3,1] >= 0
 R632: sbox[0,3,1] + rkey[0,3,1] - tkey[0,3,1] >= 0
 R633: - sbox[0,3,1] + rkey[0,3,1] + tkey[0,3,1] >= 0
 R634: - tbox[1,3,1] + forg[1,3,1] <= 0
 R635: sbox[1,3,1] - rkey[1,3,1] + tkey[1,3,1] >= 0
 R636: sbox[1,3,1] + rkey[1,3,1] - tkey[1,3,1] >= 0
 R637: - sbox[1,3,1] + rkey[1,3,1] + tkey[1,3,1] >= 0
 R638: - tbox[2,3,1] + forg[2,3,1] <= 0
 R639: - sbox[2,3,1] + rkey[2,3,1] = 0
 R640: - tbox[3,3,1] + forg[3,3,1] <= 0
 R641: - sbox[3,3,1] + rkey[3,3,1] = 0
 R642: - sbox[3,3,2] + rkey[0,3,1] + rkey[2,1,1] >= 0
 R643: sbox[3,3,2] + rkey[0,3,1] - rkey[2,1,1] >= 0
 R644: sbox[3,3,2] - rkey[0,3,1] + rkey[2,1,1] >= 0
 R645: - sbox[0,3,2] + sbox[3,3,2] + rkey[3,0,1] >= 0
 R646: sbox[0,3,2] + sbox[3,3,2] - rkey[3,0,1] >= 0
 R647: sbox[0,3,2] - sbox[3,3,2] + rkey[3,0,1] >= 0
 R648: - sbox[2,3,2] + rkey[1,2,1] + rkey[2,1,1] >= 0
 R649: sbox[2,3,2] + rkey[1,2,1] - rkey[2,1,1] >= 0
 R650: sbox[2,3,2] - rkey[1,2,1] + rkey[2,1,1] >= 0
 R651: sbox[1,3,2] - rkey[0,3,1] = 0
 R652: tbox[0,3,1] + tbox[2,1,1] - tbox[3,3,2] >= 0
 R653: tbox[0,3,1] - tbox[2,1,1] + tbox[3,3,2] >= 0
 R654: - tbox[0,3,1] + tbox[2,1,1] + tbox[3,3,2] >= 0
 R655: - tbox[0,3,2] + tbox[3,0,1] + tbox[3,3,2] >= 0
 R656: tbox[0,3,2] - tbox[3,0,1] + tbox[3,3,2] >= 0
 R657: tbox[0,3,2] + tbox[3,0,1] - tbox[3,3,2] >= 0
 R658: tbox[1,2,1] + tbox[2,1,1] - tbox[2,3,2] >= 0
 R659: tbox[1,2,1] - tbox[2,1,1] + tbox[2,3,2] >= 0
 R660: - tbox[1,2,1] + tbox[2,1,1] + tbox[2,3,2] >= 0
 R661: - tbox[0,3,1] + tbox[1,3,2] = 0
 R662: unkn[0,3,1] - unkn[3,3,2] + forg[3,3,2] <= 0
 R663: unkn[2,1,1] - unkn[3,3,2] + forg[3,3,2] <= 0
 R664: unkn[0,3,1] + unkn[2,1,1] - unkn[3,3,2] + forg[3,3,2] >= 0
 R665: - unkn[0,3,2] + unkn[3,3,2] + forg[0,3,2] - forg[3,3,2] <= 0
 R666: - unkn[0,3,2] + unkn[3,0,1] + forg[0,3,2] <= 0
 R667: - unkn[0,3,2] + unkn[3,0,1] + unkn[3,3,2] + forg[0,3,2]
   - forg[3,3,2] >= 0
 R668: unkn[1,2,1] - unkn[2,3,2] + forg[2,3,2] <= 0
 R669: unkn[2,1,1] - unkn[2,3,2] + forg[2,3,2] <= 0
 R670: unkn[1,2,1] + unkn[2,1,1] - unkn[2,3,2] + forg[2,3,2] >= 0
 R671: unkn[0,3,1] - unkn[1,3,2] + forg[1,3,2] = 0
 R672: - tbox[0,0,2] + forg[0,0,2] <= 0
 R673: sbox[0,0,2] - rkey[0,0,2] + tkey[0,0,2] >= 0
 R674: sbox[0,0,2] + rkey[0,0,2] - tkey[0,0,2] >= 0
 R675: - sbox[0,0,2] + rkey[0,0,2] + tkey[0,0,2] >= 0
 R676: - tbox[1,0,2] + forg[1,0,2] <= 0
 R677: sbox[1,0,2] - rkey[1,0,2] + tkey[1,0,2] >= 0
 R678: sbox[1,0,2] + rkey[1,0,2] - tkey[1,0,2] >= 0
 R679: - sbox[1,0,2] + rkey[1,0,2] + tkey[1,0,2] >= 0
 R680: - tbox[2,0,2] + forg[2,0,2] <= 0
 R681: - sbox[2,0,2] + rkey[2,0,2] = 0
 R682: - tbox[3,0,2] + forg[3,0,2] <= 0
 R683: - sbox[3,0,2] + rkey[3,0,2] = 0
 R684: - sbox[3,0,3] + rkey[0,0,2] + rkey[2,2,2] >= 0
 R685: sbox[3,0,3] + rkey[0,0,2] - rkey[2,2,2] >= 0
 R686: sbox[3,0,3] - rkey[0,0,2] + rkey[2,2,2] >= 0
 R687: - sbox[0,0,3] + sbox[3,0,3] + rkey[3,1,2] >= 0
 R688: sbox[0,0,3] + sbox[3,0,3] - rkey[3,1,2] >= 0
 R689: sbox[0,0,3] - sbox[3,0,3] + rkey[3,1,2] >= 0
 R690: - sbox[2,0,3] + rkey[1,3,2] + rkey[2,2,2] >= 0
 R691: sbox[2,0,3] + rkey[1,3,2] - rkey[2,2,2] >= 0
 R692: sbox[2,0,3] - rkey[1,3,2] + rkey[2,2,2] >= 0
 R693: sbox[1,0,3] - rkey[0,0,2] = 0
 R694: tbox[0,0,2] + tbox[2,2,2] - tbox[3,0,3] >= 0
 R695: tbox[0,0,2] - tbox[2,2,2] + tbox[3,0,3] >= 0
 R696: - tbox[0,0,2] + tbox[2,2,2] + tbox[3,0,3] >= 0
 R697: - tbox[0,0,3] + tbox[3,0,3] + tbox[3,1,2] >= 0
 R698: tbox[0,0,3] + tbox[3,0,3] - tbox[3,1,2] >= 0
 R699: tbox[0,0,3] - tbox[3,0,3] + tbox[3,1,2] >= 0
 R700: tbox[1,3,2] - tbox[2,0,3] + tbox[2,2,2] >= 0
 R701: tbox[1,3,2] + tbox[2,0,3] - tbox[2,2,2] >= 0
 R702: - tbox[1,3,2] + tbox[2,0,3] + tbox[2,2,2] >= 0
 R703: - tbox[0,0,2] + tbox[1,0,3] = 0
 R704: unkn[0,0,2] - unkn[3,0,3] + forg[3,0,3] <= 0
 R705: unkn[2,2,2] - unkn[3,0,3] + forg[3,0,3] <= 0
 R706: unkn[0,0,2] + unkn[2,2,2] - unkn[3,0,3] + forg[3,0,3] >= 0
 R707: - unkn[0,0,3] + unkn[3,0,3] + forg[0,0,3] - forg[3,0,3] <= 0
 R708: - unkn[0,0,3] + unkn[3,1,2] + forg[0,0,3] <= 0
 R709: - unkn[0,0,3] + unkn[3,0,3] + unkn[3,1,2] + forg[0,0,3]
   - forg[3,0,3] >= 0
 R710: unkn[1,3,2] - unkn[2,0,3] + forg[2,0,3] <= 0
 R711: - unkn[2,0,3] + unkn[2,2,2] + forg[2,0,3] <= 0
 R712: unkn[1,3,2] - unkn[2,0,3] + unkn[2,2,2] + forg[2,0,3] >= 0
 R713: unkn[0,0,2] - unkn[1,0,3] + forg[1,0,3] = 0
 R714: - tbox[0,1,2] + forg[0,1,2] <= 0
 R715: sbox[0,1,2] - rkey[0,1,2] + tkey[0,1,2] >= 0
 R716: sbox[0,1,2] + rkey[0,1,2] - tkey[0,1,2] >= 0
 R717: - sbox[0,1,2] + rkey[0,1,2] + tkey[0,1,2] >= 0
 R718: - tbox[1,1,2] + forg[1,1,2] <= 0
 R719: sbox[1,1,2] - rkey[1,1,2] + tkey[1,1,2] >= 0
 R720: sbox[1,1,2] + rkey[1,1,2] - tkey[1,1,2] >= 0
 R721: - sbox[1,1,2] + rkey[1,1,2] + tkey[1,1,2] >= 0
 R722: - tbox[2,1,2] + forg[2,1,2] <= 0
 R723: - sbox[2,1,2] + rkey[2,1,2] = 0
 R724: - tbox[3,1,2] + forg[3,1,2] <= 0
 R725: - sbox[3,1,2] + rkey[3,1,2] = 0
 R726: - sbox[3,1,3] + rkey[0,1,2] + rkey[2,3,2] >= 0
 R727: sbox[3,1,3] + rkey[0,1,2] - rkey[2,3,2] >= 0
 R728: sbox[3,1,3] - rkey[0,1,2] + rkey[2,3,2] >= 0
 R729: - sbox[0,1,3] + sbox[3,1,3] + rkey[3,2,2] >= 0
 R730: sbox[0,1,3] + sbox[3,1,3] - rkey[3,2,2] >= 0
 R731: sbox[0,1,3] - sbox[3,1,3] + rkey[3,2,2] >= 0
 R732: - sbox[2,1,3] + rkey[1,0,2] + rkey[2,3,2] >= 0
 R733: sbox[2,1,3] + rkey[1,0,2] - rkey[2,3,2] >= 0
 R734: sbox[2,1,3] - rkey[1,0,2] + rkey[2,3,2] >= 0
 R735: sbox[1,1,3] - rkey[0,1,2] = 0
 R736: tbox[0,1,2] + tbox[2,3,2] - tbox[3,1,3] >= 0
 R737: tbox[0,1,2] - tbox[2,3,2] + tbox[3,1,3] >= 0
 R738: - tbox[0,1,2] + tbox[2,3,2] + tbox[3,1,3] >= 0
 R739: - tbox[0,1,3] + tbox[3,1,3] + tbox[3,2,2] >= 0
 R740: tbox[0,1,3] + tbox[3,1,3] - tbox[3,2,2] >= 0
 R741: tbox[0,1,3] - tbox[3,1,3] + tbox[3,2,2] >= 0
 R742: tbox[1,0,2] - tbox[2,1,3] + tbox[2,3,2] >= 0
 R743: tbox[1,0,2] + tbox[2,1,3] - tbox[2,3,2] >= 0
 R744: - tbox[1,0,2] + tbox[2,1,3] + tbox[2,3,2] >= 0
 R745: - tbox[0,1,2] + tbox[1,1,3] = 0
 R746: unkn[0,1,2] - unkn[3,1,3] + forg[3,1,3] <= 0
 R747: unkn[2,3,2] - unkn[3,1,3] + forg[3,1,3] <= 0
 R748: unkn[0,1,2] + unkn[2,3,2] - unkn[3,1,3] + forg[3,1,3] >= 0
 R749: - unkn[0,1,3] + unkn[3,1,3] + forg[0,1,3] - forg[3,1,3] <= 0
 R750: - unkn[0,1,3] + unkn[3,2,2] + forg[0,1,3] <= 0
 R751: - unkn[0,1,3] + unkn[3,1,3] + unkn[3,2,2] + forg[0,1,3]
   - forg[3,1,3] >= 0
 R752: unkn[1,0,2] - unkn[2,1,3] + forg[2,1,3] <= 0
 R753: - unkn[2,1,3] + unkn[2,3,2] + forg[2,1,3] <= 0
 R754: unkn[1,0,2] - unkn[2,1,3] + unkn[2,3,2] + forg[2,1,3] >= 0
 R755: unkn[0,1,2] - unkn[1,1,3] + forg[1,1,3] = 0
 R756: - tbox[0,2,2] + forg[0,2,2] <= 0
 R757: sbox[0,2,2] - rkey[0,2,2] + tkey[0,2,2] >= 0
 R758: sbox[0,2,2] + rkey[0,2,2] - tkey[0,2,2] >= 0
 R759: - sbox[0,2,2] + rkey[0,2,2] + tkey[0,2,2] >= 0
 R760: - tbox[1,2,2] + forg[1,2,2] <= 0
 R761: sbox[1,2,2] - rkey[1,2,2] + tkey[1,2,2] >= 0
 R762: sbox[1,2,2] + rkey[1,2,2] - tkey[1,2,2] >= 0
 R763: - sbox[1,2,2] + rkey[1,2,2] + tkey[1,2,2] >= 0
 R764: - tbox[2,2,2] + forg[2,2,2] <= 0
 R765: - sbox[2,2,2] + rkey[2,2,2] = 0
 R766: - tbox[3,2,2] + forg[3,2,2] <= 0
 R767: - sbox[3,2,2] + rkey[3,2,2] = 0
 R768: - sbox[3,2,3] + rkey[0,2,2] + rkey[2,0,2] >= 0
 R769: sbox[3,2,3] + rkey[0,2,2] - rkey[2,0,2] >= 0
 R770: sbox[3,2,3] - rkey[0,2,2] + rkey[2,0,2] >= 0
 R771: - sbox[0,2,3] + sbox[3,2,3] + rkey[3,3,2] >= 0
 R772: sbox[0,2,3] + sbox[3,2,3] - rkey[3,3,2] >= 0
 R773: sbox[0,2,3] - sbox[3,2,3] + rkey[3,3,2] >= 0
 R774: - sbox[2,2,3] + rkey[1,1,2] + rkey[2,0,2] >= 0
 R775: sbox[2,2,3] + rkey[1,1,2] - rkey[2,0,2] >= 0
 R776: sbox[2,2,3] - rkey[1,1,2] + rkey[2,0,2] >= 0
 R777: sbox[1,2,3] - rkey[0,2,2] = 0
 R778: tbox[0,2,2] + tbox[2,0,2] - tbox[3,2,3] >= 0
 R779: tbox[0,2,2] - tbox[2,0,2] + tbox[3,2,3] >= 0
 R780: - tbox[0,2,2] + tbox[2,0,2] + tbox[3,2,3] >= 0
 R781: - tbox[0,2,3] + tbox[3,2,3] + tbox[3,3,2] >= 0
 R782: tbox[0,2,3] + tbox[3,2,3] - tbox[3,3,2] >= 0
 R783: tbox[0,2,3] - tbox[3,2,3] + tbox[3,3,2] >= 0
 R784: tbox[1,1,2] + tbox[2,0,2] - tbox[2,2,3] >= 0
 R785: tbox[1,1,2] - tbox[2,0,2] + tbox[2,2,3] >= 0
 R786: - tbox[1,1,2] + tbox[2,0,2] + tbox[2,2,3] >= 0
 R787: - tbox[0,2,2] + tbox[1,2,3] = 0
 R788: unkn[0,2,2] - unkn[3,2,3] + forg[3,2,3] <= 0
 R789: unkn[2,0,2] - unkn[3,2,3] + forg[3,2,3] <= 0
 R790: unkn[0,2,2] + unkn[2,0,2] - unkn[3,2,3] + forg[3,2,3] >= 0
 R791: - unkn[0,2,3] + unkn[3,2,3] + forg[0,2,3] - forg[3,2,3] <= 0
 R792: - unkn[0,2,3] + unkn[3,3,2] + forg[0,2,3] <= 0
 R793: - unkn[0,2,3] + unkn[3,2,3] + unkn[3,3,2] + forg[0,2,3]
   - forg[3,2,3] >= 0
 R794: unkn[1,1,2] - unkn[2,2,3] + forg[2,2,3] <= 0
 R795: unkn[2,0,2] - unkn[2,2,3] + forg[2,2,3] <= 0
 R796: unkn[1,1,2] + unkn[2,0,2] - unkn[2,2,3] + forg[2,2,3] >= 0
 R797: unkn[0,2,2] - unkn[1,2,3] + forg[1,2,3] = 0
 R798: - tbox[0,3,2] + forg[0,3,2] <= 0
 R799: sbox[0,3,2] - rkey[0,3,2] + tkey[0,3,2] >= 0
 R800: sbox[0,3,2] + rkey[0,3,2] - tkey[0,3,2] >= 0
 R801: - sbox[0,3,2] + rkey[0,3,2] + tkey[0,3,2] >= 0
 R802: - tbox[1,3,2] + forg[1,3,2] <= 0
 R803: sbox[1,3,2] - rkey[1,3,2] + tkey[1,3,2] >= 0
 R804: sbox[1,3,2] + rkey[1,3,2] - tkey[1,3,2] >= 0
 R805: - sbox[1,3,2] + rkey[1,3,2] + tkey[1,3,2] >= 0
 R806: - tbox[2,3,2] + forg[2,3,2] <= 0
 R807: - sbox[2,3,2] + rkey[2,3,2] = 0
 R808: - tbox[3,3,2] + forg[3,3,2] <= 0
 R809: - sbox[3,3,2] + rkey[3,3,2] = 0
 R810: - sbox[3,3,3] + rkey[0,3,2] + rkey[2,1,2] >= 0
 R811: sbox[3,3,3] + rkey[0,3,2] - rkey[2,1,2] >= 0
 R812: sbox[3,3,3] - rkey[0,3,2] + rkey[2,1,2] >= 0
 R813: - sbox[0,3,3] + sbox[3,3,3] + rkey[3,0,2] >= 0
 R814: sbox[0,3,3] + sbox[3,3,3] - rkey[3,0,2] >= 0
 R815: sbox[0,3,3] - sbox[3,3,3] + rkey[3,0,2] >= 0
 R816: - sbox[2,3,3] + rkey[1,2,2] + rkey[2,1,2] >= 0
 R817: sbox[2,3,3] + rkey[1,2,2] - rkey[2,1,2] >= 0
 R818: sbox[2,3,3] - rkey[1,2,2] + rkey[2,1,2] >= 0
 R819: sbox[1,3,3] - rkey[0,3,2] = 0
 R820: tbox[0,3,2] + tbox[2,1,2] - tbox[3,3,3] >= 0
 R821: tbox[0,3,2] - tbox[2,1,2] + tbox[3,3,3] >= 0
 R822: - tbox[0,3,2] + tbox[2,1,2] + tbox[3,3,3] >= 0
 R823: - tbox[0,3,3] + tbox[3,0,2] + tbox[3,3,3] >= 0
 R824: tbox[0,3,3] - tbox[3,0,2] + tbox[3,3,3] >= 0
 R825: tbox[0,3,3] + tbox[3,0,2] - tbox[3,3,3] >= 0
 R826: tbox[1,2,2] + tbox[2,1,2] - tbox[2,3,3] >= 0
 R827: tbox[1,2,2] - tbox[2,1,2] + tbox[2,3,3] >= 0
 R828: - tbox[1,2,2] + tbox[2,1,2] + tbox[2,3,3] >= 0
 R829: - tbox[0,3,2] + tbox[1,3,3] = 0
 R830: unkn[0,3,2] - unkn[3,3,3] + forg[3,3,3] <= 0
 R831: unkn[2,1,2] - unkn[3,3,3] + forg[3,3,3] <= 0
 R832: unkn[0,3,2] + unkn[2,1,2] - unkn[3,3,3] + forg[3,3,3] >= 0
 R833: - unkn[0,3,3] + unkn[3,3,3] + forg[0,3,3] - forg[3,3,3] <= 0
 R834: - unkn[0,3,3] + unkn[3,0,2] + forg[0,3,3] <= 0
 R835: - unkn[0,3,3] + unkn[3,0,2] + unkn[3,3,3] + forg[0,3,3]
   - forg[3,3,3] >= 0
 R836: unkn[1,2,2] - unkn[2,3,3] + forg[2,3,3] <= 0
 R837: unkn[2,1,2] - unkn[2,3,3] + forg[2,3,3] <= 0
 R838: unkn[1,2,2] + unkn[2,1,2] - unkn[2,3,3] + forg[2,3,3] >= 0
 R839: unkn[0,3,2] - unkn[1,3,3] + forg[1,3,3] = 0
 R840: - tbox[0,0,3] + forg[0,0,3] <= 0
 R841: sbox[0,0,3] - rkey[0,0,3] + tkey[0,0,3] >= 0
 R842: sbox[0,0,3] + rkey[0,0,3] - tkey[0,0,3] >= 0
 R843: - sbox[0,0,3] + rkey[0,0,3] + tkey[0,0,3] >= 0
 R844: - tbox[1,0,3] + forg[1,0,3] <= 0
 R845: sbox[1,0,3] - rkey[1,0,3] + tkey[1,0,3] >= 0
 R846: sbox[1,0,3] + rkey[1,0,3] - tkey[1,0,3] >= 0
 R847: - sbox[1,0,3] + rkey[1,0,3] + tkey[1,0,3] >= 0
 R848: - tbox[2,0,3] + forg[2,0,3] <= 0
 R849: - sbox[2,0,3] + rkey[2,0,3] = 0
 R850: - tbox[3,0,3] + forg[3,0,3] <= 0
 R851: - sbox[3,0,3] + rkey[3,0,3] = 0
 R852: - sbox[3,0,4] + rkey[0,0,3] + rkey[2,2,3] >= 0
 R853: sbox[3,0,4] + rkey[0,0,3] - rkey[2,2,3] >= 0
 R854: sbox[3,0,4] - rkey[0,0,3] + rkey[2,2,3] >= 0
 R855: - sbox[0,0,4] + sbox[3,0,4] + rkey[3,1,3] >= 0
 R856: sbox[0,0,4] + sbox[3,0,4] - rkey[3,1,3] >= 0
 R857: sbox[0,0,4] - sbox[3,0,4] + rkey[3,1,3] >= 0
 R858: - sbox[2,0,4] + rkey[1,3,3] + rkey[2,2,3] >= 0
 R859: sbox[2,0,4] + rkey[1,3,3] - rkey[2,2,3] >= 0
 R860: sbox[2,0,4] - rkey[1,3,3] + rkey[2,2,3] >= 0
 R861: sbox[1,0,4] - rkey[0,0,3] = 0
 R862: tbox[0,0,3] + tbox[2,2,3] - tbox[3,0,4] >= 0
 R863: tbox[0,0,3] - tbox[2,2,3] + tbox[3,0,4] >= 0
 R864: - tbox[0,0,3] + tbox[2,2,3] + tbox[3,0,4] >= 0
 R865: - tbox[0,0,4] + tbox[3,0,4] + tbox[3,1,3] >= 0
 R866: tbox[0,0,4] + tbox[3,0,4] - tbox[3,1,3] >= 0
 R867: tbox[0,0,4] - tbox[3,0,4] + tbox[3,1,3] >= 0
 R868: tbox[1,3,3] - tbox[2,0,4] + tbox[2,2,3] >= 0
 R869: tbox[1,3,3] + tbox[2,0,4] - tbox[2,2,3] >= 0
 R870: - tbox[1,3,3] + tbox[2,0,4] + tbox[2,2,3] >= 0
 R871: - tbox[0,0,3] + tbox[1,0,4] = 0
 R872: unkn[0,0,3] - unkn[3,0,4] + forg[3,0,4] <= 0
 R873: unkn[2,2,3] - unkn[3,0,4] + forg[3,0,4] <= 0
 R874: unkn[0,0,3] + unkn[2,2,3] - unkn[3,0,4] + forg[3,0,4] >= 0
 R875: - unkn[0,0,4] + unkn[3,0,4] + forg[0,0,4] - forg[3,0,4] <= 0
 R876: - unkn[0,0,4] + unkn[3,1,3] + forg[0,0,4] <= 0
 R877: - unkn[0,0,4] + unkn[3,0,4] + unkn[3,1,3] + forg[0,0,4]
   - forg[3,0,4] >= 0
 R878: unkn[1,3,3] - unkn[2,0,4] + forg[2,0,4] <= 0
 R879: - unkn[2,0,4] + unkn[2,2,3] + forg[2,0,4] <= 0
 R880: unkn[1,3,3] - unkn[2,0,4] + unkn[2,2,3] + forg[2,0,4] >= 0
 R881: unkn[0,0,3] - unkn[1,0,4] + forg[1,0,4] = 0
 R882: - tbox[0,1,3] + forg[0,1,3] <= 0
 R883: sbox[0,1,3] - rkey[0,1,3] + tkey[0,1,3] >= 0
 R884: sbox[0,1,3] + rkey[0,1,3] - tkey[0,1,3] >= 0
 R885: - sbox[0,1,3] + rkey[0,1,3] + tkey[0,1,3] >= 0
 R886: - tbox[1,1,3] + forg[1,1,3] <= 0
 R887: sbox[1,1,3] - rkey[1,1,3] + tkey[1,1,3] >= 0
 R888: sbox[1,1,3] + rkey[1,1,3] - tkey[1,1,3] >= 0
 R889: - sbox[1,1,3] + rkey[1,1,3] + tkey[1,1,3] >= 0
 R890: - tbox[2,1,3] + forg[2,1,3] <= 0
 R891: - sbox[2,1,3] + rkey[2,1,3] = 0
 R892: - tbox[3,1,3] + forg[3,1,3] <= 0
 R893: - sbox[3,1,3] + rkey[3,1,3] = 0
 R894: - sbox[3,1,4] + rkey[0,1,3] + rkey[2,3,3] >= 0
 R895: sbox[3,1,4] + rkey[0,1,3] - rkey[2,3,3] >= 0
 R896: sbox[3,1,4] - rkey[0,1,3] + rkey[2,3,3] >= 0
 R897: - sbox[0,1,4] + sbox[3,1,4] + rkey[3,2,3] >= 0
 R898: sbox[0,1,4] + sbox[3,1,4] - rkey[3,2,3] >= 0
 R899: sbox[0,1,4] - sbox[3,1,4] + rkey[3,2,3] >= 0
 R900: - sbox[2,1,4] + rkey[1,0,3] + rkey[2,3,3] >= 0
 R901: sbox[2,1,4] + rkey[1,0,3] - rkey[2,3,3] >= 0
 R902: sbox[2,1,4] - rkey[1,0,3] + rkey[2,3,3] >= 0
 R903: sbox[1,1,4] - rkey[0,1,3] = 0
 R904: tbox[0,1,3] + tbox[2,3,3] - tbox[3,1,4] >= 0
 R905: tbox[0,1,3] - tbox[2,3,3] + tbox[3,1,4] >= 0
 R906: - tbox[0,1,3] + tbox[2,3,3] + tbox[3,1,4] >= 0
 R907: - tbox[0,1,4] + tbox[3,1,4] + tbox[3,2,3] >= 0
 R908: tbox[0,1,4] + tbox[3,1,4] - tbox[3,2,3] >= 0
 R909: tbox[0,1,4] - tbox[3,1,4] + tbox[3,2,3] >= 0
 R910: tbox[1,0,3] - tbox[2,1,4] + tbox[2,3,3] >= 0
 R911: tbox[1,0,3] + tbox[2,1,4] - tbox[2,3,3] >= 0
 R912: - tbox[1,0,3] + tbox[2,1,4] + tbox[2,3,3] >= 0
 R913: - tbox[0,1,3] + tbox[1,1,4] = 0
 R914: unkn[0,1,3] - unkn[3,1,4] + forg[3,1,4] <= 0
 R915: unkn[2,3,3] - unkn[3,1,4] + forg[3,1,4] <= 0
 R916: unkn[0,1,3] + unkn[2,3,3] - unkn[3,1,4] + forg[3,1,4] >= 0
 R917: - unkn[0,1,4] + unkn[3,1,4] + forg[0,1,4] - forg[3,1,4] <= 0
 R918: - unkn[0,1,4] + unkn[3,2,3] + forg[0,1,4] <= 0
 R919: - unkn[0,1,4] + unkn[3,1,4] + unkn[3,2,3] + forg[0,1,4]
   - forg[3,1,4] >= 0
 R920: unkn[1,0,3] - unkn[2,1,4] + forg[2,1,4] <= 0
 R921: - unkn[2,1,4] + unkn[2,3,3] + forg[2,1,4] <= 0
 R922: unkn[1,0,3] - unkn[2,1,4] + unkn[2,3,3] + forg[2,1,4] >= 0
 R923: unkn[0,1,3] - unkn[1,1,4] + forg[1,1,4] = 0
 R924: - tbox[0,2,3] + forg[0,2,3] <= 0
 R925: sbox[0,2,3] - rkey[0,2,3] + tkey[0,2,3] >= 0
 R926: sbox[0,2,3] + rkey[0,2,3] - tkey[0,2,3] >= 0
 R927: - sbox[0,2,3] + rkey[0,2,3] + tkey[0,2,3] >= 0
 R928: - tbox[1,2,3] + forg[1,2,3] <= 0
 R929: sbox[1,2,3] - rkey[1,2,3] + tkey[1,2,3] >= 0
 R930: sbox[1,2,3] + rkey[1,2,3] - tkey[1,2,3] >= 0
 R931: - sbox[1,2,3] + rkey[1,2,3] + tkey[1,2,3] >= 0
 R932: - tbox[2,2,3] + forg[2,2,3] <= 0
 R933: - sbox[2,2,3] + rkey[2,2,3] = 0
 R934: - tbox[3,2,3] + forg[3,2,3] <= 0
 R935: - sbox[3,2,3] + rkey[3,2,3] = 0
 R936: - sbox[3,2,4] + rkey[0,2,3] + rkey[2,0,3] >= 0
 R937: sbox[3,2,4] + rkey[0,2,3] - rkey[2,0,3] >= 0
 R938: sbox[3,2,4] - rkey[0,2,3] + rkey[2,0,3] >= 0
 R939: - sbox[0,2,4] + sbox[3,2,4] + rkey[3,3,3] >= 0
 R940: sbox[0,2,4] + sbox[3,2,4] - rkey[3,3,3] >= 0
 R941: sbox[0,2,4] - sbox[3,2,4] + rkey[3,3,3] >= 0
 R942: - sbox[2,2,4] + rkey[1,1,3] + rkey[2,0,3] >= 0
 R943: sbox[2,2,4] + rkey[1,1,3] - rkey[2,0,3] >= 0
 R944: sbox[2,2,4] - rkey[1,1,3] + rkey[2,0,3] >= 0
 R945: sbox[1,2,4] - rkey[0,2,3] = 0
 R946: tbox[0,2,3] + tbox[2,0,3] - tbox[3,2,4] >= 0
 R947: tbox[0,2,3] - tbox[2,0,3] + tbox[3,2,4] >= 0
 R948: - tbox[0,2,3] + tbox[2,0,3] + tbox[3,2,4] >= 0
 R949: - tbox[0,2,4] + tbox[3,2,4] + tbox[3,3,3] >= 0
 R950: tbox[0,2,4] + tbox[3,2,4] - tbox[3,3,3] >= 0
 R951: tbox[0,2,4] - tbox[3,2,4] + tbox[3,3,3] >= 0
 R952: tbox[1,1,3] + tbox[2,0,3] - tbox[2,2,4] >= 0
 R953: tbox[1,1,3] - tbox[2,0,3] + tbox[2,2,4] >= 0
 R954: - tbox[1,1,3] + tbox[2,0,3] + tbox[2,2,4] >= 0
 R955: - tbox[0,2,3] + tbox[1,2,4] = 0
 R956: unkn[0,2,3] - unkn[3,2,4] + forg[3,2,4] <= 0
 R957: unkn[2,0,3] - unkn[3,2,4] + forg[3,2,4] <= 0
 R958: unkn[0,2,3] + unkn[2,0,3] - unkn[3,2,4] + forg[3,2,4] >= 0
 R959: - unkn[0,2,4] + unkn[3,2,4] + forg[0,2,4] - forg[3,2,4] <= 0
 R960: - unkn[0,2,4] + unkn[3,3,3] + forg[0,2,4] <= 0
 R961: - unkn[0,2,4] + unkn[3,2,4] + unkn[3,3,3] + forg[0,2,4]
   - forg[3,2,4] >= 0
 R962: unkn[1,1,3] - unkn[2,2,4] + forg[2,2,4] <= 0
 R963: unkn[2,0,3] - unkn[2,2,4] + forg[2,2,4] <= 0
 R964: unkn[1,1,3] + unkn[2,0,3] - unkn[2,2,4] + forg[2,2,4] >= 0
 R965: unkn[0,2,3] - unkn[1,2,4] + forg[1,2,4] = 0
 R966: - tbox[0,3,3] + forg[0,3,3] <= 0
 R967: sbox[0,3,3] - rkey[0,3,3] + tkey[0,3,3] >= 0
 R968: sbox[0,3,3] + rkey[0,3,3] - tkey[0,3,3] >= 0
 R969: - sbox[0,3,3] + rkey[0,3,3] + tkey[0,3,3] >= 0
 R970: - tbox[1,3,3] + forg[1,3,3] <= 0
 R971: sbox[1,3,3] - rkey[1,3,3] + tkey[1,3,3] >= 0
 R972: sbox[1,3,3] + rkey[1,3,3] - tkey[1,3,3] >= 0
 R973: - sbox[1,3,3] + rkey[1,3,3] + tkey[1,3,3] >= 0
 R974: - tbox[2,3,3] + forg[2,3,3] <= 0
 R975: - sbox[2,3,3] + rkey[2,3,3] = 0
 R976: - tbox[3,3,3] + forg[3,3,3] <= 0
 R977: - sbox[3,3,3] + rkey[3,3,3] = 0
 R978: - sbox[3,3,4] + rkey[0,3,3] + rkey[2,1,3] >= 0
 R979: sbox[3,3,4] + rkey[0,3,3] - rkey[2,1,3] >= 0
 R980: sbox[3,3,4] - rkey[0,3,3] + rkey[2,1,3] >= 0
 R981: - sbox[0,3,4] + sbox[3,3,4] + rkey[3,0,3] >= 0
 R982: sbox[0,3,4] + sbox[3,3,4] - rkey[3,0,3] >= 0
 R983: sbox[0,3,4] - sbox[3,3,4] + rkey[3,0,3] >= 0
 R984: - sbox[2,3,4] + rkey[1,2,3] + rkey[2,1,3] >= 0
 R985: sbox[2,3,4] + rkey[1,2,3] - rkey[2,1,3] >= 0
 R986: sbox[2,3,4] - rkey[1,2,3] + rkey[2,1,3] >= 0
 R987: sbox[1,3,4] - rkey[0,3,3] = 0
 R988: tbox[0,3,3] + tbox[2,1,3] - tbox[3,3,4] >= 0
 R989: tbox[0,3,3] - tbox[2,1,3] + tbox[3,3,4] >= 0
 R990: - tbox[0,3,3] + tbox[2,1,3] + tbox[3,3,4] >= 0
 R991: - tbox[0,3,4] + tbox[3,0,3] + tbox[3,3,4] >= 0
 R992: tbox[0,3,4] - tbox[3,0,3] + tbox[3,3,4] >= 0
 R993: tbox[0,3,4] + tbox[3,0,3] - tbox[3,3,4] >= 0
 R994: tbox[1,2,3] + tbox[2,1,3] - tbox[2,3,4] >= 0
 R995: tbox[1,2,3] - tbox[2,1,3] + tbox[2,3,4] >= 0
 R996: - tbox[1,2,3] + tbox[2,1,3] + tbox[2,3,4] >= 0
 R997: - tbox[0,3,3] + tbox[1,3,4] = 0
 R998: unkn[0,3,3] - unkn[3,3,4] + forg[3,3,4] <= 0
 R999: unkn[2,1,3] - unkn[3,3,4] + forg[3,3,4] <= 0
 R1000: unkn[0,3,3] + unkn[2,1,3] - unkn[3,3,4] + forg[3,3,4] >= 0
 R1001: - unkn[0,3,4] + unkn[3,3,4] + forg[0,3,4] - forg[3,3,4] <= 0
 R1002: - unkn[0,3,4] + unkn[3,0,3] + forg[0,3,4] <= 0
 R1003: - unkn[0,3,4] + unkn[3,0,3] + unkn[3,3,4] + forg[0,3,4]
   - forg[3,3,4] >= 0
 R1004: unkn[1,2,3] - unkn[2,3,4] + forg[2,3,4] <= 0
 R1005: unkn[2,1,3] - unkn[2,3,4] + forg[2,3,4] <= 0
 R1006: unkn[1,2,3] + unkn[2,1,3] - unkn[2,3,4] + forg[2,3,4] >= 0
 R1007: unkn[0,3,3] - unkn[1,3,4] + forg[1,3,4] = 0
 R1008: - tbox[0,0,4] + forg[0,0,4] <= 0
 R1009: sbox[0,0,4] - rkey[0,0,4] + tkey[0,0,4] >= 0
 R1010: sbox[0,0,4] + rkey[0,0,4] - tkey[0,0,4] >= 0
 R1011: - sbox[0,0,4] + rkey[0,0,4] + tkey[0,0,4] >= 0
 R1012: - tbox[1,0,4] + forg[1,0,4] <= 0
 R1013: sbox[1,0,4] - rkey[1,0,4] + tkey[1,0,4] >= 0
 R1014: sbox[1,0,4] + rkey[1,0,4] - tkey[1,0,4] >= 0
 R1015: - sbox[1,0,4] + rkey[1,0,4] + tkey[1,0,4] >= 0
 R1016: - tbox[2,0,4] + forg[2,0,4] <= 0
 R1017: - sbox[2,0,4] + rkey[2,0,4] = 0
 R1018: - tbox[3,0,4] + forg[3,0,4] <= 0
 R1019: - sbox[3,0,4] + rkey[3,0,4] = 0
 R1020: - sbox[3,0,5] + rkey[0,0,4] + rkey[2,2,4] >= 0
 R1021: sbox[3,0,5] + rkey[0,0,4] - rkey[2,2,4] >= 0
 R1022: sbox[3,0,5] - rkey[0,0,4] + rkey[2,2,4] >= 0
 R1023: - sbox[0,0,5] + sbox[3,0,5] + rkey[3,1,4] >= 0
 R1024: sbox[0,0,5] + sbox[3,0,5] - rkey[3,1,4] >= 0
 R1025: sbox[0,0,5] - sbox[3,0,5] + rkey[3,1,4] >= 0
 R1026: - sbox[2,0,5] + rkey[1,3,4] + rkey[2,2,4] >= 0
 R1027: sbox[2,0,5] + rkey[1,3,4] - rkey[2,2,4] >= 0
 R1028: sbox[2,0,5] - rkey[1,3,4] + rkey[2,2,4] >= 0
 R1029: sbox[1,0,5] - rkey[0,0,4] = 0
 R1030: tbox[0,0,4] + tbox[2,2,4] - tbox[3,0,5] >= 0
 R1031: tbox[0,0,4] - tbox[2,2,4] + tbox[3,0,5] >= 0
 R1032: - tbox[0,0,4] + tbox[2,2,4] + tbox[3,0,5] >= 0
 R1033: - tbox[0,0,5] + tbox[3,0,5] + tbox[3,1,4] >= 0
 R1034: tbox[0,0,5] + tbox[3,0,5] - tbox[3,1,4] >= 0
 R1035: tbox[0,0,5] - tbox[3,0,5] + tbox[3,1,4] >= 0
 R1036: tbox[1,3,4] - tbox[2,0,5] + tbox[2,2,4] >= 0
 R1037: tbox[1,3,4] + tbox[2,0,5] - tbox[2,2,4] >= 0
 R1038: - tbox[1,3,4] + tbox[2,0,5] + tbox[2,2,4] >= 0
 R1039: - tbox[0,0,4] + tbox[1,0,5] = 0
 R1040: unkn[0,0,4] - unkn[3,0,5] + forg[3,0,5] <= 0
 R1041: unkn[2,2,4] - unkn[3,0,5] + forg[3,0,5] <= 0
 R1042: unkn[0,0,4] + unkn[2,2,4] - unkn[3,0,5] + forg[3,0,5] >= 0
 R1043: - unkn[0,0,5] + unkn[3,0,5] + forg[0,0,5] - forg[3,0,5] <= 0
 R1044: - unkn[0,0,5] + unkn[3,1,4] + forg[0,0,5] <= 0
 R1045: - unkn[0,0,5] + unkn[3,0,5] + unkn[3,1,4] + forg[0,0,5]
   - forg[3,0,5] >= 0
 R1046: unkn[1,3,4] - unkn[2,0,5] + forg[2,0,5] <= 0
 R1047: - unkn[2,0,5] + unkn[2,2,4] + forg[2,0,5] <= 0
 R1048: unkn[1,3,4] - unkn[2,0,5] + unkn[2,2,4] + forg[2,0,5] >= 0
 R1049: unkn[0,0,4] - unkn[1,0,5] + forg[1,0,5] = 0
 R1050: - tbox[0,1,4] + forg[0,1,4] <= 0
 R1051: sbox[0,1,4] - rkey[0,1,4] + tkey[0,1,4] >= 0
 R1052: sbox[0,1,4] + rkey[0,1,4] - tkey[0,1,4] >= 0
 R1053: - sbox[0,1,4] + rkey[0,1,4] + tkey[0,1,4] >= 0
 R1054: - tbox[1,1,4] + forg[1,1,4] <= 0
 R1055: sbox[1,1,4] - rkey[1,1,4] + tkey[1,1,4] >= 0
 R1056: sbox[1,1,4] + rkey[1,1,4] - tkey[1,1,4] >= 0
 R1057: - sbox[1,1,4] + rkey[1,1,4] + tkey[1,1,4] >= 0
 R1058: - tbox[2,1,4] + forg[2,1,4] <= 0
 R1059: - sbox[2,1,4] + rkey[2,1,4] = 0
 R1060: - tbox[3,1,4] + forg[3,1,4] <= 0
 R1061: - sbox[3,1,4] + rkey[3,1,4] = 0
 R1062: - sbox[3,1,5] + rkey[0,1,4] + rkey[2,3,4] >= 0
 R1063: sbox[3,1,5] + rkey[0,1,4] - rkey[2,3,4] >= 0
 R1064: sbox[3,1,5] - rkey[0,1,4] + rkey[2,3,4] >= 0
 R1065: - sbox[0,1,5] + sbox[3,1,5] + rkey[3,2,4] >= 0
 R1066: sbox[0,1,5] + sbox[3,1,5] - rkey[3,2,4] >= 0
 R1067: sbox[0,1,5] - sbox[3,1,5] + rkey[3,2,4] >= 0
 R1068: - sbox[2,1,5] + rkey[1,0,4] + rkey[2,3,4] >= 0
 R1069: sbox[2,1,5] + rkey[1,0,4] - rkey[2,3,4] >= 0
 R1070: sbox[2,1,5] - rkey[1,0,4] + rkey[2,3,4] >= 0
 R1071: sbox[1,1,5] - rkey[0,1,4] = 0
 R1072: tbox[0,1,4] + tbox[2,3,4] - tbox[3,1,5] >= 0
 R1073: tbox[0,1,4] - tbox[2,3,4] + tbox[3,1,5] >= 0
 R1074: - tbox[0,1,4] + tbox[2,3,4] + tbox[3,1,5] >= 0
 R1075: - tbox[0,1,5] + tbox[3,1,5] + tbox[3,2,4] >= 0
 R1076: tbox[0,1,5] + tbox[3,1,5] - tbox[3,2,4] >= 0
 R1077: tbox[0,1,5] - tbox[3,1,5] + tbox[3,2,4] >= 0
 R1078: tbox[1,0,4] - tbox[2,1,5] + tbox[2,3,4] >= 0
 R1079: tbox[1,0,4] + tbox[2,1,5] - tbox[2,3,4] >= 0
 R1080: - tbox[1,0,4] + tbox[2,1,5] + tbox[2,3,4] >= 0
 R1081: - tbox[0,1,4] + tbox[1,1,5] = 0
 R1082: unkn[0,1,4] - unkn[3,1,5] + forg[3,1,5] <= 0
 R1083: unkn[2,3,4] - unkn[3,1,5] + forg[3,1,5] <= 0
 R1084: unkn[0,1,4] + unkn[2,3,4] - unkn[3,1,5] + forg[3,1,5] >= 0
 R1085: - unkn[0,1,5] + unkn[3,1,5] + forg[0,1,5] - forg[3,1,5] <= 0
 R1086: - unkn[0,1,5] + unkn[3,2,4] + forg[0,1,5] <= 0
 R1087: - unkn[0,1,5] + unkn[3,1,5] + unkn[3,2,4] + forg[0,1,5]
   - forg[3,1,5] >= 0
 R1088: unkn[1,0,4] - unkn[2,1,5] + forg[2,1,5] <= 0
 R1089: - unkn[2,1,5] + unkn[2,3,4] + forg[2,1,5] <= 0
 R1090: unkn[1,0,4] - unkn[2,1,5] + unkn[2,3,4] + forg[2,1,5] >= 0
 R1091: unkn[0,1,4] - unkn[1,1,5] + forg[1,1,5] = 0
 R1092: - tbox[0,2,4] + forg[0,2,4] <= 0
 R1093: sbox[0,2,4] - rkey[0,2,4] + tkey[0,2,4] >= 0
 R1094: sbox[0,2,4] + rkey[0,2,4] - tkey[0,2,4] >= 0
 R1095: - sbox[0,2,4] + rkey[0,2,4] + tkey[0,2,4] >= 0
 R1096: - tbox[1,2,4] + forg[1,2,4] <= 0
 R1097: sbox[1,2,4] - rkey[1,2,4] + tkey[1,2,4] >= 0
 R1098: sbox[1,2,4] + rkey[1,2,4] - tkey[1,2,4] >= 0
 R1099: - sbox[1,2,4] + rkey[1,2,4] + tkey[1,2,4] >= 0
 R1100: - tbox[2,2,4] + forg[2,2,4] <= 0
 R1101: - sbox[2,2,4] + rkey[2,2,4] = 0
 R1102: - tbox[3,2,4] + forg[3,2,4] <= 0
 R1103: - sbox[3,2,4] + rkey[3,2,4] = 0
 R1104: - sbox[3,2,5] + rkey[0,2,4] + rkey[2,0,4] >= 0
 R1105: sbox[3,2,5] + rkey[0,2,4] - rkey[2,0,4] >= 0
 R1106: sbox[3,2,5] - rkey[0,2,4] + rkey[2,0,4] >= 0
 R1107: - sbox[0,2,5] + sbox[3,2,5] + rkey[3,3,4] >= 0
 R1108: sbox[0,2,5] + sbox[3,2,5] - rkey[3,3,4] >= 0
 R1109: sbox[0,2,5] - sbox[3,2,5] + rkey[3,3,4] >= 0
 R1110: - sbox[2,2,5] + rkey[1,1,4] + rkey[2,0,4] >= 0
 R1111: sbox[2,2,5] + rkey[1,1,4] - rkey[2,0,4] >= 0
 R1112: sbox[2,2,5] - rkey[1,1,4] + rkey[2,0,4] >= 0
 R1113: sbox[1,2,5] - rkey[0,2,4] = 0
 R1114: tbox[0,2,4] + tbox[2,0,4] - tbox[3,2,5] >= 0
 R1115: tbox[0,2,4] - tbox[2,0,4] + tbox[3,2,5] >= 0
 R1116: - tbox[0,2,4] + tbox[2,0,4] + tbox[3,2,5] >= 0
 R1117: - tbox[0,2,5] + tbox[3,2,5] + tbox[3,3,4] >= 0
 R1118: tbox[0,2,5] + tbox[3,2,5] - tbox[3,3,4] >= 0
 R1119: tbox[0,2,5] - tbox[3,2,5] + tbox[3,3,4] >= 0
 R1120: tbox[1,1,4] + tbox[2,0,4] - tbox[2,2,5] >= 0
 R1121: tbox[1,1,4] - tbox[2,0,4] + tbox[2,2,5] >= 0
 R1122: - tbox[1,1,4] + tbox[2,0,4] + tbox[2,2,5] >= 0
 R1123: - tbox[0,2,4] + tbox[1,2,5] = 0
 R1124: unkn[0,2,4] - unkn[3,2,5] + forg[3,2,5] <= 0
 R1125: unkn[2,0,4] - unkn[3,2,5] + forg[3,2,5] <= 0
 R1126: unkn[0,2,4] + unkn[2,0,4] - unkn[3,2,5] + forg[3,2,5] >= 0
 R1127: - unkn[0,2,5] + unkn[3,2,5] + forg[0,2,5] - forg[3,2,5] <= 0
 R1128: - unkn[0,2,5] + unkn[3,3,4] + forg[0,2,5] <= 0
 R1129: - unkn[0,2,5] + unkn[3,2,5] + unkn[3,3,4] + forg[0,2,5]
   - forg[3,2,5] >= 0
 R1130: unkn[1,1,4] - unkn[2,2,5] + forg[2,2,5] <= 0
 R1131: unkn[2,0,4] - unkn[2,2,5] + forg[2,2,5] <= 0
 R1132: unkn[1,1,4] + unkn[2,0,4] - unkn[2,2,5] + forg[2,2,5] >= 0
 R1133: unkn[0,2,4] - unkn[1,2,5] + forg[1,2,5] = 0
 R1134: - tbox[0,3,4] + forg[0,3,4] <= 0
 R1135: sbox[0,3,4] - rkey[0,3,4] + tkey[0,3,4] >= 0
 R1136: sbox[0,3,4] + rkey[0,3,4] - tkey[0,3,4] >= 0
 R1137: - sbox[0,3,4] + rkey[0,3,4] + tkey[0,3,4] >= 0
 R1138: - tbox[1,3,4] + forg[1,3,4] <= 0
 R1139: sbox[1,3,4] - rkey[1,3,4] + tkey[1,3,4] >= 0
 R1140: sbox[1,3,4] + rkey[1,3,4] - tkey[1,3,4] >= 0
 R1141: - sbox[1,3,4] + rkey[1,3,4] + tkey[1,3,4] >= 0
 R1142: - tbox[2,3,4] + forg[2,3,4] <= 0
 R1143: - sbox[2,3,4] + rkey[2,3,4] = 0
 R1144: - tbox[3,3,4] + forg[3,3,4] <= 0
 R1145: - sbox[3,3,4] + rkey[3,3,4] = 0
 R1146: - sbox[3,3,5] + rkey[0,3,4] + rkey[2,1,4] >= 0
 R1147: sbox[3,3,5] + rkey[0,3,4] - rkey[2,1,4] >= 0
 R1148: sbox[3,3,5] - rkey[0,3,4] + rkey[2,1,4] >= 0
 R1149: - sbox[0,3,5] + sbox[3,3,5] + rkey[3,0,4] >= 0
 R1150: sbox[0,3,5] + sbox[3,3,5] - rkey[3,0,4] >= 0
 R1151: sbox[0,3,5] - sbox[3,3,5] + rkey[3,0,4] >= 0
 R1152: - sbox[2,3,5] + rkey[1,2,4] + rkey[2,1,4] >= 0
 R1153: sbox[2,3,5] + rkey[1,2,4] - rkey[2,1,4] >= 0
 R1154: sbox[2,3,5] - rkey[1,2,4] + rkey[2,1,4] >= 0
 R1155: sbox[1,3,5] - rkey[0,3,4] = 0
 R1156: tbox[0,3,4] + tbox[2,1,4] - tbox[3,3,5] >= 0
 R1157: tbox[0,3,4] - tbox[2,1,4] + tbox[3,3,5] >= 0
 R1158: - tbox[0,3,4] + tbox[2,1,4] + tbox[3,3,5] >= 0
 R1159: - tbox[0,3,5] + tbox[3,0,4] + tbox[3,3,5] >= 0
 R1160: tbox[0,3,5] - tbox[3,0,4] + tbox[3,3,5] >= 0
 R1161: tbox[0,3,5] + tbox[3,0,4] - tbox[3,3,5] >= 0
 R1162: tbox[1,2,4] + tbox[2,1,4] - tbox[2,3,5] >= 0
 R1163: tbox[1,2,4] - tbox[2,1,4] + tbox[2,3,5] >= 0
 R1164: - tbox[1,2,4] + tbox[2,1,4] + tbox[2,3,5] >= 0
 R1165: - tbox[0,3,4] + tbox[1,3,5] = 0
 R1166: unkn[0,3,4] - unkn[3,3,5] + forg[3,3,5] <= 0
 R1167: unkn[2,1,4] - unkn[3,3,5] + forg[3,3,5] <= 0
 R1168: unkn[0,3,4] + unkn[2,1,4] - unkn[3,3,5] + forg[3,3,5] >= 0
 R1169: - unkn[0,3,5] + unkn[3,3,5] + forg[0,3,5] - forg[3,3,5] <= 0
 R1170: - unkn[0,3,5] + unkn[3,0,4] + forg[0,3,5] <= 0
 R1171: - unkn[0,3,5] + unkn[3,0,4] + unkn[3,3,5] + forg[0,3,5]
   - forg[3,3,5] >= 0
 R1172: unkn[1,2,4] - unkn[2,3,5] + forg[2,3,5] <= 0
 R1173: unkn[2,1,4] - unkn[2,3,5] + forg[2,3,5] <= 0
 R1174: unkn[1,2,4] + unkn[2,1,4] - unkn[2,3,5] + forg[2,3,5] >= 0
 R1175: unkn[0,3,4] - unkn[1,3,5] + forg[1,3,5] = 0
 R1176: sbox[0,0,0] + sbox[0,1,0] + sbox[0,2,0] + sbox[0,3,0] + sbox[1,0,0]
   + sbox[1,1,0] + sbox[1,2,0] + sbox[1,3,0] + sbox[2,0,0] + sbox[2,1,0]
   + sbox[2,2,0] + sbox[2,3,0] + sbox[3,0,0] + sbox[3,1,0] + sbox[3,2,0]
   + sbox[3,3,0] + tkey[0,0,0] + tkey[0,1,0] + tkey[0,2,0] + tkey[0,3,0]
   + tkey[1,0,0] + tkey[1,1,0] + tkey[1,2,0] + tkey[1,3,0] + tkey[2,0,0]
   + tkey[2,1,0] + tkey[2,2,0] + tkey[2,3,0] + tkey[3,0,0] + tkey[3,1,0]
   + tkey[3,2,0] + tkey[3,3,0] >= 1
 R1177: sbox[0,0,1] + sbox[0,1,1] + sbox[0,2,1] + sbox[0,3,1] + sbox[1,0,1]
   + sbox[1,1,1] + sbox[1,2,1] + sbox[1,3,1] + sbox[2,0,1] + sbox[2,1,1]
   + sbox[2,2,1] + sbox[2,3,1] + sbox[3,0,1] + sbox[3,1,1] + sbox[3,2,1]
   + sbox[3,3,1] + tkey[0,0,1] + tkey[0,1,1] + tkey[0,2,1] + tkey[0,3,1]
   + tkey[1,0,1] + tkey[1,1,1] + tkey[1,2,1] + tkey[1,3,1] + tkey[2,0,1]
   + tkey[2,1,1] + tkey[2,2,1] + tkey[2,3,1] + tkey[3,0,1] + tkey[3,1,1]
   + tkey[3,2,1] + tkey[3,3,1] >= 1
 R1178: sbox[0,0,2] + sbox[0,1,2] + sbox[0,2,2] + sbox[0,3,2] + sbox[1,0,2]
   + sbox[1,1,2] + sbox[1,2,2] + sbox[1,3,2] + sbox[2,0,2] + sbox[2,1,2]
   + sbox[2,2,2] + sbox[2,3,2] + sbox[3,0,2] + sbox[3,1,2] + sbox[3,2,2]
   + sbox[3,3,2] + tkey[0,0,2] + tkey[0,1,2] + tkey[0,2,2] + tkey[0,3,2]
   + tkey[1,0,2] + tkey[1,1,2] + tkey[1,2,2] + tkey[1,3,2] + tkey[2,0,2]
   + tkey[2,1,2] + tkey[2,2,2] + tkey[2,3,2] + tkey[3,0,2] + tkey[3,1,2]
   + tkey[3,2,2] + tkey[3,3,2] >= 1
 R1179: sbox[0,0,3] + sbox[0,1,3] + sbox[0,2,3] + sbox[0,3,3] + sbox[1,0,3]
   + sbox[1,1,3] + sbox[1,2,3] + sbox[1,3,3] + sbox[2,0,3] + sbox[2,1,3]
   + sbox[2,2,3] + sbox[2,3,3] + sbox[3,0,3] + sbox[3,1,3] + sbox[3,2,3]
   + sbox[3,3,3] + tkey[0,0,3] + tkey[0,1,3] + tkey[0,2,3] + tkey[0,3,3]
   + tkey[1,0,3] + tkey[1,1,3] + tkey[1,2,3] + tkey[1,3,3] + tkey[2,0,3]
   + tkey[2,1,3] + tkey[2,2,3] + tkey[2,3,3] + tkey[3,0,3] + tkey[3,1,3]
   + tkey[3,2,3] + tkey[3,3,3] >= 1
 R1180: sbox[0,0,4] + sbox[0,1,4] + sbox[0,2,4] + sbox[0,3,4] + sbox[1,0,4]
   + sbox[1,1,4] + sbox[1,2,4] + sbox[1,3,4] + sbox[2,0,4] + sbox[2,1,4]
   + sbox[2,2,4] + sbox[2,3,4] + sbox[3,0,4] + sbox[3,1,4] + sbox[3,2,4]
   + sbox[3,3,4] + tkey[0,0,4] + tkey[0,1,4] + tkey[0,2,4] + tkey[0,3,4]
   + tkey[1,0,4] + tkey[1,1,4] + tkey[1,2,4] + tkey[1,3,4] + tkey[2,0,4]
   + tkey[2,1,4] + tkey[2,2,4] + tkey[2,3,4] + tkey[3,0,4] + tkey[3,1,4]
   + tkey[3,2,4] + tkey[3,3,4] >= 1
 R1181: sbox[0,0,0] + sbox[0,1,0] + sbox[0,2,0] + sbox[0,3,0] + sbox[1,0,0]
   + sbox[1,1,0] + sbox[1,2,0] + sbox[1,3,0] + sbox[2,0,0] + sbox[2,1,0]
   + sbox[2,2,0] + sbox[2,3,0] + sbox[3,0,0] + sbox[3,1,0] + sbox[3,2,0]
   + sbox[3,3,0] = 0
 R1182: sbox[0,0,5] + sbox[0,1,5] + sbox[0,2,5] + sbox[0,3,5] + sbox[1,0,5]
   + sbox[1,1,5] + sbox[1,2,5] + sbox[1,3,5] + sbox[2,0,5] + sbox[2,1,5]
   + sbox[2,2,5] + sbox[2,3,5] + sbox[3,0,5] + sbox[3,1,5] + sbox[3,2,5]
   + sbox[3,3,5] = 0
 R1183: tbox[0,0,0] = 1
 R1184: unkn[0,0,0] - forg[0,0,0] = 0
 R1185: tbox[0,1,0] = 0
 R1186: unkn[0,1,0] - forg[0,1,0] = 0
 R1187: tbox[0,2,0] = 0
 R1188: unkn[0,2,0] - forg[0,2,0] = 0
 R1189: tbox[0,3,0] = 0
 R1190: unkn[0,3,0] - forg[0,3,0] = 0
 R1191: tbox[1,0,0] = 0
 R1192: unkn[1,0,0] - forg[1,0,0] = 0
 R1193: tbox[1,1,0] = 0
 R1194: unkn[1,1,0] - forg[1,1,0] = 0
 R1195: tbox[1,2,0] = 0
 R1196: unkn[1,2,0] - forg[1,2,0] = 0
 R1197: tbox[1,3,0] = 0
 R1198: unkn[1,3,0] - forg[1,3,0] = 0
 R1199: tbox[2,0,0] = 0
 R1200: unkn[2,0,0] - forg[2,0,0] = 0
 R1201: tbox[2,1,0] = 0
 R1202: unkn[2,1,0] - forg[2,1,0] = 0
 R1203: tbox[2,2,0] = 0
 R1204: unkn[2,2,0] - forg[2,2,0] = 0
 R1205: tbox[2,3,0] = 0
 R1206: unkn[2,3,0] - forg[2,3,0] = 0
 R1207: tbox[3,0,0] = 0
 R1208: unkn[3,0,0] - forg[3,0,0] = 0
 R1209: tbox[3,1,0] = 0
 R1210: unkn[3,1,0] - forg[3,1,0] = 0
 R1211: tbox[3,2,0] = 0
 R1212: unkn[3,2,0] - forg[3,2,0] = 0
 R1213: tbox[3,3,0] = 0
 R1214: unkn[3,3,0] - forg[3,3,0] = 0
Bounds
 cost[0,0,0] <= 2
 cost[0,0,1] <= 2
 cost[0,0,2] <= 2
 cost[0,0,3] <= 2
 cost[0,0,4] <= 2
 cost[0,0,5] <= 2
 cost[0,1,0] <= 2
 cost[0,1,1] <= 2
 cost[0,1,2] <= 2
 cost[0,1,3] <= 2
 cost[0,1,4] <= 2
 cost[0,1,5] <= 2
 cost[0,2,0] <= 2
 cost[0,2,1] <= 2
 cost[0,2,2] <= 2
 cost[0,2,3] <= 2
 cost[0,2,4] <= 2
 cost[0,2,5] <= 2
 cost[0,3,0] <= 2
 cost[0,3,1] <= 2
 cost[0,3,2] <= 2
 cost[0,3,3] <= 2
 cost[0,3,4] <= 2
 cost[0,3,5] <= 2
 cost[1,0,0] <= 2
 cost[1,0,1] <= 2
 cost[1,0,2] <= 2
 cost[1,0,3] <= 2
 cost[1,0,4] <= 2
 cost[1,0,5] <= 2
 cost[1,1,0] <= 2
 cost[1,1,1] <= 2
 cost[1,1,2] <= 2
 cost[1,1,3] <= 2
 cost[1,1,4] <= 2
 cost[1,1,5] <= 2
 cost[1,2,0] <= 2
 cost[1,2,1] <= 2
 cost[1,2,2] <= 2
 cost[1,2,3] <= 2
 cost[1,2,4] <= 2
 cost[1,2,5] <= 2
 cost[1,3,0] <= 2
 cost[1,3,1] <= 2
 cost[1,3,2] <= 2
 cost[1,3,3] <= 2
 cost[1,3,4] <= 2
 cost[1,3,5] <= 2
 cost[2,0,0] <= 2
 cost[2,0,1] <= 2
 cost[2,0,2] <= 2
 cost[2,0,3] <= 2
 cost[2,0,4] <= 2
 cost[2,0,5] <= 2
 cost[2,1,0] <= 2
 cost[2,1,1] <= 2
 cost[2,1,2] <= 2
 cost[2,1,3] <= 2
 cost[2,1,4] <= 2
 cost[2,1,5] <= 2
 cost[2,2,0] <= 2
 cost[2,2,1] <= 2
 cost[2,2,2] <= 2
 cost[2,2,3] <= 2
 cost[2,2,4] <= 2
 cost[2,2,5] <= 2
 cost[2,3,0] <= 2
 cost[2,3,1] <= 2
 cost[2,3,2] <= 2
 cost[2,3,3] <= 2
 cost[2,3,4] <= 2
 cost[2,3,5] <= 2
 cost[3,0,0] <= 2
 cost[3,0,1] <= 2
 cost[3,0,2] <= 2
 cost[3,0,3] <= 2
 cost[3,0,4] <= 2
 cost[3,0,5] <= 2
 cost[3,1,0] <= 2
 cost[3,1,1] <= 2
 cost[3,1,2] <= 2
 cost[3,1,3] <= 2
 cost[3,1,4] <= 2
 cost[3,1,5] <= 2
 cost[3,2,0] <= 2
 cost[3,2,1] <= 2
 cost[3,2,2] <= 2
 cost[3,2,3] <= 2
 cost[3,2,4] <= 2
 cost[3,2,5] <= 2
 cost[3,3,0] <= 2
 cost[3,3,1] <= 2
 cost[3,3,2] <= 2
 cost[3,3,3] <= 2
 cost[3,3,4] <= 2
 cost[3,3,5] <= 2
Binaries
 sbox[0,0,0] sbox[0,0,1] sbox[0,0,2] sbox[0,0,3] sbox[0,0,4] sbox[0,0,5]
 sbox[0,1,0] sbox[0,1,1] sbox[0,1,2] sbox[0,1,3] sbox[0,1,4] sbox[0,1,5]
 sbox[0,2,0] sbox[0,2,1] sbox[0,2,2] sbox[0,2,3] sbox[0,2,4] sbox[0,2,5]
 sbox[0,3,0] sbox[0,3,1] sbox[0,3,2] sbox[0,3,3] sbox[0,3,4] sbox[0,3,5]
 sbox[1,0,0] sbox[1,0,1] sbox[1,0,2] sbox[1,0,3] sbox[1,0,4] sbox[1,0,5]
 sbox[1,1,0] sbox[1,1,1] sbox[1,1,2] sbox[1,1,3] sbox[1,1,4] sbox[1,1,5]
 sbox[1,2,0] sbox[1,2,1] sbox[1,2,2] sbox[1,2,3] sbox[1,2,4] sbox[1,2,5]
 sbox[1,3,0] sbox[1,3,1] sbox[1,3,2] sbox[1,3,3] sbox[1,3,4] sbox[1,3,5]
 sbox[2,0,0] sbox[2,0,1] sbox[2,0,2] sbox[2,0,3] sbox[2,0,4] sbox[2,0,5]
 sbox[2,1,0] sbox[2,1,1] sbox[2,1,2] sbox[2,1,3] sbox[2,1,4] sbox[2,1,5]
 sbox[2,2,0] sbox[2,2,1] sbox[2,2,2] sbox[2,2,3] sbox[2,2,4] sbox[2,2,5]
 sbox[2,3,0] sbox[2,3,1] sbox[2,3,2] sbox[2,3,3] sbox[2,3,4] sbox[2,3,5]
 sbox[3,0,0] sbox[3,0,1] sbox[3,0,2] sbox[3,0,3] sbox[3,0,4] sbox[3,0,5]
 sbox[3,1,0] sbox[3,1,1] sbox[3,1,2] sbox[3,1,3] sbox[3,1,4] sbox[3,1,5]
 sbox[3,2,0] sbox[3,2,1] sbox[3,2,2] sbox[3,2,3] sbox[3,2,4] sbox[3,2,5]
 sbox[3,3,0] sbox[3,3,1] sbox[3,3,2] sbox[3,3,3] sbox[3,3,4] sbox[3,3,5]
 rkey[0,0,0] rkey[0,0,1] rkey[0,0,2] rkey[0,0,3] rkey[0,0,4] rkey[0,0,5]
 rkey[0,1,0] rkey[0,1,1] rkey[0,1,2] rkey[0,1,3] rkey[0,1,4] rkey[0,1,5]
 rkey[0,2,0] rkey[0,2,1] rkey[0,2,2] rkey[0,2,3] rkey[0,2,4] rkey[0,2,5]
 rkey[0,3,0] rkey[0,3,1] rkey[0,3,2] rkey[0,3,3] rkey[0,3,4] rkey[0,3,5]
 rkey[1,0,0] rkey[1,0,1] rkey[1,0,2] rkey[1,0,3] rkey[1,0,4] rkey[1,0,5]
 rkey[1,1,0] rkey[1,1,1] rkey[1,1,2] rkey[1,1,3] rkey[1,1,4] rkey[1,1,5]
 rkey[1,2,0] rkey[1,2,1] rkey[1,2,2] rkey[1,2,3] rkey[1,2,4] rkey[1,2,5]
 rkey[1,3,0] rkey[1,3,1] rkey[1,3,2] rkey[1,3,3] rkey[1,3,4] rkey[1,3,5]
 rkey[2,0,0] rkey[2,0,1] rkey[2,0,2] rkey[2,0,3] rkey[2,0,4] rkey[2,0,5]
 rkey[2,1,0] rkey[2,1,1] rkey[2,1,2] rkey[2,1,3] rkey[2,1,4] rkey[2,1,5]
 rkey[2,2,0] rkey[2,2,1] rkey[2,2,2] rkey[2,2,3] rkey[2,2,4] rkey[2,2,5]
 rkey[2,3,0] rkey[2,3,1] rkey[2,3,2] rkey[2,3,3] rkey[2,3,4] rkey[2,3,5]
 rkey[3,0,0] rkey[3,0,1] rkey[3,0,2] rkey[3,0,3] rkey[3,0,4] rkey[3,0,5]
 rkey[3,1,0] rkey[3,1,1] rkey[3,1,2] rkey[3,1,3] rkey[3,1,4] rkey[3,1,5]
 rkey[3,2,0] rkey[3,2,1] rkey[3,2,2] rkey[3,2,3] rkey[3,2,4] rkey[3,2,5]
 rkey[3,3,0] rkey[3,3,1] rkey[3,3,2] rkey[3,3,3] rkey[3,3,4] rkey[3,3,5]
 tkey[0,0,0] tkey[0,0,1] tkey[0,0,2] tkey[0,0,3] tkey[0,0,4] tkey[0,0,5]
 tkey[0,1,0] tkey[0,1,1] tkey[0,1,2] tkey[0,1,3] tkey[0,1,4] tkey[0,1,5]
 tkey[0,2,0] tkey[0,2,1] tkey[0,2,2] tkey[0,2,3] tkey[0,2,4] tkey[0,2,5]
 tkey[0,3,0] tkey[0,3,1] tkey[0,3,2] tkey[0,3,3] tkey[0,3,4] tkey[0,3,5]
 tkey[1,0,0] tkey[1,0,1] tkey[1,0,2] tkey[1,0,3] tkey[1,0,4] tkey[1,0,5]
 tkey[1,1,0] tkey[1,1,1] tkey[1,1,2] tkey[1,1,3] tkey[1,1,4] tkey[1,1,5]
 tkey[1,2,0] tkey[1,2,1] tkey[1,2,2] tkey[1,2,3] tkey[1,2,4] tkey[1,2,5]
 tkey[1,3,0] tkey[1,3,1] tkey[1,3,2] tkey[1,3,3] tkey[1,3,4] tkey[1,3,5]
 tkey[2,0,0] tkey[2,0,1] tkey[2,0,2] tkey[2,0,3] tkey[2,0,4] tkey[2,0,5]
 tkey[2,1,0] tkey[2,1,1] tkey[2,1,2] tkey[2,1,3] tkey[2,1,4] tkey[2,1,5]
 tkey[2,2,0] tkey[2,2,1] tkey[2,2,2] tkey[2,2,3] tkey[2,2,4] tkey[2,2,5]
 tkey[2,3,0] tkey[2,3,1] tkey[2,3,2] tkey[2,3,3] tkey[2,3,4] tkey[2,3,5]
 tkey[3,0,0] tkey[3,0,1] tkey[3,0,2] tkey[3,0,3] tkey[3,0,4] tkey[3,0,5]
 tkey[3,1,0] tkey[3,1,1] tkey[3,1,2] tkey[3,1,3] tkey[3,1,4] tkey[3,1,5]
 tkey[3,2,0] tkey[3,2,1] tkey[3,2,2] tkey[3,2,3] tkey[3,2,4] tkey[3,2,5]
 tkey[3,3,0] tkey[3,3,1] tkey[3,3,2] tkey[3,3,3] tkey[3,3,4] tkey[3,3,5]
 tbox[0,0,0] tbox[0,0,1] tbox[0,0,2] tbox[0,0,3] tbox[0,0,4] tbox[0,0,5]
 tbox[0,1,0] tbox[0,1,1] tbox[0,1,2] tbox[0,1,3] tbox[0,1,4] tbox[0,1,5]
 tbox[0,2,0] tbox[0,2,1] tbox[0,2,2] tbox[0,2,3] tbox[0,2,4] tbox[0,2,5]
 tbox[0,3,0] tbox[0,3,1] tbox[0,3,2] tbox[0,3,3] tbox[0,3,4] tbox[0,3,5]
 tbox[1,0,0] tbox[1,0,1] tbox[1,0,2] tbox[1,0,3] tbox[1,0,4] tbox[1,0,5]
 tbox[1,1,0] tbox[1,1,1] tbox[1,1,2] tbox[1,1,3] tbox[1,1,4] tbox[1,1,5]
 tbox[1,2,0] tbox[1,2,1] tbox[1,2,2] tbox[1,2,3] tbox[1,2,4] tbox[1,2,5]
 tbox[1,3,0] tbox[1,3,1] tbox[1,3,2] tbox[1,3,3] tbox[1,3,4] tbox[1,3,5]
 tbox[2,0,0] tbox[2,0,1] tbox[2,0,2] tbox[2,0,3] tbox[2,0,4] tbox[2,0,5]
 tbox[2,1,0] tbox[2,1,1] tbox[2,1,2] tbox[2,1,3] tbox[2,1,4] tbox[2,1,5]
 tbox[2,2,0] tbox[2,2,1] tbox[2,2,2] tbox[2,2,3] tbox[2,2,4] tbox[2,2,5]
 tbox[2,3,0] tbox[2,3,1] tbox[2,3,2] tbox[2,3,3] tbox[2,3,4] tbox[2,3,5]
 tbox[3,0,0] tbox[3,0,1] tbox[3,0,2] tbox[3,0,3] tbox[3,0,4] tbox[3,0,5]
 tbox[3,1,0] tbox[3,1,1] tbox[3,1,2] tbox[3,1,3] tbox[3,1,4] tbox[3,1,5]
 tbox[3,2,0] tbox[3,2,1] tbox[3,2,2] tbox[3,2,3] tbox[3,2,4] tbox[3,2,5]
 tbox[3,3,0] tbox[3,3,1] tbox[3,3,2] tbox[3,3,3] tbox[3,3,4] tbox[3,3,5]
 unkn[0,0,0] unkn[0,0,1] unkn[0,0,2] unkn[0,0,3] unkn[0,0,4] unkn[0,0,5]
 unkn[0,1,0] unkn[0,1,1] unkn[0,1,2] unkn[0,1,3] unkn[0,1,4] unkn[0,1,5]
 unkn[0,2,0] unkn[0,2,1] unkn[0,2,2] unkn[0,2,3] unkn[0,2,4] unkn[0,2,5]
 unkn[0,3,0] unkn[0,3,1] unkn[0,3,2] unkn[0,3,3] unkn[0,3,4] unkn[0,3,5]
 unkn[1,0,0] unkn[1,0,1] unkn[1,0,2] unkn[1,0,3] unkn[1,0,4] unkn[1,0,5]
 unkn[1,1,0] unkn[1,1,1] unkn[1,1,2] unkn[1,1,3] unkn[1,1,4] unkn[1,1,5]
 unkn[1,2,0] unkn[1,2,1] unkn[1,2,2] unkn[1,2,3] unkn[1,2,4] unkn[1,2,5]
 unkn[1,3,0] unkn[1,3,1] unkn[1,3,2] unkn[1,3,3] unkn[1,3,4] unkn[1,3,5]
 unkn[2,0,0] unkn[2,0,1] unkn[2,0,2] unkn[2,0,3] unkn[2,0,4] unkn[2,0,5]
 unkn[2,1,0] unkn[2,1,1] unkn[2,1,2] unkn[2,1,3] unkn[2,1,4] unkn[2,1,5]
 unkn[2,2,0] unkn[2,2,1] unkn[2,2,2] unkn[2,2,3] unkn[2,2,4] unkn[2,2,5]
 unkn[2,3,0] unkn[2,3,1] unkn[2,3,2] unkn[2,3,3] unkn[2,3,4] unkn[2,3,5]
 unkn[3,0,0] unkn[3,0,1] unkn[3,0,2] unkn[3,0,3] unkn[3,0,4] unkn[3,0,5]
 unkn[3,1,0] unkn[3,1,1] unkn[3,1,2] unkn[3,1,3] unkn[3,1,4] unkn[3,1,5]
 unkn[3,2,0] unkn[3,2,1] unkn[3,2,2] unkn[3,2,3] unkn[3,2,4] unkn[3,2,5]
 unkn[3,3,0] unkn[3,3,1] unkn[3,3,2] unkn[3,3,3] unkn[3,3,4] unkn[3,3,5]
 forg[0,0,0] forg[0,0,1] forg[0,0,2] forg[0,0,3] forg[0,0,4] forg[0,0,5]
 forg[0,1,0] forg[0,1,1] forg[0,1,2] forg[0,1,3] forg[0,1,4] forg[0,1,5]
 forg[0,2,0] forg[0,2,1] forg[0,2,2] forg[0,2,3] forg[0,2,4] forg[0,2,5]
 forg[0,3,0] forg[0,3,1] forg[0,3,2] forg[0,3,3] forg[0,3,4] forg[0,3,5]
 forg[1,0,0] forg[1,0,1] forg[1,0,2] forg[1,0,3] forg[1,0,4] forg[1,0,5]
 forg[1,1,0] forg[1,1,1] forg[1,1,2] forg[1,1,3] forg[1,1,4] forg[1,1,5]
 forg[1,2,0] forg[1,2,1] forg[1,2,2] forg[1,2,3] forg[1,2,4] forg[1,2,5]
 forg[1,3,0] forg[1,3,1] forg[1,3,2] forg[1,3,3] forg[1,3,4] forg[1,3,5]
 forg[2,0,0] forg[2,0,1] forg[2,0,2] forg[2,0,3] forg[2,0,4] forg[2,0,5]
 forg[2,1,0] forg[2,1,1] forg[2,1,2] forg[2,1,3] forg[2,1,4] forg[2,1,5]
 forg[2,2,0] forg[2,2,1] forg[2,2,2] forg[2,2,3] forg[2,2,4] forg[2,2,5]
 forg[2,3,0] forg[2,3,1] forg[2,3,2] forg[2,3,3] forg[2,3,4] forg[2,3,5]
 forg[3,0,0] forg[3,0,1] forg[3,0,2] forg[3,0,3] forg[3,0,4] forg[3,0,5]
 forg[3,1,0] forg[3,1,1] forg[3,1,2] forg[3,1,3] forg[3,1,4] forg[3,1,5]
 forg[3,2,0] forg[3,2,1] forg[3,2,2] forg[3,2,3] forg[3,2,4] forg[3,2,5]
 forg[3,3,0] forg[3,3,1] forg[3,3,2] forg[3,3,3] forg[3,3,4] forg[3,3,5]
 lane[0,0] lane[0,1] lane[0,2] lane[0,3] lane[1,0] lane[1,1] lane[1,2]
 lane[1,3] lane[2,0] lane[2,1] lane[2,2] lane[2,3] lane[3,0] lane[3,1]
 lane[3,2] lane[3,3]
Generals
 cost[0,0,0] cost[0,0,1] cost[0,0,2] cost[0,0,3] cost[0,0,4] cost[0,0,5]
 cost[0,1,0] cost[0,1,1] cost[0,1,2] cost[0,1,3] cost[0,1,4] cost[0,1,5]
 cost[0,2,0] cost[0,2,1] cost[0,2,2] cost[0,2,3] cost[0,2,4] cost[0,2,5]
 cost[0,3,0] cost[0,3,1] cost[0,3,2] cost[0,3,3] cost[0,3,4] cost[0,3,5]
 cost[1,0,0] cost[1,0,1] cost[1,0,2] cost[1,0,3] cost[1,0,4] cost[1,0,5]
 cost[1,1,0] cost[1,1,1] cost[1,1,2] cost[1,1,3] cost[1,1,4] cost[1,1,5]
 cost[1,2,0] cost[1,2,1] cost[1,2,2] cost[1,2,3] cost[1,2,4] cost[1,2,5]
 cost[1,3,0] cost[1,3,1] cost[1,3,2] cost[1,3,3] cost[1,3,4] cost[1,3,5]
 cost[2,0,0] cost[2,0,1] cost[2,0,2] cost[2,0,3] cost[2,0,4] cost[2,0,5]
 cost[2,1,0] cost[2,1,1] cost[2,1,2] cost[2,1,3] cost[2,1,4] cost[2,1,5]
 cost[2,2,0] cost[2,2,1] cost[2,2,2] cost[2,2,3] cost[2,2,4] cost[2,2,5]
 cost[2,3,0] cost[2,3,1] cost[2,3,2] cost[2,3,3] cost[2,3,4] cost[2,3,5]
 cost[3,0,0] cost[3,0,1] cost[3,0,2] cost[3,0,3] cost[3,0,4] cost[3,0,5]
 cost[3,1,0] cost[3,1,1] cost[3,1,2] cost[3,1,3] cost[3,1,4] cost[3,1,5]
 cost[3,2,0] cost[3,2,1] cost[3,2,2] cost[3,2,3] cost[3,2,4] cost[3,2,5]
 cost[3,3,0] cost[3,3,1] cost[3,3,2] cost[3,3,3] cost[3,3,4] cost[3,3,5]
End
