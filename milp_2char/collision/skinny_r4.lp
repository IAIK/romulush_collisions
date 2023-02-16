\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 rkey[0,0,4] + 0 rkey[0,1,4] + 0 rkey[0,2,4] + 0 rkey[0,3,4]
   + 0 rkey[1,0,4] + 0 rkey[1,1,4] + 0 rkey[1,2,4] + 0 rkey[1,3,4]
   + 0 rkey[2,0,4] + 0 rkey[2,1,4] + 0 rkey[2,2,4] + 0 rkey[2,3,4]
   + 0 rkey[3,0,4] + 0 rkey[3,1,4] + 0 rkey[3,2,4] + 0 rkey[3,3,4]
   + 0 tkey[0,0,4] + 0 tkey[0,1,4] + 0 tkey[0,2,4] + 0 tkey[0,3,4]
   + 0 tkey[1,0,4] + 0 tkey[1,1,4] + 0 tkey[1,2,4] + 0 tkey[1,3,4]
   + 0 tkey[2,0,4] + 0 tkey[2,1,4] + 0 tkey[2,2,4] + 0 tkey[2,3,4]
   + 0 tkey[3,0,4] + 0 tkey[3,1,4] + 0 tkey[3,2,4] + 0 tkey[3,3,4]
   + cost[0,0,0] + cost[0,0,1] + cost[0,0,2] + cost[0,0,3] + 0 cost[0,0,4]
   + cost[0,1,0] + cost[0,1,1] + cost[0,1,2] + cost[0,1,3] + 0 cost[0,1,4]
   + cost[0,2,0] + cost[0,2,1] + cost[0,2,2] + cost[0,2,3] + 0 cost[0,2,4]
   + cost[0,3,0] + cost[0,3,1] + cost[0,3,2] + cost[0,3,3] + 0 cost[0,3,4]
   + cost[1,0,0] + cost[1,0,1] + cost[1,0,2] + cost[1,0,3] + 0 cost[1,0,4]
   + cost[1,1,0] + cost[1,1,1] + cost[1,1,2] + cost[1,1,3] + 0 cost[1,1,4]
   + cost[1,2,0] + cost[1,2,1] + cost[1,2,2] + cost[1,2,3] + 0 cost[1,2,4]
   + cost[1,3,0] + cost[1,3,1] + cost[1,3,2] + cost[1,3,3] + 0 cost[1,3,4]
   + cost[2,0,0] + cost[2,0,1] + cost[2,0,2] + cost[2,0,3] + 0 cost[2,0,4]
   + cost[2,1,0] + cost[2,1,1] + cost[2,1,2] + cost[2,1,3] + 0 cost[2,1,4]
   + cost[2,2,0] + cost[2,2,1] + cost[2,2,2] + cost[2,2,3] + 0 cost[2,2,4]
   + cost[2,3,0] + cost[2,3,1] + cost[2,3,2] + cost[2,3,3] + 0 cost[2,3,4]
   + cost[3,0,0] + cost[3,0,1] + cost[3,0,2] + cost[3,0,3] + 0 cost[3,0,4]
   + cost[3,1,0] + cost[3,1,1] + cost[3,1,2] + cost[3,1,3] + 0 cost[3,1,4]
   + cost[3,2,0] + cost[3,2,1] + cost[3,2,2] + cost[3,2,3] + 0 cost[3,2,4]
   + cost[3,3,0] + cost[3,3,1] + cost[3,3,2] + cost[3,3,3] + 0 cost[3,3,4]
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
 R192: tkey[0,0,0] - lane[0,0] <= 0
 R193: tkey[2,0,1] - lane[0,0] <= 0
 R194: tkey[0,2,2] - lane[0,0] <= 0
 R195: tkey[2,2,3] - lane[0,0] <= 0
 R196: tkey[0,1,0] - lane[0,1] <= 0
 R197: tkey[2,1,1] - lane[0,1] <= 0
 R198: tkey[0,0,2] - lane[0,1] <= 0
 R199: tkey[2,0,3] - lane[0,1] <= 0
 R200: tkey[0,2,0] - lane[0,2] <= 0
 R201: tkey[2,2,1] - lane[0,2] <= 0
 R202: tkey[1,0,2] - lane[0,2] <= 0
 R203: tkey[3,0,3] - lane[0,2] <= 0
 R204: tkey[0,3,0] - lane[0,3] <= 0
 R205: tkey[2,3,1] - lane[0,3] <= 0
 R206: tkey[1,3,2] - lane[0,3] <= 0
 R207: tkey[3,3,3] - lane[0,3] <= 0
 R208: tkey[1,0,0] - lane[1,0] <= 0
 R209: tkey[3,0,1] - lane[1,0] <= 0
 R210: tkey[1,2,2] - lane[1,0] <= 0
 R211: tkey[3,2,3] - lane[1,0] <= 0
 R212: tkey[1,1,0] - lane[1,1] <= 0
 R213: tkey[3,1,1] - lane[1,1] <= 0
 R214: tkey[0,3,2] - lane[1,1] <= 0
 R215: tkey[2,3,3] - lane[1,1] <= 0
 R216: tkey[1,2,0] - lane[1,2] <= 0
 R217: tkey[3,2,1] - lane[1,2] <= 0
 R218: tkey[1,1,2] - lane[1,2] <= 0
 R219: tkey[3,1,3] - lane[1,2] <= 0
 R220: tkey[1,3,0] - lane[1,3] <= 0
 R221: tkey[3,3,1] - lane[1,3] <= 0
 R222: tkey[0,1,2] - lane[1,3] <= 0
 R223: tkey[2,1,3] - lane[1,3] <= 0
 R224: tkey[2,0,0] - lane[2,0] <= 0
 R225: tkey[0,2,1] - lane[2,0] <= 0
 R226: tkey[2,2,2] - lane[2,0] <= 0
 R227: tkey[1,0,3] - lane[2,0] <= 0
 R228: tkey[2,1,0] - lane[2,1] <= 0
 R229: tkey[0,0,1] - lane[2,1] <= 0
 R230: tkey[2,0,2] - lane[2,1] <= 0
 R231: tkey[0,2,3] - lane[2,1] <= 0
 R232: tkey[2,2,0] - lane[2,2] <= 0
 R233: tkey[1,0,1] - lane[2,2] <= 0
 R234: tkey[3,0,2] - lane[2,2] <= 0
 R235: tkey[1,2,3] - lane[2,2] <= 0
 R236: tkey[2,3,0] - lane[2,3] <= 0
 R237: tkey[1,3,1] - lane[2,3] <= 0
 R238: tkey[3,3,2] - lane[2,3] <= 0
 R239: tkey[0,1,3] - lane[2,3] <= 0
 R240: tkey[3,0,0] - lane[3,0] <= 0
 R241: tkey[1,2,1] - lane[3,0] <= 0
 R242: tkey[3,2,2] - lane[3,0] <= 0
 R243: tkey[1,1,3] - lane[3,0] <= 0
 R244: tkey[3,1,0] - lane[3,1] <= 0
 R245: tkey[0,3,1] - lane[3,1] <= 0
 R246: tkey[2,3,2] - lane[3,1] <= 0
 R247: tkey[1,3,3] - lane[3,1] <= 0
 R248: tkey[3,2,0] - lane[3,2] <= 0
 R249: tkey[1,1,1] - lane[3,2] <= 0
 R250: tkey[3,1,2] - lane[3,2] <= 0
 R251: tkey[0,3,3] - lane[3,2] <= 0
 R252: tkey[3,3,0] - lane[3,3] <= 0
 R253: tkey[0,1,1] - lane[3,3] <= 0
 R254: tkey[2,1,2] - lane[3,3] <= 0
 R255: tkey[0,0,3] - lane[3,3] <= 0
 R256: - tkey[0,0,0] - tkey[0,2,2] - tkey[2,0,1] - tkey[2,2,3]
   + 4 lane[0,0] <= 1
 R257: - tkey[0,0,2] - tkey[0,1,0] - tkey[2,0,3] - tkey[2,1,1]
   + 4 lane[0,1] <= 1
 R258: - tkey[0,2,0] - tkey[1,0,2] - tkey[2,2,1] - tkey[3,0,3]
   + 4 lane[0,2] <= 1
 R259: - tkey[0,3,0] - tkey[1,3,2] - tkey[2,3,1] - tkey[3,3,3]
   + 4 lane[0,3] <= 1
 R260: - tkey[1,0,0] - tkey[1,2,2] - tkey[3,0,1] - tkey[3,2,3]
   + 4 lane[1,0] <= 1
 R261: - tkey[0,3,2] - tkey[1,1,0] - tkey[2,3,3] - tkey[3,1,1]
   + 4 lane[1,1] <= 1
 R262: - tkey[1,1,2] - tkey[1,2,0] - tkey[3,1,3] - tkey[3,2,1]
   + 4 lane[1,2] <= 1
 R263: - tkey[0,1,2] - tkey[1,3,0] - tkey[2,1,3] - tkey[3,3,1]
   + 4 lane[1,3] <= 1
 R264: - tkey[0,2,1] - tkey[1,0,3] - tkey[2,0,0] - tkey[2,2,2]
   + 4 lane[2,0] <= 1
 R265: - tkey[0,0,1] - tkey[0,2,3] - tkey[2,0,2] - tkey[2,1,0]
   + 4 lane[2,1] <= 1
 R266: - tkey[1,0,1] - tkey[1,2,3] - tkey[2,2,0] - tkey[3,0,2]
   + 4 lane[2,2] <= 1
 R267: - tkey[0,1,3] - tkey[1,3,1] - tkey[2,3,0] - tkey[3,3,2]
   + 4 lane[2,3] <= 1
 R268: - tkey[1,1,3] - tkey[1,2,1] - tkey[3,0,0] - tkey[3,2,2]
   + 4 lane[3,0] <= 1
 R269: - tkey[0,3,1] - tkey[1,3,3] - tkey[2,3,2] - tkey[3,1,0]
   + 4 lane[3,1] <= 1
 R270: - tkey[0,3,3] - tkey[1,1,1] - tkey[3,1,2] - tkey[3,2,0]
   + 4 lane[3,2] <= 1
 R271: - tkey[0,0,3] - tkey[0,1,1] - tkey[2,1,2] - tkey[3,3,0]
   + 4 lane[3,3] <= 1
 R272: - tbox[0,0,0] + forg[0,0,0] <= 0
 R273: sbox[0,0,0] - rkey[0,0,0] + tkey[0,0,0] >= 0
 R274: sbox[0,0,0] + rkey[0,0,0] - tkey[0,0,0] >= 0
 R275: - sbox[0,0,0] + rkey[0,0,0] + tkey[0,0,0] >= 0
 R276: - tbox[1,0,0] + forg[1,0,0] <= 0
 R277: sbox[1,0,0] - rkey[1,0,0] + tkey[1,0,0] >= 0
 R278: sbox[1,0,0] + rkey[1,0,0] - tkey[1,0,0] >= 0
 R279: - sbox[1,0,0] + rkey[1,0,0] + tkey[1,0,0] >= 0
 R280: - tbox[2,0,0] + forg[2,0,0] <= 0
 R281: - sbox[2,0,0] + rkey[2,0,0] = 0
 R282: - tbox[3,0,0] + forg[3,0,0] <= 0
 R283: - sbox[3,0,0] + rkey[3,0,0] = 0
 R284: - sbox[3,0,1] + rkey[0,0,0] + rkey[2,2,0] >= 0
 R285: sbox[3,0,1] + rkey[0,0,0] - rkey[2,2,0] >= 0
 R286: sbox[3,0,1] - rkey[0,0,0] + rkey[2,2,0] >= 0
 R287: - sbox[0,0,1] + sbox[3,0,1] + rkey[3,1,0] >= 0
 R288: sbox[0,0,1] + sbox[3,0,1] - rkey[3,1,0] >= 0
 R289: sbox[0,0,1] - sbox[3,0,1] + rkey[3,1,0] >= 0
 R290: - sbox[2,0,1] + rkey[1,3,0] + rkey[2,2,0] >= 0
 R291: sbox[2,0,1] + rkey[1,3,0] - rkey[2,2,0] >= 0
 R292: sbox[2,0,1] - rkey[1,3,0] + rkey[2,2,0] >= 0
 R293: sbox[1,0,1] - rkey[0,0,0] = 0
 R294: tbox[0,0,0] + tbox[2,2,0] - tbox[3,0,1] >= 0
 R295: tbox[0,0,0] - tbox[2,2,0] + tbox[3,0,1] >= 0
 R296: - tbox[0,0,0] + tbox[2,2,0] + tbox[3,0,1] >= 0
 R297: - tbox[0,0,1] + tbox[3,0,1] + tbox[3,1,0] >= 0
 R298: tbox[0,0,1] + tbox[3,0,1] - tbox[3,1,0] >= 0
 R299: tbox[0,0,1] - tbox[3,0,1] + tbox[3,1,0] >= 0
 R300: tbox[1,3,0] - tbox[2,0,1] + tbox[2,2,0] >= 0
 R301: tbox[1,3,0] + tbox[2,0,1] - tbox[2,2,0] >= 0
 R302: - tbox[1,3,0] + tbox[2,0,1] + tbox[2,2,0] >= 0
 R303: - tbox[0,0,0] + tbox[1,0,1] = 0
 R304: unkn[0,0,0] - unkn[3,0,1] + forg[3,0,1] <= 0
 R305: unkn[2,2,0] - unkn[3,0,1] + forg[3,0,1] <= 0
 R306: unkn[0,0,0] + unkn[2,2,0] - unkn[3,0,1] + forg[3,0,1] >= 0
 R307: - unkn[0,0,1] + unkn[3,0,1] + forg[0,0,1] - forg[3,0,1] <= 0
 R308: - unkn[0,0,1] + unkn[3,1,0] + forg[0,0,1] <= 0
 R309: - unkn[0,0,1] + unkn[3,0,1] + unkn[3,1,0] + forg[0,0,1]
   - forg[3,0,1] >= 0
 R310: unkn[1,3,0] - unkn[2,0,1] + forg[2,0,1] <= 0
 R311: - unkn[2,0,1] + unkn[2,2,0] + forg[2,0,1] <= 0
 R312: unkn[1,3,0] - unkn[2,0,1] + unkn[2,2,0] + forg[2,0,1] >= 0
 R313: unkn[0,0,0] - unkn[1,0,1] + forg[1,0,1] = 0
 R314: - tbox[0,1,0] + forg[0,1,0] <= 0
 R315: sbox[0,1,0] - rkey[0,1,0] + tkey[0,1,0] >= 0
 R316: sbox[0,1,0] + rkey[0,1,0] - tkey[0,1,0] >= 0
 R317: - sbox[0,1,0] + rkey[0,1,0] + tkey[0,1,0] >= 0
 R318: - tbox[1,1,0] + forg[1,1,0] <= 0
 R319: sbox[1,1,0] - rkey[1,1,0] + tkey[1,1,0] >= 0
 R320: sbox[1,1,0] + rkey[1,1,0] - tkey[1,1,0] >= 0
 R321: - sbox[1,1,0] + rkey[1,1,0] + tkey[1,1,0] >= 0
 R322: - tbox[2,1,0] + forg[2,1,0] <= 0
 R323: - sbox[2,1,0] + rkey[2,1,0] = 0
 R324: - tbox[3,1,0] + forg[3,1,0] <= 0
 R325: - sbox[3,1,0] + rkey[3,1,0] = 0
 R326: - sbox[3,1,1] + rkey[0,1,0] + rkey[2,3,0] >= 0
 R327: sbox[3,1,1] + rkey[0,1,0] - rkey[2,3,0] >= 0
 R328: sbox[3,1,1] - rkey[0,1,0] + rkey[2,3,0] >= 0
 R329: - sbox[0,1,1] + sbox[3,1,1] + rkey[3,2,0] >= 0
 R330: sbox[0,1,1] + sbox[3,1,1] - rkey[3,2,0] >= 0
 R331: sbox[0,1,1] - sbox[3,1,1] + rkey[3,2,0] >= 0
 R332: - sbox[2,1,1] + rkey[1,0,0] + rkey[2,3,0] >= 0
 R333: sbox[2,1,1] + rkey[1,0,0] - rkey[2,3,0] >= 0
 R334: sbox[2,1,1] - rkey[1,0,0] + rkey[2,3,0] >= 0
 R335: sbox[1,1,1] - rkey[0,1,0] = 0
 R336: tbox[0,1,0] + tbox[2,3,0] - tbox[3,1,1] >= 0
 R337: tbox[0,1,0] - tbox[2,3,0] + tbox[3,1,1] >= 0
 R338: - tbox[0,1,0] + tbox[2,3,0] + tbox[3,1,1] >= 0
 R339: - tbox[0,1,1] + tbox[3,1,1] + tbox[3,2,0] >= 0
 R340: tbox[0,1,1] + tbox[3,1,1] - tbox[3,2,0] >= 0
 R341: tbox[0,1,1] - tbox[3,1,1] + tbox[3,2,0] >= 0
 R342: tbox[1,0,0] - tbox[2,1,1] + tbox[2,3,0] >= 0
 R343: tbox[1,0,0] + tbox[2,1,1] - tbox[2,3,0] >= 0
 R344: - tbox[1,0,0] + tbox[2,1,1] + tbox[2,3,0] >= 0
 R345: - tbox[0,1,0] + tbox[1,1,1] = 0
 R346: unkn[0,1,0] - unkn[3,1,1] + forg[3,1,1] <= 0
 R347: unkn[2,3,0] - unkn[3,1,1] + forg[3,1,1] <= 0
 R348: unkn[0,1,0] + unkn[2,3,0] - unkn[3,1,1] + forg[3,1,1] >= 0
 R349: - unkn[0,1,1] + unkn[3,1,1] + forg[0,1,1] - forg[3,1,1] <= 0
 R350: - unkn[0,1,1] + unkn[3,2,0] + forg[0,1,1] <= 0
 R351: - unkn[0,1,1] + unkn[3,1,1] + unkn[3,2,0] + forg[0,1,1]
   - forg[3,1,1] >= 0
 R352: unkn[1,0,0] - unkn[2,1,1] + forg[2,1,1] <= 0
 R353: - unkn[2,1,1] + unkn[2,3,0] + forg[2,1,1] <= 0
 R354: unkn[1,0,0] - unkn[2,1,1] + unkn[2,3,0] + forg[2,1,1] >= 0
 R355: unkn[0,1,0] - unkn[1,1,1] + forg[1,1,1] = 0
 R356: - tbox[0,2,0] + forg[0,2,0] <= 0
 R357: sbox[0,2,0] - rkey[0,2,0] + tkey[0,2,0] >= 0
 R358: sbox[0,2,0] + rkey[0,2,0] - tkey[0,2,0] >= 0
 R359: - sbox[0,2,0] + rkey[0,2,0] + tkey[0,2,0] >= 0
 R360: - tbox[1,2,0] + forg[1,2,0] <= 0
 R361: sbox[1,2,0] - rkey[1,2,0] + tkey[1,2,0] >= 0
 R362: sbox[1,2,0] + rkey[1,2,0] - tkey[1,2,0] >= 0
 R363: - sbox[1,2,0] + rkey[1,2,0] + tkey[1,2,0] >= 0
 R364: - tbox[2,2,0] + forg[2,2,0] <= 0
 R365: - sbox[2,2,0] + rkey[2,2,0] = 0
 R366: - tbox[3,2,0] + forg[3,2,0] <= 0
 R367: - sbox[3,2,0] + rkey[3,2,0] = 0
 R368: - sbox[3,2,1] + rkey[0,2,0] + rkey[2,0,0] >= 0
 R369: sbox[3,2,1] + rkey[0,2,0] - rkey[2,0,0] >= 0
 R370: sbox[3,2,1] - rkey[0,2,0] + rkey[2,0,0] >= 0
 R371: - sbox[0,2,1] + sbox[3,2,1] + rkey[3,3,0] >= 0
 R372: sbox[0,2,1] + sbox[3,2,1] - rkey[3,3,0] >= 0
 R373: sbox[0,2,1] - sbox[3,2,1] + rkey[3,3,0] >= 0
 R374: - sbox[2,2,1] + rkey[1,1,0] + rkey[2,0,0] >= 0
 R375: sbox[2,2,1] + rkey[1,1,0] - rkey[2,0,0] >= 0
 R376: sbox[2,2,1] - rkey[1,1,0] + rkey[2,0,0] >= 0
 R377: sbox[1,2,1] - rkey[0,2,0] = 0
 R378: tbox[0,2,0] + tbox[2,0,0] - tbox[3,2,1] >= 0
 R379: tbox[0,2,0] - tbox[2,0,0] + tbox[3,2,1] >= 0
 R380: - tbox[0,2,0] + tbox[2,0,0] + tbox[3,2,1] >= 0
 R381: - tbox[0,2,1] + tbox[3,2,1] + tbox[3,3,0] >= 0
 R382: tbox[0,2,1] + tbox[3,2,1] - tbox[3,3,0] >= 0
 R383: tbox[0,2,1] - tbox[3,2,1] + tbox[3,3,0] >= 0
 R384: tbox[1,1,0] + tbox[2,0,0] - tbox[2,2,1] >= 0
 R385: tbox[1,1,0] - tbox[2,0,0] + tbox[2,2,1] >= 0
 R386: - tbox[1,1,0] + tbox[2,0,0] + tbox[2,2,1] >= 0
 R387: - tbox[0,2,0] + tbox[1,2,1] = 0
 R388: unkn[0,2,0] - unkn[3,2,1] + forg[3,2,1] <= 0
 R389: unkn[2,0,0] - unkn[3,2,1] + forg[3,2,1] <= 0
 R390: unkn[0,2,0] + unkn[2,0,0] - unkn[3,2,1] + forg[3,2,1] >= 0
 R391: - unkn[0,2,1] + unkn[3,2,1] + forg[0,2,1] - forg[3,2,1] <= 0
 R392: - unkn[0,2,1] + unkn[3,3,0] + forg[0,2,1] <= 0
 R393: - unkn[0,2,1] + unkn[3,2,1] + unkn[3,3,0] + forg[0,2,1]
   - forg[3,2,1] >= 0
 R394: unkn[1,1,0] - unkn[2,2,1] + forg[2,2,1] <= 0
 R395: unkn[2,0,0] - unkn[2,2,1] + forg[2,2,1] <= 0
 R396: unkn[1,1,0] + unkn[2,0,0] - unkn[2,2,1] + forg[2,2,1] >= 0
 R397: unkn[0,2,0] - unkn[1,2,1] + forg[1,2,1] = 0
 R398: - tbox[0,3,0] + forg[0,3,0] <= 0
 R399: sbox[0,3,0] - rkey[0,3,0] + tkey[0,3,0] >= 0
 R400: sbox[0,3,0] + rkey[0,3,0] - tkey[0,3,0] >= 0
 R401: - sbox[0,3,0] + rkey[0,3,0] + tkey[0,3,0] >= 0
 R402: - tbox[1,3,0] + forg[1,3,0] <= 0
 R403: sbox[1,3,0] - rkey[1,3,0] + tkey[1,3,0] >= 0
 R404: sbox[1,3,0] + rkey[1,3,0] - tkey[1,3,0] >= 0
 R405: - sbox[1,3,0] + rkey[1,3,0] + tkey[1,3,0] >= 0
 R406: - tbox[2,3,0] + forg[2,3,0] <= 0
 R407: - sbox[2,3,0] + rkey[2,3,0] = 0
 R408: - tbox[3,3,0] + forg[3,3,0] <= 0
 R409: - sbox[3,3,0] + rkey[3,3,0] = 0
 R410: - sbox[3,3,1] + rkey[0,3,0] + rkey[2,1,0] >= 0
 R411: sbox[3,3,1] + rkey[0,3,0] - rkey[2,1,0] >= 0
 R412: sbox[3,3,1] - rkey[0,3,0] + rkey[2,1,0] >= 0
 R413: - sbox[0,3,1] + sbox[3,3,1] + rkey[3,0,0] >= 0
 R414: sbox[0,3,1] + sbox[3,3,1] - rkey[3,0,0] >= 0
 R415: sbox[0,3,1] - sbox[3,3,1] + rkey[3,0,0] >= 0
 R416: - sbox[2,3,1] + rkey[1,2,0] + rkey[2,1,0] >= 0
 R417: sbox[2,3,1] + rkey[1,2,0] - rkey[2,1,0] >= 0
 R418: sbox[2,3,1] - rkey[1,2,0] + rkey[2,1,0] >= 0
 R419: sbox[1,3,1] - rkey[0,3,0] = 0
 R420: tbox[0,3,0] + tbox[2,1,0] - tbox[3,3,1] >= 0
 R421: tbox[0,3,0] - tbox[2,1,0] + tbox[3,3,1] >= 0
 R422: - tbox[0,3,0] + tbox[2,1,0] + tbox[3,3,1] >= 0
 R423: - tbox[0,3,1] + tbox[3,0,0] + tbox[3,3,1] >= 0
 R424: tbox[0,3,1] - tbox[3,0,0] + tbox[3,3,1] >= 0
 R425: tbox[0,3,1] + tbox[3,0,0] - tbox[3,3,1] >= 0
 R426: tbox[1,2,0] + tbox[2,1,0] - tbox[2,3,1] >= 0
 R427: tbox[1,2,0] - tbox[2,1,0] + tbox[2,3,1] >= 0
 R428: - tbox[1,2,0] + tbox[2,1,0] + tbox[2,3,1] >= 0
 R429: - tbox[0,3,0] + tbox[1,3,1] = 0
 R430: unkn[0,3,0] - unkn[3,3,1] + forg[3,3,1] <= 0
 R431: unkn[2,1,0] - unkn[3,3,1] + forg[3,3,1] <= 0
 R432: unkn[0,3,0] + unkn[2,1,0] - unkn[3,3,1] + forg[3,3,1] >= 0
 R433: - unkn[0,3,1] + unkn[3,3,1] + forg[0,3,1] - forg[3,3,1] <= 0
 R434: - unkn[0,3,1] + unkn[3,0,0] + forg[0,3,1] <= 0
 R435: - unkn[0,3,1] + unkn[3,0,0] + unkn[3,3,1] + forg[0,3,1]
   - forg[3,3,1] >= 0
 R436: unkn[1,2,0] - unkn[2,3,1] + forg[2,3,1] <= 0
 R437: unkn[2,1,0] - unkn[2,3,1] + forg[2,3,1] <= 0
 R438: unkn[1,2,0] + unkn[2,1,0] - unkn[2,3,1] + forg[2,3,1] >= 0
 R439: unkn[0,3,0] - unkn[1,3,1] + forg[1,3,1] = 0
 R440: - tbox[0,0,1] + forg[0,0,1] <= 0
 R441: sbox[0,0,1] - rkey[0,0,1] + tkey[0,0,1] >= 0
 R442: sbox[0,0,1] + rkey[0,0,1] - tkey[0,0,1] >= 0
 R443: - sbox[0,0,1] + rkey[0,0,1] + tkey[0,0,1] >= 0
 R444: - tbox[1,0,1] + forg[1,0,1] <= 0
 R445: sbox[1,0,1] - rkey[1,0,1] + tkey[1,0,1] >= 0
 R446: sbox[1,0,1] + rkey[1,0,1] - tkey[1,0,1] >= 0
 R447: - sbox[1,0,1] + rkey[1,0,1] + tkey[1,0,1] >= 0
 R448: - tbox[2,0,1] + forg[2,0,1] <= 0
 R449: - sbox[2,0,1] + rkey[2,0,1] = 0
 R450: - tbox[3,0,1] + forg[3,0,1] <= 0
 R451: - sbox[3,0,1] + rkey[3,0,1] = 0
 R452: - sbox[3,0,2] + rkey[0,0,1] + rkey[2,2,1] >= 0
 R453: sbox[3,0,2] + rkey[0,0,1] - rkey[2,2,1] >= 0
 R454: sbox[3,0,2] - rkey[0,0,1] + rkey[2,2,1] >= 0
 R455: - sbox[0,0,2] + sbox[3,0,2] + rkey[3,1,1] >= 0
 R456: sbox[0,0,2] + sbox[3,0,2] - rkey[3,1,1] >= 0
 R457: sbox[0,0,2] - sbox[3,0,2] + rkey[3,1,1] >= 0
 R458: - sbox[2,0,2] + rkey[1,3,1] + rkey[2,2,1] >= 0
 R459: sbox[2,0,2] + rkey[1,3,1] - rkey[2,2,1] >= 0
 R460: sbox[2,0,2] - rkey[1,3,1] + rkey[2,2,1] >= 0
 R461: sbox[1,0,2] - rkey[0,0,1] = 0
 R462: tbox[0,0,1] + tbox[2,2,1] - tbox[3,0,2] >= 0
 R463: tbox[0,0,1] - tbox[2,2,1] + tbox[3,0,2] >= 0
 R464: - tbox[0,0,1] + tbox[2,2,1] + tbox[3,0,2] >= 0
 R465: - tbox[0,0,2] + tbox[3,0,2] + tbox[3,1,1] >= 0
 R466: tbox[0,0,2] + tbox[3,0,2] - tbox[3,1,1] >= 0
 R467: tbox[0,0,2] - tbox[3,0,2] + tbox[3,1,1] >= 0
 R468: tbox[1,3,1] - tbox[2,0,2] + tbox[2,2,1] >= 0
 R469: tbox[1,3,1] + tbox[2,0,2] - tbox[2,2,1] >= 0
 R470: - tbox[1,3,1] + tbox[2,0,2] + tbox[2,2,1] >= 0
 R471: - tbox[0,0,1] + tbox[1,0,2] = 0
 R472: unkn[0,0,1] - unkn[3,0,2] + forg[3,0,2] <= 0
 R473: unkn[2,2,1] - unkn[3,0,2] + forg[3,0,2] <= 0
 R474: unkn[0,0,1] + unkn[2,2,1] - unkn[3,0,2] + forg[3,0,2] >= 0
 R475: - unkn[0,0,2] + unkn[3,0,2] + forg[0,0,2] - forg[3,0,2] <= 0
 R476: - unkn[0,0,2] + unkn[3,1,1] + forg[0,0,2] <= 0
 R477: - unkn[0,0,2] + unkn[3,0,2] + unkn[3,1,1] + forg[0,0,2]
   - forg[3,0,2] >= 0
 R478: unkn[1,3,1] - unkn[2,0,2] + forg[2,0,2] <= 0
 R479: - unkn[2,0,2] + unkn[2,2,1] + forg[2,0,2] <= 0
 R480: unkn[1,3,1] - unkn[2,0,2] + unkn[2,2,1] + forg[2,0,2] >= 0
 R481: unkn[0,0,1] - unkn[1,0,2] + forg[1,0,2] = 0
 R482: - tbox[0,1,1] + forg[0,1,1] <= 0
 R483: sbox[0,1,1] - rkey[0,1,1] + tkey[0,1,1] >= 0
 R484: sbox[0,1,1] + rkey[0,1,1] - tkey[0,1,1] >= 0
 R485: - sbox[0,1,1] + rkey[0,1,1] + tkey[0,1,1] >= 0
 R486: - tbox[1,1,1] + forg[1,1,1] <= 0
 R487: sbox[1,1,1] - rkey[1,1,1] + tkey[1,1,1] >= 0
 R488: sbox[1,1,1] + rkey[1,1,1] - tkey[1,1,1] >= 0
 R489: - sbox[1,1,1] + rkey[1,1,1] + tkey[1,1,1] >= 0
 R490: - tbox[2,1,1] + forg[2,1,1] <= 0
 R491: - sbox[2,1,1] + rkey[2,1,1] = 0
 R492: - tbox[3,1,1] + forg[3,1,1] <= 0
 R493: - sbox[3,1,1] + rkey[3,1,1] = 0
 R494: - sbox[3,1,2] + rkey[0,1,1] + rkey[2,3,1] >= 0
 R495: sbox[3,1,2] + rkey[0,1,1] - rkey[2,3,1] >= 0
 R496: sbox[3,1,2] - rkey[0,1,1] + rkey[2,3,1] >= 0
 R497: - sbox[0,1,2] + sbox[3,1,2] + rkey[3,2,1] >= 0
 R498: sbox[0,1,2] + sbox[3,1,2] - rkey[3,2,1] >= 0
 R499: sbox[0,1,2] - sbox[3,1,2] + rkey[3,2,1] >= 0
 R500: - sbox[2,1,2] + rkey[1,0,1] + rkey[2,3,1] >= 0
 R501: sbox[2,1,2] + rkey[1,0,1] - rkey[2,3,1] >= 0
 R502: sbox[2,1,2] - rkey[1,0,1] + rkey[2,3,1] >= 0
 R503: sbox[1,1,2] - rkey[0,1,1] = 0
 R504: tbox[0,1,1] + tbox[2,3,1] - tbox[3,1,2] >= 0
 R505: tbox[0,1,1] - tbox[2,3,1] + tbox[3,1,2] >= 0
 R506: - tbox[0,1,1] + tbox[2,3,1] + tbox[3,1,2] >= 0
 R507: - tbox[0,1,2] + tbox[3,1,2] + tbox[3,2,1] >= 0
 R508: tbox[0,1,2] + tbox[3,1,2] - tbox[3,2,1] >= 0
 R509: tbox[0,1,2] - tbox[3,1,2] + tbox[3,2,1] >= 0
 R510: tbox[1,0,1] - tbox[2,1,2] + tbox[2,3,1] >= 0
 R511: tbox[1,0,1] + tbox[2,1,2] - tbox[2,3,1] >= 0
 R512: - tbox[1,0,1] + tbox[2,1,2] + tbox[2,3,1] >= 0
 R513: - tbox[0,1,1] + tbox[1,1,2] = 0
 R514: unkn[0,1,1] - unkn[3,1,2] + forg[3,1,2] <= 0
 R515: unkn[2,3,1] - unkn[3,1,2] + forg[3,1,2] <= 0
 R516: unkn[0,1,1] + unkn[2,3,1] - unkn[3,1,2] + forg[3,1,2] >= 0
 R517: - unkn[0,1,2] + unkn[3,1,2] + forg[0,1,2] - forg[3,1,2] <= 0
 R518: - unkn[0,1,2] + unkn[3,2,1] + forg[0,1,2] <= 0
 R519: - unkn[0,1,2] + unkn[3,1,2] + unkn[3,2,1] + forg[0,1,2]
   - forg[3,1,2] >= 0
 R520: unkn[1,0,1] - unkn[2,1,2] + forg[2,1,2] <= 0
 R521: - unkn[2,1,2] + unkn[2,3,1] + forg[2,1,2] <= 0
 R522: unkn[1,0,1] - unkn[2,1,2] + unkn[2,3,1] + forg[2,1,2] >= 0
 R523: unkn[0,1,1] - unkn[1,1,2] + forg[1,1,2] = 0
 R524: - tbox[0,2,1] + forg[0,2,1] <= 0
 R525: sbox[0,2,1] - rkey[0,2,1] + tkey[0,2,1] >= 0
 R526: sbox[0,2,1] + rkey[0,2,1] - tkey[0,2,1] >= 0
 R527: - sbox[0,2,1] + rkey[0,2,1] + tkey[0,2,1] >= 0
 R528: - tbox[1,2,1] + forg[1,2,1] <= 0
 R529: sbox[1,2,1] - rkey[1,2,1] + tkey[1,2,1] >= 0
 R530: sbox[1,2,1] + rkey[1,2,1] - tkey[1,2,1] >= 0
 R531: - sbox[1,2,1] + rkey[1,2,1] + tkey[1,2,1] >= 0
 R532: - tbox[2,2,1] + forg[2,2,1] <= 0
 R533: - sbox[2,2,1] + rkey[2,2,1] = 0
 R534: - tbox[3,2,1] + forg[3,2,1] <= 0
 R535: - sbox[3,2,1] + rkey[3,2,1] = 0
 R536: - sbox[3,2,2] + rkey[0,2,1] + rkey[2,0,1] >= 0
 R537: sbox[3,2,2] + rkey[0,2,1] - rkey[2,0,1] >= 0
 R538: sbox[3,2,2] - rkey[0,2,1] + rkey[2,0,1] >= 0
 R539: - sbox[0,2,2] + sbox[3,2,2] + rkey[3,3,1] >= 0
 R540: sbox[0,2,2] + sbox[3,2,2] - rkey[3,3,1] >= 0
 R541: sbox[0,2,2] - sbox[3,2,2] + rkey[3,3,1] >= 0
 R542: - sbox[2,2,2] + rkey[1,1,1] + rkey[2,0,1] >= 0
 R543: sbox[2,2,2] + rkey[1,1,1] - rkey[2,0,1] >= 0
 R544: sbox[2,2,2] - rkey[1,1,1] + rkey[2,0,1] >= 0
 R545: sbox[1,2,2] - rkey[0,2,1] = 0
 R546: tbox[0,2,1] + tbox[2,0,1] - tbox[3,2,2] >= 0
 R547: tbox[0,2,1] - tbox[2,0,1] + tbox[3,2,2] >= 0
 R548: - tbox[0,2,1] + tbox[2,0,1] + tbox[3,2,2] >= 0
 R549: - tbox[0,2,2] + tbox[3,2,2] + tbox[3,3,1] >= 0
 R550: tbox[0,2,2] + tbox[3,2,2] - tbox[3,3,1] >= 0
 R551: tbox[0,2,2] - tbox[3,2,2] + tbox[3,3,1] >= 0
 R552: tbox[1,1,1] + tbox[2,0,1] - tbox[2,2,2] >= 0
 R553: tbox[1,1,1] - tbox[2,0,1] + tbox[2,2,2] >= 0
 R554: - tbox[1,1,1] + tbox[2,0,1] + tbox[2,2,2] >= 0
 R555: - tbox[0,2,1] + tbox[1,2,2] = 0
 R556: unkn[0,2,1] - unkn[3,2,2] + forg[3,2,2] <= 0
 R557: unkn[2,0,1] - unkn[3,2,2] + forg[3,2,2] <= 0
 R558: unkn[0,2,1] + unkn[2,0,1] - unkn[3,2,2] + forg[3,2,2] >= 0
 R559: - unkn[0,2,2] + unkn[3,2,2] + forg[0,2,2] - forg[3,2,2] <= 0
 R560: - unkn[0,2,2] + unkn[3,3,1] + forg[0,2,2] <= 0
 R561: - unkn[0,2,2] + unkn[3,2,2] + unkn[3,3,1] + forg[0,2,2]
   - forg[3,2,2] >= 0
 R562: unkn[1,1,1] - unkn[2,2,2] + forg[2,2,2] <= 0
 R563: unkn[2,0,1] - unkn[2,2,2] + forg[2,2,2] <= 0
 R564: unkn[1,1,1] + unkn[2,0,1] - unkn[2,2,2] + forg[2,2,2] >= 0
 R565: unkn[0,2,1] - unkn[1,2,2] + forg[1,2,2] = 0
 R566: - tbox[0,3,1] + forg[0,3,1] <= 0
 R567: sbox[0,3,1] - rkey[0,3,1] + tkey[0,3,1] >= 0
 R568: sbox[0,3,1] + rkey[0,3,1] - tkey[0,3,1] >= 0
 R569: - sbox[0,3,1] + rkey[0,3,1] + tkey[0,3,1] >= 0
 R570: - tbox[1,3,1] + forg[1,3,1] <= 0
 R571: sbox[1,3,1] - rkey[1,3,1] + tkey[1,3,1] >= 0
 R572: sbox[1,3,1] + rkey[1,3,1] - tkey[1,3,1] >= 0
 R573: - sbox[1,3,1] + rkey[1,3,1] + tkey[1,3,1] >= 0
 R574: - tbox[2,3,1] + forg[2,3,1] <= 0
 R575: - sbox[2,3,1] + rkey[2,3,1] = 0
 R576: - tbox[3,3,1] + forg[3,3,1] <= 0
 R577: - sbox[3,3,1] + rkey[3,3,1] = 0
 R578: - sbox[3,3,2] + rkey[0,3,1] + rkey[2,1,1] >= 0
 R579: sbox[3,3,2] + rkey[0,3,1] - rkey[2,1,1] >= 0
 R580: sbox[3,3,2] - rkey[0,3,1] + rkey[2,1,1] >= 0
 R581: - sbox[0,3,2] + sbox[3,3,2] + rkey[3,0,1] >= 0
 R582: sbox[0,3,2] + sbox[3,3,2] - rkey[3,0,1] >= 0
 R583: sbox[0,3,2] - sbox[3,3,2] + rkey[3,0,1] >= 0
 R584: - sbox[2,3,2] + rkey[1,2,1] + rkey[2,1,1] >= 0
 R585: sbox[2,3,2] + rkey[1,2,1] - rkey[2,1,1] >= 0
 R586: sbox[2,3,2] - rkey[1,2,1] + rkey[2,1,1] >= 0
 R587: sbox[1,3,2] - rkey[0,3,1] = 0
 R588: tbox[0,3,1] + tbox[2,1,1] - tbox[3,3,2] >= 0
 R589: tbox[0,3,1] - tbox[2,1,1] + tbox[3,3,2] >= 0
 R590: - tbox[0,3,1] + tbox[2,1,1] + tbox[3,3,2] >= 0
 R591: - tbox[0,3,2] + tbox[3,0,1] + tbox[3,3,2] >= 0
 R592: tbox[0,3,2] - tbox[3,0,1] + tbox[3,3,2] >= 0
 R593: tbox[0,3,2] + tbox[3,0,1] - tbox[3,3,2] >= 0
 R594: tbox[1,2,1] + tbox[2,1,1] - tbox[2,3,2] >= 0
 R595: tbox[1,2,1] - tbox[2,1,1] + tbox[2,3,2] >= 0
 R596: - tbox[1,2,1] + tbox[2,1,1] + tbox[2,3,2] >= 0
 R597: - tbox[0,3,1] + tbox[1,3,2] = 0
 R598: unkn[0,3,1] - unkn[3,3,2] + forg[3,3,2] <= 0
 R599: unkn[2,1,1] - unkn[3,3,2] + forg[3,3,2] <= 0
 R600: unkn[0,3,1] + unkn[2,1,1] - unkn[3,3,2] + forg[3,3,2] >= 0
 R601: - unkn[0,3,2] + unkn[3,3,2] + forg[0,3,2] - forg[3,3,2] <= 0
 R602: - unkn[0,3,2] + unkn[3,0,1] + forg[0,3,2] <= 0
 R603: - unkn[0,3,2] + unkn[3,0,1] + unkn[3,3,2] + forg[0,3,2]
   - forg[3,3,2] >= 0
 R604: unkn[1,2,1] - unkn[2,3,2] + forg[2,3,2] <= 0
 R605: unkn[2,1,1] - unkn[2,3,2] + forg[2,3,2] <= 0
 R606: unkn[1,2,1] + unkn[2,1,1] - unkn[2,3,2] + forg[2,3,2] >= 0
 R607: unkn[0,3,1] - unkn[1,3,2] + forg[1,3,2] = 0
 R608: - tbox[0,0,2] + forg[0,0,2] <= 0
 R609: sbox[0,0,2] - rkey[0,0,2] + tkey[0,0,2] >= 0
 R610: sbox[0,0,2] + rkey[0,0,2] - tkey[0,0,2] >= 0
 R611: - sbox[0,0,2] + rkey[0,0,2] + tkey[0,0,2] >= 0
 R612: - tbox[1,0,2] + forg[1,0,2] <= 0
 R613: sbox[1,0,2] - rkey[1,0,2] + tkey[1,0,2] >= 0
 R614: sbox[1,0,2] + rkey[1,0,2] - tkey[1,0,2] >= 0
 R615: - sbox[1,0,2] + rkey[1,0,2] + tkey[1,0,2] >= 0
 R616: - tbox[2,0,2] + forg[2,0,2] <= 0
 R617: - sbox[2,0,2] + rkey[2,0,2] = 0
 R618: - tbox[3,0,2] + forg[3,0,2] <= 0
 R619: - sbox[3,0,2] + rkey[3,0,2] = 0
 R620: - sbox[3,0,3] + rkey[0,0,2] + rkey[2,2,2] >= 0
 R621: sbox[3,0,3] + rkey[0,0,2] - rkey[2,2,2] >= 0
 R622: sbox[3,0,3] - rkey[0,0,2] + rkey[2,2,2] >= 0
 R623: - sbox[0,0,3] + sbox[3,0,3] + rkey[3,1,2] >= 0
 R624: sbox[0,0,3] + sbox[3,0,3] - rkey[3,1,2] >= 0
 R625: sbox[0,0,3] - sbox[3,0,3] + rkey[3,1,2] >= 0
 R626: - sbox[2,0,3] + rkey[1,3,2] + rkey[2,2,2] >= 0
 R627: sbox[2,0,3] + rkey[1,3,2] - rkey[2,2,2] >= 0
 R628: sbox[2,0,3] - rkey[1,3,2] + rkey[2,2,2] >= 0
 R629: sbox[1,0,3] - rkey[0,0,2] = 0
 R630: tbox[0,0,2] + tbox[2,2,2] - tbox[3,0,3] >= 0
 R631: tbox[0,0,2] - tbox[2,2,2] + tbox[3,0,3] >= 0
 R632: - tbox[0,0,2] + tbox[2,2,2] + tbox[3,0,3] >= 0
 R633: - tbox[0,0,3] + tbox[3,0,3] + tbox[3,1,2] >= 0
 R634: tbox[0,0,3] + tbox[3,0,3] - tbox[3,1,2] >= 0
 R635: tbox[0,0,3] - tbox[3,0,3] + tbox[3,1,2] >= 0
 R636: tbox[1,3,2] - tbox[2,0,3] + tbox[2,2,2] >= 0
 R637: tbox[1,3,2] + tbox[2,0,3] - tbox[2,2,2] >= 0
 R638: - tbox[1,3,2] + tbox[2,0,3] + tbox[2,2,2] >= 0
 R639: - tbox[0,0,2] + tbox[1,0,3] = 0
 R640: unkn[0,0,2] - unkn[3,0,3] + forg[3,0,3] <= 0
 R641: unkn[2,2,2] - unkn[3,0,3] + forg[3,0,3] <= 0
 R642: unkn[0,0,2] + unkn[2,2,2] - unkn[3,0,3] + forg[3,0,3] >= 0
 R643: - unkn[0,0,3] + unkn[3,0,3] + forg[0,0,3] - forg[3,0,3] <= 0
 R644: - unkn[0,0,3] + unkn[3,1,2] + forg[0,0,3] <= 0
 R645: - unkn[0,0,3] + unkn[3,0,3] + unkn[3,1,2] + forg[0,0,3]
   - forg[3,0,3] >= 0
 R646: unkn[1,3,2] - unkn[2,0,3] + forg[2,0,3] <= 0
 R647: - unkn[2,0,3] + unkn[2,2,2] + forg[2,0,3] <= 0
 R648: unkn[1,3,2] - unkn[2,0,3] + unkn[2,2,2] + forg[2,0,3] >= 0
 R649: unkn[0,0,2] - unkn[1,0,3] + forg[1,0,3] = 0
 R650: - tbox[0,1,2] + forg[0,1,2] <= 0
 R651: sbox[0,1,2] - rkey[0,1,2] + tkey[0,1,2] >= 0
 R652: sbox[0,1,2] + rkey[0,1,2] - tkey[0,1,2] >= 0
 R653: - sbox[0,1,2] + rkey[0,1,2] + tkey[0,1,2] >= 0
 R654: - tbox[1,1,2] + forg[1,1,2] <= 0
 R655: sbox[1,1,2] - rkey[1,1,2] + tkey[1,1,2] >= 0
 R656: sbox[1,1,2] + rkey[1,1,2] - tkey[1,1,2] >= 0
 R657: - sbox[1,1,2] + rkey[1,1,2] + tkey[1,1,2] >= 0
 R658: - tbox[2,1,2] + forg[2,1,2] <= 0
 R659: - sbox[2,1,2] + rkey[2,1,2] = 0
 R660: - tbox[3,1,2] + forg[3,1,2] <= 0
 R661: - sbox[3,1,2] + rkey[3,1,2] = 0
 R662: - sbox[3,1,3] + rkey[0,1,2] + rkey[2,3,2] >= 0
 R663: sbox[3,1,3] + rkey[0,1,2] - rkey[2,3,2] >= 0
 R664: sbox[3,1,3] - rkey[0,1,2] + rkey[2,3,2] >= 0
 R665: - sbox[0,1,3] + sbox[3,1,3] + rkey[3,2,2] >= 0
 R666: sbox[0,1,3] + sbox[3,1,3] - rkey[3,2,2] >= 0
 R667: sbox[0,1,3] - sbox[3,1,3] + rkey[3,2,2] >= 0
 R668: - sbox[2,1,3] + rkey[1,0,2] + rkey[2,3,2] >= 0
 R669: sbox[2,1,3] + rkey[1,0,2] - rkey[2,3,2] >= 0
 R670: sbox[2,1,3] - rkey[1,0,2] + rkey[2,3,2] >= 0
 R671: sbox[1,1,3] - rkey[0,1,2] = 0
 R672: tbox[0,1,2] + tbox[2,3,2] - tbox[3,1,3] >= 0
 R673: tbox[0,1,2] - tbox[2,3,2] + tbox[3,1,3] >= 0
 R674: - tbox[0,1,2] + tbox[2,3,2] + tbox[3,1,3] >= 0
 R675: - tbox[0,1,3] + tbox[3,1,3] + tbox[3,2,2] >= 0
 R676: tbox[0,1,3] + tbox[3,1,3] - tbox[3,2,2] >= 0
 R677: tbox[0,1,3] - tbox[3,1,3] + tbox[3,2,2] >= 0
 R678: tbox[1,0,2] - tbox[2,1,3] + tbox[2,3,2] >= 0
 R679: tbox[1,0,2] + tbox[2,1,3] - tbox[2,3,2] >= 0
 R680: - tbox[1,0,2] + tbox[2,1,3] + tbox[2,3,2] >= 0
 R681: - tbox[0,1,2] + tbox[1,1,3] = 0
 R682: unkn[0,1,2] - unkn[3,1,3] + forg[3,1,3] <= 0
 R683: unkn[2,3,2] - unkn[3,1,3] + forg[3,1,3] <= 0
 R684: unkn[0,1,2] + unkn[2,3,2] - unkn[3,1,3] + forg[3,1,3] >= 0
 R685: - unkn[0,1,3] + unkn[3,1,3] + forg[0,1,3] - forg[3,1,3] <= 0
 R686: - unkn[0,1,3] + unkn[3,2,2] + forg[0,1,3] <= 0
 R687: - unkn[0,1,3] + unkn[3,1,3] + unkn[3,2,2] + forg[0,1,3]
   - forg[3,1,3] >= 0
 R688: unkn[1,0,2] - unkn[2,1,3] + forg[2,1,3] <= 0
 R689: - unkn[2,1,3] + unkn[2,3,2] + forg[2,1,3] <= 0
 R690: unkn[1,0,2] - unkn[2,1,3] + unkn[2,3,2] + forg[2,1,3] >= 0
 R691: unkn[0,1,2] - unkn[1,1,3] + forg[1,1,3] = 0
 R692: - tbox[0,2,2] + forg[0,2,2] <= 0
 R693: sbox[0,2,2] - rkey[0,2,2] + tkey[0,2,2] >= 0
 R694: sbox[0,2,2] + rkey[0,2,2] - tkey[0,2,2] >= 0
 R695: - sbox[0,2,2] + rkey[0,2,2] + tkey[0,2,2] >= 0
 R696: - tbox[1,2,2] + forg[1,2,2] <= 0
 R697: sbox[1,2,2] - rkey[1,2,2] + tkey[1,2,2] >= 0
 R698: sbox[1,2,2] + rkey[1,2,2] - tkey[1,2,2] >= 0
 R699: - sbox[1,2,2] + rkey[1,2,2] + tkey[1,2,2] >= 0
 R700: - tbox[2,2,2] + forg[2,2,2] <= 0
 R701: - sbox[2,2,2] + rkey[2,2,2] = 0
 R702: - tbox[3,2,2] + forg[3,2,2] <= 0
 R703: - sbox[3,2,2] + rkey[3,2,2] = 0
 R704: - sbox[3,2,3] + rkey[0,2,2] + rkey[2,0,2] >= 0
 R705: sbox[3,2,3] + rkey[0,2,2] - rkey[2,0,2] >= 0
 R706: sbox[3,2,3] - rkey[0,2,2] + rkey[2,0,2] >= 0
 R707: - sbox[0,2,3] + sbox[3,2,3] + rkey[3,3,2] >= 0
 R708: sbox[0,2,3] + sbox[3,2,3] - rkey[3,3,2] >= 0
 R709: sbox[0,2,3] - sbox[3,2,3] + rkey[3,3,2] >= 0
 R710: - sbox[2,2,3] + rkey[1,1,2] + rkey[2,0,2] >= 0
 R711: sbox[2,2,3] + rkey[1,1,2] - rkey[2,0,2] >= 0
 R712: sbox[2,2,3] - rkey[1,1,2] + rkey[2,0,2] >= 0
 R713: sbox[1,2,3] - rkey[0,2,2] = 0
 R714: tbox[0,2,2] + tbox[2,0,2] - tbox[3,2,3] >= 0
 R715: tbox[0,2,2] - tbox[2,0,2] + tbox[3,2,3] >= 0
 R716: - tbox[0,2,2] + tbox[2,0,2] + tbox[3,2,3] >= 0
 R717: - tbox[0,2,3] + tbox[3,2,3] + tbox[3,3,2] >= 0
 R718: tbox[0,2,3] + tbox[3,2,3] - tbox[3,3,2] >= 0
 R719: tbox[0,2,3] - tbox[3,2,3] + tbox[3,3,2] >= 0
 R720: tbox[1,1,2] + tbox[2,0,2] - tbox[2,2,3] >= 0
 R721: tbox[1,1,2] - tbox[2,0,2] + tbox[2,2,3] >= 0
 R722: - tbox[1,1,2] + tbox[2,0,2] + tbox[2,2,3] >= 0
 R723: - tbox[0,2,2] + tbox[1,2,3] = 0
 R724: unkn[0,2,2] - unkn[3,2,3] + forg[3,2,3] <= 0
 R725: unkn[2,0,2] - unkn[3,2,3] + forg[3,2,3] <= 0
 R726: unkn[0,2,2] + unkn[2,0,2] - unkn[3,2,3] + forg[3,2,3] >= 0
 R727: - unkn[0,2,3] + unkn[3,2,3] + forg[0,2,3] - forg[3,2,3] <= 0
 R728: - unkn[0,2,3] + unkn[3,3,2] + forg[0,2,3] <= 0
 R729: - unkn[0,2,3] + unkn[3,2,3] + unkn[3,3,2] + forg[0,2,3]
   - forg[3,2,3] >= 0
 R730: unkn[1,1,2] - unkn[2,2,3] + forg[2,2,3] <= 0
 R731: unkn[2,0,2] - unkn[2,2,3] + forg[2,2,3] <= 0
 R732: unkn[1,1,2] + unkn[2,0,2] - unkn[2,2,3] + forg[2,2,3] >= 0
 R733: unkn[0,2,2] - unkn[1,2,3] + forg[1,2,3] = 0
 R734: - tbox[0,3,2] + forg[0,3,2] <= 0
 R735: sbox[0,3,2] - rkey[0,3,2] + tkey[0,3,2] >= 0
 R736: sbox[0,3,2] + rkey[0,3,2] - tkey[0,3,2] >= 0
 R737: - sbox[0,3,2] + rkey[0,3,2] + tkey[0,3,2] >= 0
 R738: - tbox[1,3,2] + forg[1,3,2] <= 0
 R739: sbox[1,3,2] - rkey[1,3,2] + tkey[1,3,2] >= 0
 R740: sbox[1,3,2] + rkey[1,3,2] - tkey[1,3,2] >= 0
 R741: - sbox[1,3,2] + rkey[1,3,2] + tkey[1,3,2] >= 0
 R742: - tbox[2,3,2] + forg[2,3,2] <= 0
 R743: - sbox[2,3,2] + rkey[2,3,2] = 0
 R744: - tbox[3,3,2] + forg[3,3,2] <= 0
 R745: - sbox[3,3,2] + rkey[3,3,2] = 0
 R746: - sbox[3,3,3] + rkey[0,3,2] + rkey[2,1,2] >= 0
 R747: sbox[3,3,3] + rkey[0,3,2] - rkey[2,1,2] >= 0
 R748: sbox[3,3,3] - rkey[0,3,2] + rkey[2,1,2] >= 0
 R749: - sbox[0,3,3] + sbox[3,3,3] + rkey[3,0,2] >= 0
 R750: sbox[0,3,3] + sbox[3,3,3] - rkey[3,0,2] >= 0
 R751: sbox[0,3,3] - sbox[3,3,3] + rkey[3,0,2] >= 0
 R752: - sbox[2,3,3] + rkey[1,2,2] + rkey[2,1,2] >= 0
 R753: sbox[2,3,3] + rkey[1,2,2] - rkey[2,1,2] >= 0
 R754: sbox[2,3,3] - rkey[1,2,2] + rkey[2,1,2] >= 0
 R755: sbox[1,3,3] - rkey[0,3,2] = 0
 R756: tbox[0,3,2] + tbox[2,1,2] - tbox[3,3,3] >= 0
 R757: tbox[0,3,2] - tbox[2,1,2] + tbox[3,3,3] >= 0
 R758: - tbox[0,3,2] + tbox[2,1,2] + tbox[3,3,3] >= 0
 R759: - tbox[0,3,3] + tbox[3,0,2] + tbox[3,3,3] >= 0
 R760: tbox[0,3,3] - tbox[3,0,2] + tbox[3,3,3] >= 0
 R761: tbox[0,3,3] + tbox[3,0,2] - tbox[3,3,3] >= 0
 R762: tbox[1,2,2] + tbox[2,1,2] - tbox[2,3,3] >= 0
 R763: tbox[1,2,2] - tbox[2,1,2] + tbox[2,3,3] >= 0
 R764: - tbox[1,2,2] + tbox[2,1,2] + tbox[2,3,3] >= 0
 R765: - tbox[0,3,2] + tbox[1,3,3] = 0
 R766: unkn[0,3,2] - unkn[3,3,3] + forg[3,3,3] <= 0
 R767: unkn[2,1,2] - unkn[3,3,3] + forg[3,3,3] <= 0
 R768: unkn[0,3,2] + unkn[2,1,2] - unkn[3,3,3] + forg[3,3,3] >= 0
 R769: - unkn[0,3,3] + unkn[3,3,3] + forg[0,3,3] - forg[3,3,3] <= 0
 R770: - unkn[0,3,3] + unkn[3,0,2] + forg[0,3,3] <= 0
 R771: - unkn[0,3,3] + unkn[3,0,2] + unkn[3,3,3] + forg[0,3,3]
   - forg[3,3,3] >= 0
 R772: unkn[1,2,2] - unkn[2,3,3] + forg[2,3,3] <= 0
 R773: unkn[2,1,2] - unkn[2,3,3] + forg[2,3,3] <= 0
 R774: unkn[1,2,2] + unkn[2,1,2] - unkn[2,3,3] + forg[2,3,3] >= 0
 R775: unkn[0,3,2] - unkn[1,3,3] + forg[1,3,3] = 0
 R776: - tbox[0,0,3] + forg[0,0,3] <= 0
 R777: sbox[0,0,3] - rkey[0,0,3] + tkey[0,0,3] >= 0
 R778: sbox[0,0,3] + rkey[0,0,3] - tkey[0,0,3] >= 0
 R779: - sbox[0,0,3] + rkey[0,0,3] + tkey[0,0,3] >= 0
 R780: - tbox[1,0,3] + forg[1,0,3] <= 0
 R781: sbox[1,0,3] - rkey[1,0,3] + tkey[1,0,3] >= 0
 R782: sbox[1,0,3] + rkey[1,0,3] - tkey[1,0,3] >= 0
 R783: - sbox[1,0,3] + rkey[1,0,3] + tkey[1,0,3] >= 0
 R784: - tbox[2,0,3] + forg[2,0,3] <= 0
 R785: - sbox[2,0,3] + rkey[2,0,3] = 0
 R786: - tbox[3,0,3] + forg[3,0,3] <= 0
 R787: - sbox[3,0,3] + rkey[3,0,3] = 0
 R788: - sbox[3,0,4] + rkey[0,0,3] + rkey[2,2,3] >= 0
 R789: sbox[3,0,4] + rkey[0,0,3] - rkey[2,2,3] >= 0
 R790: sbox[3,0,4] - rkey[0,0,3] + rkey[2,2,3] >= 0
 R791: - sbox[0,0,4] + sbox[3,0,4] + rkey[3,1,3] >= 0
 R792: sbox[0,0,4] + sbox[3,0,4] - rkey[3,1,3] >= 0
 R793: sbox[0,0,4] - sbox[3,0,4] + rkey[3,1,3] >= 0
 R794: - sbox[2,0,4] + rkey[1,3,3] + rkey[2,2,3] >= 0
 R795: sbox[2,0,4] + rkey[1,3,3] - rkey[2,2,3] >= 0
 R796: sbox[2,0,4] - rkey[1,3,3] + rkey[2,2,3] >= 0
 R797: sbox[1,0,4] - rkey[0,0,3] = 0
 R798: tbox[0,0,3] + tbox[2,2,3] - tbox[3,0,4] >= 0
 R799: tbox[0,0,3] - tbox[2,2,3] + tbox[3,0,4] >= 0
 R800: - tbox[0,0,3] + tbox[2,2,3] + tbox[3,0,4] >= 0
 R801: - tbox[0,0,4] + tbox[3,0,4] + tbox[3,1,3] >= 0
 R802: tbox[0,0,4] + tbox[3,0,4] - tbox[3,1,3] >= 0
 R803: tbox[0,0,4] - tbox[3,0,4] + tbox[3,1,3] >= 0
 R804: tbox[1,3,3] - tbox[2,0,4] + tbox[2,2,3] >= 0
 R805: tbox[1,3,3] + tbox[2,0,4] - tbox[2,2,3] >= 0
 R806: - tbox[1,3,3] + tbox[2,0,4] + tbox[2,2,3] >= 0
 R807: - tbox[0,0,3] + tbox[1,0,4] = 0
 R808: unkn[0,0,3] - unkn[3,0,4] + forg[3,0,4] <= 0
 R809: unkn[2,2,3] - unkn[3,0,4] + forg[3,0,4] <= 0
 R810: unkn[0,0,3] + unkn[2,2,3] - unkn[3,0,4] + forg[3,0,4] >= 0
 R811: - unkn[0,0,4] + unkn[3,0,4] + forg[0,0,4] - forg[3,0,4] <= 0
 R812: - unkn[0,0,4] + unkn[3,1,3] + forg[0,0,4] <= 0
 R813: - unkn[0,0,4] + unkn[3,0,4] + unkn[3,1,3] + forg[0,0,4]
   - forg[3,0,4] >= 0
 R814: unkn[1,3,3] - unkn[2,0,4] + forg[2,0,4] <= 0
 R815: - unkn[2,0,4] + unkn[2,2,3] + forg[2,0,4] <= 0
 R816: unkn[1,3,3] - unkn[2,0,4] + unkn[2,2,3] + forg[2,0,4] >= 0
 R817: unkn[0,0,3] - unkn[1,0,4] + forg[1,0,4] = 0
 R818: - tbox[0,1,3] + forg[0,1,3] <= 0
 R819: sbox[0,1,3] - rkey[0,1,3] + tkey[0,1,3] >= 0
 R820: sbox[0,1,3] + rkey[0,1,3] - tkey[0,1,3] >= 0
 R821: - sbox[0,1,3] + rkey[0,1,3] + tkey[0,1,3] >= 0
 R822: - tbox[1,1,3] + forg[1,1,3] <= 0
 R823: sbox[1,1,3] - rkey[1,1,3] + tkey[1,1,3] >= 0
 R824: sbox[1,1,3] + rkey[1,1,3] - tkey[1,1,3] >= 0
 R825: - sbox[1,1,3] + rkey[1,1,3] + tkey[1,1,3] >= 0
 R826: - tbox[2,1,3] + forg[2,1,3] <= 0
 R827: - sbox[2,1,3] + rkey[2,1,3] = 0
 R828: - tbox[3,1,3] + forg[3,1,3] <= 0
 R829: - sbox[3,1,3] + rkey[3,1,3] = 0
 R830: - sbox[3,1,4] + rkey[0,1,3] + rkey[2,3,3] >= 0
 R831: sbox[3,1,4] + rkey[0,1,3] - rkey[2,3,3] >= 0
 R832: sbox[3,1,4] - rkey[0,1,3] + rkey[2,3,3] >= 0
 R833: - sbox[0,1,4] + sbox[3,1,4] + rkey[3,2,3] >= 0
 R834: sbox[0,1,4] + sbox[3,1,4] - rkey[3,2,3] >= 0
 R835: sbox[0,1,4] - sbox[3,1,4] + rkey[3,2,3] >= 0
 R836: - sbox[2,1,4] + rkey[1,0,3] + rkey[2,3,3] >= 0
 R837: sbox[2,1,4] + rkey[1,0,3] - rkey[2,3,3] >= 0
 R838: sbox[2,1,4] - rkey[1,0,3] + rkey[2,3,3] >= 0
 R839: sbox[1,1,4] - rkey[0,1,3] = 0
 R840: tbox[0,1,3] + tbox[2,3,3] - tbox[3,1,4] >= 0
 R841: tbox[0,1,3] - tbox[2,3,3] + tbox[3,1,4] >= 0
 R842: - tbox[0,1,3] + tbox[2,3,3] + tbox[3,1,4] >= 0
 R843: - tbox[0,1,4] + tbox[3,1,4] + tbox[3,2,3] >= 0
 R844: tbox[0,1,4] + tbox[3,1,4] - tbox[3,2,3] >= 0
 R845: tbox[0,1,4] - tbox[3,1,4] + tbox[3,2,3] >= 0
 R846: tbox[1,0,3] - tbox[2,1,4] + tbox[2,3,3] >= 0
 R847: tbox[1,0,3] + tbox[2,1,4] - tbox[2,3,3] >= 0
 R848: - tbox[1,0,3] + tbox[2,1,4] + tbox[2,3,3] >= 0
 R849: - tbox[0,1,3] + tbox[1,1,4] = 0
 R850: unkn[0,1,3] - unkn[3,1,4] + forg[3,1,4] <= 0
 R851: unkn[2,3,3] - unkn[3,1,4] + forg[3,1,4] <= 0
 R852: unkn[0,1,3] + unkn[2,3,3] - unkn[3,1,4] + forg[3,1,4] >= 0
 R853: - unkn[0,1,4] + unkn[3,1,4] + forg[0,1,4] - forg[3,1,4] <= 0
 R854: - unkn[0,1,4] + unkn[3,2,3] + forg[0,1,4] <= 0
 R855: - unkn[0,1,4] + unkn[3,1,4] + unkn[3,2,3] + forg[0,1,4]
   - forg[3,1,4] >= 0
 R856: unkn[1,0,3] - unkn[2,1,4] + forg[2,1,4] <= 0
 R857: - unkn[2,1,4] + unkn[2,3,3] + forg[2,1,4] <= 0
 R858: unkn[1,0,3] - unkn[2,1,4] + unkn[2,3,3] + forg[2,1,4] >= 0
 R859: unkn[0,1,3] - unkn[1,1,4] + forg[1,1,4] = 0
 R860: - tbox[0,2,3] + forg[0,2,3] <= 0
 R861: sbox[0,2,3] - rkey[0,2,3] + tkey[0,2,3] >= 0
 R862: sbox[0,2,3] + rkey[0,2,3] - tkey[0,2,3] >= 0
 R863: - sbox[0,2,3] + rkey[0,2,3] + tkey[0,2,3] >= 0
 R864: - tbox[1,2,3] + forg[1,2,3] <= 0
 R865: sbox[1,2,3] - rkey[1,2,3] + tkey[1,2,3] >= 0
 R866: sbox[1,2,3] + rkey[1,2,3] - tkey[1,2,3] >= 0
 R867: - sbox[1,2,3] + rkey[1,2,3] + tkey[1,2,3] >= 0
 R868: - tbox[2,2,3] + forg[2,2,3] <= 0
 R869: - sbox[2,2,3] + rkey[2,2,3] = 0
 R870: - tbox[3,2,3] + forg[3,2,3] <= 0
 R871: - sbox[3,2,3] + rkey[3,2,3] = 0
 R872: - sbox[3,2,4] + rkey[0,2,3] + rkey[2,0,3] >= 0
 R873: sbox[3,2,4] + rkey[0,2,3] - rkey[2,0,3] >= 0
 R874: sbox[3,2,4] - rkey[0,2,3] + rkey[2,0,3] >= 0
 R875: - sbox[0,2,4] + sbox[3,2,4] + rkey[3,3,3] >= 0
 R876: sbox[0,2,4] + sbox[3,2,4] - rkey[3,3,3] >= 0
 R877: sbox[0,2,4] - sbox[3,2,4] + rkey[3,3,3] >= 0
 R878: - sbox[2,2,4] + rkey[1,1,3] + rkey[2,0,3] >= 0
 R879: sbox[2,2,4] + rkey[1,1,3] - rkey[2,0,3] >= 0
 R880: sbox[2,2,4] - rkey[1,1,3] + rkey[2,0,3] >= 0
 R881: sbox[1,2,4] - rkey[0,2,3] = 0
 R882: tbox[0,2,3] + tbox[2,0,3] - tbox[3,2,4] >= 0
 R883: tbox[0,2,3] - tbox[2,0,3] + tbox[3,2,4] >= 0
 R884: - tbox[0,2,3] + tbox[2,0,3] + tbox[3,2,4] >= 0
 R885: - tbox[0,2,4] + tbox[3,2,4] + tbox[3,3,3] >= 0
 R886: tbox[0,2,4] + tbox[3,2,4] - tbox[3,3,3] >= 0
 R887: tbox[0,2,4] - tbox[3,2,4] + tbox[3,3,3] >= 0
 R888: tbox[1,1,3] + tbox[2,0,3] - tbox[2,2,4] >= 0
 R889: tbox[1,1,3] - tbox[2,0,3] + tbox[2,2,4] >= 0
 R890: - tbox[1,1,3] + tbox[2,0,3] + tbox[2,2,4] >= 0
 R891: - tbox[0,2,3] + tbox[1,2,4] = 0
 R892: unkn[0,2,3] - unkn[3,2,4] + forg[3,2,4] <= 0
 R893: unkn[2,0,3] - unkn[3,2,4] + forg[3,2,4] <= 0
 R894: unkn[0,2,3] + unkn[2,0,3] - unkn[3,2,4] + forg[3,2,4] >= 0
 R895: - unkn[0,2,4] + unkn[3,2,4] + forg[0,2,4] - forg[3,2,4] <= 0
 R896: - unkn[0,2,4] + unkn[3,3,3] + forg[0,2,4] <= 0
 R897: - unkn[0,2,4] + unkn[3,2,4] + unkn[3,3,3] + forg[0,2,4]
   - forg[3,2,4] >= 0
 R898: unkn[1,1,3] - unkn[2,2,4] + forg[2,2,4] <= 0
 R899: unkn[2,0,3] - unkn[2,2,4] + forg[2,2,4] <= 0
 R900: unkn[1,1,3] + unkn[2,0,3] - unkn[2,2,4] + forg[2,2,4] >= 0
 R901: unkn[0,2,3] - unkn[1,2,4] + forg[1,2,4] = 0
 R902: - tbox[0,3,3] + forg[0,3,3] <= 0
 R903: sbox[0,3,3] - rkey[0,3,3] + tkey[0,3,3] >= 0
 R904: sbox[0,3,3] + rkey[0,3,3] - tkey[0,3,3] >= 0
 R905: - sbox[0,3,3] + rkey[0,3,3] + tkey[0,3,3] >= 0
 R906: - tbox[1,3,3] + forg[1,3,3] <= 0
 R907: sbox[1,3,3] - rkey[1,3,3] + tkey[1,3,3] >= 0
 R908: sbox[1,3,3] + rkey[1,3,3] - tkey[1,3,3] >= 0
 R909: - sbox[1,3,3] + rkey[1,3,3] + tkey[1,3,3] >= 0
 R910: - tbox[2,3,3] + forg[2,3,3] <= 0
 R911: - sbox[2,3,3] + rkey[2,3,3] = 0
 R912: - tbox[3,3,3] + forg[3,3,3] <= 0
 R913: - sbox[3,3,3] + rkey[3,3,3] = 0
 R914: - sbox[3,3,4] + rkey[0,3,3] + rkey[2,1,3] >= 0
 R915: sbox[3,3,4] + rkey[0,3,3] - rkey[2,1,3] >= 0
 R916: sbox[3,3,4] - rkey[0,3,3] + rkey[2,1,3] >= 0
 R917: - sbox[0,3,4] + sbox[3,3,4] + rkey[3,0,3] >= 0
 R918: sbox[0,3,4] + sbox[3,3,4] - rkey[3,0,3] >= 0
 R919: sbox[0,3,4] - sbox[3,3,4] + rkey[3,0,3] >= 0
 R920: - sbox[2,3,4] + rkey[1,2,3] + rkey[2,1,3] >= 0
 R921: sbox[2,3,4] + rkey[1,2,3] - rkey[2,1,3] >= 0
 R922: sbox[2,3,4] - rkey[1,2,3] + rkey[2,1,3] >= 0
 R923: sbox[1,3,4] - rkey[0,3,3] = 0
 R924: tbox[0,3,3] + tbox[2,1,3] - tbox[3,3,4] >= 0
 R925: tbox[0,3,3] - tbox[2,1,3] + tbox[3,3,4] >= 0
 R926: - tbox[0,3,3] + tbox[2,1,3] + tbox[3,3,4] >= 0
 R927: - tbox[0,3,4] + tbox[3,0,3] + tbox[3,3,4] >= 0
 R928: tbox[0,3,4] - tbox[3,0,3] + tbox[3,3,4] >= 0
 R929: tbox[0,3,4] + tbox[3,0,3] - tbox[3,3,4] >= 0
 R930: tbox[1,2,3] + tbox[2,1,3] - tbox[2,3,4] >= 0
 R931: tbox[1,2,3] - tbox[2,1,3] + tbox[2,3,4] >= 0
 R932: - tbox[1,2,3] + tbox[2,1,3] + tbox[2,3,4] >= 0
 R933: - tbox[0,3,3] + tbox[1,3,4] = 0
 R934: unkn[0,3,3] - unkn[3,3,4] + forg[3,3,4] <= 0
 R935: unkn[2,1,3] - unkn[3,3,4] + forg[3,3,4] <= 0
 R936: unkn[0,3,3] + unkn[2,1,3] - unkn[3,3,4] + forg[3,3,4] >= 0
 R937: - unkn[0,3,4] + unkn[3,3,4] + forg[0,3,4] - forg[3,3,4] <= 0
 R938: - unkn[0,3,4] + unkn[3,0,3] + forg[0,3,4] <= 0
 R939: - unkn[0,3,4] + unkn[3,0,3] + unkn[3,3,4] + forg[0,3,4]
   - forg[3,3,4] >= 0
 R940: unkn[1,2,3] - unkn[2,3,4] + forg[2,3,4] <= 0
 R941: unkn[2,1,3] - unkn[2,3,4] + forg[2,3,4] <= 0
 R942: unkn[1,2,3] + unkn[2,1,3] - unkn[2,3,4] + forg[2,3,4] >= 0
 R943: unkn[0,3,3] - unkn[1,3,4] + forg[1,3,4] = 0
 R944: sbox[0,0,0] + sbox[0,1,0] + sbox[0,2,0] + sbox[0,3,0] + sbox[1,0,0]
   + sbox[1,1,0] + sbox[1,2,0] + sbox[1,3,0] + sbox[2,0,0] + sbox[2,1,0]
   + sbox[2,2,0] + sbox[2,3,0] + sbox[3,0,0] + sbox[3,1,0] + sbox[3,2,0]
   + sbox[3,3,0] + tkey[0,0,0] + tkey[0,1,0] + tkey[0,2,0] + tkey[0,3,0]
   + tkey[1,0,0] + tkey[1,1,0] + tkey[1,2,0] + tkey[1,3,0] + tkey[2,0,0]
   + tkey[2,1,0] + tkey[2,2,0] + tkey[2,3,0] + tkey[3,0,0] + tkey[3,1,0]
   + tkey[3,2,0] + tkey[3,3,0] >= 1
 R945: sbox[0,0,1] + sbox[0,1,1] + sbox[0,2,1] + sbox[0,3,1] + sbox[1,0,1]
   + sbox[1,1,1] + sbox[1,2,1] + sbox[1,3,1] + sbox[2,0,1] + sbox[2,1,1]
   + sbox[2,2,1] + sbox[2,3,1] + sbox[3,0,1] + sbox[3,1,1] + sbox[3,2,1]
   + sbox[3,3,1] + tkey[0,0,1] + tkey[0,1,1] + tkey[0,2,1] + tkey[0,3,1]
   + tkey[1,0,1] + tkey[1,1,1] + tkey[1,2,1] + tkey[1,3,1] + tkey[2,0,1]
   + tkey[2,1,1] + tkey[2,2,1] + tkey[2,3,1] + tkey[3,0,1] + tkey[3,1,1]
   + tkey[3,2,1] + tkey[3,3,1] >= 1
 R946: sbox[0,0,2] + sbox[0,1,2] + sbox[0,2,2] + sbox[0,3,2] + sbox[1,0,2]
   + sbox[1,1,2] + sbox[1,2,2] + sbox[1,3,2] + sbox[2,0,2] + sbox[2,1,2]
   + sbox[2,2,2] + sbox[2,3,2] + sbox[3,0,2] + sbox[3,1,2] + sbox[3,2,2]
   + sbox[3,3,2] + tkey[0,0,2] + tkey[0,1,2] + tkey[0,2,2] + tkey[0,3,2]
   + tkey[1,0,2] + tkey[1,1,2] + tkey[1,2,2] + tkey[1,3,2] + tkey[2,0,2]
   + tkey[2,1,2] + tkey[2,2,2] + tkey[2,3,2] + tkey[3,0,2] + tkey[3,1,2]
   + tkey[3,2,2] + tkey[3,3,2] >= 1
 R947: sbox[0,0,3] + sbox[0,1,3] + sbox[0,2,3] + sbox[0,3,3] + sbox[1,0,3]
   + sbox[1,1,3] + sbox[1,2,3] + sbox[1,3,3] + sbox[2,0,3] + sbox[2,1,3]
   + sbox[2,2,3] + sbox[2,3,3] + sbox[3,0,3] + sbox[3,1,3] + sbox[3,2,3]
   + sbox[3,3,3] + tkey[0,0,3] + tkey[0,1,3] + tkey[0,2,3] + tkey[0,3,3]
   + tkey[1,0,3] + tkey[1,1,3] + tkey[1,2,3] + tkey[1,3,3] + tkey[2,0,3]
   + tkey[2,1,3] + tkey[2,2,3] + tkey[2,3,3] + tkey[3,0,3] + tkey[3,1,3]
   + tkey[3,2,3] + tkey[3,3,3] >= 1
 R948: sbox[0,0,0] + sbox[0,1,0] + sbox[0,2,0] + sbox[0,3,0] + sbox[1,0,0]
   + sbox[1,1,0] + sbox[1,2,0] + sbox[1,3,0] + sbox[2,0,0] + sbox[2,1,0]
   + sbox[2,2,0] + sbox[2,3,0] + sbox[3,0,0] + sbox[3,1,0] + sbox[3,2,0]
   + sbox[3,3,0] = 0
 R949: sbox[0,0,4] + sbox[0,1,4] + sbox[0,2,4] + sbox[0,3,4] + sbox[1,0,4]
   + sbox[1,1,4] + sbox[1,2,4] + sbox[1,3,4] + sbox[2,0,4] + sbox[2,1,4]
   + sbox[2,2,4] + sbox[2,3,4] + sbox[3,0,4] + sbox[3,1,4] + sbox[3,2,4]
   + sbox[3,3,4] = 0
 R950: tbox[0,0,0] = 1
 R951: unkn[0,0,0] - forg[0,0,0] = 0
 R952: tbox[0,1,0] = 0
 R953: unkn[0,1,0] - forg[0,1,0] = 0
 R954: tbox[0,2,0] = 0
 R955: unkn[0,2,0] - forg[0,2,0] = 0
 R956: tbox[0,3,0] = 0
 R957: unkn[0,3,0] - forg[0,3,0] = 0
 R958: tbox[1,0,0] = 0
 R959: unkn[1,0,0] - forg[1,0,0] = 0
 R960: tbox[1,1,0] = 0
 R961: unkn[1,1,0] - forg[1,1,0] = 0
 R962: tbox[1,2,0] = 0
 R963: unkn[1,2,0] - forg[1,2,0] = 0
 R964: tbox[1,3,0] = 0
 R965: unkn[1,3,0] - forg[1,3,0] = 0
 R966: tbox[2,0,0] = 0
 R967: unkn[2,0,0] - forg[2,0,0] = 0
 R968: tbox[2,1,0] = 0
 R969: unkn[2,1,0] - forg[2,1,0] = 0
 R970: tbox[2,2,0] = 0
 R971: unkn[2,2,0] - forg[2,2,0] = 0
 R972: tbox[2,3,0] = 0
 R973: unkn[2,3,0] - forg[2,3,0] = 0
 R974: tbox[3,0,0] = 0
 R975: unkn[3,0,0] - forg[3,0,0] = 0
 R976: tbox[3,1,0] = 0
 R977: unkn[3,1,0] - forg[3,1,0] = 0
 R978: tbox[3,2,0] = 0
 R979: unkn[3,2,0] - forg[3,2,0] = 0
 R980: tbox[3,3,0] = 0
 R981: unkn[3,3,0] - forg[3,3,0] = 0
Bounds
 cost[0,0,0] <= 2
 cost[0,0,1] <= 2
 cost[0,0,2] <= 2
 cost[0,0,3] <= 2
 cost[0,0,4] <= 2
 cost[0,1,0] <= 2
 cost[0,1,1] <= 2
 cost[0,1,2] <= 2
 cost[0,1,3] <= 2
 cost[0,1,4] <= 2
 cost[0,2,0] <= 2
 cost[0,2,1] <= 2
 cost[0,2,2] <= 2
 cost[0,2,3] <= 2
 cost[0,2,4] <= 2
 cost[0,3,0] <= 2
 cost[0,3,1] <= 2
 cost[0,3,2] <= 2
 cost[0,3,3] <= 2
 cost[0,3,4] <= 2
 cost[1,0,0] <= 2
 cost[1,0,1] <= 2
 cost[1,0,2] <= 2
 cost[1,0,3] <= 2
 cost[1,0,4] <= 2
 cost[1,1,0] <= 2
 cost[1,1,1] <= 2
 cost[1,1,2] <= 2
 cost[1,1,3] <= 2
 cost[1,1,4] <= 2
 cost[1,2,0] <= 2
 cost[1,2,1] <= 2
 cost[1,2,2] <= 2
 cost[1,2,3] <= 2
 cost[1,2,4] <= 2
 cost[1,3,0] <= 2
 cost[1,3,1] <= 2
 cost[1,3,2] <= 2
 cost[1,3,3] <= 2
 cost[1,3,4] <= 2
 cost[2,0,0] <= 2
 cost[2,0,1] <= 2
 cost[2,0,2] <= 2
 cost[2,0,3] <= 2
 cost[2,0,4] <= 2
 cost[2,1,0] <= 2
 cost[2,1,1] <= 2
 cost[2,1,2] <= 2
 cost[2,1,3] <= 2
 cost[2,1,4] <= 2
 cost[2,2,0] <= 2
 cost[2,2,1] <= 2
 cost[2,2,2] <= 2
 cost[2,2,3] <= 2
 cost[2,2,4] <= 2
 cost[2,3,0] <= 2
 cost[2,3,1] <= 2
 cost[2,3,2] <= 2
 cost[2,3,3] <= 2
 cost[2,3,4] <= 2
 cost[3,0,0] <= 2
 cost[3,0,1] <= 2
 cost[3,0,2] <= 2
 cost[3,0,3] <= 2
 cost[3,0,4] <= 2
 cost[3,1,0] <= 2
 cost[3,1,1] <= 2
 cost[3,1,2] <= 2
 cost[3,1,3] <= 2
 cost[3,1,4] <= 2
 cost[3,2,0] <= 2
 cost[3,2,1] <= 2
 cost[3,2,2] <= 2
 cost[3,2,3] <= 2
 cost[3,2,4] <= 2
 cost[3,3,0] <= 2
 cost[3,3,1] <= 2
 cost[3,3,2] <= 2
 cost[3,3,3] <= 2
 cost[3,3,4] <= 2
Binaries
 sbox[0,0,0] sbox[0,0,1] sbox[0,0,2] sbox[0,0,3] sbox[0,0,4] sbox[0,1,0]
 sbox[0,1,1] sbox[0,1,2] sbox[0,1,3] sbox[0,1,4] sbox[0,2,0] sbox[0,2,1]
 sbox[0,2,2] sbox[0,2,3] sbox[0,2,4] sbox[0,3,0] sbox[0,3,1] sbox[0,3,2]
 sbox[0,3,3] sbox[0,3,4] sbox[1,0,0] sbox[1,0,1] sbox[1,0,2] sbox[1,0,3]
 sbox[1,0,4] sbox[1,1,0] sbox[1,1,1] sbox[1,1,2] sbox[1,1,3] sbox[1,1,4]
 sbox[1,2,0] sbox[1,2,1] sbox[1,2,2] sbox[1,2,3] sbox[1,2,4] sbox[1,3,0]
 sbox[1,3,1] sbox[1,3,2] sbox[1,3,3] sbox[1,3,4] sbox[2,0,0] sbox[2,0,1]
 sbox[2,0,2] sbox[2,0,3] sbox[2,0,4] sbox[2,1,0] sbox[2,1,1] sbox[2,1,2]
 sbox[2,1,3] sbox[2,1,4] sbox[2,2,0] sbox[2,2,1] sbox[2,2,2] sbox[2,2,3]
 sbox[2,2,4] sbox[2,3,0] sbox[2,3,1] sbox[2,3,2] sbox[2,3,3] sbox[2,3,4]
 sbox[3,0,0] sbox[3,0,1] sbox[3,0,2] sbox[3,0,3] sbox[3,0,4] sbox[3,1,0]
 sbox[3,1,1] sbox[3,1,2] sbox[3,1,3] sbox[3,1,4] sbox[3,2,0] sbox[3,2,1]
 sbox[3,2,2] sbox[3,2,3] sbox[3,2,4] sbox[3,3,0] sbox[3,3,1] sbox[3,3,2]
 sbox[3,3,3] sbox[3,3,4] rkey[0,0,0] rkey[0,0,1] rkey[0,0,2] rkey[0,0,3]
 rkey[0,0,4] rkey[0,1,0] rkey[0,1,1] rkey[0,1,2] rkey[0,1,3] rkey[0,1,4]
 rkey[0,2,0] rkey[0,2,1] rkey[0,2,2] rkey[0,2,3] rkey[0,2,4] rkey[0,3,0]
 rkey[0,3,1] rkey[0,3,2] rkey[0,3,3] rkey[0,3,4] rkey[1,0,0] rkey[1,0,1]
 rkey[1,0,2] rkey[1,0,3] rkey[1,0,4] rkey[1,1,0] rkey[1,1,1] rkey[1,1,2]
 rkey[1,1,3] rkey[1,1,4] rkey[1,2,0] rkey[1,2,1] rkey[1,2,2] rkey[1,2,3]
 rkey[1,2,4] rkey[1,3,0] rkey[1,3,1] rkey[1,3,2] rkey[1,3,3] rkey[1,3,4]
 rkey[2,0,0] rkey[2,0,1] rkey[2,0,2] rkey[2,0,3] rkey[2,0,4] rkey[2,1,0]
 rkey[2,1,1] rkey[2,1,2] rkey[2,1,3] rkey[2,1,4] rkey[2,2,0] rkey[2,2,1]
 rkey[2,2,2] rkey[2,2,3] rkey[2,2,4] rkey[2,3,0] rkey[2,3,1] rkey[2,3,2]
 rkey[2,3,3] rkey[2,3,4] rkey[3,0,0] rkey[3,0,1] rkey[3,0,2] rkey[3,0,3]
 rkey[3,0,4] rkey[3,1,0] rkey[3,1,1] rkey[3,1,2] rkey[3,1,3] rkey[3,1,4]
 rkey[3,2,0] rkey[3,2,1] rkey[3,2,2] rkey[3,2,3] rkey[3,2,4] rkey[3,3,0]
 rkey[3,3,1] rkey[3,3,2] rkey[3,3,3] rkey[3,3,4] tkey[0,0,0] tkey[0,0,1]
 tkey[0,0,2] tkey[0,0,3] tkey[0,0,4] tkey[0,1,0] tkey[0,1,1] tkey[0,1,2]
 tkey[0,1,3] tkey[0,1,4] tkey[0,2,0] tkey[0,2,1] tkey[0,2,2] tkey[0,2,3]
 tkey[0,2,4] tkey[0,3,0] tkey[0,3,1] tkey[0,3,2] tkey[0,3,3] tkey[0,3,4]
 tkey[1,0,0] tkey[1,0,1] tkey[1,0,2] tkey[1,0,3] tkey[1,0,4] tkey[1,1,0]
 tkey[1,1,1] tkey[1,1,2] tkey[1,1,3] tkey[1,1,4] tkey[1,2,0] tkey[1,2,1]
 tkey[1,2,2] tkey[1,2,3] tkey[1,2,4] tkey[1,3,0] tkey[1,3,1] tkey[1,3,2]
 tkey[1,3,3] tkey[1,3,4] tkey[2,0,0] tkey[2,0,1] tkey[2,0,2] tkey[2,0,3]
 tkey[2,0,4] tkey[2,1,0] tkey[2,1,1] tkey[2,1,2] tkey[2,1,3] tkey[2,1,4]
 tkey[2,2,0] tkey[2,2,1] tkey[2,2,2] tkey[2,2,3] tkey[2,2,4] tkey[2,3,0]
 tkey[2,3,1] tkey[2,3,2] tkey[2,3,3] tkey[2,3,4] tkey[3,0,0] tkey[3,0,1]
 tkey[3,0,2] tkey[3,0,3] tkey[3,0,4] tkey[3,1,0] tkey[3,1,1] tkey[3,1,2]
 tkey[3,1,3] tkey[3,1,4] tkey[3,2,0] tkey[3,2,1] tkey[3,2,2] tkey[3,2,3]
 tkey[3,2,4] tkey[3,3,0] tkey[3,3,1] tkey[3,3,2] tkey[3,3,3] tkey[3,3,4]
 tbox[0,0,0] tbox[0,0,1] tbox[0,0,2] tbox[0,0,3] tbox[0,0,4] tbox[0,1,0]
 tbox[0,1,1] tbox[0,1,2] tbox[0,1,3] tbox[0,1,4] tbox[0,2,0] tbox[0,2,1]
 tbox[0,2,2] tbox[0,2,3] tbox[0,2,4] tbox[0,3,0] tbox[0,3,1] tbox[0,3,2]
 tbox[0,3,3] tbox[0,3,4] tbox[1,0,0] tbox[1,0,1] tbox[1,0,2] tbox[1,0,3]
 tbox[1,0,4] tbox[1,1,0] tbox[1,1,1] tbox[1,1,2] tbox[1,1,3] tbox[1,1,4]
 tbox[1,2,0] tbox[1,2,1] tbox[1,2,2] tbox[1,2,3] tbox[1,2,4] tbox[1,3,0]
 tbox[1,3,1] tbox[1,3,2] tbox[1,3,3] tbox[1,3,4] tbox[2,0,0] tbox[2,0,1]
 tbox[2,0,2] tbox[2,0,3] tbox[2,0,4] tbox[2,1,0] tbox[2,1,1] tbox[2,1,2]
 tbox[2,1,3] tbox[2,1,4] tbox[2,2,0] tbox[2,2,1] tbox[2,2,2] tbox[2,2,3]
 tbox[2,2,4] tbox[2,3,0] tbox[2,3,1] tbox[2,3,2] tbox[2,3,3] tbox[2,3,4]
 tbox[3,0,0] tbox[3,0,1] tbox[3,0,2] tbox[3,0,3] tbox[3,0,4] tbox[3,1,0]
 tbox[3,1,1] tbox[3,1,2] tbox[3,1,3] tbox[3,1,4] tbox[3,2,0] tbox[3,2,1]
 tbox[3,2,2] tbox[3,2,3] tbox[3,2,4] tbox[3,3,0] tbox[3,3,1] tbox[3,3,2]
 tbox[3,3,3] tbox[3,3,4] unkn[0,0,0] unkn[0,0,1] unkn[0,0,2] unkn[0,0,3]
 unkn[0,0,4] unkn[0,1,0] unkn[0,1,1] unkn[0,1,2] unkn[0,1,3] unkn[0,1,4]
 unkn[0,2,0] unkn[0,2,1] unkn[0,2,2] unkn[0,2,3] unkn[0,2,4] unkn[0,3,0]
 unkn[0,3,1] unkn[0,3,2] unkn[0,3,3] unkn[0,3,4] unkn[1,0,0] unkn[1,0,1]
 unkn[1,0,2] unkn[1,0,3] unkn[1,0,4] unkn[1,1,0] unkn[1,1,1] unkn[1,1,2]
 unkn[1,1,3] unkn[1,1,4] unkn[1,2,0] unkn[1,2,1] unkn[1,2,2] unkn[1,2,3]
 unkn[1,2,4] unkn[1,3,0] unkn[1,3,1] unkn[1,3,2] unkn[1,3,3] unkn[1,3,4]
 unkn[2,0,0] unkn[2,0,1] unkn[2,0,2] unkn[2,0,3] unkn[2,0,4] unkn[2,1,0]
 unkn[2,1,1] unkn[2,1,2] unkn[2,1,3] unkn[2,1,4] unkn[2,2,0] unkn[2,2,1]
 unkn[2,2,2] unkn[2,2,3] unkn[2,2,4] unkn[2,3,0] unkn[2,3,1] unkn[2,3,2]
 unkn[2,3,3] unkn[2,3,4] unkn[3,0,0] unkn[3,0,1] unkn[3,0,2] unkn[3,0,3]
 unkn[3,0,4] unkn[3,1,0] unkn[3,1,1] unkn[3,1,2] unkn[3,1,3] unkn[3,1,4]
 unkn[3,2,0] unkn[3,2,1] unkn[3,2,2] unkn[3,2,3] unkn[3,2,4] unkn[3,3,0]
 unkn[3,3,1] unkn[3,3,2] unkn[3,3,3] unkn[3,3,4] forg[0,0,0] forg[0,0,1]
 forg[0,0,2] forg[0,0,3] forg[0,0,4] forg[0,1,0] forg[0,1,1] forg[0,1,2]
 forg[0,1,3] forg[0,1,4] forg[0,2,0] forg[0,2,1] forg[0,2,2] forg[0,2,3]
 forg[0,2,4] forg[0,3,0] forg[0,3,1] forg[0,3,2] forg[0,3,3] forg[0,3,4]
 forg[1,0,0] forg[1,0,1] forg[1,0,2] forg[1,0,3] forg[1,0,4] forg[1,1,0]
 forg[1,1,1] forg[1,1,2] forg[1,1,3] forg[1,1,4] forg[1,2,0] forg[1,2,1]
 forg[1,2,2] forg[1,2,3] forg[1,2,4] forg[1,3,0] forg[1,3,1] forg[1,3,2]
 forg[1,3,3] forg[1,3,4] forg[2,0,0] forg[2,0,1] forg[2,0,2] forg[2,0,3]
 forg[2,0,4] forg[2,1,0] forg[2,1,1] forg[2,1,2] forg[2,1,3] forg[2,1,4]
 forg[2,2,0] forg[2,2,1] forg[2,2,2] forg[2,2,3] forg[2,2,4] forg[2,3,0]
 forg[2,3,1] forg[2,3,2] forg[2,3,3] forg[2,3,4] forg[3,0,0] forg[3,0,1]
 forg[3,0,2] forg[3,0,3] forg[3,0,4] forg[3,1,0] forg[3,1,1] forg[3,1,2]
 forg[3,1,3] forg[3,1,4] forg[3,2,0] forg[3,2,1] forg[3,2,2] forg[3,2,3]
 forg[3,2,4] forg[3,3,0] forg[3,3,1] forg[3,3,2] forg[3,3,3] forg[3,3,4]
 lane[0,0] lane[0,1] lane[0,2] lane[0,3] lane[1,0] lane[1,1] lane[1,2]
 lane[1,3] lane[2,0] lane[2,1] lane[2,2] lane[2,3] lane[3,0] lane[3,1]
 lane[3,2] lane[3,3]
Generals
 cost[0,0,0] cost[0,0,1] cost[0,0,2] cost[0,0,3] cost[0,0,4] cost[0,1,0]
 cost[0,1,1] cost[0,1,2] cost[0,1,3] cost[0,1,4] cost[0,2,0] cost[0,2,1]
 cost[0,2,2] cost[0,2,3] cost[0,2,4] cost[0,3,0] cost[0,3,1] cost[0,3,2]
 cost[0,3,3] cost[0,3,4] cost[1,0,0] cost[1,0,1] cost[1,0,2] cost[1,0,3]
 cost[1,0,4] cost[1,1,0] cost[1,1,1] cost[1,1,2] cost[1,1,3] cost[1,1,4]
 cost[1,2,0] cost[1,2,1] cost[1,2,2] cost[1,2,3] cost[1,2,4] cost[1,3,0]
 cost[1,3,1] cost[1,3,2] cost[1,3,3] cost[1,3,4] cost[2,0,0] cost[2,0,1]
 cost[2,0,2] cost[2,0,3] cost[2,0,4] cost[2,1,0] cost[2,1,1] cost[2,1,2]
 cost[2,1,3] cost[2,1,4] cost[2,2,0] cost[2,2,1] cost[2,2,2] cost[2,2,3]
 cost[2,2,4] cost[2,3,0] cost[2,3,1] cost[2,3,2] cost[2,3,3] cost[2,3,4]
 cost[3,0,0] cost[3,0,1] cost[3,0,2] cost[3,0,3] cost[3,0,4] cost[3,1,0]
 cost[3,1,1] cost[3,1,2] cost[3,1,3] cost[3,1,4] cost[3,2,0] cost[3,2,1]
 cost[3,2,2] cost[3,2,3] cost[3,2,4] cost[3,3,0] cost[3,3,1] cost[3,3,2]
 cost[3,3,3] cost[3,3,4]
End
