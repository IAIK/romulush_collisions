\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 rkey[0,0,9] + 0 rkey[0,1,9] + 0 rkey[0,2,9] + 0 rkey[0,3,9]
   + 0 rkey[1,0,9] + 0 rkey[1,1,9] + 0 rkey[1,2,9] + 0 rkey[1,3,9]
   + 0 rkey[2,0,9] + 0 rkey[2,1,9] + 0 rkey[2,2,9] + 0 rkey[2,3,9]
   + 0 rkey[3,0,9] + 0 rkey[3,1,9] + 0 rkey[3,2,9] + 0 rkey[3,3,9]
   + 0 tkey[0,0,9] + 0 tkey[0,1,9] + 0 tkey[0,2,9] + 0 tkey[0,3,9]
   + 0 tkey[1,0,9] + 0 tkey[1,1,9] + 0 tkey[1,2,9] + 0 tkey[1,3,9]
   + 0 tkey[2,0,9] + 0 tkey[2,1,9] + 0 tkey[2,2,9] + 0 tkey[2,3,9]
   + 0 tkey[3,0,9] + 0 tkey[3,1,9] + 0 tkey[3,2,9] + 0 tkey[3,3,9]
   + cost[0,0,0] + cost[0,0,1] + cost[0,0,2] + cost[0,0,3] + cost[0,0,4]
   + cost[0,0,5] + cost[0,0,6] + cost[0,0,7] + cost[0,0,8] + 0 cost[0,0,9]
   + cost[0,1,0] + cost[0,1,1] + cost[0,1,2] + cost[0,1,3] + cost[0,1,4]
   + cost[0,1,5] + cost[0,1,6] + cost[0,1,7] + cost[0,1,8] + 0 cost[0,1,9]
   + cost[0,2,0] + cost[0,2,1] + cost[0,2,2] + cost[0,2,3] + cost[0,2,4]
   + cost[0,2,5] + cost[0,2,6] + cost[0,2,7] + cost[0,2,8] + 0 cost[0,2,9]
   + cost[0,3,0] + cost[0,3,1] + cost[0,3,2] + cost[0,3,3] + cost[0,3,4]
   + cost[0,3,5] + cost[0,3,6] + cost[0,3,7] + cost[0,3,8] + 0 cost[0,3,9]
   + cost[1,0,0] + cost[1,0,1] + cost[1,0,2] + cost[1,0,3] + cost[1,0,4]
   + cost[1,0,5] + cost[1,0,6] + cost[1,0,7] + cost[1,0,8] + 0 cost[1,0,9]
   + cost[1,1,0] + cost[1,1,1] + cost[1,1,2] + cost[1,1,3] + cost[1,1,4]
   + cost[1,1,5] + cost[1,1,6] + cost[1,1,7] + cost[1,1,8] + 0 cost[1,1,9]
   + cost[1,2,0] + cost[1,2,1] + cost[1,2,2] + cost[1,2,3] + cost[1,2,4]
   + cost[1,2,5] + cost[1,2,6] + cost[1,2,7] + cost[1,2,8] + 0 cost[1,2,9]
   + cost[1,3,0] + cost[1,3,1] + cost[1,3,2] + cost[1,3,3] + cost[1,3,4]
   + cost[1,3,5] + cost[1,3,6] + cost[1,3,7] + cost[1,3,8] + 0 cost[1,3,9]
   + cost[2,0,0] + cost[2,0,1] + cost[2,0,2] + cost[2,0,3] + cost[2,0,4]
   + cost[2,0,5] + cost[2,0,6] + cost[2,0,7] + cost[2,0,8] + 0 cost[2,0,9]
   + cost[2,1,0] + cost[2,1,1] + cost[2,1,2] + cost[2,1,3] + cost[2,1,4]
   + cost[2,1,5] + cost[2,1,6] + cost[2,1,7] + cost[2,1,8] + 0 cost[2,1,9]
   + cost[2,2,0] + cost[2,2,1] + cost[2,2,2] + cost[2,2,3] + cost[2,2,4]
   + cost[2,2,5] + cost[2,2,6] + cost[2,2,7] + cost[2,2,8] + 0 cost[2,2,9]
   + cost[2,3,0] + cost[2,3,1] + cost[2,3,2] + cost[2,3,3] + cost[2,3,4]
   + cost[2,3,5] + cost[2,3,6] + cost[2,3,7] + cost[2,3,8] + 0 cost[2,3,9]
   + cost[3,0,0] + cost[3,0,1] + cost[3,0,2] + cost[3,0,3] + cost[3,0,4]
   + cost[3,0,5] + cost[3,0,6] + cost[3,0,7] + cost[3,0,8] + 0 cost[3,0,9]
   + cost[3,1,0] + cost[3,1,1] + cost[3,1,2] + cost[3,1,3] + cost[3,1,4]
   + cost[3,1,5] + cost[3,1,6] + cost[3,1,7] + cost[3,1,8] + 0 cost[3,1,9]
   + cost[3,2,0] + cost[3,2,1] + cost[3,2,2] + cost[3,2,3] + cost[3,2,4]
   + cost[3,2,5] + cost[3,2,6] + cost[3,2,7] + cost[3,2,8] + 0 cost[3,2,9]
   + cost[3,3,0] + cost[3,3,1] + cost[3,3,2] + cost[3,3,3] + cost[3,3,4]
   + cost[3,3,5] + cost[3,3,6] + cost[3,3,7] + cost[3,3,8] + 0 cost[3,3,9]
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
 R384: - sbox[0,0,8] + cost[0,0,8] >= 0
 R385: - tbox[0,0,8] + unkn[0,0,8] + cost[0,0,8] >= 0
 R386: - 2 sbox[0,0,8] - unkn[0,0,8] + forg[0,0,8] + cost[0,0,8] >= -1
 R387: - sbox[0,1,8] + cost[0,1,8] >= 0
 R388: - tbox[0,1,8] + unkn[0,1,8] + cost[0,1,8] >= 0
 R389: - 2 sbox[0,1,8] - unkn[0,1,8] + forg[0,1,8] + cost[0,1,8] >= -1
 R390: - sbox[0,2,8] + cost[0,2,8] >= 0
 R391: - tbox[0,2,8] + unkn[0,2,8] + cost[0,2,8] >= 0
 R392: - 2 sbox[0,2,8] - unkn[0,2,8] + forg[0,2,8] + cost[0,2,8] >= -1
 R393: - sbox[0,3,8] + cost[0,3,8] >= 0
 R394: - tbox[0,3,8] + unkn[0,3,8] + cost[0,3,8] >= 0
 R395: - 2 sbox[0,3,8] - unkn[0,3,8] + forg[0,3,8] + cost[0,3,8] >= -1
 R396: - sbox[1,0,8] + cost[1,0,8] >= 0
 R397: - tbox[1,0,8] + unkn[1,0,8] + cost[1,0,8] >= 0
 R398: - 2 sbox[1,0,8] - unkn[1,0,8] + forg[1,0,8] + cost[1,0,8] >= -1
 R399: - sbox[1,1,8] + cost[1,1,8] >= 0
 R400: - tbox[1,1,8] + unkn[1,1,8] + cost[1,1,8] >= 0
 R401: - 2 sbox[1,1,8] - unkn[1,1,8] + forg[1,1,8] + cost[1,1,8] >= -1
 R402: - sbox[1,2,8] + cost[1,2,8] >= 0
 R403: - tbox[1,2,8] + unkn[1,2,8] + cost[1,2,8] >= 0
 R404: - 2 sbox[1,2,8] - unkn[1,2,8] + forg[1,2,8] + cost[1,2,8] >= -1
 R405: - sbox[1,3,8] + cost[1,3,8] >= 0
 R406: - tbox[1,3,8] + unkn[1,3,8] + cost[1,3,8] >= 0
 R407: - 2 sbox[1,3,8] - unkn[1,3,8] + forg[1,3,8] + cost[1,3,8] >= -1
 R408: - sbox[2,0,8] + cost[2,0,8] >= 0
 R409: - tbox[2,0,8] + unkn[2,0,8] + cost[2,0,8] >= 0
 R410: - 2 sbox[2,0,8] - unkn[2,0,8] + forg[2,0,8] + cost[2,0,8] >= -1
 R411: - sbox[2,1,8] + cost[2,1,8] >= 0
 R412: - tbox[2,1,8] + unkn[2,1,8] + cost[2,1,8] >= 0
 R413: - 2 sbox[2,1,8] - unkn[2,1,8] + forg[2,1,8] + cost[2,1,8] >= -1
 R414: - sbox[2,2,8] + cost[2,2,8] >= 0
 R415: - tbox[2,2,8] + unkn[2,2,8] + cost[2,2,8] >= 0
 R416: - 2 sbox[2,2,8] - unkn[2,2,8] + forg[2,2,8] + cost[2,2,8] >= -1
 R417: - sbox[2,3,8] + cost[2,3,8] >= 0
 R418: - tbox[2,3,8] + unkn[2,3,8] + cost[2,3,8] >= 0
 R419: - 2 sbox[2,3,8] - unkn[2,3,8] + forg[2,3,8] + cost[2,3,8] >= -1
 R420: - sbox[3,0,8] + cost[3,0,8] >= 0
 R421: - tbox[3,0,8] + unkn[3,0,8] + cost[3,0,8] >= 0
 R422: - 2 sbox[3,0,8] - unkn[3,0,8] + forg[3,0,8] + cost[3,0,8] >= -1
 R423: - sbox[3,1,8] + cost[3,1,8] >= 0
 R424: - tbox[3,1,8] + unkn[3,1,8] + cost[3,1,8] >= 0
 R425: - 2 sbox[3,1,8] - unkn[3,1,8] + forg[3,1,8] + cost[3,1,8] >= -1
 R426: - sbox[3,2,8] + cost[3,2,8] >= 0
 R427: - tbox[3,2,8] + unkn[3,2,8] + cost[3,2,8] >= 0
 R428: - 2 sbox[3,2,8] - unkn[3,2,8] + forg[3,2,8] + cost[3,2,8] >= -1
 R429: - sbox[3,3,8] + cost[3,3,8] >= 0
 R430: - tbox[3,3,8] + unkn[3,3,8] + cost[3,3,8] >= 0
 R431: - 2 sbox[3,3,8] - unkn[3,3,8] + forg[3,3,8] + cost[3,3,8] >= -1
 R432: tkey[0,0,0] - lane[0,0] <= 0
 R433: tkey[2,0,1] - lane[0,0] <= 0
 R434: tkey[0,2,2] - lane[0,0] <= 0
 R435: tkey[2,2,3] - lane[0,0] <= 0
 R436: tkey[1,0,4] - lane[0,0] <= 0
 R437: tkey[3,0,5] - lane[0,0] <= 0
 R438: tkey[1,2,6] - lane[0,0] <= 0
 R439: tkey[3,2,7] - lane[0,0] <= 0
 R440: tkey[1,1,8] - lane[0,0] <= 0
 R441: tkey[0,1,0] - lane[0,1] <= 0
 R442: tkey[2,1,1] - lane[0,1] <= 0
 R443: tkey[0,0,2] - lane[0,1] <= 0
 R444: tkey[2,0,3] - lane[0,1] <= 0
 R445: tkey[0,2,4] - lane[0,1] <= 0
 R446: tkey[2,2,5] - lane[0,1] <= 0
 R447: tkey[1,0,6] - lane[0,1] <= 0
 R448: tkey[3,0,7] - lane[0,1] <= 0
 R449: tkey[1,2,8] - lane[0,1] <= 0
 R450: tkey[0,2,0] - lane[0,2] <= 0
 R451: tkey[2,2,1] - lane[0,2] <= 0
 R452: tkey[1,0,2] - lane[0,2] <= 0
 R453: tkey[3,0,3] - lane[0,2] <= 0
 R454: tkey[1,2,4] - lane[0,2] <= 0
 R455: tkey[3,2,5] - lane[0,2] <= 0
 R456: tkey[1,1,6] - lane[0,2] <= 0
 R457: tkey[3,1,7] - lane[0,2] <= 0
 R458: tkey[0,3,8] - lane[0,2] <= 0
 R459: tkey[0,3,0] - lane[0,3] <= 0
 R460: tkey[2,3,1] - lane[0,3] <= 0
 R461: tkey[1,3,2] - lane[0,3] <= 0
 R462: tkey[3,3,3] - lane[0,3] <= 0
 R463: tkey[0,1,4] - lane[0,3] <= 0
 R464: tkey[2,1,5] - lane[0,3] <= 0
 R465: tkey[0,0,6] - lane[0,3] <= 0
 R466: tkey[2,0,7] - lane[0,3] <= 0
 R467: tkey[0,2,8] - lane[0,3] <= 0
 R468: tkey[1,0,0] - lane[1,0] <= 0
 R469: tkey[3,0,1] - lane[1,0] <= 0
 R470: tkey[1,2,2] - lane[1,0] <= 0
 R471: tkey[3,2,3] - lane[1,0] <= 0
 R472: tkey[1,1,4] - lane[1,0] <= 0
 R473: tkey[3,1,5] - lane[1,0] <= 0
 R474: tkey[0,3,6] - lane[1,0] <= 0
 R475: tkey[2,3,7] - lane[1,0] <= 0
 R476: tkey[1,3,8] - lane[1,0] <= 0
 R477: tkey[1,1,0] - lane[1,1] <= 0
 R478: tkey[3,1,1] - lane[1,1] <= 0
 R479: tkey[0,3,2] - lane[1,1] <= 0
 R480: tkey[2,3,3] - lane[1,1] <= 0
 R481: tkey[1,3,4] - lane[1,1] <= 0
 R482: tkey[3,3,5] - lane[1,1] <= 0
 R483: tkey[0,1,6] - lane[1,1] <= 0
 R484: tkey[2,1,7] - lane[1,1] <= 0
 R485: tkey[0,0,8] - lane[1,1] <= 0
 R486: tkey[1,2,0] - lane[1,2] <= 0
 R487: tkey[3,2,1] - lane[1,2] <= 0
 R488: tkey[1,1,2] - lane[1,2] <= 0
 R489: tkey[3,1,3] - lane[1,2] <= 0
 R490: tkey[0,3,4] - lane[1,2] <= 0
 R491: tkey[2,3,5] - lane[1,2] <= 0
 R492: tkey[1,3,6] - lane[1,2] <= 0
 R493: tkey[3,3,7] - lane[1,2] <= 0
 R494: tkey[0,1,8] - lane[1,2] <= 0
 R495: tkey[1,3,0] - lane[1,3] <= 0
 R496: tkey[3,3,1] - lane[1,3] <= 0
 R497: tkey[0,1,2] - lane[1,3] <= 0
 R498: tkey[2,1,3] - lane[1,3] <= 0
 R499: tkey[0,0,4] - lane[1,3] <= 0
 R500: tkey[2,0,5] - lane[1,3] <= 0
 R501: tkey[0,2,6] - lane[1,3] <= 0
 R502: tkey[2,2,7] - lane[1,3] <= 0
 R503: tkey[1,0,8] - lane[1,3] <= 0
 R504: tkey[2,0,0] - lane[2,0] <= 0
 R505: tkey[0,2,1] - lane[2,0] <= 0
 R506: tkey[2,2,2] - lane[2,0] <= 0
 R507: tkey[1,0,3] - lane[2,0] <= 0
 R508: tkey[3,0,4] - lane[2,0] <= 0
 R509: tkey[1,2,5] - lane[2,0] <= 0
 R510: tkey[3,2,6] - lane[2,0] <= 0
 R511: tkey[1,1,7] - lane[2,0] <= 0
 R512: tkey[3,1,8] - lane[2,0] <= 0
 R513: tkey[2,1,0] - lane[2,1] <= 0
 R514: tkey[0,0,1] - lane[2,1] <= 0
 R515: tkey[2,0,2] - lane[2,1] <= 0
 R516: tkey[0,2,3] - lane[2,1] <= 0
 R517: tkey[2,2,4] - lane[2,1] <= 0
 R518: tkey[1,0,5] - lane[2,1] <= 0
 R519: tkey[3,0,6] - lane[2,1] <= 0
 R520: tkey[1,2,7] - lane[2,1] <= 0
 R521: tkey[3,2,8] - lane[2,1] <= 0
 R522: tkey[2,2,0] - lane[2,2] <= 0
 R523: tkey[1,0,1] - lane[2,2] <= 0
 R524: tkey[3,0,2] - lane[2,2] <= 0
 R525: tkey[1,2,3] - lane[2,2] <= 0
 R526: tkey[3,2,4] - lane[2,2] <= 0
 R527: tkey[1,1,5] - lane[2,2] <= 0
 R528: tkey[3,1,6] - lane[2,2] <= 0
 R529: tkey[0,3,7] - lane[2,2] <= 0
 R530: tkey[2,3,8] - lane[2,2] <= 0
 R531: tkey[2,3,0] - lane[2,3] <= 0
 R532: tkey[1,3,1] - lane[2,3] <= 0
 R533: tkey[3,3,2] - lane[2,3] <= 0
 R534: tkey[0,1,3] - lane[2,3] <= 0
 R535: tkey[2,1,4] - lane[2,3] <= 0
 R536: tkey[0,0,5] - lane[2,3] <= 0
 R537: tkey[2,0,6] - lane[2,3] <= 0
 R538: tkey[0,2,7] - lane[2,3] <= 0
 R539: tkey[2,2,8] - lane[2,3] <= 0
 R540: tkey[3,0,0] - lane[3,0] <= 0
 R541: tkey[1,2,1] - lane[3,0] <= 0
 R542: tkey[3,2,2] - lane[3,0] <= 0
 R543: tkey[1,1,3] - lane[3,0] <= 0
 R544: tkey[3,1,4] - lane[3,0] <= 0
 R545: tkey[0,3,5] - lane[3,0] <= 0
 R546: tkey[2,3,6] - lane[3,0] <= 0
 R547: tkey[1,3,7] - lane[3,0] <= 0
 R548: tkey[3,3,8] - lane[3,0] <= 0
 R549: tkey[3,1,0] - lane[3,1] <= 0
 R550: tkey[0,3,1] - lane[3,1] <= 0
 R551: tkey[2,3,2] - lane[3,1] <= 0
 R552: tkey[1,3,3] - lane[3,1] <= 0
 R553: tkey[3,3,4] - lane[3,1] <= 0
 R554: tkey[0,1,5] - lane[3,1] <= 0
 R555: tkey[2,1,6] - lane[3,1] <= 0
 R556: tkey[0,0,7] - lane[3,1] <= 0
 R557: tkey[2,0,8] - lane[3,1] <= 0
 R558: tkey[3,2,0] - lane[3,2] <= 0
 R559: tkey[1,1,1] - lane[3,2] <= 0
 R560: tkey[3,1,2] - lane[3,2] <= 0
 R561: tkey[0,3,3] - lane[3,2] <= 0
 R562: tkey[2,3,4] - lane[3,2] <= 0
 R563: tkey[1,3,5] - lane[3,2] <= 0
 R564: tkey[3,3,6] - lane[3,2] <= 0
 R565: tkey[0,1,7] - lane[3,2] <= 0
 R566: tkey[2,1,8] - lane[3,2] <= 0
 R567: tkey[3,3,0] - lane[3,3] <= 0
 R568: tkey[0,1,1] - lane[3,3] <= 0
 R569: tkey[2,1,2] - lane[3,3] <= 0
 R570: tkey[0,0,3] - lane[3,3] <= 0
 R571: tkey[2,0,4] - lane[3,3] <= 0
 R572: tkey[0,2,5] - lane[3,3] <= 0
 R573: tkey[2,2,6] - lane[3,3] <= 0
 R574: tkey[1,0,7] - lane[3,3] <= 0
 R575: tkey[3,0,8] - lane[3,3] <= 0
 R576: - tkey[0,0,0] - tkey[0,2,2] - tkey[1,0,4] - tkey[1,1,8]
   - tkey[1,2,6] - tkey[2,0,1] - tkey[2,2,3] - tkey[3,0,5] - tkey[3,2,7]
   + 9 lane[0,0] <= 1
 R577: - tkey[0,0,2] - tkey[0,1,0] - tkey[0,2,4] - tkey[1,0,6]
   - tkey[1,2,8] - tkey[2,0,3] - tkey[2,1,1] - tkey[2,2,5] - tkey[3,0,7]
   + 9 lane[0,1] <= 1
 R578: - tkey[0,2,0] - tkey[0,3,8] - tkey[1,0,2] - tkey[1,1,6]
   - tkey[1,2,4] - tkey[2,2,1] - tkey[3,0,3] - tkey[3,1,7] - tkey[3,2,5]
   + 9 lane[0,2] <= 1
 R579: - tkey[0,0,6] - tkey[0,1,4] - tkey[0,2,8] - tkey[0,3,0]
   - tkey[1,3,2] - tkey[2,0,7] - tkey[2,1,5] - tkey[2,3,1] - tkey[3,3,3]
   + 9 lane[0,3] <= 1
 R580: - tkey[0,3,6] - tkey[1,0,0] - tkey[1,1,4] - tkey[1,2,2]
   - tkey[1,3,8] - tkey[2,3,7] - tkey[3,0,1] - tkey[3,1,5] - tkey[3,2,3]
   + 9 lane[1,0] <= 1
 R581: - tkey[0,0,8] - tkey[0,1,6] - tkey[0,3,2] - tkey[1,1,0]
   - tkey[1,3,4] - tkey[2,1,7] - tkey[2,3,3] - tkey[3,1,1] - tkey[3,3,5]
   + 9 lane[1,1] <= 1
 R582: - tkey[0,1,8] - tkey[0,3,4] - tkey[1,1,2] - tkey[1,2,0]
   - tkey[1,3,6] - tkey[2,3,5] - tkey[3,1,3] - tkey[3,2,1] - tkey[3,3,7]
   + 9 lane[1,2] <= 1
 R583: - tkey[0,0,4] - tkey[0,1,2] - tkey[0,2,6] - tkey[1,0,8]
   - tkey[1,3,0] - tkey[2,0,5] - tkey[2,1,3] - tkey[2,2,7] - tkey[3,3,1]
   + 9 lane[1,3] <= 1
 R584: - tkey[0,2,1] - tkey[1,0,3] - tkey[1,1,7] - tkey[1,2,5]
   - tkey[2,0,0] - tkey[2,2,2] - tkey[3,0,4] - tkey[3,1,8] - tkey[3,2,6]
   + 9 lane[2,0] <= 1
 R585: - tkey[0,0,1] - tkey[0,2,3] - tkey[1,0,5] - tkey[1,2,7]
   - tkey[2,0,2] - tkey[2,1,0] - tkey[2,2,4] - tkey[3,0,6] - tkey[3,2,8]
   + 9 lane[2,1] <= 1
 R586: - tkey[0,3,7] - tkey[1,0,1] - tkey[1,1,5] - tkey[1,2,3]
   - tkey[2,2,0] - tkey[2,3,8] - tkey[3,0,2] - tkey[3,1,6] - tkey[3,2,4]
   + 9 lane[2,2] <= 1
 R587: - tkey[0,0,5] - tkey[0,1,3] - tkey[0,2,7] - tkey[1,3,1]
   - tkey[2,0,6] - tkey[2,1,4] - tkey[2,2,8] - tkey[2,3,0] - tkey[3,3,2]
   + 9 lane[2,3] <= 1
 R588: - tkey[0,3,5] - tkey[1,1,3] - tkey[1,2,1] - tkey[1,3,7]
   - tkey[2,3,6] - tkey[3,0,0] - tkey[3,1,4] - tkey[3,2,2] - tkey[3,3,8]
   + 9 lane[3,0] <= 1
 R589: - tkey[0,0,7] - tkey[0,1,5] - tkey[0,3,1] - tkey[1,3,3]
   - tkey[2,0,8] - tkey[2,1,6] - tkey[2,3,2] - tkey[3,1,0] - tkey[3,3,4]
   + 9 lane[3,1] <= 1
 R590: - tkey[0,1,7] - tkey[0,3,3] - tkey[1,1,1] - tkey[1,3,5]
   - tkey[2,1,8] - tkey[2,3,4] - tkey[3,1,2] - tkey[3,2,0] - tkey[3,3,6]
   + 9 lane[3,2] <= 1
 R591: - tkey[0,0,3] - tkey[0,1,1] - tkey[0,2,5] - tkey[1,0,7]
   - tkey[2,0,4] - tkey[2,1,2] - tkey[2,2,6] - tkey[3,0,8] - tkey[3,3,0]
   + 9 lane[3,3] <= 1
 R592: - tbox[0,0,0] + forg[0,0,0] <= 0
 R593: sbox[0,0,0] - rkey[0,0,0] + tkey[0,0,0] >= 0
 R594: sbox[0,0,0] + rkey[0,0,0] - tkey[0,0,0] >= 0
 R595: - sbox[0,0,0] + rkey[0,0,0] + tkey[0,0,0] >= 0
 R596: - tbox[1,0,0] + forg[1,0,0] <= 0
 R597: sbox[1,0,0] - rkey[1,0,0] + tkey[1,0,0] >= 0
 R598: sbox[1,0,0] + rkey[1,0,0] - tkey[1,0,0] >= 0
 R599: - sbox[1,0,0] + rkey[1,0,0] + tkey[1,0,0] >= 0
 R600: - tbox[2,0,0] + forg[2,0,0] <= 0
 R601: - sbox[2,0,0] + rkey[2,0,0] = 0
 R602: - tbox[3,0,0] + forg[3,0,0] <= 0
 R603: - sbox[3,0,0] + rkey[3,0,0] = 0
 R604: - sbox[3,0,1] + rkey[0,0,0] + rkey[2,2,0] >= 0
 R605: sbox[3,0,1] + rkey[0,0,0] - rkey[2,2,0] >= 0
 R606: sbox[3,0,1] - rkey[0,0,0] + rkey[2,2,0] >= 0
 R607: - sbox[0,0,1] + sbox[3,0,1] + rkey[3,1,0] >= 0
 R608: sbox[0,0,1] + sbox[3,0,1] - rkey[3,1,0] >= 0
 R609: sbox[0,0,1] - sbox[3,0,1] + rkey[3,1,0] >= 0
 R610: - sbox[2,0,1] + rkey[1,3,0] + rkey[2,2,0] >= 0
 R611: sbox[2,0,1] + rkey[1,3,0] - rkey[2,2,0] >= 0
 R612: sbox[2,0,1] - rkey[1,3,0] + rkey[2,2,0] >= 0
 R613: sbox[1,0,1] - rkey[0,0,0] = 0
 R614: tbox[0,0,0] + tbox[2,2,0] - tbox[3,0,1] >= 0
 R615: tbox[0,0,0] - tbox[2,2,0] + tbox[3,0,1] >= 0
 R616: - tbox[0,0,0] + tbox[2,2,0] + tbox[3,0,1] >= 0
 R617: - tbox[0,0,1] + tbox[3,0,1] + tbox[3,1,0] >= 0
 R618: tbox[0,0,1] + tbox[3,0,1] - tbox[3,1,0] >= 0
 R619: tbox[0,0,1] - tbox[3,0,1] + tbox[3,1,0] >= 0
 R620: tbox[1,3,0] - tbox[2,0,1] + tbox[2,2,0] >= 0
 R621: tbox[1,3,0] + tbox[2,0,1] - tbox[2,2,0] >= 0
 R622: - tbox[1,3,0] + tbox[2,0,1] + tbox[2,2,0] >= 0
 R623: - tbox[0,0,0] + tbox[1,0,1] = 0
 R624: unkn[0,0,0] - unkn[3,0,1] + forg[3,0,1] <= 0
 R625: unkn[2,2,0] - unkn[3,0,1] + forg[3,0,1] <= 0
 R626: unkn[0,0,0] + unkn[2,2,0] - unkn[3,0,1] + forg[3,0,1] >= 0
 R627: - unkn[0,0,1] + unkn[3,0,1] + forg[0,0,1] - forg[3,0,1] <= 0
 R628: - unkn[0,0,1] + unkn[3,1,0] + forg[0,0,1] <= 0
 R629: - unkn[0,0,1] + unkn[3,0,1] + unkn[3,1,0] + forg[0,0,1]
   - forg[3,0,1] >= 0
 R630: unkn[1,3,0] - unkn[2,0,1] + forg[2,0,1] <= 0
 R631: - unkn[2,0,1] + unkn[2,2,0] + forg[2,0,1] <= 0
 R632: unkn[1,3,0] - unkn[2,0,1] + unkn[2,2,0] + forg[2,0,1] >= 0
 R633: unkn[0,0,0] - unkn[1,0,1] + forg[1,0,1] = 0
 R634: - tbox[0,1,0] + forg[0,1,0] <= 0
 R635: sbox[0,1,0] - rkey[0,1,0] + tkey[0,1,0] >= 0
 R636: sbox[0,1,0] + rkey[0,1,0] - tkey[0,1,0] >= 0
 R637: - sbox[0,1,0] + rkey[0,1,0] + tkey[0,1,0] >= 0
 R638: - tbox[1,1,0] + forg[1,1,0] <= 0
 R639: sbox[1,1,0] - rkey[1,1,0] + tkey[1,1,0] >= 0
 R640: sbox[1,1,0] + rkey[1,1,0] - tkey[1,1,0] >= 0
 R641: - sbox[1,1,0] + rkey[1,1,0] + tkey[1,1,0] >= 0
 R642: - tbox[2,1,0] + forg[2,1,0] <= 0
 R643: - sbox[2,1,0] + rkey[2,1,0] = 0
 R644: - tbox[3,1,0] + forg[3,1,0] <= 0
 R645: - sbox[3,1,0] + rkey[3,1,0] = 0
 R646: - sbox[3,1,1] + rkey[0,1,0] + rkey[2,3,0] >= 0
 R647: sbox[3,1,1] + rkey[0,1,0] - rkey[2,3,0] >= 0
 R648: sbox[3,1,1] - rkey[0,1,0] + rkey[2,3,0] >= 0
 R649: - sbox[0,1,1] + sbox[3,1,1] + rkey[3,2,0] >= 0
 R650: sbox[0,1,1] + sbox[3,1,1] - rkey[3,2,0] >= 0
 R651: sbox[0,1,1] - sbox[3,1,1] + rkey[3,2,0] >= 0
 R652: - sbox[2,1,1] + rkey[1,0,0] + rkey[2,3,0] >= 0
 R653: sbox[2,1,1] + rkey[1,0,0] - rkey[2,3,0] >= 0
 R654: sbox[2,1,1] - rkey[1,0,0] + rkey[2,3,0] >= 0
 R655: sbox[1,1,1] - rkey[0,1,0] = 0
 R656: tbox[0,1,0] + tbox[2,3,0] - tbox[3,1,1] >= 0
 R657: tbox[0,1,0] - tbox[2,3,0] + tbox[3,1,1] >= 0
 R658: - tbox[0,1,0] + tbox[2,3,0] + tbox[3,1,1] >= 0
 R659: - tbox[0,1,1] + tbox[3,1,1] + tbox[3,2,0] >= 0
 R660: tbox[0,1,1] + tbox[3,1,1] - tbox[3,2,0] >= 0
 R661: tbox[0,1,1] - tbox[3,1,1] + tbox[3,2,0] >= 0
 R662: tbox[1,0,0] - tbox[2,1,1] + tbox[2,3,0] >= 0
 R663: tbox[1,0,0] + tbox[2,1,1] - tbox[2,3,0] >= 0
 R664: - tbox[1,0,0] + tbox[2,1,1] + tbox[2,3,0] >= 0
 R665: - tbox[0,1,0] + tbox[1,1,1] = 0
 R666: unkn[0,1,0] - unkn[3,1,1] + forg[3,1,1] <= 0
 R667: unkn[2,3,0] - unkn[3,1,1] + forg[3,1,1] <= 0
 R668: unkn[0,1,0] + unkn[2,3,0] - unkn[3,1,1] + forg[3,1,1] >= 0
 R669: - unkn[0,1,1] + unkn[3,1,1] + forg[0,1,1] - forg[3,1,1] <= 0
 R670: - unkn[0,1,1] + unkn[3,2,0] + forg[0,1,1] <= 0
 R671: - unkn[0,1,1] + unkn[3,1,1] + unkn[3,2,0] + forg[0,1,1]
   - forg[3,1,1] >= 0
 R672: unkn[1,0,0] - unkn[2,1,1] + forg[2,1,1] <= 0
 R673: - unkn[2,1,1] + unkn[2,3,0] + forg[2,1,1] <= 0
 R674: unkn[1,0,0] - unkn[2,1,1] + unkn[2,3,0] + forg[2,1,1] >= 0
 R675: unkn[0,1,0] - unkn[1,1,1] + forg[1,1,1] = 0
 R676: - tbox[0,2,0] + forg[0,2,0] <= 0
 R677: sbox[0,2,0] - rkey[0,2,0] + tkey[0,2,0] >= 0
 R678: sbox[0,2,0] + rkey[0,2,0] - tkey[0,2,0] >= 0
 R679: - sbox[0,2,0] + rkey[0,2,0] + tkey[0,2,0] >= 0
 R680: - tbox[1,2,0] + forg[1,2,0] <= 0
 R681: sbox[1,2,0] - rkey[1,2,0] + tkey[1,2,0] >= 0
 R682: sbox[1,2,0] + rkey[1,2,0] - tkey[1,2,0] >= 0
 R683: - sbox[1,2,0] + rkey[1,2,0] + tkey[1,2,0] >= 0
 R684: - tbox[2,2,0] + forg[2,2,0] <= 0
 R685: - sbox[2,2,0] + rkey[2,2,0] = 0
 R686: - tbox[3,2,0] + forg[3,2,0] <= 0
 R687: - sbox[3,2,0] + rkey[3,2,0] = 0
 R688: - sbox[3,2,1] + rkey[0,2,0] + rkey[2,0,0] >= 0
 R689: sbox[3,2,1] + rkey[0,2,0] - rkey[2,0,0] >= 0
 R690: sbox[3,2,1] - rkey[0,2,0] + rkey[2,0,0] >= 0
 R691: - sbox[0,2,1] + sbox[3,2,1] + rkey[3,3,0] >= 0
 R692: sbox[0,2,1] + sbox[3,2,1] - rkey[3,3,0] >= 0
 R693: sbox[0,2,1] - sbox[3,2,1] + rkey[3,3,0] >= 0
 R694: - sbox[2,2,1] + rkey[1,1,0] + rkey[2,0,0] >= 0
 R695: sbox[2,2,1] + rkey[1,1,0] - rkey[2,0,0] >= 0
 R696: sbox[2,2,1] - rkey[1,1,0] + rkey[2,0,0] >= 0
 R697: sbox[1,2,1] - rkey[0,2,0] = 0
 R698: tbox[0,2,0] + tbox[2,0,0] - tbox[3,2,1] >= 0
 R699: tbox[0,2,0] - tbox[2,0,0] + tbox[3,2,1] >= 0
 R700: - tbox[0,2,0] + tbox[2,0,0] + tbox[3,2,1] >= 0
 R701: - tbox[0,2,1] + tbox[3,2,1] + tbox[3,3,0] >= 0
 R702: tbox[0,2,1] + tbox[3,2,1] - tbox[3,3,0] >= 0
 R703: tbox[0,2,1] - tbox[3,2,1] + tbox[3,3,0] >= 0
 R704: tbox[1,1,0] + tbox[2,0,0] - tbox[2,2,1] >= 0
 R705: tbox[1,1,0] - tbox[2,0,0] + tbox[2,2,1] >= 0
 R706: - tbox[1,1,0] + tbox[2,0,0] + tbox[2,2,1] >= 0
 R707: - tbox[0,2,0] + tbox[1,2,1] = 0
 R708: unkn[0,2,0] - unkn[3,2,1] + forg[3,2,1] <= 0
 R709: unkn[2,0,0] - unkn[3,2,1] + forg[3,2,1] <= 0
 R710: unkn[0,2,0] + unkn[2,0,0] - unkn[3,2,1] + forg[3,2,1] >= 0
 R711: - unkn[0,2,1] + unkn[3,2,1] + forg[0,2,1] - forg[3,2,1] <= 0
 R712: - unkn[0,2,1] + unkn[3,3,0] + forg[0,2,1] <= 0
 R713: - unkn[0,2,1] + unkn[3,2,1] + unkn[3,3,0] + forg[0,2,1]
   - forg[3,2,1] >= 0
 R714: unkn[1,1,0] - unkn[2,2,1] + forg[2,2,1] <= 0
 R715: unkn[2,0,0] - unkn[2,2,1] + forg[2,2,1] <= 0
 R716: unkn[1,1,0] + unkn[2,0,0] - unkn[2,2,1] + forg[2,2,1] >= 0
 R717: unkn[0,2,0] - unkn[1,2,1] + forg[1,2,1] = 0
 R718: - tbox[0,3,0] + forg[0,3,0] <= 0
 R719: sbox[0,3,0] - rkey[0,3,0] + tkey[0,3,0] >= 0
 R720: sbox[0,3,0] + rkey[0,3,0] - tkey[0,3,0] >= 0
 R721: - sbox[0,3,0] + rkey[0,3,0] + tkey[0,3,0] >= 0
 R722: - tbox[1,3,0] + forg[1,3,0] <= 0
 R723: sbox[1,3,0] - rkey[1,3,0] + tkey[1,3,0] >= 0
 R724: sbox[1,3,0] + rkey[1,3,0] - tkey[1,3,0] >= 0
 R725: - sbox[1,3,0] + rkey[1,3,0] + tkey[1,3,0] >= 0
 R726: - tbox[2,3,0] + forg[2,3,0] <= 0
 R727: - sbox[2,3,0] + rkey[2,3,0] = 0
 R728: - tbox[3,3,0] + forg[3,3,0] <= 0
 R729: - sbox[3,3,0] + rkey[3,3,0] = 0
 R730: - sbox[3,3,1] + rkey[0,3,0] + rkey[2,1,0] >= 0
 R731: sbox[3,3,1] + rkey[0,3,0] - rkey[2,1,0] >= 0
 R732: sbox[3,3,1] - rkey[0,3,0] + rkey[2,1,0] >= 0
 R733: - sbox[0,3,1] + sbox[3,3,1] + rkey[3,0,0] >= 0
 R734: sbox[0,3,1] + sbox[3,3,1] - rkey[3,0,0] >= 0
 R735: sbox[0,3,1] - sbox[3,3,1] + rkey[3,0,0] >= 0
 R736: - sbox[2,3,1] + rkey[1,2,0] + rkey[2,1,0] >= 0
 R737: sbox[2,3,1] + rkey[1,2,0] - rkey[2,1,0] >= 0
 R738: sbox[2,3,1] - rkey[1,2,0] + rkey[2,1,0] >= 0
 R739: sbox[1,3,1] - rkey[0,3,0] = 0
 R740: tbox[0,3,0] + tbox[2,1,0] - tbox[3,3,1] >= 0
 R741: tbox[0,3,0] - tbox[2,1,0] + tbox[3,3,1] >= 0
 R742: - tbox[0,3,0] + tbox[2,1,0] + tbox[3,3,1] >= 0
 R743: - tbox[0,3,1] + tbox[3,0,0] + tbox[3,3,1] >= 0
 R744: tbox[0,3,1] - tbox[3,0,0] + tbox[3,3,1] >= 0
 R745: tbox[0,3,1] + tbox[3,0,0] - tbox[3,3,1] >= 0
 R746: tbox[1,2,0] + tbox[2,1,0] - tbox[2,3,1] >= 0
 R747: tbox[1,2,0] - tbox[2,1,0] + tbox[2,3,1] >= 0
 R748: - tbox[1,2,0] + tbox[2,1,0] + tbox[2,3,1] >= 0
 R749: - tbox[0,3,0] + tbox[1,3,1] = 0
 R750: unkn[0,3,0] - unkn[3,3,1] + forg[3,3,1] <= 0
 R751: unkn[2,1,0] - unkn[3,3,1] + forg[3,3,1] <= 0
 R752: unkn[0,3,0] + unkn[2,1,0] - unkn[3,3,1] + forg[3,3,1] >= 0
 R753: - unkn[0,3,1] + unkn[3,3,1] + forg[0,3,1] - forg[3,3,1] <= 0
 R754: - unkn[0,3,1] + unkn[3,0,0] + forg[0,3,1] <= 0
 R755: - unkn[0,3,1] + unkn[3,0,0] + unkn[3,3,1] + forg[0,3,1]
   - forg[3,3,1] >= 0
 R756: unkn[1,2,0] - unkn[2,3,1] + forg[2,3,1] <= 0
 R757: unkn[2,1,0] - unkn[2,3,1] + forg[2,3,1] <= 0
 R758: unkn[1,2,0] + unkn[2,1,0] - unkn[2,3,1] + forg[2,3,1] >= 0
 R759: unkn[0,3,0] - unkn[1,3,1] + forg[1,3,1] = 0
 R760: - tbox[0,0,1] + forg[0,0,1] <= 0
 R761: sbox[0,0,1] - rkey[0,0,1] + tkey[0,0,1] >= 0
 R762: sbox[0,0,1] + rkey[0,0,1] - tkey[0,0,1] >= 0
 R763: - sbox[0,0,1] + rkey[0,0,1] + tkey[0,0,1] >= 0
 R764: - tbox[1,0,1] + forg[1,0,1] <= 0
 R765: sbox[1,0,1] - rkey[1,0,1] + tkey[1,0,1] >= 0
 R766: sbox[1,0,1] + rkey[1,0,1] - tkey[1,0,1] >= 0
 R767: - sbox[1,0,1] + rkey[1,0,1] + tkey[1,0,1] >= 0
 R768: - tbox[2,0,1] + forg[2,0,1] <= 0
 R769: - sbox[2,0,1] + rkey[2,0,1] = 0
 R770: - tbox[3,0,1] + forg[3,0,1] <= 0
 R771: - sbox[3,0,1] + rkey[3,0,1] = 0
 R772: - sbox[3,0,2] + rkey[0,0,1] + rkey[2,2,1] >= 0
 R773: sbox[3,0,2] + rkey[0,0,1] - rkey[2,2,1] >= 0
 R774: sbox[3,0,2] - rkey[0,0,1] + rkey[2,2,1] >= 0
 R775: - sbox[0,0,2] + sbox[3,0,2] + rkey[3,1,1] >= 0
 R776: sbox[0,0,2] + sbox[3,0,2] - rkey[3,1,1] >= 0
 R777: sbox[0,0,2] - sbox[3,0,2] + rkey[3,1,1] >= 0
 R778: - sbox[2,0,2] + rkey[1,3,1] + rkey[2,2,1] >= 0
 R779: sbox[2,0,2] + rkey[1,3,1] - rkey[2,2,1] >= 0
 R780: sbox[2,0,2] - rkey[1,3,1] + rkey[2,2,1] >= 0
 R781: sbox[1,0,2] - rkey[0,0,1] = 0
 R782: tbox[0,0,1] + tbox[2,2,1] - tbox[3,0,2] >= 0
 R783: tbox[0,0,1] - tbox[2,2,1] + tbox[3,0,2] >= 0
 R784: - tbox[0,0,1] + tbox[2,2,1] + tbox[3,0,2] >= 0
 R785: - tbox[0,0,2] + tbox[3,0,2] + tbox[3,1,1] >= 0
 R786: tbox[0,0,2] + tbox[3,0,2] - tbox[3,1,1] >= 0
 R787: tbox[0,0,2] - tbox[3,0,2] + tbox[3,1,1] >= 0
 R788: tbox[1,3,1] - tbox[2,0,2] + tbox[2,2,1] >= 0
 R789: tbox[1,3,1] + tbox[2,0,2] - tbox[2,2,1] >= 0
 R790: - tbox[1,3,1] + tbox[2,0,2] + tbox[2,2,1] >= 0
 R791: - tbox[0,0,1] + tbox[1,0,2] = 0
 R792: unkn[0,0,1] - unkn[3,0,2] + forg[3,0,2] <= 0
 R793: unkn[2,2,1] - unkn[3,0,2] + forg[3,0,2] <= 0
 R794: unkn[0,0,1] + unkn[2,2,1] - unkn[3,0,2] + forg[3,0,2] >= 0
 R795: - unkn[0,0,2] + unkn[3,0,2] + forg[0,0,2] - forg[3,0,2] <= 0
 R796: - unkn[0,0,2] + unkn[3,1,1] + forg[0,0,2] <= 0
 R797: - unkn[0,0,2] + unkn[3,0,2] + unkn[3,1,1] + forg[0,0,2]
   - forg[3,0,2] >= 0
 R798: unkn[1,3,1] - unkn[2,0,2] + forg[2,0,2] <= 0
 R799: - unkn[2,0,2] + unkn[2,2,1] + forg[2,0,2] <= 0
 R800: unkn[1,3,1] - unkn[2,0,2] + unkn[2,2,1] + forg[2,0,2] >= 0
 R801: unkn[0,0,1] - unkn[1,0,2] + forg[1,0,2] = 0
 R802: - tbox[0,1,1] + forg[0,1,1] <= 0
 R803: sbox[0,1,1] - rkey[0,1,1] + tkey[0,1,1] >= 0
 R804: sbox[0,1,1] + rkey[0,1,1] - tkey[0,1,1] >= 0
 R805: - sbox[0,1,1] + rkey[0,1,1] + tkey[0,1,1] >= 0
 R806: - tbox[1,1,1] + forg[1,1,1] <= 0
 R807: sbox[1,1,1] - rkey[1,1,1] + tkey[1,1,1] >= 0
 R808: sbox[1,1,1] + rkey[1,1,1] - tkey[1,1,1] >= 0
 R809: - sbox[1,1,1] + rkey[1,1,1] + tkey[1,1,1] >= 0
 R810: - tbox[2,1,1] + forg[2,1,1] <= 0
 R811: - sbox[2,1,1] + rkey[2,1,1] = 0
 R812: - tbox[3,1,1] + forg[3,1,1] <= 0
 R813: - sbox[3,1,1] + rkey[3,1,1] = 0
 R814: - sbox[3,1,2] + rkey[0,1,1] + rkey[2,3,1] >= 0
 R815: sbox[3,1,2] + rkey[0,1,1] - rkey[2,3,1] >= 0
 R816: sbox[3,1,2] - rkey[0,1,1] + rkey[2,3,1] >= 0
 R817: - sbox[0,1,2] + sbox[3,1,2] + rkey[3,2,1] >= 0
 R818: sbox[0,1,2] + sbox[3,1,2] - rkey[3,2,1] >= 0
 R819: sbox[0,1,2] - sbox[3,1,2] + rkey[3,2,1] >= 0
 R820: - sbox[2,1,2] + rkey[1,0,1] + rkey[2,3,1] >= 0
 R821: sbox[2,1,2] + rkey[1,0,1] - rkey[2,3,1] >= 0
 R822: sbox[2,1,2] - rkey[1,0,1] + rkey[2,3,1] >= 0
 R823: sbox[1,1,2] - rkey[0,1,1] = 0
 R824: tbox[0,1,1] + tbox[2,3,1] - tbox[3,1,2] >= 0
 R825: tbox[0,1,1] - tbox[2,3,1] + tbox[3,1,2] >= 0
 R826: - tbox[0,1,1] + tbox[2,3,1] + tbox[3,1,2] >= 0
 R827: - tbox[0,1,2] + tbox[3,1,2] + tbox[3,2,1] >= 0
 R828: tbox[0,1,2] + tbox[3,1,2] - tbox[3,2,1] >= 0
 R829: tbox[0,1,2] - tbox[3,1,2] + tbox[3,2,1] >= 0
 R830: tbox[1,0,1] - tbox[2,1,2] + tbox[2,3,1] >= 0
 R831: tbox[1,0,1] + tbox[2,1,2] - tbox[2,3,1] >= 0
 R832: - tbox[1,0,1] + tbox[2,1,2] + tbox[2,3,1] >= 0
 R833: - tbox[0,1,1] + tbox[1,1,2] = 0
 R834: unkn[0,1,1] - unkn[3,1,2] + forg[3,1,2] <= 0
 R835: unkn[2,3,1] - unkn[3,1,2] + forg[3,1,2] <= 0
 R836: unkn[0,1,1] + unkn[2,3,1] - unkn[3,1,2] + forg[3,1,2] >= 0
 R837: - unkn[0,1,2] + unkn[3,1,2] + forg[0,1,2] - forg[3,1,2] <= 0
 R838: - unkn[0,1,2] + unkn[3,2,1] + forg[0,1,2] <= 0
 R839: - unkn[0,1,2] + unkn[3,1,2] + unkn[3,2,1] + forg[0,1,2]
   - forg[3,1,2] >= 0
 R840: unkn[1,0,1] - unkn[2,1,2] + forg[2,1,2] <= 0
 R841: - unkn[2,1,2] + unkn[2,3,1] + forg[2,1,2] <= 0
 R842: unkn[1,0,1] - unkn[2,1,2] + unkn[2,3,1] + forg[2,1,2] >= 0
 R843: unkn[0,1,1] - unkn[1,1,2] + forg[1,1,2] = 0
 R844: - tbox[0,2,1] + forg[0,2,1] <= 0
 R845: sbox[0,2,1] - rkey[0,2,1] + tkey[0,2,1] >= 0
 R846: sbox[0,2,1] + rkey[0,2,1] - tkey[0,2,1] >= 0
 R847: - sbox[0,2,1] + rkey[0,2,1] + tkey[0,2,1] >= 0
 R848: - tbox[1,2,1] + forg[1,2,1] <= 0
 R849: sbox[1,2,1] - rkey[1,2,1] + tkey[1,2,1] >= 0
 R850: sbox[1,2,1] + rkey[1,2,1] - tkey[1,2,1] >= 0
 R851: - sbox[1,2,1] + rkey[1,2,1] + tkey[1,2,1] >= 0
 R852: - tbox[2,2,1] + forg[2,2,1] <= 0
 R853: - sbox[2,2,1] + rkey[2,2,1] = 0
 R854: - tbox[3,2,1] + forg[3,2,1] <= 0
 R855: - sbox[3,2,1] + rkey[3,2,1] = 0
 R856: - sbox[3,2,2] + rkey[0,2,1] + rkey[2,0,1] >= 0
 R857: sbox[3,2,2] + rkey[0,2,1] - rkey[2,0,1] >= 0
 R858: sbox[3,2,2] - rkey[0,2,1] + rkey[2,0,1] >= 0
 R859: - sbox[0,2,2] + sbox[3,2,2] + rkey[3,3,1] >= 0
 R860: sbox[0,2,2] + sbox[3,2,2] - rkey[3,3,1] >= 0
 R861: sbox[0,2,2] - sbox[3,2,2] + rkey[3,3,1] >= 0
 R862: - sbox[2,2,2] + rkey[1,1,1] + rkey[2,0,1] >= 0
 R863: sbox[2,2,2] + rkey[1,1,1] - rkey[2,0,1] >= 0
 R864: sbox[2,2,2] - rkey[1,1,1] + rkey[2,0,1] >= 0
 R865: sbox[1,2,2] - rkey[0,2,1] = 0
 R866: tbox[0,2,1] + tbox[2,0,1] - tbox[3,2,2] >= 0
 R867: tbox[0,2,1] - tbox[2,0,1] + tbox[3,2,2] >= 0
 R868: - tbox[0,2,1] + tbox[2,0,1] + tbox[3,2,2] >= 0
 R869: - tbox[0,2,2] + tbox[3,2,2] + tbox[3,3,1] >= 0
 R870: tbox[0,2,2] + tbox[3,2,2] - tbox[3,3,1] >= 0
 R871: tbox[0,2,2] - tbox[3,2,2] + tbox[3,3,1] >= 0
 R872: tbox[1,1,1] + tbox[2,0,1] - tbox[2,2,2] >= 0
 R873: tbox[1,1,1] - tbox[2,0,1] + tbox[2,2,2] >= 0
 R874: - tbox[1,1,1] + tbox[2,0,1] + tbox[2,2,2] >= 0
 R875: - tbox[0,2,1] + tbox[1,2,2] = 0
 R876: unkn[0,2,1] - unkn[3,2,2] + forg[3,2,2] <= 0
 R877: unkn[2,0,1] - unkn[3,2,2] + forg[3,2,2] <= 0
 R878: unkn[0,2,1] + unkn[2,0,1] - unkn[3,2,2] + forg[3,2,2] >= 0
 R879: - unkn[0,2,2] + unkn[3,2,2] + forg[0,2,2] - forg[3,2,2] <= 0
 R880: - unkn[0,2,2] + unkn[3,3,1] + forg[0,2,2] <= 0
 R881: - unkn[0,2,2] + unkn[3,2,2] + unkn[3,3,1] + forg[0,2,2]
   - forg[3,2,2] >= 0
 R882: unkn[1,1,1] - unkn[2,2,2] + forg[2,2,2] <= 0
 R883: unkn[2,0,1] - unkn[2,2,2] + forg[2,2,2] <= 0
 R884: unkn[1,1,1] + unkn[2,0,1] - unkn[2,2,2] + forg[2,2,2] >= 0
 R885: unkn[0,2,1] - unkn[1,2,2] + forg[1,2,2] = 0
 R886: - tbox[0,3,1] + forg[0,3,1] <= 0
 R887: sbox[0,3,1] - rkey[0,3,1] + tkey[0,3,1] >= 0
 R888: sbox[0,3,1] + rkey[0,3,1] - tkey[0,3,1] >= 0
 R889: - sbox[0,3,1] + rkey[0,3,1] + tkey[0,3,1] >= 0
 R890: - tbox[1,3,1] + forg[1,3,1] <= 0
 R891: sbox[1,3,1] - rkey[1,3,1] + tkey[1,3,1] >= 0
 R892: sbox[1,3,1] + rkey[1,3,1] - tkey[1,3,1] >= 0
 R893: - sbox[1,3,1] + rkey[1,3,1] + tkey[1,3,1] >= 0
 R894: - tbox[2,3,1] + forg[2,3,1] <= 0
 R895: - sbox[2,3,1] + rkey[2,3,1] = 0
 R896: - tbox[3,3,1] + forg[3,3,1] <= 0
 R897: - sbox[3,3,1] + rkey[3,3,1] = 0
 R898: - sbox[3,3,2] + rkey[0,3,1] + rkey[2,1,1] >= 0
 R899: sbox[3,3,2] + rkey[0,3,1] - rkey[2,1,1] >= 0
 R900: sbox[3,3,2] - rkey[0,3,1] + rkey[2,1,1] >= 0
 R901: - sbox[0,3,2] + sbox[3,3,2] + rkey[3,0,1] >= 0
 R902: sbox[0,3,2] + sbox[3,3,2] - rkey[3,0,1] >= 0
 R903: sbox[0,3,2] - sbox[3,3,2] + rkey[3,0,1] >= 0
 R904: - sbox[2,3,2] + rkey[1,2,1] + rkey[2,1,1] >= 0
 R905: sbox[2,3,2] + rkey[1,2,1] - rkey[2,1,1] >= 0
 R906: sbox[2,3,2] - rkey[1,2,1] + rkey[2,1,1] >= 0
 R907: sbox[1,3,2] - rkey[0,3,1] = 0
 R908: tbox[0,3,1] + tbox[2,1,1] - tbox[3,3,2] >= 0
 R909: tbox[0,3,1] - tbox[2,1,1] + tbox[3,3,2] >= 0
 R910: - tbox[0,3,1] + tbox[2,1,1] + tbox[3,3,2] >= 0
 R911: - tbox[0,3,2] + tbox[3,0,1] + tbox[3,3,2] >= 0
 R912: tbox[0,3,2] - tbox[3,0,1] + tbox[3,3,2] >= 0
 R913: tbox[0,3,2] + tbox[3,0,1] - tbox[3,3,2] >= 0
 R914: tbox[1,2,1] + tbox[2,1,1] - tbox[2,3,2] >= 0
 R915: tbox[1,2,1] - tbox[2,1,1] + tbox[2,3,2] >= 0
 R916: - tbox[1,2,1] + tbox[2,1,1] + tbox[2,3,2] >= 0
 R917: - tbox[0,3,1] + tbox[1,3,2] = 0
 R918: unkn[0,3,1] - unkn[3,3,2] + forg[3,3,2] <= 0
 R919: unkn[2,1,1] - unkn[3,3,2] + forg[3,3,2] <= 0
 R920: unkn[0,3,1] + unkn[2,1,1] - unkn[3,3,2] + forg[3,3,2] >= 0
 R921: - unkn[0,3,2] + unkn[3,3,2] + forg[0,3,2] - forg[3,3,2] <= 0
 R922: - unkn[0,3,2] + unkn[3,0,1] + forg[0,3,2] <= 0
 R923: - unkn[0,3,2] + unkn[3,0,1] + unkn[3,3,2] + forg[0,3,2]
   - forg[3,3,2] >= 0
 R924: unkn[1,2,1] - unkn[2,3,2] + forg[2,3,2] <= 0
 R925: unkn[2,1,1] - unkn[2,3,2] + forg[2,3,2] <= 0
 R926: unkn[1,2,1] + unkn[2,1,1] - unkn[2,3,2] + forg[2,3,2] >= 0
 R927: unkn[0,3,1] - unkn[1,3,2] + forg[1,3,2] = 0
 R928: - tbox[0,0,2] + forg[0,0,2] <= 0
 R929: sbox[0,0,2] - rkey[0,0,2] + tkey[0,0,2] >= 0
 R930: sbox[0,0,2] + rkey[0,0,2] - tkey[0,0,2] >= 0
 R931: - sbox[0,0,2] + rkey[0,0,2] + tkey[0,0,2] >= 0
 R932: - tbox[1,0,2] + forg[1,0,2] <= 0
 R933: sbox[1,0,2] - rkey[1,0,2] + tkey[1,0,2] >= 0
 R934: sbox[1,0,2] + rkey[1,0,2] - tkey[1,0,2] >= 0
 R935: - sbox[1,0,2] + rkey[1,0,2] + tkey[1,0,2] >= 0
 R936: - tbox[2,0,2] + forg[2,0,2] <= 0
 R937: - sbox[2,0,2] + rkey[2,0,2] = 0
 R938: - tbox[3,0,2] + forg[3,0,2] <= 0
 R939: - sbox[3,0,2] + rkey[3,0,2] = 0
 R940: - sbox[3,0,3] + rkey[0,0,2] + rkey[2,2,2] >= 0
 R941: sbox[3,0,3] + rkey[0,0,2] - rkey[2,2,2] >= 0
 R942: sbox[3,0,3] - rkey[0,0,2] + rkey[2,2,2] >= 0
 R943: - sbox[0,0,3] + sbox[3,0,3] + rkey[3,1,2] >= 0
 R944: sbox[0,0,3] + sbox[3,0,3] - rkey[3,1,2] >= 0
 R945: sbox[0,0,3] - sbox[3,0,3] + rkey[3,1,2] >= 0
 R946: - sbox[2,0,3] + rkey[1,3,2] + rkey[2,2,2] >= 0
 R947: sbox[2,0,3] + rkey[1,3,2] - rkey[2,2,2] >= 0
 R948: sbox[2,0,3] - rkey[1,3,2] + rkey[2,2,2] >= 0
 R949: sbox[1,0,3] - rkey[0,0,2] = 0
 R950: tbox[0,0,2] + tbox[2,2,2] - tbox[3,0,3] >= 0
 R951: tbox[0,0,2] - tbox[2,2,2] + tbox[3,0,3] >= 0
 R952: - tbox[0,0,2] + tbox[2,2,2] + tbox[3,0,3] >= 0
 R953: - tbox[0,0,3] + tbox[3,0,3] + tbox[3,1,2] >= 0
 R954: tbox[0,0,3] + tbox[3,0,3] - tbox[3,1,2] >= 0
 R955: tbox[0,0,3] - tbox[3,0,3] + tbox[3,1,2] >= 0
 R956: tbox[1,3,2] - tbox[2,0,3] + tbox[2,2,2] >= 0
 R957: tbox[1,3,2] + tbox[2,0,3] - tbox[2,2,2] >= 0
 R958: - tbox[1,3,2] + tbox[2,0,3] + tbox[2,2,2] >= 0
 R959: - tbox[0,0,2] + tbox[1,0,3] = 0
 R960: unkn[0,0,2] - unkn[3,0,3] + forg[3,0,3] <= 0
 R961: unkn[2,2,2] - unkn[3,0,3] + forg[3,0,3] <= 0
 R962: unkn[0,0,2] + unkn[2,2,2] - unkn[3,0,3] + forg[3,0,3] >= 0
 R963: - unkn[0,0,3] + unkn[3,0,3] + forg[0,0,3] - forg[3,0,3] <= 0
 R964: - unkn[0,0,3] + unkn[3,1,2] + forg[0,0,3] <= 0
 R965: - unkn[0,0,3] + unkn[3,0,3] + unkn[3,1,2] + forg[0,0,3]
   - forg[3,0,3] >= 0
 R966: unkn[1,3,2] - unkn[2,0,3] + forg[2,0,3] <= 0
 R967: - unkn[2,0,3] + unkn[2,2,2] + forg[2,0,3] <= 0
 R968: unkn[1,3,2] - unkn[2,0,3] + unkn[2,2,2] + forg[2,0,3] >= 0
 R969: unkn[0,0,2] - unkn[1,0,3] + forg[1,0,3] = 0
 R970: - tbox[0,1,2] + forg[0,1,2] <= 0
 R971: sbox[0,1,2] - rkey[0,1,2] + tkey[0,1,2] >= 0
 R972: sbox[0,1,2] + rkey[0,1,2] - tkey[0,1,2] >= 0
 R973: - sbox[0,1,2] + rkey[0,1,2] + tkey[0,1,2] >= 0
 R974: - tbox[1,1,2] + forg[1,1,2] <= 0
 R975: sbox[1,1,2] - rkey[1,1,2] + tkey[1,1,2] >= 0
 R976: sbox[1,1,2] + rkey[1,1,2] - tkey[1,1,2] >= 0
 R977: - sbox[1,1,2] + rkey[1,1,2] + tkey[1,1,2] >= 0
 R978: - tbox[2,1,2] + forg[2,1,2] <= 0
 R979: - sbox[2,1,2] + rkey[2,1,2] = 0
 R980: - tbox[3,1,2] + forg[3,1,2] <= 0
 R981: - sbox[3,1,2] + rkey[3,1,2] = 0
 R982: - sbox[3,1,3] + rkey[0,1,2] + rkey[2,3,2] >= 0
 R983: sbox[3,1,3] + rkey[0,1,2] - rkey[2,3,2] >= 0
 R984: sbox[3,1,3] - rkey[0,1,2] + rkey[2,3,2] >= 0
 R985: - sbox[0,1,3] + sbox[3,1,3] + rkey[3,2,2] >= 0
 R986: sbox[0,1,3] + sbox[3,1,3] - rkey[3,2,2] >= 0
 R987: sbox[0,1,3] - sbox[3,1,3] + rkey[3,2,2] >= 0
 R988: - sbox[2,1,3] + rkey[1,0,2] + rkey[2,3,2] >= 0
 R989: sbox[2,1,3] + rkey[1,0,2] - rkey[2,3,2] >= 0
 R990: sbox[2,1,3] - rkey[1,0,2] + rkey[2,3,2] >= 0
 R991: sbox[1,1,3] - rkey[0,1,2] = 0
 R992: tbox[0,1,2] + tbox[2,3,2] - tbox[3,1,3] >= 0
 R993: tbox[0,1,2] - tbox[2,3,2] + tbox[3,1,3] >= 0
 R994: - tbox[0,1,2] + tbox[2,3,2] + tbox[3,1,3] >= 0
 R995: - tbox[0,1,3] + tbox[3,1,3] + tbox[3,2,2] >= 0
 R996: tbox[0,1,3] + tbox[3,1,3] - tbox[3,2,2] >= 0
 R997: tbox[0,1,3] - tbox[3,1,3] + tbox[3,2,2] >= 0
 R998: tbox[1,0,2] - tbox[2,1,3] + tbox[2,3,2] >= 0
 R999: tbox[1,0,2] + tbox[2,1,3] - tbox[2,3,2] >= 0
 R1000: - tbox[1,0,2] + tbox[2,1,3] + tbox[2,3,2] >= 0
 R1001: - tbox[0,1,2] + tbox[1,1,3] = 0
 R1002: unkn[0,1,2] - unkn[3,1,3] + forg[3,1,3] <= 0
 R1003: unkn[2,3,2] - unkn[3,1,3] + forg[3,1,3] <= 0
 R1004: unkn[0,1,2] + unkn[2,3,2] - unkn[3,1,3] + forg[3,1,3] >= 0
 R1005: - unkn[0,1,3] + unkn[3,1,3] + forg[0,1,3] - forg[3,1,3] <= 0
 R1006: - unkn[0,1,3] + unkn[3,2,2] + forg[0,1,3] <= 0
 R1007: - unkn[0,1,3] + unkn[3,1,3] + unkn[3,2,2] + forg[0,1,3]
   - forg[3,1,3] >= 0
 R1008: unkn[1,0,2] - unkn[2,1,3] + forg[2,1,3] <= 0
 R1009: - unkn[2,1,3] + unkn[2,3,2] + forg[2,1,3] <= 0
 R1010: unkn[1,0,2] - unkn[2,1,3] + unkn[2,3,2] + forg[2,1,3] >= 0
 R1011: unkn[0,1,2] - unkn[1,1,3] + forg[1,1,3] = 0
 R1012: - tbox[0,2,2] + forg[0,2,2] <= 0
 R1013: sbox[0,2,2] - rkey[0,2,2] + tkey[0,2,2] >= 0
 R1014: sbox[0,2,2] + rkey[0,2,2] - tkey[0,2,2] >= 0
 R1015: - sbox[0,2,2] + rkey[0,2,2] + tkey[0,2,2] >= 0
 R1016: - tbox[1,2,2] + forg[1,2,2] <= 0
 R1017: sbox[1,2,2] - rkey[1,2,2] + tkey[1,2,2] >= 0
 R1018: sbox[1,2,2] + rkey[1,2,2] - tkey[1,2,2] >= 0
 R1019: - sbox[1,2,2] + rkey[1,2,2] + tkey[1,2,2] >= 0
 R1020: - tbox[2,2,2] + forg[2,2,2] <= 0
 R1021: - sbox[2,2,2] + rkey[2,2,2] = 0
 R1022: - tbox[3,2,2] + forg[3,2,2] <= 0
 R1023: - sbox[3,2,2] + rkey[3,2,2] = 0
 R1024: - sbox[3,2,3] + rkey[0,2,2] + rkey[2,0,2] >= 0
 R1025: sbox[3,2,3] + rkey[0,2,2] - rkey[2,0,2] >= 0
 R1026: sbox[3,2,3] - rkey[0,2,2] + rkey[2,0,2] >= 0
 R1027: - sbox[0,2,3] + sbox[3,2,3] + rkey[3,3,2] >= 0
 R1028: sbox[0,2,3] + sbox[3,2,3] - rkey[3,3,2] >= 0
 R1029: sbox[0,2,3] - sbox[3,2,3] + rkey[3,3,2] >= 0
 R1030: - sbox[2,2,3] + rkey[1,1,2] + rkey[2,0,2] >= 0
 R1031: sbox[2,2,3] + rkey[1,1,2] - rkey[2,0,2] >= 0
 R1032: sbox[2,2,3] - rkey[1,1,2] + rkey[2,0,2] >= 0
 R1033: sbox[1,2,3] - rkey[0,2,2] = 0
 R1034: tbox[0,2,2] + tbox[2,0,2] - tbox[3,2,3] >= 0
 R1035: tbox[0,2,2] - tbox[2,0,2] + tbox[3,2,3] >= 0
 R1036: - tbox[0,2,2] + tbox[2,0,2] + tbox[3,2,3] >= 0
 R1037: - tbox[0,2,3] + tbox[3,2,3] + tbox[3,3,2] >= 0
 R1038: tbox[0,2,3] + tbox[3,2,3] - tbox[3,3,2] >= 0
 R1039: tbox[0,2,3] - tbox[3,2,3] + tbox[3,3,2] >= 0
 R1040: tbox[1,1,2] + tbox[2,0,2] - tbox[2,2,3] >= 0
 R1041: tbox[1,1,2] - tbox[2,0,2] + tbox[2,2,3] >= 0
 R1042: - tbox[1,1,2] + tbox[2,0,2] + tbox[2,2,3] >= 0
 R1043: - tbox[0,2,2] + tbox[1,2,3] = 0
 R1044: unkn[0,2,2] - unkn[3,2,3] + forg[3,2,3] <= 0
 R1045: unkn[2,0,2] - unkn[3,2,3] + forg[3,2,3] <= 0
 R1046: unkn[0,2,2] + unkn[2,0,2] - unkn[3,2,3] + forg[3,2,3] >= 0
 R1047: - unkn[0,2,3] + unkn[3,2,3] + forg[0,2,3] - forg[3,2,3] <= 0
 R1048: - unkn[0,2,3] + unkn[3,3,2] + forg[0,2,3] <= 0
 R1049: - unkn[0,2,3] + unkn[3,2,3] + unkn[3,3,2] + forg[0,2,3]
   - forg[3,2,3] >= 0
 R1050: unkn[1,1,2] - unkn[2,2,3] + forg[2,2,3] <= 0
 R1051: unkn[2,0,2] - unkn[2,2,3] + forg[2,2,3] <= 0
 R1052: unkn[1,1,2] + unkn[2,0,2] - unkn[2,2,3] + forg[2,2,3] >= 0
 R1053: unkn[0,2,2] - unkn[1,2,3] + forg[1,2,3] = 0
 R1054: - tbox[0,3,2] + forg[0,3,2] <= 0
 R1055: sbox[0,3,2] - rkey[0,3,2] + tkey[0,3,2] >= 0
 R1056: sbox[0,3,2] + rkey[0,3,2] - tkey[0,3,2] >= 0
 R1057: - sbox[0,3,2] + rkey[0,3,2] + tkey[0,3,2] >= 0
 R1058: - tbox[1,3,2] + forg[1,3,2] <= 0
 R1059: sbox[1,3,2] - rkey[1,3,2] + tkey[1,3,2] >= 0
 R1060: sbox[1,3,2] + rkey[1,3,2] - tkey[1,3,2] >= 0
 R1061: - sbox[1,3,2] + rkey[1,3,2] + tkey[1,3,2] >= 0
 R1062: - tbox[2,3,2] + forg[2,3,2] <= 0
 R1063: - sbox[2,3,2] + rkey[2,3,2] = 0
 R1064: - tbox[3,3,2] + forg[3,3,2] <= 0
 R1065: - sbox[3,3,2] + rkey[3,3,2] = 0
 R1066: - sbox[3,3,3] + rkey[0,3,2] + rkey[2,1,2] >= 0
 R1067: sbox[3,3,3] + rkey[0,3,2] - rkey[2,1,2] >= 0
 R1068: sbox[3,3,3] - rkey[0,3,2] + rkey[2,1,2] >= 0
 R1069: - sbox[0,3,3] + sbox[3,3,3] + rkey[3,0,2] >= 0
 R1070: sbox[0,3,3] + sbox[3,3,3] - rkey[3,0,2] >= 0
 R1071: sbox[0,3,3] - sbox[3,3,3] + rkey[3,0,2] >= 0
 R1072: - sbox[2,3,3] + rkey[1,2,2] + rkey[2,1,2] >= 0
 R1073: sbox[2,3,3] + rkey[1,2,2] - rkey[2,1,2] >= 0
 R1074: sbox[2,3,3] - rkey[1,2,2] + rkey[2,1,2] >= 0
 R1075: sbox[1,3,3] - rkey[0,3,2] = 0
 R1076: tbox[0,3,2] + tbox[2,1,2] - tbox[3,3,3] >= 0
 R1077: tbox[0,3,2] - tbox[2,1,2] + tbox[3,3,3] >= 0
 R1078: - tbox[0,3,2] + tbox[2,1,2] + tbox[3,3,3] >= 0
 R1079: - tbox[0,3,3] + tbox[3,0,2] + tbox[3,3,3] >= 0
 R1080: tbox[0,3,3] - tbox[3,0,2] + tbox[3,3,3] >= 0
 R1081: tbox[0,3,3] + tbox[3,0,2] - tbox[3,3,3] >= 0
 R1082: tbox[1,2,2] + tbox[2,1,2] - tbox[2,3,3] >= 0
 R1083: tbox[1,2,2] - tbox[2,1,2] + tbox[2,3,3] >= 0
 R1084: - tbox[1,2,2] + tbox[2,1,2] + tbox[2,3,3] >= 0
 R1085: - tbox[0,3,2] + tbox[1,3,3] = 0
 R1086: unkn[0,3,2] - unkn[3,3,3] + forg[3,3,3] <= 0
 R1087: unkn[2,1,2] - unkn[3,3,3] + forg[3,3,3] <= 0
 R1088: unkn[0,3,2] + unkn[2,1,2] - unkn[3,3,3] + forg[3,3,3] >= 0
 R1089: - unkn[0,3,3] + unkn[3,3,3] + forg[0,3,3] - forg[3,3,3] <= 0
 R1090: - unkn[0,3,3] + unkn[3,0,2] + forg[0,3,3] <= 0
 R1091: - unkn[0,3,3] + unkn[3,0,2] + unkn[3,3,3] + forg[0,3,3]
   - forg[3,3,3] >= 0
 R1092: unkn[1,2,2] - unkn[2,3,3] + forg[2,3,3] <= 0
 R1093: unkn[2,1,2] - unkn[2,3,3] + forg[2,3,3] <= 0
 R1094: unkn[1,2,2] + unkn[2,1,2] - unkn[2,3,3] + forg[2,3,3] >= 0
 R1095: unkn[0,3,2] - unkn[1,3,3] + forg[1,3,3] = 0
 R1096: - tbox[0,0,3] + forg[0,0,3] <= 0
 R1097: sbox[0,0,3] - rkey[0,0,3] + tkey[0,0,3] >= 0
 R1098: sbox[0,0,3] + rkey[0,0,3] - tkey[0,0,3] >= 0
 R1099: - sbox[0,0,3] + rkey[0,0,3] + tkey[0,0,3] >= 0
 R1100: - tbox[1,0,3] + forg[1,0,3] <= 0
 R1101: sbox[1,0,3] - rkey[1,0,3] + tkey[1,0,3] >= 0
 R1102: sbox[1,0,3] + rkey[1,0,3] - tkey[1,0,3] >= 0
 R1103: - sbox[1,0,3] + rkey[1,0,3] + tkey[1,0,3] >= 0
 R1104: - tbox[2,0,3] + forg[2,0,3] <= 0
 R1105: - sbox[2,0,3] + rkey[2,0,3] = 0
 R1106: - tbox[3,0,3] + forg[3,0,3] <= 0
 R1107: - sbox[3,0,3] + rkey[3,0,3] = 0
 R1108: - sbox[3,0,4] + rkey[0,0,3] + rkey[2,2,3] >= 0
 R1109: sbox[3,0,4] + rkey[0,0,3] - rkey[2,2,3] >= 0
 R1110: sbox[3,0,4] - rkey[0,0,3] + rkey[2,2,3] >= 0
 R1111: - sbox[0,0,4] + sbox[3,0,4] + rkey[3,1,3] >= 0
 R1112: sbox[0,0,4] + sbox[3,0,4] - rkey[3,1,3] >= 0
 R1113: sbox[0,0,4] - sbox[3,0,4] + rkey[3,1,3] >= 0
 R1114: - sbox[2,0,4] + rkey[1,3,3] + rkey[2,2,3] >= 0
 R1115: sbox[2,0,4] + rkey[1,3,3] - rkey[2,2,3] >= 0
 R1116: sbox[2,0,4] - rkey[1,3,3] + rkey[2,2,3] >= 0
 R1117: sbox[1,0,4] - rkey[0,0,3] = 0
 R1118: tbox[0,0,3] + tbox[2,2,3] - tbox[3,0,4] >= 0
 R1119: tbox[0,0,3] - tbox[2,2,3] + tbox[3,0,4] >= 0
 R1120: - tbox[0,0,3] + tbox[2,2,3] + tbox[3,0,4] >= 0
 R1121: - tbox[0,0,4] + tbox[3,0,4] + tbox[3,1,3] >= 0
 R1122: tbox[0,0,4] + tbox[3,0,4] - tbox[3,1,3] >= 0
 R1123: tbox[0,0,4] - tbox[3,0,4] + tbox[3,1,3] >= 0
 R1124: tbox[1,3,3] - tbox[2,0,4] + tbox[2,2,3] >= 0
 R1125: tbox[1,3,3] + tbox[2,0,4] - tbox[2,2,3] >= 0
 R1126: - tbox[1,3,3] + tbox[2,0,4] + tbox[2,2,3] >= 0
 R1127: - tbox[0,0,3] + tbox[1,0,4] = 0
 R1128: unkn[0,0,3] - unkn[3,0,4] + forg[3,0,4] <= 0
 R1129: unkn[2,2,3] - unkn[3,0,4] + forg[3,0,4] <= 0
 R1130: unkn[0,0,3] + unkn[2,2,3] - unkn[3,0,4] + forg[3,0,4] >= 0
 R1131: - unkn[0,0,4] + unkn[3,0,4] + forg[0,0,4] - forg[3,0,4] <= 0
 R1132: - unkn[0,0,4] + unkn[3,1,3] + forg[0,0,4] <= 0
 R1133: - unkn[0,0,4] + unkn[3,0,4] + unkn[3,1,3] + forg[0,0,4]
   - forg[3,0,4] >= 0
 R1134: unkn[1,3,3] - unkn[2,0,4] + forg[2,0,4] <= 0
 R1135: - unkn[2,0,4] + unkn[2,2,3] + forg[2,0,4] <= 0
 R1136: unkn[1,3,3] - unkn[2,0,4] + unkn[2,2,3] + forg[2,0,4] >= 0
 R1137: unkn[0,0,3] - unkn[1,0,4] + forg[1,0,4] = 0
 R1138: - tbox[0,1,3] + forg[0,1,3] <= 0
 R1139: sbox[0,1,3] - rkey[0,1,3] + tkey[0,1,3] >= 0
 R1140: sbox[0,1,3] + rkey[0,1,3] - tkey[0,1,3] >= 0
 R1141: - sbox[0,1,3] + rkey[0,1,3] + tkey[0,1,3] >= 0
 R1142: - tbox[1,1,3] + forg[1,1,3] <= 0
 R1143: sbox[1,1,3] - rkey[1,1,3] + tkey[1,1,3] >= 0
 R1144: sbox[1,1,3] + rkey[1,1,3] - tkey[1,1,3] >= 0
 R1145: - sbox[1,1,3] + rkey[1,1,3] + tkey[1,1,3] >= 0
 R1146: - tbox[2,1,3] + forg[2,1,3] <= 0
 R1147: - sbox[2,1,3] + rkey[2,1,3] = 0
 R1148: - tbox[3,1,3] + forg[3,1,3] <= 0
 R1149: - sbox[3,1,3] + rkey[3,1,3] = 0
 R1150: - sbox[3,1,4] + rkey[0,1,3] + rkey[2,3,3] >= 0
 R1151: sbox[3,1,4] + rkey[0,1,3] - rkey[2,3,3] >= 0
 R1152: sbox[3,1,4] - rkey[0,1,3] + rkey[2,3,3] >= 0
 R1153: - sbox[0,1,4] + sbox[3,1,4] + rkey[3,2,3] >= 0
 R1154: sbox[0,1,4] + sbox[3,1,4] - rkey[3,2,3] >= 0
 R1155: sbox[0,1,4] - sbox[3,1,4] + rkey[3,2,3] >= 0
 R1156: - sbox[2,1,4] + rkey[1,0,3] + rkey[2,3,3] >= 0
 R1157: sbox[2,1,4] + rkey[1,0,3] - rkey[2,3,3] >= 0
 R1158: sbox[2,1,4] - rkey[1,0,3] + rkey[2,3,3] >= 0
 R1159: sbox[1,1,4] - rkey[0,1,3] = 0
 R1160: tbox[0,1,3] + tbox[2,3,3] - tbox[3,1,4] >= 0
 R1161: tbox[0,1,3] - tbox[2,3,3] + tbox[3,1,4] >= 0
 R1162: - tbox[0,1,3] + tbox[2,3,3] + tbox[3,1,4] >= 0
 R1163: - tbox[0,1,4] + tbox[3,1,4] + tbox[3,2,3] >= 0
 R1164: tbox[0,1,4] + tbox[3,1,4] - tbox[3,2,3] >= 0
 R1165: tbox[0,1,4] - tbox[3,1,4] + tbox[3,2,3] >= 0
 R1166: tbox[1,0,3] - tbox[2,1,4] + tbox[2,3,3] >= 0
 R1167: tbox[1,0,3] + tbox[2,1,4] - tbox[2,3,3] >= 0
 R1168: - tbox[1,0,3] + tbox[2,1,4] + tbox[2,3,3] >= 0
 R1169: - tbox[0,1,3] + tbox[1,1,4] = 0
 R1170: unkn[0,1,3] - unkn[3,1,4] + forg[3,1,4] <= 0
 R1171: unkn[2,3,3] - unkn[3,1,4] + forg[3,1,4] <= 0
 R1172: unkn[0,1,3] + unkn[2,3,3] - unkn[3,1,4] + forg[3,1,4] >= 0
 R1173: - unkn[0,1,4] + unkn[3,1,4] + forg[0,1,4] - forg[3,1,4] <= 0
 R1174: - unkn[0,1,4] + unkn[3,2,3] + forg[0,1,4] <= 0
 R1175: - unkn[0,1,4] + unkn[3,1,4] + unkn[3,2,3] + forg[0,1,4]
   - forg[3,1,4] >= 0
 R1176: unkn[1,0,3] - unkn[2,1,4] + forg[2,1,4] <= 0
 R1177: - unkn[2,1,4] + unkn[2,3,3] + forg[2,1,4] <= 0
 R1178: unkn[1,0,3] - unkn[2,1,4] + unkn[2,3,3] + forg[2,1,4] >= 0
 R1179: unkn[0,1,3] - unkn[1,1,4] + forg[1,1,4] = 0
 R1180: - tbox[0,2,3] + forg[0,2,3] <= 0
 R1181: sbox[0,2,3] - rkey[0,2,3] + tkey[0,2,3] >= 0
 R1182: sbox[0,2,3] + rkey[0,2,3] - tkey[0,2,3] >= 0
 R1183: - sbox[0,2,3] + rkey[0,2,3] + tkey[0,2,3] >= 0
 R1184: - tbox[1,2,3] + forg[1,2,3] <= 0
 R1185: sbox[1,2,3] - rkey[1,2,3] + tkey[1,2,3] >= 0
 R1186: sbox[1,2,3] + rkey[1,2,3] - tkey[1,2,3] >= 0
 R1187: - sbox[1,2,3] + rkey[1,2,3] + tkey[1,2,3] >= 0
 R1188: - tbox[2,2,3] + forg[2,2,3] <= 0
 R1189: - sbox[2,2,3] + rkey[2,2,3] = 0
 R1190: - tbox[3,2,3] + forg[3,2,3] <= 0
 R1191: - sbox[3,2,3] + rkey[3,2,3] = 0
 R1192: - sbox[3,2,4] + rkey[0,2,3] + rkey[2,0,3] >= 0
 R1193: sbox[3,2,4] + rkey[0,2,3] - rkey[2,0,3] >= 0
 R1194: sbox[3,2,4] - rkey[0,2,3] + rkey[2,0,3] >= 0
 R1195: - sbox[0,2,4] + sbox[3,2,4] + rkey[3,3,3] >= 0
 R1196: sbox[0,2,4] + sbox[3,2,4] - rkey[3,3,3] >= 0
 R1197: sbox[0,2,4] - sbox[3,2,4] + rkey[3,3,3] >= 0
 R1198: - sbox[2,2,4] + rkey[1,1,3] + rkey[2,0,3] >= 0
 R1199: sbox[2,2,4] + rkey[1,1,3] - rkey[2,0,3] >= 0
 R1200: sbox[2,2,4] - rkey[1,1,3] + rkey[2,0,3] >= 0
 R1201: sbox[1,2,4] - rkey[0,2,3] = 0
 R1202: tbox[0,2,3] + tbox[2,0,3] - tbox[3,2,4] >= 0
 R1203: tbox[0,2,3] - tbox[2,0,3] + tbox[3,2,4] >= 0
 R1204: - tbox[0,2,3] + tbox[2,0,3] + tbox[3,2,4] >= 0
 R1205: - tbox[0,2,4] + tbox[3,2,4] + tbox[3,3,3] >= 0
 R1206: tbox[0,2,4] + tbox[3,2,4] - tbox[3,3,3] >= 0
 R1207: tbox[0,2,4] - tbox[3,2,4] + tbox[3,3,3] >= 0
 R1208: tbox[1,1,3] + tbox[2,0,3] - tbox[2,2,4] >= 0
 R1209: tbox[1,1,3] - tbox[2,0,3] + tbox[2,2,4] >= 0
 R1210: - tbox[1,1,3] + tbox[2,0,3] + tbox[2,2,4] >= 0
 R1211: - tbox[0,2,3] + tbox[1,2,4] = 0
 R1212: unkn[0,2,3] - unkn[3,2,4] + forg[3,2,4] <= 0
 R1213: unkn[2,0,3] - unkn[3,2,4] + forg[3,2,4] <= 0
 R1214: unkn[0,2,3] + unkn[2,0,3] - unkn[3,2,4] + forg[3,2,4] >= 0
 R1215: - unkn[0,2,4] + unkn[3,2,4] + forg[0,2,4] - forg[3,2,4] <= 0
 R1216: - unkn[0,2,4] + unkn[3,3,3] + forg[0,2,4] <= 0
 R1217: - unkn[0,2,4] + unkn[3,2,4] + unkn[3,3,3] + forg[0,2,4]
   - forg[3,2,4] >= 0
 R1218: unkn[1,1,3] - unkn[2,2,4] + forg[2,2,4] <= 0
 R1219: unkn[2,0,3] - unkn[2,2,4] + forg[2,2,4] <= 0
 R1220: unkn[1,1,3] + unkn[2,0,3] - unkn[2,2,4] + forg[2,2,4] >= 0
 R1221: unkn[0,2,3] - unkn[1,2,4] + forg[1,2,4] = 0
 R1222: - tbox[0,3,3] + forg[0,3,3] <= 0
 R1223: sbox[0,3,3] - rkey[0,3,3] + tkey[0,3,3] >= 0
 R1224: sbox[0,3,3] + rkey[0,3,3] - tkey[0,3,3] >= 0
 R1225: - sbox[0,3,3] + rkey[0,3,3] + tkey[0,3,3] >= 0
 R1226: - tbox[1,3,3] + forg[1,3,3] <= 0
 R1227: sbox[1,3,3] - rkey[1,3,3] + tkey[1,3,3] >= 0
 R1228: sbox[1,3,3] + rkey[1,3,3] - tkey[1,3,3] >= 0
 R1229: - sbox[1,3,3] + rkey[1,3,3] + tkey[1,3,3] >= 0
 R1230: - tbox[2,3,3] + forg[2,3,3] <= 0
 R1231: - sbox[2,3,3] + rkey[2,3,3] = 0
 R1232: - tbox[3,3,3] + forg[3,3,3] <= 0
 R1233: - sbox[3,3,3] + rkey[3,3,3] = 0
 R1234: - sbox[3,3,4] + rkey[0,3,3] + rkey[2,1,3] >= 0
 R1235: sbox[3,3,4] + rkey[0,3,3] - rkey[2,1,3] >= 0
 R1236: sbox[3,3,4] - rkey[0,3,3] + rkey[2,1,3] >= 0
 R1237: - sbox[0,3,4] + sbox[3,3,4] + rkey[3,0,3] >= 0
 R1238: sbox[0,3,4] + sbox[3,3,4] - rkey[3,0,3] >= 0
 R1239: sbox[0,3,4] - sbox[3,3,4] + rkey[3,0,3] >= 0
 R1240: - sbox[2,3,4] + rkey[1,2,3] + rkey[2,1,3] >= 0
 R1241: sbox[2,3,4] + rkey[1,2,3] - rkey[2,1,3] >= 0
 R1242: sbox[2,3,4] - rkey[1,2,3] + rkey[2,1,3] >= 0
 R1243: sbox[1,3,4] - rkey[0,3,3] = 0
 R1244: tbox[0,3,3] + tbox[2,1,3] - tbox[3,3,4] >= 0
 R1245: tbox[0,3,3] - tbox[2,1,3] + tbox[3,3,4] >= 0
 R1246: - tbox[0,3,3] + tbox[2,1,3] + tbox[3,3,4] >= 0
 R1247: - tbox[0,3,4] + tbox[3,0,3] + tbox[3,3,4] >= 0
 R1248: tbox[0,3,4] - tbox[3,0,3] + tbox[3,3,4] >= 0
 R1249: tbox[0,3,4] + tbox[3,0,3] - tbox[3,3,4] >= 0
 R1250: tbox[1,2,3] + tbox[2,1,3] - tbox[2,3,4] >= 0
 R1251: tbox[1,2,3] - tbox[2,1,3] + tbox[2,3,4] >= 0
 R1252: - tbox[1,2,3] + tbox[2,1,3] + tbox[2,3,4] >= 0
 R1253: - tbox[0,3,3] + tbox[1,3,4] = 0
 R1254: unkn[0,3,3] - unkn[3,3,4] + forg[3,3,4] <= 0
 R1255: unkn[2,1,3] - unkn[3,3,4] + forg[3,3,4] <= 0
 R1256: unkn[0,3,3] + unkn[2,1,3] - unkn[3,3,4] + forg[3,3,4] >= 0
 R1257: - unkn[0,3,4] + unkn[3,3,4] + forg[0,3,4] - forg[3,3,4] <= 0
 R1258: - unkn[0,3,4] + unkn[3,0,3] + forg[0,3,4] <= 0
 R1259: - unkn[0,3,4] + unkn[3,0,3] + unkn[3,3,4] + forg[0,3,4]
   - forg[3,3,4] >= 0
 R1260: unkn[1,2,3] - unkn[2,3,4] + forg[2,3,4] <= 0
 R1261: unkn[2,1,3] - unkn[2,3,4] + forg[2,3,4] <= 0
 R1262: unkn[1,2,3] + unkn[2,1,3] - unkn[2,3,4] + forg[2,3,4] >= 0
 R1263: unkn[0,3,3] - unkn[1,3,4] + forg[1,3,4] = 0
 R1264: - tbox[0,0,4] + forg[0,0,4] <= 0
 R1265: sbox[0,0,4] - rkey[0,0,4] + tkey[0,0,4] >= 0
 R1266: sbox[0,0,4] + rkey[0,0,4] - tkey[0,0,4] >= 0
 R1267: - sbox[0,0,4] + rkey[0,0,4] + tkey[0,0,4] >= 0
 R1268: - tbox[1,0,4] + forg[1,0,4] <= 0
 R1269: sbox[1,0,4] - rkey[1,0,4] + tkey[1,0,4] >= 0
 R1270: sbox[1,0,4] + rkey[1,0,4] - tkey[1,0,4] >= 0
 R1271: - sbox[1,0,4] + rkey[1,0,4] + tkey[1,0,4] >= 0
 R1272: - tbox[2,0,4] + forg[2,0,4] <= 0
 R1273: - sbox[2,0,4] + rkey[2,0,4] = 0
 R1274: - tbox[3,0,4] + forg[3,0,4] <= 0
 R1275: - sbox[3,0,4] + rkey[3,0,4] = 0
 R1276: - sbox[3,0,5] + rkey[0,0,4] + rkey[2,2,4] >= 0
 R1277: sbox[3,0,5] + rkey[0,0,4] - rkey[2,2,4] >= 0
 R1278: sbox[3,0,5] - rkey[0,0,4] + rkey[2,2,4] >= 0
 R1279: - sbox[0,0,5] + sbox[3,0,5] + rkey[3,1,4] >= 0
 R1280: sbox[0,0,5] + sbox[3,0,5] - rkey[3,1,4] >= 0
 R1281: sbox[0,0,5] - sbox[3,0,5] + rkey[3,1,4] >= 0
 R1282: - sbox[2,0,5] + rkey[1,3,4] + rkey[2,2,4] >= 0
 R1283: sbox[2,0,5] + rkey[1,3,4] - rkey[2,2,4] >= 0
 R1284: sbox[2,0,5] - rkey[1,3,4] + rkey[2,2,4] >= 0
 R1285: sbox[1,0,5] - rkey[0,0,4] = 0
 R1286: tbox[0,0,4] + tbox[2,2,4] - tbox[3,0,5] >= 0
 R1287: tbox[0,0,4] - tbox[2,2,4] + tbox[3,0,5] >= 0
 R1288: - tbox[0,0,4] + tbox[2,2,4] + tbox[3,0,5] >= 0
 R1289: - tbox[0,0,5] + tbox[3,0,5] + tbox[3,1,4] >= 0
 R1290: tbox[0,0,5] + tbox[3,0,5] - tbox[3,1,4] >= 0
 R1291: tbox[0,0,5] - tbox[3,0,5] + tbox[3,1,4] >= 0
 R1292: tbox[1,3,4] - tbox[2,0,5] + tbox[2,2,4] >= 0
 R1293: tbox[1,3,4] + tbox[2,0,5] - tbox[2,2,4] >= 0
 R1294: - tbox[1,3,4] + tbox[2,0,5] + tbox[2,2,4] >= 0
 R1295: - tbox[0,0,4] + tbox[1,0,5] = 0
 R1296: unkn[0,0,4] - unkn[3,0,5] + forg[3,0,5] <= 0
 R1297: unkn[2,2,4] - unkn[3,0,5] + forg[3,0,5] <= 0
 R1298: unkn[0,0,4] + unkn[2,2,4] - unkn[3,0,5] + forg[3,0,5] >= 0
 R1299: - unkn[0,0,5] + unkn[3,0,5] + forg[0,0,5] - forg[3,0,5] <= 0
 R1300: - unkn[0,0,5] + unkn[3,1,4] + forg[0,0,5] <= 0
 R1301: - unkn[0,0,5] + unkn[3,0,5] + unkn[3,1,4] + forg[0,0,5]
   - forg[3,0,5] >= 0
 R1302: unkn[1,3,4] - unkn[2,0,5] + forg[2,0,5] <= 0
 R1303: - unkn[2,0,5] + unkn[2,2,4] + forg[2,0,5] <= 0
 R1304: unkn[1,3,4] - unkn[2,0,5] + unkn[2,2,4] + forg[2,0,5] >= 0
 R1305: unkn[0,0,4] - unkn[1,0,5] + forg[1,0,5] = 0
 R1306: - tbox[0,1,4] + forg[0,1,4] <= 0
 R1307: sbox[0,1,4] - rkey[0,1,4] + tkey[0,1,4] >= 0
 R1308: sbox[0,1,4] + rkey[0,1,4] - tkey[0,1,4] >= 0
 R1309: - sbox[0,1,4] + rkey[0,1,4] + tkey[0,1,4] >= 0
 R1310: - tbox[1,1,4] + forg[1,1,4] <= 0
 R1311: sbox[1,1,4] - rkey[1,1,4] + tkey[1,1,4] >= 0
 R1312: sbox[1,1,4] + rkey[1,1,4] - tkey[1,1,4] >= 0
 R1313: - sbox[1,1,4] + rkey[1,1,4] + tkey[1,1,4] >= 0
 R1314: - tbox[2,1,4] + forg[2,1,4] <= 0
 R1315: - sbox[2,1,4] + rkey[2,1,4] = 0
 R1316: - tbox[3,1,4] + forg[3,1,4] <= 0
 R1317: - sbox[3,1,4] + rkey[3,1,4] = 0
 R1318: - sbox[3,1,5] + rkey[0,1,4] + rkey[2,3,4] >= 0
 R1319: sbox[3,1,5] + rkey[0,1,4] - rkey[2,3,4] >= 0
 R1320: sbox[3,1,5] - rkey[0,1,4] + rkey[2,3,4] >= 0
 R1321: - sbox[0,1,5] + sbox[3,1,5] + rkey[3,2,4] >= 0
 R1322: sbox[0,1,5] + sbox[3,1,5] - rkey[3,2,4] >= 0
 R1323: sbox[0,1,5] - sbox[3,1,5] + rkey[3,2,4] >= 0
 R1324: - sbox[2,1,5] + rkey[1,0,4] + rkey[2,3,4] >= 0
 R1325: sbox[2,1,5] + rkey[1,0,4] - rkey[2,3,4] >= 0
 R1326: sbox[2,1,5] - rkey[1,0,4] + rkey[2,3,4] >= 0
 R1327: sbox[1,1,5] - rkey[0,1,4] = 0
 R1328: tbox[0,1,4] + tbox[2,3,4] - tbox[3,1,5] >= 0
 R1329: tbox[0,1,4] - tbox[2,3,4] + tbox[3,1,5] >= 0
 R1330: - tbox[0,1,4] + tbox[2,3,4] + tbox[3,1,5] >= 0
 R1331: - tbox[0,1,5] + tbox[3,1,5] + tbox[3,2,4] >= 0
 R1332: tbox[0,1,5] + tbox[3,1,5] - tbox[3,2,4] >= 0
 R1333: tbox[0,1,5] - tbox[3,1,5] + tbox[3,2,4] >= 0
 R1334: tbox[1,0,4] - tbox[2,1,5] + tbox[2,3,4] >= 0
 R1335: tbox[1,0,4] + tbox[2,1,5] - tbox[2,3,4] >= 0
 R1336: - tbox[1,0,4] + tbox[2,1,5] + tbox[2,3,4] >= 0
 R1337: - tbox[0,1,4] + tbox[1,1,5] = 0
 R1338: unkn[0,1,4] - unkn[3,1,5] + forg[3,1,5] <= 0
 R1339: unkn[2,3,4] - unkn[3,1,5] + forg[3,1,5] <= 0
 R1340: unkn[0,1,4] + unkn[2,3,4] - unkn[3,1,5] + forg[3,1,5] >= 0
 R1341: - unkn[0,1,5] + unkn[3,1,5] + forg[0,1,5] - forg[3,1,5] <= 0
 R1342: - unkn[0,1,5] + unkn[3,2,4] + forg[0,1,5] <= 0
 R1343: - unkn[0,1,5] + unkn[3,1,5] + unkn[3,2,4] + forg[0,1,5]
   - forg[3,1,5] >= 0
 R1344: unkn[1,0,4] - unkn[2,1,5] + forg[2,1,5] <= 0
 R1345: - unkn[2,1,5] + unkn[2,3,4] + forg[2,1,5] <= 0
 R1346: unkn[1,0,4] - unkn[2,1,5] + unkn[2,3,4] + forg[2,1,5] >= 0
 R1347: unkn[0,1,4] - unkn[1,1,5] + forg[1,1,5] = 0
 R1348: - tbox[0,2,4] + forg[0,2,4] <= 0
 R1349: sbox[0,2,4] - rkey[0,2,4] + tkey[0,2,4] >= 0
 R1350: sbox[0,2,4] + rkey[0,2,4] - tkey[0,2,4] >= 0
 R1351: - sbox[0,2,4] + rkey[0,2,4] + tkey[0,2,4] >= 0
 R1352: - tbox[1,2,4] + forg[1,2,4] <= 0
 R1353: sbox[1,2,4] - rkey[1,2,4] + tkey[1,2,4] >= 0
 R1354: sbox[1,2,4] + rkey[1,2,4] - tkey[1,2,4] >= 0
 R1355: - sbox[1,2,4] + rkey[1,2,4] + tkey[1,2,4] >= 0
 R1356: - tbox[2,2,4] + forg[2,2,4] <= 0
 R1357: - sbox[2,2,4] + rkey[2,2,4] = 0
 R1358: - tbox[3,2,4] + forg[3,2,4] <= 0
 R1359: - sbox[3,2,4] + rkey[3,2,4] = 0
 R1360: - sbox[3,2,5] + rkey[0,2,4] + rkey[2,0,4] >= 0
 R1361: sbox[3,2,5] + rkey[0,2,4] - rkey[2,0,4] >= 0
 R1362: sbox[3,2,5] - rkey[0,2,4] + rkey[2,0,4] >= 0
 R1363: - sbox[0,2,5] + sbox[3,2,5] + rkey[3,3,4] >= 0
 R1364: sbox[0,2,5] + sbox[3,2,5] - rkey[3,3,4] >= 0
 R1365: sbox[0,2,5] - sbox[3,2,5] + rkey[3,3,4] >= 0
 R1366: - sbox[2,2,5] + rkey[1,1,4] + rkey[2,0,4] >= 0
 R1367: sbox[2,2,5] + rkey[1,1,4] - rkey[2,0,4] >= 0
 R1368: sbox[2,2,5] - rkey[1,1,4] + rkey[2,0,4] >= 0
 R1369: sbox[1,2,5] - rkey[0,2,4] = 0
 R1370: tbox[0,2,4] + tbox[2,0,4] - tbox[3,2,5] >= 0
 R1371: tbox[0,2,4] - tbox[2,0,4] + tbox[3,2,5] >= 0
 R1372: - tbox[0,2,4] + tbox[2,0,4] + tbox[3,2,5] >= 0
 R1373: - tbox[0,2,5] + tbox[3,2,5] + tbox[3,3,4] >= 0
 R1374: tbox[0,2,5] + tbox[3,2,5] - tbox[3,3,4] >= 0
 R1375: tbox[0,2,5] - tbox[3,2,5] + tbox[3,3,4] >= 0
 R1376: tbox[1,1,4] + tbox[2,0,4] - tbox[2,2,5] >= 0
 R1377: tbox[1,1,4] - tbox[2,0,4] + tbox[2,2,5] >= 0
 R1378: - tbox[1,1,4] + tbox[2,0,4] + tbox[2,2,5] >= 0
 R1379: - tbox[0,2,4] + tbox[1,2,5] = 0
 R1380: unkn[0,2,4] - unkn[3,2,5] + forg[3,2,5] <= 0
 R1381: unkn[2,0,4] - unkn[3,2,5] + forg[3,2,5] <= 0
 R1382: unkn[0,2,4] + unkn[2,0,4] - unkn[3,2,5] + forg[3,2,5] >= 0
 R1383: - unkn[0,2,5] + unkn[3,2,5] + forg[0,2,5] - forg[3,2,5] <= 0
 R1384: - unkn[0,2,5] + unkn[3,3,4] + forg[0,2,5] <= 0
 R1385: - unkn[0,2,5] + unkn[3,2,5] + unkn[3,3,4] + forg[0,2,5]
   - forg[3,2,5] >= 0
 R1386: unkn[1,1,4] - unkn[2,2,5] + forg[2,2,5] <= 0
 R1387: unkn[2,0,4] - unkn[2,2,5] + forg[2,2,5] <= 0
 R1388: unkn[1,1,4] + unkn[2,0,4] - unkn[2,2,5] + forg[2,2,5] >= 0
 R1389: unkn[0,2,4] - unkn[1,2,5] + forg[1,2,5] = 0
 R1390: - tbox[0,3,4] + forg[0,3,4] <= 0
 R1391: sbox[0,3,4] - rkey[0,3,4] + tkey[0,3,4] >= 0
 R1392: sbox[0,3,4] + rkey[0,3,4] - tkey[0,3,4] >= 0
 R1393: - sbox[0,3,4] + rkey[0,3,4] + tkey[0,3,4] >= 0
 R1394: - tbox[1,3,4] + forg[1,3,4] <= 0
 R1395: sbox[1,3,4] - rkey[1,3,4] + tkey[1,3,4] >= 0
 R1396: sbox[1,3,4] + rkey[1,3,4] - tkey[1,3,4] >= 0
 R1397: - sbox[1,3,4] + rkey[1,3,4] + tkey[1,3,4] >= 0
 R1398: - tbox[2,3,4] + forg[2,3,4] <= 0
 R1399: - sbox[2,3,4] + rkey[2,3,4] = 0
 R1400: - tbox[3,3,4] + forg[3,3,4] <= 0
 R1401: - sbox[3,3,4] + rkey[3,3,4] = 0
 R1402: - sbox[3,3,5] + rkey[0,3,4] + rkey[2,1,4] >= 0
 R1403: sbox[3,3,5] + rkey[0,3,4] - rkey[2,1,4] >= 0
 R1404: sbox[3,3,5] - rkey[0,3,4] + rkey[2,1,4] >= 0
 R1405: - sbox[0,3,5] + sbox[3,3,5] + rkey[3,0,4] >= 0
 R1406: sbox[0,3,5] + sbox[3,3,5] - rkey[3,0,4] >= 0
 R1407: sbox[0,3,5] - sbox[3,3,5] + rkey[3,0,4] >= 0
 R1408: - sbox[2,3,5] + rkey[1,2,4] + rkey[2,1,4] >= 0
 R1409: sbox[2,3,5] + rkey[1,2,4] - rkey[2,1,4] >= 0
 R1410: sbox[2,3,5] - rkey[1,2,4] + rkey[2,1,4] >= 0
 R1411: sbox[1,3,5] - rkey[0,3,4] = 0
 R1412: tbox[0,3,4] + tbox[2,1,4] - tbox[3,3,5] >= 0
 R1413: tbox[0,3,4] - tbox[2,1,4] + tbox[3,3,5] >= 0
 R1414: - tbox[0,3,4] + tbox[2,1,4] + tbox[3,3,5] >= 0
 R1415: - tbox[0,3,5] + tbox[3,0,4] + tbox[3,3,5] >= 0
 R1416: tbox[0,3,5] - tbox[3,0,4] + tbox[3,3,5] >= 0
 R1417: tbox[0,3,5] + tbox[3,0,4] - tbox[3,3,5] >= 0
 R1418: tbox[1,2,4] + tbox[2,1,4] - tbox[2,3,5] >= 0
 R1419: tbox[1,2,4] - tbox[2,1,4] + tbox[2,3,5] >= 0
 R1420: - tbox[1,2,4] + tbox[2,1,4] + tbox[2,3,5] >= 0
 R1421: - tbox[0,3,4] + tbox[1,3,5] = 0
 R1422: unkn[0,3,4] - unkn[3,3,5] + forg[3,3,5] <= 0
 R1423: unkn[2,1,4] - unkn[3,3,5] + forg[3,3,5] <= 0
 R1424: unkn[0,3,4] + unkn[2,1,4] - unkn[3,3,5] + forg[3,3,5] >= 0
 R1425: - unkn[0,3,5] + unkn[3,3,5] + forg[0,3,5] - forg[3,3,5] <= 0
 R1426: - unkn[0,3,5] + unkn[3,0,4] + forg[0,3,5] <= 0
 R1427: - unkn[0,3,5] + unkn[3,0,4] + unkn[3,3,5] + forg[0,3,5]
   - forg[3,3,5] >= 0
 R1428: unkn[1,2,4] - unkn[2,3,5] + forg[2,3,5] <= 0
 R1429: unkn[2,1,4] - unkn[2,3,5] + forg[2,3,5] <= 0
 R1430: unkn[1,2,4] + unkn[2,1,4] - unkn[2,3,5] + forg[2,3,5] >= 0
 R1431: unkn[0,3,4] - unkn[1,3,5] + forg[1,3,5] = 0
 R1432: - tbox[0,0,5] + forg[0,0,5] <= 0
 R1433: sbox[0,0,5] - rkey[0,0,5] + tkey[0,0,5] >= 0
 R1434: sbox[0,0,5] + rkey[0,0,5] - tkey[0,0,5] >= 0
 R1435: - sbox[0,0,5] + rkey[0,0,5] + tkey[0,0,5] >= 0
 R1436: - tbox[1,0,5] + forg[1,0,5] <= 0
 R1437: sbox[1,0,5] - rkey[1,0,5] + tkey[1,0,5] >= 0
 R1438: sbox[1,0,5] + rkey[1,0,5] - tkey[1,0,5] >= 0
 R1439: - sbox[1,0,5] + rkey[1,0,5] + tkey[1,0,5] >= 0
 R1440: - tbox[2,0,5] + forg[2,0,5] <= 0
 R1441: - sbox[2,0,5] + rkey[2,0,5] = 0
 R1442: - tbox[3,0,5] + forg[3,0,5] <= 0
 R1443: - sbox[3,0,5] + rkey[3,0,5] = 0
 R1444: - sbox[3,0,6] + rkey[0,0,5] + rkey[2,2,5] >= 0
 R1445: sbox[3,0,6] + rkey[0,0,5] - rkey[2,2,5] >= 0
 R1446: sbox[3,0,6] - rkey[0,0,5] + rkey[2,2,5] >= 0
 R1447: - sbox[0,0,6] + sbox[3,0,6] + rkey[3,1,5] >= 0
 R1448: sbox[0,0,6] + sbox[3,0,6] - rkey[3,1,5] >= 0
 R1449: sbox[0,0,6] - sbox[3,0,6] + rkey[3,1,5] >= 0
 R1450: - sbox[2,0,6] + rkey[1,3,5] + rkey[2,2,5] >= 0
 R1451: sbox[2,0,6] + rkey[1,3,5] - rkey[2,2,5] >= 0
 R1452: sbox[2,0,6] - rkey[1,3,5] + rkey[2,2,5] >= 0
 R1453: sbox[1,0,6] - rkey[0,0,5] = 0
 R1454: tbox[0,0,5] + tbox[2,2,5] - tbox[3,0,6] >= 0
 R1455: tbox[0,0,5] - tbox[2,2,5] + tbox[3,0,6] >= 0
 R1456: - tbox[0,0,5] + tbox[2,2,5] + tbox[3,0,6] >= 0
 R1457: - tbox[0,0,6] + tbox[3,0,6] + tbox[3,1,5] >= 0
 R1458: tbox[0,0,6] + tbox[3,0,6] - tbox[3,1,5] >= 0
 R1459: tbox[0,0,6] - tbox[3,0,6] + tbox[3,1,5] >= 0
 R1460: tbox[1,3,5] - tbox[2,0,6] + tbox[2,2,5] >= 0
 R1461: tbox[1,3,5] + tbox[2,0,6] - tbox[2,2,5] >= 0
 R1462: - tbox[1,3,5] + tbox[2,0,6] + tbox[2,2,5] >= 0
 R1463: - tbox[0,0,5] + tbox[1,0,6] = 0
 R1464: unkn[0,0,5] - unkn[3,0,6] + forg[3,0,6] <= 0
 R1465: unkn[2,2,5] - unkn[3,0,6] + forg[3,0,6] <= 0
 R1466: unkn[0,0,5] + unkn[2,2,5] - unkn[3,0,6] + forg[3,0,6] >= 0
 R1467: - unkn[0,0,6] + unkn[3,0,6] + forg[0,0,6] - forg[3,0,6] <= 0
 R1468: - unkn[0,0,6] + unkn[3,1,5] + forg[0,0,6] <= 0
 R1469: - unkn[0,0,6] + unkn[3,0,6] + unkn[3,1,5] + forg[0,0,6]
   - forg[3,0,6] >= 0
 R1470: unkn[1,3,5] - unkn[2,0,6] + forg[2,0,6] <= 0
 R1471: - unkn[2,0,6] + unkn[2,2,5] + forg[2,0,6] <= 0
 R1472: unkn[1,3,5] - unkn[2,0,6] + unkn[2,2,5] + forg[2,0,6] >= 0
 R1473: unkn[0,0,5] - unkn[1,0,6] + forg[1,0,6] = 0
 R1474: - tbox[0,1,5] + forg[0,1,5] <= 0
 R1475: sbox[0,1,5] - rkey[0,1,5] + tkey[0,1,5] >= 0
 R1476: sbox[0,1,5] + rkey[0,1,5] - tkey[0,1,5] >= 0
 R1477: - sbox[0,1,5] + rkey[0,1,5] + tkey[0,1,5] >= 0
 R1478: - tbox[1,1,5] + forg[1,1,5] <= 0
 R1479: sbox[1,1,5] - rkey[1,1,5] + tkey[1,1,5] >= 0
 R1480: sbox[1,1,5] + rkey[1,1,5] - tkey[1,1,5] >= 0
 R1481: - sbox[1,1,5] + rkey[1,1,5] + tkey[1,1,5] >= 0
 R1482: - tbox[2,1,5] + forg[2,1,5] <= 0
 R1483: - sbox[2,1,5] + rkey[2,1,5] = 0
 R1484: - tbox[3,1,5] + forg[3,1,5] <= 0
 R1485: - sbox[3,1,5] + rkey[3,1,5] = 0
 R1486: - sbox[3,1,6] + rkey[0,1,5] + rkey[2,3,5] >= 0
 R1487: sbox[3,1,6] + rkey[0,1,5] - rkey[2,3,5] >= 0
 R1488: sbox[3,1,6] - rkey[0,1,5] + rkey[2,3,5] >= 0
 R1489: - sbox[0,1,6] + sbox[3,1,6] + rkey[3,2,5] >= 0
 R1490: sbox[0,1,6] + sbox[3,1,6] - rkey[3,2,5] >= 0
 R1491: sbox[0,1,6] - sbox[3,1,6] + rkey[3,2,5] >= 0
 R1492: - sbox[2,1,6] + rkey[1,0,5] + rkey[2,3,5] >= 0
 R1493: sbox[2,1,6] + rkey[1,0,5] - rkey[2,3,5] >= 0
 R1494: sbox[2,1,6] - rkey[1,0,5] + rkey[2,3,5] >= 0
 R1495: sbox[1,1,6] - rkey[0,1,5] = 0
 R1496: tbox[0,1,5] + tbox[2,3,5] - tbox[3,1,6] >= 0
 R1497: tbox[0,1,5] - tbox[2,3,5] + tbox[3,1,6] >= 0
 R1498: - tbox[0,1,5] + tbox[2,3,5] + tbox[3,1,6] >= 0
 R1499: - tbox[0,1,6] + tbox[3,1,6] + tbox[3,2,5] >= 0
 R1500: tbox[0,1,6] + tbox[3,1,6] - tbox[3,2,5] >= 0
 R1501: tbox[0,1,6] - tbox[3,1,6] + tbox[3,2,5] >= 0
 R1502: tbox[1,0,5] - tbox[2,1,6] + tbox[2,3,5] >= 0
 R1503: tbox[1,0,5] + tbox[2,1,6] - tbox[2,3,5] >= 0
 R1504: - tbox[1,0,5] + tbox[2,1,6] + tbox[2,3,5] >= 0
 R1505: - tbox[0,1,5] + tbox[1,1,6] = 0
 R1506: unkn[0,1,5] - unkn[3,1,6] + forg[3,1,6] <= 0
 R1507: unkn[2,3,5] - unkn[3,1,6] + forg[3,1,6] <= 0
 R1508: unkn[0,1,5] + unkn[2,3,5] - unkn[3,1,6] + forg[3,1,6] >= 0
 R1509: - unkn[0,1,6] + unkn[3,1,6] + forg[0,1,6] - forg[3,1,6] <= 0
 R1510: - unkn[0,1,6] + unkn[3,2,5] + forg[0,1,6] <= 0
 R1511: - unkn[0,1,6] + unkn[3,1,6] + unkn[3,2,5] + forg[0,1,6]
   - forg[3,1,6] >= 0
 R1512: unkn[1,0,5] - unkn[2,1,6] + forg[2,1,6] <= 0
 R1513: - unkn[2,1,6] + unkn[2,3,5] + forg[2,1,6] <= 0
 R1514: unkn[1,0,5] - unkn[2,1,6] + unkn[2,3,5] + forg[2,1,6] >= 0
 R1515: unkn[0,1,5] - unkn[1,1,6] + forg[1,1,6] = 0
 R1516: - tbox[0,2,5] + forg[0,2,5] <= 0
 R1517: sbox[0,2,5] - rkey[0,2,5] + tkey[0,2,5] >= 0
 R1518: sbox[0,2,5] + rkey[0,2,5] - tkey[0,2,5] >= 0
 R1519: - sbox[0,2,5] + rkey[0,2,5] + tkey[0,2,5] >= 0
 R1520: - tbox[1,2,5] + forg[1,2,5] <= 0
 R1521: sbox[1,2,5] - rkey[1,2,5] + tkey[1,2,5] >= 0
 R1522: sbox[1,2,5] + rkey[1,2,5] - tkey[1,2,5] >= 0
 R1523: - sbox[1,2,5] + rkey[1,2,5] + tkey[1,2,5] >= 0
 R1524: - tbox[2,2,5] + forg[2,2,5] <= 0
 R1525: - sbox[2,2,5] + rkey[2,2,5] = 0
 R1526: - tbox[3,2,5] + forg[3,2,5] <= 0
 R1527: - sbox[3,2,5] + rkey[3,2,5] = 0
 R1528: - sbox[3,2,6] + rkey[0,2,5] + rkey[2,0,5] >= 0
 R1529: sbox[3,2,6] + rkey[0,2,5] - rkey[2,0,5] >= 0
 R1530: sbox[3,2,6] - rkey[0,2,5] + rkey[2,0,5] >= 0
 R1531: - sbox[0,2,6] + sbox[3,2,6] + rkey[3,3,5] >= 0
 R1532: sbox[0,2,6] + sbox[3,2,6] - rkey[3,3,5] >= 0
 R1533: sbox[0,2,6] - sbox[3,2,6] + rkey[3,3,5] >= 0
 R1534: - sbox[2,2,6] + rkey[1,1,5] + rkey[2,0,5] >= 0
 R1535: sbox[2,2,6] + rkey[1,1,5] - rkey[2,0,5] >= 0
 R1536: sbox[2,2,6] - rkey[1,1,5] + rkey[2,0,5] >= 0
 R1537: sbox[1,2,6] - rkey[0,2,5] = 0
 R1538: tbox[0,2,5] + tbox[2,0,5] - tbox[3,2,6] >= 0
 R1539: tbox[0,2,5] - tbox[2,0,5] + tbox[3,2,6] >= 0
 R1540: - tbox[0,2,5] + tbox[2,0,5] + tbox[3,2,6] >= 0
 R1541: - tbox[0,2,6] + tbox[3,2,6] + tbox[3,3,5] >= 0
 R1542: tbox[0,2,6] + tbox[3,2,6] - tbox[3,3,5] >= 0
 R1543: tbox[0,2,6] - tbox[3,2,6] + tbox[3,3,5] >= 0
 R1544: tbox[1,1,5] + tbox[2,0,5] - tbox[2,2,6] >= 0
 R1545: tbox[1,1,5] - tbox[2,0,5] + tbox[2,2,6] >= 0
 R1546: - tbox[1,1,5] + tbox[2,0,5] + tbox[2,2,6] >= 0
 R1547: - tbox[0,2,5] + tbox[1,2,6] = 0
 R1548: unkn[0,2,5] - unkn[3,2,6] + forg[3,2,6] <= 0
 R1549: unkn[2,0,5] - unkn[3,2,6] + forg[3,2,6] <= 0
 R1550: unkn[0,2,5] + unkn[2,0,5] - unkn[3,2,6] + forg[3,2,6] >= 0
 R1551: - unkn[0,2,6] + unkn[3,2,6] + forg[0,2,6] - forg[3,2,6] <= 0
 R1552: - unkn[0,2,6] + unkn[3,3,5] + forg[0,2,6] <= 0
 R1553: - unkn[0,2,6] + unkn[3,2,6] + unkn[3,3,5] + forg[0,2,6]
   - forg[3,2,6] >= 0
 R1554: unkn[1,1,5] - unkn[2,2,6] + forg[2,2,6] <= 0
 R1555: unkn[2,0,5] - unkn[2,2,6] + forg[2,2,6] <= 0
 R1556: unkn[1,1,5] + unkn[2,0,5] - unkn[2,2,6] + forg[2,2,6] >= 0
 R1557: unkn[0,2,5] - unkn[1,2,6] + forg[1,2,6] = 0
 R1558: - tbox[0,3,5] + forg[0,3,5] <= 0
 R1559: sbox[0,3,5] - rkey[0,3,5] + tkey[0,3,5] >= 0
 R1560: sbox[0,3,5] + rkey[0,3,5] - tkey[0,3,5] >= 0
 R1561: - sbox[0,3,5] + rkey[0,3,5] + tkey[0,3,5] >= 0
 R1562: - tbox[1,3,5] + forg[1,3,5] <= 0
 R1563: sbox[1,3,5] - rkey[1,3,5] + tkey[1,3,5] >= 0
 R1564: sbox[1,3,5] + rkey[1,3,5] - tkey[1,3,5] >= 0
 R1565: - sbox[1,3,5] + rkey[1,3,5] + tkey[1,3,5] >= 0
 R1566: - tbox[2,3,5] + forg[2,3,5] <= 0
 R1567: - sbox[2,3,5] + rkey[2,3,5] = 0
 R1568: - tbox[3,3,5] + forg[3,3,5] <= 0
 R1569: - sbox[3,3,5] + rkey[3,3,5] = 0
 R1570: - sbox[3,3,6] + rkey[0,3,5] + rkey[2,1,5] >= 0
 R1571: sbox[3,3,6] + rkey[0,3,5] - rkey[2,1,5] >= 0
 R1572: sbox[3,3,6] - rkey[0,3,5] + rkey[2,1,5] >= 0
 R1573: - sbox[0,3,6] + sbox[3,3,6] + rkey[3,0,5] >= 0
 R1574: sbox[0,3,6] + sbox[3,3,6] - rkey[3,0,5] >= 0
 R1575: sbox[0,3,6] - sbox[3,3,6] + rkey[3,0,5] >= 0
 R1576: - sbox[2,3,6] + rkey[1,2,5] + rkey[2,1,5] >= 0
 R1577: sbox[2,3,6] + rkey[1,2,5] - rkey[2,1,5] >= 0
 R1578: sbox[2,3,6] - rkey[1,2,5] + rkey[2,1,5] >= 0
 R1579: sbox[1,3,6] - rkey[0,3,5] = 0
 R1580: tbox[0,3,5] + tbox[2,1,5] - tbox[3,3,6] >= 0
 R1581: tbox[0,3,5] - tbox[2,1,5] + tbox[3,3,6] >= 0
 R1582: - tbox[0,3,5] + tbox[2,1,5] + tbox[3,3,6] >= 0
 R1583: - tbox[0,3,6] + tbox[3,0,5] + tbox[3,3,6] >= 0
 R1584: tbox[0,3,6] - tbox[3,0,5] + tbox[3,3,6] >= 0
 R1585: tbox[0,3,6] + tbox[3,0,5] - tbox[3,3,6] >= 0
 R1586: tbox[1,2,5] + tbox[2,1,5] - tbox[2,3,6] >= 0
 R1587: tbox[1,2,5] - tbox[2,1,5] + tbox[2,3,6] >= 0
 R1588: - tbox[1,2,5] + tbox[2,1,5] + tbox[2,3,6] >= 0
 R1589: - tbox[0,3,5] + tbox[1,3,6] = 0
 R1590: unkn[0,3,5] - unkn[3,3,6] + forg[3,3,6] <= 0
 R1591: unkn[2,1,5] - unkn[3,3,6] + forg[3,3,6] <= 0
 R1592: unkn[0,3,5] + unkn[2,1,5] - unkn[3,3,6] + forg[3,3,6] >= 0
 R1593: - unkn[0,3,6] + unkn[3,3,6] + forg[0,3,6] - forg[3,3,6] <= 0
 R1594: - unkn[0,3,6] + unkn[3,0,5] + forg[0,3,6] <= 0
 R1595: - unkn[0,3,6] + unkn[3,0,5] + unkn[3,3,6] + forg[0,3,6]
   - forg[3,3,6] >= 0
 R1596: unkn[1,2,5] - unkn[2,3,6] + forg[2,3,6] <= 0
 R1597: unkn[2,1,5] - unkn[2,3,6] + forg[2,3,6] <= 0
 R1598: unkn[1,2,5] + unkn[2,1,5] - unkn[2,3,6] + forg[2,3,6] >= 0
 R1599: unkn[0,3,5] - unkn[1,3,6] + forg[1,3,6] = 0
 R1600: - tbox[0,0,6] + forg[0,0,6] <= 0
 R1601: sbox[0,0,6] - rkey[0,0,6] + tkey[0,0,6] >= 0
 R1602: sbox[0,0,6] + rkey[0,0,6] - tkey[0,0,6] >= 0
 R1603: - sbox[0,0,6] + rkey[0,0,6] + tkey[0,0,6] >= 0
 R1604: - tbox[1,0,6] + forg[1,0,6] <= 0
 R1605: sbox[1,0,6] - rkey[1,0,6] + tkey[1,0,6] >= 0
 R1606: sbox[1,0,6] + rkey[1,0,6] - tkey[1,0,6] >= 0
 R1607: - sbox[1,0,6] + rkey[1,0,6] + tkey[1,0,6] >= 0
 R1608: - tbox[2,0,6] + forg[2,0,6] <= 0
 R1609: - sbox[2,0,6] + rkey[2,0,6] = 0
 R1610: - tbox[3,0,6] + forg[3,0,6] <= 0
 R1611: - sbox[3,0,6] + rkey[3,0,6] = 0
 R1612: - sbox[3,0,7] + rkey[0,0,6] + rkey[2,2,6] >= 0
 R1613: sbox[3,0,7] + rkey[0,0,6] - rkey[2,2,6] >= 0
 R1614: sbox[3,0,7] - rkey[0,0,6] + rkey[2,2,6] >= 0
 R1615: - sbox[0,0,7] + sbox[3,0,7] + rkey[3,1,6] >= 0
 R1616: sbox[0,0,7] + sbox[3,0,7] - rkey[3,1,6] >= 0
 R1617: sbox[0,0,7] - sbox[3,0,7] + rkey[3,1,6] >= 0
 R1618: - sbox[2,0,7] + rkey[1,3,6] + rkey[2,2,6] >= 0
 R1619: sbox[2,0,7] + rkey[1,3,6] - rkey[2,2,6] >= 0
 R1620: sbox[2,0,7] - rkey[1,3,6] + rkey[2,2,6] >= 0
 R1621: sbox[1,0,7] - rkey[0,0,6] = 0
 R1622: tbox[0,0,6] + tbox[2,2,6] - tbox[3,0,7] >= 0
 R1623: tbox[0,0,6] - tbox[2,2,6] + tbox[3,0,7] >= 0
 R1624: - tbox[0,0,6] + tbox[2,2,6] + tbox[3,0,7] >= 0
 R1625: - tbox[0,0,7] + tbox[3,0,7] + tbox[3,1,6] >= 0
 R1626: tbox[0,0,7] + tbox[3,0,7] - tbox[3,1,6] >= 0
 R1627: tbox[0,0,7] - tbox[3,0,7] + tbox[3,1,6] >= 0
 R1628: tbox[1,3,6] - tbox[2,0,7] + tbox[2,2,6] >= 0
 R1629: tbox[1,3,6] + tbox[2,0,7] - tbox[2,2,6] >= 0
 R1630: - tbox[1,3,6] + tbox[2,0,7] + tbox[2,2,6] >= 0
 R1631: - tbox[0,0,6] + tbox[1,0,7] = 0
 R1632: unkn[0,0,6] - unkn[3,0,7] + forg[3,0,7] <= 0
 R1633: unkn[2,2,6] - unkn[3,0,7] + forg[3,0,7] <= 0
 R1634: unkn[0,0,6] + unkn[2,2,6] - unkn[3,0,7] + forg[3,0,7] >= 0
 R1635: - unkn[0,0,7] + unkn[3,0,7] + forg[0,0,7] - forg[3,0,7] <= 0
 R1636: - unkn[0,0,7] + unkn[3,1,6] + forg[0,0,7] <= 0
 R1637: - unkn[0,0,7] + unkn[3,0,7] + unkn[3,1,6] + forg[0,0,7]
   - forg[3,0,7] >= 0
 R1638: unkn[1,3,6] - unkn[2,0,7] + forg[2,0,7] <= 0
 R1639: - unkn[2,0,7] + unkn[2,2,6] + forg[2,0,7] <= 0
 R1640: unkn[1,3,6] - unkn[2,0,7] + unkn[2,2,6] + forg[2,0,7] >= 0
 R1641: unkn[0,0,6] - unkn[1,0,7] + forg[1,0,7] = 0
 R1642: - tbox[0,1,6] + forg[0,1,6] <= 0
 R1643: sbox[0,1,6] - rkey[0,1,6] + tkey[0,1,6] >= 0
 R1644: sbox[0,1,6] + rkey[0,1,6] - tkey[0,1,6] >= 0
 R1645: - sbox[0,1,6] + rkey[0,1,6] + tkey[0,1,6] >= 0
 R1646: - tbox[1,1,6] + forg[1,1,6] <= 0
 R1647: sbox[1,1,6] - rkey[1,1,6] + tkey[1,1,6] >= 0
 R1648: sbox[1,1,6] + rkey[1,1,6] - tkey[1,1,6] >= 0
 R1649: - sbox[1,1,6] + rkey[1,1,6] + tkey[1,1,6] >= 0
 R1650: - tbox[2,1,6] + forg[2,1,6] <= 0
 R1651: - sbox[2,1,6] + rkey[2,1,6] = 0
 R1652: - tbox[3,1,6] + forg[3,1,6] <= 0
 R1653: - sbox[3,1,6] + rkey[3,1,6] = 0
 R1654: - sbox[3,1,7] + rkey[0,1,6] + rkey[2,3,6] >= 0
 R1655: sbox[3,1,7] + rkey[0,1,6] - rkey[2,3,6] >= 0
 R1656: sbox[3,1,7] - rkey[0,1,6] + rkey[2,3,6] >= 0
 R1657: - sbox[0,1,7] + sbox[3,1,7] + rkey[3,2,6] >= 0
 R1658: sbox[0,1,7] + sbox[3,1,7] - rkey[3,2,6] >= 0
 R1659: sbox[0,1,7] - sbox[3,1,7] + rkey[3,2,6] >= 0
 R1660: - sbox[2,1,7] + rkey[1,0,6] + rkey[2,3,6] >= 0
 R1661: sbox[2,1,7] + rkey[1,0,6] - rkey[2,3,6] >= 0
 R1662: sbox[2,1,7] - rkey[1,0,6] + rkey[2,3,6] >= 0
 R1663: sbox[1,1,7] - rkey[0,1,6] = 0
 R1664: tbox[0,1,6] + tbox[2,3,6] - tbox[3,1,7] >= 0
 R1665: tbox[0,1,6] - tbox[2,3,6] + tbox[3,1,7] >= 0
 R1666: - tbox[0,1,6] + tbox[2,3,6] + tbox[3,1,7] >= 0
 R1667: - tbox[0,1,7] + tbox[3,1,7] + tbox[3,2,6] >= 0
 R1668: tbox[0,1,7] + tbox[3,1,7] - tbox[3,2,6] >= 0
 R1669: tbox[0,1,7] - tbox[3,1,7] + tbox[3,2,6] >= 0
 R1670: tbox[1,0,6] - tbox[2,1,7] + tbox[2,3,6] >= 0
 R1671: tbox[1,0,6] + tbox[2,1,7] - tbox[2,3,6] >= 0
 R1672: - tbox[1,0,6] + tbox[2,1,7] + tbox[2,3,6] >= 0
 R1673: - tbox[0,1,6] + tbox[1,1,7] = 0
 R1674: unkn[0,1,6] - unkn[3,1,7] + forg[3,1,7] <= 0
 R1675: unkn[2,3,6] - unkn[3,1,7] + forg[3,1,7] <= 0
 R1676: unkn[0,1,6] + unkn[2,3,6] - unkn[3,1,7] + forg[3,1,7] >= 0
 R1677: - unkn[0,1,7] + unkn[3,1,7] + forg[0,1,7] - forg[3,1,7] <= 0
 R1678: - unkn[0,1,7] + unkn[3,2,6] + forg[0,1,7] <= 0
 R1679: - unkn[0,1,7] + unkn[3,1,7] + unkn[3,2,6] + forg[0,1,7]
   - forg[3,1,7] >= 0
 R1680: unkn[1,0,6] - unkn[2,1,7] + forg[2,1,7] <= 0
 R1681: - unkn[2,1,7] + unkn[2,3,6] + forg[2,1,7] <= 0
 R1682: unkn[1,0,6] - unkn[2,1,7] + unkn[2,3,6] + forg[2,1,7] >= 0
 R1683: unkn[0,1,6] - unkn[1,1,7] + forg[1,1,7] = 0
 R1684: - tbox[0,2,6] + forg[0,2,6] <= 0
 R1685: sbox[0,2,6] - rkey[0,2,6] + tkey[0,2,6] >= 0
 R1686: sbox[0,2,6] + rkey[0,2,6] - tkey[0,2,6] >= 0
 R1687: - sbox[0,2,6] + rkey[0,2,6] + tkey[0,2,6] >= 0
 R1688: - tbox[1,2,6] + forg[1,2,6] <= 0
 R1689: sbox[1,2,6] - rkey[1,2,6] + tkey[1,2,6] >= 0
 R1690: sbox[1,2,6] + rkey[1,2,6] - tkey[1,2,6] >= 0
 R1691: - sbox[1,2,6] + rkey[1,2,6] + tkey[1,2,6] >= 0
 R1692: - tbox[2,2,6] + forg[2,2,6] <= 0
 R1693: - sbox[2,2,6] + rkey[2,2,6] = 0
 R1694: - tbox[3,2,6] + forg[3,2,6] <= 0
 R1695: - sbox[3,2,6] + rkey[3,2,6] = 0
 R1696: - sbox[3,2,7] + rkey[0,2,6] + rkey[2,0,6] >= 0
 R1697: sbox[3,2,7] + rkey[0,2,6] - rkey[2,0,6] >= 0
 R1698: sbox[3,2,7] - rkey[0,2,6] + rkey[2,0,6] >= 0
 R1699: - sbox[0,2,7] + sbox[3,2,7] + rkey[3,3,6] >= 0
 R1700: sbox[0,2,7] + sbox[3,2,7] - rkey[3,3,6] >= 0
 R1701: sbox[0,2,7] - sbox[3,2,7] + rkey[3,3,6] >= 0
 R1702: - sbox[2,2,7] + rkey[1,1,6] + rkey[2,0,6] >= 0
 R1703: sbox[2,2,7] + rkey[1,1,6] - rkey[2,0,6] >= 0
 R1704: sbox[2,2,7] - rkey[1,1,6] + rkey[2,0,6] >= 0
 R1705: sbox[1,2,7] - rkey[0,2,6] = 0
 R1706: tbox[0,2,6] + tbox[2,0,6] - tbox[3,2,7] >= 0
 R1707: tbox[0,2,6] - tbox[2,0,6] + tbox[3,2,7] >= 0
 R1708: - tbox[0,2,6] + tbox[2,0,6] + tbox[3,2,7] >= 0
 R1709: - tbox[0,2,7] + tbox[3,2,7] + tbox[3,3,6] >= 0
 R1710: tbox[0,2,7] + tbox[3,2,7] - tbox[3,3,6] >= 0
 R1711: tbox[0,2,7] - tbox[3,2,7] + tbox[3,3,6] >= 0
 R1712: tbox[1,1,6] + tbox[2,0,6] - tbox[2,2,7] >= 0
 R1713: tbox[1,1,6] - tbox[2,0,6] + tbox[2,2,7] >= 0
 R1714: - tbox[1,1,6] + tbox[2,0,6] + tbox[2,2,7] >= 0
 R1715: - tbox[0,2,6] + tbox[1,2,7] = 0
 R1716: unkn[0,2,6] - unkn[3,2,7] + forg[3,2,7] <= 0
 R1717: unkn[2,0,6] - unkn[3,2,7] + forg[3,2,7] <= 0
 R1718: unkn[0,2,6] + unkn[2,0,6] - unkn[3,2,7] + forg[3,2,7] >= 0
 R1719: - unkn[0,2,7] + unkn[3,2,7] + forg[0,2,7] - forg[3,2,7] <= 0
 R1720: - unkn[0,2,7] + unkn[3,3,6] + forg[0,2,7] <= 0
 R1721: - unkn[0,2,7] + unkn[3,2,7] + unkn[3,3,6] + forg[0,2,7]
   - forg[3,2,7] >= 0
 R1722: unkn[1,1,6] - unkn[2,2,7] + forg[2,2,7] <= 0
 R1723: unkn[2,0,6] - unkn[2,2,7] + forg[2,2,7] <= 0
 R1724: unkn[1,1,6] + unkn[2,0,6] - unkn[2,2,7] + forg[2,2,7] >= 0
 R1725: unkn[0,2,6] - unkn[1,2,7] + forg[1,2,7] = 0
 R1726: - tbox[0,3,6] + forg[0,3,6] <= 0
 R1727: sbox[0,3,6] - rkey[0,3,6] + tkey[0,3,6] >= 0
 R1728: sbox[0,3,6] + rkey[0,3,6] - tkey[0,3,6] >= 0
 R1729: - sbox[0,3,6] + rkey[0,3,6] + tkey[0,3,6] >= 0
 R1730: - tbox[1,3,6] + forg[1,3,6] <= 0
 R1731: sbox[1,3,6] - rkey[1,3,6] + tkey[1,3,6] >= 0
 R1732: sbox[1,3,6] + rkey[1,3,6] - tkey[1,3,6] >= 0
 R1733: - sbox[1,3,6] + rkey[1,3,6] + tkey[1,3,6] >= 0
 R1734: - tbox[2,3,6] + forg[2,3,6] <= 0
 R1735: - sbox[2,3,6] + rkey[2,3,6] = 0
 R1736: - tbox[3,3,6] + forg[3,3,6] <= 0
 R1737: - sbox[3,3,6] + rkey[3,3,6] = 0
 R1738: - sbox[3,3,7] + rkey[0,3,6] + rkey[2,1,6] >= 0
 R1739: sbox[3,3,7] + rkey[0,3,6] - rkey[2,1,6] >= 0
 R1740: sbox[3,3,7] - rkey[0,3,6] + rkey[2,1,6] >= 0
 R1741: - sbox[0,3,7] + sbox[3,3,7] + rkey[3,0,6] >= 0
 R1742: sbox[0,3,7] + sbox[3,3,7] - rkey[3,0,6] >= 0
 R1743: sbox[0,3,7] - sbox[3,3,7] + rkey[3,0,6] >= 0
 R1744: - sbox[2,3,7] + rkey[1,2,6] + rkey[2,1,6] >= 0
 R1745: sbox[2,3,7] + rkey[1,2,6] - rkey[2,1,6] >= 0
 R1746: sbox[2,3,7] - rkey[1,2,6] + rkey[2,1,6] >= 0
 R1747: sbox[1,3,7] - rkey[0,3,6] = 0
 R1748: tbox[0,3,6] + tbox[2,1,6] - tbox[3,3,7] >= 0
 R1749: tbox[0,3,6] - tbox[2,1,6] + tbox[3,3,7] >= 0
 R1750: - tbox[0,3,6] + tbox[2,1,6] + tbox[3,3,7] >= 0
 R1751: - tbox[0,3,7] + tbox[3,0,6] + tbox[3,3,7] >= 0
 R1752: tbox[0,3,7] - tbox[3,0,6] + tbox[3,3,7] >= 0
 R1753: tbox[0,3,7] + tbox[3,0,6] - tbox[3,3,7] >= 0
 R1754: tbox[1,2,6] + tbox[2,1,6] - tbox[2,3,7] >= 0
 R1755: tbox[1,2,6] - tbox[2,1,6] + tbox[2,3,7] >= 0
 R1756: - tbox[1,2,6] + tbox[2,1,6] + tbox[2,3,7] >= 0
 R1757: - tbox[0,3,6] + tbox[1,3,7] = 0
 R1758: unkn[0,3,6] - unkn[3,3,7] + forg[3,3,7] <= 0
 R1759: unkn[2,1,6] - unkn[3,3,7] + forg[3,3,7] <= 0
 R1760: unkn[0,3,6] + unkn[2,1,6] - unkn[3,3,7] + forg[3,3,7] >= 0
 R1761: - unkn[0,3,7] + unkn[3,3,7] + forg[0,3,7] - forg[3,3,7] <= 0
 R1762: - unkn[0,3,7] + unkn[3,0,6] + forg[0,3,7] <= 0
 R1763: - unkn[0,3,7] + unkn[3,0,6] + unkn[3,3,7] + forg[0,3,7]
   - forg[3,3,7] >= 0
 R1764: unkn[1,2,6] - unkn[2,3,7] + forg[2,3,7] <= 0
 R1765: unkn[2,1,6] - unkn[2,3,7] + forg[2,3,7] <= 0
 R1766: unkn[1,2,6] + unkn[2,1,6] - unkn[2,3,7] + forg[2,3,7] >= 0
 R1767: unkn[0,3,6] - unkn[1,3,7] + forg[1,3,7] = 0
 R1768: - tbox[0,0,7] + forg[0,0,7] <= 0
 R1769: sbox[0,0,7] - rkey[0,0,7] + tkey[0,0,7] >= 0
 R1770: sbox[0,0,7] + rkey[0,0,7] - tkey[0,0,7] >= 0
 R1771: - sbox[0,0,7] + rkey[0,0,7] + tkey[0,0,7] >= 0
 R1772: - tbox[1,0,7] + forg[1,0,7] <= 0
 R1773: sbox[1,0,7] - rkey[1,0,7] + tkey[1,0,7] >= 0
 R1774: sbox[1,0,7] + rkey[1,0,7] - tkey[1,0,7] >= 0
 R1775: - sbox[1,0,7] + rkey[1,0,7] + tkey[1,0,7] >= 0
 R1776: - tbox[2,0,7] + forg[2,0,7] <= 0
 R1777: - sbox[2,0,7] + rkey[2,0,7] = 0
 R1778: - tbox[3,0,7] + forg[3,0,7] <= 0
 R1779: - sbox[3,0,7] + rkey[3,0,7] = 0
 R1780: - sbox[3,0,8] + rkey[0,0,7] + rkey[2,2,7] >= 0
 R1781: sbox[3,0,8] + rkey[0,0,7] - rkey[2,2,7] >= 0
 R1782: sbox[3,0,8] - rkey[0,0,7] + rkey[2,2,7] >= 0
 R1783: - sbox[0,0,8] + sbox[3,0,8] + rkey[3,1,7] >= 0
 R1784: sbox[0,0,8] + sbox[3,0,8] - rkey[3,1,7] >= 0
 R1785: sbox[0,0,8] - sbox[3,0,8] + rkey[3,1,7] >= 0
 R1786: - sbox[2,0,8] + rkey[1,3,7] + rkey[2,2,7] >= 0
 R1787: sbox[2,0,8] + rkey[1,3,7] - rkey[2,2,7] >= 0
 R1788: sbox[2,0,8] - rkey[1,3,7] + rkey[2,2,7] >= 0
 R1789: sbox[1,0,8] - rkey[0,0,7] = 0
 R1790: tbox[0,0,7] + tbox[2,2,7] - tbox[3,0,8] >= 0
 R1791: tbox[0,0,7] - tbox[2,2,7] + tbox[3,0,8] >= 0
 R1792: - tbox[0,0,7] + tbox[2,2,7] + tbox[3,0,8] >= 0
 R1793: - tbox[0,0,8] + tbox[3,0,8] + tbox[3,1,7] >= 0
 R1794: tbox[0,0,8] + tbox[3,0,8] - tbox[3,1,7] >= 0
 R1795: tbox[0,0,8] - tbox[3,0,8] + tbox[3,1,7] >= 0
 R1796: tbox[1,3,7] - tbox[2,0,8] + tbox[2,2,7] >= 0
 R1797: tbox[1,3,7] + tbox[2,0,8] - tbox[2,2,7] >= 0
 R1798: - tbox[1,3,7] + tbox[2,0,8] + tbox[2,2,7] >= 0
 R1799: - tbox[0,0,7] + tbox[1,0,8] = 0
 R1800: unkn[0,0,7] - unkn[3,0,8] + forg[3,0,8] <= 0
 R1801: unkn[2,2,7] - unkn[3,0,8] + forg[3,0,8] <= 0
 R1802: unkn[0,0,7] + unkn[2,2,7] - unkn[3,0,8] + forg[3,0,8] >= 0
 R1803: - unkn[0,0,8] + unkn[3,0,8] + forg[0,0,8] - forg[3,0,8] <= 0
 R1804: - unkn[0,0,8] + unkn[3,1,7] + forg[0,0,8] <= 0
 R1805: - unkn[0,0,8] + unkn[3,0,8] + unkn[3,1,7] + forg[0,0,8]
   - forg[3,0,8] >= 0
 R1806: unkn[1,3,7] - unkn[2,0,8] + forg[2,0,8] <= 0
 R1807: - unkn[2,0,8] + unkn[2,2,7] + forg[2,0,8] <= 0
 R1808: unkn[1,3,7] - unkn[2,0,8] + unkn[2,2,7] + forg[2,0,8] >= 0
 R1809: unkn[0,0,7] - unkn[1,0,8] + forg[1,0,8] = 0
 R1810: - tbox[0,1,7] + forg[0,1,7] <= 0
 R1811: sbox[0,1,7] - rkey[0,1,7] + tkey[0,1,7] >= 0
 R1812: sbox[0,1,7] + rkey[0,1,7] - tkey[0,1,7] >= 0
 R1813: - sbox[0,1,7] + rkey[0,1,7] + tkey[0,1,7] >= 0
 R1814: - tbox[1,1,7] + forg[1,1,7] <= 0
 R1815: sbox[1,1,7] - rkey[1,1,7] + tkey[1,1,7] >= 0
 R1816: sbox[1,1,7] + rkey[1,1,7] - tkey[1,1,7] >= 0
 R1817: - sbox[1,1,7] + rkey[1,1,7] + tkey[1,1,7] >= 0
 R1818: - tbox[2,1,7] + forg[2,1,7] <= 0
 R1819: - sbox[2,1,7] + rkey[2,1,7] = 0
 R1820: - tbox[3,1,7] + forg[3,1,7] <= 0
 R1821: - sbox[3,1,7] + rkey[3,1,7] = 0
 R1822: - sbox[3,1,8] + rkey[0,1,7] + rkey[2,3,7] >= 0
 R1823: sbox[3,1,8] + rkey[0,1,7] - rkey[2,3,7] >= 0
 R1824: sbox[3,1,8] - rkey[0,1,7] + rkey[2,3,7] >= 0
 R1825: - sbox[0,1,8] + sbox[3,1,8] + rkey[3,2,7] >= 0
 R1826: sbox[0,1,8] + sbox[3,1,8] - rkey[3,2,7] >= 0
 R1827: sbox[0,1,8] - sbox[3,1,8] + rkey[3,2,7] >= 0
 R1828: - sbox[2,1,8] + rkey[1,0,7] + rkey[2,3,7] >= 0
 R1829: sbox[2,1,8] + rkey[1,0,7] - rkey[2,3,7] >= 0
 R1830: sbox[2,1,8] - rkey[1,0,7] + rkey[2,3,7] >= 0
 R1831: sbox[1,1,8] - rkey[0,1,7] = 0
 R1832: tbox[0,1,7] + tbox[2,3,7] - tbox[3,1,8] >= 0
 R1833: tbox[0,1,7] - tbox[2,3,7] + tbox[3,1,8] >= 0
 R1834: - tbox[0,1,7] + tbox[2,3,7] + tbox[3,1,8] >= 0
 R1835: - tbox[0,1,8] + tbox[3,1,8] + tbox[3,2,7] >= 0
 R1836: tbox[0,1,8] + tbox[3,1,8] - tbox[3,2,7] >= 0
 R1837: tbox[0,1,8] - tbox[3,1,8] + tbox[3,2,7] >= 0
 R1838: tbox[1,0,7] - tbox[2,1,8] + tbox[2,3,7] >= 0
 R1839: tbox[1,0,7] + tbox[2,1,8] - tbox[2,3,7] >= 0
 R1840: - tbox[1,0,7] + tbox[2,1,8] + tbox[2,3,7] >= 0
 R1841: - tbox[0,1,7] + tbox[1,1,8] = 0
 R1842: unkn[0,1,7] - unkn[3,1,8] + forg[3,1,8] <= 0
 R1843: unkn[2,3,7] - unkn[3,1,8] + forg[3,1,8] <= 0
 R1844: unkn[0,1,7] + unkn[2,3,7] - unkn[3,1,8] + forg[3,1,8] >= 0
 R1845: - unkn[0,1,8] + unkn[3,1,8] + forg[0,1,8] - forg[3,1,8] <= 0
 R1846: - unkn[0,1,8] + unkn[3,2,7] + forg[0,1,8] <= 0
 R1847: - unkn[0,1,8] + unkn[3,1,8] + unkn[3,2,7] + forg[0,1,8]
   - forg[3,1,8] >= 0
 R1848: unkn[1,0,7] - unkn[2,1,8] + forg[2,1,8] <= 0
 R1849: - unkn[2,1,8] + unkn[2,3,7] + forg[2,1,8] <= 0
 R1850: unkn[1,0,7] - unkn[2,1,8] + unkn[2,3,7] + forg[2,1,8] >= 0
 R1851: unkn[0,1,7] - unkn[1,1,8] + forg[1,1,8] = 0
 R1852: - tbox[0,2,7] + forg[0,2,7] <= 0
 R1853: sbox[0,2,7] - rkey[0,2,7] + tkey[0,2,7] >= 0
 R1854: sbox[0,2,7] + rkey[0,2,7] - tkey[0,2,7] >= 0
 R1855: - sbox[0,2,7] + rkey[0,2,7] + tkey[0,2,7] >= 0
 R1856: - tbox[1,2,7] + forg[1,2,7] <= 0
 R1857: sbox[1,2,7] - rkey[1,2,7] + tkey[1,2,7] >= 0
 R1858: sbox[1,2,7] + rkey[1,2,7] - tkey[1,2,7] >= 0
 R1859: - sbox[1,2,7] + rkey[1,2,7] + tkey[1,2,7] >= 0
 R1860: - tbox[2,2,7] + forg[2,2,7] <= 0
 R1861: - sbox[2,2,7] + rkey[2,2,7] = 0
 R1862: - tbox[3,2,7] + forg[3,2,7] <= 0
 R1863: - sbox[3,2,7] + rkey[3,2,7] = 0
 R1864: - sbox[3,2,8] + rkey[0,2,7] + rkey[2,0,7] >= 0
 R1865: sbox[3,2,8] + rkey[0,2,7] - rkey[2,0,7] >= 0
 R1866: sbox[3,2,8] - rkey[0,2,7] + rkey[2,0,7] >= 0
 R1867: - sbox[0,2,8] + sbox[3,2,8] + rkey[3,3,7] >= 0
 R1868: sbox[0,2,8] + sbox[3,2,8] - rkey[3,3,7] >= 0
 R1869: sbox[0,2,8] - sbox[3,2,8] + rkey[3,3,7] >= 0
 R1870: - sbox[2,2,8] + rkey[1,1,7] + rkey[2,0,7] >= 0
 R1871: sbox[2,2,8] + rkey[1,1,7] - rkey[2,0,7] >= 0
 R1872: sbox[2,2,8] - rkey[1,1,7] + rkey[2,0,7] >= 0
 R1873: sbox[1,2,8] - rkey[0,2,7] = 0
 R1874: tbox[0,2,7] + tbox[2,0,7] - tbox[3,2,8] >= 0
 R1875: tbox[0,2,7] - tbox[2,0,7] + tbox[3,2,8] >= 0
 R1876: - tbox[0,2,7] + tbox[2,0,7] + tbox[3,2,8] >= 0
 R1877: - tbox[0,2,8] + tbox[3,2,8] + tbox[3,3,7] >= 0
 R1878: tbox[0,2,8] + tbox[3,2,8] - tbox[3,3,7] >= 0
 R1879: tbox[0,2,8] - tbox[3,2,8] + tbox[3,3,7] >= 0
 R1880: tbox[1,1,7] + tbox[2,0,7] - tbox[2,2,8] >= 0
 R1881: tbox[1,1,7] - tbox[2,0,7] + tbox[2,2,8] >= 0
 R1882: - tbox[1,1,7] + tbox[2,0,7] + tbox[2,2,8] >= 0
 R1883: - tbox[0,2,7] + tbox[1,2,8] = 0
 R1884: unkn[0,2,7] - unkn[3,2,8] + forg[3,2,8] <= 0
 R1885: unkn[2,0,7] - unkn[3,2,8] + forg[3,2,8] <= 0
 R1886: unkn[0,2,7] + unkn[2,0,7] - unkn[3,2,8] + forg[3,2,8] >= 0
 R1887: - unkn[0,2,8] + unkn[3,2,8] + forg[0,2,8] - forg[3,2,8] <= 0
 R1888: - unkn[0,2,8] + unkn[3,3,7] + forg[0,2,8] <= 0
 R1889: - unkn[0,2,8] + unkn[3,2,8] + unkn[3,3,7] + forg[0,2,8]
   - forg[3,2,8] >= 0
 R1890: unkn[1,1,7] - unkn[2,2,8] + forg[2,2,8] <= 0
 R1891: unkn[2,0,7] - unkn[2,2,8] + forg[2,2,8] <= 0
 R1892: unkn[1,1,7] + unkn[2,0,7] - unkn[2,2,8] + forg[2,2,8] >= 0
 R1893: unkn[0,2,7] - unkn[1,2,8] + forg[1,2,8] = 0
 R1894: - tbox[0,3,7] + forg[0,3,7] <= 0
 R1895: sbox[0,3,7] - rkey[0,3,7] + tkey[0,3,7] >= 0
 R1896: sbox[0,3,7] + rkey[0,3,7] - tkey[0,3,7] >= 0
 R1897: - sbox[0,3,7] + rkey[0,3,7] + tkey[0,3,7] >= 0
 R1898: - tbox[1,3,7] + forg[1,3,7] <= 0
 R1899: sbox[1,3,7] - rkey[1,3,7] + tkey[1,3,7] >= 0
 R1900: sbox[1,3,7] + rkey[1,3,7] - tkey[1,3,7] >= 0
 R1901: - sbox[1,3,7] + rkey[1,3,7] + tkey[1,3,7] >= 0
 R1902: - tbox[2,3,7] + forg[2,3,7] <= 0
 R1903: - sbox[2,3,7] + rkey[2,3,7] = 0
 R1904: - tbox[3,3,7] + forg[3,3,7] <= 0
 R1905: - sbox[3,3,7] + rkey[3,3,7] = 0
 R1906: - sbox[3,3,8] + rkey[0,3,7] + rkey[2,1,7] >= 0
 R1907: sbox[3,3,8] + rkey[0,3,7] - rkey[2,1,7] >= 0
 R1908: sbox[3,3,8] - rkey[0,3,7] + rkey[2,1,7] >= 0
 R1909: - sbox[0,3,8] + sbox[3,3,8] + rkey[3,0,7] >= 0
 R1910: sbox[0,3,8] + sbox[3,3,8] - rkey[3,0,7] >= 0
 R1911: sbox[0,3,8] - sbox[3,3,8] + rkey[3,0,7] >= 0
 R1912: - sbox[2,3,8] + rkey[1,2,7] + rkey[2,1,7] >= 0
 R1913: sbox[2,3,8] + rkey[1,2,7] - rkey[2,1,7] >= 0
 R1914: sbox[2,3,8] - rkey[1,2,7] + rkey[2,1,7] >= 0
 R1915: sbox[1,3,8] - rkey[0,3,7] = 0
 R1916: tbox[0,3,7] + tbox[2,1,7] - tbox[3,3,8] >= 0
 R1917: tbox[0,3,7] - tbox[2,1,7] + tbox[3,3,8] >= 0
 R1918: - tbox[0,3,7] + tbox[2,1,7] + tbox[3,3,8] >= 0
 R1919: - tbox[0,3,8] + tbox[3,0,7] + tbox[3,3,8] >= 0
 R1920: tbox[0,3,8] - tbox[3,0,7] + tbox[3,3,8] >= 0
 R1921: tbox[0,3,8] + tbox[3,0,7] - tbox[3,3,8] >= 0
 R1922: tbox[1,2,7] + tbox[2,1,7] - tbox[2,3,8] >= 0
 R1923: tbox[1,2,7] - tbox[2,1,7] + tbox[2,3,8] >= 0
 R1924: - tbox[1,2,7] + tbox[2,1,7] + tbox[2,3,8] >= 0
 R1925: - tbox[0,3,7] + tbox[1,3,8] = 0
 R1926: unkn[0,3,7] - unkn[3,3,8] + forg[3,3,8] <= 0
 R1927: unkn[2,1,7] - unkn[3,3,8] + forg[3,3,8] <= 0
 R1928: unkn[0,3,7] + unkn[2,1,7] - unkn[3,3,8] + forg[3,3,8] >= 0
 R1929: - unkn[0,3,8] + unkn[3,3,8] + forg[0,3,8] - forg[3,3,8] <= 0
 R1930: - unkn[0,3,8] + unkn[3,0,7] + forg[0,3,8] <= 0
 R1931: - unkn[0,3,8] + unkn[3,0,7] + unkn[3,3,8] + forg[0,3,8]
   - forg[3,3,8] >= 0
 R1932: unkn[1,2,7] - unkn[2,3,8] + forg[2,3,8] <= 0
 R1933: unkn[2,1,7] - unkn[2,3,8] + forg[2,3,8] <= 0
 R1934: unkn[1,2,7] + unkn[2,1,7] - unkn[2,3,8] + forg[2,3,8] >= 0
 R1935: unkn[0,3,7] - unkn[1,3,8] + forg[1,3,8] = 0
 R1936: - tbox[0,0,8] + forg[0,0,8] <= 0
 R1937: sbox[0,0,8] - rkey[0,0,8] + tkey[0,0,8] >= 0
 R1938: sbox[0,0,8] + rkey[0,0,8] - tkey[0,0,8] >= 0
 R1939: - sbox[0,0,8] + rkey[0,0,8] + tkey[0,0,8] >= 0
 R1940: - tbox[1,0,8] + forg[1,0,8] <= 0
 R1941: sbox[1,0,8] - rkey[1,0,8] + tkey[1,0,8] >= 0
 R1942: sbox[1,0,8] + rkey[1,0,8] - tkey[1,0,8] >= 0
 R1943: - sbox[1,0,8] + rkey[1,0,8] + tkey[1,0,8] >= 0
 R1944: - tbox[2,0,8] + forg[2,0,8] <= 0
 R1945: - sbox[2,0,8] + rkey[2,0,8] = 0
 R1946: - tbox[3,0,8] + forg[3,0,8] <= 0
 R1947: - sbox[3,0,8] + rkey[3,0,8] = 0
 R1948: - sbox[3,0,9] + rkey[0,0,8] + rkey[2,2,8] >= 0
 R1949: sbox[3,0,9] + rkey[0,0,8] - rkey[2,2,8] >= 0
 R1950: sbox[3,0,9] - rkey[0,0,8] + rkey[2,2,8] >= 0
 R1951: - sbox[0,0,9] + sbox[3,0,9] + rkey[3,1,8] >= 0
 R1952: sbox[0,0,9] + sbox[3,0,9] - rkey[3,1,8] >= 0
 R1953: sbox[0,0,9] - sbox[3,0,9] + rkey[3,1,8] >= 0
 R1954: - sbox[2,0,9] + rkey[1,3,8] + rkey[2,2,8] >= 0
 R1955: sbox[2,0,9] + rkey[1,3,8] - rkey[2,2,8] >= 0
 R1956: sbox[2,0,9] - rkey[1,3,8] + rkey[2,2,8] >= 0
 R1957: sbox[1,0,9] - rkey[0,0,8] = 0
 R1958: tbox[0,0,8] + tbox[2,2,8] - tbox[3,0,9] >= 0
 R1959: tbox[0,0,8] - tbox[2,2,8] + tbox[3,0,9] >= 0
 R1960: - tbox[0,0,8] + tbox[2,2,8] + tbox[3,0,9] >= 0
 R1961: - tbox[0,0,9] + tbox[3,0,9] + tbox[3,1,8] >= 0
 R1962: tbox[0,0,9] + tbox[3,0,9] - tbox[3,1,8] >= 0
 R1963: tbox[0,0,9] - tbox[3,0,9] + tbox[3,1,8] >= 0
 R1964: tbox[1,3,8] - tbox[2,0,9] + tbox[2,2,8] >= 0
 R1965: tbox[1,3,8] + tbox[2,0,9] - tbox[2,2,8] >= 0
 R1966: - tbox[1,3,8] + tbox[2,0,9] + tbox[2,2,8] >= 0
 R1967: - tbox[0,0,8] + tbox[1,0,9] = 0
 R1968: unkn[0,0,8] - unkn[3,0,9] + forg[3,0,9] <= 0
 R1969: unkn[2,2,8] - unkn[3,0,9] + forg[3,0,9] <= 0
 R1970: unkn[0,0,8] + unkn[2,2,8] - unkn[3,0,9] + forg[3,0,9] >= 0
 R1971: - unkn[0,0,9] + unkn[3,0,9] + forg[0,0,9] - forg[3,0,9] <= 0
 R1972: - unkn[0,0,9] + unkn[3,1,8] + forg[0,0,9] <= 0
 R1973: - unkn[0,0,9] + unkn[3,0,9] + unkn[3,1,8] + forg[0,0,9]
   - forg[3,0,9] >= 0
 R1974: unkn[1,3,8] - unkn[2,0,9] + forg[2,0,9] <= 0
 R1975: - unkn[2,0,9] + unkn[2,2,8] + forg[2,0,9] <= 0
 R1976: unkn[1,3,8] - unkn[2,0,9] + unkn[2,2,8] + forg[2,0,9] >= 0
 R1977: unkn[0,0,8] - unkn[1,0,9] + forg[1,0,9] = 0
 R1978: - tbox[0,1,8] + forg[0,1,8] <= 0
 R1979: sbox[0,1,8] - rkey[0,1,8] + tkey[0,1,8] >= 0
 R1980: sbox[0,1,8] + rkey[0,1,8] - tkey[0,1,8] >= 0
 R1981: - sbox[0,1,8] + rkey[0,1,8] + tkey[0,1,8] >= 0
 R1982: - tbox[1,1,8] + forg[1,1,8] <= 0
 R1983: sbox[1,1,8] - rkey[1,1,8] + tkey[1,1,8] >= 0
 R1984: sbox[1,1,8] + rkey[1,1,8] - tkey[1,1,8] >= 0
 R1985: - sbox[1,1,8] + rkey[1,1,8] + tkey[1,1,8] >= 0
 R1986: - tbox[2,1,8] + forg[2,1,8] <= 0
 R1987: - sbox[2,1,8] + rkey[2,1,8] = 0
 R1988: - tbox[3,1,8] + forg[3,1,8] <= 0
 R1989: - sbox[3,1,8] + rkey[3,1,8] = 0
 R1990: - sbox[3,1,9] + rkey[0,1,8] + rkey[2,3,8] >= 0
 R1991: sbox[3,1,9] + rkey[0,1,8] - rkey[2,3,8] >= 0
 R1992: sbox[3,1,9] - rkey[0,1,8] + rkey[2,3,8] >= 0
 R1993: - sbox[0,1,9] + sbox[3,1,9] + rkey[3,2,8] >= 0
 R1994: sbox[0,1,9] + sbox[3,1,9] - rkey[3,2,8] >= 0
 R1995: sbox[0,1,9] - sbox[3,1,9] + rkey[3,2,8] >= 0
 R1996: - sbox[2,1,9] + rkey[1,0,8] + rkey[2,3,8] >= 0
 R1997: sbox[2,1,9] + rkey[1,0,8] - rkey[2,3,8] >= 0
 R1998: sbox[2,1,9] - rkey[1,0,8] + rkey[2,3,8] >= 0
 R1999: sbox[1,1,9] - rkey[0,1,8] = 0
 R2000: tbox[0,1,8] + tbox[2,3,8] - tbox[3,1,9] >= 0
 R2001: tbox[0,1,8] - tbox[2,3,8] + tbox[3,1,9] >= 0
 R2002: - tbox[0,1,8] + tbox[2,3,8] + tbox[3,1,9] >= 0
 R2003: - tbox[0,1,9] + tbox[3,1,9] + tbox[3,2,8] >= 0
 R2004: tbox[0,1,9] + tbox[3,1,9] - tbox[3,2,8] >= 0
 R2005: tbox[0,1,9] - tbox[3,1,9] + tbox[3,2,8] >= 0
 R2006: tbox[1,0,8] - tbox[2,1,9] + tbox[2,3,8] >= 0
 R2007: tbox[1,0,8] + tbox[2,1,9] - tbox[2,3,8] >= 0
 R2008: - tbox[1,0,8] + tbox[2,1,9] + tbox[2,3,8] >= 0
 R2009: - tbox[0,1,8] + tbox[1,1,9] = 0
 R2010: unkn[0,1,8] - unkn[3,1,9] + forg[3,1,9] <= 0
 R2011: unkn[2,3,8] - unkn[3,1,9] + forg[3,1,9] <= 0
 R2012: unkn[0,1,8] + unkn[2,3,8] - unkn[3,1,9] + forg[3,1,9] >= 0
 R2013: - unkn[0,1,9] + unkn[3,1,9] + forg[0,1,9] - forg[3,1,9] <= 0
 R2014: - unkn[0,1,9] + unkn[3,2,8] + forg[0,1,9] <= 0
 R2015: - unkn[0,1,9] + unkn[3,1,9] + unkn[3,2,8] + forg[0,1,9]
   - forg[3,1,9] >= 0
 R2016: unkn[1,0,8] - unkn[2,1,9] + forg[2,1,9] <= 0
 R2017: - unkn[2,1,9] + unkn[2,3,8] + forg[2,1,9] <= 0
 R2018: unkn[1,0,8] - unkn[2,1,9] + unkn[2,3,8] + forg[2,1,9] >= 0
 R2019: unkn[0,1,8] - unkn[1,1,9] + forg[1,1,9] = 0
 R2020: - tbox[0,2,8] + forg[0,2,8] <= 0
 R2021: sbox[0,2,8] - rkey[0,2,8] + tkey[0,2,8] >= 0
 R2022: sbox[0,2,8] + rkey[0,2,8] - tkey[0,2,8] >= 0
 R2023: - sbox[0,2,8] + rkey[0,2,8] + tkey[0,2,8] >= 0
 R2024: - tbox[1,2,8] + forg[1,2,8] <= 0
 R2025: sbox[1,2,8] - rkey[1,2,8] + tkey[1,2,8] >= 0
 R2026: sbox[1,2,8] + rkey[1,2,8] - tkey[1,2,8] >= 0
 R2027: - sbox[1,2,8] + rkey[1,2,8] + tkey[1,2,8] >= 0
 R2028: - tbox[2,2,8] + forg[2,2,8] <= 0
 R2029: - sbox[2,2,8] + rkey[2,2,8] = 0
 R2030: - tbox[3,2,8] + forg[3,2,8] <= 0
 R2031: - sbox[3,2,8] + rkey[3,2,8] = 0
 R2032: - sbox[3,2,9] + rkey[0,2,8] + rkey[2,0,8] >= 0
 R2033: sbox[3,2,9] + rkey[0,2,8] - rkey[2,0,8] >= 0
 R2034: sbox[3,2,9] - rkey[0,2,8] + rkey[2,0,8] >= 0
 R2035: - sbox[0,2,9] + sbox[3,2,9] + rkey[3,3,8] >= 0
 R2036: sbox[0,2,9] + sbox[3,2,9] - rkey[3,3,8] >= 0
 R2037: sbox[0,2,9] - sbox[3,2,9] + rkey[3,3,8] >= 0
 R2038: - sbox[2,2,9] + rkey[1,1,8] + rkey[2,0,8] >= 0
 R2039: sbox[2,2,9] + rkey[1,1,8] - rkey[2,0,8] >= 0
 R2040: sbox[2,2,9] - rkey[1,1,8] + rkey[2,0,8] >= 0
 R2041: sbox[1,2,9] - rkey[0,2,8] = 0
 R2042: tbox[0,2,8] + tbox[2,0,8] - tbox[3,2,9] >= 0
 R2043: tbox[0,2,8] - tbox[2,0,8] + tbox[3,2,9] >= 0
 R2044: - tbox[0,2,8] + tbox[2,0,8] + tbox[3,2,9] >= 0
 R2045: - tbox[0,2,9] + tbox[3,2,9] + tbox[3,3,8] >= 0
 R2046: tbox[0,2,9] + tbox[3,2,9] - tbox[3,3,8] >= 0
 R2047: tbox[0,2,9] - tbox[3,2,9] + tbox[3,3,8] >= 0
 R2048: tbox[1,1,8] + tbox[2,0,8] - tbox[2,2,9] >= 0
 R2049: tbox[1,1,8] - tbox[2,0,8] + tbox[2,2,9] >= 0
 R2050: - tbox[1,1,8] + tbox[2,0,8] + tbox[2,2,9] >= 0
 R2051: - tbox[0,2,8] + tbox[1,2,9] = 0
 R2052: unkn[0,2,8] - unkn[3,2,9] + forg[3,2,9] <= 0
 R2053: unkn[2,0,8] - unkn[3,2,9] + forg[3,2,9] <= 0
 R2054: unkn[0,2,8] + unkn[2,0,8] - unkn[3,2,9] + forg[3,2,9] >= 0
 R2055: - unkn[0,2,9] + unkn[3,2,9] + forg[0,2,9] - forg[3,2,9] <= 0
 R2056: - unkn[0,2,9] + unkn[3,3,8] + forg[0,2,9] <= 0
 R2057: - unkn[0,2,9] + unkn[3,2,9] + unkn[3,3,8] + forg[0,2,9]
   - forg[3,2,9] >= 0
 R2058: unkn[1,1,8] - unkn[2,2,9] + forg[2,2,9] <= 0
 R2059: unkn[2,0,8] - unkn[2,2,9] + forg[2,2,9] <= 0
 R2060: unkn[1,1,8] + unkn[2,0,8] - unkn[2,2,9] + forg[2,2,9] >= 0
 R2061: unkn[0,2,8] - unkn[1,2,9] + forg[1,2,9] = 0
 R2062: - tbox[0,3,8] + forg[0,3,8] <= 0
 R2063: sbox[0,3,8] - rkey[0,3,8] + tkey[0,3,8] >= 0
 R2064: sbox[0,3,8] + rkey[0,3,8] - tkey[0,3,8] >= 0
 R2065: - sbox[0,3,8] + rkey[0,3,8] + tkey[0,3,8] >= 0
 R2066: - tbox[1,3,8] + forg[1,3,8] <= 0
 R2067: sbox[1,3,8] - rkey[1,3,8] + tkey[1,3,8] >= 0
 R2068: sbox[1,3,8] + rkey[1,3,8] - tkey[1,3,8] >= 0
 R2069: - sbox[1,3,8] + rkey[1,3,8] + tkey[1,3,8] >= 0
 R2070: - tbox[2,3,8] + forg[2,3,8] <= 0
 R2071: - sbox[2,3,8] + rkey[2,3,8] = 0
 R2072: - tbox[3,3,8] + forg[3,3,8] <= 0
 R2073: - sbox[3,3,8] + rkey[3,3,8] = 0
 R2074: - sbox[3,3,9] + rkey[0,3,8] + rkey[2,1,8] >= 0
 R2075: sbox[3,3,9] + rkey[0,3,8] - rkey[2,1,8] >= 0
 R2076: sbox[3,3,9] - rkey[0,3,8] + rkey[2,1,8] >= 0
 R2077: - sbox[0,3,9] + sbox[3,3,9] + rkey[3,0,8] >= 0
 R2078: sbox[0,3,9] + sbox[3,3,9] - rkey[3,0,8] >= 0
 R2079: sbox[0,3,9] - sbox[3,3,9] + rkey[3,0,8] >= 0
 R2080: - sbox[2,3,9] + rkey[1,2,8] + rkey[2,1,8] >= 0
 R2081: sbox[2,3,9] + rkey[1,2,8] - rkey[2,1,8] >= 0
 R2082: sbox[2,3,9] - rkey[1,2,8] + rkey[2,1,8] >= 0
 R2083: sbox[1,3,9] - rkey[0,3,8] = 0
 R2084: tbox[0,3,8] + tbox[2,1,8] - tbox[3,3,9] >= 0
 R2085: tbox[0,3,8] - tbox[2,1,8] + tbox[3,3,9] >= 0
 R2086: - tbox[0,3,8] + tbox[2,1,8] + tbox[3,3,9] >= 0
 R2087: - tbox[0,3,9] + tbox[3,0,8] + tbox[3,3,9] >= 0
 R2088: tbox[0,3,9] - tbox[3,0,8] + tbox[3,3,9] >= 0
 R2089: tbox[0,3,9] + tbox[3,0,8] - tbox[3,3,9] >= 0
 R2090: tbox[1,2,8] + tbox[2,1,8] - tbox[2,3,9] >= 0
 R2091: tbox[1,2,8] - tbox[2,1,8] + tbox[2,3,9] >= 0
 R2092: - tbox[1,2,8] + tbox[2,1,8] + tbox[2,3,9] >= 0
 R2093: - tbox[0,3,8] + tbox[1,3,9] = 0
 R2094: unkn[0,3,8] - unkn[3,3,9] + forg[3,3,9] <= 0
 R2095: unkn[2,1,8] - unkn[3,3,9] + forg[3,3,9] <= 0
 R2096: unkn[0,3,8] + unkn[2,1,8] - unkn[3,3,9] + forg[3,3,9] >= 0
 R2097: - unkn[0,3,9] + unkn[3,3,9] + forg[0,3,9] - forg[3,3,9] <= 0
 R2098: - unkn[0,3,9] + unkn[3,0,8] + forg[0,3,9] <= 0
 R2099: - unkn[0,3,9] + unkn[3,0,8] + unkn[3,3,9] + forg[0,3,9]
   - forg[3,3,9] >= 0
 R2100: unkn[1,2,8] - unkn[2,3,9] + forg[2,3,9] <= 0
 R2101: unkn[2,1,8] - unkn[2,3,9] + forg[2,3,9] <= 0
 R2102: unkn[1,2,8] + unkn[2,1,8] - unkn[2,3,9] + forg[2,3,9] >= 0
 R2103: unkn[0,3,8] - unkn[1,3,9] + forg[1,3,9] = 0
 R2104: sbox[0,0,0] + sbox[0,1,0] + sbox[0,2,0] + sbox[0,3,0] + sbox[1,0,0]
   + sbox[1,1,0] + sbox[1,2,0] + sbox[1,3,0] + sbox[2,0,0] + sbox[2,1,0]
   + sbox[2,2,0] + sbox[2,3,0] + sbox[3,0,0] + sbox[3,1,0] + sbox[3,2,0]
   + sbox[3,3,0] + tkey[0,0,0] + tkey[0,1,0] + tkey[0,2,0] + tkey[0,3,0]
   + tkey[1,0,0] + tkey[1,1,0] + tkey[1,2,0] + tkey[1,3,0] + tkey[2,0,0]
   + tkey[2,1,0] + tkey[2,2,0] + tkey[2,3,0] + tkey[3,0,0] + tkey[3,1,0]
   + tkey[3,2,0] + tkey[3,3,0] >= 1
 R2105: sbox[0,0,1] + sbox[0,1,1] + sbox[0,2,1] + sbox[0,3,1] + sbox[1,0,1]
   + sbox[1,1,1] + sbox[1,2,1] + sbox[1,3,1] + sbox[2,0,1] + sbox[2,1,1]
   + sbox[2,2,1] + sbox[2,3,1] + sbox[3,0,1] + sbox[3,1,1] + sbox[3,2,1]
   + sbox[3,3,1] + tkey[0,0,1] + tkey[0,1,1] + tkey[0,2,1] + tkey[0,3,1]
   + tkey[1,0,1] + tkey[1,1,1] + tkey[1,2,1] + tkey[1,3,1] + tkey[2,0,1]
   + tkey[2,1,1] + tkey[2,2,1] + tkey[2,3,1] + tkey[3,0,1] + tkey[3,1,1]
   + tkey[3,2,1] + tkey[3,3,1] >= 1
 R2106: sbox[0,0,2] + sbox[0,1,2] + sbox[0,2,2] + sbox[0,3,2] + sbox[1,0,2]
   + sbox[1,1,2] + sbox[1,2,2] + sbox[1,3,2] + sbox[2,0,2] + sbox[2,1,2]
   + sbox[2,2,2] + sbox[2,3,2] + sbox[3,0,2] + sbox[3,1,2] + sbox[3,2,2]
   + sbox[3,3,2] + tkey[0,0,2] + tkey[0,1,2] + tkey[0,2,2] + tkey[0,3,2]
   + tkey[1,0,2] + tkey[1,1,2] + tkey[1,2,2] + tkey[1,3,2] + tkey[2,0,2]
   + tkey[2,1,2] + tkey[2,2,2] + tkey[2,3,2] + tkey[3,0,2] + tkey[3,1,2]
   + tkey[3,2,2] + tkey[3,3,2] >= 1
 R2107: sbox[0,0,3] + sbox[0,1,3] + sbox[0,2,3] + sbox[0,3,3] + sbox[1,0,3]
   + sbox[1,1,3] + sbox[1,2,3] + sbox[1,3,3] + sbox[2,0,3] + sbox[2,1,3]
   + sbox[2,2,3] + sbox[2,3,3] + sbox[3,0,3] + sbox[3,1,3] + sbox[3,2,3]
   + sbox[3,3,3] + tkey[0,0,3] + tkey[0,1,3] + tkey[0,2,3] + tkey[0,3,3]
   + tkey[1,0,3] + tkey[1,1,3] + tkey[1,2,3] + tkey[1,3,3] + tkey[2,0,3]
   + tkey[2,1,3] + tkey[2,2,3] + tkey[2,3,3] + tkey[3,0,3] + tkey[3,1,3]
   + tkey[3,2,3] + tkey[3,3,3] >= 1
 R2108: sbox[0,0,4] + sbox[0,1,4] + sbox[0,2,4] + sbox[0,3,4] + sbox[1,0,4]
   + sbox[1,1,4] + sbox[1,2,4] + sbox[1,3,4] + sbox[2,0,4] + sbox[2,1,4]
   + sbox[2,2,4] + sbox[2,3,4] + sbox[3,0,4] + sbox[3,1,4] + sbox[3,2,4]
   + sbox[3,3,4] + tkey[0,0,4] + tkey[0,1,4] + tkey[0,2,4] + tkey[0,3,4]
   + tkey[1,0,4] + tkey[1,1,4] + tkey[1,2,4] + tkey[1,3,4] + tkey[2,0,4]
   + tkey[2,1,4] + tkey[2,2,4] + tkey[2,3,4] + tkey[3,0,4] + tkey[3,1,4]
   + tkey[3,2,4] + tkey[3,3,4] >= 1
 R2109: sbox[0,0,5] + sbox[0,1,5] + sbox[0,2,5] + sbox[0,3,5] + sbox[1,0,5]
   + sbox[1,1,5] + sbox[1,2,5] + sbox[1,3,5] + sbox[2,0,5] + sbox[2,1,5]
   + sbox[2,2,5] + sbox[2,3,5] + sbox[3,0,5] + sbox[3,1,5] + sbox[3,2,5]
   + sbox[3,3,5] + tkey[0,0,5] + tkey[0,1,5] + tkey[0,2,5] + tkey[0,3,5]
   + tkey[1,0,5] + tkey[1,1,5] + tkey[1,2,5] + tkey[1,3,5] + tkey[2,0,5]
   + tkey[2,1,5] + tkey[2,2,5] + tkey[2,3,5] + tkey[3,0,5] + tkey[3,1,5]
   + tkey[3,2,5] + tkey[3,3,5] >= 1
 R2110: sbox[0,0,6] + sbox[0,1,6] + sbox[0,2,6] + sbox[0,3,6] + sbox[1,0,6]
   + sbox[1,1,6] + sbox[1,2,6] + sbox[1,3,6] + sbox[2,0,6] + sbox[2,1,6]
   + sbox[2,2,6] + sbox[2,3,6] + sbox[3,0,6] + sbox[3,1,6] + sbox[3,2,6]
   + sbox[3,3,6] + tkey[0,0,6] + tkey[0,1,6] + tkey[0,2,6] + tkey[0,3,6]
   + tkey[1,0,6] + tkey[1,1,6] + tkey[1,2,6] + tkey[1,3,6] + tkey[2,0,6]
   + tkey[2,1,6] + tkey[2,2,6] + tkey[2,3,6] + tkey[3,0,6] + tkey[3,1,6]
   + tkey[3,2,6] + tkey[3,3,6] >= 1
 R2111: sbox[0,0,7] + sbox[0,1,7] + sbox[0,2,7] + sbox[0,3,7] + sbox[1,0,7]
   + sbox[1,1,7] + sbox[1,2,7] + sbox[1,3,7] + sbox[2,0,7] + sbox[2,1,7]
   + sbox[2,2,7] + sbox[2,3,7] + sbox[3,0,7] + sbox[3,1,7] + sbox[3,2,7]
   + sbox[3,3,7] + tkey[0,0,7] + tkey[0,1,7] + tkey[0,2,7] + tkey[0,3,7]
   + tkey[1,0,7] + tkey[1,1,7] + tkey[1,2,7] + tkey[1,3,7] + tkey[2,0,7]
   + tkey[2,1,7] + tkey[2,2,7] + tkey[2,3,7] + tkey[3,0,7] + tkey[3,1,7]
   + tkey[3,2,7] + tkey[3,3,7] >= 1
 R2112: sbox[0,0,8] + sbox[0,1,8] + sbox[0,2,8] + sbox[0,3,8] + sbox[1,0,8]
   + sbox[1,1,8] + sbox[1,2,8] + sbox[1,3,8] + sbox[2,0,8] + sbox[2,1,8]
   + sbox[2,2,8] + sbox[2,3,8] + sbox[3,0,8] + sbox[3,1,8] + sbox[3,2,8]
   + sbox[3,3,8] + tkey[0,0,8] + tkey[0,1,8] + tkey[0,2,8] + tkey[0,3,8]
   + tkey[1,0,8] + tkey[1,1,8] + tkey[1,2,8] + tkey[1,3,8] + tkey[2,0,8]
   + tkey[2,1,8] + tkey[2,2,8] + tkey[2,3,8] + tkey[3,0,8] + tkey[3,1,8]
   + tkey[3,2,8] + tkey[3,3,8] >= 1
 R2113: sbox[0,0,0] + sbox[0,1,0] + sbox[0,2,0] + sbox[0,3,0] + sbox[1,0,0]
   + sbox[1,1,0] + sbox[1,2,0] + sbox[1,3,0] + sbox[2,0,0] + sbox[2,1,0]
   + sbox[2,2,0] + sbox[2,3,0] + sbox[3,0,0] + sbox[3,1,0] + sbox[3,2,0]
   + sbox[3,3,0] = 0
 R2114: sbox[0,0,9] + sbox[0,1,9] + sbox[0,2,9] + sbox[0,3,9] + sbox[1,0,9]
   + sbox[1,1,9] + sbox[1,2,9] + sbox[1,3,9] + sbox[2,0,9] + sbox[2,1,9]
   + sbox[2,2,9] + sbox[2,3,9] + sbox[3,0,9] + sbox[3,1,9] + sbox[3,2,9]
   + sbox[3,3,9] = 0
 R2115: tbox[0,0,0] = 1
 R2116: unkn[0,0,0] - forg[0,0,0] = 0
 R2117: tbox[0,1,0] = 0
 R2118: unkn[0,1,0] - forg[0,1,0] = 0
 R2119: tbox[0,2,0] = 0
 R2120: unkn[0,2,0] - forg[0,2,0] = 0
 R2121: tbox[0,3,0] = 0
 R2122: unkn[0,3,0] - forg[0,3,0] = 0
 R2123: tbox[1,0,0] = 0
 R2124: unkn[1,0,0] - forg[1,0,0] = 0
 R2125: tbox[1,1,0] = 0
 R2126: unkn[1,1,0] - forg[1,1,0] = 0
 R2127: tbox[1,2,0] = 0
 R2128: unkn[1,2,0] - forg[1,2,0] = 0
 R2129: tbox[1,3,0] = 0
 R2130: unkn[1,3,0] - forg[1,3,0] = 0
 R2131: tbox[2,0,0] = 0
 R2132: unkn[2,0,0] - forg[2,0,0] = 0
 R2133: tbox[2,1,0] = 0
 R2134: unkn[2,1,0] - forg[2,1,0] = 0
 R2135: tbox[2,2,0] = 0
 R2136: unkn[2,2,0] - forg[2,2,0] = 0
 R2137: tbox[2,3,0] = 0
 R2138: unkn[2,3,0] - forg[2,3,0] = 0
 R2139: tbox[3,0,0] = 0
 R2140: unkn[3,0,0] - forg[3,0,0] = 0
 R2141: tbox[3,1,0] = 0
 R2142: unkn[3,1,0] - forg[3,1,0] = 0
 R2143: tbox[3,2,0] = 0
 R2144: unkn[3,2,0] - forg[3,2,0] = 0
 R2145: tbox[3,3,0] = 0
 R2146: unkn[3,3,0] - forg[3,3,0] = 0
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
 cost[0,0,9] <= 2
 cost[0,1,0] <= 2
 cost[0,1,1] <= 2
 cost[0,1,2] <= 2
 cost[0,1,3] <= 2
 cost[0,1,4] <= 2
 cost[0,1,5] <= 2
 cost[0,1,6] <= 2
 cost[0,1,7] <= 2
 cost[0,1,8] <= 2
 cost[0,1,9] <= 2
 cost[0,2,0] <= 2
 cost[0,2,1] <= 2
 cost[0,2,2] <= 2
 cost[0,2,3] <= 2
 cost[0,2,4] <= 2
 cost[0,2,5] <= 2
 cost[0,2,6] <= 2
 cost[0,2,7] <= 2
 cost[0,2,8] <= 2
 cost[0,2,9] <= 2
 cost[0,3,0] <= 2
 cost[0,3,1] <= 2
 cost[0,3,2] <= 2
 cost[0,3,3] <= 2
 cost[0,3,4] <= 2
 cost[0,3,5] <= 2
 cost[0,3,6] <= 2
 cost[0,3,7] <= 2
 cost[0,3,8] <= 2
 cost[0,3,9] <= 2
 cost[1,0,0] <= 2
 cost[1,0,1] <= 2
 cost[1,0,2] <= 2
 cost[1,0,3] <= 2
 cost[1,0,4] <= 2
 cost[1,0,5] <= 2
 cost[1,0,6] <= 2
 cost[1,0,7] <= 2
 cost[1,0,8] <= 2
 cost[1,0,9] <= 2
 cost[1,1,0] <= 2
 cost[1,1,1] <= 2
 cost[1,1,2] <= 2
 cost[1,1,3] <= 2
 cost[1,1,4] <= 2
 cost[1,1,5] <= 2
 cost[1,1,6] <= 2
 cost[1,1,7] <= 2
 cost[1,1,8] <= 2
 cost[1,1,9] <= 2
 cost[1,2,0] <= 2
 cost[1,2,1] <= 2
 cost[1,2,2] <= 2
 cost[1,2,3] <= 2
 cost[1,2,4] <= 2
 cost[1,2,5] <= 2
 cost[1,2,6] <= 2
 cost[1,2,7] <= 2
 cost[1,2,8] <= 2
 cost[1,2,9] <= 2
 cost[1,3,0] <= 2
 cost[1,3,1] <= 2
 cost[1,3,2] <= 2
 cost[1,3,3] <= 2
 cost[1,3,4] <= 2
 cost[1,3,5] <= 2
 cost[1,3,6] <= 2
 cost[1,3,7] <= 2
 cost[1,3,8] <= 2
 cost[1,3,9] <= 2
 cost[2,0,0] <= 2
 cost[2,0,1] <= 2
 cost[2,0,2] <= 2
 cost[2,0,3] <= 2
 cost[2,0,4] <= 2
 cost[2,0,5] <= 2
 cost[2,0,6] <= 2
 cost[2,0,7] <= 2
 cost[2,0,8] <= 2
 cost[2,0,9] <= 2
 cost[2,1,0] <= 2
 cost[2,1,1] <= 2
 cost[2,1,2] <= 2
 cost[2,1,3] <= 2
 cost[2,1,4] <= 2
 cost[2,1,5] <= 2
 cost[2,1,6] <= 2
 cost[2,1,7] <= 2
 cost[2,1,8] <= 2
 cost[2,1,9] <= 2
 cost[2,2,0] <= 2
 cost[2,2,1] <= 2
 cost[2,2,2] <= 2
 cost[2,2,3] <= 2
 cost[2,2,4] <= 2
 cost[2,2,5] <= 2
 cost[2,2,6] <= 2
 cost[2,2,7] <= 2
 cost[2,2,8] <= 2
 cost[2,2,9] <= 2
 cost[2,3,0] <= 2
 cost[2,3,1] <= 2
 cost[2,3,2] <= 2
 cost[2,3,3] <= 2
 cost[2,3,4] <= 2
 cost[2,3,5] <= 2
 cost[2,3,6] <= 2
 cost[2,3,7] <= 2
 cost[2,3,8] <= 2
 cost[2,3,9] <= 2
 cost[3,0,0] <= 2
 cost[3,0,1] <= 2
 cost[3,0,2] <= 2
 cost[3,0,3] <= 2
 cost[3,0,4] <= 2
 cost[3,0,5] <= 2
 cost[3,0,6] <= 2
 cost[3,0,7] <= 2
 cost[3,0,8] <= 2
 cost[3,0,9] <= 2
 cost[3,1,0] <= 2
 cost[3,1,1] <= 2
 cost[3,1,2] <= 2
 cost[3,1,3] <= 2
 cost[3,1,4] <= 2
 cost[3,1,5] <= 2
 cost[3,1,6] <= 2
 cost[3,1,7] <= 2
 cost[3,1,8] <= 2
 cost[3,1,9] <= 2
 cost[3,2,0] <= 2
 cost[3,2,1] <= 2
 cost[3,2,2] <= 2
 cost[3,2,3] <= 2
 cost[3,2,4] <= 2
 cost[3,2,5] <= 2
 cost[3,2,6] <= 2
 cost[3,2,7] <= 2
 cost[3,2,8] <= 2
 cost[3,2,9] <= 2
 cost[3,3,0] <= 2
 cost[3,3,1] <= 2
 cost[3,3,2] <= 2
 cost[3,3,3] <= 2
 cost[3,3,4] <= 2
 cost[3,3,5] <= 2
 cost[3,3,6] <= 2
 cost[3,3,7] <= 2
 cost[3,3,8] <= 2
 cost[3,3,9] <= 2
Binaries
 sbox[0,0,0] sbox[0,0,1] sbox[0,0,2] sbox[0,0,3] sbox[0,0,4] sbox[0,0,5]
 sbox[0,0,6] sbox[0,0,7] sbox[0,0,8] sbox[0,0,9] sbox[0,1,0] sbox[0,1,1]
 sbox[0,1,2] sbox[0,1,3] sbox[0,1,4] sbox[0,1,5] sbox[0,1,6] sbox[0,1,7]
 sbox[0,1,8] sbox[0,1,9] sbox[0,2,0] sbox[0,2,1] sbox[0,2,2] sbox[0,2,3]
 sbox[0,2,4] sbox[0,2,5] sbox[0,2,6] sbox[0,2,7] sbox[0,2,8] sbox[0,2,9]
 sbox[0,3,0] sbox[0,3,1] sbox[0,3,2] sbox[0,3,3] sbox[0,3,4] sbox[0,3,5]
 sbox[0,3,6] sbox[0,3,7] sbox[0,3,8] sbox[0,3,9] sbox[1,0,0] sbox[1,0,1]
 sbox[1,0,2] sbox[1,0,3] sbox[1,0,4] sbox[1,0,5] sbox[1,0,6] sbox[1,0,7]
 sbox[1,0,8] sbox[1,0,9] sbox[1,1,0] sbox[1,1,1] sbox[1,1,2] sbox[1,1,3]
 sbox[1,1,4] sbox[1,1,5] sbox[1,1,6] sbox[1,1,7] sbox[1,1,8] sbox[1,1,9]
 sbox[1,2,0] sbox[1,2,1] sbox[1,2,2] sbox[1,2,3] sbox[1,2,4] sbox[1,2,5]
 sbox[1,2,6] sbox[1,2,7] sbox[1,2,8] sbox[1,2,9] sbox[1,3,0] sbox[1,3,1]
 sbox[1,3,2] sbox[1,3,3] sbox[1,3,4] sbox[1,3,5] sbox[1,3,6] sbox[1,3,7]
 sbox[1,3,8] sbox[1,3,9] sbox[2,0,0] sbox[2,0,1] sbox[2,0,2] sbox[2,0,3]
 sbox[2,0,4] sbox[2,0,5] sbox[2,0,6] sbox[2,0,7] sbox[2,0,8] sbox[2,0,9]
 sbox[2,1,0] sbox[2,1,1] sbox[2,1,2] sbox[2,1,3] sbox[2,1,4] sbox[2,1,5]
 sbox[2,1,6] sbox[2,1,7] sbox[2,1,8] sbox[2,1,9] sbox[2,2,0] sbox[2,2,1]
 sbox[2,2,2] sbox[2,2,3] sbox[2,2,4] sbox[2,2,5] sbox[2,2,6] sbox[2,2,7]
 sbox[2,2,8] sbox[2,2,9] sbox[2,3,0] sbox[2,3,1] sbox[2,3,2] sbox[2,3,3]
 sbox[2,3,4] sbox[2,3,5] sbox[2,3,6] sbox[2,3,7] sbox[2,3,8] sbox[2,3,9]
 sbox[3,0,0] sbox[3,0,1] sbox[3,0,2] sbox[3,0,3] sbox[3,0,4] sbox[3,0,5]
 sbox[3,0,6] sbox[3,0,7] sbox[3,0,8] sbox[3,0,9] sbox[3,1,0] sbox[3,1,1]
 sbox[3,1,2] sbox[3,1,3] sbox[3,1,4] sbox[3,1,5] sbox[3,1,6] sbox[3,1,7]
 sbox[3,1,8] sbox[3,1,9] sbox[3,2,0] sbox[3,2,1] sbox[3,2,2] sbox[3,2,3]
 sbox[3,2,4] sbox[3,2,5] sbox[3,2,6] sbox[3,2,7] sbox[3,2,8] sbox[3,2,9]
 sbox[3,3,0] sbox[3,3,1] sbox[3,3,2] sbox[3,3,3] sbox[3,3,4] sbox[3,3,5]
 sbox[3,3,6] sbox[3,3,7] sbox[3,3,8] sbox[3,3,9] rkey[0,0,0] rkey[0,0,1]
 rkey[0,0,2] rkey[0,0,3] rkey[0,0,4] rkey[0,0,5] rkey[0,0,6] rkey[0,0,7]
 rkey[0,0,8] rkey[0,0,9] rkey[0,1,0] rkey[0,1,1] rkey[0,1,2] rkey[0,1,3]
 rkey[0,1,4] rkey[0,1,5] rkey[0,1,6] rkey[0,1,7] rkey[0,1,8] rkey[0,1,9]
 rkey[0,2,0] rkey[0,2,1] rkey[0,2,2] rkey[0,2,3] rkey[0,2,4] rkey[0,2,5]
 rkey[0,2,6] rkey[0,2,7] rkey[0,2,8] rkey[0,2,9] rkey[0,3,0] rkey[0,3,1]
 rkey[0,3,2] rkey[0,3,3] rkey[0,3,4] rkey[0,3,5] rkey[0,3,6] rkey[0,3,7]
 rkey[0,3,8] rkey[0,3,9] rkey[1,0,0] rkey[1,0,1] rkey[1,0,2] rkey[1,0,3]
 rkey[1,0,4] rkey[1,0,5] rkey[1,0,6] rkey[1,0,7] rkey[1,0,8] rkey[1,0,9]
 rkey[1,1,0] rkey[1,1,1] rkey[1,1,2] rkey[1,1,3] rkey[1,1,4] rkey[1,1,5]
 rkey[1,1,6] rkey[1,1,7] rkey[1,1,8] rkey[1,1,9] rkey[1,2,0] rkey[1,2,1]
 rkey[1,2,2] rkey[1,2,3] rkey[1,2,4] rkey[1,2,5] rkey[1,2,6] rkey[1,2,7]
 rkey[1,2,8] rkey[1,2,9] rkey[1,3,0] rkey[1,3,1] rkey[1,3,2] rkey[1,3,3]
 rkey[1,3,4] rkey[1,3,5] rkey[1,3,6] rkey[1,3,7] rkey[1,3,8] rkey[1,3,9]
 rkey[2,0,0] rkey[2,0,1] rkey[2,0,2] rkey[2,0,3] rkey[2,0,4] rkey[2,0,5]
 rkey[2,0,6] rkey[2,0,7] rkey[2,0,8] rkey[2,0,9] rkey[2,1,0] rkey[2,1,1]
 rkey[2,1,2] rkey[2,1,3] rkey[2,1,4] rkey[2,1,5] rkey[2,1,6] rkey[2,1,7]
 rkey[2,1,8] rkey[2,1,9] rkey[2,2,0] rkey[2,2,1] rkey[2,2,2] rkey[2,2,3]
 rkey[2,2,4] rkey[2,2,5] rkey[2,2,6] rkey[2,2,7] rkey[2,2,8] rkey[2,2,9]
 rkey[2,3,0] rkey[2,3,1] rkey[2,3,2] rkey[2,3,3] rkey[2,3,4] rkey[2,3,5]
 rkey[2,3,6] rkey[2,3,7] rkey[2,3,8] rkey[2,3,9] rkey[3,0,0] rkey[3,0,1]
 rkey[3,0,2] rkey[3,0,3] rkey[3,0,4] rkey[3,0,5] rkey[3,0,6] rkey[3,0,7]
 rkey[3,0,8] rkey[3,0,9] rkey[3,1,0] rkey[3,1,1] rkey[3,1,2] rkey[3,1,3]
 rkey[3,1,4] rkey[3,1,5] rkey[3,1,6] rkey[3,1,7] rkey[3,1,8] rkey[3,1,9]
 rkey[3,2,0] rkey[3,2,1] rkey[3,2,2] rkey[3,2,3] rkey[3,2,4] rkey[3,2,5]
 rkey[3,2,6] rkey[3,2,7] rkey[3,2,8] rkey[3,2,9] rkey[3,3,0] rkey[3,3,1]
 rkey[3,3,2] rkey[3,3,3] rkey[3,3,4] rkey[3,3,5] rkey[3,3,6] rkey[3,3,7]
 rkey[3,3,8] rkey[3,3,9] tkey[0,0,0] tkey[0,0,1] tkey[0,0,2] tkey[0,0,3]
 tkey[0,0,4] tkey[0,0,5] tkey[0,0,6] tkey[0,0,7] tkey[0,0,8] tkey[0,0,9]
 tkey[0,1,0] tkey[0,1,1] tkey[0,1,2] tkey[0,1,3] tkey[0,1,4] tkey[0,1,5]
 tkey[0,1,6] tkey[0,1,7] tkey[0,1,8] tkey[0,1,9] tkey[0,2,0] tkey[0,2,1]
 tkey[0,2,2] tkey[0,2,3] tkey[0,2,4] tkey[0,2,5] tkey[0,2,6] tkey[0,2,7]
 tkey[0,2,8] tkey[0,2,9] tkey[0,3,0] tkey[0,3,1] tkey[0,3,2] tkey[0,3,3]
 tkey[0,3,4] tkey[0,3,5] tkey[0,3,6] tkey[0,3,7] tkey[0,3,8] tkey[0,3,9]
 tkey[1,0,0] tkey[1,0,1] tkey[1,0,2] tkey[1,0,3] tkey[1,0,4] tkey[1,0,5]
 tkey[1,0,6] tkey[1,0,7] tkey[1,0,8] tkey[1,0,9] tkey[1,1,0] tkey[1,1,1]
 tkey[1,1,2] tkey[1,1,3] tkey[1,1,4] tkey[1,1,5] tkey[1,1,6] tkey[1,1,7]
 tkey[1,1,8] tkey[1,1,9] tkey[1,2,0] tkey[1,2,1] tkey[1,2,2] tkey[1,2,3]
 tkey[1,2,4] tkey[1,2,5] tkey[1,2,6] tkey[1,2,7] tkey[1,2,8] tkey[1,2,9]
 tkey[1,3,0] tkey[1,3,1] tkey[1,3,2] tkey[1,3,3] tkey[1,3,4] tkey[1,3,5]
 tkey[1,3,6] tkey[1,3,7] tkey[1,3,8] tkey[1,3,9] tkey[2,0,0] tkey[2,0,1]
 tkey[2,0,2] tkey[2,0,3] tkey[2,0,4] tkey[2,0,5] tkey[2,0,6] tkey[2,0,7]
 tkey[2,0,8] tkey[2,0,9] tkey[2,1,0] tkey[2,1,1] tkey[2,1,2] tkey[2,1,3]
 tkey[2,1,4] tkey[2,1,5] tkey[2,1,6] tkey[2,1,7] tkey[2,1,8] tkey[2,1,9]
 tkey[2,2,0] tkey[2,2,1] tkey[2,2,2] tkey[2,2,3] tkey[2,2,4] tkey[2,2,5]
 tkey[2,2,6] tkey[2,2,7] tkey[2,2,8] tkey[2,2,9] tkey[2,3,0] tkey[2,3,1]
 tkey[2,3,2] tkey[2,3,3] tkey[2,3,4] tkey[2,3,5] tkey[2,3,6] tkey[2,3,7]
 tkey[2,3,8] tkey[2,3,9] tkey[3,0,0] tkey[3,0,1] tkey[3,0,2] tkey[3,0,3]
 tkey[3,0,4] tkey[3,0,5] tkey[3,0,6] tkey[3,0,7] tkey[3,0,8] tkey[3,0,9]
 tkey[3,1,0] tkey[3,1,1] tkey[3,1,2] tkey[3,1,3] tkey[3,1,4] tkey[3,1,5]
 tkey[3,1,6] tkey[3,1,7] tkey[3,1,8] tkey[3,1,9] tkey[3,2,0] tkey[3,2,1]
 tkey[3,2,2] tkey[3,2,3] tkey[3,2,4] tkey[3,2,5] tkey[3,2,6] tkey[3,2,7]
 tkey[3,2,8] tkey[3,2,9] tkey[3,3,0] tkey[3,3,1] tkey[3,3,2] tkey[3,3,3]
 tkey[3,3,4] tkey[3,3,5] tkey[3,3,6] tkey[3,3,7] tkey[3,3,8] tkey[3,3,9]
 tbox[0,0,0] tbox[0,0,1] tbox[0,0,2] tbox[0,0,3] tbox[0,0,4] tbox[0,0,5]
 tbox[0,0,6] tbox[0,0,7] tbox[0,0,8] tbox[0,0,9] tbox[0,1,0] tbox[0,1,1]
 tbox[0,1,2] tbox[0,1,3] tbox[0,1,4] tbox[0,1,5] tbox[0,1,6] tbox[0,1,7]
 tbox[0,1,8] tbox[0,1,9] tbox[0,2,0] tbox[0,2,1] tbox[0,2,2] tbox[0,2,3]
 tbox[0,2,4] tbox[0,2,5] tbox[0,2,6] tbox[0,2,7] tbox[0,2,8] tbox[0,2,9]
 tbox[0,3,0] tbox[0,3,1] tbox[0,3,2] tbox[0,3,3] tbox[0,3,4] tbox[0,3,5]
 tbox[0,3,6] tbox[0,3,7] tbox[0,3,8] tbox[0,3,9] tbox[1,0,0] tbox[1,0,1]
 tbox[1,0,2] tbox[1,0,3] tbox[1,0,4] tbox[1,0,5] tbox[1,0,6] tbox[1,0,7]
 tbox[1,0,8] tbox[1,0,9] tbox[1,1,0] tbox[1,1,1] tbox[1,1,2] tbox[1,1,3]
 tbox[1,1,4] tbox[1,1,5] tbox[1,1,6] tbox[1,1,7] tbox[1,1,8] tbox[1,1,9]
 tbox[1,2,0] tbox[1,2,1] tbox[1,2,2] tbox[1,2,3] tbox[1,2,4] tbox[1,2,5]
 tbox[1,2,6] tbox[1,2,7] tbox[1,2,8] tbox[1,2,9] tbox[1,3,0] tbox[1,3,1]
 tbox[1,3,2] tbox[1,3,3] tbox[1,3,4] tbox[1,3,5] tbox[1,3,6] tbox[1,3,7]
 tbox[1,3,8] tbox[1,3,9] tbox[2,0,0] tbox[2,0,1] tbox[2,0,2] tbox[2,0,3]
 tbox[2,0,4] tbox[2,0,5] tbox[2,0,6] tbox[2,0,7] tbox[2,0,8] tbox[2,0,9]
 tbox[2,1,0] tbox[2,1,1] tbox[2,1,2] tbox[2,1,3] tbox[2,1,4] tbox[2,1,5]
 tbox[2,1,6] tbox[2,1,7] tbox[2,1,8] tbox[2,1,9] tbox[2,2,0] tbox[2,2,1]
 tbox[2,2,2] tbox[2,2,3] tbox[2,2,4] tbox[2,2,5] tbox[2,2,6] tbox[2,2,7]
 tbox[2,2,8] tbox[2,2,9] tbox[2,3,0] tbox[2,3,1] tbox[2,3,2] tbox[2,3,3]
 tbox[2,3,4] tbox[2,3,5] tbox[2,3,6] tbox[2,3,7] tbox[2,3,8] tbox[2,3,9]
 tbox[3,0,0] tbox[3,0,1] tbox[3,0,2] tbox[3,0,3] tbox[3,0,4] tbox[3,0,5]
 tbox[3,0,6] tbox[3,0,7] tbox[3,0,8] tbox[3,0,9] tbox[3,1,0] tbox[3,1,1]
 tbox[3,1,2] tbox[3,1,3] tbox[3,1,4] tbox[3,1,5] tbox[3,1,6] tbox[3,1,7]
 tbox[3,1,8] tbox[3,1,9] tbox[3,2,0] tbox[3,2,1] tbox[3,2,2] tbox[3,2,3]
 tbox[3,2,4] tbox[3,2,5] tbox[3,2,6] tbox[3,2,7] tbox[3,2,8] tbox[3,2,9]
 tbox[3,3,0] tbox[3,3,1] tbox[3,3,2] tbox[3,3,3] tbox[3,3,4] tbox[3,3,5]
 tbox[3,3,6] tbox[3,3,7] tbox[3,3,8] tbox[3,3,9] unkn[0,0,0] unkn[0,0,1]
 unkn[0,0,2] unkn[0,0,3] unkn[0,0,4] unkn[0,0,5] unkn[0,0,6] unkn[0,0,7]
 unkn[0,0,8] unkn[0,0,9] unkn[0,1,0] unkn[0,1,1] unkn[0,1,2] unkn[0,1,3]
 unkn[0,1,4] unkn[0,1,5] unkn[0,1,6] unkn[0,1,7] unkn[0,1,8] unkn[0,1,9]
 unkn[0,2,0] unkn[0,2,1] unkn[0,2,2] unkn[0,2,3] unkn[0,2,4] unkn[0,2,5]
 unkn[0,2,6] unkn[0,2,7] unkn[0,2,8] unkn[0,2,9] unkn[0,3,0] unkn[0,3,1]
 unkn[0,3,2] unkn[0,3,3] unkn[0,3,4] unkn[0,3,5] unkn[0,3,6] unkn[0,3,7]
 unkn[0,3,8] unkn[0,3,9] unkn[1,0,0] unkn[1,0,1] unkn[1,0,2] unkn[1,0,3]
 unkn[1,0,4] unkn[1,0,5] unkn[1,0,6] unkn[1,0,7] unkn[1,0,8] unkn[1,0,9]
 unkn[1,1,0] unkn[1,1,1] unkn[1,1,2] unkn[1,1,3] unkn[1,1,4] unkn[1,1,5]
 unkn[1,1,6] unkn[1,1,7] unkn[1,1,8] unkn[1,1,9] unkn[1,2,0] unkn[1,2,1]
 unkn[1,2,2] unkn[1,2,3] unkn[1,2,4] unkn[1,2,5] unkn[1,2,6] unkn[1,2,7]
 unkn[1,2,8] unkn[1,2,9] unkn[1,3,0] unkn[1,3,1] unkn[1,3,2] unkn[1,3,3]
 unkn[1,3,4] unkn[1,3,5] unkn[1,3,6] unkn[1,3,7] unkn[1,3,8] unkn[1,3,9]
 unkn[2,0,0] unkn[2,0,1] unkn[2,0,2] unkn[2,0,3] unkn[2,0,4] unkn[2,0,5]
 unkn[2,0,6] unkn[2,0,7] unkn[2,0,8] unkn[2,0,9] unkn[2,1,0] unkn[2,1,1]
 unkn[2,1,2] unkn[2,1,3] unkn[2,1,4] unkn[2,1,5] unkn[2,1,6] unkn[2,1,7]
 unkn[2,1,8] unkn[2,1,9] unkn[2,2,0] unkn[2,2,1] unkn[2,2,2] unkn[2,2,3]
 unkn[2,2,4] unkn[2,2,5] unkn[2,2,6] unkn[2,2,7] unkn[2,2,8] unkn[2,2,9]
 unkn[2,3,0] unkn[2,3,1] unkn[2,3,2] unkn[2,3,3] unkn[2,3,4] unkn[2,3,5]
 unkn[2,3,6] unkn[2,3,7] unkn[2,3,8] unkn[2,3,9] unkn[3,0,0] unkn[3,0,1]
 unkn[3,0,2] unkn[3,0,3] unkn[3,0,4] unkn[3,0,5] unkn[3,0,6] unkn[3,0,7]
 unkn[3,0,8] unkn[3,0,9] unkn[3,1,0] unkn[3,1,1] unkn[3,1,2] unkn[3,1,3]
 unkn[3,1,4] unkn[3,1,5] unkn[3,1,6] unkn[3,1,7] unkn[3,1,8] unkn[3,1,9]
 unkn[3,2,0] unkn[3,2,1] unkn[3,2,2] unkn[3,2,3] unkn[3,2,4] unkn[3,2,5]
 unkn[3,2,6] unkn[3,2,7] unkn[3,2,8] unkn[3,2,9] unkn[3,3,0] unkn[3,3,1]
 unkn[3,3,2] unkn[3,3,3] unkn[3,3,4] unkn[3,3,5] unkn[3,3,6] unkn[3,3,7]
 unkn[3,3,8] unkn[3,3,9] forg[0,0,0] forg[0,0,1] forg[0,0,2] forg[0,0,3]
 forg[0,0,4] forg[0,0,5] forg[0,0,6] forg[0,0,7] forg[0,0,8] forg[0,0,9]
 forg[0,1,0] forg[0,1,1] forg[0,1,2] forg[0,1,3] forg[0,1,4] forg[0,1,5]
 forg[0,1,6] forg[0,1,7] forg[0,1,8] forg[0,1,9] forg[0,2,0] forg[0,2,1]
 forg[0,2,2] forg[0,2,3] forg[0,2,4] forg[0,2,5] forg[0,2,6] forg[0,2,7]
 forg[0,2,8] forg[0,2,9] forg[0,3,0] forg[0,3,1] forg[0,3,2] forg[0,3,3]
 forg[0,3,4] forg[0,3,5] forg[0,3,6] forg[0,3,7] forg[0,3,8] forg[0,3,9]
 forg[1,0,0] forg[1,0,1] forg[1,0,2] forg[1,0,3] forg[1,0,4] forg[1,0,5]
 forg[1,0,6] forg[1,0,7] forg[1,0,8] forg[1,0,9] forg[1,1,0] forg[1,1,1]
 forg[1,1,2] forg[1,1,3] forg[1,1,4] forg[1,1,5] forg[1,1,6] forg[1,1,7]
 forg[1,1,8] forg[1,1,9] forg[1,2,0] forg[1,2,1] forg[1,2,2] forg[1,2,3]
 forg[1,2,4] forg[1,2,5] forg[1,2,6] forg[1,2,7] forg[1,2,8] forg[1,2,9]
 forg[1,3,0] forg[1,3,1] forg[1,3,2] forg[1,3,3] forg[1,3,4] forg[1,3,5]
 forg[1,3,6] forg[1,3,7] forg[1,3,8] forg[1,3,9] forg[2,0,0] forg[2,0,1]
 forg[2,0,2] forg[2,0,3] forg[2,0,4] forg[2,0,5] forg[2,0,6] forg[2,0,7]
 forg[2,0,8] forg[2,0,9] forg[2,1,0] forg[2,1,1] forg[2,1,2] forg[2,1,3]
 forg[2,1,4] forg[2,1,5] forg[2,1,6] forg[2,1,7] forg[2,1,8] forg[2,1,9]
 forg[2,2,0] forg[2,2,1] forg[2,2,2] forg[2,2,3] forg[2,2,4] forg[2,2,5]
 forg[2,2,6] forg[2,2,7] forg[2,2,8] forg[2,2,9] forg[2,3,0] forg[2,3,1]
 forg[2,3,2] forg[2,3,3] forg[2,3,4] forg[2,3,5] forg[2,3,6] forg[2,3,7]
 forg[2,3,8] forg[2,3,9] forg[3,0,0] forg[3,0,1] forg[3,0,2] forg[3,0,3]
 forg[3,0,4] forg[3,0,5] forg[3,0,6] forg[3,0,7] forg[3,0,8] forg[3,0,9]
 forg[3,1,0] forg[3,1,1] forg[3,1,2] forg[3,1,3] forg[3,1,4] forg[3,1,5]
 forg[3,1,6] forg[3,1,7] forg[3,1,8] forg[3,1,9] forg[3,2,0] forg[3,2,1]
 forg[3,2,2] forg[3,2,3] forg[3,2,4] forg[3,2,5] forg[3,2,6] forg[3,2,7]
 forg[3,2,8] forg[3,2,9] forg[3,3,0] forg[3,3,1] forg[3,3,2] forg[3,3,3]
 forg[3,3,4] forg[3,3,5] forg[3,3,6] forg[3,3,7] forg[3,3,8] forg[3,3,9]
 lane[0,0] lane[0,1] lane[0,2] lane[0,3] lane[1,0] lane[1,1] lane[1,2]
 lane[1,3] lane[2,0] lane[2,1] lane[2,2] lane[2,3] lane[3,0] lane[3,1]
 lane[3,2] lane[3,3]
Generals
 cost[0,0,0] cost[0,0,1] cost[0,0,2] cost[0,0,3] cost[0,0,4] cost[0,0,5]
 cost[0,0,6] cost[0,0,7] cost[0,0,8] cost[0,0,9] cost[0,1,0] cost[0,1,1]
 cost[0,1,2] cost[0,1,3] cost[0,1,4] cost[0,1,5] cost[0,1,6] cost[0,1,7]
 cost[0,1,8] cost[0,1,9] cost[0,2,0] cost[0,2,1] cost[0,2,2] cost[0,2,3]
 cost[0,2,4] cost[0,2,5] cost[0,2,6] cost[0,2,7] cost[0,2,8] cost[0,2,9]
 cost[0,3,0] cost[0,3,1] cost[0,3,2] cost[0,3,3] cost[0,3,4] cost[0,3,5]
 cost[0,3,6] cost[0,3,7] cost[0,3,8] cost[0,3,9] cost[1,0,0] cost[1,0,1]
 cost[1,0,2] cost[1,0,3] cost[1,0,4] cost[1,0,5] cost[1,0,6] cost[1,0,7]
 cost[1,0,8] cost[1,0,9] cost[1,1,0] cost[1,1,1] cost[1,1,2] cost[1,1,3]
 cost[1,1,4] cost[1,1,5] cost[1,1,6] cost[1,1,7] cost[1,1,8] cost[1,1,9]
 cost[1,2,0] cost[1,2,1] cost[1,2,2] cost[1,2,3] cost[1,2,4] cost[1,2,5]
 cost[1,2,6] cost[1,2,7] cost[1,2,8] cost[1,2,9] cost[1,3,0] cost[1,3,1]
 cost[1,3,2] cost[1,3,3] cost[1,3,4] cost[1,3,5] cost[1,3,6] cost[1,3,7]
 cost[1,3,8] cost[1,3,9] cost[2,0,0] cost[2,0,1] cost[2,0,2] cost[2,0,3]
 cost[2,0,4] cost[2,0,5] cost[2,0,6] cost[2,0,7] cost[2,0,8] cost[2,0,9]
 cost[2,1,0] cost[2,1,1] cost[2,1,2] cost[2,1,3] cost[2,1,4] cost[2,1,5]
 cost[2,1,6] cost[2,1,7] cost[2,1,8] cost[2,1,9] cost[2,2,0] cost[2,2,1]
 cost[2,2,2] cost[2,2,3] cost[2,2,4] cost[2,2,5] cost[2,2,6] cost[2,2,7]
 cost[2,2,8] cost[2,2,9] cost[2,3,0] cost[2,3,1] cost[2,3,2] cost[2,3,3]
 cost[2,3,4] cost[2,3,5] cost[2,3,6] cost[2,3,7] cost[2,3,8] cost[2,3,9]
 cost[3,0,0] cost[3,0,1] cost[3,0,2] cost[3,0,3] cost[3,0,4] cost[3,0,5]
 cost[3,0,6] cost[3,0,7] cost[3,0,8] cost[3,0,9] cost[3,1,0] cost[3,1,1]
 cost[3,1,2] cost[3,1,3] cost[3,1,4] cost[3,1,5] cost[3,1,6] cost[3,1,7]
 cost[3,1,8] cost[3,1,9] cost[3,2,0] cost[3,2,1] cost[3,2,2] cost[3,2,3]
 cost[3,2,4] cost[3,2,5] cost[3,2,6] cost[3,2,7] cost[3,2,8] cost[3,2,9]
 cost[3,3,0] cost[3,3,1] cost[3,3,2] cost[3,3,3] cost[3,3,4] cost[3,3,5]
 cost[3,3,6] cost[3,3,7] cost[3,3,8] cost[3,3,9]
End
