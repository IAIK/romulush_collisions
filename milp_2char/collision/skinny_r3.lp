\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 rkey[0,0,3] + 0 rkey[0,1,3] + 0 rkey[0,2,3] + 0 rkey[0,3,3]
   + 0 rkey[1,0,3] + 0 rkey[1,1,3] + 0 rkey[1,2,3] + 0 rkey[1,3,3]
   + 0 rkey[2,0,3] + 0 rkey[2,1,3] + 0 rkey[2,2,3] + 0 rkey[2,3,3]
   + 0 rkey[3,0,3] + 0 rkey[3,1,3] + 0 rkey[3,2,3] + 0 rkey[3,3,3]
   + 0 tkey[0,0,3] + 0 tkey[0,1,3] + 0 tkey[0,2,3] + 0 tkey[0,3,3]
   + 0 tkey[1,0,3] + 0 tkey[1,1,3] + 0 tkey[1,2,3] + 0 tkey[1,3,3]
   + 0 tkey[2,0,3] + 0 tkey[2,1,3] + 0 tkey[2,2,3] + 0 tkey[2,3,3]
   + 0 tkey[3,0,3] + 0 tkey[3,1,3] + 0 tkey[3,2,3] + 0 tkey[3,3,3]
   + cost[0,0,0] + cost[0,0,1] + cost[0,0,2] + 0 cost[0,0,3] + cost[0,1,0]
   + cost[0,1,1] + cost[0,1,2] + 0 cost[0,1,3] + cost[0,2,0] + cost[0,2,1]
   + cost[0,2,2] + 0 cost[0,2,3] + cost[0,3,0] + cost[0,3,1] + cost[0,3,2]
   + 0 cost[0,3,3] + cost[1,0,0] + cost[1,0,1] + cost[1,0,2]
   + 0 cost[1,0,3] + cost[1,1,0] + cost[1,1,1] + cost[1,1,2]
   + 0 cost[1,1,3] + cost[1,2,0] + cost[1,2,1] + cost[1,2,2]
   + 0 cost[1,2,3] + cost[1,3,0] + cost[1,3,1] + cost[1,3,2]
   + 0 cost[1,3,3] + cost[2,0,0] + cost[2,0,1] + cost[2,0,2]
   + 0 cost[2,0,3] + cost[2,1,0] + cost[2,1,1] + cost[2,1,2]
   + 0 cost[2,1,3] + cost[2,2,0] + cost[2,2,1] + cost[2,2,2]
   + 0 cost[2,2,3] + cost[2,3,0] + cost[2,3,1] + cost[2,3,2]
   + 0 cost[2,3,3] + cost[3,0,0] + cost[3,0,1] + cost[3,0,2]
   + 0 cost[3,0,3] + cost[3,1,0] + cost[3,1,1] + cost[3,1,2]
   + 0 cost[3,1,3] + cost[3,2,0] + cost[3,2,1] + cost[3,2,2]
   + 0 cost[3,2,3] + cost[3,3,0] + cost[3,3,1] + cost[3,3,2]
   + 0 cost[3,3,3]
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
 R144: tkey[0,0,0] - lane[0,0] <= 0
 R145: tkey[2,0,1] - lane[0,0] <= 0
 R146: tkey[0,2,2] - lane[0,0] <= 0
 R147: tkey[0,1,0] - lane[0,1] <= 0
 R148: tkey[2,1,1] - lane[0,1] <= 0
 R149: tkey[0,0,2] - lane[0,1] <= 0
 R150: tkey[0,2,0] - lane[0,2] <= 0
 R151: tkey[2,2,1] - lane[0,2] <= 0
 R152: tkey[1,0,2] - lane[0,2] <= 0
 R153: tkey[0,3,0] - lane[0,3] <= 0
 R154: tkey[2,3,1] - lane[0,3] <= 0
 R155: tkey[1,3,2] - lane[0,3] <= 0
 R156: tkey[1,0,0] - lane[1,0] <= 0
 R157: tkey[3,0,1] - lane[1,0] <= 0
 R158: tkey[1,2,2] - lane[1,0] <= 0
 R159: tkey[1,1,0] - lane[1,1] <= 0
 R160: tkey[3,1,1] - lane[1,1] <= 0
 R161: tkey[0,3,2] - lane[1,1] <= 0
 R162: tkey[1,2,0] - lane[1,2] <= 0
 R163: tkey[3,2,1] - lane[1,2] <= 0
 R164: tkey[1,1,2] - lane[1,2] <= 0
 R165: tkey[1,3,0] - lane[1,3] <= 0
 R166: tkey[3,3,1] - lane[1,3] <= 0
 R167: tkey[0,1,2] - lane[1,3] <= 0
 R168: tkey[2,0,0] - lane[2,0] <= 0
 R169: tkey[0,2,1] - lane[2,0] <= 0
 R170: tkey[2,2,2] - lane[2,0] <= 0
 R171: tkey[2,1,0] - lane[2,1] <= 0
 R172: tkey[0,0,1] - lane[2,1] <= 0
 R173: tkey[2,0,2] - lane[2,1] <= 0
 R174: tkey[2,2,0] - lane[2,2] <= 0
 R175: tkey[1,0,1] - lane[2,2] <= 0
 R176: tkey[3,0,2] - lane[2,2] <= 0
 R177: tkey[2,3,0] - lane[2,3] <= 0
 R178: tkey[1,3,1] - lane[2,3] <= 0
 R179: tkey[3,3,2] - lane[2,3] <= 0
 R180: tkey[3,0,0] - lane[3,0] <= 0
 R181: tkey[1,2,1] - lane[3,0] <= 0
 R182: tkey[3,2,2] - lane[3,0] <= 0
 R183: tkey[3,1,0] - lane[3,1] <= 0
 R184: tkey[0,3,1] - lane[3,1] <= 0
 R185: tkey[2,3,2] - lane[3,1] <= 0
 R186: tkey[3,2,0] - lane[3,2] <= 0
 R187: tkey[1,1,1] - lane[3,2] <= 0
 R188: tkey[3,1,2] - lane[3,2] <= 0
 R189: tkey[3,3,0] - lane[3,3] <= 0
 R190: tkey[0,1,1] - lane[3,3] <= 0
 R191: tkey[2,1,2] - lane[3,3] <= 0
 R192: - tkey[0,0,0] - tkey[0,2,2] - tkey[2,0,1] + 3 lane[0,0] <= 1
 R193: - tkey[0,0,2] - tkey[0,1,0] - tkey[2,1,1] + 3 lane[0,1] <= 1
 R194: - tkey[0,2,0] - tkey[1,0,2] - tkey[2,2,1] + 3 lane[0,2] <= 1
 R195: - tkey[0,3,0] - tkey[1,3,2] - tkey[2,3,1] + 3 lane[0,3] <= 1
 R196: - tkey[1,0,0] - tkey[1,2,2] - tkey[3,0,1] + 3 lane[1,0] <= 1
 R197: - tkey[0,3,2] - tkey[1,1,0] - tkey[3,1,1] + 3 lane[1,1] <= 1
 R198: - tkey[1,1,2] - tkey[1,2,0] - tkey[3,2,1] + 3 lane[1,2] <= 1
 R199: - tkey[0,1,2] - tkey[1,3,0] - tkey[3,3,1] + 3 lane[1,3] <= 1
 R200: - tkey[0,2,1] - tkey[2,0,0] - tkey[2,2,2] + 3 lane[2,0] <= 1
 R201: - tkey[0,0,1] - tkey[2,0,2] - tkey[2,1,0] + 3 lane[2,1] <= 1
 R202: - tkey[1,0,1] - tkey[2,2,0] - tkey[3,0,2] + 3 lane[2,2] <= 1
 R203: - tkey[1,3,1] - tkey[2,3,0] - tkey[3,3,2] + 3 lane[2,3] <= 1
 R204: - tkey[1,2,1] - tkey[3,0,0] - tkey[3,2,2] + 3 lane[3,0] <= 1
 R205: - tkey[0,3,1] - tkey[2,3,2] - tkey[3,1,0] + 3 lane[3,1] <= 1
 R206: - tkey[1,1,1] - tkey[3,1,2] - tkey[3,2,0] + 3 lane[3,2] <= 1
 R207: - tkey[0,1,1] - tkey[2,1,2] - tkey[3,3,0] + 3 lane[3,3] <= 1
 R208: - tbox[0,0,0] + forg[0,0,0] <= 0
 R209: sbox[0,0,0] - rkey[0,0,0] + tkey[0,0,0] >= 0
 R210: sbox[0,0,0] + rkey[0,0,0] - tkey[0,0,0] >= 0
 R211: - sbox[0,0,0] + rkey[0,0,0] + tkey[0,0,0] >= 0
 R212: - tbox[1,0,0] + forg[1,0,0] <= 0
 R213: sbox[1,0,0] - rkey[1,0,0] + tkey[1,0,0] >= 0
 R214: sbox[1,0,0] + rkey[1,0,0] - tkey[1,0,0] >= 0
 R215: - sbox[1,0,0] + rkey[1,0,0] + tkey[1,0,0] >= 0
 R216: - tbox[2,0,0] + forg[2,0,0] <= 0
 R217: - sbox[2,0,0] + rkey[2,0,0] = 0
 R218: - tbox[3,0,0] + forg[3,0,0] <= 0
 R219: - sbox[3,0,0] + rkey[3,0,0] = 0
 R220: - sbox[3,0,1] + rkey[0,0,0] + rkey[2,2,0] >= 0
 R221: sbox[3,0,1] + rkey[0,0,0] - rkey[2,2,0] >= 0
 R222: sbox[3,0,1] - rkey[0,0,0] + rkey[2,2,0] >= 0
 R223: - sbox[0,0,1] + sbox[3,0,1] + rkey[3,1,0] >= 0
 R224: sbox[0,0,1] + sbox[3,0,1] - rkey[3,1,0] >= 0
 R225: sbox[0,0,1] - sbox[3,0,1] + rkey[3,1,0] >= 0
 R226: - sbox[2,0,1] + rkey[1,3,0] + rkey[2,2,0] >= 0
 R227: sbox[2,0,1] + rkey[1,3,0] - rkey[2,2,0] >= 0
 R228: sbox[2,0,1] - rkey[1,3,0] + rkey[2,2,0] >= 0
 R229: sbox[1,0,1] - rkey[0,0,0] = 0
 R230: tbox[0,0,0] + tbox[2,2,0] - tbox[3,0,1] >= 0
 R231: tbox[0,0,0] - tbox[2,2,0] + tbox[3,0,1] >= 0
 R232: - tbox[0,0,0] + tbox[2,2,0] + tbox[3,0,1] >= 0
 R233: - tbox[0,0,1] + tbox[3,0,1] + tbox[3,1,0] >= 0
 R234: tbox[0,0,1] + tbox[3,0,1] - tbox[3,1,0] >= 0
 R235: tbox[0,0,1] - tbox[3,0,1] + tbox[3,1,0] >= 0
 R236: tbox[1,3,0] - tbox[2,0,1] + tbox[2,2,0] >= 0
 R237: tbox[1,3,0] + tbox[2,0,1] - tbox[2,2,0] >= 0
 R238: - tbox[1,3,0] + tbox[2,0,1] + tbox[2,2,0] >= 0
 R239: - tbox[0,0,0] + tbox[1,0,1] = 0
 R240: unkn[0,0,0] - unkn[3,0,1] + forg[3,0,1] <= 0
 R241: unkn[2,2,0] - unkn[3,0,1] + forg[3,0,1] <= 0
 R242: unkn[0,0,0] + unkn[2,2,0] - unkn[3,0,1] + forg[3,0,1] >= 0
 R243: - unkn[0,0,1] + unkn[3,0,1] + forg[0,0,1] - forg[3,0,1] <= 0
 R244: - unkn[0,0,1] + unkn[3,1,0] + forg[0,0,1] <= 0
 R245: - unkn[0,0,1] + unkn[3,0,1] + unkn[3,1,0] + forg[0,0,1]
   - forg[3,0,1] >= 0
 R246: unkn[1,3,0] - unkn[2,0,1] + forg[2,0,1] <= 0
 R247: - unkn[2,0,1] + unkn[2,2,0] + forg[2,0,1] <= 0
 R248: unkn[1,3,0] - unkn[2,0,1] + unkn[2,2,0] + forg[2,0,1] >= 0
 R249: unkn[0,0,0] - unkn[1,0,1] + forg[1,0,1] = 0
 R250: - tbox[0,1,0] + forg[0,1,0] <= 0
 R251: sbox[0,1,0] - rkey[0,1,0] + tkey[0,1,0] >= 0
 R252: sbox[0,1,0] + rkey[0,1,0] - tkey[0,1,0] >= 0
 R253: - sbox[0,1,0] + rkey[0,1,0] + tkey[0,1,0] >= 0
 R254: - tbox[1,1,0] + forg[1,1,0] <= 0
 R255: sbox[1,1,0] - rkey[1,1,0] + tkey[1,1,0] >= 0
 R256: sbox[1,1,0] + rkey[1,1,0] - tkey[1,1,0] >= 0
 R257: - sbox[1,1,0] + rkey[1,1,0] + tkey[1,1,0] >= 0
 R258: - tbox[2,1,0] + forg[2,1,0] <= 0
 R259: - sbox[2,1,0] + rkey[2,1,0] = 0
 R260: - tbox[3,1,0] + forg[3,1,0] <= 0
 R261: - sbox[3,1,0] + rkey[3,1,0] = 0
 R262: - sbox[3,1,1] + rkey[0,1,0] + rkey[2,3,0] >= 0
 R263: sbox[3,1,1] + rkey[0,1,0] - rkey[2,3,0] >= 0
 R264: sbox[3,1,1] - rkey[0,1,0] + rkey[2,3,0] >= 0
 R265: - sbox[0,1,1] + sbox[3,1,1] + rkey[3,2,0] >= 0
 R266: sbox[0,1,1] + sbox[3,1,1] - rkey[3,2,0] >= 0
 R267: sbox[0,1,1] - sbox[3,1,1] + rkey[3,2,0] >= 0
 R268: - sbox[2,1,1] + rkey[1,0,0] + rkey[2,3,0] >= 0
 R269: sbox[2,1,1] + rkey[1,0,0] - rkey[2,3,0] >= 0
 R270: sbox[2,1,1] - rkey[1,0,0] + rkey[2,3,0] >= 0
 R271: sbox[1,1,1] - rkey[0,1,0] = 0
 R272: tbox[0,1,0] + tbox[2,3,0] - tbox[3,1,1] >= 0
 R273: tbox[0,1,0] - tbox[2,3,0] + tbox[3,1,1] >= 0
 R274: - tbox[0,1,0] + tbox[2,3,0] + tbox[3,1,1] >= 0
 R275: - tbox[0,1,1] + tbox[3,1,1] + tbox[3,2,0] >= 0
 R276: tbox[0,1,1] + tbox[3,1,1] - tbox[3,2,0] >= 0
 R277: tbox[0,1,1] - tbox[3,1,1] + tbox[3,2,0] >= 0
 R278: tbox[1,0,0] - tbox[2,1,1] + tbox[2,3,0] >= 0
 R279: tbox[1,0,0] + tbox[2,1,1] - tbox[2,3,0] >= 0
 R280: - tbox[1,0,0] + tbox[2,1,1] + tbox[2,3,0] >= 0
 R281: - tbox[0,1,0] + tbox[1,1,1] = 0
 R282: unkn[0,1,0] - unkn[3,1,1] + forg[3,1,1] <= 0
 R283: unkn[2,3,0] - unkn[3,1,1] + forg[3,1,1] <= 0
 R284: unkn[0,1,0] + unkn[2,3,0] - unkn[3,1,1] + forg[3,1,1] >= 0
 R285: - unkn[0,1,1] + unkn[3,1,1] + forg[0,1,1] - forg[3,1,1] <= 0
 R286: - unkn[0,1,1] + unkn[3,2,0] + forg[0,1,1] <= 0
 R287: - unkn[0,1,1] + unkn[3,1,1] + unkn[3,2,0] + forg[0,1,1]
   - forg[3,1,1] >= 0
 R288: unkn[1,0,0] - unkn[2,1,1] + forg[2,1,1] <= 0
 R289: - unkn[2,1,1] + unkn[2,3,0] + forg[2,1,1] <= 0
 R290: unkn[1,0,0] - unkn[2,1,1] + unkn[2,3,0] + forg[2,1,1] >= 0
 R291: unkn[0,1,0] - unkn[1,1,1] + forg[1,1,1] = 0
 R292: - tbox[0,2,0] + forg[0,2,0] <= 0
 R293: sbox[0,2,0] - rkey[0,2,0] + tkey[0,2,0] >= 0
 R294: sbox[0,2,0] + rkey[0,2,0] - tkey[0,2,0] >= 0
 R295: - sbox[0,2,0] + rkey[0,2,0] + tkey[0,2,0] >= 0
 R296: - tbox[1,2,0] + forg[1,2,0] <= 0
 R297: sbox[1,2,0] - rkey[1,2,0] + tkey[1,2,0] >= 0
 R298: sbox[1,2,0] + rkey[1,2,0] - tkey[1,2,0] >= 0
 R299: - sbox[1,2,0] + rkey[1,2,0] + tkey[1,2,0] >= 0
 R300: - tbox[2,2,0] + forg[2,2,0] <= 0
 R301: - sbox[2,2,0] + rkey[2,2,0] = 0
 R302: - tbox[3,2,0] + forg[3,2,0] <= 0
 R303: - sbox[3,2,0] + rkey[3,2,0] = 0
 R304: - sbox[3,2,1] + rkey[0,2,0] + rkey[2,0,0] >= 0
 R305: sbox[3,2,1] + rkey[0,2,0] - rkey[2,0,0] >= 0
 R306: sbox[3,2,1] - rkey[0,2,0] + rkey[2,0,0] >= 0
 R307: - sbox[0,2,1] + sbox[3,2,1] + rkey[3,3,0] >= 0
 R308: sbox[0,2,1] + sbox[3,2,1] - rkey[3,3,0] >= 0
 R309: sbox[0,2,1] - sbox[3,2,1] + rkey[3,3,0] >= 0
 R310: - sbox[2,2,1] + rkey[1,1,0] + rkey[2,0,0] >= 0
 R311: sbox[2,2,1] + rkey[1,1,0] - rkey[2,0,0] >= 0
 R312: sbox[2,2,1] - rkey[1,1,0] + rkey[2,0,0] >= 0
 R313: sbox[1,2,1] - rkey[0,2,0] = 0
 R314: tbox[0,2,0] + tbox[2,0,0] - tbox[3,2,1] >= 0
 R315: tbox[0,2,0] - tbox[2,0,0] + tbox[3,2,1] >= 0
 R316: - tbox[0,2,0] + tbox[2,0,0] + tbox[3,2,1] >= 0
 R317: - tbox[0,2,1] + tbox[3,2,1] + tbox[3,3,0] >= 0
 R318: tbox[0,2,1] + tbox[3,2,1] - tbox[3,3,0] >= 0
 R319: tbox[0,2,1] - tbox[3,2,1] + tbox[3,3,0] >= 0
 R320: tbox[1,1,0] + tbox[2,0,0] - tbox[2,2,1] >= 0
 R321: tbox[1,1,0] - tbox[2,0,0] + tbox[2,2,1] >= 0
 R322: - tbox[1,1,0] + tbox[2,0,0] + tbox[2,2,1] >= 0
 R323: - tbox[0,2,0] + tbox[1,2,1] = 0
 R324: unkn[0,2,0] - unkn[3,2,1] + forg[3,2,1] <= 0
 R325: unkn[2,0,0] - unkn[3,2,1] + forg[3,2,1] <= 0
 R326: unkn[0,2,0] + unkn[2,0,0] - unkn[3,2,1] + forg[3,2,1] >= 0
 R327: - unkn[0,2,1] + unkn[3,2,1] + forg[0,2,1] - forg[3,2,1] <= 0
 R328: - unkn[0,2,1] + unkn[3,3,0] + forg[0,2,1] <= 0
 R329: - unkn[0,2,1] + unkn[3,2,1] + unkn[3,3,0] + forg[0,2,1]
   - forg[3,2,1] >= 0
 R330: unkn[1,1,0] - unkn[2,2,1] + forg[2,2,1] <= 0
 R331: unkn[2,0,0] - unkn[2,2,1] + forg[2,2,1] <= 0
 R332: unkn[1,1,0] + unkn[2,0,0] - unkn[2,2,1] + forg[2,2,1] >= 0
 R333: unkn[0,2,0] - unkn[1,2,1] + forg[1,2,1] = 0
 R334: - tbox[0,3,0] + forg[0,3,0] <= 0
 R335: sbox[0,3,0] - rkey[0,3,0] + tkey[0,3,0] >= 0
 R336: sbox[0,3,0] + rkey[0,3,0] - tkey[0,3,0] >= 0
 R337: - sbox[0,3,0] + rkey[0,3,0] + tkey[0,3,0] >= 0
 R338: - tbox[1,3,0] + forg[1,3,0] <= 0
 R339: sbox[1,3,0] - rkey[1,3,0] + tkey[1,3,0] >= 0
 R340: sbox[1,3,0] + rkey[1,3,0] - tkey[1,3,0] >= 0
 R341: - sbox[1,3,0] + rkey[1,3,0] + tkey[1,3,0] >= 0
 R342: - tbox[2,3,0] + forg[2,3,0] <= 0
 R343: - sbox[2,3,0] + rkey[2,3,0] = 0
 R344: - tbox[3,3,0] + forg[3,3,0] <= 0
 R345: - sbox[3,3,0] + rkey[3,3,0] = 0
 R346: - sbox[3,3,1] + rkey[0,3,0] + rkey[2,1,0] >= 0
 R347: sbox[3,3,1] + rkey[0,3,0] - rkey[2,1,0] >= 0
 R348: sbox[3,3,1] - rkey[0,3,0] + rkey[2,1,0] >= 0
 R349: - sbox[0,3,1] + sbox[3,3,1] + rkey[3,0,0] >= 0
 R350: sbox[0,3,1] + sbox[3,3,1] - rkey[3,0,0] >= 0
 R351: sbox[0,3,1] - sbox[3,3,1] + rkey[3,0,0] >= 0
 R352: - sbox[2,3,1] + rkey[1,2,0] + rkey[2,1,0] >= 0
 R353: sbox[2,3,1] + rkey[1,2,0] - rkey[2,1,0] >= 0
 R354: sbox[2,3,1] - rkey[1,2,0] + rkey[2,1,0] >= 0
 R355: sbox[1,3,1] - rkey[0,3,0] = 0
 R356: tbox[0,3,0] + tbox[2,1,0] - tbox[3,3,1] >= 0
 R357: tbox[0,3,0] - tbox[2,1,0] + tbox[3,3,1] >= 0
 R358: - tbox[0,3,0] + tbox[2,1,0] + tbox[3,3,1] >= 0
 R359: - tbox[0,3,1] + tbox[3,0,0] + tbox[3,3,1] >= 0
 R360: tbox[0,3,1] - tbox[3,0,0] + tbox[3,3,1] >= 0
 R361: tbox[0,3,1] + tbox[3,0,0] - tbox[3,3,1] >= 0
 R362: tbox[1,2,0] + tbox[2,1,0] - tbox[2,3,1] >= 0
 R363: tbox[1,2,0] - tbox[2,1,0] + tbox[2,3,1] >= 0
 R364: - tbox[1,2,0] + tbox[2,1,0] + tbox[2,3,1] >= 0
 R365: - tbox[0,3,0] + tbox[1,3,1] = 0
 R366: unkn[0,3,0] - unkn[3,3,1] + forg[3,3,1] <= 0
 R367: unkn[2,1,0] - unkn[3,3,1] + forg[3,3,1] <= 0
 R368: unkn[0,3,0] + unkn[2,1,0] - unkn[3,3,1] + forg[3,3,1] >= 0
 R369: - unkn[0,3,1] + unkn[3,3,1] + forg[0,3,1] - forg[3,3,1] <= 0
 R370: - unkn[0,3,1] + unkn[3,0,0] + forg[0,3,1] <= 0
 R371: - unkn[0,3,1] + unkn[3,0,0] + unkn[3,3,1] + forg[0,3,1]
   - forg[3,3,1] >= 0
 R372: unkn[1,2,0] - unkn[2,3,1] + forg[2,3,1] <= 0
 R373: unkn[2,1,0] - unkn[2,3,1] + forg[2,3,1] <= 0
 R374: unkn[1,2,0] + unkn[2,1,0] - unkn[2,3,1] + forg[2,3,1] >= 0
 R375: unkn[0,3,0] - unkn[1,3,1] + forg[1,3,1] = 0
 R376: - tbox[0,0,1] + forg[0,0,1] <= 0
 R377: sbox[0,0,1] - rkey[0,0,1] + tkey[0,0,1] >= 0
 R378: sbox[0,0,1] + rkey[0,0,1] - tkey[0,0,1] >= 0
 R379: - sbox[0,0,1] + rkey[0,0,1] + tkey[0,0,1] >= 0
 R380: - tbox[1,0,1] + forg[1,0,1] <= 0
 R381: sbox[1,0,1] - rkey[1,0,1] + tkey[1,0,1] >= 0
 R382: sbox[1,0,1] + rkey[1,0,1] - tkey[1,0,1] >= 0
 R383: - sbox[1,0,1] + rkey[1,0,1] + tkey[1,0,1] >= 0
 R384: - tbox[2,0,1] + forg[2,0,1] <= 0
 R385: - sbox[2,0,1] + rkey[2,0,1] = 0
 R386: - tbox[3,0,1] + forg[3,0,1] <= 0
 R387: - sbox[3,0,1] + rkey[3,0,1] = 0
 R388: - sbox[3,0,2] + rkey[0,0,1] + rkey[2,2,1] >= 0
 R389: sbox[3,0,2] + rkey[0,0,1] - rkey[2,2,1] >= 0
 R390: sbox[3,0,2] - rkey[0,0,1] + rkey[2,2,1] >= 0
 R391: - sbox[0,0,2] + sbox[3,0,2] + rkey[3,1,1] >= 0
 R392: sbox[0,0,2] + sbox[3,0,2] - rkey[3,1,1] >= 0
 R393: sbox[0,0,2] - sbox[3,0,2] + rkey[3,1,1] >= 0
 R394: - sbox[2,0,2] + rkey[1,3,1] + rkey[2,2,1] >= 0
 R395: sbox[2,0,2] + rkey[1,3,1] - rkey[2,2,1] >= 0
 R396: sbox[2,0,2] - rkey[1,3,1] + rkey[2,2,1] >= 0
 R397: sbox[1,0,2] - rkey[0,0,1] = 0
 R398: tbox[0,0,1] + tbox[2,2,1] - tbox[3,0,2] >= 0
 R399: tbox[0,0,1] - tbox[2,2,1] + tbox[3,0,2] >= 0
 R400: - tbox[0,0,1] + tbox[2,2,1] + tbox[3,0,2] >= 0
 R401: - tbox[0,0,2] + tbox[3,0,2] + tbox[3,1,1] >= 0
 R402: tbox[0,0,2] + tbox[3,0,2] - tbox[3,1,1] >= 0
 R403: tbox[0,0,2] - tbox[3,0,2] + tbox[3,1,1] >= 0
 R404: tbox[1,3,1] - tbox[2,0,2] + tbox[2,2,1] >= 0
 R405: tbox[1,3,1] + tbox[2,0,2] - tbox[2,2,1] >= 0
 R406: - tbox[1,3,1] + tbox[2,0,2] + tbox[2,2,1] >= 0
 R407: - tbox[0,0,1] + tbox[1,0,2] = 0
 R408: unkn[0,0,1] - unkn[3,0,2] + forg[3,0,2] <= 0
 R409: unkn[2,2,1] - unkn[3,0,2] + forg[3,0,2] <= 0
 R410: unkn[0,0,1] + unkn[2,2,1] - unkn[3,0,2] + forg[3,0,2] >= 0
 R411: - unkn[0,0,2] + unkn[3,0,2] + forg[0,0,2] - forg[3,0,2] <= 0
 R412: - unkn[0,0,2] + unkn[3,1,1] + forg[0,0,2] <= 0
 R413: - unkn[0,0,2] + unkn[3,0,2] + unkn[3,1,1] + forg[0,0,2]
   - forg[3,0,2] >= 0
 R414: unkn[1,3,1] - unkn[2,0,2] + forg[2,0,2] <= 0
 R415: - unkn[2,0,2] + unkn[2,2,1] + forg[2,0,2] <= 0
 R416: unkn[1,3,1] - unkn[2,0,2] + unkn[2,2,1] + forg[2,0,2] >= 0
 R417: unkn[0,0,1] - unkn[1,0,2] + forg[1,0,2] = 0
 R418: - tbox[0,1,1] + forg[0,1,1] <= 0
 R419: sbox[0,1,1] - rkey[0,1,1] + tkey[0,1,1] >= 0
 R420: sbox[0,1,1] + rkey[0,1,1] - tkey[0,1,1] >= 0
 R421: - sbox[0,1,1] + rkey[0,1,1] + tkey[0,1,1] >= 0
 R422: - tbox[1,1,1] + forg[1,1,1] <= 0
 R423: sbox[1,1,1] - rkey[1,1,1] + tkey[1,1,1] >= 0
 R424: sbox[1,1,1] + rkey[1,1,1] - tkey[1,1,1] >= 0
 R425: - sbox[1,1,1] + rkey[1,1,1] + tkey[1,1,1] >= 0
 R426: - tbox[2,1,1] + forg[2,1,1] <= 0
 R427: - sbox[2,1,1] + rkey[2,1,1] = 0
 R428: - tbox[3,1,1] + forg[3,1,1] <= 0
 R429: - sbox[3,1,1] + rkey[3,1,1] = 0
 R430: - sbox[3,1,2] + rkey[0,1,1] + rkey[2,3,1] >= 0
 R431: sbox[3,1,2] + rkey[0,1,1] - rkey[2,3,1] >= 0
 R432: sbox[3,1,2] - rkey[0,1,1] + rkey[2,3,1] >= 0
 R433: - sbox[0,1,2] + sbox[3,1,2] + rkey[3,2,1] >= 0
 R434: sbox[0,1,2] + sbox[3,1,2] - rkey[3,2,1] >= 0
 R435: sbox[0,1,2] - sbox[3,1,2] + rkey[3,2,1] >= 0
 R436: - sbox[2,1,2] + rkey[1,0,1] + rkey[2,3,1] >= 0
 R437: sbox[2,1,2] + rkey[1,0,1] - rkey[2,3,1] >= 0
 R438: sbox[2,1,2] - rkey[1,0,1] + rkey[2,3,1] >= 0
 R439: sbox[1,1,2] - rkey[0,1,1] = 0
 R440: tbox[0,1,1] + tbox[2,3,1] - tbox[3,1,2] >= 0
 R441: tbox[0,1,1] - tbox[2,3,1] + tbox[3,1,2] >= 0
 R442: - tbox[0,1,1] + tbox[2,3,1] + tbox[3,1,2] >= 0
 R443: - tbox[0,1,2] + tbox[3,1,2] + tbox[3,2,1] >= 0
 R444: tbox[0,1,2] + tbox[3,1,2] - tbox[3,2,1] >= 0
 R445: tbox[0,1,2] - tbox[3,1,2] + tbox[3,2,1] >= 0
 R446: tbox[1,0,1] - tbox[2,1,2] + tbox[2,3,1] >= 0
 R447: tbox[1,0,1] + tbox[2,1,2] - tbox[2,3,1] >= 0
 R448: - tbox[1,0,1] + tbox[2,1,2] + tbox[2,3,1] >= 0
 R449: - tbox[0,1,1] + tbox[1,1,2] = 0
 R450: unkn[0,1,1] - unkn[3,1,2] + forg[3,1,2] <= 0
 R451: unkn[2,3,1] - unkn[3,1,2] + forg[3,1,2] <= 0
 R452: unkn[0,1,1] + unkn[2,3,1] - unkn[3,1,2] + forg[3,1,2] >= 0
 R453: - unkn[0,1,2] + unkn[3,1,2] + forg[0,1,2] - forg[3,1,2] <= 0
 R454: - unkn[0,1,2] + unkn[3,2,1] + forg[0,1,2] <= 0
 R455: - unkn[0,1,2] + unkn[3,1,2] + unkn[3,2,1] + forg[0,1,2]
   - forg[3,1,2] >= 0
 R456: unkn[1,0,1] - unkn[2,1,2] + forg[2,1,2] <= 0
 R457: - unkn[2,1,2] + unkn[2,3,1] + forg[2,1,2] <= 0
 R458: unkn[1,0,1] - unkn[2,1,2] + unkn[2,3,1] + forg[2,1,2] >= 0
 R459: unkn[0,1,1] - unkn[1,1,2] + forg[1,1,2] = 0
 R460: - tbox[0,2,1] + forg[0,2,1] <= 0
 R461: sbox[0,2,1] - rkey[0,2,1] + tkey[0,2,1] >= 0
 R462: sbox[0,2,1] + rkey[0,2,1] - tkey[0,2,1] >= 0
 R463: - sbox[0,2,1] + rkey[0,2,1] + tkey[0,2,1] >= 0
 R464: - tbox[1,2,1] + forg[1,2,1] <= 0
 R465: sbox[1,2,1] - rkey[1,2,1] + tkey[1,2,1] >= 0
 R466: sbox[1,2,1] + rkey[1,2,1] - tkey[1,2,1] >= 0
 R467: - sbox[1,2,1] + rkey[1,2,1] + tkey[1,2,1] >= 0
 R468: - tbox[2,2,1] + forg[2,2,1] <= 0
 R469: - sbox[2,2,1] + rkey[2,2,1] = 0
 R470: - tbox[3,2,1] + forg[3,2,1] <= 0
 R471: - sbox[3,2,1] + rkey[3,2,1] = 0
 R472: - sbox[3,2,2] + rkey[0,2,1] + rkey[2,0,1] >= 0
 R473: sbox[3,2,2] + rkey[0,2,1] - rkey[2,0,1] >= 0
 R474: sbox[3,2,2] - rkey[0,2,1] + rkey[2,0,1] >= 0
 R475: - sbox[0,2,2] + sbox[3,2,2] + rkey[3,3,1] >= 0
 R476: sbox[0,2,2] + sbox[3,2,2] - rkey[3,3,1] >= 0
 R477: sbox[0,2,2] - sbox[3,2,2] + rkey[3,3,1] >= 0
 R478: - sbox[2,2,2] + rkey[1,1,1] + rkey[2,0,1] >= 0
 R479: sbox[2,2,2] + rkey[1,1,1] - rkey[2,0,1] >= 0
 R480: sbox[2,2,2] - rkey[1,1,1] + rkey[2,0,1] >= 0
 R481: sbox[1,2,2] - rkey[0,2,1] = 0
 R482: tbox[0,2,1] + tbox[2,0,1] - tbox[3,2,2] >= 0
 R483: tbox[0,2,1] - tbox[2,0,1] + tbox[3,2,2] >= 0
 R484: - tbox[0,2,1] + tbox[2,0,1] + tbox[3,2,2] >= 0
 R485: - tbox[0,2,2] + tbox[3,2,2] + tbox[3,3,1] >= 0
 R486: tbox[0,2,2] + tbox[3,2,2] - tbox[3,3,1] >= 0
 R487: tbox[0,2,2] - tbox[3,2,2] + tbox[3,3,1] >= 0
 R488: tbox[1,1,1] + tbox[2,0,1] - tbox[2,2,2] >= 0
 R489: tbox[1,1,1] - tbox[2,0,1] + tbox[2,2,2] >= 0
 R490: - tbox[1,1,1] + tbox[2,0,1] + tbox[2,2,2] >= 0
 R491: - tbox[0,2,1] + tbox[1,2,2] = 0
 R492: unkn[0,2,1] - unkn[3,2,2] + forg[3,2,2] <= 0
 R493: unkn[2,0,1] - unkn[3,2,2] + forg[3,2,2] <= 0
 R494: unkn[0,2,1] + unkn[2,0,1] - unkn[3,2,2] + forg[3,2,2] >= 0
 R495: - unkn[0,2,2] + unkn[3,2,2] + forg[0,2,2] - forg[3,2,2] <= 0
 R496: - unkn[0,2,2] + unkn[3,3,1] + forg[0,2,2] <= 0
 R497: - unkn[0,2,2] + unkn[3,2,2] + unkn[3,3,1] + forg[0,2,2]
   - forg[3,2,2] >= 0
 R498: unkn[1,1,1] - unkn[2,2,2] + forg[2,2,2] <= 0
 R499: unkn[2,0,1] - unkn[2,2,2] + forg[2,2,2] <= 0
 R500: unkn[1,1,1] + unkn[2,0,1] - unkn[2,2,2] + forg[2,2,2] >= 0
 R501: unkn[0,2,1] - unkn[1,2,2] + forg[1,2,2] = 0
 R502: - tbox[0,3,1] + forg[0,3,1] <= 0
 R503: sbox[0,3,1] - rkey[0,3,1] + tkey[0,3,1] >= 0
 R504: sbox[0,3,1] + rkey[0,3,1] - tkey[0,3,1] >= 0
 R505: - sbox[0,3,1] + rkey[0,3,1] + tkey[0,3,1] >= 0
 R506: - tbox[1,3,1] + forg[1,3,1] <= 0
 R507: sbox[1,3,1] - rkey[1,3,1] + tkey[1,3,1] >= 0
 R508: sbox[1,3,1] + rkey[1,3,1] - tkey[1,3,1] >= 0
 R509: - sbox[1,3,1] + rkey[1,3,1] + tkey[1,3,1] >= 0
 R510: - tbox[2,3,1] + forg[2,3,1] <= 0
 R511: - sbox[2,3,1] + rkey[2,3,1] = 0
 R512: - tbox[3,3,1] + forg[3,3,1] <= 0
 R513: - sbox[3,3,1] + rkey[3,3,1] = 0
 R514: - sbox[3,3,2] + rkey[0,3,1] + rkey[2,1,1] >= 0
 R515: sbox[3,3,2] + rkey[0,3,1] - rkey[2,1,1] >= 0
 R516: sbox[3,3,2] - rkey[0,3,1] + rkey[2,1,1] >= 0
 R517: - sbox[0,3,2] + sbox[3,3,2] + rkey[3,0,1] >= 0
 R518: sbox[0,3,2] + sbox[3,3,2] - rkey[3,0,1] >= 0
 R519: sbox[0,3,2] - sbox[3,3,2] + rkey[3,0,1] >= 0
 R520: - sbox[2,3,2] + rkey[1,2,1] + rkey[2,1,1] >= 0
 R521: sbox[2,3,2] + rkey[1,2,1] - rkey[2,1,1] >= 0
 R522: sbox[2,3,2] - rkey[1,2,1] + rkey[2,1,1] >= 0
 R523: sbox[1,3,2] - rkey[0,3,1] = 0
 R524: tbox[0,3,1] + tbox[2,1,1] - tbox[3,3,2] >= 0
 R525: tbox[0,3,1] - tbox[2,1,1] + tbox[3,3,2] >= 0
 R526: - tbox[0,3,1] + tbox[2,1,1] + tbox[3,3,2] >= 0
 R527: - tbox[0,3,2] + tbox[3,0,1] + tbox[3,3,2] >= 0
 R528: tbox[0,3,2] - tbox[3,0,1] + tbox[3,3,2] >= 0
 R529: tbox[0,3,2] + tbox[3,0,1] - tbox[3,3,2] >= 0
 R530: tbox[1,2,1] + tbox[2,1,1] - tbox[2,3,2] >= 0
 R531: tbox[1,2,1] - tbox[2,1,1] + tbox[2,3,2] >= 0
 R532: - tbox[1,2,1] + tbox[2,1,1] + tbox[2,3,2] >= 0
 R533: - tbox[0,3,1] + tbox[1,3,2] = 0
 R534: unkn[0,3,1] - unkn[3,3,2] + forg[3,3,2] <= 0
 R535: unkn[2,1,1] - unkn[3,3,2] + forg[3,3,2] <= 0
 R536: unkn[0,3,1] + unkn[2,1,1] - unkn[3,3,2] + forg[3,3,2] >= 0
 R537: - unkn[0,3,2] + unkn[3,3,2] + forg[0,3,2] - forg[3,3,2] <= 0
 R538: - unkn[0,3,2] + unkn[3,0,1] + forg[0,3,2] <= 0
 R539: - unkn[0,3,2] + unkn[3,0,1] + unkn[3,3,2] + forg[0,3,2]
   - forg[3,3,2] >= 0
 R540: unkn[1,2,1] - unkn[2,3,2] + forg[2,3,2] <= 0
 R541: unkn[2,1,1] - unkn[2,3,2] + forg[2,3,2] <= 0
 R542: unkn[1,2,1] + unkn[2,1,1] - unkn[2,3,2] + forg[2,3,2] >= 0
 R543: unkn[0,3,1] - unkn[1,3,2] + forg[1,3,2] = 0
 R544: - tbox[0,0,2] + forg[0,0,2] <= 0
 R545: sbox[0,0,2] - rkey[0,0,2] + tkey[0,0,2] >= 0
 R546: sbox[0,0,2] + rkey[0,0,2] - tkey[0,0,2] >= 0
 R547: - sbox[0,0,2] + rkey[0,0,2] + tkey[0,0,2] >= 0
 R548: - tbox[1,0,2] + forg[1,0,2] <= 0
 R549: sbox[1,0,2] - rkey[1,0,2] + tkey[1,0,2] >= 0
 R550: sbox[1,0,2] + rkey[1,0,2] - tkey[1,0,2] >= 0
 R551: - sbox[1,0,2] + rkey[1,0,2] + tkey[1,0,2] >= 0
 R552: - tbox[2,0,2] + forg[2,0,2] <= 0
 R553: - sbox[2,0,2] + rkey[2,0,2] = 0
 R554: - tbox[3,0,2] + forg[3,0,2] <= 0
 R555: - sbox[3,0,2] + rkey[3,0,2] = 0
 R556: - sbox[3,0,3] + rkey[0,0,2] + rkey[2,2,2] >= 0
 R557: sbox[3,0,3] + rkey[0,0,2] - rkey[2,2,2] >= 0
 R558: sbox[3,0,3] - rkey[0,0,2] + rkey[2,2,2] >= 0
 R559: - sbox[0,0,3] + sbox[3,0,3] + rkey[3,1,2] >= 0
 R560: sbox[0,0,3] + sbox[3,0,3] - rkey[3,1,2] >= 0
 R561: sbox[0,0,3] - sbox[3,0,3] + rkey[3,1,2] >= 0
 R562: - sbox[2,0,3] + rkey[1,3,2] + rkey[2,2,2] >= 0
 R563: sbox[2,0,3] + rkey[1,3,2] - rkey[2,2,2] >= 0
 R564: sbox[2,0,3] - rkey[1,3,2] + rkey[2,2,2] >= 0
 R565: sbox[1,0,3] - rkey[0,0,2] = 0
 R566: tbox[0,0,2] + tbox[2,2,2] - tbox[3,0,3] >= 0
 R567: tbox[0,0,2] - tbox[2,2,2] + tbox[3,0,3] >= 0
 R568: - tbox[0,0,2] + tbox[2,2,2] + tbox[3,0,3] >= 0
 R569: - tbox[0,0,3] + tbox[3,0,3] + tbox[3,1,2] >= 0
 R570: tbox[0,0,3] + tbox[3,0,3] - tbox[3,1,2] >= 0
 R571: tbox[0,0,3] - tbox[3,0,3] + tbox[3,1,2] >= 0
 R572: tbox[1,3,2] - tbox[2,0,3] + tbox[2,2,2] >= 0
 R573: tbox[1,3,2] + tbox[2,0,3] - tbox[2,2,2] >= 0
 R574: - tbox[1,3,2] + tbox[2,0,3] + tbox[2,2,2] >= 0
 R575: - tbox[0,0,2] + tbox[1,0,3] = 0
 R576: unkn[0,0,2] - unkn[3,0,3] + forg[3,0,3] <= 0
 R577: unkn[2,2,2] - unkn[3,0,3] + forg[3,0,3] <= 0
 R578: unkn[0,0,2] + unkn[2,2,2] - unkn[3,0,3] + forg[3,0,3] >= 0
 R579: - unkn[0,0,3] + unkn[3,0,3] + forg[0,0,3] - forg[3,0,3] <= 0
 R580: - unkn[0,0,3] + unkn[3,1,2] + forg[0,0,3] <= 0
 R581: - unkn[0,0,3] + unkn[3,0,3] + unkn[3,1,2] + forg[0,0,3]
   - forg[3,0,3] >= 0
 R582: unkn[1,3,2] - unkn[2,0,3] + forg[2,0,3] <= 0
 R583: - unkn[2,0,3] + unkn[2,2,2] + forg[2,0,3] <= 0
 R584: unkn[1,3,2] - unkn[2,0,3] + unkn[2,2,2] + forg[2,0,3] >= 0
 R585: unkn[0,0,2] - unkn[1,0,3] + forg[1,0,3] = 0
 R586: - tbox[0,1,2] + forg[0,1,2] <= 0
 R587: sbox[0,1,2] - rkey[0,1,2] + tkey[0,1,2] >= 0
 R588: sbox[0,1,2] + rkey[0,1,2] - tkey[0,1,2] >= 0
 R589: - sbox[0,1,2] + rkey[0,1,2] + tkey[0,1,2] >= 0
 R590: - tbox[1,1,2] + forg[1,1,2] <= 0
 R591: sbox[1,1,2] - rkey[1,1,2] + tkey[1,1,2] >= 0
 R592: sbox[1,1,2] + rkey[1,1,2] - tkey[1,1,2] >= 0
 R593: - sbox[1,1,2] + rkey[1,1,2] + tkey[1,1,2] >= 0
 R594: - tbox[2,1,2] + forg[2,1,2] <= 0
 R595: - sbox[2,1,2] + rkey[2,1,2] = 0
 R596: - tbox[3,1,2] + forg[3,1,2] <= 0
 R597: - sbox[3,1,2] + rkey[3,1,2] = 0
 R598: - sbox[3,1,3] + rkey[0,1,2] + rkey[2,3,2] >= 0
 R599: sbox[3,1,3] + rkey[0,1,2] - rkey[2,3,2] >= 0
 R600: sbox[3,1,3] - rkey[0,1,2] + rkey[2,3,2] >= 0
 R601: - sbox[0,1,3] + sbox[3,1,3] + rkey[3,2,2] >= 0
 R602: sbox[0,1,3] + sbox[3,1,3] - rkey[3,2,2] >= 0
 R603: sbox[0,1,3] - sbox[3,1,3] + rkey[3,2,2] >= 0
 R604: - sbox[2,1,3] + rkey[1,0,2] + rkey[2,3,2] >= 0
 R605: sbox[2,1,3] + rkey[1,0,2] - rkey[2,3,2] >= 0
 R606: sbox[2,1,3] - rkey[1,0,2] + rkey[2,3,2] >= 0
 R607: sbox[1,1,3] - rkey[0,1,2] = 0
 R608: tbox[0,1,2] + tbox[2,3,2] - tbox[3,1,3] >= 0
 R609: tbox[0,1,2] - tbox[2,3,2] + tbox[3,1,3] >= 0
 R610: - tbox[0,1,2] + tbox[2,3,2] + tbox[3,1,3] >= 0
 R611: - tbox[0,1,3] + tbox[3,1,3] + tbox[3,2,2] >= 0
 R612: tbox[0,1,3] + tbox[3,1,3] - tbox[3,2,2] >= 0
 R613: tbox[0,1,3] - tbox[3,1,3] + tbox[3,2,2] >= 0
 R614: tbox[1,0,2] - tbox[2,1,3] + tbox[2,3,2] >= 0
 R615: tbox[1,0,2] + tbox[2,1,3] - tbox[2,3,2] >= 0
 R616: - tbox[1,0,2] + tbox[2,1,3] + tbox[2,3,2] >= 0
 R617: - tbox[0,1,2] + tbox[1,1,3] = 0
 R618: unkn[0,1,2] - unkn[3,1,3] + forg[3,1,3] <= 0
 R619: unkn[2,3,2] - unkn[3,1,3] + forg[3,1,3] <= 0
 R620: unkn[0,1,2] + unkn[2,3,2] - unkn[3,1,3] + forg[3,1,3] >= 0
 R621: - unkn[0,1,3] + unkn[3,1,3] + forg[0,1,3] - forg[3,1,3] <= 0
 R622: - unkn[0,1,3] + unkn[3,2,2] + forg[0,1,3] <= 0
 R623: - unkn[0,1,3] + unkn[3,1,3] + unkn[3,2,2] + forg[0,1,3]
   - forg[3,1,3] >= 0
 R624: unkn[1,0,2] - unkn[2,1,3] + forg[2,1,3] <= 0
 R625: - unkn[2,1,3] + unkn[2,3,2] + forg[2,1,3] <= 0
 R626: unkn[1,0,2] - unkn[2,1,3] + unkn[2,3,2] + forg[2,1,3] >= 0
 R627: unkn[0,1,2] - unkn[1,1,3] + forg[1,1,3] = 0
 R628: - tbox[0,2,2] + forg[0,2,2] <= 0
 R629: sbox[0,2,2] - rkey[0,2,2] + tkey[0,2,2] >= 0
 R630: sbox[0,2,2] + rkey[0,2,2] - tkey[0,2,2] >= 0
 R631: - sbox[0,2,2] + rkey[0,2,2] + tkey[0,2,2] >= 0
 R632: - tbox[1,2,2] + forg[1,2,2] <= 0
 R633: sbox[1,2,2] - rkey[1,2,2] + tkey[1,2,2] >= 0
 R634: sbox[1,2,2] + rkey[1,2,2] - tkey[1,2,2] >= 0
 R635: - sbox[1,2,2] + rkey[1,2,2] + tkey[1,2,2] >= 0
 R636: - tbox[2,2,2] + forg[2,2,2] <= 0
 R637: - sbox[2,2,2] + rkey[2,2,2] = 0
 R638: - tbox[3,2,2] + forg[3,2,2] <= 0
 R639: - sbox[3,2,2] + rkey[3,2,2] = 0
 R640: - sbox[3,2,3] + rkey[0,2,2] + rkey[2,0,2] >= 0
 R641: sbox[3,2,3] + rkey[0,2,2] - rkey[2,0,2] >= 0
 R642: sbox[3,2,3] - rkey[0,2,2] + rkey[2,0,2] >= 0
 R643: - sbox[0,2,3] + sbox[3,2,3] + rkey[3,3,2] >= 0
 R644: sbox[0,2,3] + sbox[3,2,3] - rkey[3,3,2] >= 0
 R645: sbox[0,2,3] - sbox[3,2,3] + rkey[3,3,2] >= 0
 R646: - sbox[2,2,3] + rkey[1,1,2] + rkey[2,0,2] >= 0
 R647: sbox[2,2,3] + rkey[1,1,2] - rkey[2,0,2] >= 0
 R648: sbox[2,2,3] - rkey[1,1,2] + rkey[2,0,2] >= 0
 R649: sbox[1,2,3] - rkey[0,2,2] = 0
 R650: tbox[0,2,2] + tbox[2,0,2] - tbox[3,2,3] >= 0
 R651: tbox[0,2,2] - tbox[2,0,2] + tbox[3,2,3] >= 0
 R652: - tbox[0,2,2] + tbox[2,0,2] + tbox[3,2,3] >= 0
 R653: - tbox[0,2,3] + tbox[3,2,3] + tbox[3,3,2] >= 0
 R654: tbox[0,2,3] + tbox[3,2,3] - tbox[3,3,2] >= 0
 R655: tbox[0,2,3] - tbox[3,2,3] + tbox[3,3,2] >= 0
 R656: tbox[1,1,2] + tbox[2,0,2] - tbox[2,2,3] >= 0
 R657: tbox[1,1,2] - tbox[2,0,2] + tbox[2,2,3] >= 0
 R658: - tbox[1,1,2] + tbox[2,0,2] + tbox[2,2,3] >= 0
 R659: - tbox[0,2,2] + tbox[1,2,3] = 0
 R660: unkn[0,2,2] - unkn[3,2,3] + forg[3,2,3] <= 0
 R661: unkn[2,0,2] - unkn[3,2,3] + forg[3,2,3] <= 0
 R662: unkn[0,2,2] + unkn[2,0,2] - unkn[3,2,3] + forg[3,2,3] >= 0
 R663: - unkn[0,2,3] + unkn[3,2,3] + forg[0,2,3] - forg[3,2,3] <= 0
 R664: - unkn[0,2,3] + unkn[3,3,2] + forg[0,2,3] <= 0
 R665: - unkn[0,2,3] + unkn[3,2,3] + unkn[3,3,2] + forg[0,2,3]
   - forg[3,2,3] >= 0
 R666: unkn[1,1,2] - unkn[2,2,3] + forg[2,2,3] <= 0
 R667: unkn[2,0,2] - unkn[2,2,3] + forg[2,2,3] <= 0
 R668: unkn[1,1,2] + unkn[2,0,2] - unkn[2,2,3] + forg[2,2,3] >= 0
 R669: unkn[0,2,2] - unkn[1,2,3] + forg[1,2,3] = 0
 R670: - tbox[0,3,2] + forg[0,3,2] <= 0
 R671: sbox[0,3,2] - rkey[0,3,2] + tkey[0,3,2] >= 0
 R672: sbox[0,3,2] + rkey[0,3,2] - tkey[0,3,2] >= 0
 R673: - sbox[0,3,2] + rkey[0,3,2] + tkey[0,3,2] >= 0
 R674: - tbox[1,3,2] + forg[1,3,2] <= 0
 R675: sbox[1,3,2] - rkey[1,3,2] + tkey[1,3,2] >= 0
 R676: sbox[1,3,2] + rkey[1,3,2] - tkey[1,3,2] >= 0
 R677: - sbox[1,3,2] + rkey[1,3,2] + tkey[1,3,2] >= 0
 R678: - tbox[2,3,2] + forg[2,3,2] <= 0
 R679: - sbox[2,3,2] + rkey[2,3,2] = 0
 R680: - tbox[3,3,2] + forg[3,3,2] <= 0
 R681: - sbox[3,3,2] + rkey[3,3,2] = 0
 R682: - sbox[3,3,3] + rkey[0,3,2] + rkey[2,1,2] >= 0
 R683: sbox[3,3,3] + rkey[0,3,2] - rkey[2,1,2] >= 0
 R684: sbox[3,3,3] - rkey[0,3,2] + rkey[2,1,2] >= 0
 R685: - sbox[0,3,3] + sbox[3,3,3] + rkey[3,0,2] >= 0
 R686: sbox[0,3,3] + sbox[3,3,3] - rkey[3,0,2] >= 0
 R687: sbox[0,3,3] - sbox[3,3,3] + rkey[3,0,2] >= 0
 R688: - sbox[2,3,3] + rkey[1,2,2] + rkey[2,1,2] >= 0
 R689: sbox[2,3,3] + rkey[1,2,2] - rkey[2,1,2] >= 0
 R690: sbox[2,3,3] - rkey[1,2,2] + rkey[2,1,2] >= 0
 R691: sbox[1,3,3] - rkey[0,3,2] = 0
 R692: tbox[0,3,2] + tbox[2,1,2] - tbox[3,3,3] >= 0
 R693: tbox[0,3,2] - tbox[2,1,2] + tbox[3,3,3] >= 0
 R694: - tbox[0,3,2] + tbox[2,1,2] + tbox[3,3,3] >= 0
 R695: - tbox[0,3,3] + tbox[3,0,2] + tbox[3,3,3] >= 0
 R696: tbox[0,3,3] - tbox[3,0,2] + tbox[3,3,3] >= 0
 R697: tbox[0,3,3] + tbox[3,0,2] - tbox[3,3,3] >= 0
 R698: tbox[1,2,2] + tbox[2,1,2] - tbox[2,3,3] >= 0
 R699: tbox[1,2,2] - tbox[2,1,2] + tbox[2,3,3] >= 0
 R700: - tbox[1,2,2] + tbox[2,1,2] + tbox[2,3,3] >= 0
 R701: - tbox[0,3,2] + tbox[1,3,3] = 0
 R702: unkn[0,3,2] - unkn[3,3,3] + forg[3,3,3] <= 0
 R703: unkn[2,1,2] - unkn[3,3,3] + forg[3,3,3] <= 0
 R704: unkn[0,3,2] + unkn[2,1,2] - unkn[3,3,3] + forg[3,3,3] >= 0
 R705: - unkn[0,3,3] + unkn[3,3,3] + forg[0,3,3] - forg[3,3,3] <= 0
 R706: - unkn[0,3,3] + unkn[3,0,2] + forg[0,3,3] <= 0
 R707: - unkn[0,3,3] + unkn[3,0,2] + unkn[3,3,3] + forg[0,3,3]
   - forg[3,3,3] >= 0
 R708: unkn[1,2,2] - unkn[2,3,3] + forg[2,3,3] <= 0
 R709: unkn[2,1,2] - unkn[2,3,3] + forg[2,3,3] <= 0
 R710: unkn[1,2,2] + unkn[2,1,2] - unkn[2,3,3] + forg[2,3,3] >= 0
 R711: unkn[0,3,2] - unkn[1,3,3] + forg[1,3,3] = 0
 R712: sbox[0,0,0] + sbox[0,1,0] + sbox[0,2,0] + sbox[0,3,0] + sbox[1,0,0]
   + sbox[1,1,0] + sbox[1,2,0] + sbox[1,3,0] + sbox[2,0,0] + sbox[2,1,0]
   + sbox[2,2,0] + sbox[2,3,0] + sbox[3,0,0] + sbox[3,1,0] + sbox[3,2,0]
   + sbox[3,3,0] + tkey[0,0,0] + tkey[0,1,0] + tkey[0,2,0] + tkey[0,3,0]
   + tkey[1,0,0] + tkey[1,1,0] + tkey[1,2,0] + tkey[1,3,0] + tkey[2,0,0]
   + tkey[2,1,0] + tkey[2,2,0] + tkey[2,3,0] + tkey[3,0,0] + tkey[3,1,0]
   + tkey[3,2,0] + tkey[3,3,0] >= 1
 R713: sbox[0,0,1] + sbox[0,1,1] + sbox[0,2,1] + sbox[0,3,1] + sbox[1,0,1]
   + sbox[1,1,1] + sbox[1,2,1] + sbox[1,3,1] + sbox[2,0,1] + sbox[2,1,1]
   + sbox[2,2,1] + sbox[2,3,1] + sbox[3,0,1] + sbox[3,1,1] + sbox[3,2,1]
   + sbox[3,3,1] + tkey[0,0,1] + tkey[0,1,1] + tkey[0,2,1] + tkey[0,3,1]
   + tkey[1,0,1] + tkey[1,1,1] + tkey[1,2,1] + tkey[1,3,1] + tkey[2,0,1]
   + tkey[2,1,1] + tkey[2,2,1] + tkey[2,3,1] + tkey[3,0,1] + tkey[3,1,1]
   + tkey[3,2,1] + tkey[3,3,1] >= 1
 R714: sbox[0,0,2] + sbox[0,1,2] + sbox[0,2,2] + sbox[0,3,2] + sbox[1,0,2]
   + sbox[1,1,2] + sbox[1,2,2] + sbox[1,3,2] + sbox[2,0,2] + sbox[2,1,2]
   + sbox[2,2,2] + sbox[2,3,2] + sbox[3,0,2] + sbox[3,1,2] + sbox[3,2,2]
   + sbox[3,3,2] + tkey[0,0,2] + tkey[0,1,2] + tkey[0,2,2] + tkey[0,3,2]
   + tkey[1,0,2] + tkey[1,1,2] + tkey[1,2,2] + tkey[1,3,2] + tkey[2,0,2]
   + tkey[2,1,2] + tkey[2,2,2] + tkey[2,3,2] + tkey[3,0,2] + tkey[3,1,2]
   + tkey[3,2,2] + tkey[3,3,2] >= 1
 R715: sbox[0,0,0] + sbox[0,1,0] + sbox[0,2,0] + sbox[0,3,0] + sbox[1,0,0]
   + sbox[1,1,0] + sbox[1,2,0] + sbox[1,3,0] + sbox[2,0,0] + sbox[2,1,0]
   + sbox[2,2,0] + sbox[2,3,0] + sbox[3,0,0] + sbox[3,1,0] + sbox[3,2,0]
   + sbox[3,3,0] = 0
 R716: sbox[0,0,3] + sbox[0,1,3] + sbox[0,2,3] + sbox[0,3,3] + sbox[1,0,3]
   + sbox[1,1,3] + sbox[1,2,3] + sbox[1,3,3] + sbox[2,0,3] + sbox[2,1,3]
   + sbox[2,2,3] + sbox[2,3,3] + sbox[3,0,3] + sbox[3,1,3] + sbox[3,2,3]
   + sbox[3,3,3] = 0
 R717: tbox[0,0,0] = 1
 R718: unkn[0,0,0] - forg[0,0,0] = 0
 R719: tbox[0,1,0] = 0
 R720: unkn[0,1,0] - forg[0,1,0] = 0
 R721: tbox[0,2,0] = 0
 R722: unkn[0,2,0] - forg[0,2,0] = 0
 R723: tbox[0,3,0] = 0
 R724: unkn[0,3,0] - forg[0,3,0] = 0
 R725: tbox[1,0,0] = 0
 R726: unkn[1,0,0] - forg[1,0,0] = 0
 R727: tbox[1,1,0] = 0
 R728: unkn[1,1,0] - forg[1,1,0] = 0
 R729: tbox[1,2,0] = 0
 R730: unkn[1,2,0] - forg[1,2,0] = 0
 R731: tbox[1,3,0] = 0
 R732: unkn[1,3,0] - forg[1,3,0] = 0
 R733: tbox[2,0,0] = 0
 R734: unkn[2,0,0] - forg[2,0,0] = 0
 R735: tbox[2,1,0] = 0
 R736: unkn[2,1,0] - forg[2,1,0] = 0
 R737: tbox[2,2,0] = 0
 R738: unkn[2,2,0] - forg[2,2,0] = 0
 R739: tbox[2,3,0] = 0
 R740: unkn[2,3,0] - forg[2,3,0] = 0
 R741: tbox[3,0,0] = 0
 R742: unkn[3,0,0] - forg[3,0,0] = 0
 R743: tbox[3,1,0] = 0
 R744: unkn[3,1,0] - forg[3,1,0] = 0
 R745: tbox[3,2,0] = 0
 R746: unkn[3,2,0] - forg[3,2,0] = 0
 R747: tbox[3,3,0] = 0
 R748: unkn[3,3,0] - forg[3,3,0] = 0
Bounds
 cost[0,0,0] <= 2
 cost[0,0,1] <= 2
 cost[0,0,2] <= 2
 cost[0,0,3] <= 2
 cost[0,1,0] <= 2
 cost[0,1,1] <= 2
 cost[0,1,2] <= 2
 cost[0,1,3] <= 2
 cost[0,2,0] <= 2
 cost[0,2,1] <= 2
 cost[0,2,2] <= 2
 cost[0,2,3] <= 2
 cost[0,3,0] <= 2
 cost[0,3,1] <= 2
 cost[0,3,2] <= 2
 cost[0,3,3] <= 2
 cost[1,0,0] <= 2
 cost[1,0,1] <= 2
 cost[1,0,2] <= 2
 cost[1,0,3] <= 2
 cost[1,1,0] <= 2
 cost[1,1,1] <= 2
 cost[1,1,2] <= 2
 cost[1,1,3] <= 2
 cost[1,2,0] <= 2
 cost[1,2,1] <= 2
 cost[1,2,2] <= 2
 cost[1,2,3] <= 2
 cost[1,3,0] <= 2
 cost[1,3,1] <= 2
 cost[1,3,2] <= 2
 cost[1,3,3] <= 2
 cost[2,0,0] <= 2
 cost[2,0,1] <= 2
 cost[2,0,2] <= 2
 cost[2,0,3] <= 2
 cost[2,1,0] <= 2
 cost[2,1,1] <= 2
 cost[2,1,2] <= 2
 cost[2,1,3] <= 2
 cost[2,2,0] <= 2
 cost[2,2,1] <= 2
 cost[2,2,2] <= 2
 cost[2,2,3] <= 2
 cost[2,3,0] <= 2
 cost[2,3,1] <= 2
 cost[2,3,2] <= 2
 cost[2,3,3] <= 2
 cost[3,0,0] <= 2
 cost[3,0,1] <= 2
 cost[3,0,2] <= 2
 cost[3,0,3] <= 2
 cost[3,1,0] <= 2
 cost[3,1,1] <= 2
 cost[3,1,2] <= 2
 cost[3,1,3] <= 2
 cost[3,2,0] <= 2
 cost[3,2,1] <= 2
 cost[3,2,2] <= 2
 cost[3,2,3] <= 2
 cost[3,3,0] <= 2
 cost[3,3,1] <= 2
 cost[3,3,2] <= 2
 cost[3,3,3] <= 2
Binaries
 sbox[0,0,0] sbox[0,0,1] sbox[0,0,2] sbox[0,0,3] sbox[0,1,0] sbox[0,1,1]
 sbox[0,1,2] sbox[0,1,3] sbox[0,2,0] sbox[0,2,1] sbox[0,2,2] sbox[0,2,3]
 sbox[0,3,0] sbox[0,3,1] sbox[0,3,2] sbox[0,3,3] sbox[1,0,0] sbox[1,0,1]
 sbox[1,0,2] sbox[1,0,3] sbox[1,1,0] sbox[1,1,1] sbox[1,1,2] sbox[1,1,3]
 sbox[1,2,0] sbox[1,2,1] sbox[1,2,2] sbox[1,2,3] sbox[1,3,0] sbox[1,3,1]
 sbox[1,3,2] sbox[1,3,3] sbox[2,0,0] sbox[2,0,1] sbox[2,0,2] sbox[2,0,3]
 sbox[2,1,0] sbox[2,1,1] sbox[2,1,2] sbox[2,1,3] sbox[2,2,0] sbox[2,2,1]
 sbox[2,2,2] sbox[2,2,3] sbox[2,3,0] sbox[2,3,1] sbox[2,3,2] sbox[2,3,3]
 sbox[3,0,0] sbox[3,0,1] sbox[3,0,2] sbox[3,0,3] sbox[3,1,0] sbox[3,1,1]
 sbox[3,1,2] sbox[3,1,3] sbox[3,2,0] sbox[3,2,1] sbox[3,2,2] sbox[3,2,3]
 sbox[3,3,0] sbox[3,3,1] sbox[3,3,2] sbox[3,3,3] rkey[0,0,0] rkey[0,0,1]
 rkey[0,0,2] rkey[0,0,3] rkey[0,1,0] rkey[0,1,1] rkey[0,1,2] rkey[0,1,3]
 rkey[0,2,0] rkey[0,2,1] rkey[0,2,2] rkey[0,2,3] rkey[0,3,0] rkey[0,3,1]
 rkey[0,3,2] rkey[0,3,3] rkey[1,0,0] rkey[1,0,1] rkey[1,0,2] rkey[1,0,3]
 rkey[1,1,0] rkey[1,1,1] rkey[1,1,2] rkey[1,1,3] rkey[1,2,0] rkey[1,2,1]
 rkey[1,2,2] rkey[1,2,3] rkey[1,3,0] rkey[1,3,1] rkey[1,3,2] rkey[1,3,3]
 rkey[2,0,0] rkey[2,0,1] rkey[2,0,2] rkey[2,0,3] rkey[2,1,0] rkey[2,1,1]
 rkey[2,1,2] rkey[2,1,3] rkey[2,2,0] rkey[2,2,1] rkey[2,2,2] rkey[2,2,3]
 rkey[2,3,0] rkey[2,3,1] rkey[2,3,2] rkey[2,3,3] rkey[3,0,0] rkey[3,0,1]
 rkey[3,0,2] rkey[3,0,3] rkey[3,1,0] rkey[3,1,1] rkey[3,1,2] rkey[3,1,3]
 rkey[3,2,0] rkey[3,2,1] rkey[3,2,2] rkey[3,2,3] rkey[3,3,0] rkey[3,3,1]
 rkey[3,3,2] rkey[3,3,3] tkey[0,0,0] tkey[0,0,1] tkey[0,0,2] tkey[0,0,3]
 tkey[0,1,0] tkey[0,1,1] tkey[0,1,2] tkey[0,1,3] tkey[0,2,0] tkey[0,2,1]
 tkey[0,2,2] tkey[0,2,3] tkey[0,3,0] tkey[0,3,1] tkey[0,3,2] tkey[0,3,3]
 tkey[1,0,0] tkey[1,0,1] tkey[1,0,2] tkey[1,0,3] tkey[1,1,0] tkey[1,1,1]
 tkey[1,1,2] tkey[1,1,3] tkey[1,2,0] tkey[1,2,1] tkey[1,2,2] tkey[1,2,3]
 tkey[1,3,0] tkey[1,3,1] tkey[1,3,2] tkey[1,3,3] tkey[2,0,0] tkey[2,0,1]
 tkey[2,0,2] tkey[2,0,3] tkey[2,1,0] tkey[2,1,1] tkey[2,1,2] tkey[2,1,3]
 tkey[2,2,0] tkey[2,2,1] tkey[2,2,2] tkey[2,2,3] tkey[2,3,0] tkey[2,3,1]
 tkey[2,3,2] tkey[2,3,3] tkey[3,0,0] tkey[3,0,1] tkey[3,0,2] tkey[3,0,3]
 tkey[3,1,0] tkey[3,1,1] tkey[3,1,2] tkey[3,1,3] tkey[3,2,0] tkey[3,2,1]
 tkey[3,2,2] tkey[3,2,3] tkey[3,3,0] tkey[3,3,1] tkey[3,3,2] tkey[3,3,3]
 tbox[0,0,0] tbox[0,0,1] tbox[0,0,2] tbox[0,0,3] tbox[0,1,0] tbox[0,1,1]
 tbox[0,1,2] tbox[0,1,3] tbox[0,2,0] tbox[0,2,1] tbox[0,2,2] tbox[0,2,3]
 tbox[0,3,0] tbox[0,3,1] tbox[0,3,2] tbox[0,3,3] tbox[1,0,0] tbox[1,0,1]
 tbox[1,0,2] tbox[1,0,3] tbox[1,1,0] tbox[1,1,1] tbox[1,1,2] tbox[1,1,3]
 tbox[1,2,0] tbox[1,2,1] tbox[1,2,2] tbox[1,2,3] tbox[1,3,0] tbox[1,3,1]
 tbox[1,3,2] tbox[1,3,3] tbox[2,0,0] tbox[2,0,1] tbox[2,0,2] tbox[2,0,3]
 tbox[2,1,0] tbox[2,1,1] tbox[2,1,2] tbox[2,1,3] tbox[2,2,0] tbox[2,2,1]
 tbox[2,2,2] tbox[2,2,3] tbox[2,3,0] tbox[2,3,1] tbox[2,3,2] tbox[2,3,3]
 tbox[3,0,0] tbox[3,0,1] tbox[3,0,2] tbox[3,0,3] tbox[3,1,0] tbox[3,1,1]
 tbox[3,1,2] tbox[3,1,3] tbox[3,2,0] tbox[3,2,1] tbox[3,2,2] tbox[3,2,3]
 tbox[3,3,0] tbox[3,3,1] tbox[3,3,2] tbox[3,3,3] unkn[0,0,0] unkn[0,0,1]
 unkn[0,0,2] unkn[0,0,3] unkn[0,1,0] unkn[0,1,1] unkn[0,1,2] unkn[0,1,3]
 unkn[0,2,0] unkn[0,2,1] unkn[0,2,2] unkn[0,2,3] unkn[0,3,0] unkn[0,3,1]
 unkn[0,3,2] unkn[0,3,3] unkn[1,0,0] unkn[1,0,1] unkn[1,0,2] unkn[1,0,3]
 unkn[1,1,0] unkn[1,1,1] unkn[1,1,2] unkn[1,1,3] unkn[1,2,0] unkn[1,2,1]
 unkn[1,2,2] unkn[1,2,3] unkn[1,3,0] unkn[1,3,1] unkn[1,3,2] unkn[1,3,3]
 unkn[2,0,0] unkn[2,0,1] unkn[2,0,2] unkn[2,0,3] unkn[2,1,0] unkn[2,1,1]
 unkn[2,1,2] unkn[2,1,3] unkn[2,2,0] unkn[2,2,1] unkn[2,2,2] unkn[2,2,3]
 unkn[2,3,0] unkn[2,3,1] unkn[2,3,2] unkn[2,3,3] unkn[3,0,0] unkn[3,0,1]
 unkn[3,0,2] unkn[3,0,3] unkn[3,1,0] unkn[3,1,1] unkn[3,1,2] unkn[3,1,3]
 unkn[3,2,0] unkn[3,2,1] unkn[3,2,2] unkn[3,2,3] unkn[3,3,0] unkn[3,3,1]
 unkn[3,3,2] unkn[3,3,3] forg[0,0,0] forg[0,0,1] forg[0,0,2] forg[0,0,3]
 forg[0,1,0] forg[0,1,1] forg[0,1,2] forg[0,1,3] forg[0,2,0] forg[0,2,1]
 forg[0,2,2] forg[0,2,3] forg[0,3,0] forg[0,3,1] forg[0,3,2] forg[0,3,3]
 forg[1,0,0] forg[1,0,1] forg[1,0,2] forg[1,0,3] forg[1,1,0] forg[1,1,1]
 forg[1,1,2] forg[1,1,3] forg[1,2,0] forg[1,2,1] forg[1,2,2] forg[1,2,3]
 forg[1,3,0] forg[1,3,1] forg[1,3,2] forg[1,3,3] forg[2,0,0] forg[2,0,1]
 forg[2,0,2] forg[2,0,3] forg[2,1,0] forg[2,1,1] forg[2,1,2] forg[2,1,3]
 forg[2,2,0] forg[2,2,1] forg[2,2,2] forg[2,2,3] forg[2,3,0] forg[2,3,1]
 forg[2,3,2] forg[2,3,3] forg[3,0,0] forg[3,0,1] forg[3,0,2] forg[3,0,3]
 forg[3,1,0] forg[3,1,1] forg[3,1,2] forg[3,1,3] forg[3,2,0] forg[3,2,1]
 forg[3,2,2] forg[3,2,3] forg[3,3,0] forg[3,3,1] forg[3,3,2] forg[3,3,3]
 lane[0,0] lane[0,1] lane[0,2] lane[0,3] lane[1,0] lane[1,1] lane[1,2]
 lane[1,3] lane[2,0] lane[2,1] lane[2,2] lane[2,3] lane[3,0] lane[3,1]
 lane[3,2] lane[3,3]
Generals
 cost[0,0,0] cost[0,0,1] cost[0,0,2] cost[0,0,3] cost[0,1,0] cost[0,1,1]
 cost[0,1,2] cost[0,1,3] cost[0,2,0] cost[0,2,1] cost[0,2,2] cost[0,2,3]
 cost[0,3,0] cost[0,3,1] cost[0,3,2] cost[0,3,3] cost[1,0,0] cost[1,0,1]
 cost[1,0,2] cost[1,0,3] cost[1,1,0] cost[1,1,1] cost[1,1,2] cost[1,1,3]
 cost[1,2,0] cost[1,2,1] cost[1,2,2] cost[1,2,3] cost[1,3,0] cost[1,3,1]
 cost[1,3,2] cost[1,3,3] cost[2,0,0] cost[2,0,1] cost[2,0,2] cost[2,0,3]
 cost[2,1,0] cost[2,1,1] cost[2,1,2] cost[2,1,3] cost[2,2,0] cost[2,2,1]
 cost[2,2,2] cost[2,2,3] cost[2,3,0] cost[2,3,1] cost[2,3,2] cost[2,3,3]
 cost[3,0,0] cost[3,0,1] cost[3,0,2] cost[3,0,3] cost[3,1,0] cost[3,1,1]
 cost[3,1,2] cost[3,1,3] cost[3,2,0] cost[3,2,1] cost[3,2,2] cost[3,2,3]
 cost[3,3,0] cost[3,3,1] cost[3,3,2] cost[3,3,3]
End
