\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 rkey[0,0,2] + 0 rkey[0,1,2] + 0 rkey[0,2,2] + 0 rkey[0,3,2]
   + 0 rkey[1,0,2] + 0 rkey[1,1,2] + 0 rkey[1,2,2] + 0 rkey[1,3,2]
   + 0 rkey[2,0,2] + 0 rkey[2,1,2] + 0 rkey[2,2,2] + 0 rkey[2,3,2]
   + 0 rkey[3,0,2] + 0 rkey[3,1,2] + 0 rkey[3,2,2] + 0 rkey[3,3,2]
   + 0 tkey[0,0,2] + 0 tkey[0,1,2] + 0 tkey[0,2,2] + 0 tkey[0,3,2]
   + 0 tkey[1,0,2] + 0 tkey[1,1,2] + 0 tkey[1,2,2] + 0 tkey[1,3,2]
   + 0 tkey[2,0,2] + 0 tkey[2,1,2] + 0 tkey[2,2,2] + 0 tkey[2,3,2]
   + 0 tkey[3,0,2] + 0 tkey[3,1,2] + 0 tkey[3,2,2] + 0 tkey[3,3,2]
   + cost[0,0,0] + cost[0,0,1] + 0 cost[0,0,2] + cost[0,1,0] + cost[0,1,1]
   + 0 cost[0,1,2] + cost[0,2,0] + cost[0,2,1] + 0 cost[0,2,2]
   + cost[0,3,0] + cost[0,3,1] + 0 cost[0,3,2] + cost[1,0,0] + cost[1,0,1]
   + 0 cost[1,0,2] + cost[1,1,0] + cost[1,1,1] + 0 cost[1,1,2]
   + cost[1,2,0] + cost[1,2,1] + 0 cost[1,2,2] + cost[1,3,0] + cost[1,3,1]
   + 0 cost[1,3,2] + cost[2,0,0] + cost[2,0,1] + 0 cost[2,0,2]
   + cost[2,1,0] + cost[2,1,1] + 0 cost[2,1,2] + cost[2,2,0] + cost[2,2,1]
   + 0 cost[2,2,2] + cost[2,3,0] + cost[2,3,1] + 0 cost[2,3,2]
   + cost[3,0,0] + cost[3,0,1] + 0 cost[3,0,2] + cost[3,1,0] + cost[3,1,1]
   + 0 cost[3,1,2] + cost[3,2,0] + cost[3,2,1] + 0 cost[3,2,2]
   + cost[3,3,0] + cost[3,3,1] + 0 cost[3,3,2]
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
 R96: tkey[0,0,0] - lane[0,0] <= 0
 R97: tkey[2,0,1] - lane[0,0] <= 0
 R98: tkey[0,1,0] - lane[0,1] <= 0
 R99: tkey[2,1,1] - lane[0,1] <= 0
 R100: tkey[0,2,0] - lane[0,2] <= 0
 R101: tkey[2,2,1] - lane[0,2] <= 0
 R102: tkey[0,3,0] - lane[0,3] <= 0
 R103: tkey[2,3,1] - lane[0,3] <= 0
 R104: tkey[1,0,0] - lane[1,0] <= 0
 R105: tkey[3,0,1] - lane[1,0] <= 0
 R106: tkey[1,1,0] - lane[1,1] <= 0
 R107: tkey[3,1,1] - lane[1,1] <= 0
 R108: tkey[1,2,0] - lane[1,2] <= 0
 R109: tkey[3,2,1] - lane[1,2] <= 0
 R110: tkey[1,3,0] - lane[1,3] <= 0
 R111: tkey[3,3,1] - lane[1,3] <= 0
 R112: tkey[2,0,0] - lane[2,0] <= 0
 R113: tkey[0,2,1] - lane[2,0] <= 0
 R114: tkey[2,1,0] - lane[2,1] <= 0
 R115: tkey[0,0,1] - lane[2,1] <= 0
 R116: tkey[2,2,0] - lane[2,2] <= 0
 R117: tkey[1,0,1] - lane[2,2] <= 0
 R118: tkey[2,3,0] - lane[2,3] <= 0
 R119: tkey[1,3,1] - lane[2,3] <= 0
 R120: tkey[3,0,0] - lane[3,0] <= 0
 R121: tkey[1,2,1] - lane[3,0] <= 0
 R122: tkey[3,1,0] - lane[3,1] <= 0
 R123: tkey[0,3,1] - lane[3,1] <= 0
 R124: tkey[3,2,0] - lane[3,2] <= 0
 R125: tkey[1,1,1] - lane[3,2] <= 0
 R126: tkey[3,3,0] - lane[3,3] <= 0
 R127: tkey[0,1,1] - lane[3,3] <= 0
 R128: - tkey[0,0,0] - tkey[2,0,1] + 2 lane[0,0] <= 1
 R129: - tkey[0,1,0] - tkey[2,1,1] + 2 lane[0,1] <= 1
 R130: - tkey[0,2,0] - tkey[2,2,1] + 2 lane[0,2] <= 1
 R131: - tkey[0,3,0] - tkey[2,3,1] + 2 lane[0,3] <= 1
 R132: - tkey[1,0,0] - tkey[3,0,1] + 2 lane[1,0] <= 1
 R133: - tkey[1,1,0] - tkey[3,1,1] + 2 lane[1,1] <= 1
 R134: - tkey[1,2,0] - tkey[3,2,1] + 2 lane[1,2] <= 1
 R135: - tkey[1,3,0] - tkey[3,3,1] + 2 lane[1,3] <= 1
 R136: - tkey[0,2,1] - tkey[2,0,0] + 2 lane[2,0] <= 1
 R137: - tkey[0,0,1] - tkey[2,1,0] + 2 lane[2,1] <= 1
 R138: - tkey[1,0,1] - tkey[2,2,0] + 2 lane[2,2] <= 1
 R139: - tkey[1,3,1] - tkey[2,3,0] + 2 lane[2,3] <= 1
 R140: - tkey[1,2,1] - tkey[3,0,0] + 2 lane[3,0] <= 1
 R141: - tkey[0,3,1] - tkey[3,1,0] + 2 lane[3,1] <= 1
 R142: - tkey[1,1,1] - tkey[3,2,0] + 2 lane[3,2] <= 1
 R143: - tkey[0,1,1] - tkey[3,3,0] + 2 lane[3,3] <= 1
 R144: - tbox[0,0,0] + forg[0,0,0] <= 0
 R145: sbox[0,0,0] - rkey[0,0,0] + tkey[0,0,0] >= 0
 R146: sbox[0,0,0] + rkey[0,0,0] - tkey[0,0,0] >= 0
 R147: - sbox[0,0,0] + rkey[0,0,0] + tkey[0,0,0] >= 0
 R148: - tbox[1,0,0] + forg[1,0,0] <= 0
 R149: sbox[1,0,0] - rkey[1,0,0] + tkey[1,0,0] >= 0
 R150: sbox[1,0,0] + rkey[1,0,0] - tkey[1,0,0] >= 0
 R151: - sbox[1,0,0] + rkey[1,0,0] + tkey[1,0,0] >= 0
 R152: - tbox[2,0,0] + forg[2,0,0] <= 0
 R153: - sbox[2,0,0] + rkey[2,0,0] = 0
 R154: - tbox[3,0,0] + forg[3,0,0] <= 0
 R155: - sbox[3,0,0] + rkey[3,0,0] = 0
 R156: - sbox[3,0,1] + rkey[0,0,0] + rkey[2,2,0] >= 0
 R157: sbox[3,0,1] + rkey[0,0,0] - rkey[2,2,0] >= 0
 R158: sbox[3,0,1] - rkey[0,0,0] + rkey[2,2,0] >= 0
 R159: - sbox[0,0,1] + sbox[3,0,1] + rkey[3,1,0] >= 0
 R160: sbox[0,0,1] + sbox[3,0,1] - rkey[3,1,0] >= 0
 R161: sbox[0,0,1] - sbox[3,0,1] + rkey[3,1,0] >= 0
 R162: - sbox[2,0,1] + rkey[1,3,0] + rkey[2,2,0] >= 0
 R163: sbox[2,0,1] + rkey[1,3,0] - rkey[2,2,0] >= 0
 R164: sbox[2,0,1] - rkey[1,3,0] + rkey[2,2,0] >= 0
 R165: sbox[1,0,1] - rkey[0,0,0] = 0
 R166: tbox[0,0,0] + tbox[2,2,0] - tbox[3,0,1] >= 0
 R167: tbox[0,0,0] - tbox[2,2,0] + tbox[3,0,1] >= 0
 R168: - tbox[0,0,0] + tbox[2,2,0] + tbox[3,0,1] >= 0
 R169: - tbox[0,0,1] + tbox[3,0,1] + tbox[3,1,0] >= 0
 R170: tbox[0,0,1] + tbox[3,0,1] - tbox[3,1,0] >= 0
 R171: tbox[0,0,1] - tbox[3,0,1] + tbox[3,1,0] >= 0
 R172: tbox[1,3,0] - tbox[2,0,1] + tbox[2,2,0] >= 0
 R173: tbox[1,3,0] + tbox[2,0,1] - tbox[2,2,0] >= 0
 R174: - tbox[1,3,0] + tbox[2,0,1] + tbox[2,2,0] >= 0
 R175: - tbox[0,0,0] + tbox[1,0,1] = 0
 R176: unkn[0,0,0] - unkn[3,0,1] + forg[3,0,1] <= 0
 R177: unkn[2,2,0] - unkn[3,0,1] + forg[3,0,1] <= 0
 R178: unkn[0,0,0] + unkn[2,2,0] - unkn[3,0,1] + forg[3,0,1] >= 0
 R179: - unkn[0,0,1] + unkn[3,0,1] + forg[0,0,1] - forg[3,0,1] <= 0
 R180: - unkn[0,0,1] + unkn[3,1,0] + forg[0,0,1] <= 0
 R181: - unkn[0,0,1] + unkn[3,0,1] + unkn[3,1,0] + forg[0,0,1]
   - forg[3,0,1] >= 0
 R182: unkn[1,3,0] - unkn[2,0,1] + forg[2,0,1] <= 0
 R183: - unkn[2,0,1] + unkn[2,2,0] + forg[2,0,1] <= 0
 R184: unkn[1,3,0] - unkn[2,0,1] + unkn[2,2,0] + forg[2,0,1] >= 0
 R185: unkn[0,0,0] - unkn[1,0,1] + forg[1,0,1] = 0
 R186: - tbox[0,1,0] + forg[0,1,0] <= 0
 R187: sbox[0,1,0] - rkey[0,1,0] + tkey[0,1,0] >= 0
 R188: sbox[0,1,0] + rkey[0,1,0] - tkey[0,1,0] >= 0
 R189: - sbox[0,1,0] + rkey[0,1,0] + tkey[0,1,0] >= 0
 R190: - tbox[1,1,0] + forg[1,1,0] <= 0
 R191: sbox[1,1,0] - rkey[1,1,0] + tkey[1,1,0] >= 0
 R192: sbox[1,1,0] + rkey[1,1,0] - tkey[1,1,0] >= 0
 R193: - sbox[1,1,0] + rkey[1,1,0] + tkey[1,1,0] >= 0
 R194: - tbox[2,1,0] + forg[2,1,0] <= 0
 R195: - sbox[2,1,0] + rkey[2,1,0] = 0
 R196: - tbox[3,1,0] + forg[3,1,0] <= 0
 R197: - sbox[3,1,0] + rkey[3,1,0] = 0
 R198: - sbox[3,1,1] + rkey[0,1,0] + rkey[2,3,0] >= 0
 R199: sbox[3,1,1] + rkey[0,1,0] - rkey[2,3,0] >= 0
 R200: sbox[3,1,1] - rkey[0,1,0] + rkey[2,3,0] >= 0
 R201: - sbox[0,1,1] + sbox[3,1,1] + rkey[3,2,0] >= 0
 R202: sbox[0,1,1] + sbox[3,1,1] - rkey[3,2,0] >= 0
 R203: sbox[0,1,1] - sbox[3,1,1] + rkey[3,2,0] >= 0
 R204: - sbox[2,1,1] + rkey[1,0,0] + rkey[2,3,0] >= 0
 R205: sbox[2,1,1] + rkey[1,0,0] - rkey[2,3,0] >= 0
 R206: sbox[2,1,1] - rkey[1,0,0] + rkey[2,3,0] >= 0
 R207: sbox[1,1,1] - rkey[0,1,0] = 0
 R208: tbox[0,1,0] + tbox[2,3,0] - tbox[3,1,1] >= 0
 R209: tbox[0,1,0] - tbox[2,3,0] + tbox[3,1,1] >= 0
 R210: - tbox[0,1,0] + tbox[2,3,0] + tbox[3,1,1] >= 0
 R211: - tbox[0,1,1] + tbox[3,1,1] + tbox[3,2,0] >= 0
 R212: tbox[0,1,1] + tbox[3,1,1] - tbox[3,2,0] >= 0
 R213: tbox[0,1,1] - tbox[3,1,1] + tbox[3,2,0] >= 0
 R214: tbox[1,0,0] - tbox[2,1,1] + tbox[2,3,0] >= 0
 R215: tbox[1,0,0] + tbox[2,1,1] - tbox[2,3,0] >= 0
 R216: - tbox[1,0,0] + tbox[2,1,1] + tbox[2,3,0] >= 0
 R217: - tbox[0,1,0] + tbox[1,1,1] = 0
 R218: unkn[0,1,0] - unkn[3,1,1] + forg[3,1,1] <= 0
 R219: unkn[2,3,0] - unkn[3,1,1] + forg[3,1,1] <= 0
 R220: unkn[0,1,0] + unkn[2,3,0] - unkn[3,1,1] + forg[3,1,1] >= 0
 R221: - unkn[0,1,1] + unkn[3,1,1] + forg[0,1,1] - forg[3,1,1] <= 0
 R222: - unkn[0,1,1] + unkn[3,2,0] + forg[0,1,1] <= 0
 R223: - unkn[0,1,1] + unkn[3,1,1] + unkn[3,2,0] + forg[0,1,1]
   - forg[3,1,1] >= 0
 R224: unkn[1,0,0] - unkn[2,1,1] + forg[2,1,1] <= 0
 R225: - unkn[2,1,1] + unkn[2,3,0] + forg[2,1,1] <= 0
 R226: unkn[1,0,0] - unkn[2,1,1] + unkn[2,3,0] + forg[2,1,1] >= 0
 R227: unkn[0,1,0] - unkn[1,1,1] + forg[1,1,1] = 0
 R228: - tbox[0,2,0] + forg[0,2,0] <= 0
 R229: sbox[0,2,0] - rkey[0,2,0] + tkey[0,2,0] >= 0
 R230: sbox[0,2,0] + rkey[0,2,0] - tkey[0,2,0] >= 0
 R231: - sbox[0,2,0] + rkey[0,2,0] + tkey[0,2,0] >= 0
 R232: - tbox[1,2,0] + forg[1,2,0] <= 0
 R233: sbox[1,2,0] - rkey[1,2,0] + tkey[1,2,0] >= 0
 R234: sbox[1,2,0] + rkey[1,2,0] - tkey[1,2,0] >= 0
 R235: - sbox[1,2,0] + rkey[1,2,0] + tkey[1,2,0] >= 0
 R236: - tbox[2,2,0] + forg[2,2,0] <= 0
 R237: - sbox[2,2,0] + rkey[2,2,0] = 0
 R238: - tbox[3,2,0] + forg[3,2,0] <= 0
 R239: - sbox[3,2,0] + rkey[3,2,0] = 0
 R240: - sbox[3,2,1] + rkey[0,2,0] + rkey[2,0,0] >= 0
 R241: sbox[3,2,1] + rkey[0,2,0] - rkey[2,0,0] >= 0
 R242: sbox[3,2,1] - rkey[0,2,0] + rkey[2,0,0] >= 0
 R243: - sbox[0,2,1] + sbox[3,2,1] + rkey[3,3,0] >= 0
 R244: sbox[0,2,1] + sbox[3,2,1] - rkey[3,3,0] >= 0
 R245: sbox[0,2,1] - sbox[3,2,1] + rkey[3,3,0] >= 0
 R246: - sbox[2,2,1] + rkey[1,1,0] + rkey[2,0,0] >= 0
 R247: sbox[2,2,1] + rkey[1,1,0] - rkey[2,0,0] >= 0
 R248: sbox[2,2,1] - rkey[1,1,0] + rkey[2,0,0] >= 0
 R249: sbox[1,2,1] - rkey[0,2,0] = 0
 R250: tbox[0,2,0] + tbox[2,0,0] - tbox[3,2,1] >= 0
 R251: tbox[0,2,0] - tbox[2,0,0] + tbox[3,2,1] >= 0
 R252: - tbox[0,2,0] + tbox[2,0,0] + tbox[3,2,1] >= 0
 R253: - tbox[0,2,1] + tbox[3,2,1] + tbox[3,3,0] >= 0
 R254: tbox[0,2,1] + tbox[3,2,1] - tbox[3,3,0] >= 0
 R255: tbox[0,2,1] - tbox[3,2,1] + tbox[3,3,0] >= 0
 R256: tbox[1,1,0] + tbox[2,0,0] - tbox[2,2,1] >= 0
 R257: tbox[1,1,0] - tbox[2,0,0] + tbox[2,2,1] >= 0
 R258: - tbox[1,1,0] + tbox[2,0,0] + tbox[2,2,1] >= 0
 R259: - tbox[0,2,0] + tbox[1,2,1] = 0
 R260: unkn[0,2,0] - unkn[3,2,1] + forg[3,2,1] <= 0
 R261: unkn[2,0,0] - unkn[3,2,1] + forg[3,2,1] <= 0
 R262: unkn[0,2,0] + unkn[2,0,0] - unkn[3,2,1] + forg[3,2,1] >= 0
 R263: - unkn[0,2,1] + unkn[3,2,1] + forg[0,2,1] - forg[3,2,1] <= 0
 R264: - unkn[0,2,1] + unkn[3,3,0] + forg[0,2,1] <= 0
 R265: - unkn[0,2,1] + unkn[3,2,1] + unkn[3,3,0] + forg[0,2,1]
   - forg[3,2,1] >= 0
 R266: unkn[1,1,0] - unkn[2,2,1] + forg[2,2,1] <= 0
 R267: unkn[2,0,0] - unkn[2,2,1] + forg[2,2,1] <= 0
 R268: unkn[1,1,0] + unkn[2,0,0] - unkn[2,2,1] + forg[2,2,1] >= 0
 R269: unkn[0,2,0] - unkn[1,2,1] + forg[1,2,1] = 0
 R270: - tbox[0,3,0] + forg[0,3,0] <= 0
 R271: sbox[0,3,0] - rkey[0,3,0] + tkey[0,3,0] >= 0
 R272: sbox[0,3,0] + rkey[0,3,0] - tkey[0,3,0] >= 0
 R273: - sbox[0,3,0] + rkey[0,3,0] + tkey[0,3,0] >= 0
 R274: - tbox[1,3,0] + forg[1,3,0] <= 0
 R275: sbox[1,3,0] - rkey[1,3,0] + tkey[1,3,0] >= 0
 R276: sbox[1,3,0] + rkey[1,3,0] - tkey[1,3,0] >= 0
 R277: - sbox[1,3,0] + rkey[1,3,0] + tkey[1,3,0] >= 0
 R278: - tbox[2,3,0] + forg[2,3,0] <= 0
 R279: - sbox[2,3,0] + rkey[2,3,0] = 0
 R280: - tbox[3,3,0] + forg[3,3,0] <= 0
 R281: - sbox[3,3,0] + rkey[3,3,0] = 0
 R282: - sbox[3,3,1] + rkey[0,3,0] + rkey[2,1,0] >= 0
 R283: sbox[3,3,1] + rkey[0,3,0] - rkey[2,1,0] >= 0
 R284: sbox[3,3,1] - rkey[0,3,0] + rkey[2,1,0] >= 0
 R285: - sbox[0,3,1] + sbox[3,3,1] + rkey[3,0,0] >= 0
 R286: sbox[0,3,1] + sbox[3,3,1] - rkey[3,0,0] >= 0
 R287: sbox[0,3,1] - sbox[3,3,1] + rkey[3,0,0] >= 0
 R288: - sbox[2,3,1] + rkey[1,2,0] + rkey[2,1,0] >= 0
 R289: sbox[2,3,1] + rkey[1,2,0] - rkey[2,1,0] >= 0
 R290: sbox[2,3,1] - rkey[1,2,0] + rkey[2,1,0] >= 0
 R291: sbox[1,3,1] - rkey[0,3,0] = 0
 R292: tbox[0,3,0] + tbox[2,1,0] - tbox[3,3,1] >= 0
 R293: tbox[0,3,0] - tbox[2,1,0] + tbox[3,3,1] >= 0
 R294: - tbox[0,3,0] + tbox[2,1,0] + tbox[3,3,1] >= 0
 R295: - tbox[0,3,1] + tbox[3,0,0] + tbox[3,3,1] >= 0
 R296: tbox[0,3,1] - tbox[3,0,0] + tbox[3,3,1] >= 0
 R297: tbox[0,3,1] + tbox[3,0,0] - tbox[3,3,1] >= 0
 R298: tbox[1,2,0] + tbox[2,1,0] - tbox[2,3,1] >= 0
 R299: tbox[1,2,0] - tbox[2,1,0] + tbox[2,3,1] >= 0
 R300: - tbox[1,2,0] + tbox[2,1,0] + tbox[2,3,1] >= 0
 R301: - tbox[0,3,0] + tbox[1,3,1] = 0
 R302: unkn[0,3,0] - unkn[3,3,1] + forg[3,3,1] <= 0
 R303: unkn[2,1,0] - unkn[3,3,1] + forg[3,3,1] <= 0
 R304: unkn[0,3,0] + unkn[2,1,0] - unkn[3,3,1] + forg[3,3,1] >= 0
 R305: - unkn[0,3,1] + unkn[3,3,1] + forg[0,3,1] - forg[3,3,1] <= 0
 R306: - unkn[0,3,1] + unkn[3,0,0] + forg[0,3,1] <= 0
 R307: - unkn[0,3,1] + unkn[3,0,0] + unkn[3,3,1] + forg[0,3,1]
   - forg[3,3,1] >= 0
 R308: unkn[1,2,0] - unkn[2,3,1] + forg[2,3,1] <= 0
 R309: unkn[2,1,0] - unkn[2,3,1] + forg[2,3,1] <= 0
 R310: unkn[1,2,0] + unkn[2,1,0] - unkn[2,3,1] + forg[2,3,1] >= 0
 R311: unkn[0,3,0] - unkn[1,3,1] + forg[1,3,1] = 0
 R312: - tbox[0,0,1] + forg[0,0,1] <= 0
 R313: sbox[0,0,1] - rkey[0,0,1] + tkey[0,0,1] >= 0
 R314: sbox[0,0,1] + rkey[0,0,1] - tkey[0,0,1] >= 0
 R315: - sbox[0,0,1] + rkey[0,0,1] + tkey[0,0,1] >= 0
 R316: - tbox[1,0,1] + forg[1,0,1] <= 0
 R317: sbox[1,0,1] - rkey[1,0,1] + tkey[1,0,1] >= 0
 R318: sbox[1,0,1] + rkey[1,0,1] - tkey[1,0,1] >= 0
 R319: - sbox[1,0,1] + rkey[1,0,1] + tkey[1,0,1] >= 0
 R320: - tbox[2,0,1] + forg[2,0,1] <= 0
 R321: - sbox[2,0,1] + rkey[2,0,1] = 0
 R322: - tbox[3,0,1] + forg[3,0,1] <= 0
 R323: - sbox[3,0,1] + rkey[3,0,1] = 0
 R324: - sbox[3,0,2] + rkey[0,0,1] + rkey[2,2,1] >= 0
 R325: sbox[3,0,2] + rkey[0,0,1] - rkey[2,2,1] >= 0
 R326: sbox[3,0,2] - rkey[0,0,1] + rkey[2,2,1] >= 0
 R327: - sbox[0,0,2] + sbox[3,0,2] + rkey[3,1,1] >= 0
 R328: sbox[0,0,2] + sbox[3,0,2] - rkey[3,1,1] >= 0
 R329: sbox[0,0,2] - sbox[3,0,2] + rkey[3,1,1] >= 0
 R330: - sbox[2,0,2] + rkey[1,3,1] + rkey[2,2,1] >= 0
 R331: sbox[2,0,2] + rkey[1,3,1] - rkey[2,2,1] >= 0
 R332: sbox[2,0,2] - rkey[1,3,1] + rkey[2,2,1] >= 0
 R333: sbox[1,0,2] - rkey[0,0,1] = 0
 R334: tbox[0,0,1] + tbox[2,2,1] - tbox[3,0,2] >= 0
 R335: tbox[0,0,1] - tbox[2,2,1] + tbox[3,0,2] >= 0
 R336: - tbox[0,0,1] + tbox[2,2,1] + tbox[3,0,2] >= 0
 R337: - tbox[0,0,2] + tbox[3,0,2] + tbox[3,1,1] >= 0
 R338: tbox[0,0,2] + tbox[3,0,2] - tbox[3,1,1] >= 0
 R339: tbox[0,0,2] - tbox[3,0,2] + tbox[3,1,1] >= 0
 R340: tbox[1,3,1] - tbox[2,0,2] + tbox[2,2,1] >= 0
 R341: tbox[1,3,1] + tbox[2,0,2] - tbox[2,2,1] >= 0
 R342: - tbox[1,3,1] + tbox[2,0,2] + tbox[2,2,1] >= 0
 R343: - tbox[0,0,1] + tbox[1,0,2] = 0
 R344: unkn[0,0,1] - unkn[3,0,2] + forg[3,0,2] <= 0
 R345: unkn[2,2,1] - unkn[3,0,2] + forg[3,0,2] <= 0
 R346: unkn[0,0,1] + unkn[2,2,1] - unkn[3,0,2] + forg[3,0,2] >= 0
 R347: - unkn[0,0,2] + unkn[3,0,2] + forg[0,0,2] - forg[3,0,2] <= 0
 R348: - unkn[0,0,2] + unkn[3,1,1] + forg[0,0,2] <= 0
 R349: - unkn[0,0,2] + unkn[3,0,2] + unkn[3,1,1] + forg[0,0,2]
   - forg[3,0,2] >= 0
 R350: unkn[1,3,1] - unkn[2,0,2] + forg[2,0,2] <= 0
 R351: - unkn[2,0,2] + unkn[2,2,1] + forg[2,0,2] <= 0
 R352: unkn[1,3,1] - unkn[2,0,2] + unkn[2,2,1] + forg[2,0,2] >= 0
 R353: unkn[0,0,1] - unkn[1,0,2] + forg[1,0,2] = 0
 R354: - tbox[0,1,1] + forg[0,1,1] <= 0
 R355: sbox[0,1,1] - rkey[0,1,1] + tkey[0,1,1] >= 0
 R356: sbox[0,1,1] + rkey[0,1,1] - tkey[0,1,1] >= 0
 R357: - sbox[0,1,1] + rkey[0,1,1] + tkey[0,1,1] >= 0
 R358: - tbox[1,1,1] + forg[1,1,1] <= 0
 R359: sbox[1,1,1] - rkey[1,1,1] + tkey[1,1,1] >= 0
 R360: sbox[1,1,1] + rkey[1,1,1] - tkey[1,1,1] >= 0
 R361: - sbox[1,1,1] + rkey[1,1,1] + tkey[1,1,1] >= 0
 R362: - tbox[2,1,1] + forg[2,1,1] <= 0
 R363: - sbox[2,1,1] + rkey[2,1,1] = 0
 R364: - tbox[3,1,1] + forg[3,1,1] <= 0
 R365: - sbox[3,1,1] + rkey[3,1,1] = 0
 R366: - sbox[3,1,2] + rkey[0,1,1] + rkey[2,3,1] >= 0
 R367: sbox[3,1,2] + rkey[0,1,1] - rkey[2,3,1] >= 0
 R368: sbox[3,1,2] - rkey[0,1,1] + rkey[2,3,1] >= 0
 R369: - sbox[0,1,2] + sbox[3,1,2] + rkey[3,2,1] >= 0
 R370: sbox[0,1,2] + sbox[3,1,2] - rkey[3,2,1] >= 0
 R371: sbox[0,1,2] - sbox[3,1,2] + rkey[3,2,1] >= 0
 R372: - sbox[2,1,2] + rkey[1,0,1] + rkey[2,3,1] >= 0
 R373: sbox[2,1,2] + rkey[1,0,1] - rkey[2,3,1] >= 0
 R374: sbox[2,1,2] - rkey[1,0,1] + rkey[2,3,1] >= 0
 R375: sbox[1,1,2] - rkey[0,1,1] = 0
 R376: tbox[0,1,1] + tbox[2,3,1] - tbox[3,1,2] >= 0
 R377: tbox[0,1,1] - tbox[2,3,1] + tbox[3,1,2] >= 0
 R378: - tbox[0,1,1] + tbox[2,3,1] + tbox[3,1,2] >= 0
 R379: - tbox[0,1,2] + tbox[3,1,2] + tbox[3,2,1] >= 0
 R380: tbox[0,1,2] + tbox[3,1,2] - tbox[3,2,1] >= 0
 R381: tbox[0,1,2] - tbox[3,1,2] + tbox[3,2,1] >= 0
 R382: tbox[1,0,1] - tbox[2,1,2] + tbox[2,3,1] >= 0
 R383: tbox[1,0,1] + tbox[2,1,2] - tbox[2,3,1] >= 0
 R384: - tbox[1,0,1] + tbox[2,1,2] + tbox[2,3,1] >= 0
 R385: - tbox[0,1,1] + tbox[1,1,2] = 0
 R386: unkn[0,1,1] - unkn[3,1,2] + forg[3,1,2] <= 0
 R387: unkn[2,3,1] - unkn[3,1,2] + forg[3,1,2] <= 0
 R388: unkn[0,1,1] + unkn[2,3,1] - unkn[3,1,2] + forg[3,1,2] >= 0
 R389: - unkn[0,1,2] + unkn[3,1,2] + forg[0,1,2] - forg[3,1,2] <= 0
 R390: - unkn[0,1,2] + unkn[3,2,1] + forg[0,1,2] <= 0
 R391: - unkn[0,1,2] + unkn[3,1,2] + unkn[3,2,1] + forg[0,1,2]
   - forg[3,1,2] >= 0
 R392: unkn[1,0,1] - unkn[2,1,2] + forg[2,1,2] <= 0
 R393: - unkn[2,1,2] + unkn[2,3,1] + forg[2,1,2] <= 0
 R394: unkn[1,0,1] - unkn[2,1,2] + unkn[2,3,1] + forg[2,1,2] >= 0
 R395: unkn[0,1,1] - unkn[1,1,2] + forg[1,1,2] = 0
 R396: - tbox[0,2,1] + forg[0,2,1] <= 0
 R397: sbox[0,2,1] - rkey[0,2,1] + tkey[0,2,1] >= 0
 R398: sbox[0,2,1] + rkey[0,2,1] - tkey[0,2,1] >= 0
 R399: - sbox[0,2,1] + rkey[0,2,1] + tkey[0,2,1] >= 0
 R400: - tbox[1,2,1] + forg[1,2,1] <= 0
 R401: sbox[1,2,1] - rkey[1,2,1] + tkey[1,2,1] >= 0
 R402: sbox[1,2,1] + rkey[1,2,1] - tkey[1,2,1] >= 0
 R403: - sbox[1,2,1] + rkey[1,2,1] + tkey[1,2,1] >= 0
 R404: - tbox[2,2,1] + forg[2,2,1] <= 0
 R405: - sbox[2,2,1] + rkey[2,2,1] = 0
 R406: - tbox[3,2,1] + forg[3,2,1] <= 0
 R407: - sbox[3,2,1] + rkey[3,2,1] = 0
 R408: - sbox[3,2,2] + rkey[0,2,1] + rkey[2,0,1] >= 0
 R409: sbox[3,2,2] + rkey[0,2,1] - rkey[2,0,1] >= 0
 R410: sbox[3,2,2] - rkey[0,2,1] + rkey[2,0,1] >= 0
 R411: - sbox[0,2,2] + sbox[3,2,2] + rkey[3,3,1] >= 0
 R412: sbox[0,2,2] + sbox[3,2,2] - rkey[3,3,1] >= 0
 R413: sbox[0,2,2] - sbox[3,2,2] + rkey[3,3,1] >= 0
 R414: - sbox[2,2,2] + rkey[1,1,1] + rkey[2,0,1] >= 0
 R415: sbox[2,2,2] + rkey[1,1,1] - rkey[2,0,1] >= 0
 R416: sbox[2,2,2] - rkey[1,1,1] + rkey[2,0,1] >= 0
 R417: sbox[1,2,2] - rkey[0,2,1] = 0
 R418: tbox[0,2,1] + tbox[2,0,1] - tbox[3,2,2] >= 0
 R419: tbox[0,2,1] - tbox[2,0,1] + tbox[3,2,2] >= 0
 R420: - tbox[0,2,1] + tbox[2,0,1] + tbox[3,2,2] >= 0
 R421: - tbox[0,2,2] + tbox[3,2,2] + tbox[3,3,1] >= 0
 R422: tbox[0,2,2] + tbox[3,2,2] - tbox[3,3,1] >= 0
 R423: tbox[0,2,2] - tbox[3,2,2] + tbox[3,3,1] >= 0
 R424: tbox[1,1,1] + tbox[2,0,1] - tbox[2,2,2] >= 0
 R425: tbox[1,1,1] - tbox[2,0,1] + tbox[2,2,2] >= 0
 R426: - tbox[1,1,1] + tbox[2,0,1] + tbox[2,2,2] >= 0
 R427: - tbox[0,2,1] + tbox[1,2,2] = 0
 R428: unkn[0,2,1] - unkn[3,2,2] + forg[3,2,2] <= 0
 R429: unkn[2,0,1] - unkn[3,2,2] + forg[3,2,2] <= 0
 R430: unkn[0,2,1] + unkn[2,0,1] - unkn[3,2,2] + forg[3,2,2] >= 0
 R431: - unkn[0,2,2] + unkn[3,2,2] + forg[0,2,2] - forg[3,2,2] <= 0
 R432: - unkn[0,2,2] + unkn[3,3,1] + forg[0,2,2] <= 0
 R433: - unkn[0,2,2] + unkn[3,2,2] + unkn[3,3,1] + forg[0,2,2]
   - forg[3,2,2] >= 0
 R434: unkn[1,1,1] - unkn[2,2,2] + forg[2,2,2] <= 0
 R435: unkn[2,0,1] - unkn[2,2,2] + forg[2,2,2] <= 0
 R436: unkn[1,1,1] + unkn[2,0,1] - unkn[2,2,2] + forg[2,2,2] >= 0
 R437: unkn[0,2,1] - unkn[1,2,2] + forg[1,2,2] = 0
 R438: - tbox[0,3,1] + forg[0,3,1] <= 0
 R439: sbox[0,3,1] - rkey[0,3,1] + tkey[0,3,1] >= 0
 R440: sbox[0,3,1] + rkey[0,3,1] - tkey[0,3,1] >= 0
 R441: - sbox[0,3,1] + rkey[0,3,1] + tkey[0,3,1] >= 0
 R442: - tbox[1,3,1] + forg[1,3,1] <= 0
 R443: sbox[1,3,1] - rkey[1,3,1] + tkey[1,3,1] >= 0
 R444: sbox[1,3,1] + rkey[1,3,1] - tkey[1,3,1] >= 0
 R445: - sbox[1,3,1] + rkey[1,3,1] + tkey[1,3,1] >= 0
 R446: - tbox[2,3,1] + forg[2,3,1] <= 0
 R447: - sbox[2,3,1] + rkey[2,3,1] = 0
 R448: - tbox[3,3,1] + forg[3,3,1] <= 0
 R449: - sbox[3,3,1] + rkey[3,3,1] = 0
 R450: - sbox[3,3,2] + rkey[0,3,1] + rkey[2,1,1] >= 0
 R451: sbox[3,3,2] + rkey[0,3,1] - rkey[2,1,1] >= 0
 R452: sbox[3,3,2] - rkey[0,3,1] + rkey[2,1,1] >= 0
 R453: - sbox[0,3,2] + sbox[3,3,2] + rkey[3,0,1] >= 0
 R454: sbox[0,3,2] + sbox[3,3,2] - rkey[3,0,1] >= 0
 R455: sbox[0,3,2] - sbox[3,3,2] + rkey[3,0,1] >= 0
 R456: - sbox[2,3,2] + rkey[1,2,1] + rkey[2,1,1] >= 0
 R457: sbox[2,3,2] + rkey[1,2,1] - rkey[2,1,1] >= 0
 R458: sbox[2,3,2] - rkey[1,2,1] + rkey[2,1,1] >= 0
 R459: sbox[1,3,2] - rkey[0,3,1] = 0
 R460: tbox[0,3,1] + tbox[2,1,1] - tbox[3,3,2] >= 0
 R461: tbox[0,3,1] - tbox[2,1,1] + tbox[3,3,2] >= 0
 R462: - tbox[0,3,1] + tbox[2,1,1] + tbox[3,3,2] >= 0
 R463: - tbox[0,3,2] + tbox[3,0,1] + tbox[3,3,2] >= 0
 R464: tbox[0,3,2] - tbox[3,0,1] + tbox[3,3,2] >= 0
 R465: tbox[0,3,2] + tbox[3,0,1] - tbox[3,3,2] >= 0
 R466: tbox[1,2,1] + tbox[2,1,1] - tbox[2,3,2] >= 0
 R467: tbox[1,2,1] - tbox[2,1,1] + tbox[2,3,2] >= 0
 R468: - tbox[1,2,1] + tbox[2,1,1] + tbox[2,3,2] >= 0
 R469: - tbox[0,3,1] + tbox[1,3,2] = 0
 R470: unkn[0,3,1] - unkn[3,3,2] + forg[3,3,2] <= 0
 R471: unkn[2,1,1] - unkn[3,3,2] + forg[3,3,2] <= 0
 R472: unkn[0,3,1] + unkn[2,1,1] - unkn[3,3,2] + forg[3,3,2] >= 0
 R473: - unkn[0,3,2] + unkn[3,3,2] + forg[0,3,2] - forg[3,3,2] <= 0
 R474: - unkn[0,3,2] + unkn[3,0,1] + forg[0,3,2] <= 0
 R475: - unkn[0,3,2] + unkn[3,0,1] + unkn[3,3,2] + forg[0,3,2]
   - forg[3,3,2] >= 0
 R476: unkn[1,2,1] - unkn[2,3,2] + forg[2,3,2] <= 0
 R477: unkn[2,1,1] - unkn[2,3,2] + forg[2,3,2] <= 0
 R478: unkn[1,2,1] + unkn[2,1,1] - unkn[2,3,2] + forg[2,3,2] >= 0
 R479: unkn[0,3,1] - unkn[1,3,2] + forg[1,3,2] = 0
 R480: sbox[0,0,0] + sbox[0,1,0] + sbox[0,2,0] + sbox[0,3,0] + sbox[1,0,0]
   + sbox[1,1,0] + sbox[1,2,0] + sbox[1,3,0] + sbox[2,0,0] + sbox[2,1,0]
   + sbox[2,2,0] + sbox[2,3,0] + sbox[3,0,0] + sbox[3,1,0] + sbox[3,2,0]
   + sbox[3,3,0] + tkey[0,0,0] + tkey[0,1,0] + tkey[0,2,0] + tkey[0,3,0]
   + tkey[1,0,0] + tkey[1,1,0] + tkey[1,2,0] + tkey[1,3,0] + tkey[2,0,0]
   + tkey[2,1,0] + tkey[2,2,0] + tkey[2,3,0] + tkey[3,0,0] + tkey[3,1,0]
   + tkey[3,2,0] + tkey[3,3,0] >= 1
 R481: sbox[0,0,1] + sbox[0,1,1] + sbox[0,2,1] + sbox[0,3,1] + sbox[1,0,1]
   + sbox[1,1,1] + sbox[1,2,1] + sbox[1,3,1] + sbox[2,0,1] + sbox[2,1,1]
   + sbox[2,2,1] + sbox[2,3,1] + sbox[3,0,1] + sbox[3,1,1] + sbox[3,2,1]
   + sbox[3,3,1] + tkey[0,0,1] + tkey[0,1,1] + tkey[0,2,1] + tkey[0,3,1]
   + tkey[1,0,1] + tkey[1,1,1] + tkey[1,2,1] + tkey[1,3,1] + tkey[2,0,1]
   + tkey[2,1,1] + tkey[2,2,1] + tkey[2,3,1] + tkey[3,0,1] + tkey[3,1,1]
   + tkey[3,2,1] + tkey[3,3,1] >= 1
 R482: sbox[0,0,0] + sbox[0,1,0] + sbox[0,2,0] + sbox[0,3,0] + sbox[1,0,0]
   + sbox[1,1,0] + sbox[1,2,0] + sbox[1,3,0] + sbox[2,0,0] + sbox[2,1,0]
   + sbox[2,2,0] + sbox[2,3,0] + sbox[3,0,0] + sbox[3,1,0] + sbox[3,2,0]
   + sbox[3,3,0] = 0
 R483: sbox[0,0,2] + sbox[0,1,2] + sbox[0,2,2] + sbox[0,3,2] + sbox[1,0,2]
   + sbox[1,1,2] + sbox[1,2,2] + sbox[1,3,2] + sbox[2,0,2] + sbox[2,1,2]
   + sbox[2,2,2] + sbox[2,3,2] + sbox[3,0,2] + sbox[3,1,2] + sbox[3,2,2]
   + sbox[3,3,2] = 0
 R484: tbox[0,0,0] = 1
 R485: unkn[0,0,0] - forg[0,0,0] = 0
 R486: tbox[0,1,0] = 0
 R487: unkn[0,1,0] - forg[0,1,0] = 0
 R488: tbox[0,2,0] = 0
 R489: unkn[0,2,0] - forg[0,2,0] = 0
 R490: tbox[0,3,0] = 0
 R491: unkn[0,3,0] - forg[0,3,0] = 0
 R492: tbox[1,0,0] = 0
 R493: unkn[1,0,0] - forg[1,0,0] = 0
 R494: tbox[1,1,0] = 0
 R495: unkn[1,1,0] - forg[1,1,0] = 0
 R496: tbox[1,2,0] = 0
 R497: unkn[1,2,0] - forg[1,2,0] = 0
 R498: tbox[1,3,0] = 0
 R499: unkn[1,3,0] - forg[1,3,0] = 0
 R500: tbox[2,0,0] = 0
 R501: unkn[2,0,0] - forg[2,0,0] = 0
 R502: tbox[2,1,0] = 0
 R503: unkn[2,1,0] - forg[2,1,0] = 0
 R504: tbox[2,2,0] = 0
 R505: unkn[2,2,0] - forg[2,2,0] = 0
 R506: tbox[2,3,0] = 0
 R507: unkn[2,3,0] - forg[2,3,0] = 0
 R508: tbox[3,0,0] = 0
 R509: unkn[3,0,0] - forg[3,0,0] = 0
 R510: tbox[3,1,0] = 0
 R511: unkn[3,1,0] - forg[3,1,0] = 0
 R512: tbox[3,2,0] = 0
 R513: unkn[3,2,0] - forg[3,2,0] = 0
 R514: tbox[3,3,0] = 0
 R515: unkn[3,3,0] - forg[3,3,0] = 0
Bounds
 cost[0,0,0] <= 2
 cost[0,0,1] <= 2
 cost[0,0,2] <= 2
 cost[0,1,0] <= 2
 cost[0,1,1] <= 2
 cost[0,1,2] <= 2
 cost[0,2,0] <= 2
 cost[0,2,1] <= 2
 cost[0,2,2] <= 2
 cost[0,3,0] <= 2
 cost[0,3,1] <= 2
 cost[0,3,2] <= 2
 cost[1,0,0] <= 2
 cost[1,0,1] <= 2
 cost[1,0,2] <= 2
 cost[1,1,0] <= 2
 cost[1,1,1] <= 2
 cost[1,1,2] <= 2
 cost[1,2,0] <= 2
 cost[1,2,1] <= 2
 cost[1,2,2] <= 2
 cost[1,3,0] <= 2
 cost[1,3,1] <= 2
 cost[1,3,2] <= 2
 cost[2,0,0] <= 2
 cost[2,0,1] <= 2
 cost[2,0,2] <= 2
 cost[2,1,0] <= 2
 cost[2,1,1] <= 2
 cost[2,1,2] <= 2
 cost[2,2,0] <= 2
 cost[2,2,1] <= 2
 cost[2,2,2] <= 2
 cost[2,3,0] <= 2
 cost[2,3,1] <= 2
 cost[2,3,2] <= 2
 cost[3,0,0] <= 2
 cost[3,0,1] <= 2
 cost[3,0,2] <= 2
 cost[3,1,0] <= 2
 cost[3,1,1] <= 2
 cost[3,1,2] <= 2
 cost[3,2,0] <= 2
 cost[3,2,1] <= 2
 cost[3,2,2] <= 2
 cost[3,3,0] <= 2
 cost[3,3,1] <= 2
 cost[3,3,2] <= 2
Binaries
 sbox[0,0,0] sbox[0,0,1] sbox[0,0,2] sbox[0,1,0] sbox[0,1,1] sbox[0,1,2]
 sbox[0,2,0] sbox[0,2,1] sbox[0,2,2] sbox[0,3,0] sbox[0,3,1] sbox[0,3,2]
 sbox[1,0,0] sbox[1,0,1] sbox[1,0,2] sbox[1,1,0] sbox[1,1,1] sbox[1,1,2]
 sbox[1,2,0] sbox[1,2,1] sbox[1,2,2] sbox[1,3,0] sbox[1,3,1] sbox[1,3,2]
 sbox[2,0,0] sbox[2,0,1] sbox[2,0,2] sbox[2,1,0] sbox[2,1,1] sbox[2,1,2]
 sbox[2,2,0] sbox[2,2,1] sbox[2,2,2] sbox[2,3,0] sbox[2,3,1] sbox[2,3,2]
 sbox[3,0,0] sbox[3,0,1] sbox[3,0,2] sbox[3,1,0] sbox[3,1,1] sbox[3,1,2]
 sbox[3,2,0] sbox[3,2,1] sbox[3,2,2] sbox[3,3,0] sbox[3,3,1] sbox[3,3,2]
 rkey[0,0,0] rkey[0,0,1] rkey[0,0,2] rkey[0,1,0] rkey[0,1,1] rkey[0,1,2]
 rkey[0,2,0] rkey[0,2,1] rkey[0,2,2] rkey[0,3,0] rkey[0,3,1] rkey[0,3,2]
 rkey[1,0,0] rkey[1,0,1] rkey[1,0,2] rkey[1,1,0] rkey[1,1,1] rkey[1,1,2]
 rkey[1,2,0] rkey[1,2,1] rkey[1,2,2] rkey[1,3,0] rkey[1,3,1] rkey[1,3,2]
 rkey[2,0,0] rkey[2,0,1] rkey[2,0,2] rkey[2,1,0] rkey[2,1,1] rkey[2,1,2]
 rkey[2,2,0] rkey[2,2,1] rkey[2,2,2] rkey[2,3,0] rkey[2,3,1] rkey[2,3,2]
 rkey[3,0,0] rkey[3,0,1] rkey[3,0,2] rkey[3,1,0] rkey[3,1,1] rkey[3,1,2]
 rkey[3,2,0] rkey[3,2,1] rkey[3,2,2] rkey[3,3,0] rkey[3,3,1] rkey[3,3,2]
 tkey[0,0,0] tkey[0,0,1] tkey[0,0,2] tkey[0,1,0] tkey[0,1,1] tkey[0,1,2]
 tkey[0,2,0] tkey[0,2,1] tkey[0,2,2] tkey[0,3,0] tkey[0,3,1] tkey[0,3,2]
 tkey[1,0,0] tkey[1,0,1] tkey[1,0,2] tkey[1,1,0] tkey[1,1,1] tkey[1,1,2]
 tkey[1,2,0] tkey[1,2,1] tkey[1,2,2] tkey[1,3,0] tkey[1,3,1] tkey[1,3,2]
 tkey[2,0,0] tkey[2,0,1] tkey[2,0,2] tkey[2,1,0] tkey[2,1,1] tkey[2,1,2]
 tkey[2,2,0] tkey[2,2,1] tkey[2,2,2] tkey[2,3,0] tkey[2,3,1] tkey[2,3,2]
 tkey[3,0,0] tkey[3,0,1] tkey[3,0,2] tkey[3,1,0] tkey[3,1,1] tkey[3,1,2]
 tkey[3,2,0] tkey[3,2,1] tkey[3,2,2] tkey[3,3,0] tkey[3,3,1] tkey[3,3,2]
 tbox[0,0,0] tbox[0,0,1] tbox[0,0,2] tbox[0,1,0] tbox[0,1,1] tbox[0,1,2]
 tbox[0,2,0] tbox[0,2,1] tbox[0,2,2] tbox[0,3,0] tbox[0,3,1] tbox[0,3,2]
 tbox[1,0,0] tbox[1,0,1] tbox[1,0,2] tbox[1,1,0] tbox[1,1,1] tbox[1,1,2]
 tbox[1,2,0] tbox[1,2,1] tbox[1,2,2] tbox[1,3,0] tbox[1,3,1] tbox[1,3,2]
 tbox[2,0,0] tbox[2,0,1] tbox[2,0,2] tbox[2,1,0] tbox[2,1,1] tbox[2,1,2]
 tbox[2,2,0] tbox[2,2,1] tbox[2,2,2] tbox[2,3,0] tbox[2,3,1] tbox[2,3,2]
 tbox[3,0,0] tbox[3,0,1] tbox[3,0,2] tbox[3,1,0] tbox[3,1,1] tbox[3,1,2]
 tbox[3,2,0] tbox[3,2,1] tbox[3,2,2] tbox[3,3,0] tbox[3,3,1] tbox[3,3,2]
 unkn[0,0,0] unkn[0,0,1] unkn[0,0,2] unkn[0,1,0] unkn[0,1,1] unkn[0,1,2]
 unkn[0,2,0] unkn[0,2,1] unkn[0,2,2] unkn[0,3,0] unkn[0,3,1] unkn[0,3,2]
 unkn[1,0,0] unkn[1,0,1] unkn[1,0,2] unkn[1,1,0] unkn[1,1,1] unkn[1,1,2]
 unkn[1,2,0] unkn[1,2,1] unkn[1,2,2] unkn[1,3,0] unkn[1,3,1] unkn[1,3,2]
 unkn[2,0,0] unkn[2,0,1] unkn[2,0,2] unkn[2,1,0] unkn[2,1,1] unkn[2,1,2]
 unkn[2,2,0] unkn[2,2,1] unkn[2,2,2] unkn[2,3,0] unkn[2,3,1] unkn[2,3,2]
 unkn[3,0,0] unkn[3,0,1] unkn[3,0,2] unkn[3,1,0] unkn[3,1,1] unkn[3,1,2]
 unkn[3,2,0] unkn[3,2,1] unkn[3,2,2] unkn[3,3,0] unkn[3,3,1] unkn[3,3,2]
 forg[0,0,0] forg[0,0,1] forg[0,0,2] forg[0,1,0] forg[0,1,1] forg[0,1,2]
 forg[0,2,0] forg[0,2,1] forg[0,2,2] forg[0,3,0] forg[0,3,1] forg[0,3,2]
 forg[1,0,0] forg[1,0,1] forg[1,0,2] forg[1,1,0] forg[1,1,1] forg[1,1,2]
 forg[1,2,0] forg[1,2,1] forg[1,2,2] forg[1,3,0] forg[1,3,1] forg[1,3,2]
 forg[2,0,0] forg[2,0,1] forg[2,0,2] forg[2,1,0] forg[2,1,1] forg[2,1,2]
 forg[2,2,0] forg[2,2,1] forg[2,2,2] forg[2,3,0] forg[2,3,1] forg[2,3,2]
 forg[3,0,0] forg[3,0,1] forg[3,0,2] forg[3,1,0] forg[3,1,1] forg[3,1,2]
 forg[3,2,0] forg[3,2,1] forg[3,2,2] forg[3,3,0] forg[3,3,1] forg[3,3,2]
 lane[0,0] lane[0,1] lane[0,2] lane[0,3] lane[1,0] lane[1,1] lane[1,2]
 lane[1,3] lane[2,0] lane[2,1] lane[2,2] lane[2,3] lane[3,0] lane[3,1]
 lane[3,2] lane[3,3]
Generals
 cost[0,0,0] cost[0,0,1] cost[0,0,2] cost[0,1,0] cost[0,1,1] cost[0,1,2]
 cost[0,2,0] cost[0,2,1] cost[0,2,2] cost[0,3,0] cost[0,3,1] cost[0,3,2]
 cost[1,0,0] cost[1,0,1] cost[1,0,2] cost[1,1,0] cost[1,1,1] cost[1,1,2]
 cost[1,2,0] cost[1,2,1] cost[1,2,2] cost[1,3,0] cost[1,3,1] cost[1,3,2]
 cost[2,0,0] cost[2,0,1] cost[2,0,2] cost[2,1,0] cost[2,1,1] cost[2,1,2]
 cost[2,2,0] cost[2,2,1] cost[2,2,2] cost[2,3,0] cost[2,3,1] cost[2,3,2]
 cost[3,0,0] cost[3,0,1] cost[3,0,2] cost[3,1,0] cost[3,1,1] cost[3,1,2]
 cost[3,2,0] cost[3,2,1] cost[3,2,2] cost[3,3,0] cost[3,3,1] cost[3,3,2]
End
