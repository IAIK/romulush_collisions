\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 rkey[0,0,1] + 0 rkey[0,1,1] + 0 rkey[0,2,1] + 0 rkey[0,3,1]
   + 0 rkey[1,0,1] + 0 rkey[1,1,1] + 0 rkey[1,2,1] + 0 rkey[1,3,1]
   + 0 rkey[2,0,1] + 0 rkey[2,1,1] + 0 rkey[2,2,1] + 0 rkey[2,3,1]
   + 0 rkey[3,0,1] + 0 rkey[3,1,1] + 0 rkey[3,2,1] + 0 rkey[3,3,1]
   + 0 tkey[0,0,1] + 0 tkey[0,1,1] + 0 tkey[0,2,1] + 0 tkey[0,3,1]
   + 0 tkey[1,0,1] + 0 tkey[1,1,1] + 0 tkey[1,2,1] + 0 tkey[1,3,1]
   + 0 tkey[2,0,1] + 0 tkey[2,1,1] + 0 tkey[2,2,1] + 0 tkey[2,3,1]
   + 0 tkey[3,0,1] + 0 tkey[3,1,1] + 0 tkey[3,2,1] + 0 tkey[3,3,1]
   + cost[0,0,0] + 0 cost[0,0,1] + cost[0,1,0] + 0 cost[0,1,1]
   + cost[0,2,0] + 0 cost[0,2,1] + cost[0,3,0] + 0 cost[0,3,1]
   + cost[1,0,0] + 0 cost[1,0,1] + cost[1,1,0] + 0 cost[1,1,1]
   + cost[1,2,0] + 0 cost[1,2,1] + cost[1,3,0] + 0 cost[1,3,1]
   + cost[2,0,0] + 0 cost[2,0,1] + cost[2,1,0] + 0 cost[2,1,1]
   + cost[2,2,0] + 0 cost[2,2,1] + cost[2,3,0] + 0 cost[2,3,1]
   + cost[3,0,0] + 0 cost[3,0,1] + cost[3,1,0] + 0 cost[3,1,1]
   + cost[3,2,0] + 0 cost[3,2,1] + cost[3,3,0] + 0 cost[3,3,1]
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
 R48: tkey[0,0,0] - lane[0,0] <= 0
 R49: tkey[0,1,0] - lane[0,1] <= 0
 R50: tkey[0,2,0] - lane[0,2] <= 0
 R51: tkey[0,3,0] - lane[0,3] <= 0
 R52: tkey[1,0,0] - lane[1,0] <= 0
 R53: tkey[1,1,0] - lane[1,1] <= 0
 R54: tkey[1,2,0] - lane[1,2] <= 0
 R55: tkey[1,3,0] - lane[1,3] <= 0
 R56: tkey[2,0,0] - lane[2,0] <= 0
 R57: tkey[2,1,0] - lane[2,1] <= 0
 R58: tkey[2,2,0] - lane[2,2] <= 0
 R59: tkey[2,3,0] - lane[2,3] <= 0
 R60: tkey[3,0,0] - lane[3,0] <= 0
 R61: tkey[3,1,0] - lane[3,1] <= 0
 R62: tkey[3,2,0] - lane[3,2] <= 0
 R63: tkey[3,3,0] - lane[3,3] <= 0
 R64: - tkey[0,0,0] + lane[0,0] <= 1
 R65: - tkey[0,1,0] + lane[0,1] <= 1
 R66: - tkey[0,2,0] + lane[0,2] <= 1
 R67: - tkey[0,3,0] + lane[0,3] <= 1
 R68: - tkey[1,0,0] + lane[1,0] <= 1
 R69: - tkey[1,1,0] + lane[1,1] <= 1
 R70: - tkey[1,2,0] + lane[1,2] <= 1
 R71: - tkey[1,3,0] + lane[1,3] <= 1
 R72: - tkey[2,0,0] + lane[2,0] <= 1
 R73: - tkey[2,1,0] + lane[2,1] <= 1
 R74: - tkey[2,2,0] + lane[2,2] <= 1
 R75: - tkey[2,3,0] + lane[2,3] <= 1
 R76: - tkey[3,0,0] + lane[3,0] <= 1
 R77: - tkey[3,1,0] + lane[3,1] <= 1
 R78: - tkey[3,2,0] + lane[3,2] <= 1
 R79: - tkey[3,3,0] + lane[3,3] <= 1
 R80: - tbox[0,0,0] + forg[0,0,0] <= 0
 R81: sbox[0,0,0] - rkey[0,0,0] + tkey[0,0,0] >= 0
 R82: sbox[0,0,0] + rkey[0,0,0] - tkey[0,0,0] >= 0
 R83: - sbox[0,0,0] + rkey[0,0,0] + tkey[0,0,0] >= 0
 R84: - tbox[1,0,0] + forg[1,0,0] <= 0
 R85: sbox[1,0,0] - rkey[1,0,0] + tkey[1,0,0] >= 0
 R86: sbox[1,0,0] + rkey[1,0,0] - tkey[1,0,0] >= 0
 R87: - sbox[1,0,0] + rkey[1,0,0] + tkey[1,0,0] >= 0
 R88: - tbox[2,0,0] + forg[2,0,0] <= 0
 R89: - sbox[2,0,0] + rkey[2,0,0] = 0
 R90: - tbox[3,0,0] + forg[3,0,0] <= 0
 R91: - sbox[3,0,0] + rkey[3,0,0] = 0
 R92: - sbox[3,0,1] + rkey[0,0,0] + rkey[2,2,0] >= 0
 R93: sbox[3,0,1] + rkey[0,0,0] - rkey[2,2,0] >= 0
 R94: sbox[3,0,1] - rkey[0,0,0] + rkey[2,2,0] >= 0
 R95: - sbox[0,0,1] + sbox[3,0,1] + rkey[3,1,0] >= 0
 R96: sbox[0,0,1] + sbox[3,0,1] - rkey[3,1,0] >= 0
 R97: sbox[0,0,1] - sbox[3,0,1] + rkey[3,1,0] >= 0
 R98: - sbox[2,0,1] + rkey[1,3,0] + rkey[2,2,0] >= 0
 R99: sbox[2,0,1] + rkey[1,3,0] - rkey[2,2,0] >= 0
 R100: sbox[2,0,1] - rkey[1,3,0] + rkey[2,2,0] >= 0
 R101: sbox[1,0,1] - rkey[0,0,0] = 0
 R102: tbox[0,0,0] + tbox[2,2,0] - tbox[3,0,1] >= 0
 R103: tbox[0,0,0] - tbox[2,2,0] + tbox[3,0,1] >= 0
 R104: - tbox[0,0,0] + tbox[2,2,0] + tbox[3,0,1] >= 0
 R105: - tbox[0,0,1] + tbox[3,0,1] + tbox[3,1,0] >= 0
 R106: tbox[0,0,1] + tbox[3,0,1] - tbox[3,1,0] >= 0
 R107: tbox[0,0,1] - tbox[3,0,1] + tbox[3,1,0] >= 0
 R108: tbox[1,3,0] - tbox[2,0,1] + tbox[2,2,0] >= 0
 R109: tbox[1,3,0] + tbox[2,0,1] - tbox[2,2,0] >= 0
 R110: - tbox[1,3,0] + tbox[2,0,1] + tbox[2,2,0] >= 0
 R111: - tbox[0,0,0] + tbox[1,0,1] = 0
 R112: unkn[0,0,0] - unkn[3,0,1] + forg[3,0,1] <= 0
 R113: unkn[2,2,0] - unkn[3,0,1] + forg[3,0,1] <= 0
 R114: unkn[0,0,0] + unkn[2,2,0] - unkn[3,0,1] + forg[3,0,1] >= 0
 R115: - unkn[0,0,1] + unkn[3,0,1] + forg[0,0,1] - forg[3,0,1] <= 0
 R116: - unkn[0,0,1] + unkn[3,1,0] + forg[0,0,1] <= 0
 R117: - unkn[0,0,1] + unkn[3,0,1] + unkn[3,1,0] + forg[0,0,1]
   - forg[3,0,1] >= 0
 R118: unkn[1,3,0] - unkn[2,0,1] + forg[2,0,1] <= 0
 R119: - unkn[2,0,1] + unkn[2,2,0] + forg[2,0,1] <= 0
 R120: unkn[1,3,0] - unkn[2,0,1] + unkn[2,2,0] + forg[2,0,1] >= 0
 R121: unkn[0,0,0] - unkn[1,0,1] + forg[1,0,1] = 0
 R122: - tbox[0,1,0] + forg[0,1,0] <= 0
 R123: sbox[0,1,0] - rkey[0,1,0] + tkey[0,1,0] >= 0
 R124: sbox[0,1,0] + rkey[0,1,0] - tkey[0,1,0] >= 0
 R125: - sbox[0,1,0] + rkey[0,1,0] + tkey[0,1,0] >= 0
 R126: - tbox[1,1,0] + forg[1,1,0] <= 0
 R127: sbox[1,1,0] - rkey[1,1,0] + tkey[1,1,0] >= 0
 R128: sbox[1,1,0] + rkey[1,1,0] - tkey[1,1,0] >= 0
 R129: - sbox[1,1,0] + rkey[1,1,0] + tkey[1,1,0] >= 0
 R130: - tbox[2,1,0] + forg[2,1,0] <= 0
 R131: - sbox[2,1,0] + rkey[2,1,0] = 0
 R132: - tbox[3,1,0] + forg[3,1,0] <= 0
 R133: - sbox[3,1,0] + rkey[3,1,0] = 0
 R134: - sbox[3,1,1] + rkey[0,1,0] + rkey[2,3,0] >= 0
 R135: sbox[3,1,1] + rkey[0,1,0] - rkey[2,3,0] >= 0
 R136: sbox[3,1,1] - rkey[0,1,0] + rkey[2,3,0] >= 0
 R137: - sbox[0,1,1] + sbox[3,1,1] + rkey[3,2,0] >= 0
 R138: sbox[0,1,1] + sbox[3,1,1] - rkey[3,2,0] >= 0
 R139: sbox[0,1,1] - sbox[3,1,1] + rkey[3,2,0] >= 0
 R140: - sbox[2,1,1] + rkey[1,0,0] + rkey[2,3,0] >= 0
 R141: sbox[2,1,1] + rkey[1,0,0] - rkey[2,3,0] >= 0
 R142: sbox[2,1,1] - rkey[1,0,0] + rkey[2,3,0] >= 0
 R143: sbox[1,1,1] - rkey[0,1,0] = 0
 R144: tbox[0,1,0] + tbox[2,3,0] - tbox[3,1,1] >= 0
 R145: tbox[0,1,0] - tbox[2,3,0] + tbox[3,1,1] >= 0
 R146: - tbox[0,1,0] + tbox[2,3,0] + tbox[3,1,1] >= 0
 R147: - tbox[0,1,1] + tbox[3,1,1] + tbox[3,2,0] >= 0
 R148: tbox[0,1,1] + tbox[3,1,1] - tbox[3,2,0] >= 0
 R149: tbox[0,1,1] - tbox[3,1,1] + tbox[3,2,0] >= 0
 R150: tbox[1,0,0] - tbox[2,1,1] + tbox[2,3,0] >= 0
 R151: tbox[1,0,0] + tbox[2,1,1] - tbox[2,3,0] >= 0
 R152: - tbox[1,0,0] + tbox[2,1,1] + tbox[2,3,0] >= 0
 R153: - tbox[0,1,0] + tbox[1,1,1] = 0
 R154: unkn[0,1,0] - unkn[3,1,1] + forg[3,1,1] <= 0
 R155: unkn[2,3,0] - unkn[3,1,1] + forg[3,1,1] <= 0
 R156: unkn[0,1,0] + unkn[2,3,0] - unkn[3,1,1] + forg[3,1,1] >= 0
 R157: - unkn[0,1,1] + unkn[3,1,1] + forg[0,1,1] - forg[3,1,1] <= 0
 R158: - unkn[0,1,1] + unkn[3,2,0] + forg[0,1,1] <= 0
 R159: - unkn[0,1,1] + unkn[3,1,1] + unkn[3,2,0] + forg[0,1,1]
   - forg[3,1,1] >= 0
 R160: unkn[1,0,0] - unkn[2,1,1] + forg[2,1,1] <= 0
 R161: - unkn[2,1,1] + unkn[2,3,0] + forg[2,1,1] <= 0
 R162: unkn[1,0,0] - unkn[2,1,1] + unkn[2,3,0] + forg[2,1,1] >= 0
 R163: unkn[0,1,0] - unkn[1,1,1] + forg[1,1,1] = 0
 R164: - tbox[0,2,0] + forg[0,2,0] <= 0
 R165: sbox[0,2,0] - rkey[0,2,0] + tkey[0,2,0] >= 0
 R166: sbox[0,2,0] + rkey[0,2,0] - tkey[0,2,0] >= 0
 R167: - sbox[0,2,0] + rkey[0,2,0] + tkey[0,2,0] >= 0
 R168: - tbox[1,2,0] + forg[1,2,0] <= 0
 R169: sbox[1,2,0] - rkey[1,2,0] + tkey[1,2,0] >= 0
 R170: sbox[1,2,0] + rkey[1,2,0] - tkey[1,2,0] >= 0
 R171: - sbox[1,2,0] + rkey[1,2,0] + tkey[1,2,0] >= 0
 R172: - tbox[2,2,0] + forg[2,2,0] <= 0
 R173: - sbox[2,2,0] + rkey[2,2,0] = 0
 R174: - tbox[3,2,0] + forg[3,2,0] <= 0
 R175: - sbox[3,2,0] + rkey[3,2,0] = 0
 R176: - sbox[3,2,1] + rkey[0,2,0] + rkey[2,0,0] >= 0
 R177: sbox[3,2,1] + rkey[0,2,0] - rkey[2,0,0] >= 0
 R178: sbox[3,2,1] - rkey[0,2,0] + rkey[2,0,0] >= 0
 R179: - sbox[0,2,1] + sbox[3,2,1] + rkey[3,3,0] >= 0
 R180: sbox[0,2,1] + sbox[3,2,1] - rkey[3,3,0] >= 0
 R181: sbox[0,2,1] - sbox[3,2,1] + rkey[3,3,0] >= 0
 R182: - sbox[2,2,1] + rkey[1,1,0] + rkey[2,0,0] >= 0
 R183: sbox[2,2,1] + rkey[1,1,0] - rkey[2,0,0] >= 0
 R184: sbox[2,2,1] - rkey[1,1,0] + rkey[2,0,0] >= 0
 R185: sbox[1,2,1] - rkey[0,2,0] = 0
 R186: tbox[0,2,0] + tbox[2,0,0] - tbox[3,2,1] >= 0
 R187: tbox[0,2,0] - tbox[2,0,0] + tbox[3,2,1] >= 0
 R188: - tbox[0,2,0] + tbox[2,0,0] + tbox[3,2,1] >= 0
 R189: - tbox[0,2,1] + tbox[3,2,1] + tbox[3,3,0] >= 0
 R190: tbox[0,2,1] + tbox[3,2,1] - tbox[3,3,0] >= 0
 R191: tbox[0,2,1] - tbox[3,2,1] + tbox[3,3,0] >= 0
 R192: tbox[1,1,0] + tbox[2,0,0] - tbox[2,2,1] >= 0
 R193: tbox[1,1,0] - tbox[2,0,0] + tbox[2,2,1] >= 0
 R194: - tbox[1,1,0] + tbox[2,0,0] + tbox[2,2,1] >= 0
 R195: - tbox[0,2,0] + tbox[1,2,1] = 0
 R196: unkn[0,2,0] - unkn[3,2,1] + forg[3,2,1] <= 0
 R197: unkn[2,0,0] - unkn[3,2,1] + forg[3,2,1] <= 0
 R198: unkn[0,2,0] + unkn[2,0,0] - unkn[3,2,1] + forg[3,2,1] >= 0
 R199: - unkn[0,2,1] + unkn[3,2,1] + forg[0,2,1] - forg[3,2,1] <= 0
 R200: - unkn[0,2,1] + unkn[3,3,0] + forg[0,2,1] <= 0
 R201: - unkn[0,2,1] + unkn[3,2,1] + unkn[3,3,0] + forg[0,2,1]
   - forg[3,2,1] >= 0
 R202: unkn[1,1,0] - unkn[2,2,1] + forg[2,2,1] <= 0
 R203: unkn[2,0,0] - unkn[2,2,1] + forg[2,2,1] <= 0
 R204: unkn[1,1,0] + unkn[2,0,0] - unkn[2,2,1] + forg[2,2,1] >= 0
 R205: unkn[0,2,0] - unkn[1,2,1] + forg[1,2,1] = 0
 R206: - tbox[0,3,0] + forg[0,3,0] <= 0
 R207: sbox[0,3,0] - rkey[0,3,0] + tkey[0,3,0] >= 0
 R208: sbox[0,3,0] + rkey[0,3,0] - tkey[0,3,0] >= 0
 R209: - sbox[0,3,0] + rkey[0,3,0] + tkey[0,3,0] >= 0
 R210: - tbox[1,3,0] + forg[1,3,0] <= 0
 R211: sbox[1,3,0] - rkey[1,3,0] + tkey[1,3,0] >= 0
 R212: sbox[1,3,0] + rkey[1,3,0] - tkey[1,3,0] >= 0
 R213: - sbox[1,3,0] + rkey[1,3,0] + tkey[1,3,0] >= 0
 R214: - tbox[2,3,0] + forg[2,3,0] <= 0
 R215: - sbox[2,3,0] + rkey[2,3,0] = 0
 R216: - tbox[3,3,0] + forg[3,3,0] <= 0
 R217: - sbox[3,3,0] + rkey[3,3,0] = 0
 R218: - sbox[3,3,1] + rkey[0,3,0] + rkey[2,1,0] >= 0
 R219: sbox[3,3,1] + rkey[0,3,0] - rkey[2,1,0] >= 0
 R220: sbox[3,3,1] - rkey[0,3,0] + rkey[2,1,0] >= 0
 R221: - sbox[0,3,1] + sbox[3,3,1] + rkey[3,0,0] >= 0
 R222: sbox[0,3,1] + sbox[3,3,1] - rkey[3,0,0] >= 0
 R223: sbox[0,3,1] - sbox[3,3,1] + rkey[3,0,0] >= 0
 R224: - sbox[2,3,1] + rkey[1,2,0] + rkey[2,1,0] >= 0
 R225: sbox[2,3,1] + rkey[1,2,0] - rkey[2,1,0] >= 0
 R226: sbox[2,3,1] - rkey[1,2,0] + rkey[2,1,0] >= 0
 R227: sbox[1,3,1] - rkey[0,3,0] = 0
 R228: tbox[0,3,0] + tbox[2,1,0] - tbox[3,3,1] >= 0
 R229: tbox[0,3,0] - tbox[2,1,0] + tbox[3,3,1] >= 0
 R230: - tbox[0,3,0] + tbox[2,1,0] + tbox[3,3,1] >= 0
 R231: - tbox[0,3,1] + tbox[3,0,0] + tbox[3,3,1] >= 0
 R232: tbox[0,3,1] - tbox[3,0,0] + tbox[3,3,1] >= 0
 R233: tbox[0,3,1] + tbox[3,0,0] - tbox[3,3,1] >= 0
 R234: tbox[1,2,0] + tbox[2,1,0] - tbox[2,3,1] >= 0
 R235: tbox[1,2,0] - tbox[2,1,0] + tbox[2,3,1] >= 0
 R236: - tbox[1,2,0] + tbox[2,1,0] + tbox[2,3,1] >= 0
 R237: - tbox[0,3,0] + tbox[1,3,1] = 0
 R238: unkn[0,3,0] - unkn[3,3,1] + forg[3,3,1] <= 0
 R239: unkn[2,1,0] - unkn[3,3,1] + forg[3,3,1] <= 0
 R240: unkn[0,3,0] + unkn[2,1,0] - unkn[3,3,1] + forg[3,3,1] >= 0
 R241: - unkn[0,3,1] + unkn[3,3,1] + forg[0,3,1] - forg[3,3,1] <= 0
 R242: - unkn[0,3,1] + unkn[3,0,0] + forg[0,3,1] <= 0
 R243: - unkn[0,3,1] + unkn[3,0,0] + unkn[3,3,1] + forg[0,3,1]
   - forg[3,3,1] >= 0
 R244: unkn[1,2,0] - unkn[2,3,1] + forg[2,3,1] <= 0
 R245: unkn[2,1,0] - unkn[2,3,1] + forg[2,3,1] <= 0
 R246: unkn[1,2,0] + unkn[2,1,0] - unkn[2,3,1] + forg[2,3,1] >= 0
 R247: unkn[0,3,0] - unkn[1,3,1] + forg[1,3,1] = 0
 R248: sbox[0,0,0] + sbox[0,1,0] + sbox[0,2,0] + sbox[0,3,0] + sbox[1,0,0]
   + sbox[1,1,0] + sbox[1,2,0] + sbox[1,3,0] + sbox[2,0,0] + sbox[2,1,0]
   + sbox[2,2,0] + sbox[2,3,0] + sbox[3,0,0] + sbox[3,1,0] + sbox[3,2,0]
   + sbox[3,3,0] + tkey[0,0,0] + tkey[0,1,0] + tkey[0,2,0] + tkey[0,3,0]
   + tkey[1,0,0] + tkey[1,1,0] + tkey[1,2,0] + tkey[1,3,0] + tkey[2,0,0]
   + tkey[2,1,0] + tkey[2,2,0] + tkey[2,3,0] + tkey[3,0,0] + tkey[3,1,0]
   + tkey[3,2,0] + tkey[3,3,0] >= 1
 R249: sbox[0,0,0] + sbox[0,1,0] + sbox[0,2,0] + sbox[0,3,0] + sbox[1,0,0]
   + sbox[1,1,0] + sbox[1,2,0] + sbox[1,3,0] + sbox[2,0,0] + sbox[2,1,0]
   + sbox[2,2,0] + sbox[2,3,0] + sbox[3,0,0] + sbox[3,1,0] + sbox[3,2,0]
   + sbox[3,3,0] = 0
 R250: sbox[0,0,1] + sbox[0,1,1] + sbox[0,2,1] + sbox[0,3,1] + sbox[1,0,1]
   + sbox[1,1,1] + sbox[1,2,1] + sbox[1,3,1] + sbox[2,0,1] + sbox[2,1,1]
   + sbox[2,2,1] + sbox[2,3,1] + sbox[3,0,1] + sbox[3,1,1] + sbox[3,2,1]
   + sbox[3,3,1] = 0
 R251: tbox[0,0,0] = 1
 R252: unkn[0,0,0] - forg[0,0,0] = 0
 R253: tbox[0,1,0] = 0
 R254: unkn[0,1,0] - forg[0,1,0] = 0
 R255: tbox[0,2,0] = 0
 R256: unkn[0,2,0] - forg[0,2,0] = 0
 R257: tbox[0,3,0] = 0
 R258: unkn[0,3,0] - forg[0,3,0] = 0
 R259: tbox[1,0,0] = 0
 R260: unkn[1,0,0] - forg[1,0,0] = 0
 R261: tbox[1,1,0] = 0
 R262: unkn[1,1,0] - forg[1,1,0] = 0
 R263: tbox[1,2,0] = 0
 R264: unkn[1,2,0] - forg[1,2,0] = 0
 R265: tbox[1,3,0] = 0
 R266: unkn[1,3,0] - forg[1,3,0] = 0
 R267: tbox[2,0,0] = 0
 R268: unkn[2,0,0] - forg[2,0,0] = 0
 R269: tbox[2,1,0] = 0
 R270: unkn[2,1,0] - forg[2,1,0] = 0
 R271: tbox[2,2,0] = 0
 R272: unkn[2,2,0] - forg[2,2,0] = 0
 R273: tbox[2,3,0] = 0
 R274: unkn[2,3,0] - forg[2,3,0] = 0
 R275: tbox[3,0,0] = 0
 R276: unkn[3,0,0] - forg[3,0,0] = 0
 R277: tbox[3,1,0] = 0
 R278: unkn[3,1,0] - forg[3,1,0] = 0
 R279: tbox[3,2,0] = 0
 R280: unkn[3,2,0] - forg[3,2,0] = 0
 R281: tbox[3,3,0] = 0
 R282: unkn[3,3,0] - forg[3,3,0] = 0
Bounds
 cost[0,0,0] <= 2
 cost[0,0,1] <= 2
 cost[0,1,0] <= 2
 cost[0,1,1] <= 2
 cost[0,2,0] <= 2
 cost[0,2,1] <= 2
 cost[0,3,0] <= 2
 cost[0,3,1] <= 2
 cost[1,0,0] <= 2
 cost[1,0,1] <= 2
 cost[1,1,0] <= 2
 cost[1,1,1] <= 2
 cost[1,2,0] <= 2
 cost[1,2,1] <= 2
 cost[1,3,0] <= 2
 cost[1,3,1] <= 2
 cost[2,0,0] <= 2
 cost[2,0,1] <= 2
 cost[2,1,0] <= 2
 cost[2,1,1] <= 2
 cost[2,2,0] <= 2
 cost[2,2,1] <= 2
 cost[2,3,0] <= 2
 cost[2,3,1] <= 2
 cost[3,0,0] <= 2
 cost[3,0,1] <= 2
 cost[3,1,0] <= 2
 cost[3,1,1] <= 2
 cost[3,2,0] <= 2
 cost[3,2,1] <= 2
 cost[3,3,0] <= 2
 cost[3,3,1] <= 2
Binaries
 sbox[0,0,0] sbox[0,0,1] sbox[0,1,0] sbox[0,1,1] sbox[0,2,0] sbox[0,2,1]
 sbox[0,3,0] sbox[0,3,1] sbox[1,0,0] sbox[1,0,1] sbox[1,1,0] sbox[1,1,1]
 sbox[1,2,0] sbox[1,2,1] sbox[1,3,0] sbox[1,3,1] sbox[2,0,0] sbox[2,0,1]
 sbox[2,1,0] sbox[2,1,1] sbox[2,2,0] sbox[2,2,1] sbox[2,3,0] sbox[2,3,1]
 sbox[3,0,0] sbox[3,0,1] sbox[3,1,0] sbox[3,1,1] sbox[3,2,0] sbox[3,2,1]
 sbox[3,3,0] sbox[3,3,1] rkey[0,0,0] rkey[0,0,1] rkey[0,1,0] rkey[0,1,1]
 rkey[0,2,0] rkey[0,2,1] rkey[0,3,0] rkey[0,3,1] rkey[1,0,0] rkey[1,0,1]
 rkey[1,1,0] rkey[1,1,1] rkey[1,2,0] rkey[1,2,1] rkey[1,3,0] rkey[1,3,1]
 rkey[2,0,0] rkey[2,0,1] rkey[2,1,0] rkey[2,1,1] rkey[2,2,0] rkey[2,2,1]
 rkey[2,3,0] rkey[2,3,1] rkey[3,0,0] rkey[3,0,1] rkey[3,1,0] rkey[3,1,1]
 rkey[3,2,0] rkey[3,2,1] rkey[3,3,0] rkey[3,3,1] tkey[0,0,0] tkey[0,0,1]
 tkey[0,1,0] tkey[0,1,1] tkey[0,2,0] tkey[0,2,1] tkey[0,3,0] tkey[0,3,1]
 tkey[1,0,0] tkey[1,0,1] tkey[1,1,0] tkey[1,1,1] tkey[1,2,0] tkey[1,2,1]
 tkey[1,3,0] tkey[1,3,1] tkey[2,0,0] tkey[2,0,1] tkey[2,1,0] tkey[2,1,1]
 tkey[2,2,0] tkey[2,2,1] tkey[2,3,0] tkey[2,3,1] tkey[3,0,0] tkey[3,0,1]
 tkey[3,1,0] tkey[3,1,1] tkey[3,2,0] tkey[3,2,1] tkey[3,3,0] tkey[3,3,1]
 tbox[0,0,0] tbox[0,0,1] tbox[0,1,0] tbox[0,1,1] tbox[0,2,0] tbox[0,2,1]
 tbox[0,3,0] tbox[0,3,1] tbox[1,0,0] tbox[1,0,1] tbox[1,1,0] tbox[1,1,1]
 tbox[1,2,0] tbox[1,2,1] tbox[1,3,0] tbox[1,3,1] tbox[2,0,0] tbox[2,0,1]
 tbox[2,1,0] tbox[2,1,1] tbox[2,2,0] tbox[2,2,1] tbox[2,3,0] tbox[2,3,1]
 tbox[3,0,0] tbox[3,0,1] tbox[3,1,0] tbox[3,1,1] tbox[3,2,0] tbox[3,2,1]
 tbox[3,3,0] tbox[3,3,1] unkn[0,0,0] unkn[0,0,1] unkn[0,1,0] unkn[0,1,1]
 unkn[0,2,0] unkn[0,2,1] unkn[0,3,0] unkn[0,3,1] unkn[1,0,0] unkn[1,0,1]
 unkn[1,1,0] unkn[1,1,1] unkn[1,2,0] unkn[1,2,1] unkn[1,3,0] unkn[1,3,1]
 unkn[2,0,0] unkn[2,0,1] unkn[2,1,0] unkn[2,1,1] unkn[2,2,0] unkn[2,2,1]
 unkn[2,3,0] unkn[2,3,1] unkn[3,0,0] unkn[3,0,1] unkn[3,1,0] unkn[3,1,1]
 unkn[3,2,0] unkn[3,2,1] unkn[3,3,0] unkn[3,3,1] forg[0,0,0] forg[0,0,1]
 forg[0,1,0] forg[0,1,1] forg[0,2,0] forg[0,2,1] forg[0,3,0] forg[0,3,1]
 forg[1,0,0] forg[1,0,1] forg[1,1,0] forg[1,1,1] forg[1,2,0] forg[1,2,1]
 forg[1,3,0] forg[1,3,1] forg[2,0,0] forg[2,0,1] forg[2,1,0] forg[2,1,1]
 forg[2,2,0] forg[2,2,1] forg[2,3,0] forg[2,3,1] forg[3,0,0] forg[3,0,1]
 forg[3,1,0] forg[3,1,1] forg[3,2,0] forg[3,2,1] forg[3,3,0] forg[3,3,1]
 lane[0,0] lane[0,1] lane[0,2] lane[0,3] lane[1,0] lane[1,1] lane[1,2]
 lane[1,3] lane[2,0] lane[2,1] lane[2,2] lane[2,3] lane[3,0] lane[3,1]
 lane[3,2] lane[3,3]
Generals
 cost[0,0,0] cost[0,0,1] cost[0,1,0] cost[0,1,1] cost[0,2,0] cost[0,2,1]
 cost[0,3,0] cost[0,3,1] cost[1,0,0] cost[1,0,1] cost[1,1,0] cost[1,1,1]
 cost[1,2,0] cost[1,2,1] cost[1,3,0] cost[1,3,1] cost[2,0,0] cost[2,0,1]
 cost[2,1,0] cost[2,1,1] cost[2,2,0] cost[2,2,1] cost[2,3,0] cost[2,3,1]
 cost[3,0,0] cost[3,0,1] cost[3,1,0] cost[3,1,1] cost[3,2,0] cost[3,2,1]
 cost[3,3,0] cost[3,3,1]
End
