#!/usr/bin/env python3
"""Optional check: are the flagged I-1000 slices roughly evenly spaced?"""
import numpy as np

flagged_z = [65,75,85,185,195,205,330,340,350,425,435,445,775,785,795]
d = np.diff(sorted(flagged_z))
print("spacing between flagged regions:", d)