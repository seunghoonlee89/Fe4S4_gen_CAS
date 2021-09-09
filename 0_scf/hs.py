#!/usr/bin/env python

from pyscf import scf
from pyscf import gto,dft

mol = gto.Mole()
mol.verbose = 6
mol.output = 'hs_bp86.out'
mol.atom = '''
S   0.04  -1.78  -1.292   
S  -0.04  1.78  -1.293   
S   1.78  -0.04  1.294   
S  -1.78  0.04   1.295   
Fe  0.05  -1.37  1.016   
Fe  -1.38  0.05  -1.007   
Fe  -0.05  1.38   1.008   
Fe  1.37  -0.05  -1.019   
S   0.24   3.30   2.1410  
S  -0.24  -3.29  2.1411  
S  -3.29  -0.24  -2.1412  
S   3.29   0.24  -2.1413  
C  -3.80  -1.84  -1.3814  
H  -3.91  -1.71  -0.2915  
H  -4.76  -2.17  -1.8116  
H  -3.03  -2.60  -1.5617  
C   3.80   1.83  -1.3818  
H   3.91   1.71  -0.2919  
H   4.76   2.16  -1.8120  
H   3.03   2.59  -1.5521  
C  -1.83  -3.80  1.3822  
H  -2.16  -4.76  1.8123  
H  -2.59  -3.03  1.5524  
H  -1.70  -3.91  0.2925  
C   1.84   3.80   1.3826  
H   2.17   4.76   1.8127  
H   2.60   3.03   1.5628  
H   1.71   3.91   0.29
'''
mol.basis = 'tzp-dkh'
mol.charge = -2
mol.spin = 18 #na-nb
mol.build()

mf = scf.sfx2c(scf.UKS(mol))
mf.chkfile = 'hs_bp86.chk'
mf.max_cycle = 500
mf.conv_tol = 1.e-4
mf.xc = 'b88,p86' 
mf.scf()

mf2 = scf.newton(mf)
#mf2.ah_start_tol = 1.e-8
#mf2.ah_conv_tol = 1.e-14
mf2.conv_tol = 1.e-12
mf2.kernel()
