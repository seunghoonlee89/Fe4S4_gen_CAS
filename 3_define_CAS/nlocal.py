#
# Localization based on UNO from UHF/UKS check files
#
import numpy
import scipy.linalg
import h5py
from pyscf import tools,gto,scf,dft
from pyscf.tools import molden
import pmloc 
import ulocal

def sqrtm(s):
   e, v = numpy.linalg.eigh(s)
   return numpy.dot(v*numpy.sqrt(e), v.T.conj())

def lowdin(s):
   e, v = numpy.linalg.eigh(s)
   return numpy.dot(v/numpy.sqrt(e), v.T.conj())

def dumpAct(fname,flmo,actlst,neact,base=1):
   # Load
   chkfile = fname+'.chk'
   outfile = fname+'_cmo.molden'
   tools.molden.from_chkfile(outfile, chkfile)
   mol,mf = scf.chkfile.load_scf(chkfile)
   mo_coeff = mf['mo_coeff']
   nb = mo_coeff.shape[1]
   nalpha = (mol.nelectron+mol.spin)/2
   nbeta  = (mol.nelectron-mol.spin)/2
   ova = mol.intor_symmetric('cint1e_ovlp_sph')
   # Active
   actlst2 = [i-base for i in actlst]
   ka = len(actlst2)
   assert (mol.nelectron-neact)%2 == 0
   kc = (mol.nelectron-neact)/2
   kv = nb - ka - kc
   # Generate indices
   torb = set(range(nb))
   aorb = set(actlst2)
   rorb = torb.difference(aorb)
   aorb = numpy.sort(numpy.array(list(aorb)))
   rorb = numpy.sort(numpy.array(list(rorb)))
   corb = rorb[:kc]
   vorb = rorb[kc:]
   print '[dumpAct]'
   print ' corb=',len(corb)
   print ' aorb=',len(aorb)
   print ' vorb=',len(vorb)
   print ' List of aorbs =',aorb
   print ' List of corbs =',corb
   print ' List of vorbs =',vorb
   # Load
   f = h5py.File(flmo+'.h5')
   lmo = f['lmo'].value
   f.close()
   clmo = lmo[:,corb].copy() 
   almo = lmo[:,aorb].copy() 
   vlmo = lmo[:,vorb].copy() 
   # Localization of active space
   ierr,ua = pmloc.loc(mol,almo)
   almo = almo.dot(ua)

   # <F> and <P> +> enorb,occ
   # Spin-averaged DM
   ma = mo_coeff[0]
   mb = mo_coeff[1]
   pTa = numpy.dot(ma[:,:nalpha],ma[:,:nalpha].T)
   pTb = numpy.dot(mb[:,:nbeta],mb[:,:nbeta].T)
   pav = pTa+pTb
   # Averaged Fock
   enorb = mf["mo_energy"]
   fa = reduce(numpy.dot,(ma,numpy.diag(enorb[0]),ma.T))
   fb = reduce(numpy.dot,(mb,numpy.diag(enorb[1]),mb.T))
   fav = 0.5*(fa+fb)
   almo,n_o,e_o = ulocal.psort(ova,fav,pav,almo)

   # Dump
   lmo2 = numpy.hstack((clmo,almo,vlmo))
   # Expectation value of natural orbitals <i|F|i>
   pexpt = reduce(numpy.dot,(lmo2.T,ova,pav,ova,lmo2))
   fexpt = reduce(numpy.dot,(lmo2.T,ova,fav,ova,lmo2))
   occ = numpy.diag(pexpt)
   enorb = numpy.diag(fexpt)
   # CHECK
   diff = reduce(numpy.dot,(lmo2.T,ova,lmo2)) - numpy.identity(nb)
   diff = numpy.linalg.norm(diff)
   print 'diff=',diff
   assert diff<1.e-10
   ulocal.dumpLMO(mol,fname+'_new',lmo2)
   ulocal.lowdinPop(mol,lmo2,ova,enorb,occ)
   print 'nalpha,nbeta,mol.spin:',nalpha,nbeta,mol.spin
   print 'kc,ka,kv,nb=',kc,ka,kv,nb

   ulocal.dumpLMO(mol,fname+'_ActiveOnly',almo)

   return lmo2,kc,ka,kv

if __name__ == '__main__':
   import time
   t0 = time.time()
   fname = 'hs_bp86' 
   flmo = 'hs_bp86'
   act1 = [i+109 for i in range(12)]
   act2 = [121,122,123,124] 
   act3 = [i+125 for i in range(20)]
   actlst = act1+act2+act3 
   assert len(actlst) == 36 
   neact = 54 
   dumpAct(fname,flmo,actlst,neact,base=1)
   t1 = time.time()
   print 'finished t =',t1-t0
