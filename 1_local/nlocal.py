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

def dumpLUNO(fname,thresh=0.05):
   chkfile = fname+'.chk'
   outfile = fname+'_cmo.molden'
   tools.molden.from_chkfile(outfile, chkfile)
   #=============================
   # Natural orbitals
   # Lowdin basis X=S{-1/2}
   # psi = chi * C 
   #     = chi' * C'
   #     = chi*X*(X{-1}C')
   #=============================
   mol,mf = scf.chkfile.load_scf(chkfile)
   mo_coeff = mf["mo_coeff"]
   ova=mol.intor_symmetric("cint1e_ovlp_sph")
   nb = mo_coeff.shape[1]
   # Check overlap
   diff = reduce(numpy.dot,(mo_coeff[0].T,ova,mo_coeff[0])) - numpy.identity(nb)
   print numpy.linalg.norm(diff)
   diff = reduce(numpy.dot,(mo_coeff[1].T,ova,mo_coeff[1])) - numpy.identity(nb)
   print numpy.linalg.norm(diff)
   # UHF-alpha/beta
   ma = mo_coeff[0]
   mb = mo_coeff[1]
   nalpha = (mol.nelectron+mol.spin)/2
   nbeta  = (mol.nelectron-mol.spin)/2
   # Spin-averaged DM
   pTa = numpy.dot(ma[:,:nalpha],ma[:,:nalpha].T)
   pTb = numpy.dot(mb[:,:nbeta],mb[:,:nbeta].T)
   pT = 0.5*(pTa+pTb)
   # Lowdin basis
   s12 = sqrtm(ova)
   s12inv = lowdin(ova)
   pTOAO = reduce(numpy.dot,(s12,pT,s12))
   eig,coeff = scipy.linalg.eigh(-pTOAO)
   eig = -2.0*eig
   eig[eig<0.0]=0.0
   eig[abs(eig)<1.e-14]=0.0
   ifplot = False #True
   if ifplot:
      import matplotlib.pyplot as plt
      plt.plot(range(nb),eig,'ro')
      plt.show()
   # Back to AO basis
   coeff = numpy.dot(s12inv,coeff)
   diff = reduce(numpy.dot,(coeff.T,ova,coeff)) - numpy.identity(nb)
   print 'CtSC-I',numpy.linalg.norm(diff)
   # 
   # Averaged Fock
   #
   enorb = mf["mo_energy"]
   fa = reduce(numpy.dot,(ma,numpy.diag(enorb[0]),ma.T))
   fb = reduce(numpy.dot,(mb,numpy.diag(enorb[1]),mb.T))
   # Non-orthogonal cases: FC=SCE
   # Fao = SC*e*C{-1} = S*C*e*Ct*S
   fav = 0.5*(fa+fb)
   # Expectation value of natural orbitals <i|F|i>
   fexpt = reduce(numpy.dot,(coeff.T,ova,fav,ova,coeff))
   enorb = numpy.diag(fexpt)
   nocc = eig.copy()
   #
   # Reordering and define active space according to thresh
   #
   idx = 0
   active=[]
   for i in range(nb):
      if nocc[i]<=2.0-thresh and nocc[i]>=thresh:
         active.append(True)
      else:
         active.append(False)
   print '\nNatural orbitals:'
   for i in range(nb):
      print 'orb:',i,active[i],nocc[i],enorb[i]
   active = numpy.array(active)
   actIndices = list(numpy.argwhere(active==True).flatten())
   cOrbs = coeff[:,:actIndices[0]]
   aOrbs = coeff[:,actIndices]
   vOrbs = coeff[:,actIndices[-1]+1:]
   nb = cOrbs.shape[0]
   nc = cOrbs.shape[1]
   na = aOrbs.shape[1]
   nv = vOrbs.shape[1]
   print 'core orbs:',cOrbs.shape
   print 'act  orbs:',aOrbs.shape
   print 'vir  orbs:',vOrbs.shape
   assert nc+na+nv == nb
   # dump UNO
   with open(fname+'_uno.molden','w') as thefile:
       molden.header(mol,thefile)
       molden.orbital_coeff(mol,thefile,coeff)
   #=====================
   # Population analysis
   #=====================
   aux = s12inv

   #clmo = ulocal.scdm(cOrbs,ova,aux)
   #almo = ulocal.scdm(aOrbs,ova,aux)

   clmo = cOrbs
   almo = aOrbs
   ierr,uc = pmloc.loc(mol,clmo)
   ierr,ua = pmloc.loc(mol,almo)
   clmo = clmo.dot(uc)
   almo = almo.dot(ua)

   vlmo = ulocal.scdm(vOrbs,ova,aux)
   # P-SORT
   mo_c,n_c,e_c = ulocal.psort(ova,fav,pT,clmo)
   mo_o,n_o,e_o = ulocal.psort(ova,fav,pT,almo)
   mo_v,n_v,e_v = ulocal.psort(ova,fav,pT,vlmo)
   lmo = numpy.hstack((mo_c,mo_o,mo_v)).copy()
   enorb = numpy.hstack([e_c,e_o,e_v])
   occ = numpy.hstack([n_c,n_o,n_v])
   # CHECK
   diff = reduce(numpy.dot,(lmo.T,ova,lmo)) - numpy.identity(nb)
   print 'diff=',numpy.linalg.norm(diff)
   ulocal.lowdinPop(mol,lmo,ova,enorb,occ)
   ulocal.dumpLMO(mol,fname,lmo)
   print 'nalpha,nbeta,mol.spin,nb:',\
          nalpha,nbeta,mol.spin,nb
   return mol,ova,fav,pT,nb,nalpha,nbeta,nc,na,nv,lmo,enorb,occ

def dumpAct(fname,info,actlst,base=1):
   actlst2 = [i-base for i in actlst]
   mol,ova,fav,pT,nb,nalpha,nbeta,nc,na,nv,lmo,enorb,occ = info
   corb = set(range(nc)) 
   aorb = set(range(nc,nc+na))
   vorb = set(range(nc+na,nc+na+nv))
   print '[dumpAct]'
   print ' corb=',len(corb)
   print ' aorb=',len(aorb)
   print ' vorb=',len(vorb)
   sorb = set(actlst2)
   rcorb = corb.difference(corb.intersection(sorb))
   #assuming act in actlst
   #raorb = aorb.difference(aorb.intersection(sorb))
   rvorb = vorb.difference(vorb.intersection(sorb))
   corb = list(rcorb)
   aorb = list(sorb)
   vorb = list(rvorb)
   print ' corb_new=',len(corb)
   print ' aorb_new=',len(aorb)
   print ' vorb_new=',len(vorb)
   clmo = lmo[:,corb].copy() 
   almo = lmo[:,aorb].copy() 
   vlmo = lmo[:,vorb].copy() 
   ierr,ua = pmloc.loc(mol,almo)
   almo = almo.dot(ua)
   #>>> DUMP <<<#
   # P-SORT
   mo_c = clmo
   mo_v = vlmo
   e_c = enorb[corb].copy()
   e_v = enorb[vorb].copy()
   n_c = occ[corb].copy()
   n_v = occ[vorb].copy()
   mo_o,n_o,e_o = ulocal.psort(ova,fav,pT,almo)
   lmo2 = numpy.hstack((mo_c,mo_o,mo_v))
   enorb = numpy.hstack([e_c,e_o,e_v])
   occ = numpy.hstack([n_c,n_o,n_v])
   print len(enorb)
   print nb
   assert len(enorb)==nb
   assert len(occ)==nb
   # CHECK
   diff = reduce(numpy.dot,(lmo2.T,ova,lmo2)) - numpy.identity(nb)
   print 'diff=',numpy.linalg.norm(diff)
   ulocal.lowdinPop(mol,lmo2,ova,enorb,occ)
   ulocal.dumpLMO(mol,fname+'_new',lmo2)
   print 'nalpha,nbeta,mol.spin,nb:',\
          nalpha,nbeta,mol.spin,nb
   print 'diff(LMO2-LMO)=',numpy.linalg.norm(lmo2-lmo)
   nc = len(e_c)
   na = len(e_o)
   nv = len(e_v)
   assert na == len(actlst)
   assert nc+na+nv == nb
   print 'nc,na,nv,nb=',nc,na,nv,nb
   return lmo2,nc,na,nv

if __name__ == '__main__':
   import time
   t0 = time.time()
   fname = 'hs_bp86' 
   info = dumpLUNO(fname)
#   actlst = [483,\
#	     597]
#   dumpAct(fname,info,actlst,base=1)
   t1 = time.time()
   print 'finished t =',t1-t0
