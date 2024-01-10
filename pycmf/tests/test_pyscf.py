import pyscf
import pyscf.tools

molecule = """
He  0.0     0.0     0.0 
He  2.0     0.0     0.0  
He  2.0     2.0     0.0  
He  2.0     2.0     2.0  
"""
basis = "6-31g*"
pymol = pyscf.gto.Mole(
        atom    =   molecule,
        symmetry=   False,
        spin    =   0, # number of unpaired electrons
        charge  =   0,
        basis   =   basis).build()


pymol.build()
mf = pyscf.scf.RHF(pymol)
# mf.verbose = 4
mf.conv_tol = 1e-8
mf.conv_tol_grad = 1e-5
mf.run()

print(" Hartree-Fock Energy: %12.8f" % mf.e_tot)
# Get data
F = mf.get_fock()
C = mf.mo_coeff
S = mf.get_ovlp()

# Just use alpha orbitals
Cdocc = mf.mo_coeff[:,mf.mo_occ==2]
Csing = mf.mo_coeff[:,mf.mo_occ==1]
Cvirt = mf.mo_coeff[:,mf.mo_occ==0]
ndocc = Cdocc.shape[1]
nsing = Csing.shape[1]


import pycmf
cmf = pycmf.CMF(mf)
cmf.lowdin()
cmf.init([[0, 1], [2, 3], [4, 5], [6, 7]], [(1,1), (1,1), (1,1), (1,1)])
cmf.do_local_casci(1)

print(cmf.cluster_energies)