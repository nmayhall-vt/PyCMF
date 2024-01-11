import pyscf
import pyscf.tools
import pycmf

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


def test1():
    cmf = pycmf.CMF(mf)
    C = cmf.lowdin()
    cmf.set_Cfrz(C[:,0])
    cmf.set_Cdoc(C[:,2])
    cmf.set_Cact(C[:,[4,5,6,7]], [[0,1], [2,3]], [(1,1), (1,1)])
    cmf.set_Cvir(C[:,3])

    cmf.build_active_integrals()
    cmf.do_local_casci(0)
    cmf.do_local_casci(1)
    # pyscf.tools.molden.from_mo(mf.mol, "C.molden", cmf.get_mo_coeffs())
    print(cmf.active_h0)
    print(cmf.cluster_energies)
    return cmf.cluster_energies

e1 = test1()

def test2():
    print()
    print(" Test 2")
    cmf = pycmf.CMF(mf)
    C = cmf.lowdin()
    # cmf.set_Cfrz(C[:,0])
    cmf.set_Cdoc(C[:,[0,2]])
    cmf.set_Cact(C[:,[4,5,6,7]], [[0,1], [2,3]], [(1,1), (1,1)])
    cmf.set_Cvir(C[:,3])

    cmf.build_active_integrals()
    cmf.do_local_casci(0)
    cmf.do_local_casci(1)
    # pyscf.tools.molden.from_mo(mf.mol, "C.molden", cmf.get_mo_coeffs())
    print(cmf.active_h0)
    print(cmf.cluster_energies)
    return cmf.cluster_energies

e2 = test2()

assert(abs(e1[0] - e2[0]) < 1e-16)
assert(abs(e1[1] - e2[1]) < 1e-16)
