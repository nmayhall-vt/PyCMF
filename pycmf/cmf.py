import numpy as np
import pyscf
from pyscf import gto, scf, fci

class CMF:
    """
    Class to organize CMF calculation
    """
    def __init__(self, mf:pyscf.scf.hf.SCF):
        self.mf = mf
        self.F = self.mf.get_fock()
        self.C = self.mf.mo_coeff
        self.S = self.mf.get_ovlp()
    
    def init(self, clusters:list, fspace:list):
        #   Get data
        self.clusters = clusters 
        self.fspace   = fspace 

    def set_mo_coeffs(self, C:np.ndarray) -> None:
        self.C = C

    def get_mo_coeffs(self) -> np.ndarray:
        return self.C
    
    def lowdin(self):
        print(" Form lowdin orthogonalized orbitals")
        #forming S^-1/2 to transform to A and B block.
        sal, svec = np.linalg.eigh(self.S)
        idx = sal.argsort()[::-1]
        sal = sal[idx]
        svec = svec[:, idx]
        sal = sal**-0.5
        sal = np.diagflat(sal)
        X = svec @ sal @ svec.T
        return X

    def do_local_casci(self, i):
        """
        Solve local problem for cluster i
        """

        mol = self.mf.mol

        h0 = self.mf.energy_nuc()
        h1  = self.mf.get_hcore()

        norb_i = len(self.clusters[i])

        Ci = self.C[:,self.clusters[i]]

        # now rotate to MO basis
        h1 = Ci.T @ h1 @ Ci
        h2 = pyscf.ao2mo.kernel(mol, Ci, aosym="s4", compact=False)
        h2.shape = (norb_i, norb_i, norb_i, norb_i)

        nelec = sum(self.fspace[i])
        cisolver = fci.direct_spin1.FCI()
        efci, ci = cisolver.kernel(h1, h2, norb_i, nelec, ecore=h0, nroots=1, verbose=100)
        
        fci_dim = ci.shape[0]*ci.shape[1]
        d1 = cisolver.make_rdm1(ci, norb_i, nelec)
        print(" PYSCF 1RDM: ")
        occs = np.linalg.eig(d1)[0]
        [print("%4i %12.8f"%(i,occs[i])) for i in range(len(occs))]
        with np.printoptions(precision=6, suppress=True):
            print(d1)
        print(" FCI:        %12.8f Dim:%6d"%(efci,fci_dim))

