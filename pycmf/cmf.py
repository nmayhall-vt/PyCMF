import numpy as np
import pyscf
from pyscf import gto, scf, fci

from .helpers import * 

class CMF:
    """
    Class to organize CMF calculation
    """
    def __init__(self, mf:pyscf.scf.hf.SCF):
        self.mf = mf
        self.F = self.mf.get_fock()
        self.C = self.mf.mo_coeff
        self.S = self.mf.get_ovlp()

    def init(self, clusters:list[list[int]], fspace:list[tuple[int,int]]):
        """Initialize a CMF calculation.
    
        
        Parameters
        ----------
        clusters : list
            A list of lists indicating which MOs are in each cluster.  
        fspace : list 
            A list of tuples indicating how many alpha and beta electrons are in each cluster 
        
        Returns
        -------
        None
        """

        #   Get data
        self.clusters = clusters 
        self.fspace   = fspace 
        self.cluster_energies:list[float] = [0.0 for i in clusters]
        self.cluster_1rdm:list[np.ndarray] = [np.zeros((len(i),len(i))) for i in clusters]
        self.roots = [1 for i in clusters] 
    
    def rotate_mo_coeffs(self, U:np.ndarray) -> None:
        self.C = self.C @ U

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

    def do_local_casci(self, i) -> None:
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

        cisolver = fci.direct_spin1.FCI()
        efci, ci = cisolver.kernel(h1, h2, norb_i, self.fspace[i], ecore=h0, nroots=1, verbose=100)

        self.cluster_energies[i] = efci

        fci_dim = ci.shape[0]*ci.shape[1]
        (d1a, d1b)  = cisolver.make_rdm1s(ci, norb_i, self.fspace[i])
        
        # d1 = cisolver.make_rdm1(ci, norb_i, nelec)
        # (d1a, d1b), (d2aa, d2ab, d2bb)  = cisolver.make_rdm12s(ci, no, (na,nb))
        
        d1 = d1a + d1b

        print(" PYSCF 1RDM: ")
        matrix_print(d1)
        
        occs = np.linalg.eig(d1)[0]

        print(" Occupation Numbers:")
        [print("%4i %12.8f"%(i,occs[i])) for i in range(len(occs))]
        print(" FCI:        %12.8f Dim:%6d"%(efci,fci_dim))

    def energy(self):
        """
        Compute current CMF energy
        """
        pass
    
    def orbital_gradient(self):
        """
        Compute current CMF orbital gradient 
        """
        pass
