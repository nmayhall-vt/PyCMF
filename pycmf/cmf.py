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
        self.nbas = mf.mol.nbas
        
        self.Cfrz = np.zeros((mf.mol.nbas,0))   # frozen core - no orbital optimization 
        self.Cdoc = np.zeros((mf.mol.nbas,0))   # doubly occupied - no correlation
        self.Cact = np.zeros((mf.mol.nbas,0))   # active 
        self.Cvir = np.zeros((mf.mol.nbas,0))   # virtual - no correlation
        self.S = self.mf.get_ovlp()

        # Active space clustering
        self.clusters = []
        self.fspace = []

        # Active space integrals
        self.active_h0 = 0.0
        self.active_h1 = np.zeros((0,0))
        self.active_h2 = np.zeros((0,0,0,0))

    def set_Cfrz(self, C:np.ndarray):
        """Define the frozen orbitals which are not optimized. 
        """
        self.Cfrz = C
        if len(self.Cfrz.shape) == 1:
            self.Cfrz.shape = (self.Cfrz.shape[0], 1)
    
    def set_Cdoc(self, C:np.ndarray):
        """Define the doubly occupied orbitals.
        These represent a closed shell cluster.
        """
        self.Cdoc = C
        if len(C.shape) == 1:
            self.Cdoc.shape = (C.shape[0], 1)
    
    def set_Cvir(self, C:np.ndarray):
        """Define the completely virtual orbitals.
        """
        self.Cvir = C

    def set_Cact(self, C:np.ndarray, clusters:list[list[int]], fspace:list[tuple[int,int]]):
        """Define the active orbitals, with the associated clustering.
        
        Parameters
        ----------
        C: np.ndarray
            MO Coefficients of active space
        clusters : list
            A list of lists indicating which MOs are in each cluster.  
        fspace : list 
            A list of tuples indicating how many alpha and beta electrons are in each cluster 
        Returns
        -------
        None
        """

        #   Get data
        self.Cact = C
        self.clusters = clusters 
        self.fspace   = fspace 
        self.cluster_energies:list[float] = [0.0 for i in clusters]
        self.cluster_1rdm:list[np.ndarray] = [np.zeros((len(i),len(i))) for i in clusters]
        self.roots = [1 for i in clusters] 
    
    def rotate_mo_coeffs(self, U:np.ndarray) -> None:
        self.Cdoc = self.Cdoc @ U
        self.Cact = self.Cact @ U
        self.Cvir = self.Cvir @ U

    def get_mo_coeffs(self) -> np.ndarray:
        return np.hstack((self.Cdoc, self.Cact, self.Cvir)) 
    
    def lowdin(self) -> np.ndarray:
        """
        Get symetrically orthogonalized localized basis
        """
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

    def build_active_integrals(self) -> None:
        """
        Build effective integrals for all active space clusters 
        """
        mol = self.mf.mol

        h0 = self.mf.energy_nuc()
        h1  = self.mf.get_hcore()

        nbas = self.nbas
        nact = self.Cact.shape[1]


        # Build AO 1rdm 
        rdm1_embed = np.zeros((nbas, nbas))
        rdm1_embed += self.Cfrz @ self.Cfrz.T
        rdm1_embed += self.Cdoc @ self.Cdoc.T

        j, k = self.mf.get_jk(self.mf.mol, rdm1_embed, hermi=1)

        h0 += np.trace(rdm1_embed * ( h1 + .5*j - .25*k))
        
        h1 += j - .5*k;

        # now rotate to MO basis
        h1 = self.Cact.T @ h1 @ self.Cact 
        h2 = pyscf.ao2mo.kernel(mol, self.Cact, aosym="s4", compact=False)
        h2.shape = (nact, nact, nact, nact)

        self.active_h0 = h0
        self.active_h1 = h1
        self.active_h2 = h2
        return

    def extract_cluster_integrals(self, i) -> (float, np.ndarray, np.ndarray):
        """
        Build effective integrals for cluster `i`
        """
        mol = self.mf.mol
        
        ci = self.clusters[i]

        h0  = self.active_h0
        h1  = self.active_h1[ci, :][:, ci]
        h2  = self.active_h2[ci,:,:,:][:,ci,:,:][:,:,ci,:][:, :, :, ci]

        return h0, h1, h2

        # for j in range(len(self.clusters)):
        #     if j == i:
        #         continue
        #     Cj = self.Cact[:, self.clusters[j]]
        #     rdm1 += Cj @ (self.cluster_1rdm[j] @ Cj.T)

        # Ci = self.Cact[:,self.clusters[i]]

        # # now rotate to MO basis
        # h1 = Ci.T @ h1 @ Ci
        # h2 = pyscf.ao2mo.kernel(mol, Ci, aosym="s4", compact=False)
        # h2.shape = (norb_i, norb_i, norb_i, norb_i)

        # return h0, h1, h2


    def do_local_casci(self, i) -> None:
        """
        Solve local problem for cluster i
        """
        
        h0, h1, h2 = self.extract_cluster_integrals(i)

        norb_i = h1.shape[1]

        cisolver = fci.direct_spin1.FCI()
        efci, ci = cisolver.kernel(h1, h2, norb_i, self.fspace[i], ecore=h0, nroots=1, verbose=100)

        self.cluster_energies[i] = efci

        fci_dim = ci.shape[0]*ci.shape[1]
        # (d1a, d1b)  = cisolver.make_rdm1s(ci, norb_i, self.fspace[i])
        
        # d1 = cisolver.make_rdm1(ci, norb_i, nelec)
        (d1a, d1b), (d2aa, d2ab, d2bb)  = cisolver.make_rdm12s(ci, norb_i, self.fspace[i])
        
        d1 = d1a + d1b

        print(" PYSCF 1RDM: ")
        matrix_print(d1a)
        
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
