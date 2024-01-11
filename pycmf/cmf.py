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
        
        self.C = np.zeros((mf.mol.nbas,0))   # mo coeffs 
        self.S = self.mf.get_ovlp()          

        # Active space clustering
        self.clusters = []
        self.fspace = []

    
    def init(self, C:np.ndarray, clusters:list[list[int]], fspace:list[tuple[int,int]]):
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
        self.C = C
        self.clusters = clusters 
        self.fspace   = fspace 
        self.cluster_energies:list[float] = [0.0 for i in clusters]
        self.cluster_1rdm:list[np.ndarray] = [np.zeros((len(i),len(i))) for i in clusters]
        self.roots = [1 for i in clusters] 

        self.cluster_core_energies:list[float] = [0.0 for i in clusters]

    def rotate_mo_coeffs(self, U:np.ndarray) -> None:
        self.C = self.C @ U

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

    def update_veff_matrices(self) -> None:
        """Compute the 'embedding' term for all clusters simultaneously.
        """
        full_dm = self.get_rdm1()

        nelec = sum([sum(i) for i in self.fspace])
        # Start with full dm, and remove each cluster's local dm
        dms = [full_dm*1.0 for i in self.clusters]
        for i,ci in enumerate(self.clusters):
            Ci = self.C[:,ci]
            dms[i] -= Ci @ self.cluster_1rdm[i] @ Ci.T

            # nelec_i = sum(self.fspace[i])
            # warn(abs(np.trace(dms[i]@self.S) - (nelec - nelec_i)) < 1e-16)
        
        self.v_effs = self.mf.get_veff(self.mf.mol, dms, hermi=1)


        # energy_core += numpy.einsum('ij,ji', core_dm[0], hcore[0])
        # energy_core += numpy.einsum('ij,ji', core_dm[1], hcore[1])
        # energy_core += numpy.einsum('ij,ji', core_dm[0], corevhf[0]) * .5
        # energy_core += numpy.einsum('ij,ji', core_dm[1], corevhf[1]) * .5


    def get_rdm1(self):
        N = self.mf.mol.nao
        dm = np.zeros((N,N))
        for i,ci in enumerate(self.clusters):
            Ci = self.C[:,ci]
            dm += Ci @ self.cluster_1rdm[i] @ Ci.T

        # print(np.trace(dm @ self.S))
        return dm

    def build_cluster_integrals(self, i) -> (float, np.ndarray, np.ndarray):
        """Build integrals for local cluster
        """ 

        Ni = len(self.clusters[i])

        Ci = self.C[:,self.clusters[i]] 
        h0 = self.mf.energy_nuc()
        h1 = self.mf.get_hcore() + self.v_effs[i]
        h1 = Ci.T @ h1 @ Ci

        h2 = pyscf.ao2mo.kernel(self.mf.mol, Ci, aosym="s4", compact=False)
        h2.shape = (Ni, Ni, Ni, Ni)
        
        return h0, h1, h2

    def do_local_casci(self, i, verbose=1) -> None:
        """
        Solve local problem for cluster i

        This will update both the cluster energy and the cluster_rdm1 data
        """

        if verbose > 0:
            print()
            print(" Solve local FCI problem for cluster ", i) 
        
        h0, h1, h2 = self.build_cluster_integrals(i)

        norb_i = h1.shape[1]

        cisolver = fci.direct_spin1.FCI()
        efci, ci = cisolver.kernel(h1, h2, norb_i, self.fspace[i], ecore=h0, nroots=1, verbose=100)

        self.cluster_energies[i] = efci

        fci_dim = ci.shape[0]*ci.shape[1]
        # (d1a, d1b)  = cisolver.make_rdm1s(ci, norb_i, self.fspace[i])
        
        # d1 = cisolver.make_rdm1(ci, norb_i, nelec)
        (d1a, d1b), (d2aa, d2ab, d2bb)  = cisolver.make_rdm12s(ci, norb_i, self.fspace[i])
        
        d1 = d1a + d1b

        self.cluster_1rdm[i] = d1

        if verbose > 3:
            print(" PYSCF 1RDM: ")
            matrix_print(d1a)
        
            occs = np.linalg.eig(d1)[0]

            print(" Occupation Numbers:")
            [print("%4i %12.8f"%(i,occs[i])) for i in range(len(occs))]
        
        if verbose > 0:
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
