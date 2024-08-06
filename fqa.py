import scipy
import vqe_methods 
import operator_pools
import pyscf_helper 

import pyscf
from pyscf import lib
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, cc
from pyscf.cc import ccsd

import openfermion 
from openfermion import *
from tVQE import *
    
r = 1.5
geometry = [('H', (0,0,1*r)), ('H', (0,0,2*r)), ('H', (0,0,3*r)), ('H', (0,0,4*r))]

charge = 0
spin = 0
basis = 'sto-3g'

[n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = pyscf_helper.init(geometry,charge,spin,basis)

print(" n_orb: %4i" %n_orb)
print(" n_a  : %4i" %n_a)
print(" n_b  : %4i" %n_b)

sq_ham = pyscf_helper.SQ_Hamiltonian()
sq_ham.init(h, g, C, S)
print(" HF Energy: %12.8f" %(E_nuc + sq_ham.energy_of_determinant(range(n_a),range(n_b))))

fermi_ham  = sq_ham.export_FermionOperator()

hamiltonian = openfermion.linalg.get_sparse_operator(fermi_ham)

s2 = vqe_methods.Make_S2(n_orb)

#build reference configuration
occupied_list = []
for i in range(n_a):
    occupied_list.append(i*2)
for i in range(n_b):
    occupied_list.append(i*2+1)

fermi_ham += FermionOperator((),E_nuc)

pool = operator_pools.singlet_GSD()
pool.init(n_orb, n_occ_a=n_a, n_occ_b=n_b, n_vir_a=n_orb-n_a, n_vir_b=n_orb-n_b)

onebody_ham = sq_ham.export_onebody()

onebody = openfermion.linalg.get_sparse_operator(onebody_ham, n_qubits = pool.n_spin_orb)

[e,v] = scipy.sparse.linalg.eigsh(onebody.real,1,which='SA',v0=reference_ket.todense())

reference_ket = scipy.sparse.csc_matrix(v[:,0]).transpose()

twobody_ham = sq_ham.export_twobody()

twobody = openfermion.linalg.get_sparse_operator(twobody_ham, n_qubits = pool.n_spin_orb)

ref_energy = reference_ket.T.conj().dot(hamiltonian.dot(reference_ket))[0,0].real

print('reference energy:', ref_energy)

repeat = 0

for k in range(repeat):

    ss = k * 10

    filenam = 'h4_15a_fqa_before_100_H1_ref_%d.csv' % k

    print(filenam)
    
    # [e,v] = vqe_methods.adapt_vqe_layer_grad(fermi_ham, pool, reference_ket, filenam,
    #     theta_thresh=1e-9, adapt_thresh=1e-7)

    [e,v] = vqe_methods.adapt_qaoa_norm_before(fermi_ham, pool, reference_ket, filenam, onebody, twobody,
        theta_thresh=1e-9, adapt_thresh=1e-7, learningrate = ss)

print(" Final ADAPT-VQE energy: %12.8f" %e)
# print(" <S^2> of final state  : %12.8f" %(v.conj().T.dot(s2.dot(v))[0,0].real))
