import qiskit.aqua.components.variational_forms as vf
import math
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.utils.run_circuits import find_regs_by_name
from qiskit.quantum_info import Pauli
from qiskit import Aer, QuantumCircuit
import numpy as np
from qiskit.aqua.components.optimizers.cg import Optimizer

class QNG(Optimizer):
    CONFIGURATION = {
        'name': 'QNG',
        'description': 'Quantum Natural Gradient',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'qng_schema',
            'type': 'object',
            'properties': {
                'maxiter': {
                    'type': 'integer',
                    'default': 20
                },
                'disp': {
                    'type': 'boolean',
                    'default': False
                },
                'eps': {
                    'type': 'number',
                    'default': 1.4901161193847656e-08
                },
                'eta': {
                    'type': 'number',
                    'default': 1.4901161193847656e-08
                }
            },
            'additionalProperties': False
        },
        'support_level': {
            'gradient': Optimizer.SupportLevel.supported,
            'bounds': Optimizer.SupportLevel.ignored,
            'initial_point': Optimizer.SupportLevel.required
        },
        'options': ['maxiter', 'disp', 'eps', 'eta'],
        'optimizer': ['local']
    }

    def __init__(self, num_qbits, ry_depth, eps=1.4901161193847656e-08, eta=0.0001):
        self.validate(locals())
        super().__init__()
        for k, v in locals().items():
            if k in self._configuration['options']:
                self._options[k] = v

        self.num_qbits = num_qbits
        self.ry_depth = ry_depth

    def optimize(self, num_vars, objective_function, gradient_function=None, variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function, variable_bounds, initial_point)

        epsilon = self._options['eps']
        eta = self._options['eta']
        theta = initial_point

        gradient_function = Optimizer.wrap_function(Optimizer.gradient_num_diff,
                                                    (objective_function, epsilon, self._max_evals_grouped))(theta)

        QGT = self.determine_QGT(theta)
        inv_QGT = np.linalg.pinv(QGT)

        print(gradient_function)

        updated_theta = theta - eta * inv_QGT @ gradient_function

        return updated_theta, objective_function(updated_theta), 1 # no of iters

    def determine_QGT(self, theta):
        QGT = []
        pauli_str = 'I' * self.num_qbits

        for layer_idx in range(1, self.ry_depth+1):
            for qb_idx in range(self.num_qbits):
                QGT.append(1 - self.calc_expectataion(pauli_str[qb_idx], self.get_subcricuit(theta, layer_idx)) ** 2)

        QGT = np.diag(QGT)

        return QGT

    def calc_expectataion(self, pauli_str, sub_circuit):
        qubit_op = WeightedPauliOperator([[1 / 4, Pauli.from_label(pauli_str)]])
        sv_mode = False

        qi = QuantumInstance(backend=Aer.get_backend('statevector_simulator'), shots=1, seed_simulator=100, seed_transpiler=2)

        # Make sure that the eval quantum/ classical registers in the circuit are named 'q'/'c'
        qc = qubit_op.construct_evaluation_circuit(statevector_mode=sv_mode,
                                                   wave_function=sub_circuit,
                                                   qr=find_regs_by_name(sub_circuit, 'q'),
                                                   use_simulator_operator_mode=True)

        result = qi.execute(qc)
        avg, std = qubit_op.evaluate_with_result(statevector_mode=sv_mode,
                                                 result=result,
                                                 use_simulator_operator_mode=True,
                                                 )

        return avg

    def get_subcricuit(self, theta_params, layer):
        assert layer >= 0

        circ = vf.RY(num_qubits=self.num_qbits, depth=layer, skip_final_ry=True)
        p = theta_params[0:self.num_qbits * layer]

        return circ.construct_circuit(parameters=p)


if __name__ == "__main__":
    ## TESTING

    from qiskit.aqua.translators.ising import tsp
    from qiskit.aqua.components.variational_forms import RY

    n = 2
    ins = tsp.random_tsp(n)
    qubitOp, offset = tsp.get_tsp_qubitops(ins)

    n_qbits = qubitOp.num_qubits
    n_layers = 3


    seed = 10598

    QNG_opti = QNG(n_qbits, ry_depth=n_layers)
    ry = RY(n_qbits, depth=n_layers, entanglement='linear')

    vqe = VQE(qubitOp, ry, optimizer=QNG_opti)

    backend = Aer.get_backend('statevector_simulator')
    quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)

    result = vqe.run(quantum_instance)
