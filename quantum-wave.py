# Copyright (C) 2024  Tony Boutwell  3/30/24 tonyboutwell@gmail.com
# This program is free software: you can redistribute it and/or modify it under the terms of the 
# GNU General Public License as published by the Free Software Foundation, either version 3 of the License, 
# or (at your option) any later version.

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit, transpile
import numpy as np
from qiskit_aer import Aer, AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error

def text_to_binary(text):
    """
    Converts a given ASCII text string into its binary representation.
    Each character is represented by an 8-bit binary code.
    For example:
      "A" -> "01000001"
    """
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_text(binary):
    """
    Converts a binary string (in 8-bit segments) back into ASCII text.
    For example:
      "01000001" -> "A"
    """
    text = ''
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        char = chr(int(byte, 2))
        text += char
    return text

def create_quantum_walk_circuit(apply_phase, phase_angle, steps, rz_angle=0.0, ry_angle=0.0):
    """
    Constructs a quantum circuit to encode a single bit of information by applying (or not) a phase.
    The circuit then undergoes a series of known operations and a quantum walk, resulting in distinct
    interference patterns that reveal whether a '0' or '1' was encoded.
    """
    num_position_qubits = 4
    qc = QuantumCircuit(num_position_qubits + 3, num_position_qubits)

    # Initial known state preparation
    qc.h(0)
    qc.cx(0, 4)

    # If encoding '1', apply a small phase + optional rotations to qubit 4
    if apply_phase:
        qc.p(phase_angle, 4)
        if rz_angle != 0.0:
            qc.rz(rz_angle, 4)
        if ry_angle != 0.0:
            qc.ry(ry_angle, 4)

    # Additional operations to establish a known entangled state and prepare for the quantum walk
    qc.h(5)
    qc.cx(5, 6)
    qc.cx(4, 0)
    qc.h(4)
    qc.cx(0, 6)
    qc.cz(4, 6)

    # Initiate the quantum walk steps
    qc.h(6)
    for _ in range(steps):
        for i in range(num_position_qubits):
            qc.cry(np.pi/2, 6, i+1)

    qc.measure(range(1, num_position_qubits+1), range(num_position_qubits))
    return qc

def decode_with_threshold(bit_data, binary_message, shots=8192):
    """
    Decodes the full message by computing a threshold from known plaintext ('42').

    Method:
    - Compute a threshold from the first 16 bits (known plaintext).
    - Use a metric (p('1000') - p('0000')) to distinguish '0' from '1'.
    - Apply this threshold to all bits to decode the message.

    Returns:
    - decoded_message: ASCII decoded message
    - overall_accuracy: Fraction of bits correctly decoded
    - decoded_binary: Binary string of decoded message
    - threshold: Computed threshold value
    - diffs_per_bit: List of difference values per bit
    """
    known_bits_count = 16
    known_data = bit_data[:known_bits_count]
    known_binary = binary_message[:known_bits_count]

    zero_diffs = []
    one_diffs = []

    # Compute differences for known plaintext bits
    for i, info in enumerate(known_data):
        baseline_prob = info["test"].get('0000',0)/shots
        ref_prob = info["test"].get('1000',0)/shots
        diff = ref_prob - baseline_prob
        if known_binary[i] == '0':
            zero_diffs.append(diff)
        else:
            one_diffs.append(diff)

    # Determine threshold from known plaintext
    if zero_diffs and one_diffs:
        threshold = (max(zero_diffs) + min(one_diffs))/2
    else:
        threshold = 0.0

    decoded_bits = []
    diffs_per_bit = []
    # Decode all bits using the computed threshold
    for info in bit_data:
        baseline_prob = info["test"].get('0000',0)/shots
        ref_prob = info["test"].get('1000',0)/shots
        diff = ref_prob - baseline_prob
        decoded_bit = '1' if diff > threshold else '0'
        decoded_bits.append(decoded_bit)
        diffs_per_bit.append(diff)

    decoded_message = binary_to_text(''.join(decoded_bits))
    correct_bits = sum(db == ob for db, ob in zip(decoded_bits, binary_message))
    overall_accuracy = correct_bits / len(binary_message)

    return decoded_message, overall_accuracy, ''.join(decoded_bits), threshold, diffs_per_bit

if __name__ == "__main__":
    # Main Execution Flow:
    # 1. Set simulation/hardware mode and noise preferences.
    # 2. Define encoding parameters and messages.
    # 3. Construct circuits, run them, and decode results.

    # Toggle between simulator and hardware
    use_simulator = True
    use_noise = True

    # Pre-agreed parameters
    phase_angle = 1.9921
    steps = 1
    rz_angle = 0.1278
    ry_angle = -0.3335

    known_plaintext = "42"
    secret_message = "Douglas"
    full_message = known_plaintext + secret_message
    binary_message = text_to_binary(full_message)
    shots = 8192

    # If using simulator, set up local Aer simulator (with optional noise)
    if use_simulator:
        if use_noise:
            # Noise model parameters
            single_qubit_error_rate = 0.05
            two_qubit_error_rate = 0.10
            amp_damp_param = 0.05

            noise_model = NoiseModel()
            error_1q_depol = depolarizing_error(single_qubit_error_rate, 1)
            error_2q_depol = depolarizing_error(two_qubit_error_rate, 2)
            error_amp_damp = amplitude_damping_error(amp_damp_param)
            combined_1q = error_1q_depol.compose(error_amp_damp)

            noise_model.add_all_qubit_quantum_error(combined_1q, ['rz', 'ry', 'p', 'h'])
            noise_model.add_all_qubit_quantum_error(error_2q_depol, ['cx', 'cz'])

            backend = AerSimulator(noise_model=noise_model)
        else:
            # No noise simulation
            backend = Aer.get_backend('aer_simulator')
    else:
        # Running on IBM Quantum Hardware:
        # Ensure token/account is configured. Then select a backend (e.g. "ibm_brisbane").
        service = QiskitRuntimeService(token="YOUR_TOKEN_HERE", channel="ibm_quantum")
        backend = service.backend("ibm_brisbane")

    # Construct circuits for each bit
    circuits = []
    for i, bit in enumerate(binary_message):
        # Baseline circuit
        qc_baseline = create_quantum_walk_circuit(False, phase_angle, steps, rz_angle, ry_angle)
        qc_baseline.name = f"bit_{i}_baseline"
        circuits.append(qc_baseline)

        # Test circuit
        apply_phase = (bit == '1')
        qc_test = create_quantum_walk_circuit(apply_phase, phase_angle, steps, rz_angle, ry_angle)
        qc_test.name = f"bit_{i}_test"
        circuits.append(qc_test)

    # Transpile and run the circuits
    transpiled = transpile(circuits, backend=backend, optimization_level=1)
    job = backend.run(transpiled, shots=shots)
    print("Job submitted. Job ID:", job.job_id())
    result = job.result()
    print("Job completed!")

    # Gather results
    num_bits = len(binary_message)
    bit_data = []
    for i in range(num_bits):
        baseline_index = 2*i
        test_index = 2*i + 1
        baseline_counts = result.get_counts(baseline_index)
        test_counts = result.get_counts(test_index)
        bit_data.append({
            "baseline": baseline_counts,
            "test": test_counts,
            "original_bit": binary_message[i]
        })

    # Decode using known plaintext '42' as a reference
    decoded_message, overall_accuracy, decoded_binary, threshold, diffs_per_bit = decode_with_threshold(
        bit_data, binary_message, shots=shots
    )

    # Remove known plaintext from displayed final message
    known_len = len(known_plaintext)
    actual_decoded_message = decoded_message[known_len:]

    print("Threshold calibrated using known plaintext '42' for optimization.")
    print(f"Threshold: {threshold:.4f}")
    print(f"Overall Accuracy: {overall_accuracy*100:.2f}%")

    # Show the actual message without the "42" prefix
    print("Actual Message:", secret_message)
    print("Decoded Message:", actual_decoded_message)

    # Additional Reporting:  
    # Show the full original binary vs. the full decoded binary for clarity.
    print("\nFull Original Binary Message:")
    print(binary_message)
    print("Full Decoded Binary Message:")
    print(decoded_binary)

    # Compute known and unknown bit accuracy separately
    known_binary = binary_message[:16]  # '42' is 2 chars = 16 bits
    known_decoded = decoded_binary[:16]
    unknown_binary = binary_message[16:]
    unknown_decoded = decoded_binary[16:]

    def accuracy_stats(original_bits, decoded_bits):
        correct = sum(o == d for o, d in zip(original_bits, decoded_bits))
        total = len(original_bits)
        accuracy = (correct / total) * 100 if total > 0 else 0.0

        zeros_in_original = sum(o == '0' for o in original_bits)
        ones_in_original = sum(o == '1' for o in original_bits)

        zeros_wrong = sum((o != d) and (o == '0') for o, d in zip(original_bits, decoded_bits))
        ones_wrong = sum((o != d) and (o == '1') for o, d in zip(original_bits, decoded_bits))

        zero_wrong_rate = (zeros_wrong / zeros_in_original * 100) if zeros_in_original > 0 else 0.0
        one_wrong_rate = (ones_wrong / ones_in_original * 100) if ones_in_original > 0 else 0.0

        return accuracy, zero_wrong_rate, one_wrong_rate

    known_accuracy, known_zero_wrong, known_one_wrong = accuracy_stats(known_binary, known_decoded)
    unknown_accuracy, unknown_zero_wrong, unknown_one_wrong = accuracy_stats(unknown_binary, unknown_decoded)
    overall_accuracy_percent = overall_accuracy * 100

    print("\nAccuracy Breakdown:")
    print(f"Known Bits Accuracy:   ~{known_accuracy:.2f}% | 0's wrong {known_zero_wrong:.2f}% | 1's wrong {known_one_wrong:.2f}%")
    print(f"Unknown Bits Accuracy: ~{unknown_accuracy:.2f}% | 0's wrong {unknown_zero_wrong:.2f}% | 1's wrong {unknown_one_wrong:.2f}%")
    print(f"Overall Average Accuracy: {overall_accuracy_percent:.2f}% | (All bits combined)")

    # Now the Per-bit analysis follows.
    print("\nPer-bit Analysis:")
    for i, (orig_bit, dec_bit, diff) in enumerate(zip(binary_message, decoded_binary, diffs_per_bit)):
        print(f"Bit {i}: Original={orig_bit}, Decoded={dec_bit}, Diff={diff:.4f}")
