from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit, transpile
import numpy as np
from qiskit_aer import Aer, AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error

def text_to_binary(text):
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_text(binary):
    text = ''
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        char = chr(int(byte, 2))
        text += char
    return text

def create_quantum_walk_circuit(apply_phase, phase_angle, steps, rz_angle=0.0, ry_angle=0.0):
    num_position_qubits = 4
    qc = QuantumCircuit(num_position_qubits + 3, num_position_qubits)

    # Initial entanglement (Alice:0, Bob:4)
    qc.h(0)
    qc.cx(0, 4)

    # Apply phase if needed
    if apply_phase:
        qc.p(phase_angle, 4)
        if rz_angle != 0.0:
            qc.rz(rz_angle, 4)
        if ry_angle != 0.0:
            qc.ry(ry_angle, 4)

    # Teleportation-like steps
    qc.h(5)
    qc.cx(5, 6)
    qc.cx(4, 0)
    qc.h(4)
    qc.cx(0, 6)
    qc.cz(4, 6)

    # Quantum walk steps
    qc.h(6)
    for _ in range(steps):
        for i in range(num_position_qubits):
            qc.cry(np.pi/2, 6, i+1)

    qc.measure(range(1, num_position_qubits+1), range(num_position_qubits))
    return qc

def decode_with_threshold(bit_data, binary_message, shots=3000):
    # Compute threshold from known plaintext '42' bits (first 16 bits)
    known_bits_count = 16
    known_data = bit_data[:known_bits_count]
    known_binary = binary_message[:known_bits_count]

    zero_diffs = []
    one_diffs = []

    # Using diffA = p('1000') - p('0000') as the metric
    for i, info in enumerate(known_data):
        baseline_prob = info["test"].get('0000',0)/shots
        ref_prob = info["test"].get('1000',0)/shots
        diff = ref_prob - baseline_prob
        if known_binary[i] == '0':
            zero_diffs.append(diff)
        else:
            one_diffs.append(diff)

    # Simple threshold from known plaintext
    if zero_diffs and one_diffs:
        threshold = (max(zero_diffs) + min(one_diffs))/2
    else:
        threshold = 0.0

    decoded_bits = []
    diffs_per_bit = []
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
    # Insert your IBM Quantum API token once and run:
    # QiskitRuntimeService.save_account(token="IBM_API_TOKEN_HERE", channel="ibm_quantum", overwrite=True)
    #
    # After the account is saved, you can comment the line above out and then just use:
    # service = QiskitRuntimeService()
    #
    # When you're ready to run on hardware, set use_simulator = False and uncomment:
    service = QiskitRuntimeService()
    backend = service.backend("ibm_brisbane")

    phase_angle = 1.9921
    steps = 1
    rz_angle = 0.1278
    ry_angle = -0.3335

    # Toggle between simulator and hardware
    use_simulator = False  # True = Aer simulator, False = IBM Hardware
    use_noise = True      # True = Add noise model to simulator runs, False = No noise in simulation

    known_plaintext = "42"
    secret_message = "Wow"

    # Convert secret message to binary
    secret_binary = text_to_binary(secret_message)

    # Strategy: Start with "42", after each bit of secret_message add a known '0' bit, end with "42".
    # This gives multiple known reference points.
    # full pattern: 42 + [secret_bit + '0'] * len(secret_bits) + 42
    # For example, if secret_message = "Hello", we interleave '0' after each bit.
    # We'll have to strip these extra bits out after decoding.

    # Binary of known_plaintext '42'
    known_binary = text_to_binary(known_plaintext)

    # Interleave a '0' after each secret bit
    interleaved_bits = []
    for b in secret_binary:
        interleaved_bits.append(b)
        interleaved_bits.append('0')  # control bit after each message bit

    # Final binary message:
    # Start with known_plaintext
    # Then interleaved message bits
    # Then end with known_plaintext again
    enhanced_binary_message = known_binary + ''.join(interleaved_bits) + known_binary

    binary_message = enhanced_binary_message
    shots=3000

    if use_simulator:
        if use_noise:
            # Define a noise model (same as before)
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
            # No noise
            backend = Aer.get_backend('aer_simulator')
    else:
        # For real hardware:
        service = QiskitRuntimeService(channel="ibm_quantum")
        backend = service.backend("ibm_brisbane")
        pass

    # Prepare circuits
    circuits = []
    for i, bit in enumerate(binary_message):
        qc_baseline = create_quantum_walk_circuit(False, phase_angle, steps, rz_angle, ry_angle)
        qc_baseline.name = f"bit_{i}_baseline"
        circuits.append(qc_baseline)

        apply_phase = (bit == '1')
        qc_test = create_quantum_walk_circuit(apply_phase, phase_angle, steps, rz_angle, ry_angle)
        qc_test.name = f"bit_{i}_test"
        circuits.append(qc_test)

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

    # Decode results using known plaintext
    decoded_message, overall_accuracy, decoded_binary, threshold, diffs_per_bit = decode_with_threshold(
        bit_data, binary_message, shots=shots
    )

    print("Threshold calibrated using known plaintext '42' for optimization.")

    # Now we must remove the known pattern from the decoded message:
    # We started with: 42 + (secret_bit + '0') * each bit + 42
    # That's known_binary at start and end. Length of known_binary = 16 bits.
    # Remove first 16 bits and last 16 bits from decoded_binary:
    stripped = decoded_binary[16:-16]

    # Now 'stripped' is (secret_bit + '0') repeated for the entire secret message.
    # Remove every second bit (the '0' control bit) to get the original secret bits back.
    secret_recovered_bits = stripped[0::2]  # take every other bit starting at 0
    actual_decoded_message = binary_to_text(secret_recovered_bits)

    print(f"Threshold: {threshold:.4f}")
    print(f"Overall Accuracy: {overall_accuracy*100:.2f}%")
    print("Decoded Binary:", decoded_binary)
    print("Decoded Message (after removing known patterns):", actual_decoded_message)

    # Per-bit analysis (debugging info)
    print("\nPer-bit Analysis:")
    for i, (orig_bit, dec_bit, diff) in enumerate(zip(binary_message, decoded_binary, diffs_per_bit)):
        print(f"Bit {i}: Original={orig_bit}, Decoded={dec_bit}, Diff={diff:.4f}")
