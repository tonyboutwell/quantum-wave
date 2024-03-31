# Code Author: Tony Boutwell 3/30/2024 tonyboutwell@gmail.com

from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import Aer
import numpy as np

def run_quantum_walk_interference(steps, backend, apply_phase, phase_angle=np.pi/4):
    # Set the number of qubits used for the position in the quantum walk
    num_position_qubits = 4
    # Create a quantum circuit with extra qubits for teleportation and measurement
    qc = QuantumCircuit(num_position_qubits + 3, num_position_qubits)

    # Initial entanglement between Alice's and Bob's qubits to establish quantum correlation
    qc.h(0) # Apply Hadamard gate to Alice's qubit (qubit 0) to create superposition
    qc.cx(0, 4) # Apply CNOT gate with Alice's qubit as control and Bob's qubit as target (qubit 4)

    # Apply a phase shift to Bob's qubit based on the input flag (simulate changing conditions or messages)
    if apply_phase:
        qc.p(phase_angle, 4) # Apply phase gate to Bob's qubit

    # Prepare additional qubits for teleportation process on Alice's side
    qc.h(5) # Hadamard gate on qubit 5 creates superposition for teleportation
    qc.cx(5, 6)  # CNOT gate entangles qubit 5 with qubit 6 for teleportation

    # Perform Bell-state measurement to correlate the qubits for teleportation
    qc.cx(4, 0) # CNOT gate for Bell-state measurement
    qc.h(4) # Hadamard gate to complete the Bell-state preparation

    # Apply conditional operations based on the Bell-state measurement for teleportation
    qc.cx(0, 6) # Conditional operation based on the result of the Bell-state measurement
    qc.cz(4, 6) # Apply a controlled-Z gate to complete the teleportation process

    # Execute the quantum walk on the teleported state, now on qubit 6
    qc.h(6) # Hadamard gate to start the walk in a superposition state
    for i in range(num_position_qubits):
        # Apply controlled rotations to simulate the quantum walk
        qc.cry(np.pi/2, 6, i+1)

    # Measure the state of the quantum walk qubits to observe the interference pattern
    qc.measure(range(1, num_position_qubits + 1), range(num_position_qubits))

    # Execute the quantum circuit on the provided backend and return the measurement counts
    transpiled_circuit = transpile(qc, backend)
    result = backend.run(transpiled_circuit, shots=8192).result()
    counts = result.get_counts()

    return counts

def analyze_pattern_changes(counts_with_phase, counts_without_phase):
    # Initialize a dictionary to store patterns where significant changes are observed
    significant_patterns = {}
    threshold = 500  # Define a threshold to determine significant changes

    # Compare the counts with and without phase shift to find significant changes
    for pattern in counts_with_phase.keys():
        count_with = counts_with_phase.get(pattern, 0)
        count_without = counts_without_phase.get(pattern, 0)

        # If the difference in counts exceeds the threshold, consider it significant
        if abs(count_with - count_without) > threshold:
            significant_patterns[pattern] = (count_with, count_without)

    return significant_patterns

def is_phase_change_detected(counts, significant_patterns):
    # Check if any of the significant patterns are observed in the provided counts
    for pattern, (count_with, count_without) in significant_patterns.items():
        # If a pattern with a higher count than the baseline is detected, return True
        if counts.get(pattern, 0) > count_without:
            return True
    return False

def text_to_binary(text):
    # Convert the input text to a binary string
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_text(binary):
    # Convert a binary string back to text
    text = ''
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        try:
            char = chr(int(byte, 2))
            if not 0 <= ord(char) < 128:
                raise ValueError(f"Non-ASCII byte: {byte}")
            text += char
        except ValueError as e:
            print(f"Error decoding byte: {byte}, Exception: {e}")
            text += '?'
    return text

def send_message_binary(message, backend, num_steps=10, optimized_phase_angle=np.pi/4):
    # Convert the message to binary and prepare for sending
    binary_message = text_to_binary(message)
    decoded_binary = ''
    
    # Run the quantum walk without phase shift to establish a baseline for comparison
    counts_without_phase = run_quantum_walk_interference(num_steps, backend, False, optimized_phase_angle)
    
    for bit in binary_message:
        # Determine whether to apply a phase shift based on the binary message content
        apply_phase = (bit == '1')
        counts_with_phase = run_quantum_walk_interference(num_steps, backend, apply_phase, optimized_phase_angle)
        
        # Analyze the interference patterns to detect the presence of a phase shift
        significant_patterns = analyze_pattern_changes(counts_with_phase, counts_without_phase)

        # Determine the bit value based on the detected changes in the interference pattern
        if is_phase_change_detected(counts_with_phase, significant_patterns):
            decoded_bit = '1'
        else:
            decoded_bit = '0'
        
        decoded_binary += decoded_bit

    # Convert the binary string back to text to obtain the decoded message
    decoded_message = binary_to_text(decoded_binary)
    return decoded_message

if __name__ == "__main__":
    # Set up the quantum simulation backend
    backend = Aer.get_backend('aer_simulator')

    # Define optimized parameters for the simulation
    optimized_phase_angle = 1.06028752
    num_steps = 10

    # Input message to be sent via quantum communication
    original_message = "Improbability factor infinitely high but not zero. :)"
    print("Original Message:", original_message)

    # Process the message using the quantum communication protocol and display the result
    decoded_message = send_message_binary(original_message, backend, num_steps, optimized_phase_angle)
    print("Decoded Message:", decoded_message)