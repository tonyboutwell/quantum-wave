# Copyright (C) 2024  Tony Boutwell  12/13/24 tonyboutwell@gmail.com
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

import numpy as np
import random
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

###################
# Utility Functions
###################

def text_to_binary(text):
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_text(binary):
    text = ''
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        char = chr(int(byte, 2))
        text += char
    return text

##########################################
# Circuit Generation with Parameterization
##########################################

def create_custom_circuit(apply_phase, phase_angle, steps, rotations):
    """
    Similar to create_quantum_walk_circuit but more flexible:
    - apply_phase: bool, whether to apply a phase at qubit 4.
    - phase_angle: angle for the phase gate or possibly replaced by rotations.
    - steps: how many CRy steps for the quantum walk.
    - rotations: a dict that might specify additional rotations like:
      {
        "rz_angle": float, # rotation around z on a certain qubit
        "ry_angle": float, # rotation around y on a certain qubit
      }

    You can expand this dict to try rotations on different qubits or different points in the circuit.
    For now, we’ll just show one extra rotation (rz) after the phase application as an example.
    """
    num_position_qubits = 4
    qc = QuantumCircuit(num_position_qubits + 3, num_position_qubits)

    # Initial steps same as before
    qc.h(0)
    qc.cx(0, 4)

    if apply_phase:
        # Instead of just p(phase_angle), we could try different rotations:
        qc.p(phase_angle, 4)

    # Apply optional extra rotations if specified
    if rotations.get("rz_angle", 0.0) != 0.0:
        qc.rz(rotations["rz_angle"], 4)

    if rotations.get("ry_angle", 0.0) != 0.0:
        qc.ry(rotations["ry_angle"], 4)

    qc.h(5)
    qc.cx(5, 6)
    qc.cx(4, 0)
    qc.h(4)
    qc.cx(0, 6)
    qc.cz(4, 6)
    qc.h(6)

    for _ in range(steps):
        for i in range(num_position_qubits):
            qc.cry(np.pi/2, 6, i+1)

    qc.measure(range(1, num_position_qubits+1), range(num_position_qubits))
    return qc

def run_experiment(message, phase_angle, steps, rotations, shots=8192):
    """
    Run the encoding and measurement with given parameters.
    Using known plaintext "42" as before.
    """
    known_plaintext = "42"
    full_message = known_plaintext + message
    binary_message = text_to_binary(full_message)

    backend = Aer.get_backend('aer_simulator')

    circuits = []
    for i, bit in enumerate(binary_message):
        # Baseline circuit: apply_phase=False
        qc_baseline = create_custom_circuit(False, phase_angle, steps, rotations)
        qc_baseline.name = f"bit_{i}_baseline"
        circuits.append(qc_baseline)

        # Test circuit: apply_phase if bit is '1'
        apply_phase = (bit == '1')
        qc_test = create_custom_circuit(apply_phase, phase_angle, steps, rotations)
        qc_test.name = f"bit_{i}_test"
        circuits.append(qc_test)

    transpiled = transpile(circuits, backend=backend, optimization_level=1)
    job = backend.run(transpiled, shots=shots)
    result = job.result()

    bit_data = []
    num_bits = len(binary_message)
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

    return bit_data, binary_message

def compute_margin(bit_data, binary_message, shots=8192):
    """
    Compute a margin or a similar metric from the known plaintext bits.
    As before, known plaintext = "42" = 16 bits.
    We'll measure how well diffA separates '0' and '1'.

    Margin = (min(one_diffs) - max(zero_diffs)) for diffA,
    higher margin means better separation.
    """
    known_bits_count = 16
    known_data = bit_data[:known_bits_count]
    known_binary = binary_message[:known_bits_count]

    zero_diffs_A, one_diffs_A = [], []

    for i, info in enumerate(known_data):
        test_0000 = info["test"].get('0000',0)/shots
        test_1000 = info["test"].get('1000',0)/shots
        diffA = test_1000 - test_0000
        if known_binary[i] == '0':
            zero_diffs_A.append(diffA)
        else:
            one_diffs_A.append(diffA)

    if not zero_diffs_A or not one_diffs_A:
        return -1.0  # Invalid scenario if no data
    margin = (min(one_diffs_A) - max(zero_diffs_A))
    return margin

##############################
# Genetic Algorithm Framework
##############################

def random_parameters():
    """
    Generate a random set of parameters.
    Feel free to adjust ranges as desired.

    phase_angle: random angle between 0 and 2π.
    steps: integer steps for the quantum walk, e.g. between 1 and 3.
    rotations: random rotations for rz, ry within a range.
    """
    phase_angle = random.uniform(0, 2*np.pi)
    steps = random.randint(1, 3)
    rz_angle = random.uniform(-np.pi, np.pi)
    ry_angle = random.uniform(-np.pi, np.pi)
    rotations = {"rz_angle": rz_angle, "ry_angle": ry_angle}
    return phase_angle, steps, rotations

def mutate_parameters(params):
    """
    Mutate parameters slightly. This introduces small random changes.
    """
    phase_angle, steps, rotations = params
    # Small mutations
    if random.random() < 0.5:
        phase_angle += random.uniform(-0.1, 0.1)
        # Keep angles in 0 to 2π
        phase_angle %= (2*np.pi)

    if random.random() < 0.3:
        steps = steps + random.choice([-1,1])
        steps = max(1, min(3, steps)) # keep steps in [1,3]

    if random.random() < 0.5:
        rotations["rz_angle"] += random.uniform(-0.1,0.1)

    if random.random() < 0.5:
        rotations["ry_angle"] += random.uniform(-0.1,0.1)

    return phase_angle, steps, rotations

def crossover_parameters(p1, p2):
    """
    Combine parameters from two parents.
    """
    phase_angle_1, steps_1, rotations_1 = p1
    phase_angle_2, steps_2, rotations_2 = p2

    # Crossover by mixing values
    phase_angle = random.choice([phase_angle_1, phase_angle_2])
    steps = random.choice([steps_1, steps_2])
    rz_angle = random.choice([rotations_1["rz_angle"], rotations_2["rz_angle"]])
    ry_angle = random.choice([rotations_1["ry_angle"], rotations_2["ry_angle"]])
    rotations = {"rz_angle": rz_angle, "ry_angle": ry_angle}

    return phase_angle, steps, rotations

def initialize_population(pop_size=10):
    """
    Create an initial population of random parameter sets.
    """
    population = []
    for _ in range(pop_size):
        population.append(random_parameters())
    return population

def evaluate_individual(params, message="Hello"):
    """
    Evaluate one individual (set of parameters) by running the experiment
    and computing the margin.

    We use margin on known plaintext as a fitness measure.
    Higher margin = better.
    """
    phase_angle, steps, rotations = params
    bit_data, binary_message = run_experiment(message, phase_angle, steps, rotations, shots=8192)
    margin = compute_margin(bit_data, binary_message)
    return margin

def genetic_algorithm(generations=10, pop_size=10, elitism=2, mutation_rate=0.3):
    """
    A simple genetic algorithm loop:
    - Initialize population
    - Evaluate fitness (margin)
    - Select elites
    - Crossover and mutate to produce new generation
    - Keep track of the best solutions found.

    This is a rough sketch. You can refine selection strategies (tournament, roulette wheel),
    adjust mutation rates, etc.
    """
    population = initialize_population(pop_size)
    best_solutions = []  # Store (margin, params)

    for gen in range(generations):
        # Evaluate all
        scores = []
        for individual in population:
            score = evaluate_individual(individual)
            scores.append((score, individual))
        scores.sort(key=lambda x: x[0], reverse=True)

        # Keep track of best
        best_solutions.extend(scores)
        best_solutions.sort(key=lambda x: x[0], reverse=True)
        best_solutions = best_solutions[:20]  # Keep top 20 overall

        print(f"Generation {gen}: Best margin = {scores[0][0]:.6f}, Params={scores[0][1]}")

        # Selection: take top 'elitism' as elites directly
        new_population = [x[1] for x in scores[:elitism]]

        # Breed new offspring
        while len(new_population) < pop_size:
            p1 = random.choice(scores[:5])[1]  # pick from top 5 for better convergence
            p2 = random.choice(scores[:5])[1]
            child = crossover_parameters(p1, p2)
            if random.random() < mutation_rate:
                child = mutate_parameters(child)
            new_population.append(child)

        population = new_population

    return best_solutions

if __name__ == "__main__":
    # Run the GA to try and find better parameters
    # Feel free to tweak message, generations, pop_size, etc.
    results = genetic_algorithm(generations=5, pop_size=6, elitism=2, mutation_rate=0.5)

    print("\nTop Candidates Found:")
    for margin, params in results[:5]:
        print(f"Margin={margin:.6f}, Params={params}")
    print("You can rerun with more generations/pop or tweak params to explore more thoroughly.")
