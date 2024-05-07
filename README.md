Quantum Wave Communication Experiment using Qiskit and Python

Overview:
This experiment utilizes entanglement, quantum teleportation, and quantum walks to simulate instantaneous information transfer between two parties, Alice and Bob. The experiment aims to simulate instant communication without the need for a classical verification channel, while preserving the entanglement connection between Alice and Bob.

Simulation Key Steps:

Entanglement Distribution:
Alice and Bob share a pair of entangled qubits.
Alice transports her entangled qubit to a distant location.

Phase Change Encoding:
Alice encodes her message by applying phase changes to her qubit based on a pre-shared interval.
The phase changes instantly affect the state of Bob's qubit due to the entanglement.

Quantum Teleportation:
Instead of directly using the entangled qubit, Bob performs quantum teleportation to transfer the state of his entangled qubit to a separate qubit.
This preserves the entanglement connection between Alice and Bob, allowing for continuous communication without consuming the entanglement.

Quantum Walk and Interference:
Bob performs a quantum walk on his side using the teleported qubit.
The quantum walk creates interference patterns sensitive to the phase changes applied by Alice.

Measurement and Decoding:
Bob measures the position qubits after the quantum walk.
By comparing the measured patterns with baseline patterns, Bob can detect the phase changes applied by Alice.
The changes in the interference patterns reflect the message encoded by Alice.

It is important to note this works in the Qiskit simulation since it appears to not be truly random. It would require more research/access to quantum computers to test against true quantum randomness.
