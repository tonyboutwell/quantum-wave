# Quantum Interference-Based Communication: A Qiskit and Python Approach

## Overview
This project demonstrates how a carefully arranged quantum protocol can transform subtle local phase tweaks into clearly distinguishable interference patterns. Starting with a pre-shared entangled state and a known plaintext for calibration, Alice encodes bits using minimal phase shifts. Bob, following a predetermined strategy, then decodes these bits with high accuracy—often near 100% in simulation and about 88% when tested on real IBM quantum hardware.

## Key Idea
Alice and Bob agree on the circuit structure, a known plaintext (“42”), and their measurement basis before sending any data. By examining these known bits, Bob sets a decoding threshold that applies to the entire message. Although Alice’s phase changes are tiny, the chosen quantum walk and unitaries significantly amplify these differences, turning subtle shifts into reliably distinguishable outcome patterns.

## What’s Happening Under the Hood?
- **Pre-Shared Protocol & Calibration:**  
  Before running the experiment, Alice and Bob define the entire circuit and identify a known plaintext segment. Bob uses these reference bits to find the decoding threshold.
  
- **Encoding with Phase:**  
  Alice encodes each bit by applying (or not applying) a minimal phase shift.
  
- **Interference Amplification:**  
  Bob’s operations amplify these phase differences, making their effects clear in the final measurement distributions.
  
- **Measurement & Decoding:**  
  After the circuit runs—either in simulation or on IBM’s quantum hardware—Bob compares the measured outcomes against the known plaintext pattern and applies the derived threshold to decode the message.

## Performance Under Noise and Hardware Tests
Simulations show that the approach holds up under significant depolarizing and amplitude damping noise, and it remains accurate even with fewer shots in simulation. Real hardware tests have also demonstrated about 88% accuracy—showing that these interference patterns can survive real-world imperfections. Higher shot counts may be required on actual devices to maintain accuracy levels.

## Running the Experiment
**Requirements:**  
- Python 3  
- Qiskit  
- Matplotlib

**Process:**  
1. Run the provided Python script.  
2. The script encodes a secret message preceded by “42.”  
3. It applies the agreed-upon circuit, simulates noise if desired, or runs on IBM hardware (using your quantum token).  
4. Known plaintext bits are used to determine a decoding threshold for the entire message.

## In Summary
This repository offers a practical environment for exploring how a fully predetermined quantum protocol, combined with a known calibration segment, might harness quantum interference to decode subtle phase-encoded information—without requiring additional classical communication once the initial conditions are set. While initial simulations and early hardware tests show promising accuracy, further review and analysis are encouraged to fully understand its limitations, underlying mechanisms, and potential applications.
