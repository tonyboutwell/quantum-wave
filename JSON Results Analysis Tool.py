import json
import numpy as np

def text_to_binary(text):
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_text(binary):
    text = ''
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        char = chr(int(byte, 2))
        text += char
    return text

def decode_with_threshold(bit_data, binary_message, shots=8192):
    """
    Decodes the full message by computing a threshold from the first 16 bits (known plaintext '42').
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
        threshold = (max(zero_diffs) + min(one_diffs)) / 2
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

def main():
    # The known plaintext and message:
    known_plaintext = "42"
    secret_message = "Douglas"
    full_message = known_plaintext + secret_message
    binary_message = text_to_binary(full_message)
    shots = 8192

    # Load previously saved result JSON
    filename = "brisbane-result.json"
    with open(filename, 'r') as f:
        qiskit_result = json.load(f)

    # Extracting bit_data from the qiskit_result
    # The code that generated the JSON stored circuits as "bit_i_baseline" and "bit_i_test"
    # We'll reconstruct bit_data similar to how the generation code did.
    results = qiskit_result["results"]
    num_bits = len(binary_message)
    bit_data = []

    # We'll create dictionaries by index
    baseline_dicts = {}
    test_dicts = {}

    for res in results:
        name = res["header"]["name"]
        counts_hex = res["data"]["counts"]
        # Convert hex keys to binary keys (the code that generated was using get_counts which returns normal dict)
        # We can attempt same decode or assume keys are hex. If keys look like hex:
        counts_bin = {}
        for k, v in counts_hex.items():
            # k is hex string like '0x0', '0x8' etc. Convert to binary (4-bit)
            val = int(k, 16)
            bkey = format(val, '04b')
            counts_bin[bkey] = v

        if "baseline" in name:
            bit_index = int(name.split("_")[1])
            baseline_dicts[bit_index] = counts_bin
        elif "test" in name:
            bit_index = int(name.split("_")[1])
            test_dicts[bit_index] = counts_bin

    for i in range(num_bits):
        bit_data.append({
            "baseline": baseline_dicts[i],
            "test": test_dicts[i],
            "original_bit": binary_message[i]
        })

    # Decode using the known plaintext threshold method
    decoded_message, overall_accuracy, decoded_binary, threshold, diffs_per_bit = decode_with_threshold(
        bit_data, binary_message, shots=shots
    )

    # Remove known plaintext from displayed final message
    known_len = len(known_plaintext)
    actual_decoded_message = decoded_message[known_len:]

    print("Threshold calibrated using known plaintext '42' for optimization.")
    print(f"Threshold: {threshold:.4f}")
    print(f"Overall Accuracy: {overall_accuracy*100:.2f}%")

    print("Actual Message:", full_message)
    print("Decoded Message:", actual_decoded_message)

    # Additional Reporting:
    print("\nFull Original Binary Message:")
    print(binary_message)
    print("Full Decoded Binary Message:")
    print(decoded_binary)

    # Compute accuracies for known/unknown portions
    known_binary = binary_message[:16]
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

    # Per-bit analysis
    print("\nPer-bit Analysis:")
    for i, (orig_bit, dec_bit, diff) in enumerate(zip(binary_message, decoded_binary, diffs_per_bit)):
        print(f"Bit {i}: Original={orig_bit}, Decoded={dec_bit}, Diff={diff:.4f}")

if __name__ == "__main__":
    main()
