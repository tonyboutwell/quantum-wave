import sys
import subprocess
import random

def flip_bit(binary_vector, index):
    """Flip a single bit in the binary vector at the specified index."""
    flipped = list(binary_vector)
    flipped[index] = '1' if binary_vector[index] == '0' else '0'
    return ''.join(flipped)

def extract_binary_output(output):
    """Extract the binary vector from program output, ensuring only valid binary characters remain."""
    for line in output.strip().split('\n'):
        line = line.strip()
        # Remove any labels like 'Encoded Binary Vector:'
        if ':' in line:
            line = line.split(':', 1)[1].strip()
        if line and all(c in '01' for c in line):
            return line
    return ""

def call_program(program, mode, input_value):
    """Call the specified Python program in encode (-e) or decode (-d) mode."""
    try:
        result = subprocess.run(["python", program, mode, input_value], capture_output=True, text=True)
        if mode == "-e":
            # Extract binary output for encoding mode
            return extract_binary_output(result.stdout), result.stdout.strip()
        else:
            # Return the raw decoded output for decode mode
            for line in result.stdout.strip().split('\n'):
                if "Decoded Text String:" in line:
                    return line.split(":", 1)[1].strip(), result.stdout.strip()
            return "", result.stdout.strip()  # Default to empty string if not found
    except Exception as e:
        print(f"Error calling program {program}: {e}")
        sys.exit(1)

def run_tests(program, text, iterations, show_modified=False):
    """Run the specified number of tests, flipping bits until errors occur."""
    print("Running tests...")
    correct_vector, _ = call_program(program, "-e", text)
    if not correct_vector:
        print("Error: Encoding failed or result is not a valid binary vector.")
        sys.exit(1)

    print(f"Correct Binary Vector: {correct_vector}")
    binary_length = len(correct_vector)
    results = []

    for set_number in range(iterations):
        flips = 0
        test_vector = correct_vector  # Always reset to the correct binary vector

        while True:
            # Flip one additional bit and reset test vector
            modified_vector = flip_bit(test_vector, random.randint(0, binary_length - 1))
            flips += 1

            # Call the program with the modified vector to decode
            decoded_text, raw_output = call_program(program, "-d", modified_vector)

            # Show the modified vector and decoded text if -x flag is enabled
            if show_modified:
                print(f"Set {set_number + 1}, Flip {flips}: {modified_vector} -> Decoded: '{decoded_text}'")

            # Check if the decoded text matches the original
            if decoded_text != text:
                results.append(flips)
                print(f"Set {set_number + 1}: Error after {flips} bit flips.")
                break
            
            test_vector = modified_vector  # Carry forward the last modified vector
    return results

def summarize_results(results):
    """Summarize and display the count of iterations before errors occurred."""
    summary = {}
    for flips in results:
        summary[flips] = summary.get(flips, 0) + 1

    print("\nSummary of Results:")
    for flips in sorted(summary.keys()):
        print(f"{flips} iteration(s): {summary[flips]} sets")

def print_usage():
    """Print the command-line usage."""
    print("Usage: python bit_flip_tester.py -p <program> -e <text> -n <iterations> [-x]")
    print("Options:")
    print("  -p <program>   Name of the Python program to call (e.g., bch_tool.py)")
    print("  -e <text>      Text string to encode and decode")
    print("  -n <iterations> Number of test sets to run")
    print("  -x             Output the modified binary vector for each iteration")
    print("  -h             Display this help message")

def main():
    if "-h" in sys.argv:
        print_usage()
        sys.exit(0)

    try:
        # Parse command-line arguments
        if "-p" in sys.argv:
            program_index = sys.argv.index("-p") + 1
            program = sys.argv[program_index]
        else:
            raise ValueError("Missing -p flag for program name.")

        if "-e" in sys.argv:
            text_index = sys.argv.index("-e") + 1
            text = sys.argv[text_index]
        else:
            raise ValueError("Missing -e flag for text string.")

        if "-n" in sys.argv:
            iterations_index = sys.argv.index("-n") + 1
            iterations = int(sys.argv[iterations_index])
        else:
            raise ValueError("Missing -n flag for number of iterations.")

        show_modified = "-x" in sys.argv

        # Run tests and summarize results
        results = run_tests(program, text, iterations, show_modified)
        summarize_results(results)
    
    except (ValueError, IndexError) as e:
        print(f"Error: {e}")
        print_usage()
        sys.exit(1)

if __name__ == "__main__":
    main()
