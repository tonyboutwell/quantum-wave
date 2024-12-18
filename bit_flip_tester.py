import sys
import subprocess
import random
import matplotlib.pyplot as plt

def flip_bit(binary_vector, index):
    """Flip a single bit in the binary vector at the specified index."""
    flipped = list(binary_vector)
    flipped[index] = '1' if binary_vector[index] == '0' else '0'
    return ''.join(flipped)

def call_program(program, mode, input_value):
    """Call the specified Python program in encode (-e) or decode (-d) mode."""
    try:
        result = subprocess.run(["python", program, mode, input_value], capture_output=True, text=True)
        # Parse the output and extract the relevant result
        output_lines = result.stdout.strip().split('\n')
        for line in output_lines:
            if line and ':' not in line:
                return line.strip()
        return ""
    except Exception as e:
        print(f"Error calling program {program}: {e}")
        sys.exit(1)

def run_tests(program, text, iterations):
    """Run the specified number of tests, flipping bits until errors occur."""
    print("Running tests...")
    correct_vector = call_program(program, "-e", text)
    if not correct_vector or not all(c in '01' for c in correct_vector):
        print("Error: Encoding failed or result is not a valid binary vector.")
        sys.exit(1)

    results = []
    binary_length = len(correct_vector)

    for _ in range(iterations):
        test_vector = correct_vector
        flips = 0

        while True:
            # Randomly select a bit to flip
            index = random.randint(0, binary_length - 1)
            test_vector = flip_bit(test_vector, index)
            flips += 1

            # Call the program with the modified vector to decode
            decoded_text = call_program(program, "-d", test_vector)

            # Check if the decoded text matches the original
            if decoded_text != text:
                results.append(flips)
                break
    return results

def plot_histogram(data):
    """Plot a histogram of the results."""
    plt.hist(data, bins=range(1, max(data) + 2), edgecolor='black', align='left')
    plt.title("Distribution of Bit Flips Before Errors Occur")
    plt.xlabel("Number of Bit Flips")
    plt.ylabel("Frequency")
    plt.show()

def print_usage():
    """Print the command-line usage."""
    print("Usage: python bit_flip_tester.py -p <program> -e <text> -n <iterations>")
    print("Options:")
    print("  -p <program>   Name of the Python program to call (e.g., bch_tool.py)")
    print("  -e <text>      Text string to encode and decode")
    print("  -n <iterations> Number of test sets to run")
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

        # Run tests and plot results
        results = run_tests(program, text, iterations)
        plot_histogram(results)
    
    except (ValueError, IndexError) as e:
        print(f"Error: {e}")
        print_usage()
        sys.exit(1)

if __name__ == "__main__":
    main()
