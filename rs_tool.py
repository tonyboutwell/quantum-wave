import sys
import reedsolo

class ReedSolomon:
    def __init__(self):
        """
        Initialize Reed-Solomon encoder and decoder.
        Parameters:
        - n = 255: Codeword length (maximum for Reed-Solomon in GF(256))
        - k = 223: Message length
        - t = 16: Error correction capability (2t parity bytes)
        """
        self.n = 255  # Codeword length
        self.k = 223  # Message length
        self.t = 16   # Error correction capability (2 * t parity bytes)
        self.rs = reedsolo.RSCodec(2 * self.t)  # Initialize Reed-Solomon with 32 parity bytes

    def encode(self, message):
        """Encode a message using Reed-Solomon."""
        try:
            data = bytearray(message, 'utf-8')
            encoded = self.rs.encode(data)
            return encoded
        except Exception as e:
            print(f"Encoding Error: {e}")
            sys.exit(1)

def decode_codeword(self, codeword):
    """Decode a Reed-Solomon codeword and correct errors."""
    try:
        corrected = self.rs.decode(codeword)  # Returns only the corrected data
        return corrected.decode('utf-8', errors='ignore')
    except reedsolo.ReedSolomonError:
        return "Decoding Failed"

def text_to_binary(text):
    """Convert a text string to a binary list."""
    return ''.join(format(b, '08b') for b in bytearray(text, 'utf-8'))

def binary_to_text(binary):
    """Convert a binary string to a text string."""
    return bytearray(int(binary[i:i+8], 2) for i in range(0, len(binary), 8)).decode('utf-8', errors='ignore')

def main():
    rs = ReedSolomon()
    args = sys.argv

    if "-e" in args:
        index = args.index("-e")
        if index + 1 < len(args):
            text = args[index + 1]
            encoded = rs.encode(text)
            print("Encoded Binary Vector:", ''.join(format(b, '08b') for b in encoded))
        else:
            print("Error: No text string provided for encoding.")

    elif "-d" in args:
        index = args.index("-d")
        if index + 1 < len(args):
            binary_vector = args[index + 1]
            codeword = bytearray(int(binary_vector[i:i+8], 2) for i in range(0, len(binary_vector), 8))
            decoded_text = rs.decode_codeword(codeword)
            print("Decoded Text String:", decoded_text)
        else:
            print("Error: No binary vector provided for decoding.")

    else:
        print("Usage: python rs_tool.py -e <text> | -d <binary_vector>")

if __name__ == "__main__":
    main()
