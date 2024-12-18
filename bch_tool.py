import sys
from pyfinite import ffield

class BCH:
    def __init__(self, m=4, t=2):
        """
        Initialize BCH encoder and decoder with Galois Field GF(2^m) and error correction capability t.
        :param m: Degree of the Galois Field
        :param t: Error correction capability
        """
        self.m = m
        self.t = t
        self.field = ffield.FField(m)
        self.n = (1 << m) - 1  # Codeword length = 2^m - 1
        self.k = self.n - m * t  # Message length

    def encode(self, message):
        """Encode a text message into a BCH codeword."""
        # Ensure valid ASCII input
        if not all(32 <= ord(char) <= 126 for char in message):
            raise ValueError("Message contains unsupported characters. Only ASCII printable characters are allowed.")
        
        binary_message = ''.join(format(ord(char), '08b') for char in message)
        codeword = []
        for i in range(0, len(binary_message), self.k):
            chunk = binary_message[i:i+self.k].ljust(self.k, '0')
            parity = [0] * (self.n - self.k)
            for j, bit in enumerate(chunk):
                if bit == '1':
                    for p in range(self.n - self.k):
                        parity[p] ^= self.field.Multiply(1 << (j % self.m), p)
            codeword.extend(list(chunk) + [str(bit % 2) for bit in parity])
        return ''.join(codeword)

    def decode(self, codeword):
        """Decode a BCH codeword into a text message."""
        # Validate that the codeword contains only binary digits
        if not all(c in '01' for c in codeword):
            raise ValueError("Invalid binary vector: Input must contain only '0' and '1'.")
        
        decoded_bits = []
        for i in range(0, len(codeword), self.n):
            chunk = codeword[i:i+self.k]
            decoded_bits.extend(chunk[:self.k])
        decoded_bytes = [int(''.join(decoded_bits[i:i+8]), 2) for i in range(0, len(decoded_bits), 8)]
        return ''.join(chr(byte) for byte in decoded_bytes if byte != 0)

def main():
    bch = BCH(m=4, t=2)
    args = sys.argv

    if "-e" in args:
        index = args.index("-e")
        if index + 1 < len(args):
            message = args[index + 1]
            try:
                encoded = bch.encode(message)
                print(f"Encoded Binary Vector: {encoded}")
            except ValueError as e:
                print(f"Error: {e}")
        else:
            print("Error: No text string provided for encoding.")

    elif "-d" in args:
        index = args.index("-d")
        if index + 1 < len(args):
            binary_vector = args[index + 1]
            try:
                decoded = bch.decode(binary_vector)
                print(f"Decoded Text String: {decoded}")
            except ValueError as e:
                print(f"Error: {e}")
        else:
            print("Error: No binary vector provided for decoding.")
    else:
        print("Usage: python bch_tool.py -e <text> | -d <binary_vector>")

if __name__ == "__main__":
    main()
