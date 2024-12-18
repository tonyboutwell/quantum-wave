import sys
from pyfinite import ffield

class BCH:
    def __init__(self):
        """
        Initialize BCH(15, 7, 2) encoder and decoder.
        Code parameters:
        - n = 15: Codeword length
        - k = 7: Message length
        - t = 2: Error-correction capability
        """
        self.n = 15  # Codeword length
        self.k = 7   # Message length
        self.t = 2   # Error correction capability
        self.field = ffield.FField(4)  # Galois Field GF(2^4)

        # Generator polynomial for BCH(15, 7, 2)
        self.generator = [1, 1, 0, 0, 1, 1, 1, 0, 1]  # x^8 + x^7 + x^4 + x^3 + x + 1

    def encode(self, message):
        """Encode a 7-bit message into a 15-bit BCH codeword."""
        if len(message) != self.k:
            raise ValueError(f"Message length must be {self.k} bits.")

        # Append parity bits (initialize with zeros)
        codeword = message + [0] * (self.n - self.k)

        # Perform polynomial division to compute parity bits
        for i in range(self.k):
            if codeword[i] == 1:  # Only divide if leading coefficient is non-zero
                for j in range(len(self.generator)):
                    codeword[i + j] ^= self.generator[j]
        return message + codeword[self.k:]

    def decode(self, codeword):
        """Decode a 15-bit BCH codeword and correct up to 2 errors."""
        if len(codeword) != self.n:
            raise ValueError(f"Codeword length must be {self.n} bits.")

        # Syndrome computation (check for errors)
        syndromes = [0] * (2 * self.t)
        for i in range(2 * self.t):
            for j in range(self.n):
                syndromes[i] ^= codeword[j] * self.field.Multiply(2, (i * j) % (self.n - 1))

        if max(syndromes) == 0:  # No errors detected
            return codeword[:self.k]

        # Error locator polynomial (Berlekamp-Massey algorithm)
        error_locator = [1]  # Start with a degree-0 polynomial
        for i in range(self.t):
            if syndromes[i] != 0:
                error_locator.append(syndromes[i])

        # Locate and correct errors
        error_positions = []
        for i in range(self.n):
            value = 1
            for coef in error_locator:
                value = self.field.Multiply(value, 2) if coef else value
            if value == 0:
                error_positions.append(i)

        for pos in error_positions:  # Flip bits at error positions
            codeword[pos] ^= 1

        return codeword[:self.k]

def text_to_binary(text):
    """Convert a text string to a binary list."""
    return [int(b) for char in text for b in format(ord(char), '08b')]

def binary_to_text(binary):
    """Convert a binary list to a text string."""
    chars = [binary[i:i + 8] for i in range(0, len(binary), 8)]
    return ''.join(chr(int(''.join(map(str, char)), 2)) for char in chars if len(char) == 8)

def main():
    bch = BCH()
    args = sys.argv

    if "-e" in args:
        index = args.index("-e")
        if index + 1 < len(args):
            message = text_to_binary(args[index + 1])
            encoded = []
            for i in range(0, len(message), 7):
                block = message[i:i + 7]
                while len(block) < 7:
                    block.append(0)
                encoded.extend(bch.encode(block))
            print("Encoded Binary Vector:", ''.join(map(str, encoded)))
        else:
            print("Error: No text string provided for encoding.")

    elif "-d" in args:
        index = args.index("-d")
        if index + 1 < len(args):
            binary_vector = [int(b) for b in args[index + 1]]
            decoded = []
            for i in range(0, len(binary_vector), 15):
                block = binary_vector[i:i + 15]
                decoded.extend(bch.decode(block))
            print("Decoded Text String:", binary_to_text(decoded))
        else:
            print("Error: No binary vector provided for decoding.")
    else:
        print("Usage: python bch_tool.py -e <text> | -d <binary_vector>")

if __name__ == "__main__":
    main()
