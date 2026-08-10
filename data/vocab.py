from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        text = sorted(text)
        unique = set()
        self.stoi = {}
        self.itos = {}
        j = 0
        for i, c in enumerate(text):
            if c not in unique:
                unique.add(c)
                self.stoi[c] = i - j
                self.itos[i - j] = c
            else:
                j += 1
        return self.stoi, self.itos
        

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        out = []
        for c in text:
            out.append(stoi[c])
        return out

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        out = ""
        for i in ids:
            out += (itos[i])
        return out




