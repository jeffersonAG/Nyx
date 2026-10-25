"""Tokenizador determinista de bytes UTF-8, entrenable sin vocabulario de terceros.

Se usa para el primer prototipo. No comprime español como un tokenizador BPE;
futuros cambios de tokenizador requieren reentrenar desde cero.
"""
BOS = 256
EOS = 257
PAD = 258
VOCAB_SIZE = 259


def encode(text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
    if not isinstance(text, str):
        raise TypeError("El texto debe ser str.")
    return ([BOS] if add_bos else []) + list(text.encode("utf-8")) + (
        [EOS] if add_eos else []
    )


def decode(ids: list[int]) -> str:
    """Ignora BOS/PAD, detiene la generación en EOS."""
    buffer = bytearray()
    for index in ids:
        token = int(index)
        if token == EOS:
            break
        if 0 <= token <= 255:
            buffer.append(token)
    return bytes(buffer).decode("utf-8", errors="replace")
