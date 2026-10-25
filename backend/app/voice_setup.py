"""Descarga bajo demanda de los modelos de voz; nunca se distribuyen en Git."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

BASE = Path(os.environ.get("NYX_VOICE_PATH", "/voices"))
VOICE_NAME = "es_ES-davefx-medium"
PIPER_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/"
PIPER_SHA256 = "6658b03b1a6c316ee4c265a9896abc1393353c2d9e1bca7d66c2c442e222a917"
VOSK_NAME = "vosk-model-small-es-0.42"
VOSK_URL = f"https://alphacephei.com/vosk/models/{VOSK_NAME}.zip"


def download(url: str, target: Path, checksum: str | None = None) -> None:
    if target.is_file():
        if not checksum or hashlib.sha256(target.read_bytes()).hexdigest() == checksum:
            print(f"Ya existe: {target.name}")
            return
        target.unlink()
    temp = target.with_name(target.name + ".part")
    try:
        print(f"Descargando {target.name}...")
        req = urllib.request.Request(url, headers={"User-Agent": "NYX-Voice-Setup/0.2"})
        sha = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=120) as source, temp.open("wb") as dest:
            while True:
                block = source.read(512 * 1024)
                if not block:
                    break
                dest.write(block)
                sha.update(block)
        if checksum and sha.hexdigest() != checksum:
            raise ValueError(f"SHA256 inválido para {target.name}; descarga descartada")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def install_tts() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    download(PIPER_URL + VOICE_NAME + ".onnx", BASE / (VOICE_NAME + ".onnx"), PIPER_SHA256)
    download(PIPER_URL + VOICE_NAME + ".onnx.json", BASE / (VOICE_NAME + ".onnx.json"))
    print("Piper preparado. Voz masculina española davefx-medium (datos CC0).")


def install_stt() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    target = BASE / VOSK_NAME
    if target.is_dir() and (target / "am").is_dir():
        print("Vosk ya está instalado.")
        return
    archive = BASE / (VOSK_NAME + ".zip")
    download(VOSK_URL, archive)
    # Solo se admiten archivos dentro del subdirectorio esperado.
    root = BASE.resolve()
    with zipfile.ZipFile(archive) as zf:
        for item in zf.infolist():
            dest = (BASE / item.filename).resolve()
            if not dest.is_relative_to(root) or not item.filename.startswith(VOSK_NAME + "/"):
                raise ValueError("Archivo ZIP con ruta inesperada")
            if item.file_size > 100_000_000:
                raise ValueError("Archivo ZIP excesivamente grande")
        zf.extractall(BASE)
    if not (target / "am").is_dir():
        raise RuntimeError("Estructura Vosk inválida")
    archive.unlink(missing_ok=True)
    print("Vosk preparado para transcripción local en español.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Instala los modelos de voz locales de NYX")
    parser.add_argument("--tts", action="store_true", help="Descargar Piper")
    parser.add_argument("--stt", action="store_true", help="Descargar Vosk")
    args = parser.parse_args()
    if not args.tts and not args.stt:
        args.tts = args.stt = True
    if args.tts:
        install_tts()
    if args.stt:
        install_stt()


if __name__ == "__main__":
    main()
