"""Bootstrap administrador de NYX. Solo se ejecuta en consola LOCAL dentro de Docker.

Uso: docker compose exec api python -m app.setup_admin
Jamás escribe credenciales ni hashes en el repositorio.
"""
from __future__ import annotations

import getpass
import sys

from .security import bootstrap_admin, init_auth_db, account_count, validate_username


def main() -> None:
    if not sys.stdin.isatty():
        raise SystemExit("Ejecuta el comando interactivamente (sin -T).")
    init_auth_db()
    if account_count():
        raise SystemExit("NYX ya tiene cuentas. La inicialización solo puede ejecutarse una vez.")
    username = input("Usuario administrador (no aparecerá en la interfaz pública): ").strip()
    try:
        validate_username(username)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    password = getpass.getpass("Contraseña provisional (no se mostrará): ")
    confirm = getpass.getpass("Confirma la contraseña: ")
    if password != confirm:
        raise SystemExit("Las contraseñas no coinciden.")
    try:
        bootstrap_admin(username, password)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print("Administrador creado. Los datos previos quedaron asociados a esa cuenta.")
    print("En el primer inicio de sesión debes reemplazar la contraseña provisional "
          "por una clave de al menos 12 caracteres.")


if __name__ == "__main__":
    main()
