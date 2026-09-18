"""Generate project-local TLS credentials; never modify the OS trust store."""

import argparse
import csv
import os
from pathlib import Path
import secrets
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def protect(directory):
    if os.name == "nt":
        user = subprocess.run(
            ["whoami", "/user", "/fo", "csv", "/nh"], check=True, capture_output=True, text=True
        ).stdout.strip()
        sid = next(csv.reader([user]))[1]
        subprocess.run(
            ["icacls", str(directory), "/inheritance:r", "/grant:r", f"*{sid}:(OI)(CI)F"],
            check=True, capture_output=True,
        )
    else:
        directory.chmod(0o700)


def prepare_credentials(directory, openssl=None):
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError("Credentials directory already exists; never overwrite it silently.")
    openssl = openssl or shutil.which("openssl") or r"C:\ProgramData\miniconda3\Library\bin\openssl.exe"
    if not Path(openssl).is_file():
        raise FileNotFoundError("OpenSSL not found; provide --openssl with its executable path.")
    directory.mkdir(parents=True, mode=0o700)
    protect(directory)
    config = directory / "openssl.cnf"
    config.write_text(
        "[req]\ndistinguished_name=dn\nprompt=no\n[dn]\nCN=FedSEA Local Development CA\n"
        "[server]\nbasicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n"
        "subjectAltName=DNS:localhost,IP:127.0.0.1\n"
        "subjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer\n",
        encoding="ascii",
    )

    def run(*args):
        result = subprocess.run(
            [str(openssl), *args], cwd=directory, capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode:
            raise RuntimeError(f"OpenSSL failed ({args[0]}): {result.stderr[-1000:]}")

    run("req", "-x509", "-newkey", "rsa:3072", "-noenc", "-sha256", "-days", "365",
        "-config", str(config), "-keyout", "ca.key", "-out", "ca.crt",
        "-addext", "basicConstraints=critical,CA:TRUE,pathlen:0",
        "-addext", "keyUsage=critical,keyCertSign,cRLSign", "-addext", "subjectKeyIdentifier=hash")
    run("req", "-new", "-newkey", "rsa:3072", "-noenc", "-sha256", "-config", str(config),
        "-subj", "/CN=localhost", "-keyout", "server.key", "-out", "server.csr")
    run("x509", "-req", "-in", "server.csr", "-CA", "ca.crt", "-CAkey", "ca.key",
        "-CAcreateserial", "-out", "server.crt", "-days", "90", "-sha256",
        "-extfile", str(config), "-extensions", "server")
    run("verify", "-CAfile", "ca.crt", "-verify_hostname", "localhost", "server.crt")
    run("verify", "-CAfile", "ca.crt", "-verify_ip", "127.0.0.1", "server.crt")
    (directory / "token.txt").write_text(secrets.token_hex(32) + "\n", encoding="ascii")
    if os.name != "nt":
        for file in directory.iterdir():
            file.chmod(0o600)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "certs/localhost")
    parser.add_argument("--openssl")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(ROOT / "certs") or directory == ROOT / "certs":
        raise ValueError("Project credentials must stay in a new subdirectory under certs/.")
    result = prepare_credentials(directory, args.openssl)
    print(f"TLS credentials ready: {result}")
    print("Server certificate: 90 days; local CA: 365 days. No OS trust-store changes.")
    print("Keep ca.key, server.key, and token.txt private. Existing credentials are never overwritten.")


if __name__ == "__main__":
    main()
