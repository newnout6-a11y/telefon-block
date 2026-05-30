#!/usr/bin/env python3
"""ECDSA P-256 manifest signing for SpamBlocker (P0 #3).

Helper for generating the keypair and signing `releases/latest/manifest.json`
so that `RemoteUpdateWorker.verifyManifestSignature(...)` will accept it.

Subcommands:
  keygen   --out DIR
      Generates a new ECDSA P-256 keypair and writes:
        - manifest_priv.pem  (PEM, private — keep secret, NEVER commit)
        - manifest_pub.pem   (PEM, public — bundle into APK at
                              app/src/main/assets/manifest_pubkey.pem)

  sign     --key PRIV.pem  --manifest manifest.json
      Signs `manifest.json` with SHA256-with-ECDSA-P256 and writes
      `manifest.json.sig` next to it. The signature format is the standard
      DER ECDSA encoding that Android `Signature.getInstance("SHA256withECDSA")`
      accepts.

  verify   --pub  PUB.pem  --manifest manifest.json
      Local sanity check; useful for CI and pre-flight before publishing.

Dependencies:
  - cryptography (`pip install cryptography`) is the easiest path.
  - If `cryptography` is unavailable, falls back to invoking `openssl`
    (works on macOS/Linux; on Windows install OpenSSL from
    https://slproweb.com/products/Win32OpenSSL.html or use Git-Bash's
    bundled openssl).

Examples (PowerShell):
  py scripts/sign_manifest.py keygen --out releases/keys
  copy releases/keys/manifest_pub.pem app/src/main/assets/manifest_pubkey.pem
  py scripts/sign_manifest.py sign --key releases/keys/manifest_priv.pem \
        --manifest releases/latest/manifest.json
  py scripts/sign_manifest.py verify --pub releases/keys/manifest_pub.pem \
        --manifest releases/latest/manifest.json
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ── crypto backend ──────────────────────────────────────────────────────────
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.exceptions import InvalidSignature
    HAVE_CRYPTOGRAPHY = True
except ImportError:
    HAVE_CRYPTOGRAPHY = False


# ── cryptography backend ────────────────────────────────────────────────────
def _keygen_cryptography(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    priv = ec.generate_private_key(ec.SECP256R1())
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    priv_path = out_dir / "manifest_priv.pem"
    pub_path = out_dir / "manifest_pub.pem"
    priv_path.write_bytes(priv_pem)
    pub_path.write_bytes(pub_pem)
    return priv_path, pub_path


def _sign_cryptography(priv_pem: Path, manifest: Path) -> Path:
    priv = serialization.load_pem_private_key(priv_pem.read_bytes(), password=None)
    if not isinstance(priv, ec.EllipticCurvePrivateKey):
        sys.exit("private key is not ECDSA — expected secp256r1 / P-256")
    data = manifest.read_bytes()
    sig = priv.sign(data, ec.ECDSA(hashes.SHA256()))
    out = manifest.with_suffix(manifest.suffix + ".sig")
    out.write_bytes(sig)
    return out


def _verify_cryptography(pub_pem: Path, manifest: Path) -> bool:
    pub = serialization.load_pem_public_key(pub_pem.read_bytes())
    if not isinstance(pub, ec.EllipticCurvePublicKey):
        sys.exit("public key is not ECDSA")
    data = manifest.read_bytes()
    sig_path = manifest.with_suffix(manifest.suffix + ".sig")
    if not sig_path.exists():
        sys.exit(f"signature file not found: {sig_path}")
    try:
        pub.verify(sig_path.read_bytes(), data, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


# ── openssl fallback ────────────────────────────────────────────────────────
def _need_openssl() -> str:
    path = shutil.which("openssl")
    if not path:
        sys.exit(
            "Neither `cryptography` Python package nor `openssl` binary "
            "available. Install one:\n"
            "  - py -m pip install cryptography\n"
            "  - or install OpenSSL from https://slproweb.com/products/Win32OpenSSL.html"
        )
    return path


def _keygen_openssl(out_dir: Path) -> tuple[Path, Path]:
    openssl = _need_openssl()
    out_dir.mkdir(parents=True, exist_ok=True)
    priv = out_dir / "manifest_priv.pem"
    pub = out_dir / "manifest_pub.pem"
    subprocess.check_call([
        openssl, "ecparam", "-name", "prime256v1", "-genkey", "-noout",
        "-out", str(priv),
    ])
    subprocess.check_call([
        openssl, "ec", "-in", str(priv), "-pubout", "-out", str(pub),
    ])
    return priv, pub


def _sign_openssl(priv_pem: Path, manifest: Path) -> Path:
    openssl = _need_openssl()
    out = manifest.with_suffix(manifest.suffix + ".sig")
    subprocess.check_call([
        openssl, "dgst", "-sha256", "-sign", str(priv_pem),
        "-out", str(out), str(manifest),
    ])
    return out


def _verify_openssl(pub_pem: Path, manifest: Path) -> bool:
    openssl = _need_openssl()
    sig = manifest.with_suffix(manifest.suffix + ".sig")
    if not sig.exists():
        sys.exit(f"signature file not found: {sig}")
    try:
        out = subprocess.check_output(
            [openssl, "dgst", "-sha256", "-verify", str(pub_pem),
             "-signature", str(sig), str(manifest)],
            stderr=subprocess.STDOUT,
        ).decode().strip()
    except subprocess.CalledProcessError as e:
        out = e.output.decode().strip()
    return "Verified OK" in out


# ── CLI ─────────────────────────────────────────────────────────────────────
def cmd_keygen(args: argparse.Namespace) -> None:
    if HAVE_CRYPTOGRAPHY:
        priv, pub = _keygen_cryptography(Path(args.out))
    else:
        priv, pub = _keygen_openssl(Path(args.out))
    print(f"private key (KEEP SECRET, NEVER COMMIT): {priv}")
    print(f"public  key (bundle into APK):          {pub}")
    print()
    print("Next steps:")
    print(f"  1. copy {pub} app/src/main/assets/manifest_pubkey.pem")
    print(f"  2. add {priv.parent}/ to .gitignore")
    print("  3. py scripts/sign_manifest.py sign --key <priv> --manifest releases/latest/manifest.json")


def cmd_sign(args: argparse.Namespace) -> None:
    priv = Path(args.key)
    manifest = Path(args.manifest)
    if HAVE_CRYPTOGRAPHY:
        out = _sign_cryptography(priv, manifest)
    else:
        out = _sign_openssl(priv, manifest)
    print(f"signed → {out} ({out.stat().st_size} bytes)")


def cmd_verify(args: argparse.Namespace) -> None:
    pub = Path(args.pub)
    manifest = Path(args.manifest)
    if HAVE_CRYPTOGRAPHY:
        ok = _verify_cryptography(pub, manifest)
    else:
        ok = _verify_openssl(pub, manifest)
    if ok:
        print("Verified OK")
    else:
        sys.exit("Verification FAILED")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_keygen = sub.add_parser("keygen", help="generate ECDSA P-256 keypair")
    p_keygen.add_argument("--out", required=True, help="output directory")
    p_keygen.set_defaults(func=cmd_keygen)

    p_sign = sub.add_parser("sign", help="sign manifest.json with private key")
    p_sign.add_argument("--key", required=True, help="path to manifest_priv.pem")
    p_sign.add_argument("--manifest", required=True, help="path to manifest.json")
    p_sign.set_defaults(func=cmd_sign)

    p_verify = sub.add_parser("verify", help="locally verify signature")
    p_verify.add_argument("--pub", required=True, help="path to manifest_pub.pem")
    p_verify.add_argument("--manifest", required=True, help="path to manifest.json")
    p_verify.set_defaults(func=cmd_verify)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
