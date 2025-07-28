"""Simple BIP39 solver using PyOpenCL.

This script loads the OpenCL kernels from the ``cl`` directory and runs the
``int_to_address`` kernel.  A starting mnemonic index and the target address are
passed to the GPU.  When the target address is found the kernel writes the
matching mnemonic to the output buffer.

The script is intentionally minimal and meant only as a reference.  It does not
implement the original work-server logic or transaction broadcasting that the
Rust version handled.
"""

import argparse
import hashlib
import itertools
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np

try:
    import pyopencl as cl
except Exception as exc:  # pragma: no cover - library may be missing
    raise SystemExit(
        "PyOpenCL is required to run this script. Install it with 'pip install pyopencl'."
    ) from exc


KERNEL_FILES = [
    "common",
    "ripemd",
    "sha2",
    "secp256k1_common",
    "secp256k1_scalar",
    "secp256k1_field",
    "secp256k1_group",
    "secp256k1_prec",
    "secp256k1",
    "address",
    "mnemonic_constants",
    "int_to_address",
]


def load_kernel_source() -> str:
    """Concatenate all kernel source files into a single string."""
    src_parts = []
    base = Path(__file__).resolve().parent / "cl"
    for name in KERNEL_FILES:
        with open(base / f"{name}.cl", "r", encoding="utf-8") as f:
            src_parts.append(f.read())
            src_parts.append("\n")
    return "".join(src_parts)


def parse_wordlist() -> list[str]:
    """Extract the BIP39 English word list from the OpenCL constants."""
    text = (Path(__file__).resolve().parent / "cl" / "mnemonic_constants.cl").read_text()
    start = text.index("{") + 1
    end = text.index("};", start)
    words = [w.strip().strip('"') for w in text[start:end].split(',')]
    return words


ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58decode_check(addr: str) -> bytes:
    """Decode a Base58Check string to bytes."""
    num = 0
    for char in addr:
        num *= 58
        if char not in ALPHABET:
            raise ValueError("invalid base58 character")
        num += ALPHABET.index(char)

    byte_len = (num.bit_length() + 7) // 8
    combined = num.to_bytes(byte_len, "big") if byte_len else b"\x00"

    # Handle leading zeros
    n_pad = len(addr) - len(addr.lstrip("1"))
    decoded = b"\x00" * n_pad + combined.lstrip(b"\x00")

    if len(decoded) != 25:
        raise ValueError("invalid address length")
    payload, checksum = decoded[:-4], decoded[-4:]
    check = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    if check != checksum:
        raise ValueError("checksum mismatch")
    return decoded


def entropy_from_indices(indices: list[int], last7: int) -> int:
    bitstr = "".join(f"{i:011b}" for i in indices) + f"{last7:07b}"
    return int(bitstr, 2)


def run_batch(
    prog: cl.Program,
    queue: cl.CommandQueue,
    ctx: cl.Context,
    hi_batch: np.ndarray,
    lo_batch: np.ndarray,
    target_buf: cl.Buffer,
) -> str | None:
    """Run the GPU kernel on a batch of entropy values."""

    n = hi_batch.size

    output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=120)
    found_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=1)
    hi_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=hi_batch)
    lo_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=lo_batch)

    kernel = prog.int_to_address
    kernel.set_args(hi_buf, lo_buf, output_buf, found_buf, target_buf)

    cl.enqueue_nd_range_kernel(queue, kernel, (n,), None)
    cl.enqueue_barrier(queue)

    output = np.empty(120, dtype=np.uint8)
    found = np.empty(1, dtype=np.uint8)

    cl.enqueue_copy(queue, output, output_buf)
    cl.enqueue_copy(queue, found, found_buf)
    queue.finish()

    if found[0] == 1:
        return bytes(output).rstrip(b"\x00").decode()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU based mnemonic search")
    parser.add_argument("--mnemonic", required=True, help="12 word mnemonic with '*' for unknown words")
    parser.add_argument("--target", required=True,
                        help="target address in standard Base58Check form")
    parser.add_argument("--batch-size", type=int, default=262144,
                        help="number of mnemonics to test per GPU batch")
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="number of CPU threads used to dispatch GPU work",
    )
    args = parser.parse_args()

    print("Starting search at", datetime.utcnow().isoformat(sep=" ", timespec="seconds"))

    try:
        target_bytes = b58decode_check(args.target)
    except ValueError as exc:
        raise SystemExit(f"invalid target address: {exc}")

    words = args.mnemonic.strip().split()
    if len(words) != 12:
        raise SystemExit("mnemonic must contain exactly 12 words")

    wordlist = parse_wordlist()

    first11_indices = []
    unknown_pos = []
    for i, w in enumerate(words[:11]):
        if w == "*":
            first11_indices.append(None)
            unknown_pos.append(i)
        else:
            if w not in wordlist:
                raise SystemExit(f"unknown word: {w}")
            first11_indices.append(wordlist.index(w))

    last_word_given = words[11]
    last_word_idx = None
    if last_word_given != "*":
        if last_word_given not in wordlist:
            raise SystemExit(f"unknown word: {last_word_given}")
        last_word_idx = wordlist.index(last_word_given)

    ctx = cl.create_some_context()
    queues = [cl.CommandQueue(ctx) for _ in range(max(1, args.threads))]
    prog = cl.Program(ctx, load_kernel_source()).build()
    target_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=target_bytes)

    hi_batch: list[int] = []
    lo_batch: list[int] = []

    total_candidates = (2048 ** len(unknown_pos)) * (1 if last_word_idx is not None else 128)
    processed = 0
    found_mnemonic: str | None = None

    executor = ThreadPoolExecutor(max_workers=args.threads)
    pending: set = set()
    sizes: dict[object, int] = {}
    next_queue = 0

    def submit_batch() -> None:
        nonlocal next_queue
        if not hi_batch:
            return
        hi_array = np.array(hi_batch, dtype=np.uint64)
        lo_array = np.array(lo_batch, dtype=np.uint64)
        hi_batch.clear()
        lo_batch.clear()
        q = queues[next_queue % len(queues)]
        next_queue += 1
        fut = executor.submit(run_batch, prog, q, ctx, hi_array, lo_array, target_buf)
        pending.add(fut)
        sizes[fut] = hi_array.size

    def collect_done(block: bool = False) -> bool:
        nonlocal processed, found_mnemonic
        if not pending:
            return False
        wait_time = None if block else 0
        done, _ = wait(pending, timeout=wait_time, return_when=FIRST_COMPLETED)
        for fut in list(done):
            pending.remove(fut)
            batch_size = sizes.pop(fut)
            res = fut.result()
            processed += batch_size
            percent = processed / total_candidates * 100
            print(f"Processed {processed}/{total_candidates} mnemonics ({percent:.2f}%)")
            if res and found_mnemonic is None:
                found_mnemonic = res
                return True
        return False

    outer_break = False
    for combo in itertools.product(range(2048), repeat=len(unknown_pos)):
        indices = first11_indices.copy()
        for pos, idx in zip(unknown_pos, combo):
            indices[pos] = idx

        for last7 in range(128):
            entropy = entropy_from_indices(indices, last7)
            checksum = hashlib.sha256(entropy.to_bytes(16, "big")).digest()[0] >> 4
            w11 = (last7 << 4) | checksum
            if w11 >= 2048:
                continue
            if last_word_idx is not None and w11 != last_word_idx:
                continue

            hi_batch.append(entropy >> 64)
            lo_batch.append(entropy & ((1 << 64) - 1))
            if len(hi_batch) >= args.batch_size:
                submit_batch()
                if collect_done():
                    outer_break = True
                    break
        if outer_break or found_mnemonic is not None:
            break

    if not found_mnemonic:
        submit_batch()
        while pending and not found_mnemonic:
            if collect_done(block=True):
                break

    executor.shutdown(cancel_futures=True)

    if found_mnemonic:
        print(
            "Seed found at",
            datetime.utcnow().isoformat(sep=" ", timespec="seconds"),
        )
        print("Found mnemonic:", found_mnemonic)
    else:
        print("Mnemonic not found")


if __name__ == "__main__":
    main()

