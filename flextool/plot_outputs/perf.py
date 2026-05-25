import time
from contextlib import contextmanager

# Performance tracking accumulator
PERF_STATS: dict = {}


@contextmanager
def time_block(name: str, verbose: bool = False):
    """Context manager to time a block of code and accumulate stats."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if name not in PERF_STATS:
            PERF_STATS[name] = {'total': 0}
        PERF_STATS[name]['total'] += elapsed
        if verbose:
            print(f"  [{name}] {elapsed:.4f}s")


def print_perf_summary() -> None:
    """Print summary of performance statistics."""
    if not PERF_STATS:
        return

    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Operation':<50}{'Total':>10}")
    print("-"*80)

    for name, stats in sorted(PERF_STATS.items(), key=lambda x: x[1]['total'], reverse=True):
        print(f"{name:<50} {stats['total']:>10.4f}s")

    print("="*80 + "\n")
