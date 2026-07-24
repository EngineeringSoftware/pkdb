"""
Entry point for python -m pkdb
"""

import sys
from .runtime.main import main

if __name__ == "__main__":
    # profile flag fixed
    if "--profile" in sys.argv:
        import cProfile
        import pstats
        from datetime import datetime

        profile_file = f"pkdb_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
        print(f"\n[PROFILING] Running with cProfile, output: {profile_file}")

        profiler = cProfile.Profile()
        profiler.enable()

        try:
            main()
        finally:
            profiler.disable()
            profiler.dump_stats(profile_file)

            # Print summary
            print(f"\n[PROFILING] Profile saved to: {profile_file}")
            stats = pstats.Stats(profile_file)
            stats.strip_dirs()
            stats.sort_stats("cumulative")
            stats.print_stats(20)
    else:
        main()
