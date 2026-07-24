import argparse
import os

try:
    from boltzmann import BoltzmannDSMC
    from kernels import stack_profiler as sp
except:
    from boltzmann import BoltzmannDSMC
    from kernels import stack_profiler as sp


def main(
    num_steps, num_particles, gridpts, ecov, join, space, store_results, compare_results
):
    boltzmann = BoltzmannDSMC(num_particles, gridpts, ecov, join, space)
    stable: bool = store_results or compare_results
    boltzmann.run(num_steps, stable)

    if store_results:
        boltzmann.store_results(num_steps)

    if compare_results:
        boltzmann.compare_results(num_steps)


def run(
    num_steps,
    num_particles,
    gridpts,
    no_warmup,
    ecov,
    join,
    space,
    store_results,
    compare_results,
):
    if not no_warmup:
        print("warming up")
        sp.profile(main, 10, num_particles, gridpts, ecov, join, space, False, False)
        sp.reset_profile()

    print("RESET")
    sp.profile(
        main,
        num_steps,
        num_particles,
        gridpts,
        ecov,
        join,
        space,
        store_results,
        compare_results,
    )

    profile = sp.get_profile()
    sp.reset_profile()

    return profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--steps", type=int, default=400)
    parser.add_argument("-N", "--num_particles", type=int, default=200000)
    parser.add_argument("-g", "--gridpts", type=int, default=256)
    parser.add_argument("-nw", "--no_warmup", action="store_true")
    parser.add_argument("-e", "--Ecov", type=float, default=5.0)
    parser.add_argument("-j", "--join", action="store_true")
    parser.add_argument("-space", "--space", type=str)
    parser.add_argument("-store", "--store_results", action="store_true")
    parser.add_argument("-compare", "--compare_results", action="store_true")
    parser.add_argument(
        "--verbose-timings",
        action="store_true",
        help="Print per-phase timings from boltzmann.py",
    )
    parser.add_argument(
        "--verbose-timings-bigsteps",
        type=int,
        default=1,
        help="Only print timings for the first N timesteps (bigstep loop)",
    )
    parser.add_argument(
        "--verbose-timings-ng",
        type=int,
        default=0,
        help="Only print timings for GPU index ng==N",
    )
    args = parser.parse_args()

    if args.verbose_timings:
        os.environ["PKDB_BOLTZ_VERBOSE_TIMINGS"] = "1"
        os.environ["PKDB_BOLTZ_VERBOSE_TIMINGS_BIGSTEPS"] = str(
            args.verbose_timings_bigsteps
        )
        os.environ["PKDB_BOLTZ_VERBOSE_TIMINGS_NG"] = str(args.verbose_timings_ng)

    profile = run(
        args.steps,
        args.num_particles,
        args.gridpts,
        args.no_warmup,
        args.Ecov,
        args.join,
        args.space,
        args.store_results,
        args.compare_results,
    )
    from pprint import pprint

    pprint(profile)
