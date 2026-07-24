from .helper import (
    configure_runtime_flags,
    get_script_path_from_argv,
    run_setup_mode,
    validate_script,
)


# Main entry point for the debugger
def main():
    configure_runtime_flags()
    script_path = get_script_path_from_argv()
    validate_script(script_path)
    run_setup_mode(script_path)


if __name__ == "__main__":
    main()
