import jax

jax.config.update("jax_enable_x64", True)

from test_speed import run_tests


def main():
    print("Using double precision.\n")
    run_tests(elastic=False)
    run_tests(elastic=True)


if __name__ == "__main__":
    main()
