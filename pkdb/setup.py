"""Setup script for PyKokkos Debugger"""

from setuptools import setup

setup(
    name="pkdb",
    version="0.1.0",
    author="PyKokkos Debugger Contributors",
    description="Interactive debugger for PyKokkos workunits",
    long_description="Interactive debugger for PyKokkos kernels with Python front-end and on-device execution.",
    long_description_content_type="text/plain",
    packages=[
        "pkdb",
        "pkdb.runtime",
        "pkdb.core",
        "pkdb.analyzers",
        "pkdb.commands",
        "pkdb.controllers",
        "pkdb.translators",
    ],
    package_dir={"pkdb": "."},
    python_requires=">=3.11",
    install_requires=["pygdbmi>=0.11.0.0"],
    entry_points={
        "console_scripts": [
            "pkdb=pkdb.runtime.main:main",
            "pykokkos-debug=pkdb.runtime.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Debuggers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
