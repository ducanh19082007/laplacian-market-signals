"""
Build the L3 C++ extension (Tarjan SCC + per-SCC Bellman-Ford).

    cd L3_TarjanSCC
    python setup.py build_ext --inplace

Produces  tarjan_arb.<abi>.so  right next to TarjanSCC.py, which imports it.
Pybind11Extension pulls the pybind11 + Python headers in automatically, so no
cmake and no manual -I flags are needed -- just g++ (or any C++17 compiler).
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "tarjan_arb",
        ["TarjanSCC.cpp"],
        cxx_std=17,
    ),
]

setup(
    name="tarjan_arb",
    version="0.1.0",
    description="L3: Tarjan SCC + Bellman-Ford arbitrage search for FOREX_farming",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
#this is to set up that new custom library.
