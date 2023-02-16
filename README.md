Code for the paper ["Finding Collisions for Round-Reduced Romulus-H"](https://eprint.iacr.org/2022/1630)
========================================================================================================


## Usage

To run the Python code in the `sat_modelling` directory you first need to run the following command to compile the Cython extension modules.
```bash
cd sat_modelling
python3 setup.py build_ext -i
```

You may need the following dependencies
```bash
apt install cython3 python3-z3
pip install pycryptosat
```
