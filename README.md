# BIP39 Solver GPU

This project was used to iterate through all possible BIP39 mnemonics given a certain amount of known words and a target address. It utilizes all available GPUs on the system. This is not high quality production ready code, it was thrown together as quickly as possible. Read more here: <insert medium post>.

## Requirements

The new implementation is written in Python and uses PyOpenCL.  You will need
Python 3, the ``pyopencl`` package and OpenCL drivers for your GPU.  On
Debian/Ubuntu based systems you can install the generic development package:

```
sudo apt-get install ocl-icd-opencl-dev
```

On Windows make sure you have installed Python 3 from https://python.org and
the latest drivers for your GPU that provide an OpenCL runtime (NVIDIA, AMD or
Intel).

Then install the Python dependency:

```
pip install pyopencl numpy
```

## Usage

Run the solver by providing a 12 word mnemonic and the target address. Unknown
words can be represented with ``*``. ``target`` should be provided as a standard
Bitcoin address in Base58Check format.  Use ``--batch-size`` to control how many
candidate mnemonics are processed per GPU launch.  A value around ``262144``
works well on an RTX 3070.

After each batch the solver prints how many candidate mnemonics have been
tested and the overall percentage completed so you can monitor progress.
It also reports the start time and the exact time a matching mnemonic is found.

```
python solver.py --mnemonic "abandon ability * about above absent * * * * * *" \
    --target 3HX5tttedDehKWTTGpxaPAbo157fnjn89s --batch-size 262144
```

The program will iterate over all possible combinations of the ``*`` positions
and print the mnemonic once the provided target address is generated.

