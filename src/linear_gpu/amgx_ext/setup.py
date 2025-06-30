from pathlib import Path
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup
import os

AMGX_INC = os.environ.get("AMGX_INC")
AMGX_LIB = os.environ.get("AMGX_LIB")
define_macros = []
extra_compile_args = {
    'cxx': ['-O3', '-std=c++17']
}
include_dirs = []
library_dirs = []
libraries = []

if AMGX_INC and AMGX_LIB:
    include_dirs.append(AMGX_INC)
    library_dirs.append(AMGX_LIB)
    libraries.extend(['amgx', 'cudart', 'cuda'])
    define_macros.append(('HAS_AMGX', '1'))
else:
    print("WARNING: AMGX_INC/AMGX_LIB не заданы – будет собрана версия без AMGX (fallback на torch.solve)")

setup(
    name='amgx_ext',
    ext_modules=[
        CUDAExtension(
            'amgx_ext',
            ['amgx_ext.cpp'],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)},
) 