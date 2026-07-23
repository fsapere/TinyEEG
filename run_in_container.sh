#!/bin/bash
export PULP_RISCV_GCC_TOOLCHAIN=/scratch/fsapere/toolchains/v1.0.16-pulp-riscv-gcc-centos-7
export RISCV=/scratch/fsapere/toolchains/v1.0.16-pulp-riscv-gcc-centos-7
export PATH=$PULP_RISCV_GCC_TOOLCHAIN/bin:/app/install/gvsoc/bin:$PATH
source /app/install/pulp-sdk/configs/siracusa.sh
cd /scratch/fsapere/TinyEEG/ares_artifacts/c_project
make clean all PROJECT_ROOT=/scratch/fsapere/ARES/ARES MINIMAL_OUTPUT=0 > make_log.txt 2>&1
make run PROJECT_ROOT=/scratch/fsapere/ARES/ARES MINIMAL_OUTPUT=0 >> make_log.txt 2>&1
