#!/usr/bin/bash
export PULP_RISCV_GCC_TOOLCHAIN=/scratch/fsapere/toolchains/v1.0.16-pulp-riscv-gcc-centos-7
export RISCV=/scratch/fsapere/toolchains/v1.0.16-pulp-riscv-gcc-centos-7
export PATH=$PULP_RISCV_GCC_TOOLCHAIN/bin:/app/install/gvsoc/bin:$PATH
source /app/install/pulp-sdk/configs/siracusa.sh
cd /scratch/fsapere/TinyEEG/ares_artifacts/c_project
make clean all run PROJECT_ROOT=/scratch/fsapere/ARES/ARES MINIMAL_OUTPUT=0 > full_sim_output.txt 2>&1
