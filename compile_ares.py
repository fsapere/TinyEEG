import os
import sys

def main():
    print("Compiling and running ARES on Siracusa (GVSOC)...")
    work_dir = "/scratch/fsapere/TinyEEG/ares_artifacts/c_project"
    project_root = "/scratch/fsapere/ARES/ARES"
    
    cmd = f"""
    export PULP_RISCV_GCC_TOOLCHAIN=/scratch/fsapere/toolchains/v1.0.16-pulp-riscv-gcc-centos-7
    export RISCV=/scratch/fsapere/toolchains/v1.0.16-pulp-riscv-gcc-centos-7
    export PATH=$PULP_RISCV_GCC_TOOLCHAIN/bin:/app/install/gvsoc/bin:$PATH
    source /app/install/pulp-sdk/configs/siracusa.sh
    cd {work_dir}
    make clean all PROJECT_ROOT={project_root} MINIMAL_OUTPUT=0
    make run PROJECT_ROOT={project_root} MINIMAL_OUTPUT=0
    """
    
    # We can write it to a script and execute it via bash if bash is there, or sh
    # Or just execute using os.system with sh
    ret = os.system(cmd)
    sys.exit(ret)

if __name__ == "__main__":
    main()
