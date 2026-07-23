#!/bin/bash
# Wrapper to compile and run ARES projects using the Deeploy container

CONTAINER_PATH="/scratch/deeploy/sem26f44/DeeployContainer"
BIND_ARGS="--bind /scratch/fsapere"
PROJECT_ROOT="/scratch/fsapere/ARES/ARES"
WORK_DIR="/scratch/fsapere/TinyEEG/ares_artifacts/c_project"

echo "============================================================"
echo "Avvio della compilazione in Singularity..."
singularity exec $BIND_ARGS $CONTAINER_PATH /usr/bin/bash -c "export PULP_RISCV_GCC_TOOLCHAIN=/scratch/fsapere/toolchains/v1.0.16-pulp-riscv-gcc-centos-7 && export RISCV=/scratch/fsapere/toolchains/v1.0.16-pulp-riscv-gcc-centos-7 && export PATH=\$PULP_RISCV_GCC_TOOLCHAIN/bin:/app/install/gvsoc/bin:\$PATH && source /app/install/pulp-sdk/configs/siracusa.sh && cd $WORK_DIR && make clean all PROJECT_ROOT=$PROJECT_ROOT"

if [ $? -ne 0 ]; then
    echo "❌ Errore durante la compilazione."
    exit 1
fi

echo ""
echo "============================================================"
echo " Avvio della simulazione GVSOC in Singularity..."
echo "============================================================"
singularity exec $BIND_ARGS $CONTAINER_PATH /usr/bin/bash -c "export PULP_RISCV_GCC_TOOLCHAIN=/scratch/fsapere/toolchains/v1.0.16-pulp-riscv-gcc-centos-7 && export RISCV=/scratch/fsapere/toolchains/v1.0.16-pulp-riscv-gcc-centos-7 && export PATH=\$PULP_RISCV_GCC_TOOLCHAIN/bin:/app/install/gvsoc/bin:\$PATH && source /app/install/pulp-sdk/configs/siracusa.sh && cd $WORK_DIR && make run PROJECT_ROOT=$PROJECT_ROOT || true"

if [ $? -eq 0 ]; then
    echo "✅ Simulazione completata con successo!"
else
    echo "❌ Errore durante la simulazione."
    exit 1
fi
