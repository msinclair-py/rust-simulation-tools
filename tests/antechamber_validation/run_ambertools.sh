#!/bin/bash
# Run AmberTools antechamber on each test molecule and extract results.
# Usage: bash run_ambertools.sh

AMBER_BIN=~/mamba/envs/ambertools/bin
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

for mol in methanol acetic_acid benzene methylamine; do
    echo "============================================"
    echo "Processing: $mol"
    echo "============================================"

    mkdir -p "amber_${mol}"
    cd "amber_${mol}"

    # Run antechamber with AM1-BCC charges
    $AMBER_BIN/antechamber \
        -i "../${mol}.sdf" \
        -fi sdf \
        -o "${mol}_amber.mol2" \
        -fo mol2 \
        -c bcc \
        -nc 0 \
        -at gaff2 \
        -s 2 2>&1 | tail -5

    if [ -f "${mol}_amber.mol2" ]; then
        echo ""
        echo "--- Atom types and charges (AmberTools) ---"
        # Extract atom section from mol2
        awk '/@<TRIPOS>ATOM/{found=1; next} /@<TRIPOS>/{found=0} found{print}' "${mol}_amber.mol2"
    else
        echo "FAILED: ${mol}_amber.mol2 not produced"
    fi

    cd "$SCRIPT_DIR"
    echo ""
done
