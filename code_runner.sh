#!/bin/bash

# Bank Dataset training and generation

start_bank="$(date -u +%s)"

python3.8 -m main --dataset="bank" --epochs=300 --reproduce_paper

end_bank="$(date -u +%s)"
elapsed_bank="$(($end_bank-$start_bank))"


cat <<EOF > time_generation.txt
Bank dataset: $elapsed_bank seconds.
EOF