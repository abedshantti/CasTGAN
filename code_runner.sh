#!/bin/bash

cat <<EOF > time_python.txt
EOF

# Adult Dataset training and generation

start_adult="$(date -u +%s)"

python3.8 -m main --dataset="adult" --epochs=300 --reproduce_paper

end_adult="$(date -u +%s)"
elapsed_adult="$(($end_adult-$start_adult))"

# Bank Dataset training and generation

start_bank="$(date -u +%s)"

python3.8 -m main --dataset="bank" --epochs=300 --reproduce_paper

end_bank="$(date -u +%s)"
elapsed_bank="$(($end_bank-$start_bank))"

# Cars Dataset training and generation

start_cars="$(date -u +%s)"

python3.8 -m main --dataset="cars" --epochs=300 --reproduce_paper

end_cars="$(date -u +%s)"
elapsed_cars="$(($end_cars-$start_cars))"

# Credit Dataset training and generation

start_credit="$(date -u +%s)"

python3.8 -m main --dataset="credit" --epochs=300 --reproduce_paper

end_credit="$(date -u +%s)"
elapsed_credit="$(($end_credit-$start_credit))"

# Diabetes Dataset training and generation

start_diabetes="$(date -u +%s)"

python3.8 -m main --dataset="diabetes" --epochs=300 --reproduce_paper

end_diabetes="$(date -u +%s)"
elapsed_diabetes="$(($end_diabetes-$start_diabetes))"

# Housing Dataset training and generation

start_housing="$(date -u +%s)"

python3.8 -m main --dataset="housing" --epochs=300 --reproduce_paper

end_housing="$(date -u +%s)"
elapsed_housing="$(($end_housing-$start_housing))"

# cat <<EOF > time_generation.txt
# Adult dataset: $elapsed_adult seconds.
# Bank dataset: $elapsed_bank seconds.
# Cars dataset: $elapsed_cars seconds.
# Credit dataset: $elapsed_credit seconds.
# Diabetes dataset: $elapsed_diabetes seconds.
# Housing dataset: $elapsed_housing seconds.
# EOF