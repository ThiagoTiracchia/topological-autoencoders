#!/bin/bash



# Comando 1
#echo "=== Ejecutando SwissRoll ConoCycles ==="
# python -m exp.train_model -F results/SwissRoll with experiments/myexpsv2/Spheres/ConoCycles.json device='cuda'

# Comando 2  
# echo "=== Ejecutando SwissRoll  ConoSinCycles ==="
# python -m exp.train_model -F results/SwissRoll with experiments/myexpsv2/SwissRoll/ConoSinCycles.json device='cuda'

# Comando 3  
# echo "=== Ejecutando  SwissRoll  TopoRedgeSymmetric ==="
# python -m exp.train_model -F results/SwissRoll with experiments/myexpsv2/SwissRoll/TopoRegEdgeSymmetric.json device='cuda'


# Comando 4
echo "=== Ejecutando Spheres ConoCycles  ==="
python -m exp.train_model -F results/Spheres with experiments/myexpsv2/Spheres/ConoCycles.json device='cuda'

# Comando 5
echo "=== Ejecutando Spheres ConoSinCycles ==="
#python -m exp.train_model -F results/Spheres with experiments/myexpsv2/Spheres/ConoSinCycles.json device='cuda'

# Comando 6
# echo "=== Ejecutando Spheres TopoRegEdgeSymmetric ==="
python -m exp.train_model -F results/Spheres with experiments/myexpsv2/Spheres/TopoRegEdgeSymmetric.json device='cuda'


echo "Todos los experimentos se ejecutaron"

echo "Graficando"

# Generar gráficos
mkdir -p graficos/Spheres
# mkdir -p graficos/SwissRoll


python scripts/plot_metrics.py results/Spheres --output graficos/Spheres
# python scripts/plot_metrics.py results/SwissRoll --output graficos/SwissRoll