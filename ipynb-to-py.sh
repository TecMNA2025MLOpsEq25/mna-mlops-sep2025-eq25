#!/bin/zsh
# ==========================================================
# Script: ipynb-to-py.sh
# Descripción: Convierte y limpia notebooks para ejecución en DVC.
# ==========================================================

NOTEBOOK_DIR="notebooks/a01313663"
SCRIPTS_DIR="src/scripts/a01313663"
CONVERTED=0

echo "Buscando notebooks nuevos o modificados..."

for nb in "$NOTEBOOK_DIR"/*.ipynb; do
    base_name=$(basename "$nb" .ipynb)
    py_file="$SCRIPTS_DIR/${base_name}.py"

    if [[ ! -f "$py_file" || "$nb" -nt "$py_file" ]]; then
        echo "Exportando: $nb → $py_file"
        jupyter nbconvert --to script "$nb" --output-dir "$SCRIPTS_DIR" >/dev/null 2>&1

        # --- Limpieza automática --- #

        # Elimina encabezados de Jupyter (shebang y codificación)
        sed -i '' '/^#!\/usr\/bin\/env python/d' "$py_file"
        sed -i '' '/^# *coding[:=]/d' "$py_file"

        # Elimina anotaciones de celdas Jupyter
        sed -i '' '/^# In\[/d' "$py_file"

        # Reemplaza display() por print()
        sed -i '' 's/display(\(.*\))/print(\1)/g' "$py_file"

        # Asegura que df.head() y df.info() impriman
        sed -i '' 's/^\( *\)\(df[^=]*\.head()\)/\1print(\2)/' "$py_file"
        sed -i '' 's/^\( *\)\(df[^=]*\.info()\)/\1print(\2)/' "$py_file"

        # Agrega guard clause si no existe
        if ! grep -q "__main__" "$py_file"; then
            echo -e "\nif __name__ == '__main__':\n    pass  # main guard added" >> "$py_file"
        fi

        ((CONVERTED++))
    fi
done

if [[ $CONVERTED -eq 0 ]]; then
    echo "Todos los notebooks están actualizados."
else
    echo "$CONVERTED notebooks convertidos y limpiados."
fi
