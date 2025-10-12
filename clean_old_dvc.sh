#!/bin/zsh
# ==========================================================
# Script: clean_old_dvc.sh
# Descripción: Interactivamente elimina .dvc antiguos que ahora
#              están en dvc.yaml
# ==========================================================

DRY_RUN=false

# Listar todos los archivos .dvc, ignorando directorios .dvc/
old_dvc_files=( $(find . -type f -name "*.dvc" ! -path "./.dvc/*") )

# Extraer todos los outs del dvc.yaml usando Python
yaml_outs=( $(python3 - <<'EOF'
import yaml
with open('dvc.yaml') as f:
    dvc = yaml.safe_load(f)
outs = []
for stage in dvc.get('stages', {}).values():
    for out in stage.get('outs', []):
        outs.append(out)
print("\n".join(outs))
EOF
) )

echo "\nArchivos encontrados (.dvc):"
for f in "${old_dvc_files[@]}"; do echo "  $f"; done

echo "\nArchivos en dvc.yaml:"
for f in "${yaml_outs[@]}"; do echo "  $f"; done

echo "\nArchivos .dvc antiguos que ahora están en dvc.yaml:\n"

for dvc_file in "${old_dvc_files[@]}"; do
    # Extraer outs del .dvc usando Python
    target_files=( $(python3 - <<EOF
import yaml
with open("$dvc_file") as f:
    d = yaml.safe_load(f)
for out in d.get("outs", []):
    print(out.get("path"))
EOF
) )

    for out in "${target_files[@]}"; do
        # Checar si el out está en dvc.yaml
        if [[ "${yaml_outs[@]}" =~ "${out}" ]]; then
            echo "CANDIDATO: $dvc_file → $out"
            
            read "confirm?  ¿Eliminar este archivo .dvc y desregistrarlo de git? (y/n) "
            
            if [[ "$confirm" =~ ^[Yy]$ ]]; then
                if [[ "$DRY_RUN" == true ]]; then
                    echo "    ✔ Dry-run: no se elimina (cambiar DRY_RUN=false para borrar)"
                else
                    git rm --cached "$dvc_file"
                    rm "$dvc_file"
                    echo "    ✔ Eliminado"
                fi
            else
                echo "    ✖ Omitido"
            fi
        fi
    done
done

echo "\nRevisión completa."
