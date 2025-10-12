#!/usr/bin/env python3
"""
tools/find_extra_dvc.py

Detecta archivos .dvc (creados con `dvc add`) cuyos outputs
NO estén cubiertos por los outs declarados en dvc.yaml.

Modo safe (no borra nada). Requiere PyYAML.
"""

import os
import yaml
import argparse
import subprocess
from pathlib import Path

def load_yaml_outs(dvc_yaml_path="dvc.yaml"):
    with open(dvc_yaml_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    outs = []
    for stage in d.get("stages", {}).values():
        for o in stage.get("outs", []):
            # outs pueden ser strings o dicts con 'path'
            if isinstance(o, dict):
                path = o.get("path")
            else:
                path = o
            if path:
                outs.append(os.path.normpath(path))
    return outs

def parse_dvc_file_outs(dvc_file_path):
    with open(dvc_file_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    outs = []
    for o in d.get("outs", []):
        if isinstance(o, dict):
            p = o.get("path")
        else:
            p = o
        if p:
            outs.append(os.path.normpath(p))
    return outs

def abs_path(p):
    return os.path.normpath(os.path.abspath(os.path.join(os.getcwd(), p)))

def is_covered(out, yaml_outs):
    # normalizar absolutos
    out_abs = abs_path(out)
    for yo in yaml_outs:
        yo_abs = abs_path(yo)
        # si yaml_out es igual al out
        if out_abs == yo_abs:
            return True
        # si yaml_out es carpeta y out está dentro
        try:
            common = os.path.commonpath([out_abs, yo_abs])
        except ValueError:
            # paths en distintos montajes? no cubre
            continue
        if common == yo_abs:  # out está bajo yo_abs
            return True
    return False

def git_tracked(path):
    try:
        subprocess.run(["git", "ls-files", "--error-unmatch", path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def human_size(path):
    try:
        s = os.path.getsize(path)
        for unit in ['B','KB','MB','GB']:
            if s < 1024.0:
                return f"{s:.1f}{unit}"
            s /= 1024.0
        return f"{s:.1f}TB"
    except Exception:
        return "n/a"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dvc-yaml", default="dvc.yaml", help="Ruta a dvc.yaml")
    parser.add_argument("--list-all", action="store_true", help="Mostrar todos los .dvc y sus outs")
    args = parser.parse_args()

    if not os.path.exists(args.dvc_yaml):
        print(f"ERROR: no se encontró {args.dvc_yaml} en el directorio actual.")
        return

    yaml_outs = load_yaml_outs(args.dvc_yaml)
    print("Outs declarados en dvc.yaml:")
    for yo in yaml_outs:
        print("  -", yo)
    print()

    # encontrar .dvc files (ignorar carpeta .dvc interna)
    dvc_files = []
    for root, dirs, files in os.walk("."):
        # evitar ./.dvc
        if root.startswith("./.dvc") or root == "./.dvc":
            continue
        for f in files:
            if f.endswith(".dvc"):
                dvc_files.append(os.path.join(root, f))

    if not dvc_files:
        print("No se encontraron archivos .dvc.")
        return

    candidates = []

    for dvcf in sorted(dvc_files):
        outs = parse_dvc_file_outs(dvcf)
        if args.list_all:
            print(f"{dvcf}:")
            for o in outs:
                exists = os.path.exists(o)
                print(f"   - {o}  exists={exists}")
        # ¿alguno de los outs está cubierto por dvc.yaml?
        covered_any = False
        for o in outs:
            if is_covered(o, yaml_outs):
                covered_any = True
                break
        if not covered_any:
            candidates.append((dvcf, outs))

    if not candidates:
        print("No se detectaron .dvc independientes que NO estén cubiertos por dvc.yaml.")
        return

    print("\nArchivos .dvc candidatos (no referenciados por dvc.yaml):\n")
    for dvcf, outs in candidates:
        print("->", dvcf)
        for o in outs:
            p = o
            exists = os.path.exists(p)
            tracked = git_tracked(dvcf)
            size = human_size(p) if exists else "n/a"
            print(f"     out: {p}   exists={exists}   size={size}")
        print(f"   git-tracked .dvc? {tracked}")
        print()

    print("Revisa con cuidado antes de eliminar. Para borrar manualmente:")
    print("  git rm --cached <archivo.dvc>")
    print("  rm <archivo.dvc>")

if __name__ == "__main__":
    main()
