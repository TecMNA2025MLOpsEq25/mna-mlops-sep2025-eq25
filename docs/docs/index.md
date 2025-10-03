# mna-mlops-sep2025-eq25 documentation!

## Description

Repositorio de trabajo sobre la materia de mlops para desarrollo del equipo 25 de la maestría de inteligencia artificial aplicada en el tecnológico de monterrey

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `az storage blob upload-batch -d` to recursively sync files in `data/` up to `dvc-data-remote/data/`.
* `make sync_data_down` will use `az storage blob upload-batch -d` to recursively sync files from `dvc-data-remote/data/` to `data/`.


