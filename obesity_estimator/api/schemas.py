# -*- coding: utf-8 -*-
"""
obesity_estimator/api/schemas.py

Esquemas Pydantic para validación de entrada en la API.
Compatible con Pydantic v2.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal


class ObesityInput(BaseModel):
    # --- columnas categóricas ---
    Gender: Literal["Male", "Female"] = Field(..., description="Género de la persona")
    CAEC: Literal["no", "Sometimes", "Frequently", "Always"] = Field(..., description="Consumo de alimentos entre comidas")
    CALC: Literal["no", "Sometimes", "Frequently", "Always"] = Field(..., description="Frecuencia de consumo de alcohol")
    MTRANS: Literal["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"] = Field(
        ..., description="Medio de transporte principal"
    )

    # --- columnas binarias (sí/no) ---
    family_history_with_overweight: Literal["yes", "no"] = Field(..., description="Historial familiar de sobrepeso")
    FAVC: Literal["yes", "no"] = Field(..., description="Consume alimentos hipercalóricos frecuentemente")
    SMOKE: Literal["yes", "no"] = Field(..., description="Fuma habitualmente")
    SCC: Literal["yes", "no"] = Field(..., description="Monitorea consumo de calorías")

    # --- columnas numéricas ---
    Age: float = Field(..., ge=0, le=120, description="Edad en años")
    Height: float = Field(..., ge=0.5, le=2.5, description="Altura en metros")
    Weight: float = Field(..., ge=10, le=300, description="Peso en kilogramos")
    FCVC: float = Field(..., ge=0, le=3, description="Frecuencia de consumo de vegetales (1–3)")
    NCP: float = Field(..., ge=1, le=4, description="Número de comidas principales al día")
    CH2O: float = Field(..., ge=0, le=3, description="Consumo de agua diario (litros)")
    FAF: float = Field(..., ge=0, le=3, description="Frecuencia de actividad física")
    TUE: float = Field(..., ge=0, le=2, description="Tiempo usando dispositivos electrónicos (horas)")

    # --- validadores adicionales ---
    @field_validator("Height", "Weight")
    def check_positive(cls, v, field):
        if v <= 0:
            raise ValueError(f"{field.name} debe ser positivo")
        return v
