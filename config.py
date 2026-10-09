"""Rutas y constantes compartidas por todo el proyecto."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent

DATA_RAW = RAIZ / "data" / "raw"
DATA_PROCESSED = RAIZ / "data" / "processed"
OUTPUTS = RAIZ / "outputs"
DOCS = RAIZ / "docs"

TRAIN_RAW = DATA_RAW / "train_data.jsonlines"
TEST_RAW = DATA_RAW / "test_data.jsonlines"

TARGET = "condition"
CLASE_POSITIVA = "used"
SEED = 42
