"""
Configurações da aplicação.
"""
from pathlib import Path

# Diretório base do projeto (raiz)
BASE_DIR = Path(__file__).parent.parent.parent

# Caminho do modelo ML
MODEL_PATH = BASE_DIR / "model" / "modelo_atraso_voos_xgb_te_v1.pkl"

