# ================================================================
# PROJETO: NOISEPRINT - PROTOTIPO (VIDEO + TUTORIAL)
#
# HEITOR MACIEL - 10402559
# VITOR PEPE - 10339754
# VINICIUS MAGNO - 10401365
# KAIKI BELLINI BARBOSA - 10402509
#
# ARQUIVO: UTILS/IMAGE_IO.PY - FUNCOES DE ENTRADA E SAIDA DE IMAGENS
# DESCRICAO: TRATA VALIDACAO, CARREGAMENTO, EXIF E SALVAMENTO DE IMAGENS
# ================================================================
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image, ExifTags

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm"}


# ======================================================
# FUNCAO VALIDAR_CAMINHO_IMAGEM: CONFERE EXISTENCIA E EXTENSAO
# ======================================================
def validar_caminho_imagem(caminho: Path) -> bool:
    return caminho.exists() and caminho.suffix.lower() in ALLOWED_EXTENSIONS


# ==========================================================
# FUNCAO CARREGAR_IMAGEM: LE MATRIZ BGR DO ARQUIVO INDICADO
# ==========================================================
def carregar_imagem(caminho: Path) -> np.ndarray:
    imagem = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if imagem is None:
        raise ValueError("Formato invalido. Use JPG, PNG, BMP ou TIFF.")
    return imagem


# ==============================================================
# FUNCAO EXTRAIR_EXIF: COLETA METADADOS BASICOS DE UM ARQUIVO
# ==============================================================
def extrair_exif(caminho: Path) -> Dict[str, str]:
    try:
        with Image.open(caminho) as img:
            exif = img.getexif()
            if not exif:
                return {}
            tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            return {k: str(tags[k]) for k in ("Model", "Make", "DateTime", "Software") if k in tags}
    except Exception:
        return {}


# ==========================================================
# FUNCAO SALVAR_IMAGEM: ESCREVE MATRIZ EM DISCO NO FORMATO BGR
# ==========================================================
def salvar_imagem(destino: Path, imagem: np.ndarray) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destino), imagem)
