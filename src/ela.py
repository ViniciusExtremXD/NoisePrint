# ================================================================
# PROJETO: NOISEPRINT - PROTOTIPO (VIDEO + TUTORIAL)
#
# HEITOR MACIEL - 10402559
# VITOR PEPE - 10339754
# VINICIUS MAGNO - 10401365
# KAIKI BELLINI BARBOSA - 10402509
#
# ARQUIVO: SRC/ELA.PY - ANALISE ELA COM COMPRESSAO JPEG DE REFERENCIA
# DESCRICAO: GERA MAPA DE ERROS DE COMPRESSAO E HEATMAP COLORIDO
# ================================================================
from __future__ import annotations

import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


# ============================================================
# FUNCAO APLICAR_ELA: CALCULA ERRO DE NIVEIS E MAPA COLORIDO
# ============================================================
def aplicar_ela(imagem_bgr: np.ndarray, qualidade: int = 90, amplify: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """Aplica analise ELA retornando mapa bruto e mapa normalizado."""
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray((imagem_rgb).astype(np.uint8))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=qualidade)
    buffer.seek(0)
    recompress = Image.open(buffer)
    recompress_rgb = np.array(recompress).astype(np.float32)
    diff = np.abs(imagem_rgb.astype(np.float32) - recompress_rgb)
    diff *= amplify
    diff = np.clip(diff, 0, 255)
    diff_bgr = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_RGB2BGR)
    diff_gray = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY)
    diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
    return diff_bgr, cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
