# ================================================================
# PROJETO: NOISEPRINT - PROTOTIPO (VIDEO + TUTORIAL)
#
# HEITOR MACIEL - 10402559
# VITOR PEPE - 10339754
# VINICIUS MAGNO - 10401365
# KAIKI BELLINI BARBOSA - 10402509
#
# ARQUIVO: SRC/NOISEPRINT.PY - OPERACOES DE RESIDUAL E NORMALIZACAO
# DESCRICAO: OFERECE FUNCAO PARA GERAR NOISEPRINT SIMPLIFICADO VIA FILTRO PASSA-ALTA
# ================================================================
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# ===========================================================
# FUNCAO EXTRAIR_NOISEPRINT: CALCULA RESIDUAL E MAPA NORMAL
# ===========================================================
def extrair_noiseprint(imagem_bgr: np.ndarray, ksize: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Gera mapa de ruido (noiseprint) usando filtro passa-alta basico."""
    imagem_float = imagem_bgr.astype(np.float32) / 255.0
    cinza = cv2.cvtColor(imagem_float, cv2.COLOR_BGR2GRAY)
    borrada = cv2.GaussianBlur(cinza, (ksize, ksize), 0)
    residual = cinza - borrada
    residual_normalizado = cv2.normalize(residual, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    return residual, residual_normalizado
