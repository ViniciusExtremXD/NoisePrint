# ================================================================
# PROJETO: NOISEPRINT - PROTOTIPO (VIDEO + TUTORIAL)
#
# HEITOR MACIEL - 10402559
# VITOR PEPE - 10339754
# VINICIUS MAGNO - 10401365
# KAIKI BELLINI BARBOSA - 10402509
#
# ARQUIVO: SRC/PRNU.PY - ESTIMATIVA SIMPLIFICADA DO PADRAO PRNU
# DESCRICAO: CALCULA RESIDUAL POR BLUR GAUSSIANO E GERA MAPA COLORIDO
# ================================================================
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# ============================================================
# FUNCAO EXTRAIR_PRNU: ESTIMA PRNU E GERA MAPA COLORIDO
# ============================================================
def extrair_prnu(imagem_bgr: np.ndarray, h: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Estima o padrao PRNU usando filtro Wiener simplificado."""
    imagem_float = imagem_bgr.astype(np.float32) / 255.0
    cinza = cv2.cvtColor(imagem_float, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cinza, (h*2+1, h*2+1), 0)
    wiener = cinza - blur
    wiener_norm = cv2.normalize(wiener, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    mapa_color = cv2.applyColorMap((wiener_norm * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
    return wiener, mapa_color
