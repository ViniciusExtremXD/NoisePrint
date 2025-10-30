# ================================================================
# PROJETO: NOISEPRINT - PROTOTIPO (VIDEO + TUTORIAL)
#
# HEITOR MACIEL - 10402559
# VITOR PEPE - 10339754
# VINICIUS MAGNO - 10401365
# KAIKI BELLINI BARBOSA - 10402509
#
# ARQUIVO: UTILS/LOGGING_UTILS.PY - CONFIGURACAO CENTRAL DE LOGS
# DESCRICAO: DEFINE FORMATO PADRAO E FUNCOES PARA REGISTRAR EVENTOS
# ================================================================
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


# ===========================================================
# FUNCAO CONFIGURAR_LOGGING: MONTA LOGGER COM ARQUIVO E CONSOLE
# ===========================================================
def configurar_logging(base_dir: Path, nivel: str = "INFO") -> logging.Logger:
    base_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("noiseprint_app")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, nivel.upper(), logging.INFO))
    formatter = logging.Formatter(LOG_FORMAT)

    arquivo = logging.FileHandler(base_dir / "forensic.log", encoding="utf-8")
    arquivo.setFormatter(formatter)
    logger.addHandler(arquivo)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


# =======================================================
# FUNCAO REGISTRAR: ENCAMINHA MENSAGEM PARA O LOGGER
# =======================================================
def registrar(logger: logging.Logger, mensagem: str, nivel: str = "info") -> None:
    metodo = getattr(logger, nivel.lower(), logger.info)
    metodo(mensagem)
