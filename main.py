# ================================================================
# PROJETO: NOISEPRINT - PROTOTIPO (VIDEO + TUTORIAL)
#
# HEITOR MACIEL - 10402559
# VITOR PEPE - 10339754
# VINICIUS MAGNO - 10401365
# KAIKI BELLINI BARBOSA - 10402509
#
# ARQUIVO: MAIN.PY - INICIALIZACAO DO APLICATIVO FORENSE EM PYQT5
# DESCRICAO: CONFIGURA LOGGING, PREPARA PASTAS E ABRE A INTERFACE PRINCIPAL
# ================================================================
from __future__ import annotations

import sys
from pathlib import Path

from PyQt5 import QtWidgets

from ui.main_window import MainWindow
from utils.logging_utils import configurar_logging
from utils.presets import preparar_pastas


# ================================================================
# FUNCAO EXECUTAR: CONFIGURA LOGS, PREPARA DIRETORIOS E INICIA A GUI
# ================================================================
def executar() -> int:
    """Inicializa logging, prepara pastas e inicia a interface."""
    raiz_saida = Path("outputs")
    logger = configurar_logging(raiz_saida, nivel="INFO")
    preparar_pastas()

    app = QtWidgets.QApplication(sys.argv)
    janela = MainWindow(logger, raiz_saida)
    janela.show()
    return app.exec_()


# =======================================================
# BLOCO PRINCIPAL: GARANTE SAIDA ELEGANTE DO APLICATIVO
# =======================================================
if __name__ == "__main__":
    raise SystemExit(executar())
