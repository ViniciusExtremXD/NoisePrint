# ================================================================
# PROJETO: NOISEPRINT - PROTOTIPO (VIDEO + TUTORIAL)
#
# HEITOR MACIEL - 10402559
# VITOR PEPE - 10339754
# VINICIUS MAGNO - 10401365
# KAIKI BELLINI BARBOSA - 10402509
#
# ARQUIVO: UTILS/REPORT.PY - GERA PDF COM CONTEXTO E IMAGENS
# DESCRICAO: USA FPDF PARA COMPILAR INFORMACOES E MAPAS PRODUZIDOS
# ================================================================
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from fpdf import FPDF


# ============================================================
# FUNCAO GERAR_RELATORIO: MONTA PDF COM CONTEXTO E FIGURAS
# ============================================================
def gerar_relatorio(destino: Path, contexto: Dict[str, str], imagens: List[Tuple[str, Path, str]]) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Forensic Image Analysis Report", ln=True)

    pdf.set_font("Arial", size=12)
    for chave, valor in contexto.items():
        pdf.cell(0, 8, f"{chave}: {valor}", ln=True)

    for titulo, caminho, legenda in imagens:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, titulo, ln=True)
        if legenda:
            pdf.ln(4)
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 8, legenda)
            pdf.ln(2)
        pdf.image(str(caminho), w=180)

    pdf.output(str(destino))
    return destino
