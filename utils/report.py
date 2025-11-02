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

import re
import unicodedata

from fpdf import FPDF


def _normalizar_texto(texto: str) -> str:
    """Normaliza espacos e caracteres especiais antes de enviar ao PDF."""
    if not texto:
        return ""
    texto = unicodedata.normalize("NFKC", texto)
    for alvo, substituto in (
        ("\u00a0", " "),
        ("\u202f", " "),
        ("\u2007", " "),
        ("\u2060", ""),
        ("\ufeff", ""),
    ):
        texto = texto.replace(alvo, substituto)
    texto = texto.replace("&nbsp;", " ")
    texto = re.sub(r"(?<=\S)&(?=\S)", " ", texto)
    return texto


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
        chave_limpo = _normalizar_texto(chave)
        valor_limpo = _normalizar_texto(valor)
        linha = f"{chave_limpo}: {valor_limpo}".rstrip()
        if "\n" in linha:
            pdf.multi_cell(0, 8, linha, align="L")
        else:
            pdf.cell(0, 8, linha, ln=True)

    for titulo, caminho, legenda in imagens:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, _normalizar_texto(titulo), ln=True)
        if legenda:
            pdf.ln(4)
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 8, _normalizar_texto(legenda), align="L")
            pdf.ln(2)
        pdf.image(str(caminho), w=180)

    pdf.output(str(destino))
    return destino
