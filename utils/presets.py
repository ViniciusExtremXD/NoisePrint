# ================================================================
# PROJETO: NOISEPRINT - PROTOTIPO (VIDEO + TUTORIAL)
#
# HEITOR MACIEL - 10402559
# VITOR PEPE - 10339754
# VINICIUS MAGNO - 10401365
# KAIKI BELLINI BARBOSA - 10402509
#
# ARQUIVO: UTILS/PRESETS.PY - GERENCIAMENTO DE PRESETS DE PESOS
# DESCRICAO: DEFINE OBJETOS DE PRESET E PREPARA ARQUIVOS .PTH EXEMPLARES
# ================================================================
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


# ============================================================
# CLASSE PRESETINFO: AGRUPA IDENTIDADE, NOME, ARQUIVO E NIVEL
# ============================================================
@dataclass(frozen=True)
class PresetInfo:
    """Resumo de um preset cadastrado."""

    identificador: int
    nome: str
    arquivo: Path
    nivel: int


# ===============================================================
# CONSTANTE PRESETS_PADRAO: MAPEIA PRESETS PRINCIPAIS DO PROJETO
# ===============================================================
PRESETS_PADRAO: List[PresetInfo] = [
    PresetInfo(1, "Basico", Path("weights/noiseprint_base.pth"), 1),
    PresetInfo(2, "Detalhado", Path("weights/noiseprint_v1.pth"), 2),
    PresetInfo(3, "Alta sensibilidade", Path("weights/noiseprint_v2.pth"), 3),
    PresetInfo(4, "Mobile", Path("weights/noiseprint_mobile.pth"), 4),
    PresetInfo(5, "Laboratorio", Path("weights/noiseprint_lab.pth"), 5),
    PresetInfo(6, "Demo", Path("weights/noiseprint_demo.pth"), 6),
]


# ======================================================================
# FUNCAO PREPARAR_PASTAS: CRIA PASTAS PADRAO E GERA PRESETS EXEMPLARES
# ======================================================================
def preparar_pastas() -> None:
    """Cria estrutura de pastas e gera presets base com conteudo numerico."""

    Path("weights").mkdir(parents=True, exist_ok=True)

    guia = Path("weights/LEIA.txt")
    if not guia.exists():
        guia.write_text(
            "Coloque arquivos .pth reais nesta pasta quando desejar substituir os presets exemplo.\n",
            encoding="ascii",
        )

    try:
        import torch  # type: ignore[import-not-found]
    except (ImportError, OSError):
        torch = None  # type: ignore[assignment]

    for info in PRESETS_PADRAO:
        if info.arquivo.exists():
            continue
        if torch is not None:
            estado = {
                "meta": {
                    "preset_id": info.identificador,
                    "preset_nome": info.nome,
                    "preset_nivel": info.nivel,
                    "descricao": "Preset de exemplo (sem pesos oficiais).",
                },
                "escala": torch.full((1,), float(info.nivel)),
            }
            torch.save(estado, info.arquivo)
        else:
            info.arquivo.write_text(
                (
                    "Preset de exemplo sem dependencias.\n"
                    f"ID: {info.identificador}\n"
                    f"Nome: {info.nome}\n"
                    f"Nivel: {info.nivel}\n"
                    "Substitua por pesos reais (.pth) quando disponivel.\n"
                ),
                encoding="ascii",
            )


# ======================================================================
# FUNCAO LISTAR_PRESETS: AGREGA PRESETS PADRAO E EXTRAS DO DIRETORIO
# ======================================================================
def listar_presets() -> List[PresetInfo]:
    """Retorna lista de presets disponiveis, incluindo extras no diretorio weights."""

    preparar_pastas()
    encontrados: List[PresetInfo] = []
    vistos: set[Path] = set()

    for info in PRESETS_PADRAO:
        if info.arquivo.exists():
            encontrados.append(info)
            vistos.add(info.arquivo.resolve())

    identificador_extra = 100
    for arquivo in Path("weights").glob("*.pth"):
        caminho = arquivo.resolve()
        if caminho in vistos:
            continue
        identificador_extra += 1
        encontrados.append(
            PresetInfo(
                identificador_extra,
                f"Extra {identificador_extra}",
                arquivo,
                nivel=identificador_extra,
            )
        )
        vistos.add(caminho)

    return encontrados
