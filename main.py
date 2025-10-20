from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import ExifTags, Image, __version__ as PIL_VERSION
import skimage  # type: ignore
import torch
import torchvision  # type: ignore
import matplotlib
from matplotlib import cm

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ===== Constantes principais =====
RUN_PREFIX = "run_"
PRESETS_PADRAO = [Path("weights/noiseprint.pth"), Path("weights/noiseprint_demo.pth")]
SEMENTE_PADRAO = 123
NOME_LOGGER = "noiseprint"


# ===== Preparacao das pastas padrao =====
def preparar_pastas_padrao() -> None:
    """Cria as pastas principais do prototipo caso nao existam."""
    for caminho in (Path("data/input"), Path("data/output"), Path("weights")):
        caminho.mkdir(parents=True, exist_ok=True)
    placeholder = Path("weights/LEIA.txt")
    if not placeholder.exists():
        placeholder.write_text(
            "Coloque arquivos .pth nesta pasta para usar a opcao de preset.\n",
            encoding="utf-8",
        )


def listar_presets_disponiveis() -> List[Path]:
    """Retorna a lista de pesos conhecidos e arquivos .pth encontrados em weights/."""
    preparar_pastas_padrao()
    candidatos: List[Path] = []
    candidatos.extend(PRESETS_PADRAO)
    candidatos.extend(Path("weights").glob("*.pth"))
    unicos: List[Path] = []
    vistos: set[str] = set()
    for item in candidatos:
        chave = str(item.resolve())
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(item)
    return unicos
# ===== Estruturas de dados =====
@dataclass
class DadosConfiguracao:
    entradas: List[Path]
    modo_pesos: str
    caminho_pesos: Optional[Path]
    resize_max: Optional[int]
    salvar_heatmap: bool
    salvar_overlay: bool
    salvar_intermediarios: bool
    diretorio_saida: Path
    nivel_log: str
    evento_cancelar: Optional[threading.Event] = None


@dataclass
class ResultadoProcessamento:
    sucesso: bool
    pasta_execucao: Path
    ultimo_arquivo: Optional[Path]


@dataclass
class MetricasImagem:
    arquivo: str
    tempo_segundos: float
    usou_modelo: bool
    forma_entrada: Tuple[int, int]
    forma_processada: Tuple[int, int]
    estatisticas: Dict[str, float]


class ManipuladorTextoTk(logging.Handler):
    """Encaminha mensagens de log para o widget de texto da interface."""

    def __init__(self, widget: tk.Text) -> None:
        super().__init__()
        self.widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        mensagem = self.format(record)

        def adicionar() -> None:
            try:
                self.widget.configure(state="normal")
                self.widget.insert("end", mensagem + "\n")
                self.widget.see("end")
                self.widget.configure(state="disabled")
            except Exception:
                pass

        try:
            self.widget.after(0, adicionar)
        except Exception:
            pass
# ===== Funcoes utilitarias de ambiente e arquivos =====
def configurar_determinismo(semente: int) -> None:
    """Configura geradores para resultados reproduziveis."""
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def calcular_hash_arquivo(arquivo: Optional[Path]) -> Optional[str]:
    """Calcula o hash SHA256 de um arquivo de pesos (quando fornecido)."""
    if arquivo is None or not arquivo.exists():
        return None
    resumo = hashlib.sha256()
    with arquivo.open("rb") as ponteiro:
        for bloco in iter(lambda: ponteiro.read(8192), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def coletar_informacoes_ambiente(pesos: Optional[Path], resize_max: Optional[int], total_imagens: int) -> Dict[str, object]:
    """Reune informacoes de ambiente e configuracao da rodada."""
    dados: Dict[str, object] = {
        "data_hora": datetime.now().isoformat(timespec="seconds"),
        "sistema": platform.platform(),
        "python": platform.python_version(),
        "bibliotecas": {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pillow": PIL_VERSION,
            "matplotlib": matplotlib.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "skimage": skimage.__version__,
        },
        "cuda_disponivel": bool(torch.cuda.is_available()),
        "modo_pesos": "residual" if pesos is None else "modelo",
        "arquivo_pesos": str(pesos) if pesos else None,
        "hash_sha256_pesos": calcular_hash_arquivo(pesos),
        "resize_max": resize_max,
        "total_imagens": total_imagens,
    }
    if torch.cuda.is_available():
        dispositivo = torch.cuda.get_device_properties(0)
        dados["cuda_dispositivo"] = dispositivo.name
        dados["cuda_capacidade"] = f"{dispositivo.major}.{dispositivo.minor}"
    else:
        dados["cuda_dispositivo"] = "cpu"
    return dados


def gerar_linhas_yaml(valor: object, nivel: int = 0) -> List[str]:
    """Converte um dicionario em linhas estilo YAML (sem dependencia externa)."""
    prefixo = "  " * nivel
    if isinstance(valor, dict):
        linhas: List[str] = []
        for chave, conteudo in valor.items():
            if isinstance(conteudo, (dict, list)):
                linhas.append(f"{prefixo}{chave}:")
                linhas.extend(gerar_linhas_yaml(conteudo, nivel + 1))
            else:
                linhas.append(f"{prefixo}{chave}: {conteudo if conteudo is not None else 'null'}")
        return linhas
    if isinstance(valor, list):
        linhas = []
        for item in valor:
            if isinstance(item, (dict, list)):
                linhas.append(f"{prefixo}-")
                linhas.extend(gerar_linhas_yaml(item, nivel + 1))
            else:
                linhas.append(f"{prefixo}- {item}")
        return linhas
    return [f"{prefixo}{valor if valor is not None else 'null'}"]


def salvar_yaml(caminho: Path, dados: Dict[str, object]) -> None:
    """Grava o arquivo execucao.yaml com a configuracao completa."""
    with caminho.open("w", encoding="utf-8") as ponteiro:
        for linha in gerar_linhas_yaml(dados):
            ponteiro.write(linha + "\n")


def salvar_relatorio_markdown(caminho: Path, contexto: Dict[str, object], metricas: List[MetricasImagem]) -> None:
    """Gera relatorio_execucao.md resumindo ambiente e metricas."""
    with caminho.open("w", encoding="utf-8") as ponteiro:
        ponteiro.write("# Relatorio de execucao\n\n")
        ponteiro.write("## Ambiente\n")
        for chave, valor in contexto.items():
            if isinstance(valor, dict):
                ponteiro.write(f"- {chave}:\n")
                for sub_chave, sub_valor in valor.items():
                    ponteiro.write(f"  - {sub_chave}: {sub_valor}\n")
            else:
                ponteiro.write(f"- {chave}: {valor}\n")
        ponteiro.write("\n## Imagens processadas\n")
        for item in metricas:
            texto_estatisticas = ", ".join(f"{nome}={valor:.4f}" for nome, valor in item.estatisticas.items())
            ponteiro.write(
                f"- {item.arquivo} | tempo={item.tempo_segundos:.3f}s | modelo={'sim' if item.usou_modelo else 'nao'} | "
                f"entrada={item.forma_entrada} | processada={item.forma_processada} | {texto_estatisticas}\n"
            )


def ler_resumo_exif(caminho: Path) -> Dict[str, object]:
    """Extrai campos EXIF basicos quando disponiveis."""
    resumo: Dict[str, object] = {}
    try:
        with Image.open(str(caminho)) as imagem:
            exif = imagem.getexif()
            if not exif:
                return resumo
            tabela = {ExifTags.TAGS.get(chave, chave): valor for chave, valor in exif.items()}
            for campo in ("Model", "Make", "DateTime", "Software"):
                if campo in tabela:
                    resumo[campo] = tabela[campo]
    except Exception:
        return {}
    return resumo
# ===== Funcoes centrais de imagem =====
def carregar_bgr(caminho: Path) -> np.ndarray:
    """Carrega imagem no formato BGR."""
    imagem = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if imagem is None:
        raise IOError(f"Falha ao carregar imagem: {caminho}")
    return imagem


def converter_para_cinza(img: np.ndarray) -> np.ndarray:
    """Converte imagem BGR em escala de cinza float32."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def residual_passa_alta(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Aplica filtro passa-alta simples (gaussiana + subtracao)."""
    cinza = converter_para_cinza(img)
    borrada = cv2.GaussianBlur(cinza, (ksize, ksize), 0)
    return (cinza - borrada).astype(np.float32)


def normalizar_mapa(mapa: np.ndarray) -> np.ndarray:
    """Normaliza mapa para o intervalo [0, 1]."""
    arr = mapa.astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if np.isclose(max_val - min_val, 0.0):
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_val) / (max_val - min_val)


def redimensionar_se_necessario(img: np.ndarray, maximo: Optional[int]) -> Tuple[np.ndarray, float]:
    """Redimensiona imagem preservando proporcao quando excede maximo."""
    if not maximo or maximo <= 0:
        return img, 1.0
    altura, largura = img.shape[:2]
    maior = max(altura, largura)
    if maior <= maximo:
        return img, 1.0
    escala = maximo / float(maior)
    novo_tamanho = (int(largura * escala), int(altura * escala))
    redimensionada = cv2.resize(img, novo_tamanho, interpolation=cv2.INTER_AREA)
    return redimensionada, escala


def mapa_para_rgb(mapa: np.ndarray, nome_cmap: str = "jet") -> np.ndarray:
    """Converte mapa normalizado em RGB usando cmap do matplotlib."""
    cmap = cm.get_cmap(nome_cmap)
    rgb = cmap(mapa)[..., :3]
    return (rgb * 255).astype(np.uint8)


def salvar_heatmap(mapa: np.ndarray, caminho: Path) -> None:
    """Salva heatmap colorido em disco."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    rgb = mapa_para_rgb(mapa)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(caminho), bgr):
        raise IOError(f"Falha ao salvar heatmap em {caminho}")


def salvar_overlay(base: np.ndarray, mapa: np.ndarray, caminho: Path, alpha: float = 0.6) -> None:
    """Salva overlay do mapa sobre a imagem original."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    rgb = mapa_para_rgb(mapa)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if bgr.shape[:2] != base.shape[:2]:
        bgr = cv2.resize(bgr, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    mistura = cv2.addWeighted(base.astype(np.float32), 1.0 - alpha, bgr.astype(np.float32), alpha, 0.0)
    resultado = np.clip(mistura, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(caminho), resultado):
        raise IOError(f"Falha ao salvar overlay em {caminho}")


def salvar_intermediarios(pasta: Path, base: str, cinza: Optional[np.ndarray], residual: Optional[np.ndarray]) -> None:
    """Salva intermediarios (cinza e residual) quando solicitado."""
    pasta.mkdir(parents=True, exist_ok=True)
    if cinza is not None:
        cv2.imwrite(str(pasta / f"{base}_cinza.png"), np.clip(cinza, 0, 255).astype(np.uint8))
    if residual is not None:
        mapa = normalizar_mapa(residual)
        salvar_heatmap(mapa, pasta / f"{base}_residual.png")
# ===== Modelo placeholder =====
class RedeNoiseprint(torch.nn.Module):
    def __init__(self) -> None:
        """Constroi rede convolucional rasa usada como placeholder."""
        super().__init__()
        self.bloco = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
        )
        self.cabeca = torch.nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Realiza passagem direta pela rede."""
        tensor = self.bloco(tensor)
        return self.cabeca(tensor)


def carregar_modelo(caminho: Optional[Path], logger: logging.Logger) -> Optional[torch.nn.Module]:
    """Carrega pesos opcionais do Noiseprint caso disponiveis."""
    if caminho is None:
        logger.info("Executando em modo residual (sem pesos).")
        return None
    if not caminho.exists():
        logger.warning("Arquivo de pesos nao encontrado: %s. Usando residual.", caminho)
        return None
    modelo = RedeNoiseprint()
    try:
        checkpoint = torch.load(str(caminho), map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        modelo.load_state_dict(checkpoint)
        modelo.eval()
        logger.info("Pesos carregados de %s.", caminho)
        return modelo
    except Exception as erro:  # noqa: BLE001
        logger.error("Falha ao carregar pesos: %s", erro)
        return None


def executar_modelo(modelo: torch.nn.Module, imagem: np.ndarray) -> np.ndarray:
    """Executa o modelo placeholder sobre a imagem normalizada."""
    tensor = torch.from_numpy(imagem.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    dispositivo = next(modelo.parameters()).device  # type: ignore[arg-type]
    tensor = tensor.to(dispositivo)
    with torch.no_grad():
        saida = modelo(tensor)
    if saida.ndim == 4:
        saida = saida.squeeze(0)
    if saida.ndim == 3 and saida.shape[0] == 1:
        saida = saida.squeeze(0)
    return saida.detach().cpu().numpy().astype(np.float32)


def calcular_noiseprint(imagem: np.ndarray, modelo: Optional[torch.nn.Module]) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
    """Retorna mapa normalizado, residual cru e flag indicando uso do modelo."""
    if modelo is None:
        residual = residual_passa_alta(imagem)
        return normalizar_mapa(residual), residual, False
    residual = executar_modelo(modelo, imagem)
    return normalizar_mapa(residual), residual, True
# ===== Coleta de entradas =====
def obter_entradas(padrao: str) -> List[Path]:
    """Expande caminho ou glob em lista de arquivos unicos."""
    tokens = [item.strip() for item in padrao.replace(",", ";").split(";")]
    resultado: List[Path] = []
    vistos = set()
    for token in tokens:
        if not token:
            continue
        candidato = Path(token)
        linhas: List[Path]
        if candidato.exists():
            if candidato.is_dir():
                linhas = sorted(candidato.glob("*"))
            else:
                linhas = [candidato]
        else:
            linhas = sorted(Path(".").glob(token))
        for caminho in linhas:
            if caminho.is_file() and caminho not in vistos:
                vistos.add(caminho)
                resultado.append(caminho)
    return resultado
# ===== Configuracao de logs e processamento =====
def configurar_logger(pasta: Path, nivel: str) -> logging.Logger:
    """Configura logger com saida em texto e JSONL para reproducibilidade."""
    logger = logging.getLogger(NOME_LOGGER)
    logger.handlers.clear()
    logger.setLevel(getattr(logging, nivel.upper(), logging.INFO))
    logger.propagate = False

    formato = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    arquivo_txt = logging.FileHandler(pasta / "logs.txt", encoding="utf-8")
    arquivo_txt.setFormatter(formato)
    logger.addHandler(arquivo_txt)

    console = logging.StreamHandler()
    console.setFormatter(formato)
    logger.addHandler(console)

    class ManipuladorJson(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            carga = {
                "timestamp": time.time(),
                "nivel": record.levelname,
                "mensagem": record.getMessage(),
                "logger": record.name,
            }
            with (pasta / "logs.jsonl").open("a", encoding="utf-8") as ponteiro:
                ponteiro.write(json.dumps(carga, ensure_ascii=False) + "\n")

    logger.addHandler(ManipuladorJson())
    return logger


def processar_imagens(
    configuracao: DadosConfiguracao,
    manipulador_tk: Optional[ManipuladorTextoTk] = None,
    progresso_callback: Optional[Callable[[int, int], None]] = None,
) -> ResultadoProcessamento:
    """Executa a pipeline sobre todas as imagens e gera artefatos academicos."""
    total = len(configuracao.entradas)
    pasta_execucao = configuracao.diretorio_saida / f"{RUN_PREFIX}{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    pasta_execucao.mkdir(parents=True, exist_ok=True)

    logger = configurar_logger(pasta_execucao, configuracao.nivel_log)
    if manipulador_tk is not None:
        manipulador_tk.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(manipulador_tk)

    configurar_determinismo(SEMENTE_PADRAO)
    modelo = carregar_modelo(configuracao.caminho_pesos, logger)

    contexto = coletar_informacoes_ambiente(configuracao.caminho_pesos, configuracao.resize_max, total)
    salvar_yaml(pasta_execucao / "execucao.yaml", contexto)

    csv_path = pasta_execucao / "manifesto.csv"
    campos = [
        "arquivo",
        "largura",
        "altura",
        "resize_aplicado",
        "modo_pesos",
        "tempo_total_ms",
        "tempo_io_ms",
        "tempo_preproc_ms",
        "tempo_modelo_ms",
        "tempo_normalizacao_ms",
        "tempo_salvar_ms",
        "min",
        "max",
        "mean",
        "std",
        "p95",
        "entropia",
    ]
    arquivo_csv = csv_path.open("w", newline="", encoding="utf-8")
    escritor = csv.DictWriter(arquivo_csv, fieldnames=campos)
    escritor.writeheader()

    metricas: List[MetricasImagem] = []
    ultimo_arquivo: Optional[Path] = None
    sucesso = True

    if progresso_callback:
        progresso_callback(0, total)

    for indice, caminho in enumerate(configuracao.entradas, start=1):
        if configuracao.evento_cancelar and configuracao.evento_cancelar.is_set():
            logger.warning("Processamento cancelado pelo usuario.")
            sucesso = False
            break
        inicio_total = time.perf_counter()
        try:
            t_io = time.perf_counter()
            imagem_original = carregar_bgr(caminho)
            tempo_io = time.perf_counter() - t_io

            altura, largura = imagem_original.shape[:2]
            t_pre = time.perf_counter()
            imagem_trabalho, escala = redimensionar_se_necessario(imagem_original, configuracao.resize_max)
            tempo_pre = time.perf_counter() - t_pre

            mapa_cinza = converter_para_cinza(imagem_trabalho) if configuracao.salvar_intermediarios else None

            t_modelo = time.perf_counter()
            mapa_normalizado, residual, usou_modelo = calcular_noiseprint(imagem_trabalho, modelo)
            tempo_modelo = time.perf_counter() - t_modelo

            arr_stats = np.clip(mapa_normalizado.astype(np.float32), 1e-6, 1.0 - 1e-6)
            estatisticas = {
                "min": float(arr_stats.min()),
                "max": float(arr_stats.max()),
                "mean": float(arr_stats.mean()),
                "std": float(arr_stats.std()),
                "p95": float(np.percentile(arr_stats, 95)),
                "entropia": float((-arr_stats * np.log2(arr_stats) - (1.0 - arr_stats) * np.log2(1.0 - arr_stats)).mean()),
            }

            t_salvar = time.perf_counter()
            base = caminho.stem
            if configuracao.salvar_intermediarios:
                salvar_intermediarios(pasta_execucao, base, mapa_cinza, residual)
            if configuracao.salvar_heatmap:
                caminho_heatmap = pasta_execucao / f"{base}_noiseprint.png"
                salvar_heatmap(mapa_normalizado, caminho_heatmap)
                ultimo_arquivo = caminho_heatmap
            if configuracao.salvar_overlay:
                base_overlay = imagem_trabalho if escala != 1.0 else imagem_original
                caminho_overlay = pasta_execucao / f"{base}_overlay.png"
                salvar_overlay(base_overlay, mapa_normalizado, caminho_overlay)
                ultimo_arquivo = caminho_overlay
            tempo_salvar = time.perf_counter() - t_salvar

            tempo_total = time.perf_counter() - inicio_total

            escritor.writerow(
                {
                    "arquivo": str(caminho),
                    "largura": largura,
                    "altura": altura,
                    "resize_aplicado": int(escala != 1.0),
                    "modo_pesos": "modelo" if usou_modelo else "residual",
                    "tempo_total_ms": round(tempo_total * 1000, 2),
                    "tempo_io_ms": round(tempo_io * 1000, 2),
                    "tempo_preproc_ms": round(tempo_pre * 1000, 2),
                    "tempo_modelo_ms": round(tempo_modelo * 1000, 2),
                    "tempo_normalizacao_ms": 0.0,
                    "tempo_salvar_ms": round(tempo_salvar * 1000, 2),
                    **estatisticas,
                }
            )

            metricas.append(
                MetricasImagem(
                    arquivo=str(caminho),
                    tempo_segundos=tempo_total,
                    usou_modelo=usou_modelo,
                    forma_entrada=(altura, largura),
                    forma_processada=imagem_trabalho.shape[:2],
                    estatisticas=estatisticas,
                )
            )

            logger.info("Processado %s em %.2fs.", caminho.name, tempo_total)
            resumo_exif = ler_resumo_exif(caminho)
            if resumo_exif:
                logger.debug("EXIF %s: %s", caminho.name, resumo_exif)
        except Exception as erro:  # noqa: BLE001
            sucesso = False
            logger.exception("Falha ao processar %s: %s", caminho, erro)

        if progresso_callback:
            progresso_callback(indice, total)

    arquivo_csv.close()
    salvar_relatorio_markdown(pasta_execucao / "relatorio_execucao.md", contexto, metricas)

    if sucesso:
        logger.info("Execucao concluida com sucesso.")
    else:
        logger.warning("Execucao concluida com erros ou cancelamento.")

    return ResultadoProcessamento(sucesso=sucesso, pasta_execucao=pasta_execucao, ultimo_arquivo=ultimo_arquivo)
# ===== Interface grafica =====
class AplicacaoNoiseprint:
    """Interface grafica principal do prototipo."""
    def __init__(self, raiz: tk.Tk) -> None:
        self.raiz = raiz
        self.raiz.title("Noiseprint - Prototipo Academico")
        self.raiz.option_add("*Font", ("Segoe UI", 10))

        self.var_entrada = tk.StringVar(value="data/input/*.jpg")
        self.var_modo_peso = tk.StringVar(value="sem")
        self.var_peso_arquivo = tk.StringVar()
        self.var_preset = tk.StringVar()
        self.var_saida = tk.StringVar(value="data/output")
        self.var_manter_resolucao = tk.BooleanVar(value=True)
        self.var_nivel_log = tk.StringVar(value="INFO")
        self.var_salvar_heatmap = tk.BooleanVar(value=True)
        self.var_salvar_overlay = tk.BooleanVar(value=True)
        self.var_salvar_intermediarios = tk.BooleanVar(value=False)

        self.evento_cancelar = threading.Event()
        self.thread_trabalho: Optional[threading.Thread] = None
        self.ultimo_arquivo: Optional[Path] = None
        self.ultima_pasta: Optional[Path] = None
        self.manipulador_tk: Optional[ManipuladorTextoTk] = None

        self._construir_interface()

    def _construir_interface(self) -> None:
        quadro = ttk.Frame(self.raiz, padding=10)
        quadro.pack(fill="both", expand=True)

        ttk.Label(quadro, text="Entrada (arquivo, glob ou lista ';'):").grid(row=0, column=0, sticky="w")
        entrada = ttk.Entry(quadro, textvariable=self.var_entrada, width=55)
        entrada.grid(row=1, column=0, sticky="we")
        caixa_botoes = ttk.Frame(quadro)
        caixa_botoes.grid(row=1, column=1, padx=(5, 0))
        ttk.Button(caixa_botoes, text="Arquivos...", command=self._selecionar_arquivos).pack(side="left", padx=2)
        ttk.Button(caixa_botoes, text="Pasta...", command=self._selecionar_pasta).pack(side="left", padx=2)

        quadro_pesos = ttk.LabelFrame(quadro, text="Pesos (opcional)")
        quadro_pesos.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky="we")
        ttk.Radiobutton(quadro_pesos, text="Sem pesos (residual)", variable=self.var_modo_peso, value="sem", command=self._atualizar_controles_pesos).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(quadro_pesos, text="Arquivo (.pth)", variable=self.var_modo_peso, value="arquivo", command=self._atualizar_controles_pesos).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(quadro_pesos, text="Preset manual", variable=self.var_modo_peso, value="preset", command=self._atualizar_controles_pesos).grid(row=2, column=0, sticky="w")

        self.campo_peso = ttk.Entry(quadro_pesos, textvariable=self.var_peso_arquivo, width=45)
        self.campo_peso.grid(row=1, column=1, sticky="we", padx=5)
        self.botao_peso = ttk.Button(quadro_pesos, text="Selecionar...", command=self._selecionar_arquivo_pesos)
        self.botao_peso.grid(row=1, column=2, padx=5)

        ttk.Label(quadro_pesos, text="Caminho manual:").grid(row=2, column=1, sticky="w", padx=5)
        self.campo_preset = ttk.Entry(quadro_pesos, textvariable=self.var_preset, width=40)
        self.campo_preset.grid(row=2, column=1, sticky="e", padx=(110, 5))
        if not self.var_preset.get():
            self.var_preset.set(str(PRESETS_PADRAO[0]))
        self.botao_placeholder = ttk.Button(quadro_pesos, text="Criar placeholder", command=self._criar_placeholder_peso)
        self.botao_placeholder.grid(row=2, column=2, padx=5)

        ttk.Label(quadro, text="Diretorio de saida:").grid(row=3, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(quadro, textvariable=self.var_saida, width=55).grid(row=4, column=0, sticky="we")
        ttk.Button(quadro, text="Selecionar...", command=self._selecionar_saida).grid(row=4, column=1, padx=(5, 0))

        quadro_opcoes = ttk.Frame(quadro)
        quadro_opcoes.grid(row=5, column=0, columnspan=2, pady=(10, 0), sticky="we")
        ttk.Checkbutton(quadro_opcoes, text="Salvar heatmap", variable=self.var_salvar_heatmap).pack(side="left", padx=5)
        ttk.Checkbutton(quadro_opcoes, text="Salvar overlay", variable=self.var_salvar_overlay).pack(side="left", padx=5)
        ttk.Checkbutton(quadro_opcoes, text="Salvar intermediarios", variable=self.var_salvar_intermediarios).pack(side="left", padx=5)
        ttk.Checkbutton(quadro_opcoes, text="Manter resolucao original", variable=self.var_manter_resolucao, command=self._atualizar_resize).pack(side="left", padx=5)
        ttk.Label(quadro_opcoes, text="Resize max (px):").pack(side="left", padx=5)
        self.campo_resize = ttk.Entry(quadro_opcoes, width=8)
        self.campo_resize.insert(0, "2048")
        self.campo_resize.pack(side="left")
        ttk.Label(quadro_opcoes, text="Nivel de log:").pack(side="left", padx=5)
        ttk.Combobox(quadro_opcoes, values=["DEBUG", "INFO", "WARNING", "ERROR"], textvariable=self.var_nivel_log, width=8, state="readonly").pack(side="left")

        quadro_botoes = ttk.Frame(quadro)
        quadro_botoes.grid(row=6, column=0, columnspan=2, pady=(15, 5))
        self.botao_executar = ttk.Button(quadro_botoes, text="Executar", command=self._iniciar_processamento)
        self.botao_executar.pack(side="left", padx=5)
        self.botao_cancelar = ttk.Button(quadro_botoes, text="Cancelar", command=self._cancelar_processamento, state="disabled")
        self.botao_cancelar.pack(side="left", padx=5)
        self.botao_abre_saida = ttk.Button(quadro_botoes, text="Abrir pasta de saida", command=self._abrir_saida, state="disabled")
        self.botao_abre_saida.pack(side="left", padx=5)
        self.botao_abre_resultado = ttk.Button(quadro_botoes, text="Abrir ultimo resultado", command=self._abrir_ultimo, state="disabled")
        self.botao_abre_resultado.pack(side="left", padx=5)

        self.barra_progresso = ttk.Progressbar(quadro, mode="determinate", length=360)
        self.barra_progresso.grid(row=7, column=0, columnspan=2, sticky="we")

        ttk.Label(quadro, text="Logs:").grid(row=8, column=0, sticky="w")
        self.caixa_logs = tk.Text(quadro, height=14, width=90, state="disabled")
        self.caixa_logs.grid(row=9, column=0, columnspan=2, sticky="nsew")

        quadro.rowconfigure(9, weight=1)
        quadro.columnconfigure(0, weight=1)

    def _selecionar_arquivos(self) -> None:
        arquivos = filedialog.askopenfilenames(
            title="Selecione imagens",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Todos", "*.*")],
        )
        if arquivos:
            self.var_entrada.set(";".join(arquivos))

    def _selecionar_pasta(self) -> None:
        pasta = filedialog.askdirectory(title="Selecione a pasta de entrada")
        if pasta:
            self.var_entrada.set(str(Path(pasta) / "*.*"))

    def _selecionar_arquivo_pesos(self) -> None:
        arquivo = filedialog.askopenfilename(
            title="Selecione arquivo .pth",
            filetypes=[("Pesos PyTorch", "*.pth *.pt"), ("Todos", "*.*")],
        )
        if arquivo:
            self.var_peso_arquivo.set(arquivo)

    def _selecionar_saida(self) -> None:
        pasta = filedialog.askdirectory(title="Selecione a pasta de saida")
        if pasta:
            self.var_saida.set(pasta)

    def _criar_placeholder_peso(self) -> None:
        """Cria arquivo placeholder de pesos para demonstracao."""
        caminho = self.var_preset.get().strip()
        if not caminho:
            caminho = str(PRESETS_PADRAO[0])
            self.var_preset.set(caminho)
        alvo = Path(caminho)
        alvo.parent.mkdir(parents=True, exist_ok=True)
        try:
            alvo.write_text("Placeholder demo de pesos Noiseprint.\nSubstitua por um arquivo .pth real quando necessario.\n", encoding="utf-8")
            messagebox.showinfo("Placeholder criado", f"Arquivo demo criado em: {alvo}")
        except Exception as erro:  # noqa: BLE001
            messagebox.showerror("Erro", f"Falha ao criar placeholder: {erro}")

    def _atualizar_controles_pesos(self) -> None:
        modo = self.var_modo_peso.get()
        estado_arquivo = "normal" if modo == "arquivo" else "disabled"
        estado_preset = "readonly" if modo == "preset" else "disabled"
        self.campo_peso.configure(state=estado_arquivo)
        self.botao_peso.configure(state=estado_arquivo)
        self.campo_preset.configure(state="normal" if modo == "preset" else "disabled")
        self.botao_placeholder.configure(state="normal" if modo == "preset" else "disabled")

    def _atualizar_resize(self) -> None:
        if self.var_manter_resolucao.get():
            self.campo_resize.configure(state="disabled")
        else:
            self.campo_resize.configure(state="normal")

    def _obter_caminho_pesos(self) -> Optional[Path]:
        modo = self.var_modo_peso.get()
        if modo == "sem":
            return None
        if modo == "arquivo":
            valor = self.var_peso_arquivo.get().strip()
            return Path(valor) if valor else None
        valor = self.var_preset.get().strip()
        return Path(valor) if valor else None

    def _obter_resize_maximo(self) -> Optional[int]:
        if self.var_manter_resolucao.get():
            return None
        valor = self.campo_resize.get().strip()
        if not valor.isdigit():
            return None
        numero = int(valor)
        return numero if numero > 0 else None

    def _iniciar_processamento(self) -> None:
        if self.thread_trabalho and self.thread_trabalho.is_alive():
            messagebox.showinfo("Processando", "Uma execucao esta em andamento.")
            return

        entradas = obter_entradas(self.var_entrada.get())
        if not entradas:
            messagebox.showerror("Erro", "Nenhuma imagem encontrada.")
            return

        caminho_pesos = self._obter_caminho_pesos()
        modo = self.var_modo_peso.get()
        if modo in {"arquivo", "preset"} and (caminho_pesos is None or not caminho_pesos.exists()):
            messagebox.showerror("Erro", "O arquivo de pesos informado nao foi encontrado.")
            return

        resize_max = self._obter_resize_maximo()
        pasta_saida = Path(self.var_saida.get().strip() or "data/output")

        self._limpar_logs()
        self._adicionar_log("Iniciando processamento...")
        self.barra_progresso.configure(value=0, maximum=max(1, len(entradas)))
        self.botao_executar.configure(state="disabled")
        self.botao_cancelar.configure(state="normal")
        self.botao_abre_saida.configure(state="disabled")
        self.botao_abre_resultado.configure(state="disabled")
        self.evento_cancelar.clear()
        self.ultimo_arquivo = None
        self.ultima_pasta = None

        self.manipulador_tk = ManipuladorTextoTk(self.caixa_logs)
        configuracao = DadosConfiguracao(
            entradas=entradas,
            modo_pesos=modo,
            caminho_pesos=caminho_pesos,
            resize_max=resize_max,
            salvar_heatmap=self.var_salvar_heatmap.get(),
            salvar_overlay=self.var_salvar_overlay.get(),
            salvar_intermediarios=self.var_salvar_intermediarios.get(),
            diretorio_saida=pasta_saida,
            nivel_log=self.var_nivel_log.get(),
            evento_cancelar=self.evento_cancelar,
        )

        def alvo() -> None:
            resultado = ResultadoProcessamento(False, pasta_saida, None)
            try:
                resultado = processar_imagens(
                    configuracao,
                    manipulador_tk=self.manipulador_tk,
                    progresso_callback=self._atualizar_progresso,
                )
            finally:
                def finalizar() -> None:
                    logger = logging.getLogger(NOME_LOGGER)
                    if self.manipulador_tk and self.manipulador_tk in logger.handlers:
                        logger.removeHandler(self.manipulador_tk)
                    self.manipulador_tk = None
                    self.ultimo_arquivo = resultado.ultimo_arquivo
                    self.ultima_pasta = resultado.pasta_execucao
                    if resultado.sucesso:
                        self._adicionar_log("Execucao concluida com sucesso.")
                    else:
                        self._adicionar_log("Execucao finalizada com erros ou cancelamento.")
                    self._finalizar_processamento()
                self.raiz.after(0, finalizar)

        self.thread_trabalho = threading.Thread(target=alvo, daemon=True)
        self.thread_trabalho.start()

    def _finalizar_processamento(self) -> None:
        self.botao_executar.configure(state="normal")
        self.botao_cancelar.configure(state="disabled")
        self.botao_abre_saida.configure(state="normal" if self.ultima_pasta else "disabled")
        habilitar_resultado = self.ultimo_arquivo and Path(self.ultimo_arquivo).exists()
        self.botao_abre_resultado.configure(state="normal" if habilitar_resultado else "disabled")

    def _cancelar_processamento(self) -> None:
        if self.thread_trabalho and self.thread_trabalho.is_alive():
            self.evento_cancelar.set()
            self._adicionar_log("Solicitado cancelamento pelo usuario.")

    def _abrir_saida(self) -> None:
        alvo = self.ultima_pasta or Path(self.var_saida.get().strip() or "data/output")
        alvo.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(alvo))
        except Exception as erro:  # noqa: BLE001
            messagebox.showerror("Erro", f"Nao foi possivel abrir: {erro}")

    def _abrir_ultimo(self) -> None:
        if not self.ultimo_arquivo or not Path(self.ultimo_arquivo).exists():
            messagebox.showerror("Erro", "Nenhum resultado disponivel.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(self.ultimo_arquivo))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(self.ultimo_arquivo)], check=False)
            else:
                subprocess.run(["xdg-open", str(self.ultimo_arquivo)], check=False)
        except Exception as erro:  # noqa: BLE001
            messagebox.showerror("Erro", f"Nao foi possivel abrir: {erro}")

    def _atualizar_progresso(self, atual: int, total: int) -> None:
        def atualizar() -> None:
            self.barra_progresso.configure(maximum=max(1, total), value=atual)
        self.raiz.after(0, atualizar)

    def _limpar_logs(self) -> None:
        self.caixa_logs.configure(state="normal")
        self.caixa_logs.delete("1.0", "end")
        self.caixa_logs.configure(state="disabled")

    def _adicionar_log(self, mensagem: str) -> None:
        self.caixa_logs.configure(state="normal")
        self.caixa_logs.insert("end", mensagem + "\n")
        self.caixa_logs.see("end")
        self.caixa_logs.configure(state="disabled")
# ===== CLI =====
def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ferramenta Noiseprint (ASCII)")
    parser.add_argument("--entrada", required=True, help="Arquivos ou glob (ex: data/input/*.jpg)")
    parser.add_argument("--pesos", default=None, help="Caminho para pesos .pth")
    parser.add_argument("--preset", default=None, help="Nome do preset em weights/")
    parser.add_argument("--saida", default="data/output", help="Diretorio base de saida")
    parser.add_argument("--resize-max", type=int, default=None, help="Resize maximo (px)")
    parser.add_argument("--salvar-heatmap", action="store_true", help="Salvar heatmap")
    parser.add_argument("--salvar-overlay", action="store_true", help="Salvar overlay")
    parser.add_argument("--salvar-intermediarios", action="store_true", help="Salvar intermediarios")
    parser.add_argument("--nivel-log", default="INFO", help="Nivel de log")
    return parser


def executar_cli(argv: Sequence[str]) -> int:
    parser = criar_parser()
    args = parser.parse_args(argv)
    preparar_pastas_padrao()
    entradas = obter_entradas(args.entrada)
    if not entradas:
        print("Nenhum arquivo encontrado.", file=sys.stderr)
        return 1
    if args.pesos and args.preset:
        print("Informe apenas --pesos ou --preset.", file=sys.stderr)
        return 1

    caminho_pesos: Optional[Path] = None
    modo = "sem"
    if args.pesos:
        caminho_pesos = Path(args.pesos)
        modo = "arquivo"
    elif args.preset:
        caminho_pesos = Path("weights") / args.preset
        modo = "preset"
        if not caminho_pesos.exists():
            print("Aviso: preset indicado nao foi encontrado. Executando em modo residual.", file=sys.stderr)
            caminho_pesos = None
            modo = "sem"

    configuracao = DadosConfiguracao(
        entradas=entradas,
        modo_pesos=modo,
        caminho_pesos=caminho_pesos,
        resize_max=args.resize_max,
        salvar_heatmap=args.salvar_heatmap,
        salvar_overlay=args.salvar_overlay,
        salvar_intermediarios=args.salvar_intermediarios,
        diretorio_saida=Path(args.saida),
        nivel_log=args.nivel_log,
    )
    resultado = processar_imagens(configuracao)
    print(f"Artefatos salvos em: {resultado.pasta_execucao}")
    return 0 if resultado.sucesso else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        preparar_pastas_padrao()
        raiz = tk.Tk()
        app = AplicacaoNoiseprint(raiz)
        raiz.mainloop()
        return 0
    preparar_pastas_padrao()
    return executar_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())


