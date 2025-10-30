# ================================================================
# PROJETO: NOISEPRINT - PROTOTIPO (VIDEO + TUTORIAL)
#
# HEITOR MACIEL - 10402559
# VITOR PEPE - 10339754
# VINICIUS MAGNO - 10401365
# KAIKI BELLINI BARBOSA - 10402509
#
# ARQUIVO: UI/MAIN_WINDOW.PY - INTERFACE GRAFICA COMPLETA EM PYQT5
# DESCRICAO: IMPLEMENTA A GUI COM ABA DE IMPORTACAO, ANALISES E RELATORIOS
# ================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from src.noiseprint import extrair_noiseprint
from src.ela import aplicar_ela
from src.prnu import extrair_prnu
from utils.image_io import carregar_imagem, salvar_imagem, extrair_exif
from utils.logging_utils import registrar
from utils.report import gerar_relatorio
from utils.presets import listar_presets, PRESETS_PADRAO, PresetInfo


# ================================================================
# CLASSE ANALISERESULTADO: AGREGA METADADOS DE CADA PROCESSAMENTO
# ================================================================
@dataclass
class AnaliseResultado:
    metodo: str
    mapa_path: Path
    overlay_path: Optional[Path]
    descricao: str
    pesos: Optional[Path]
    pesos_descricao: str


# ================================================================
# CLASSE QTLOGHANDLER: ENVIA ENTRADAS DE LOG PARA O PAINEL DE TEXTO
# ================================================================
class QtLogHandler(logging.Handler):
    """Encaminha registros do logger para o painel de logs da GUI."""

    def __init__(self, callback) -> None:
        super().__init__()
        self.callback = callback

    # ================================================================
    # FUNCAO EMIT: REPASSA MENSAGENS FORMATADAS PARA O CALLBACK DA GUI
    # ================================================================
    def emit(self, record: logging.LogRecord) -> None:
        self.callback(self.format(record))


# ================================================================
# CLASSE IMAGEVIEWER: PROVE ZOOM E PAN PARA VISUALIZAR IMAGENS
# ================================================================
class ImageViewer(QtWidgets.QGraphicsView):
    """Visualizador de imagem com suporte a zoom e pan."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    # ============================================================
    # FUNCAO SET_IMAGE: CARREGA MATRIZ BGR EM UM QPIXMAP AJUSTADO
    # ============================================================
    def set_image(self, imagem: np.ndarray) -> None:
        if imagem.size == 0:
            return
        rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self._pixmap_item.setPixmap(QtGui.QPixmap.fromImage(qimg))
        self.fitInView(self._pixmap_item, QtCore.Qt.KeepAspectRatio)

    # =========================================================
    # FUNCAO WHEELEVENT: APLICA ZOOM GRADUAL NO MOVIMENTO DO MOUSE
    # =========================================================
    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802
        if self._pixmap_item.pixmap().isNull():
            return
        fator = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(fator, fator)


# ================================================================
# CLASSE DROPLABEL: RECUPERA CAMINHOS VIA DRAG-AND-DROP
# ================================================================
class DropLabel(QtWidgets.QLabel):
    """Area para drag-and-drop de imagens."""

    arquivo_recebido = QtCore.pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__("Arraste uma imagem aqui ou clique em `Abrir`.")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("border: 2px dashed #999; padding: 24px; color: #555;")
        self.setAcceptDrops(True)

    # ===========================================================
    # FUNCAO DRAGENTEREVENT: ACEITA ARQUIVOS ARRASTADOS PARA A LABEL
    # ===========================================================
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    # ===========================================================
    # FUNCAO DROPEVENT: EMITE SINAL COM O PRIMEIRO CAMINHO RECEBIDO
    # ===========================================================
    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if urls:
            self.arquivo_recebido.emit(urls[0].toLocalFile())


# ================================================================
# CLASSE MAINWINDOW: COORDENA TODAS AS ABAS E FLUXOS DA APLICACAO
# ================================================================
class MainWindow(QtWidgets.QMainWindow):
    """Janela principal com abas de analise e relatorio."""

    def __init__(self, logger: logging.Logger, trabalho_dir: Path) -> None:
        super().__init__()
        self.logger = logger
        self.trabalho_dir = trabalho_dir
        self.trabalho_dir.mkdir(parents=True, exist_ok=True)

        self.arquivo_atual: Optional[Path] = None
        self.imagem_atual: Optional[np.ndarray] = None
        self.resultados: List[AnaliseResultado] = []
        self.historico: List[Path] = []
        self.caminho_pesos_manual: Optional[Path] = None
        self.campo_pesos: Optional[QtWidgets.QLineEdit] = None
        self.preset_group: Optional[QtWidgets.QButtonGroup] = None
        self._preset_id_para_info: Dict[int, PresetInfo] = {}
        self.preset_infos: List[PresetInfo] = listar_presets()
        self._atualizando_pesos: bool = False

        self.setWindowTitle("Forensic Analyzer - NoisePrint | ELA | PRNU")
        self.resize(1280, 720)

        self._montar_interface()

        handler = QtLogHandler(self._adicionar_log)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(handler)
        registrar(self.logger, "Aplicacao iniciada.")

    # ----- construcao de interface -----
    # ============================================================
    # FUNCAO _MONTAR_INTERFACE: CONSTROI ABAS, LOGS E STATUS BAR
    # ============================================================
    def _montar_interface(self) -> None:
        layout_principal = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(layout_principal)

        self.tabs = QtWidgets.QTabWidget()
        layout_principal.addWidget(self.tabs)

        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        layout_principal.addWidget(self.log_box)
        layout_principal.setSizes([900, 380])

        self.tabs.addTab(self._criar_tab_importacao(), "Import Image")
        self.tabs.addTab(self._criar_tab_metodo("NoisePrint"), "NoisePrint")
        self.tabs.addTab(self._criar_tab_metodo("ELA"), "ELA")
        self.tabs.addTab(self._criar_tab_metodo("PRNU"), "PRNU")
        self.tabs.addTab(self._criar_tab_comparacao(), "Comparison")
        self.tabs.addTab(self._criar_tab_relatorio(), "Report")

        self.statusBar().showMessage("Pronto para iniciar")
        self._resetar(silencioso=True)

    # ===============================================================
    # FUNCAO _CRIAR_TAB_IMPORTACAO: MONTA A ABA DE INPUT DE IMAGENS
    # ===============================================================
    def _criar_tab_importacao(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        self.drop_label = DropLabel()
        self.drop_label.arquivo_recebido.connect(self._carregar_arquivo)
        layout.addWidget(self.drop_label)

        botoes = QtWidgets.QHBoxLayout()
        btn_abrir = QtWidgets.QPushButton("Abrir imagem...")
        btn_abrir.clicked.connect(self._selecionar_arquivo)
        botoes.addWidget(btn_abrir)

        btn_reset = QtWidgets.QPushButton("Reset")
        btn_reset.clicked.connect(self._resetar)
        botoes.addWidget(btn_reset)

        btn_help = QtWidgets.QPushButton("Help")
        btn_help.clicked.connect(self._mostrar_ajuda)
        botoes.addWidget(btn_help)
        botoes.addStretch()
        layout.addLayout(botoes)

        # ===== CONTROLE MANUAL DE PESOS =====
        grupo_pesos = QtWidgets.QGroupBox("Pesos NoisePrint (opcional)")
        layout_pesos = QtWidgets.QVBoxLayout(grupo_pesos)
        layout_pesos.addWidget(QtWidgets.QLabel("Escolha um preset numerico ou informe caminho manual:"))

        grade_preset = QtWidgets.QGridLayout()
        self.preset_group = QtWidgets.QButtonGroup(self)
        self.preset_group.setExclusive(True)
        if not self.preset_infos:
            layout_pesos.addWidget(QtWidgets.QLabel("Nenhum preset encontrado em weights/."))
        else:
            for indice, info in enumerate(self.preset_infos[:5], start=1):
                rotulo = f"{info.identificador} - {info.nome} (nivel {info.nivel})"
                botao = QtWidgets.QRadioButton(rotulo)
                self.preset_group.addButton(botao, info.identificador)
                self._preset_id_para_info[info.identificador] = info
                botao.clicked.connect(lambda _, preset_info=info: self._selecionar_preset(preset_info))
                linha = (indice - 1) // 3
                coluna = (indice - 1) % 3
                grade_preset.addWidget(botao, linha, coluna)
            layout_pesos.addLayout(grade_preset)

        manual_layout = QtWidgets.QHBoxLayout()
        self.campo_pesos = QtWidgets.QLineEdit()
        self.campo_pesos.setPlaceholderText("Digite caminho completo para pesos .pth ou deixe vazio")
        self.campo_pesos.textChanged.connect(self._ao_mudar_campo_pesos)
        manual_layout.addWidget(self.campo_pesos)

        btn_pesos = QtWidgets.QPushButton("Selecionar...")
        btn_pesos.clicked.connect(self._selecionar_pesos)
        manual_layout.addWidget(btn_pesos)

        btn_limpar_pesos = QtWidgets.QPushButton("Usar residual")
        btn_limpar_pesos.clicked.connect(self._limpar_pesos)
        manual_layout.addWidget(btn_limpar_pesos)

        layout_pesos.addLayout(manual_layout)

        layout.addWidget(grupo_pesos)

        self.viewer_original = ImageViewer()
        layout.addWidget(self.viewer_original)

        layout.addWidget(QtWidgets.QLabel("Historico recente:"))
        self.lista_historico = QtWidgets.QListWidget()
        self.lista_historico.itemClicked.connect(lambda item: self._carregar_arquivo(item.text()))
        layout.addWidget(self.lista_historico)

        return tab

    # ============================================================
    # FUNCAO _CRIAR_TAB_METODO: MONTA A ABA DE CADA PROCESSAMENTO
    # ============================================================
    def _criar_tab_metodo(self, metodo: str) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        layout.addWidget(QtWidgets.QLabel("Fluxo: importe -> ajuste parametros -> execute -> exporte."))

        parametros = QtWidgets.QHBoxLayout()
        if metodo == "NoisePrint":
            parametros.addWidget(QtWidgets.QLabel("Kernel passa-alta:"))
            self.spin_noise = QtWidgets.QSpinBox()
            self.spin_noise.setRange(3, 99)
            self.spin_noise.setSingleStep(2)
            self.spin_noise.setValue(7)
            parametros.addWidget(self.spin_noise)
        elif metodo == "ELA":
            parametros.addWidget(QtWidgets.QLabel("Qualidade JPEG:"))
            self.spin_ela = QtWidgets.QSpinBox()
            self.spin_ela.setRange(1, 100)
            self.spin_ela.setValue(90)
            parametros.addWidget(self.spin_ela)
        else:
            parametros.addWidget(QtWidgets.QLabel("Kernel suavizacao:"))
            self.spin_prnu = QtWidgets.QSpinBox()
            self.spin_prnu.setRange(1, 25)
            self.spin_prnu.setValue(5)
            parametros.addWidget(self.spin_prnu)
        parametros.addStretch()
        layout.addLayout(parametros)

        btn_exec = QtWidgets.QPushButton("Executar analise")
        btn_exec.clicked.connect(lambda _, nome=metodo: self._executar_metodo(nome))
        layout.addWidget(btn_exec)

        viewer = ImageViewer()
        layout.addWidget(QtWidgets.QLabel("Resultado:"))
        layout.addWidget(viewer)

        if metodo == "NoisePrint":
            self.viewer_noise = viewer
        elif metodo == "ELA":
            self.viewer_ela = viewer
        else:
            self.viewer_prnu = viewer
        return tab

    # =============================================================
    # FUNCAO _CRIAR_TAB_COMPARACAO: CONSTRUCAO DO PAINEL DE COMPARO
    # =============================================================
    def _criar_tab_comparacao(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)
        self.viewer_comp1 = ImageViewer()
        self.viewer_comp2 = ImageViewer()
        layout.addWidget(self.viewer_comp1)
        layout.addWidget(self.viewer_comp2)
        return tab

    # ===========================================================
    # FUNCAO _CRIAR_TAB_RELATORIO: CRIA CAMPOS PARA GERAR PDFs
    # ===========================================================
    def _criar_tab_relatorio(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        self.texto_relatorio = QtWidgets.QTextEdit()
        self.texto_relatorio.setPlaceholderText("Observacoes para o relatorio PDF...")
        layout.addWidget(self.texto_relatorio)

        botoes = QtWidgets.QHBoxLayout()
        btn_presets = QtWidgets.QPushButton("Presets detectados")
        btn_presets.clicked.connect(self._listar_presets_dialogo)
        botoes.addWidget(btn_presets)

        btn_exportar = QtWidgets.QPushButton("Exportar relatorio PDF")
        btn_exportar.clicked.connect(self._exportar_relatorio)
        botoes.addWidget(btn_exportar)
        botoes.addStretch()

        layout.addLayout(botoes)
        return tab

    # ----- operacoes de fluxo -----
    # ================================================================
    # FUNCAO _SELECIONAR_ARQUIVO: ABRE DIALOGO PARA ESCOLHER IMAGEM
    # ================================================================
    def _selecionar_arquivo(self) -> None:
        arquivo, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Selecionar imagem",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.ppm)",
        )
        if arquivo:
            self._carregar_arquivo(arquivo)

    # ==================================================================
    # FUNCAO _CARREGAR_ARQUIVO: LE IMAGEM, ATUALIZA VIEWER E HISTORICO
    # ==================================================================
    def _carregar_arquivo(self, caminho: str) -> None:
        try:
            path = Path(caminho)
            imagem = carregar_imagem(path)
        except Exception as erro:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Erro", str(erro))
            return
        self.arquivo_atual = path
        self.imagem_atual = imagem
        self.viewer_original.set_image(imagem)
        self.statusBar().showMessage(f"Imagem carregada: {path.name}")
        registrar(self.logger, f"Imagem importada: {path}")
        self._adicionar_historico(path)

    # ===============================================================
    # FUNCAO _ADICIONAR_HISTORICO: MANTEM LISTA DE IMAGENS RECENTES
    # ===============================================================
    def _adicionar_historico(self, caminho: Path) -> None:
        if caminho in self.historico:
            self.historico.remove(caminho)
        self.historico.insert(0, caminho)
        self.historico = self.historico[:10]
        self.lista_historico.clear()
        for item in self.historico:
            self.lista_historico.addItem(str(item))

    # ===============================================================
    # FUNCAO _SELECIONAR_PESOS: COLETA CAMINHO DE PESOS PELO SISTEMA
    # ===============================================================
    def _selecionar_pesos(self) -> None:
        """Permite escolher um arquivo .pth manualmente."""
        caminho, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Selecionar pesos NoisePrint",
            str(Path.cwd() / "weights"),
            "Modelos (*.pth)",
        )
        if caminho:
            self.caminho_pesos_manual = Path(caminho)
            self._atualizando_pesos = True
            self.campo_pesos.setText(caminho)
            self._atualizando_pesos = False
            self._limpar_preset_selecao()
            registrar(self.logger, f"Pesos definidos manualmente: {caminho}")

    # ===========================================================
    # FUNCAO _LIMPAR_PESOS: REMOVE PESOS E RESETA PARA RESIDUAL
    # ===========================================================
    def _limpar_pesos(self) -> None:
        """Restaura modo residual removendo pesos manuais."""
        self.caminho_pesos_manual = None
        if self.campo_pesos is not None:
            self._atualizando_pesos = True
            self.campo_pesos.clear()
            self._atualizando_pesos = False
        self._limpar_preset_selecao()
        registrar(self.logger, "Pesos limpos. Residual sera utilizado.")

    # ===============================================================
    # FUNCAO _AO_MUDAR_CAMPO_PESOS: SINCRONIZA RADIO BUTTONS PELOS DADOS
    # ===============================================================
    def _ao_mudar_campo_pesos(self, texto: str) -> None:
        if self._atualizando_pesos:
            return
        caminho = texto.strip()
        if not caminho:
            self._limpar_preset_selecao()
            self.caminho_pesos_manual = None
            return
        self.caminho_pesos_manual = Path(caminho)
        for botao in self.preset_group.buttons() if self.preset_group else []:
            identificador = self.preset_group.id(botao)
            info = self._preset_id_para_info.get(identificador)
            if info and str(info.arquivo) == caminho:
                botao.setChecked(True)
                return
        self._limpar_preset_selecao()

    # =========================================================
    # FUNCAO _LIMPAR_PRESET_SELECAO: DESMARCA OPCOES DE PRESET
    # =========================================================
    def _limpar_preset_selecao(self) -> None:
        if not self.preset_group:
            return
        self.preset_group.setExclusive(False)
        for botao in self.preset_group.buttons():
            botao.setChecked(False)
        self.preset_group.setExclusive(True)

    # =============================================================
    # FUNCAO _RESOLVER_PRESET_POR_CAMINHO: BUSCA PRESET ASSOCIADO
    # =============================================================
    def _resolver_preset_por_caminho(self, caminho: Path) -> Optional[PresetInfo]:
        caminho_real = caminho.resolve()
        for info in listar_presets():
            if info.arquivo.resolve() == caminho_real:
                return info
        return None

    # ==============================================================
    # FUNCAO _SELECIONAR_PRESET: APLICA PRESET ESCOLHIDO AO CAMPO
    # ==============================================================
    def _selecionar_preset(self, info: PresetInfo) -> None:
        """Preenche o campo manual com o preset escolhido."""
        self._atualizando_pesos = True
        if self.campo_pesos is not None:
            self.campo_pesos.setText(str(info.arquivo))
        self._atualizando_pesos = False
        self.caminho_pesos_manual = info.arquivo

    # ===========================================================================
    # FUNCAO _OBTER_INFO_PESOS: RETORNA CAMINHO EFETIVO E TEXTO DESCRITIVO DO PESO
    # ===========================================================================
    def _obter_info_pesos(self) -> Tuple[Optional[Path], str]:
        texto_manual = ""
        if self.campo_pesos is not None:
            texto_manual = self.campo_pesos.text().strip()

        if texto_manual:
            caminho_manual = Path(texto_manual)
            if caminho_manual.exists():
                info_manual = self._resolver_preset_por_caminho(caminho_manual)
                self.caminho_pesos_manual = caminho_manual
                if info_manual:
                    return (
                        caminho_manual,
                        f"Preset {info_manual.identificador} ({info_manual.nome}) nivel {info_manual.nivel}",
                    )
                return caminho_manual, f"Arquivo manual ({caminho_manual.name})"
            QtWidgets.QMessageBox.warning(
                self,
                "Aviso",
                "Arquivo informado para pesos nao existe. Residual sera utilizado.",
            )
            registrar(self.logger, "Arquivo de pesos informado nao encontrado. Residual aplicado.", nivel="warning")
            self._limpar_pesos()

        selecionado = self.preset_group.checkedId() if self.preset_group else -1
        if selecionado != -1:
            info = self._preset_id_para_info.get(selecionado)
            if info and info.arquivo.exists():
                self.caminho_pesos_manual = info.arquivo
                if self.campo_pesos is not None:
                    self.campo_pesos.setText(str(info.arquivo))
                return info.arquivo, f"Preset {info.identificador} ({info.nome}) nivel {info.nivel}"

        info_lista = listar_presets()
        if info_lista:
            info = info_lista[0]
            if info.arquivo.exists():
                self.caminho_pesos_manual = info.arquivo
                if self.campo_pesos is not None:
                    self.campo_pesos.setText(str(info.arquivo))
                return info.arquivo, f"Preset {info.identificador} ({info.nome}) nivel {info.nivel}"

        self.caminho_pesos_manual = None
        return None, "Residual (filtro passa-alta)"

    # ============================================================
    # FUNCAO _ABRIR_ARQUIVO: CHAMA O SISTEMA PARA ABRIR UM CAMINHO
    # ============================================================
    def _abrir_arquivo(self, caminho: Path) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(caminho.resolve())))

    # =============================================================
    # FUNCAO _EXECUTAR_METODO: PROCESSA A IMAGEM COM O ALGORITMO ESCOLHIDO
    # =============================================================
    def _executar_metodo(self, metodo: str) -> None:
        if self.imagem_atual is None or self.arquivo_atual is None:
            QtWidgets.QMessageBox.warning(self, "Aviso", "Carregue uma imagem primeiro.")
            return

        pasta_imagem = self.trabalho_dir / self.arquivo_atual.stem
        pasta_imagem.mkdir(parents=True, exist_ok=True)
        caminho_pesos, descricao_pesos = self._obter_info_pesos()
        registrar(self.logger, f"Pesos utilizados: {descricao_pesos}")

        if metodo == "NoisePrint":
            kernel = self.spin_noise.value()
            _, mapa_norm = extrair_noiseprint(self.imagem_atual, ksize=kernel)
            mapa = cv2.applyColorMap((mapa_norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
            destino = pasta_imagem / "noiseprint_mapa.png"
            salvar_imagem(destino, mapa)
            self.viewer_noise.set_image(mapa)
            self.viewer_comp1.set_image(self.imagem_atual)
            self.viewer_comp2.set_image(mapa)
            self.resultados.append(
                AnaliseResultado("NoisePrint", destino, None, f"Kernel: {kernel}", caminho_pesos, descricao_pesos)
            )
        elif metodo == "ELA":
            qualidade = self.spin_ela.value()
            _, mapa = aplicar_ela(self.imagem_atual, qualidade=qualidade)
            destino = pasta_imagem / "ela_heatmap.png"
            salvar_imagem(destino, mapa)
            self.viewer_ela.set_image(mapa)
            self.viewer_comp1.set_image(self.imagem_atual)
            self.viewer_comp2.set_image(mapa)
            self.resultados.append(
                AnaliseResultado("ELA", destino, None, f"Qualidade JPEG: {qualidade}", caminho_pesos, descricao_pesos)
            )
        else:
            kernel = self.spin_prnu.value()
            _, mapa = extrair_prnu(self.imagem_atual, h=kernel)
            destino = pasta_imagem / "prnu_mapa.png"
            salvar_imagem(destino, mapa)
            self.viewer_prnu.set_image(mapa)
            self.viewer_comp1.set_image(self.imagem_atual)
            self.viewer_comp2.set_image(mapa)
            self.resultados.append(
                AnaliseResultado("PRNU", destino, None, f"Kernel: {kernel}", caminho_pesos, descricao_pesos)
            )

        self.statusBar().showMessage(f"Metodo {metodo} concluido")
        registrar(self.logger, f"Metodo {metodo} executado para {self.arquivo_atual}")

    # =============================================================
    # FUNCAO _EXPORTAR_RELATORIO: MONTA PDF COM CONTEXTO E MAPAS
    # =============================================================
    def _exportar_relatorio(self) -> None:
        if not self.resultados or self.arquivo_atual is None:
            QtWidgets.QMessageBox.warning(self, "Aviso", "Realize pelo menos uma analise antes do relatorio.")
            return
        contexto: Dict[str, str] = {
            "Arquivo": self.arquivo_atual.name,
            "Observacoes": self.texto_relatorio.toPlainText() or "(sem observacoes)",
        }
        contexto.update(extrair_exif(self.arquivo_atual))
        imagens: List[Tuple[str, Path, str]] = []
        imagens.append(("Imagem original", self.arquivo_atual, "Imagem importada para analise."))
        for resultado in self.resultados:
            legenda = f"{resultado.descricao} | Pesos: {resultado.pesos_descricao}"
            imagens.append((resultado.metodo, resultado.mapa_path, legenda))
        destino = self.trabalho_dir / f"relatorio_{self.arquivo_atual.stem}.pdf"
        gerar_relatorio(destino, contexto, imagens)
        caixa = QtWidgets.QMessageBox(self)
        caixa.setWindowTitle("Relatorio")
        caixa.setText(f"Relatorio salvo em {destino}")
        botao_abrir = caixa.addButton("Abrir PDF", QtWidgets.QMessageBox.AcceptRole)
        caixa.addButton("Fechar", QtWidgets.QMessageBox.RejectRole)
        caixa.exec_()
        if caixa.clickedButton() is botao_abrir:
            self._abrir_arquivo(destino)
        registrar(self.logger, f"Relatorio gerado em {destino}")

    # ============================================================
    # FUNCAO _LISTAR_PRESETS_DIALOGO: EXIBE ARQUIVOS DE PESOS DISPONIVEIS
    # ============================================================
    def _listar_presets_dialogo(self) -> None:
        lista = listar_presets()
        if not lista:
            QtWidgets.QMessageBox.information(self, "Presets", "Nenhum arquivo .pth encontrado em weights/.")
            return
        linhas = []
        for info in lista:
            linhas.append(f"{info.identificador} | {info.nome} | nivel {info.nivel} | {info.arquivo}")
        QtWidgets.QMessageBox.information(self, "Presets detectados", "\n".join(linhas))
    # ======================================================
    # FUNCAO _OBTER_RESIZE_MAXIMO: PONTO DE EXTENSAO PARA REDIMENSIONAMENTO
    # ======================================================
    def _obter_resize_maximo(self) -> Optional[int]:
        return None

    # ============================================================
    # FUNCAO _RESETAR: LIMPA VISUALIZADORES, RESULTADOS E ESTADO
    # ============================================================
    def _resetar(self, silencioso: bool = False) -> None:
        vazio = np.zeros((40, 40, 3), dtype=np.uint8)
        self.arquivo_atual = None
        self.imagem_atual = None
        self.viewer_original.set_image(vazio)
        if hasattr(self, "viewer_noise"):
            self.viewer_noise.set_image(vazio)
        if hasattr(self, "viewer_ela"):
            self.viewer_ela.set_image(vazio)
        if hasattr(self, "viewer_prnu"):
            self.viewer_prnu.set_image(vazio)
        if hasattr(self, "viewer_comp1"):
            self.viewer_comp1.set_image(vazio)
        if hasattr(self, "viewer_comp2"):
            self.viewer_comp2.set_image(vazio)
        self.resultados.clear()
        if not silencioso:
            self.statusBar().showMessage("Aplicacao resetada")
            registrar(self.logger, "Reset executado")

    # =========================================================
    # FUNCAO _MOSTRAR_AJUDA: EXIBE ORIENTACOES DE USO
    # =========================================================
    def _mostrar_ajuda(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Ajuda",
            (
                "Fluxo recomendado:\n"
                "1) Import Image: arraste ou abra um arquivo.\n"
                "2) NoisePrint/ELA/PRNU: ajuste parametros.\n"
                "3) Comparison: avalie resultados.\n"
                "4) Report: exporte PDF."
            ),
        )

    # =========================================================
    # FUNCAO _ADICIONAR_LOG: ANEXA TEXTO AO QUADRO DE LOGS
    # =========================================================
    def _adicionar_log(self, mensagem: str) -> None:
        self.log_box.append(mensagem)

