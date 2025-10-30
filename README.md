# ================================================================
# PROJETO: NOISEPRINT - PROTOTIPO (VIDEO + TUTORIAL)
#
# HEITOR MACIEL - 10402559
# VITOR PEPE - 10339754
# VINICIUS MAGNO - 10401365
# KAIKI BELLINI BARBOSA - 10402509
# ================================================================

Analise forense digital que reune NoisePrint simplificado, Error Level Analysis (ELA) e PRNU. O foco e demonstrar usabilidade, clareza visual e fluxo linear para investigar autenticidade de fotografias.

## Estrutura do projeto
main.py                 # ponto de entrada (PyQt5)
requirements.txt        # dependencias
src/noiseprint.py       # extracao NoisePrint
src/ela.py              # Error Level Analysis
src/prnu.py             # estimativa PRNU
ui/main_window.py       # interface grafica PyQt
utils/image_io.py       # leitura/escrita de imagens
utils/logging_utils.py  # configuracao de logs
utils/report.py         # exportacao de relatorio PDF
samples/demo.ppm        # exemplo colorido
samples/demo_manipulated.ppm  # exemplo com manipulacao
outputs/                # gerado em tempo de execucao

## Pre-requisitos
- Python 3.9+
- Ambiente virtual:
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1

- Instale dependencias:
  python -m pip install -r requirements.txt

## Execucao
python main.py

### Fluxo da interface
1. Import Image: arraste/solte ou selecione arquivo; historico visivel; botoes Reset e Help.
2. NoisePrint / ELA / PRNU: ajuste parametros, clique em "Executar analise" e visualize original vs resultado (zoom/pan disponiveis).
3. Comparison: compara original com o ultimo mapa gerado.
4. Report: adicione observacoes e exporte PDF com data, EXIF e imagens.

## Usabilidade demonstrada
- Menus intuitivos seguindo o fluxo Import -> Metodo -> Comparison -> Report.
- Feedback continuo via status bar, painel de logs e mensagens.
- Prevencao de erros: validacao de formato, alerta de preset ausente, botoes de reset/ajuda.
- Memorabilidade: interface autoexplicativa, tooltips curtos, historico para repetir analises.
- Exportacao completa: PNGs, log (outputs/forensic.log) e relatorio PDF.

## Pesos NoisePrint
Na aba "Import Image" utilize o campo "Pesos NoisePrint (opcional)" para digitar o caminho do arquivo .pth ou clique em "Selecionar pesos...". O botao "Usar residual" limpa a escolha manual. 

O botão "Presets detectados" ajuda a mapear a pasta weights/. 
Caso nao haja pesos, o modo residual e usado automaticamente.

## Observacoes tecnicas
- Metodos simplificados para fins academicos (NoisePrint fallback, ELA via recompressao JPEG, PRNU por subtracao suave).
- Multiplataforma (Windows/Linux). Em macOS substitua os.startfile por open se necessario.
- Relatorio PDF gerado com FPDF em outputs/<imagem>/.

## Roadmap sugerido
- Integrar NoisePrint oficial (CNN) e PRNU completo.
- Adicionar metricas quantitativas (PSNR, SSIM).
- Automatizar comparacao entre duas imagens (aba "Comparison").

