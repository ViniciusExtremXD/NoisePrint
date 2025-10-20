# Noiseprint - Video + Tutorial (Prototipo simples)

Noiseprint e uma tecnica de foto forense que aprende um fingerprint caracteristico do modelo de camera. Diferente do PRNU tradicional (ruido de resposta do sensor), o Noiseprint usa uma CNN para destacar padroes de processamento na cadeia de formacao da imagem. Este prototipo academico entrega uma versao simplificada com residual fallback, interface grafica opcional e organizacao pensada para o video/tutorial da disciplina.

## Como executar (Windows)
### Ambiente
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Modo grafico
```
python main.py
```
1. Escolha os arquivos (ou glob) em **Entrada**.
2. Opcional: selecione pesos reais (`weights/noiseprint.pth`).
3. Defina pasta de saida e opcoes desejadas.
4. Clique em **Executar**. Os logs aparecem na janela.
5. Ao final, use **Abrir pasta de saida** ou **Abrir ultimo resultado**.

### Modo CLI
```
python main.py --entrada "data/input\*.jpg" --salvar-heatmap --salvar-overlay
```
Opcoes uteis:
- `--pesos caminho\para\noiseprint.pth`
- `--preset noiseprint.pth` (procura dentro de `weights/`)
- `--saida data/output_custom`
- `--resize-max 2048`
- `--salvar-intermediarios`
- `--nivel-log DEBUG`

## Roteiro do video (<= 10 min)
- 0:00-1:00 - Abertura, apresentacao do grupo, objetivo do tutorial.
- 1:00-3:00 - Teoria: ruido de sensor, fingerprints, diferenca entre Noiseprint e PRNU.
- 3:00-7:00 - Demo: executar GUI/CLI em duas imagens (original vs manipulada), mostrar heatmap e overlay.
- 7:00-8:30 - Limitacoes: sem pesos oficiais, CNN placeholder, situacoes em que o residual falha.
- 8:30-10:00 - Proximos passos: integrar pesos reais, comparar com PRNU, montar testes controlados.

## Referencias
- Cozzolino et al., "Noiseprint: a CNN-based camera model fingerprint", 2019.
- Lukacs et al., "PRNU-based camera identification", 2015.
- Repositorio oficial Noiseprint: https://github.com/grip-unina/noiseprint
- Documentacao OpenCV: https://docs.opencv.org/

## Aviso
Nenhum peso real acompanha este prototipo. O modelo incluido e apenas um placeholder para fins didaticos; a demonstracao mostra o fluxo geral antes de plugar a rede treinada.

## Estrutura resumida
- `main.py` reune funcoes de processamento (carregar_imagem, residual_highpass, normalizar_mapa), CNN placeholder, pipeline em lote com geracao de artefatos (execucao.yaml, manifesto.csv, logs.txt/jsonl, relatorio_execucao.md), alem da CLI e da GUI.
- `data/input`, `data/output`, `weights` armazenam as imagens de entrada/saida e pesos opcionais (mantidos vazios com `.gitkeep`).

## Observacoes finais
- Trabalho academico de Computacao Visual (video-tutorial ate 10 minutos sobre Noiseprint).
- Cada execucao gera uma pasta `data/output/run_YYYY-MM-DD_HH-MM-SS/` com resultados, metricas por imagem e logs reprodutiveis.
- CLI e GUI compartilham exatamente a mesma pipeline, garantindo demonstracao pratica consistente.

