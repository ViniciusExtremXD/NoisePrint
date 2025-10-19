# Noiseprint - Prototipo (Video + Tutorial)

## Integrantes e RA
- Heitor Maciel - RA 10402599
- Vitor Pepe - RA 10339754
- Vinicius Magno - RA 10401365
- Kaiki Bellini Barbosa - RA 10402509

## Contexto do projeto
Prototipo desenvolvido para a disciplina de Computacao Visual - Universidade Presbiteriana Mackenzie, opcao 3 (video-tutorial de 10 minutos sobre **Noiseprint** - fingerprint de modelo de camera via CNN). O material combina explicacao teorica e demonstracao pratica com referencias ao trabalho original de Noiseprint e estudos correlatos sobre PRNU (Photo-Response Non-Uniformity).

## Como rodar (Windows)
1. `py -3 -m venv .venv`
   `.\.venv\Scripts\Activate.ps1`
2. `python -m pip install -r requirements.txt`
3. Adicione 2-3 imagens em `data\input`.
4. (Opcional) copie `noiseprint.pth` para `weights\`.
5. Execute a CLI:
   `python scripts\extract_noiseprint.py --input "data/input\*.jpg" --save-heatmap --save-overlay`
6. Para o notebook:
   `jupyter notebook notebooks\01_noiseprint_demo.ipynb`

## Arquitetura do prototipo
- **Fallback**: aplica residual de alta frequencia (`Gaussian blur` + subtracao) para simular o comportamento de Noiseprint em ambientes sem pesos reais.
- **Modelo real**: se `weights/noiseprint.pth` estiver presente, a pipeline normaliza a imagem, envia para uma CNN rasa (`NoiseprintCNN`) e normaliza a saida para `[0,1]`. O prototipo espera pesos compativeis com a arquitetura placeholder; para integrar o Noiseprint original basta ajustar a classe/modelo e carregar os pesos adequados.
- **Pos-processamento**: geracao de heatmap colorido e sobreposicao com a imagem original.

## Roteiro sugerido para o video (10 min)
1. **Introducao teorica (3-4 min)**
   - Foto-forense e fingerprint de sensores.
   - Conceito de Noiseprint vs. PRNU.
   - Principais referencias e datasets sugeridos.
2. **Demonstracao pratica (6-7 min)**
   - Mostrar estrutura do repositorio e requisitos.
   - Executar a CLI com fallback e, se disponivel, com pesos reais.
   - Navegar pelo notebook destacando analise dos mapas (resposta forte em bordas/texturas, limitacoes do residual manual).

## Limitacoes e proximos passos
- Substituir o fallback pela implementacao oficial do Noiseprint com pesos treinados.
- Comparar resultados com PRNU tradicional para estabelecer baseline.
- Montar conjunto de testes automatizados (unitarios e validacao de performance) e investigar aceleracao em GPU.

## Aviso sobre pesos
Os pesos reais nao estao incluidos por restricoes de distribuicao. Assim que `weights/noiseprint.pth` for adicionado, a pipeline detecta automaticamente e passa a usar o modelo neural em vez do residual manual.
