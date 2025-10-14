# Noiseprint – Protótipo (Vídeo + Tutorial)

## Integrantes e RA
- Heitor Maciel — RA 10402599  
- Vitor Pepe — RA 10339754  
- Vinícius Magno — RA 10401365  
- Kaiki Bellini Barbosa — RA 10402509

## Contexto do projeto
Protótipo desenvolvido para a disciplina de Computação Visual – Universidade Presbiteriana Mackenzie, opção 3 (vídeo-tutorial de 10 minutos sobre **Noiseprint** – fingerprint de modelo de câmera via CNN). O material combina explicação teórica e demonstração prática com referências ao trabalho original de Noiseprint e estudos correlatos sobre PRNU (Photo-Response Non-Uniformity).

## Como rodar (Windows)
1. `py -3 -m venv .venv`  
   `.\.venv\Scripts\Activate.ps1`
2. `pip install -r requirements.txt`
3. Adicione 2–3 imagens em `data\input`.
4. (Opcional) copie `noiseprint.pth` para `weights\`.
5. Execute a CLI:  
   `python scripts\extract_noiseprint.py --input "data/input\*.jpg" --save-heatmap --save-overlay`
6. Para o notebook:  
   `jupyter notebook notebooks\01_noiseprint_demo.ipynb`

## Arquitetura do protótipo
- **Fallback**: aplica residual de alta frequência (`Gaussian blur` → subtração) para simular o comportamento de Noiseprint em ambientes sem pesos reais.
- **Modelo real**: se `weights/noiseprint.pth` estiver presente, a pipeline normaliza a imagem, envia para uma CNN rasa (`NoiseprintCNN`) e normaliza a saída para `[0,1]`. O protótipo espera pesos compatíveis com a arquitetura placeholder; para integrar o Noiseprint original basta ajustar a classe/modelo e carregar os pesos adequados.
- **Pós-processamento**: geração de heatmap colorido e sobreposição com a imagem original.

## Roteiro sugerido para o vídeo (10 min)
1. **Introdução teórica (3–4 min)**  
   - Foto-forense e fingerprint de sensores.  
   - Conceito de Noiseprint vs. PRNU.  
   - Principais referências e datasets sugeridos.
2. **Demonstração prática (6–7 min)**  
   - Mostrar estrutura do repositório e requisitos.  
   - Executar a CLI com fallback e, se disponível, com pesos reais.  
   - Navegar pelo notebook destacando análise dos mapas (resposta forte em bordas/texturas, limitações do residual manual).

## Limitações e próximos passos
- Substituir o fallback pela implementação oficial do Noiseprint com pesos treinados.
- Comparar resultados com PRNU tradicional para estabelecer baseline.
- Montar conjunto de testes automatizados (unitários e validação de performance) e investigar aceleração em GPU.

## Aviso sobre pesos
Os pesos reais não estão incluídos por restrições de distribuição. Assim que `weights/noiseprint.pth` for adicionado, a pipeline detecta automaticamente e passa a usar o modelo neural ao invés do residual manual.

