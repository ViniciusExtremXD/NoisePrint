# Noiseprint - Video + Tutorial (Prototipo simples)

Noiseprint e uma tecnica de foto forense que aprende um fingerprint caracteristico do modelo de camera. Diferente do PRNU tradicional (ruido de resposta do sensor), o Noiseprint usa uma CNN para destacar padroes de processamento da cadeia de formacao da imagem.

## Como executar (Windows)
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\noiseprint.py --input "data/input\*.jpg" --save-heatmap --save-overlay
```

## Roteiro do video (<= 10 min)
- 0:00-1:00  Abertura, apresentacao do grupo, objetivo do tutorial.
- 1:00-3:00  Teoria: ruido de sensor, fingerprints, diferenca entre Noiseprint e PRNU.
- 3:00-7:00  Demo: rodar o script em duas imagens (original vs manipulada), mostrar heatmap e overlay.
- 7:00-8:30  Limitacoes: sem pesos oficiais, CNN placeholder, casos em que o residual falha.
- 8:30-10:00 Proximos passos: integrar pesos reais, comparar com PRNU, montar testes controlados.

## Referencias
- Cozzolino et al., "Noiseprint: a CNN-based camera model fingerprint," 2019.
- Lukacs et al., "PRNU-based camera identification," 2015.
- Noiseprint project repo: https://github.com/grip-unina/noiseprint
- OpenCV documentation: https://docs.opencv.org/

## Aviso
Nenhum peso real acompanha este prototipo. O modelo incluso e apenas um placeholder para fins didaticos; a demonstracao serve para ilustrar o fluxo geral antes de plugar a rede treinada.
