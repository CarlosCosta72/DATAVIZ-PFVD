# Escada do Saneamento — Dashboard interativo (PFVD)

Dashboard em Plotly Dash para diagnóstico de água e esgoto por estado,
com identificação do melhor e pior município segundo critério escolhido
pelo usuário. Dados: SNIS 2010–2021 (via dankkom/snis-rawdata), com
médias ponderadas por população.

## Como rodar

```bash
pip install -r requirements.txt
python data_prep.py   # 1ª vez apenas: baixa o SNIS (~57 MB) e gera snis_slim.parquet
python app.py         # abre em http://127.0.0.1:8050
```

## Estrutura

| Arquivo | Papel |
|---|---|
| `data_prep.py` | Pré-processamento (roda uma vez). Corrige o parsing numérico brasileiro (`decimal=','` + `thousands='.'`) e gera o parquet enxuto |
| `snis_slim.parquet` | Dados prontos (~2 MB) — o app nunca lê o CSV bruto |
| `app.py` | O dashboard: layout, figuras e callback |

## Decisões documentadas

Cada gráfico tem um painel recolhível "ⓘ Por que este gráfico é assim?" que explica a escolha feita.

Regras de elegibilidade do ranking municipal: população urbana ≥ 20 mil
habitantes e os três indicadores da cascata preenchidos no ano — evita que
ruído declaratório de municípios minúsculos domine o ranking.
