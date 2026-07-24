"""
data_prep.py — Pré-processamento do SNIS para o dashboard.

Roda UMA vez: baixa o consolidado municipal do SNIS (~57 MB), corrige o
parsing numérico, filtra 2010-2021 e salva um parquet enxuto (~2 MB) que o
app carrega instantaneamente.

Uso:
    python data_prep.py
"""

import os
import re
import zipfile

import pandas as pd
import requests

URL = ("https://github.com/dankkom/snis-rawdata/releases/download/"
       "v2024.02/municipios-consolidado-municipio.zip")
CACHE_ZIP = "snis_consolidado.zip"
SAIDA = "snis_slim.parquet"

# Indicadores usados no dashboard
INDICADORES = ["IN023_AE", "IN055_AE", "IN015_AE", "IN016_AE", "IN047_AE"]


def baixar():
    if os.path.exists(CACHE_ZIP):
        print(f"Usando cache local: {CACHE_ZIP}")
        return
    print("Baixando SNIS (~57 MB)...")
    r = requests.get(URL, timeout=300)
    r.raise_for_status()
    with open(CACHE_ZIP, "wb") as f:
        f.write(r.content)
    print("Download concluído.")


def preparar():
    with zipfile.ZipFile(CACHE_ZIP) as z:
        with z.open("consolidado-municipio.csv") as f:
            # ATENÇÃO ao formato numérico brasileiro do SNIS:
            #   decimal=','    -> "83,86"   vira 83.86  (indicadores)
            #   thousands='.'  -> "825.796" vira 825796 (populações)
            # Sem thousands='.', a população de João Pessoa viraria 825
            # habitantes e corromperia silenciosamente toda média ponderada.
            bruto = pd.read_csv(
                f, sep=";", encoding="utf-8",
                decimal=",", thousands=".",
                low_memory=False,
            )

    # Colunas vêm como "IN055_AE - Índice de ..." -> mantemos só o código
    bruto.columns = [re.sub(r"\s*-\s*.*$", "", c).strip() for c in bruto.columns]

    cols = (["ano_referencia", "codigo_municipio", "nome_municipio",
             "sigla_uf", "POP_TOT", "POP_URB"] + INDICADORES)
    dados = bruto[cols].copy()

    for c in INDICADORES + ["POP_TOT", "POP_URB"]:
        dados[c] = pd.to_numeric(dados[c], errors="coerce")

    # Recorte temporal confiável (baixa completude antes de 2010; quebra de
    # série a partir de 2022)
    dados = dados[
        (dados.ano_referencia >= 2010) & (dados.ano_referencia <= 2021)
    ].copy()

    dados["ano_referencia"] = dados["ano_referencia"].astype("int16")
    dados["sigla_uf"] = dados["sigla_uf"].astype("category")

    dados.to_parquet(SAIDA, index=False)

    ne = dados[(dados.ano_referencia == 2021)]
    print(f"\nSalvo: {SAIDA}")
    print(f"Linhas: {len(dados):,} | Municípios em 2021: {len(ne):,}")
    print(f"UFs: {dados.sigla_uf.nunique()}")
    # Sanidade: pop. urbana do Brasil em 2021 deve ficar perto de 182 mi
    print(f"Pop. urbana Brasil 2021: {ne.POP_URB.sum():,.0f} hab.")


if __name__ == "__main__":
    baixar()
    preparar()
