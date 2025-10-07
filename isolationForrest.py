# -*- coding: utf-8 -*-
"""
Script: consolidar_medicoes_cond_evap_superaq.py
Autor: Matheus Blanco
Descrição:
  Lê todos os arquivos Excel (.xlsx) da pasta informada contendo colunas:
  [data_hora, id_equipamento, condensacao, evaporacao, superaquecimento],
  e consolida tudo em um único DataFrame.
"""

import os
import pandas as pd

# Diretório com os arquivos Excel
pasta = r"C:\Users\Matheus Blanco\OneDrive - ESTECH ESCO & ENGENHARIA LTDA\Documentos\modelos\anomalia\dados"

# Lista para armazenar os DataFrames individuais
lista_dfs = []

# Percorre todos os arquivos .xlsx
for arquivo in os.listdir(pasta):
    if arquivo.endswith(".xlsx"):
        caminho_arquivo = os.path.join(pasta, arquivo)
        print(f"Lendo arquivo: {arquivo}")

        try:
            df = pd.read_excel(caminho_arquivo)
            df.columns = df.columns.str.strip().str.lower()

            # Confere se contém as colunas esperadas
            colunas_esperadas = ["data_hora", "id_equipamento", "condensacao", "evaporacao", "superaquecimento"]
            faltantes = [c for c in colunas_esperadas if c not in df.columns]
            if faltantes:
                print(f"Aviso: colunas faltando em {arquivo}: {faltantes}")
                continue

            # Conversão e limpeza
            df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")
            df = df.dropna(subset=["data_hora"])
            df["id_equipamento"] = df["id_equipamento"].astype(str)

            lista_dfs.append(df)

        except Exception as e:
            print(f"Erro ao processar {arquivo}: {e}")
            
            
            

# Garantir que a coluna está no formato datetime
df['data_hora'] = pd.to_datetime(df['data_hora'])

# Criar as novas colunas
df['data'] = df['data_hora'].dt.date
df['hora'] = df['data_hora'].dt.time
df['id_equipamento'] = df['id_equipamento'].astype('category')




from sklearn.ensemble import IsolationForest


params_iforest = {
    "n_estimators": 300,       # número de árvores
    "contamination": 0.02,     # fração esperada de anomalias
    "max_samples": "auto",     # amostragem automática
    "max_features": 3,         # número de variáveis usadas em cada split
    "bootstrap": False,        # amostragem sem reposição
    "random_state": 42,        # reprodutibilidade
    "n_jobs": -1               # usa todos os núcleos da CPU
}


model = IsolationForest(**params_iforest).fit(df[['condensacao', 'evaporacao','superaquecimento']])


df['anomalia'] = model.predict(df[['condensacao', 'evaporacao','superaquecimento']])
df['anomaliaScore'] = model.decision_function(df[['condensacao', 'evaporacao','superaquecimento']])

sns.jointplot(data=df, x="condensacao", y="evaporacao")


df.head()

pasta = r"C:\Users\Matheus Blanco\OneDrive - ESTECH ESCO & ENGENHARIA LTDA\Documentos\modelos\anomalia"


df.to_excel(pasta + '/anomalias/dados_predicoes.xlsx')





import seaborn as sns
import matplotlib.pyplot as plt
import os

# Caminho base — note o "r" no início (raw string)
pasta = r"C:\Users\Matheus Blanco\OneDrive - ESTECH ESCO & ENGENHARIA LTDA\Documentos\modelos\anomalia"

# Cria subpasta 'figuras' dentro de 'dados'
pasta_figuras = os.path.join(pasta, "figuras")
os.makedirs(pasta_figuras, exist_ok=True)

# === Gráfico 1: sem hue ===
g1 = sns.jointplot(data=df, x="condensacao", y="evaporacao")
fig_path1 = os.path.join(pasta_figuras, "condensacao_evaporacao_base.png")
g1.figure.savefig(fig_path1, dpi=300, bbox_inches="tight")
plt.close(g1.figure)
print(f"Gráfico 1 salvo em: {fig_path1}")

# === Gráfico 2: com hue (anomalias) ===
g2 = sns.jointplot(
    data=df,
    x="condensacao",
    y="evaporacao",
    hue="anomalia",
    palette={-1: "black", 1: "red"}
)
fig_path2 = os.path.join(pasta_figuras, "condensacao_evaporacao_anomalias.png")
g2.figure.savefig(fig_path2, dpi=300, bbox_inches="tight")
plt.close(g2.figure)
print(f"Gráfico 2 salvo em: {fig_path2}")



# Configuração MLflow
import mlflow
import os 

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://164.90.253.107:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "estech77"
os.environ["AWS_SECRET_ACCESS_KEY"] = "evs741oACEp6"


mlflow.set_experiment("Anomalias_racks")






# Nome da execução e caminho do artefato
run_name = "isolationForrest"
artifact_path = "isolationForrest"

# Inicia run no MLflow
with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params(params_iforest)

    # Loga modelo com input_example tratado
    mlflow.sklearn.log_model(
        sk_model=model,
        input_example=None,
        name=artifact_path,
        registered_model_name='isolationForrest',
        signature=None
    )

    print(f"Run registrada em: {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")





