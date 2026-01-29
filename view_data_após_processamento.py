import pandas as pd
import preprocessamento as pp
 
# 1. Função de pré-processamento
df_norm, df_real, labels = pp.preprocessing_pts()

# 2. Preparação para o CSV de Visualização
# Ex: "TSH_real" e "TSH_zscore"
# Dados reais vs dados normalizados 
df_visual = df_real.add_suffix('_real')        # Adiciona sufixo nos valores reais
df_norm_visual = df_norm.add_suffix('_zscore') # Adiciona sufixo nos valores normalizados

# 3. Juntar tudo em um único DataFrame 
# axis=1 significa juntar colunas, index=True garante que os pacientes alinhem corretamente
df_final_visual = pd.concat([df_visual, df_norm_visual, labels], axis=1)

# 4. (Opcional) Reordenar colunas para facilitar a leitura
# Coloca o label no começo, seguido de parzinhos (TSH_real, TSH_zscore, ...)
cols = ['severity_label']
for col in df_real.columns:
    cols.append(f"{col}_real")
    cols.append(f"{col}_zscore")
try:
    df_final_visual = df_final_visual[cols]
except KeyError:
    pass 

# 5. Salvar o CSV
nome_arquivo = "dados_processados.csv"
df_final_visual.to_csv(nome_arquivo, index=True)

print(f"Arquivo '{nome_arquivo}' gerado com sucesso!")
print("Para conferir se o Z-Score faz sentido comparado ao valor real.")