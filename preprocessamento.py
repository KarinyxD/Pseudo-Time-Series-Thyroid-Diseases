import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# --- CONFIGURAÇÕES ---
# Focamos apenas em Negativos e Hipotireoidismo
classes = ['-', 'E', 'F', 'G'] 

# Mapeamento para referência futura (preservando sub-classes)
# Vamos usar isso para colorir o gráfico depois

class_mapping_details = {
    '-': 0,  # Healthy
    'G': 1,  # Compensated (Início da falha) TSH elevado e T3, T4 normais.
    'F': 2,  # Primary (Falha estabelecida) TSH bem elevado, T4 baixo, T3 baixo ou normal.
    'E': 2,  # Hypothyroid (Severo) TSH muito elevado, T4 e T3 muito baixo.
    # Mesclar severidade (E) na 2 (F), já que temos poucos casos de E (1 paciente).
    # 'H': 4   # Secondary (Ramo paralelo) TSH baixo/normal, T4, T3 baixo.
}

# Features Numéricas (Essenciais para a Distância Euclidiana)
# Incluímos 'age' aqui como feature, não como constraint
num_features = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'age']

# Limites para limpeza de outliers
limits = {
    'TSH': (0, 500), 'T3': (0, 20), 'TT4': (0, 400),
    'T4U': (0, 2), 'FTI': (0, 300), 'age': (0, 100),
}

def preprocessing_pts():
    # Carregar Dataset
    df = pd.read_csv("csv/thyroidDF.csv") 
    
    # Limpeza Básica de Target
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].str.strip().str.upper()
    
    # FILTRAGEM: Manter apenas Saudáveis e Hipo (Remove Hiper)
    pattern = '|'.join(classes)
    df = df[df['target'].str.contains(pattern, na=False, regex=True)]
    
    # Limpeza de Labels (Pega a primeira letra relevante)
    # Ex: "ABFC" vira "F"
    def clean_target(x):
        for letter in classes:
            if letter in x:
                return letter
        return np.nan
    
    df['target_clean'] = df['target'].apply(clean_target)
    df = df.dropna(subset=['target_clean'])
    
    # Mapear para números ordinais 
    df['severity_label'] = df['target_clean'].map(class_mapping_details)
    
    # Aplicação dos Limites (Outliers)
    for col, (min_val, max_val) in limits.items():
        if col in df.columns:
            df = df[df[col].isna() | ((df[col] >= min_val) & (df[col] <= max_val))]
            
    # Seleção de Features
    # Focamos nas numéricas para a construção da trajetória
    df_model = df[num_features].copy()
    
    # Imputação (KNN)
    imputer = KNNImputer(n_neighbors=10, weights="distance")
    df_imputed_vals = imputer.fit_transform(df_model)
    df_imputed = pd.DataFrame(df_imputed_vals, columns=num_features, index=df_model.index)
    
    # PADRONIZAÇÃO (Z-SCORE)
    # Deixar as features na mesma escala de importância
    scaler = StandardScaler()
    df_normalized_vals = scaler.fit_transform(df_imputed)
    df_normalized = pd.DataFrame(df_normalized_vals, columns=num_features, index=df_model.index)

    # Retornamos:
    # 1. df_normalized: Para calcular a matriz de distância (padronizado) 
    # 2. df_imputed: Os valores reais para plotar gráficos depois (valores originais)
    # 3. df['severity_label']: As classes (feature de severidade da doença (0 1 2 3 4))
    
    print(f"Dados processados. Total de pacientes: {len(df)}")
    print("Distribuição das classes:\n", df['target_clean'].value_counts())
    
    return df_normalized, df_imputed, df['severity_label']

# Executar
# df_norm, df_real, label_severity = preprocessing_pts()