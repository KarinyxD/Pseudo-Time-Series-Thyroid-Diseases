import matplotlib.pyplot as plt
import seaborn as sns

def plot_trajectory(df_final):
    plt.figure(figsize=(12, 6))
    
    # Definindo cores para ficar igual artigo científico
    # Saudável (Cinza/Azul), G (Amarelo), F (Laranja), E (Vermelho), H (Roxo)
    palette = {'-': 'tab:blue', 'G': 'tab:olive', 'F': 'tab:orange', 'E': 'tab:red', 'H': 'tab:purple'}
    
    # Plot de dispersão
    sns.scatterplot(
        data=df_final, 
        x='pseudo_time', 
        y='TSH', 
        hue='target_clean',
        palette=palette,
        alpha=0.6, # Transparência para ver a densidade
        edgecolor=None
    )
    
    plt.title('Reconstrução da Trajetória do Hipotireoidismo (Pseudo-Time vs TSH)')
    plt.xlabel('Pseudo Time (Distância Geodésica da Saúde)')
    plt.ylabel('TSH (Nível Sérico)')
    plt.grid(True, alpha=0.3)
    
    # Focar o gráfico onde importa (TSH < 150 geralmente pega a maioria)
    # Se quiser ver tudo, comente a linha abaixo
    plt.ylim(-5, 200) 
    
    plt.show()

# --- EXECUÇÃO DO PASSO 4 ---
# plot_trajectory(df_final)