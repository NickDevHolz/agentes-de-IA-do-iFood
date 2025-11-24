# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------
# POC - Agente Interno iFood com RAG
# ---------------------------------------------------------------------------
# Este script implementa um agente de perguntas e respostas simples
# utilizando a técnica de RAG (Retrieval-Augmented Generation) baseada
# em similaridade de cosseno.
#
# Dependências:
# - pandas: Para leitura e manipulação do arquivo CSV da base de conhecimento.
# - sentence-transformers: Para geração de embeddings e cálculo de similaridade.
#
# Modelo de Embeddings: all-MiniLM-L6-v2
# ---------------------------------------------------------------------------

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os

# --- Configurações Iniciais ---

# Nome do arquivo CSV que contém a base de conhecimento.
# O arquivo deve ter as colunas 'pergunta' e 'resposta'.
NOME_ARQUIVO_CSV = 'base_conhecimento_ifood_genai-exemplo.csv'

# Modelo de embeddings da biblioteca sentence-transformers.
# 'all-MiniLM-L6-v2' é um modelo leve e eficiente para tarefas de similaridade semântica.
MODELO_EMBEDDINGS = 'all-MiniLM-L6-v2'

# Limiar de similaridade. Se a pontuação da melhor resposta for menor que este valor,
# o agente retornará uma mensagem de fallback. O valor deve ser entre 0.0 e 1.0.
LIMIAR_SIMILARIDADE = 0.40

# Nome do arquivo para salvar/carregar os embeddings cacheados.
ARQUIVO_EMBEDDINGS_CACHE = 'embeddings_cache.pt'

# --- Funções do Agente ---

def carregar_base_conhecimento(caminho_arquivo: str):
    """
    Carrega a base de conhecimento a partir de um arquivo CSV.

    Args:
        caminho_arquivo (str): O caminho para o arquivo CSV.

    Returns:
        pandas.DataFrame: Um DataFrame com as perguntas e respostas, ou None se o arquivo não for encontrado.
    """
    if not os.path.exists(caminho_arquivo):
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
        print("Por favor, certifique-se de que o arquivo CSV está no mesmo diretório que o script.")
        return None
        
    print(f"Carregando a base de conhecimento de '{caminho_arquivo}'...")
    try:
        # Lê o CSV e o retorna como um DataFrame do pandas
        df = pd.read_csv(caminho_arquivo, encoding='utf-8')
        if 'pergunta' not in df.columns or 'resposta' not in df.columns or 'fonte' not in df.columns:
            print("Erro: O arquivo CSV deve conter as colunas 'pergunta' e 'resposta'.")
            return None
        print("Base de conhecimento carregada com sucesso!")
        return df
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo CSV: {e}")
        return None

def criar_embeddings_base(base_conhecimento: pd.DataFrame, modelo: SentenceTransformer):
    """
    Gera os embeddings para todas as perguntas na base de conhecimento.

    Args:
        base_conhecimento (pd.DataFrame): O DataFrame contendo a coluna 'pergunta'.
        modelo (SentenceTransformer): O modelo pré-treinado para gerar os embeddings.

    Returns:
        torch.Tensor: Um tensor contendo os embeddings de todas as perguntas.
    """
    # Verifica se um arquivo de cache de embeddings já existe
    if os.path.exists(ARQUIVO_EMBEDDINGS_CACHE):
        print(f"Carregando embeddings do cache '{ARQUIVO_EMBEDDINGS_CACHE}'...")
        embeddings_base = torch.load(ARQUIVO_EMBEDDINGS_CACHE)
        print("Embeddings carregados do cache com sucesso.")
        return embeddings_base

    print("Criando embeddings para a base de conhecimento... (Isso pode levar um momento na primeira execução)")
    # Extrai a lista de perguntas do DataFrame
    perguntas_base = base_conhecimento['pergunta'].tolist()
    # Codifica as perguntas para gerar os vetores de embedding
    embeddings_base = modelo.encode(perguntas_base, convert_to_tensor=True)
    print("Embeddings criados com sucesso.")
    # Salva os embeddings em um arquivo de cache para uso futuro
    torch.save(embeddings_base, ARQUIVO_EMBEDDINGS_CACHE)
    print(f"Embeddings criados e salvos em '{ARQUIVO_EMBEDDINGS_CACHE}'.")
    return embeddings_base

def encontrar_melhor_resposta(pergunta_usuario: str, base_conhecimento: pd.DataFrame, embeddings_base: torch.Tensor, modelo: SentenceTransformer):
    """
    Encontra a resposta mais similar para a pergunta do usuário na base de conhecimento.

    Args:
        pergunta_usuario (str): A pergunta feita pelo usuário.
        base_conhecimento (pd.DataFrame): O DataFrame com as perguntas e respostas.
        embeddings_base (torch.Tensor): Os embeddings pré-calculados da base.
        modelo (SentenceTransformer): O modelo de embeddings.

    Returns:
        tuple[str, str | None]: Uma tupla contendo a resposta e a fonte, ou uma mensagem de fallback e None.
    """
    # Gera o embedding para a pergunta do usuário
    embedding_usuario = modelo.encode(pergunta_usuario, convert_to_tensor=True)

    # Calcula a similaridade de cosseno entre o embedding do usuário e todos os embeddings da base
    similaridades = util.cos_sim(embedding_usuario, embeddings_base)

    # Encontra a maior pontuação de similaridade e seu índice
    # O [0] é para extrair o tensor de dentro da lista de resultados
    melhor_indice = torch.argmax(similaridades[0]).item()
    maior_similaridade = similaridades[0][melhor_indice].item()

    # Compara a maior similaridade com o limiar definido
    if maior_similaridade >= LIMIAR_SIMILARIDADE:
        # Se for alta o suficiente, retorna a resposta correspondente
        resposta = base_conhecimento.iloc[melhor_indice]['resposta']
        fonte = base_conhecimento.iloc[melhor_indice]['fonte']
        # print(f"(Debug: Similaridade de {maior_similaridade:.2f})") # Descomente para depuração
        return resposta, fonte
    else:
        # Se a similaridade for muito baixa, retorna a mensagem de fallback
        return "Desculpe, não consegui encontrar uma resposta para sua pergunta. Por favor, tente reformulá-la.", None

# --- Função Principal (Loop Interativo) ---

def main():
    """
    Função principal que inicializa o agente e inicia o loop de interação com o usuário.
    """
    # 1. Carregar a base de conhecimento do arquivo CSV
    base_conhecimento = carregar_base_conhecimento(NOME_ARQUIVO_CSV)
    if base_conhecimento is None:
        return # Encerra a execução se a base não puder ser carregada

    # 2. Carregar o modelo de sentence-transformer
    try:
        modelo = SentenceTransformer(MODELO_EMBEDDINGS)
    except Exception as e:
        print(f"Erro ao carregar o modelo '{MODELO_EMBEDDINGS}'. Verifique sua conexão com a internet ou a instalação.")
        print(f"Detalhes do erro: {e}")
        return
        
    # 3. Gerar embeddings para a base de conhecimento
    embeddings_base = criar_embeddings_base(base_conhecimento, modelo)

    # 4. Iniciar o loop interativo
    print("\n--- Agente iFood (POC) ---")
    print("Olá! Faça uma pergunta sobre nossos serviços. Digite 'sair' para encerrar.")
    
    while True:
        # 5. Receber pergunta do usuário
        pergunta_usuario = input("\nVocê: ")

        if pergunta_usuario.lower() == 'sair':
            print("Agente encerrado. Até logo!")
            break

        # 6. Encontrar e exibir a melhor resposta
        resposta, fonte = encontrar_melhor_resposta(pergunta_usuario, base_conhecimento, embeddings_base, modelo)
        print(f"\nAgente: {resposta}")
        if fonte:
            print(f"(Fonte: {fonte})")

if __name__ == "__main__":
    main()