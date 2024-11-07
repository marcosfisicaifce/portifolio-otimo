# portfolio_otimo.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def main():
    st.set_page_config(page_title="Portfólio Ótimo - Otimização de Investimentos", layout="wide")
    st.title("Portfólio Ótimo: Otimize Seu Portfólio de Investimentos")

    st.markdown("""
    ## Introdução

    **Portfólio Ótimo** é uma plataforma projetada para ajudar investidores a otimizar seus portfólios utilizando a **Teoria Moderna do Portfólio** desenvolvida por **Harry Markowitz**. A plataforma permite que os usuários selecionem múltiplas ações e fornece uma alocação otimizada com base na relação risco-retorno.

    ### Como Funciona

    A plataforma utiliza dados históricos das ações para calcular:

    - **Retornos Esperados**: A média dos retornos históricos.
    - **Volatilidade (Risco)**: Medida pelo desvio padrão dos retornos.
    - **Matriz de Covariância**: Mede como as ações se movem juntas.
    - **Fronteira Eficiente**: Um conjunto de portfólios ótimos que oferecem o maior retorno esperado para um nível definido de risco.

    **Portfólio Ótimo** realiza simulações para gerar uma variedade de portfólios, calcula seus retornos esperados e riscos, e identifica o portfólio com o maior **Índice de Sharpe**.

    ## Fundamentos Teóricos

    ### Teoria Moderna do Portfólio (TMP)

    Desenvolvida por Harry Markowitz em 1952, a TMP é uma estrutura matemática para montar um portfólio de ativos de forma que o retorno esperado seja maximizado para um determinado nível de risco.

    #### Conceitos Chave

    - **Retorno Esperado (\( E(R) \))**: O valor médio da distribuição de probabilidade dos retornos possíveis.
    - **Risco (\( \sigma \))**: O desvio padrão dos retornos, representando a volatilidade.
    - **Covariância (\( \sigma_{ij} \))**: Mede como dois ativos se movem juntos.
    - **Coeficiente de Correlação (\( \rho_{ij} \))**: Medida padronizada da covariância.

    #### Fórmulas

    - **Retorno Esperado do Portfólio**:
      $$
      E(R_p) = \sum_{i=1}^{n} w_i \cdot E(R_i)
      $$

    - **Variância do Portfólio**:
      $$
      \sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i \cdot w_j \cdot \sigma_{ij}
      $$

    - **Desvio Padrão do Portfólio (Risco)**:
      $$
      \sigma_p = \sqrt{\sigma_p^2}
      $$

    - **Índice de Sharpe**:
      $$
      S_p = \frac{E(R_p) - R_f}{\sigma_p}
      $$

      Onde:
      - \( E(R_p) \): Retorno esperado do portfólio.
      - \( w_i \): Peso do ativo \( i \) no portfólio.
      - \( E(R_i) \): Retorno esperado do ativo \( i \).
      - \( \sigma_{ij} \): Covariância entre os ativos \( i \) e \( j \).
      - \( R_f \): Taxa livre de risco (assumida como zero para simplificação).
      - \( \sigma_p \): Desvio padrão do portfólio.

    ### Fronteira Eficiente

    A Fronteira Eficiente representa portfólios que oferecem o maior retorno esperado para um determinado nível de risco. Portfólios abaixo da fronteira são subótimos.

    ## Como Começar

    Insira os códigos das ações (tickers) nas quais você está interessado e o valor total que deseja investir. O **Portfólio Ótimo** gerará um portfólio otimizado para você.

    ---
    """)

    st.sidebar.header("Seleção de Portfólio")
    st.sidebar.write("Insira os códigos das ações (separados por vírgula):")
    tickers_input = st.sidebar.text_input("Tickers", "PETR4.SA, VALE3.SA, ITUB4.SA, ABEV3.SA")
    amount = st.sidebar.number_input("Valor Total do Investimento (R$):", min_value=0.0, value=10000.0, step=1000.0)

    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

    if st.sidebar.button("Otimizar Portfólio"):
        data = yf.download(tickers, period='1y')['Adj Close']

        if data.isnull().values.any():
            st.error("Não foi possível recuperar dados de algumas ações. Verifique os códigos dos tickers e tente novamente.")
            return

        # Calcular retornos diários
        returns = data.pct_change().dropna()

        # Calcular retornos esperados e matriz de covariância
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        num_portfolios = 50000
        results = np.zeros((3, num_portfolios))
        weight_array = []

        for i in range(num_portfolios):
            weights = np.random.dirichlet(np.ones(len(tickers)), size=1)
            weights = weights.flatten()
            weight_array.append(weights)

            portfolio_return = np.sum(mean_returns * weights) * 252
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            sharpe_ratio = portfolio_return / portfolio_std_dev

            results[0,i] = portfolio_return * 100  # Converter para porcentagem
            results[1,i] = portfolio_std_dev * 100  # Converter para porcentagem
            results[2,i] = sharpe_ratio

        max_sharpe_idx = np.argmax(results[2])
        max_sharpe_return = results[0, max_sharpe_idx]
        max_sharpe_std_dev = results[1, max_sharpe_idx]
        optimal_weights = weight_array[max_sharpe_idx]

        st.subheader("Alocação Otimizada do Portfólio")
        allocation = pd.DataFrame({'Ticker': tickers, 'Peso': optimal_weights})
        allocation['Valor Investido (R$)'] = allocation['Peso'] * amount
        st.table(allocation[['Ticker', 'Peso', 'Valor Investido (R$)']])

        st.subheader("Retorno Esperado Anual e Risco")
        st.write(f"Retorno Anual Esperado: **{max_sharpe_return:.2f}%**")
        st.write(f"Volatilidade Anual (Risco): **{max_sharpe_std_dev:.2f}%**")
        st.write(f"Índice de Sharpe: **{results[2, max_sharpe_idx]:.2f}**")

        # Plotar a Fronteira Eficiente
        st.subheader("Fronteira Eficiente")
        plt.figure(figsize=(10,6))
        plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.3)
        plt.scatter(max_sharpe_std_dev, max_sharpe_return, c='red', marker='*', s=500)
        plt.colorbar(label='Índice de Sharpe')
        plt.xlabel('Volatilidade (Desvio Padrão)')
        plt.ylabel('Retorno Esperado')
        plt.title('Fronteira Eficiente')
        st.pyplot(plt)

        st.markdown("""
        ## Interpretação

        - **Portfólio Ótimo**: A estrela vermelha no gráfico representa o portfólio com o maior Índice de Sharpe.
        - **Fronteira Eficiente**: O gráfico de dispersão mostra todos os portfólios simulados. Portfólios na fronteira superior esquerda são mais eficientes.
        - **Risco vs. Retorno**: Os investidores podem usar essas informações para selecionar um portfólio que corresponda à sua tolerância ao risco.

        ## Próximos Passos

        - **Ajuste Seu Portfólio**: Tente adicionar ou remover ações para ver como isso afeta a alocação ótima.
        - **Análise Adicional**: Considere outros fatores como condições de mercado, análise individual de ações e seus objetivos de investimento.

        **Aviso**: Esta ferramenta fornece uma alocação teórica otimizada com base em dados históricos. Desempenhos passados não são indicativos de resultados futuros. Sempre conduza pesquisas aprofundadas ou consulte um consultor financeiro antes de tomar decisões de investimento.
        """)

if __name__ == "__main__":
    main()
