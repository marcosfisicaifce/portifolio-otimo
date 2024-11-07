# portfolio_otimo.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

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

    Utilize o campo abaixo para pesquisar e adicionar ações ao seu portfólio. Você pode selecionar múltiplas ações usando o campo de seleção com pesquisa.

    ---
    """)

    # Carregar a lista de ações
    @st.cache_data
    def carregar_lista_acoes():
        df_acoes = pd.read_csv('lista_acoes.csv')
        return df_acoes

    df_acoes = carregar_lista_acoes()
    df_acoes['display'] = df_acoes['nome'] + ' (' + df_acoes['ticker'] + ')'

    st.sidebar.header("Seleção de Portfólio")
    st.sidebar.write("Pesquise e selecione as ações para o seu portfólio:")

    # Campo de seleção múltipla com pesquisa
    selected_options = st.sidebar.multiselect(
        "Selecione as ações:",
        options=df_acoes['display'].tolist()
    )

    # Obter tickers das opções selecionadas
    tickers = [option.split('(')[-1].strip(')') for option in selected_options]

    amount = st.sidebar.number_input("Valor Total do Investimento (R$):", min_value=0.0, value=10000.0, step=1000.0)

    if st.sidebar.button("Otimizar Portfólio"):
        if tickers:
            try:
                data = yf.download(tickers, period='1y')['Adj Close']
            except Exception as e:
                st.error(f"Erro ao baixar dados das ações: {e}")
                return

            if data.isnull().values.any():
                missing_tickers = data.columns[data.isnull().any()].tolist()
                st.error(f"Não foi possível recuperar dados das seguintes ações: {', '.join(missing_tickers)}. Verifique os códigos e tente novamente.")
                return

            # Calcular retornos diários
            returns = data.pct_change().dropna()

            # Calcular retornos esperados e matriz de covariância
            mean_returns = returns.mean()
            cov_matrix = returns.cov()

            num_portfolios = 50000
            results = np.zeros((4 + len(tickers), num_portfolios))
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
                results[3,i] = i  # Índice do portfólio

                # Armazenar pesos
                for j in range(len(weights)):
                    results[4 + j, i] = weights[j]

            max_sharpe_idx = np.argmax(results[2])
            max_sharpe_return = results[0, max_sharpe_idx]
            max_sharpe_std_dev = results[1, max_sharpe_idx]
            optimal_weights = weight_array[max_sharpe_idx]

            # Criar DataFrame com os resultados
            columns = ['Retorno', 'Risco', 'Sharpe', 'Índice'] + tickers
            portfolios = pd.DataFrame(results.T, columns=columns)

            st.subheader("Alocação Otimizada do Portfólio")
            allocation = pd.DataFrame({'Ticker': tickers, 'Peso': optimal_weights})

            # Obter nomes das empresas para exibição
            nomes_selecionados = []
            for ticker in tickers:
                nome_empresa = df_acoes.loc[df_acoes['ticker'] == ticker, 'nome'].values[0]
                nomes_selecionados.append(nome_empresa)

            allocation['Empresa'] = nomes_selecionados

            # Obter preços atuais das ações
            try:
                current_prices = yf.download(tickers, period='1d')['Adj Close'].iloc[0]
            except Exception as e:
                st.error(f"Erro ao obter preços atuais das ações: {e}")
                return

            allocation['Preço Atual (R$)'] = allocation['Ticker'].apply(lambda x: current_prices[x])

            # Calcular valor inicial alocado a cada ação
            allocation['Valor Alocado Inicial (R$)'] = allocation['Peso'] * amount

            # Calcular quantidade inteira de ações que pode ser comprada
            allocation['Quantidade de Ações'] = (allocation['Valor Alocado Inicial (R$)'] / allocation['Preço Atual (R$)']).astype(int)

            # Recalcular o valor investido com base na quantidade inteira de ações
            allocation['Valor Investido (R$)'] = allocation['Quantidade de Ações'] * allocation['Preço Atual (R$)']

            # Calcular o peso real após ajuste
            total_invested = allocation['Valor Investido (R$)'].sum()
            allocation['Peso Ajustado'] = allocation['Valor Investido (R$)'] / total_invested

            # Reordenar colunas para exibição
            allocation = allocation[['Ticker', 'Empresa', 'Quantidade de Ações', 'Preço Atual (R$)', 'Valor Investido (R$)', 'Peso Ajustado']]

            st.table(allocation)

            st.subheader("Retorno Esperado Anual e Risco (Ajustados)")
            # Recalcular o retorno e risco com os pesos ajustados
            adjusted_weights = allocation['Peso Ajustado'].values

            adjusted_return = np.sum(mean_returns * adjusted_weights) * 252 * 100  # Converter para porcentagem
            adjusted_std_dev = np.sqrt(np.dot(adjusted_weights.T, np.dot(cov_matrix * 252, adjusted_weights))) * 100  # Converter para porcentagem
            adjusted_sharpe_ratio = adjusted_return / adjusted_std_dev

            st.write(f"Retorno Anual Esperado Ajustado: **{adjusted_return:.2f}%**")
            st.write(f"Volatilidade Anual (Risco) Ajustada: **{adjusted_std_dev:.2f}%**")
            st.write(f"Índice de Sharpe Ajustado: **{adjusted_sharpe_ratio:.2f}**")

            # Plotar a Fronteira Eficiente Interativa
            st.subheader("Fronteira Eficiente Interativa")

            # Dados para o gráfico
            portfolios['Retorno'] = portfolios['Retorno']
            portfolios['Risco'] = portfolios['Risco']
            portfolios['Índice de Sharpe'] = portfolios['Sharpe']

            # Preparar informações para hover
            hover_data = {
                'Retorno': ':.2f',
                'Risco': ':.2f',
                'Índice de Sharpe': ':.2f',
            }

            for ticker in tickers:
                portfolios[ticker] = portfolios[ticker]
                hover_data[ticker] = ':.2f'

            # Criar gráfico interativo com Plotly
            fig = px.scatter(
                portfolios,
                x='Risco',
                y='Retorno',
                color='Índice de Sharpe',
                hover_data=hover_data,
                title='Fronteira Eficiente',
                template='plotly_white'
            )

            # Adicionar o portfólio ótimo ajustado
            fig.add_scatter(
                x=[adjusted_std_dev],
                y=[adjusted_return],
                mode='markers',
                marker=dict(color='red', size=15, symbol='star'),
                name='Portfólio Ótimo Ajustado'
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            ## Interpretação

            - **Portfólio Ótimo Ajustado**: A estrela vermelha no gráfico representa o portfólio ajustado com o maior Índice de Sharpe considerando a compra de quantidades inteiras de ações.
            - **Fronteira Eficiente**: O gráfico de dispersão mostra todos os portfólios simulados. Passe o mouse sobre os pontos para ver os detalhes de cada portfólio.
            - **Risco vs. Retorno**: Os investidores podem usar essas informações para selecionar um portfólio que corresponda à sua tolerância ao risco.

            ## Próximos Passos

            - **Ajuste Seu Portfólio**: Tente adicionar ou remover ações para ver como isso afeta a alocação ótima.
            - **Análise Adicional**: Considere outros fatores como condições de mercado, análise individual de ações e seus objetivos de investimento.

            **Aviso**: Esta ferramenta fornece uma alocação teórica otimizada com base em dados históricos. Desempenhos passados não são indicativos de resultados futuros. Sempre conduza pesquisas aprofundadas ou consulte um consultor financeiro antes de tomar decisões de investimento.
            """)
        else:
            st.error("Por favor, selecione pelo menos uma ação para otimizar o portfólio.")

if __name__ == "__main__":
    main()
