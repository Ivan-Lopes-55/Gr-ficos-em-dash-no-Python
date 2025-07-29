import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from dash import Dash, dcc, html
import statsmodels.api as sm

df = pd.read_csv("ecommerce_estatistica.csv")

numeric_df = df.select_dtypes(include='number')

#Histograma
hist_fig = px.histogram(df, x="Preço", title="Histograma de Preço")

#Gráfico de dispersão: Nota vs Preço
scatter_fig = px.scatter(df, x="Nota", y="Preço", title="Dispersão: Nota vs Preço")

#Mapa de calor de correlação
heatmap_fig = px.imshow(numeric_df.corr(), text_auto=True, title="Mapa de Calor (Correlação)")

#Gráfico de barras: Preço por Marca
bar_fig = px.bar(df, x="Marca", y="Preço", title="Preço por Marca", color="Marca")

#Gráfico de pizza: Distribuição por Gênero
pie_data = df["Gênero"].value_counts()
pie_fig = px.pie(names=pie_data.index, values=pie_data.values, title="Distribuição por Gênero")

#Gráfico de densidade: Preço
density_fig = ff.create_distplot([df["Preço"].dropna()], ["Preço"], show_hist=False)

#Gráfico de regressão: Nota vs Preço
X = df["Nota"]
Y = df["Preço"]
X_const = sm.add_constant(X)
model = sm.OLS(Y, X_const).fit()
df["regression_line"] = model.predict(X_const)

regression_fig = go.Figure()
regression_fig.add_trace(go.Scatter(x=df["Nota"], y=df["Preço"], mode='markers', name='Dados'))
regression_fig.add_trace(go.Scatter(x=df["Nota"], y=df["regression_line"], mode='lines', name='Regressão'))
regression_fig.update_layout(title="Regressão Linear: Nota vs Preço")

# Criar a aplicação Dash
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Dashboard de Estatísticas de E-commerce"),
    dcc.Graph(figure=hist_fig),
    dcc.Graph(figure=scatter_fig),
    dcc.Graph(figure=heatmap_fig),
    dcc.Graph(figure=bar_fig),
    dcc.Graph(figure=pie_fig),
    dcc.Graph(figure=density_fig),
    dcc.Graph(figure=regression_fig)
])

# Executar o servidor (versão atualizada)
if __name__ == "__main__":
    app.run(debug=True)
