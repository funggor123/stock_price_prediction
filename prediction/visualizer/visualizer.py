from prediction.cron import cron_stock
import plotly.graph_objects as go

# visualize a stock preprocess through graph
# input: stock_data     pandas preprocess frame object
def visualize(stock_data):
    visualize_close_price(stock_data)


def visualize_close_price(stock_data):
    data = go.Scatter(
        x=stock_data.index.to_series(),
        y=stock_data['Open'].values,
    )

    layout = go.Layout(dict(title="Closing prices of the stock",
                            xaxis=dict(title='Month'),
                            yaxis=dict(title='Price'),
                            ), legend=dict(
        orientation="h"))
    fig = go.Figure(dict(data=data, layout=layout))
    fig.show()


visualize(cron_stock.fetch_stock())
