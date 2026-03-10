import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

def make_graph(x_data, y_data):
    df = pd.read_csv('healthy-train_results.csv')
    x_datas = df[x_data]
    y_datas = df[y_data]
    plt.plot(x_datas, y_datas, marker='x', linestyle='') # Create the line plot with markers

    df = pd.read_csv('unhealthy-train_results.csv')
    x_datas = df[x_data]
    y_datas = df[y_data]
    plt.plot(x_datas, y_datas, marker='o', linestyle='') # Create the line plot with markers

    plt.xlabel(x_data)
    plt.ylabel(y_data)
    plt.title('x = Healthy, O = Unhealthy: ' + x_data + ' vs ' + y_data)

    plt.grid(True) # Optional: add a grid
    plt.show()

make_graph('fractal_dimension', 'lacunarity')
make_graph('fractal_dimension', 'succolarity')
make_graph('lacunarity', 'succolarity')