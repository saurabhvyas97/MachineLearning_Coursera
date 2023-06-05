import numpy as np
import time
import matplotlib.pyplot as plt
import csv
import pandas as pd

##############Funcions##################
def pred_func(w,b,x):
    y_pred = np.dot(w,x) + b
    return y_pred

def cost_func(y, y_pred, N):
    cost = np.sum((y - y_pred)**2)/(2*N)
    return cost

def get_cost_plot(x_data, y_data, N, cost_history, w_history, b_history):
    cost_mesh = np.array([])
    w = np.linspace(np.min(w_history)-5,np.max(w_history)+5,10)
    b = np.linspace(np.min(b_history)-5,np.max(b_history)+5,10)

    for i in b:
        for j in w:
            y_pred = pred_func(j, i, x_data)
            cost = cost_func(y_data, y_pred, N)
            cost_mesh = np.append(cost_mesh,cost)

    cost_mesh = np.reshape(cost_mesh,[len(b),len(w)])

    fig1 = plt.figure()
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(w,b)
    ax.plot_wireframe(X,Y,cost_mesh,color="black",linewidth=1)
    ax.plot3D(w_history,b_history,cost_history,color="red",linewidth=2)
    ax.set_xlabel("Slope")
    ax.set_ylabel("Offset")
    ax.set_zlabel("Cost")
    ax.set_title("Gradient Descent Cross-plot")

    fig2 = plt.figure()
    plt.plot(cost_history)
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.title('Cost history')
    plt.grid(True)
    
def gradient_desc(w_old, b_old, alpha, N, x_data, y_data, n):
    w_new = np.array([])
    for i in range(0,n):
        w = w_old[i] - (alpha/N)*(np.dot(x_data[i], (pred_func(w_old, b_old, x_data) - y_data)))
        w_new = np.append(w_new, w)
    b_new = b_old - (alpha/N)*(np.sum(pred_func(w_old, b_old, x_data) - y_data))
    return w_new, b_new

def run_gradient_desc(x_data, y_data, N, alpha, w, b, n):
    cost_history = np.array([])
    w_history = np.array([])
    b_history = np.array([])
    iteration = 0
    cost = cost_func(y_data, pred_func(w,b,x_data), N)

    while cost > 0.0001 and iteration<3000:
        w_history = np.append(w_history, w)
        b_history = np.append(b_history, b)
        cost_history = np.append(cost_history,cost)
        w, b = gradient_desc(w, b, alpha, N, x_data, y_data, n)
        cost = cost_func(y_data, pred_func(w,b,x_data), N)
        iteration = iteration+1
    return cost_history, w_history, b_history

def plot_grad_desc_result(x_data, y_data, w, b):
    fig3 = plt.figure()
    plt.plot(x_data,y_data,'r.')
    x_all = np.linspace(np.min(x_data),np.max(x_data),1000)
    y_all = pred_func(w,b,x_all)
    plt.plot(x_all,y_all)
    plt.title("Linear regression result")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend("Raw data","Linear approximation")
    plt.grid(True)

#############MainProg##################
def main():

    ##DataGeneration##
    df = pd.read_csv('Fish.csv')
    x_data_raw_1 = df['Length1'].to_numpy()
    x_data_raw_2 = df['Length2'].to_numpy()
    x_data_raw_3 = df['Length3'].to_numpy()
    x_data_raw_4 = df['Height'].to_numpy()
    x_data_raw_5 = df['Width'].to_numpy()
    x_data_raw = np.array([x_data_raw_1, x_data_raw_2, x_data_raw_3, x_data_raw_4, x_data_raw_5])
    x_data = np.array([x_data_raw_1, x_data_raw_2, x_data_raw_3, x_data_raw_4, x_data_raw_5])
    y_data = df['Length1'].to_numpy()
    n = len(x_data) #No. of features
    for i in range(0,n):
        x_data[i,:] = x_data[i,:]/np.max(x_data[i,:])
    N = len(y_data)
    
    ##ParametersForAlgo##
    alpha = 0.1 #learning rate
    w = np.array([1,1,1,1,1])
    b = 3
    
    ##StartOfAlgo##
    tic = time.perf_counter()
    cost_history, w_history, b_history = run_gradient_desc(x_data, y_data, N, alpha, w, b, n)
    toc = time.perf_counter()
    plt.plot(cost_history)
    
    w_last = w_history[-5:]
    w = np.array([])
    for i in range(0,n):
        w_temp = w_last[i]/np.max(x_data_raw[i,:]) #rescaling
        w = np.append(w, w_temp)
    print(w)
    print(b_history[-1])
    print(cost_history[-1])
    
    fig2 = plt.figure()
    plt.plot(x_data_raw_1, y_data,'r.')
    plt.plot(x_data_raw_1, pred_func(w, b_history[-1], x_data_raw))
    
    fig3 = plt.figure()
    plt.plot(x_data_raw_3, y_data,'r.')
    plt.plot(x_data_raw_3, pred_func(w, b_history[-1], x_data_raw))
    '''
    w_history = w_history/np.max(x_data_raw) #rescaling

    get_cost_plot(x_data_raw, y_data, N, cost_history, w_history, b_history)
    plot_grad_desc_result(x_data_raw, y_data, w_history[-1], b_history[-1])
    
    print('Time elasped =',(toc-tic),'seconds.')
    print('Param w = ',w_history[-1])
    print('Param b = ',b_history[-1])
    print('Cost = ', cost_history[-1])
    '''
    plt.show()
    
##Running Main##
if __name__ == "__main__":
    main()