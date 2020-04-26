import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## ref: https://medium.com/engineer-quant/alphaai-using-machine-learning-to-predict-stocks-79c620f87e53
def wavelets_transform():
    print("undefine")


## ref: https://github.com/borisbanushev/stockpredictionai
def fourier_transform(stock_data):
    close_fft = np.fft.fft(np.asarray(stock_data['Close'].values.tolist()))
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))


    #plt.figure(figsize=(14, 7), dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3, 6, 9]:
        fft_list_m10 = np.copy(fft_list)
        fft_list_m10[num_:-num_] = 0
        #plt.plot(np.fft.ifft(fft_list_m10).real, label='Fourier transform with {} components'.format(num_))
        stock_data['FT_' + str(num_)] = np.fft.ifft(fft_list_m10).real

    '''
    plt.plot(stock_data['Close'].values, label='Real')
    plt.xlabel('Days')
    plt.ylabel('USD')
    plt.title('Figure 3: Goldman Sachs (close) stock prices & Fourier transforms')
    plt.legend()
    plt.show()
    '''

