import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def plot_prediction(predictions, measurements, time_point, method, mode, sector):

    #matplotlib.rcParams['timezone'] = 'Germany/Berlin'
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(measurements, c="forestgreen", label="measured")
    ax.plot(predictions, c="red", label="predicted")
    #ax.set_ylim(0, 90000)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Load, kW', fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    # Axis formatter
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
    # placement for ticks
    plt.tick_params(axis='y', direction='out', labelsize=8)
    plt.tick_params(axis='x', direction='out', labelsize=8, rotation=45)
    # plt.legend(fontsize=8)
    fig.tight_layout()
    date_for_name = time_point.strftime("%Y-%m-%d_%H-%M")
    plt.savefig("./results/"+method+"/"+mode+"/"+sector+"/figures/"+method+mode+sector+date_for_name+".jpg", dpi=300)
    plt.close()