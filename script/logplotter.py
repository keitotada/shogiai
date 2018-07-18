import argparse, re, matplotlib.pyplot as plt, os, json, numpy as np

def plotter(filename):
    if os.path.exists(filename):
        with open(filename) as tmp_f:
            data_log = json.load(tmp_f)
            date_train = data_log.key().sort()
            result_train = {'loss': [], 'accuracy': []}
            for i in date_train:
                result_train['loss'].append(data_log[i]['loss'])
                result_train['acuracy'].append(data_log[i]['accuracy'])

            plt.hold(True)
            x = np.linspace(0, len(data_train), 1)
            plt.plot(x, result_train['loss'], 'loss')
            plt.plot(x, result_train['accuracy'], 'loss')
            plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('log', type=str)

plotter(parser.parse_args().log)



