import argparse, matplotlib.pyplot as plt, os, json


def plotter(filename):
    if os.path.exists(filename):
        with open(filename) as tmp_f:
            data_log = json.load(tmp_f)
            print(data_log)
            ite_train = {}
            for i in data_log.values():
                print(i)
                ite_train[i['iteration']] = (int(i['loss']), int(i['accuracy']))

            print(ite_train)
            ite = sorted(ite_train.items())
            print(ite)
            result_train = [[], [], []]
            for i in ite:
                result_train[0].append(i[0])
                result_train[1].append(i[1][0])
                result_train[2].append(i[1][1])

            print(result_train)
            x = result_train[0]
            plt.plot(x, result_train[1], label='loss')
            plt.plot(x, result_train[2], label='accuracy')
            plt.legend()
            plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('log', type=str)

plotter(parser.parse_args().log)



