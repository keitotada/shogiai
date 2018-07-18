import chainer
from chainer import cuda, Variable, optimizers, serializers
import chainer.functions as func

import numpy as np, argparse, random, pickle, os, re
from datetime import datetime

import policynet


parser = argparse.ArgumentParser()
parser.add_argument('kifu_train', type=str)
parser.add_argument('kifu_test', type=str)
parser.add_argument('--train_batchsize', '-b', type=int, default=32)
parser.add_argument('--test_batchsize', '-tb', type=int, default=512)
parser.add_argument('--epoch', '-e', type=int, default=1)
parser.add_argument('--model', '-m', type=str, default='model/model_policy')
parser.add_argument('--state_opt', '-s', type=str, default='model/state_policy')
parser.add_argument('--init_model', '-im', default='')
parser.add_argument('--resume_opt', '-r', default='')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--eval_interval', '-ei', type=int, default=1000)
parser.add_argument('--log', '-l', default='log_train.json')
args = parser.parse_args()

time_start = datetime.now()
time_now = datetime.now()
name_log = re.sub('\.', time_start.strftime('%Y/%m/%d/%H/%M/%S') + '.', args.log)
log = {'epoch': 0, 'iteration': 0, 'loss': 0, 'accuracy': 0}
with open(name_log, 'a+') as log_f:
    log_f.writelines('{\n\t')

model = policynet.Policynet()
model.to_gpu()

optimizer = optimizers.SGD(lr=args.learning_rate)
optimizer.setup(model)



if args.init_model:
    serializers.load_npz(args.init_model, model)
    print('Loaded model from {}'.format(args.init_model))

if args.resume:
    serializers.load_npz(args.resume, optimizer)
    print('Loaded optimizer state from {}'.format(args.resume_opt))


name_train = re.sub('\.*?$', '.pickle', args.kifu_train)
if os.path.exists(name_train):
    with open(name_train, 'rb') as tmp_f:
        positions_train = pickle.load(tmp_f)

    print('Loaded train pickle')

else:
    #read kifu
    with open(name_train, 'wb') as tmp_f:
        #pickle.dump(positions_train, tmp_f, pickle.HIGHEST_PROTOCOL)
        print('Saved train pickle')

name_test = re.sub('\.*?$', '.pickle', args.kifu_test)
if os.path.exists(name_test):
    with open(name_test) as tmp_f:
        positions_test = pickle.load(tmp_f)
    print('Loaded test pickle')

else:
    #read kifu
    with open(name_test) as tmp_f:
        #pickle.dump(positions_test, tmp_f, pickle.HIGHEST_PROTOCOL)
        print('Saved test pickle')


'''def mini_batch(positions, i, batchsize):
    mini_batch_data = []
    mini_batch_move = []
    for b in range(batchsize):
        features, move, win = make_features(positions[i + b])
        mini_batch_data.append(features)
        mini_batch_move.append(move)

    return (Variable(cuda.to_gpu(np.array(mini_batch_data, dtype=np.float32))),
            Variable(cuda.to_gpu(np.array(mini_batch_move, dtype=np.int32))))

def mini_batch_for_test(positions, batchsize):
    mini_batch_data = []
    mini_batch_move = []
    for b in range(batchsize):
        features, move, win = make_features(random.choice(positions))
        mini_batch_data.append(features)
        mini_batch_move.append(move)

    return (Variable(cuda.to_gpu(np.array(mini_batch_data, dtype=np.float32))),
            Variable(cuda.to_gpu(np.array(mini_batch_move, dtype=np.int32))))
'''
print('Training starts')
ite = 0
sum_loss = 0
for i in range(args.epoch):
    positions_train = random.sample(positions_train, len(positions_train))

    ite_epoch = 0
    sum_loss_epoch = 0
    for j in range(0, len(positions_train) - args.train_batchsize, args.train_batchsize):
        x = 0#mini_batch(positions_train, j, args.train_batchsize)
        y = model(x)

        model.cleargrads()
        loss = func.softmax_cross_entropy(y, x)
        loss.backward()
        optimizer.update()

        ite += 1
        sum_loss += loss.data
        ite_epoch += 1
        sum_loss_epoch += loss.data

        if optimizer.t % args.eval_interval == 0:
            x= 0#mini_batch_for_test(positions_test, args.test_batchsize)
            y = model(x)

            log['epoch'] = optimizer.epoch + 1
            log['iteration'] = optimizer.t
            log['loss'] = sum_loss / ite
            log['accuracy'] = func.accuracy(y, x).data
            time_now = datetime.now()
            with open(name_log, 'a+') as log_f:
                log_f.writelines(time_now.strftime('"%Y_%m_%d_%H_%M_%S": ') + log)


            print(time_now.strftime('%Y/%m/&d %H/%M/&S    ') + 'epoch: {}, iter: {}, loss: {}, accuracy: {}'.format(
                log['epoch'], log['iteration'],
                log['loss'], log['accuracy']))
            ite = 0
            sum_loss = 0

    print('Check against test data')
    ite_test, sum_test_acc = 0
    for j in range(0, len(positions_test) - args.train_batchsize, args.train_batchsize):
        x = 0#mini_batch(positions_test, j, args.train_batchsize)
        y = model(x)
        ite_test += 1
        sum_test_acc += func.accuracy(y, x).data

    log['epoch'] = optimizer.epoch + 1
    log['iteration'] = optimizer.t
    log['loss'] = sum_loss_epoch / ite_epoch
    log['accuracy'] = sum_test_acc / ite_test
    time_now = datetime.now()
    with open(name_log, 'a+') as log_f:
         log_f.writelines(time_now.strftime('"%Y_%m_%d_%H_%M_%S": ') + log)


    print(time_now.strftime('%Y/%m/&d %H/%M/&S    ') + 'epoch: {}, iter: {}, loss: {}, accuracy: {}'.format(log['epoch'], log['iteration'],log['loss'], log['accuracy']))

    optimizer.new_epoch()

serializers.save_npz(args.model, model)
print('Saved model')
serializers.save_npz(args.state, optimizer)
print('Saved optimizer state')

with open(name_log, 'a+') as log_f:
    log_f.writelines('\n}')

print('Finished!!')






