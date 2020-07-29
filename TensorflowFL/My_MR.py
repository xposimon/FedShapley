from __future__ import absolute_import, division, print_function
import tensorflow_federated as tff
import tensorflow.compat.v1 as tf
import numpy as np
import time
from scipy.special import comb, perm
from itertools import permutations

import os

# tf.compat.v1.enable_v2_behavior()
# tf.compat.v1.enable_eager_execution()

# NUM_EXAMPLES_PER_USER = 1000
BATCH_SIZE = 100
NUM_AGENT = 5
DECAY_FACTOR = 0.8


def get_data_for_digit(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    for i in range(0, len(all_samples), BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        output_sequence.append({
            'x': np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                          dtype=np.float32),
            'y': np.array([source[1][i] for i in batch_samples], dtype=np.int32)})
    return output_sequence


def get_data_for_digit_test(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    for i in range(0, len(all_samples)):
        output_sequence.append({
            'x': np.array(source[0][all_samples[i]].flatten() / 255.0,
                          dtype=np.float32),
            'y': np.array(source[1][all_samples[i]], dtype=np.int32)})
    return output_sequence

def checkRange(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0

        if x[i] > 1:
            x[i] = 1
    return x

def get_data_for_federated_agents(source, num, weight=False, noiseX=False, noiseY=False):
    
    if weight:
        # default 1:2:3:4:5
        weights = [(i+1) for i in range(NUM_AGENT)]
        PIECE = int(5421/sum(weights))
        left=sum(weights[:num])
        print((left,left+num+1))
        output_sequence = []
        Samples = []

        for digit in range(0, 10):
            samples = [i for i, d in enumerate(source[1]) if d == digit]
            samples = samples[0:5421]
            Samples.append(samples)

        all_samples = []
        for sample in Samples:
            for sample_index in range(left*PIECE,(left+num+1)*PIECE):
                if sample_index >= len(sample):
                    break
                all_samples.append(sample[sample_index])

        # all_samples = [i for i in range(int(num*(len(source[1])/NUM_AGENT)), int((num+1)*(len(source[1])/NUM_AGENT)))]

        for i in range(0, len(all_samples), BATCH_SIZE):
            batch_samples = all_samples[i:i + BATCH_SIZE]
            output_sequence.append({
                'x': np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                            dtype=np.float32),
                'y': np.array([source[1][i] for i in batch_samples], dtype=np.int32)})
    else:
        output_sequence = []

        Samples = []
        for digit in range(0, 10):
            samples = [i for i, d in enumerate(source[1]) if d == digit]
            samples = samples[0:5421]
            Samples.append(samples)

        all_samples = []
        for sample in Samples:
            for sample_index in range(int(num * (len(sample) / NUM_AGENT)), int((num + 1) * (len(sample) / NUM_AGENT))):
                all_samples.append(sample[sample_index])

        # all_samples = [i for i in range(int(num*(len(source[1])/NUM_AGENT)), int((num+1)*(len(source[1])/NUM_AGENT)))]

        for i in range(0, len(all_samples), BATCH_SIZE):
            batch_samples = all_samples[i:i + BATCH_SIZE]
            output_sequence.append({
                'x': np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                            dtype=np.float32),
                'y': np.array([source[1][i] for i in batch_samples], dtype=np.int32)})

    if noiseX:
        # add noise 3:0.3x, 4:0.4x
        ratio = 0
        if num > 2:
            ratio = num * 0.1
        sum_agent = int(len(all_samples))
        index = 0
        for i in range(0, sum_agent):
            noiseHere = ratio * np.random.randn(28*28)
            output_sequence[int(i/BATCH_SIZE)]['x'][i % BATCH_SIZE] = checkRange(np.add(
                output_sequence[int(i/BATCH_SIZE)]['x'][i % BATCH_SIZE], noiseHere))
        
    if noiseY:
        rand_index = []
        rand_label = []
        with open(os.path.join(os.path.dirname(__file__), "random_index.txt"), "r") as randomIndex:
            lines = randomIndex.readlines()
        for line in lines:
            rand_index.append(eval(line))
        with open(os.path.join(os.path.dirname(__file__), "random_label.txt"), "r") as randomLabel:
            lines = randomLabel.readlines()
            for line in lines:
                rand_label.append(eval(line))

        noiseList = rand_index[num][0:int(ratio*sum_agent)]
        noiseLabel = rand_label[num][0:int(ratio*sum_agent)]
        # noiseList = random.sample(range(0, sum_agent), int(ratio*sum_agent))
        # noiseLabel = []
        index = 0
        for i in noiseList:
            # noiseHere = random.randint(1, 9)
            # noiseLabel.append(noiseHere)
            noiseHere = noiseLabel[index]
            index = index + 1
            output_sequence[int(i/BATCH_SIZE)]['y'][i % BATCH_SIZE] = (
                output_sequence[int(i/BATCH_SIZE)]['y'][i % BATCH_SIZE]+noiseHere) % 10
   
    return output_sequence    
        
BATCH_TYPE = tff.NamedTupleType([
    ('x', tff.TensorType(tf.float32, [None, 784])),
    ('y', tff.TensorType(tf.int32, [None]))])

MODEL_TYPE = tff.NamedTupleType([
    ('weights', tff.TensorType(tf.float32, [784, 10])),
    ('bias', tff.TensorType(tf.float32, [10]))])


@tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
def batch_loss(model, batch):
    predicted_y = tf.nn.softmax(tf.matmul(batch.x, model.weights) + model.bias)
    return -tf.reduce_mean(tf.reduce_sum(
        tf.one_hot(batch.y, 10) * tf.log(predicted_y), axis=[1]))


@tff.tf_computation(MODEL_TYPE, BATCH_TYPE, tf.float32)
def batch_train(initial_model, batch, learning_rate):
    # Define a group of model variables and set them to `initial_model`.
    model_vars = tff.utils.create_variables('v', MODEL_TYPE)
    init_model = tff.utils.assign(model_vars, initial_model)

    # Perform one step of gradient descent using loss from `batch_loss`.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    with tf.control_dependencies([init_model]):
        train_model = optimizer.minimize(batch_loss(model_vars, batch))

    # Return the model vars after performing this gradient descent step.
    with tf.control_dependencies([train_model]):
        return tff.utils.identity(model_vars)


LOCAL_DATA_TYPE = tff.SequenceType(BATCH_TYPE)


@tff.federated_computation(MODEL_TYPE, tf.float32, LOCAL_DATA_TYPE)
def local_train(initial_model, learning_rate, all_batches):
    # Mapping function to apply to each batch.
    @tff.federated_computation(MODEL_TYPE, BATCH_TYPE)
    def batch_fn(model, batch):
        return batch_train(model, batch, learning_rate)

    l = tff.sequence_reduce(all_batches, initial_model, batch_fn)
    return l


@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_eval(model, all_batches):
    #
    return tff.sequence_sum(
        tff.sequence_map(
            tff.federated_computation(lambda b: batch_loss(model, b), BATCH_TYPE),
            all_batches))


SERVER_MODEL_TYPE = tff.FederatedType(MODEL_TYPE, tff.SERVER, all_equal=True)
CLIENT_DATA_TYPE = tff.FederatedType(LOCAL_DATA_TYPE, tff.CLIENTS)


@tff.federated_computation(SERVER_MODEL_TYPE, CLIENT_DATA_TYPE)
def federated_eval(model, data):
    return tff.federated_mean(
        tff.federated_map(local_eval, [tff.federated_broadcast(model), data]))


SERVER_FLOAT_TYPE = tff.FederatedType(tf.float32, tff.SERVER, all_equal=True)


@tff.federated_computation(
    SERVER_MODEL_TYPE, SERVER_FLOAT_TYPE, CLIENT_DATA_TYPE)
def federated_train(model, learning_rate, data):
    l = tff.federated_map(
        local_train,
        [tff.federated_broadcast(model),
         tff.federated_broadcast(learning_rate),
         data])
    return l
    # return tff.federated_mean()


def readTestImagesFromFile(distr_same):
    ret = []
    if distr_same:
        f = open(os.path.join(os.path.dirname(__file__), "test_images1_.txt"), encoding="utf-8")
    else:
        f = open(os.path.join(os.path.dirname(__file__), "test_images1_.txt"), encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        tem_ret = []
        p = line.replace("[", "").replace("]", "").replace("\n", "").split("\t")
        for i in p:
            if i != "":
                tem_ret.append(float(i))
        ret.append(tem_ret)
    return np.asarray(ret)


def readTestLabelsFromFile(distr_same):
    ret = []
    if distr_same:
        f = open(os.path.join(os.path.dirname(__file__), "test_labels_.txt"), encoding="utf-8")
    else:
        f = open(os.path.join(os.path.dirname(__file__), "test_labels_.txt"), encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        tem_ret = []
        p = line.replace("[", "").replace("]", "").replace("\n", "").split(" ")
        for i in p:
            if i != "":
                tem_ret.append(float(i))
        ret.append(tem_ret)
    return np.asarray(ret)


def getParmsAndLearningRate(agent_no):
    f = open(os.path.join(os.path.dirname(__file__), "weights_" + str(agent_no) + ".txt"))
    content = f.read()
    g_ = content.split("***\n--------------------------------------------------")
    parm_local = []
    learning_rate_list = []
    for j in range(len(g_) - 1):
        line = g_[j].split("\n")
        if j == 0:
            weights_line = line[0:784]
            learning_rate_list.append(float(line[784].replace("*", "").replace("\n", "")))
        else:
            weights_line = line[1:785]
            learning_rate_list.append(float(line[785].replace("*", "").replace("\n", "")))
        valid_weights_line = []
        for l in weights_line:
            w_list = l.split("\t")
            w_list = w_list[0:len(w_list) - 1]
            w_list = [float(i) for i in w_list]
            valid_weights_line.append(w_list)
        parm_local.append(valid_weights_line)
    f.close()

    f = open(os.path.join(os.path.dirname(__file__), "bias_" + str(agent_no) + ".txt"))
    content = f.read()
    g_ = content.split("***\n--------------------------------------------------")
    bias_local = []
    for j in range(len(g_) - 1):
        line = g_[j].split("\n")
        if j == 0:
            weights_line = line[0]
        else:
            weights_line = line[1]
        b_list = weights_line.split("\t")
        b_list = b_list[0:len(b_list) - 1]
        b_list = [float(i) for i in b_list]
        bias_local.append(b_list)
    f.close()
    ret = {
        'weights': np.asarray(parm_local),
        'bias': np.asarray(bias_local),
        'learning_rate': np.asarray(learning_rate_list)
    }
    return ret


def train_with_gradient_and_valuation(agent_list, grad, bi, lr, distr_type, g_m, datanum):
    model_g = {
        'weights': g_m[0],
        'bias': g_m[1]
    }

    gradient_w = np.zeros([784, 10], dtype=np.float32)
    gradient_b = np.zeros([10], dtype=np.float32)
    #print(agent_list, len(grad), agent_shapley, local_model_index)
    
    data_sum = 0
    for i in agent_list:
        data_sum += datanum[i]
    agents_w = [0 for _ in range(NUM_AGENT)]
    for i in agent_list:
        agents_w[i] = datanum[i] / data_sum
    #print(agents_w)
    for j in agent_list:
        gradient_w = np.add(np.multiply(grad[j], agents_w[j]), gradient_w)
        gradient_b = np.add(np.multiply(bi[j], agents_w[j]), gradient_b)

    model_g['weights'] = np.subtract(model_g['weights'], np.multiply(lr, gradient_w))
    model_g['bias'] = np.subtract(model_g['bias'], np.multiply(lr, gradient_b))

    test_images = readTestImagesFromFile(False)
    test_labels_onehot = readTestLabelsFromFile(False)
    m = np.dot(test_images, np.asarray(model_g['weights']))
    test_result = m + np.asarray(model_g['bias'])
    y = tf.nn.softmax(test_result)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(test_labels_onehot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy.numpy()


def remove_list_indexed(removed_ele, original_l, ll):
    new_original_l = []
    for i in original_l:
        new_original_l.append(i)
    for i in new_original_l:
        if i == removed_ele:
            new_original_l.remove(i)
    for i in range(len(ll)):
        if set(ll[i]) == set(new_original_l):
            return i
    return -1


def shapley_list_indexed(original_l, ll):
    for i in range(len(ll)):
        if set(ll[i]) == set(original_l):
            return i
    return -1


def PowerSetsBinary(items):
    N = len(items)
    set_all = []
    for i in range(2 ** N):
        combo = []
        for j in range(N):
            if (i >> j) % 2 == 1:
                combo.append(items[j])
        set_all.append(combo)
    return set_all


def loadHistoryModels():
    f = open(os.path.join(os.path.dirname(__file__), "gradientplus_models.txt"), "r")
    lines = f.readlines()
    ret_models = []

    f_ini_p = open(os.path.join(os.path.dirname(__file__), "initial_model_parameters.txt"), "r")
    para_lines = f_ini_p.readlines()
    w_paras = para_lines[0].split("\t")
    w_paras = [float(i) for i in w_paras]
    b_paras = para_lines[1].split("\t")
    b_paras = [float(i) for i in b_paras]
    w_initial = np.asarray(w_paras, dtype=np.float32).reshape([784, 10])
    b_initial = np.asarray(b_paras, dtype=np.float32).reshape([10])
    f_ini_p.close()

    ret_models.append([w_initial,b_initial])

    tem_model = []
    for i, line in enumerate(lines):
        if i % 2 == 0:
            lis = line.strip().replace("[", "").replace("]", "").split(",")
            lis = [float(i.strip()) for i in lis]
            lis = np.array(lis).reshape([784, 10])
            tem_model = [lis]
        else:
            lis = line.strip().replace("[", "").replace("]", "").split(",")
            lis = [float(i.strip()) for i in lis]
            lis = np.array(lis)
            tem_model.append(lis)
            ret_models.append(tem_model)
    f.close()
    return ret_models


if __name__ == "__main__":
    start_time = time.time()
    # data_num = np.asarray([5923, 6742, 5958, 6131, 5842])
    # agents_weights = np.divide(data_num, data_num.sum())
    # for index in range(NUM_AGENT):
    #     f = open(os.path.join(os.path.dirname(__file__), "weights_" + str(index) + ".txt"), "w")
    #     f.close()
    #     f = open(os.path.join(os.path.dirname(__file__), "bias_" + str(index) + ".txt"), "w")
    #     f.close()
    # f = open(os.path.join(os.path.dirname(__file__), "gradientplus_models.txt"), "w")
    # f.close()
    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

    DISTRIBUTION_TYPE = "SAME"

    federated_train_data_divide = None
    federated_train_data = None
    if DISTRIBUTION_TYPE == "SAME":
        federated_train_data_divide = [get_data_for_federated_agents(mnist_train, d, noiseX=True) for d in range(NUM_AGENT)]
        federated_train_data = federated_train_data_divide

    f_ini_p = open(os.path.join(os.path.dirname(__file__), "initial_model_parameters.txt"), "r")
    para_lines = f_ini_p.readlines()
    w_paras = para_lines[0].split("\t")
    w_paras = [float(i) for i in w_paras]
    b_paras = para_lines[1].split("\t")
    b_paras = [float(i) for i in b_paras]
    w_initial = np.asarray(w_paras, dtype=np.float32).reshape([784, 10])
    b_initial = np.asarray(b_paras, dtype=np.float32).reshape([10])
    f_ini_p.close()

    initial_model = {
        'weights': w_initial,
        'bias': b_initial
    }
    model = initial_model
    all_sets = PowerSetsBinary([i for i in range(NUM_AGENT)])
    learning_rate = 0.1
    pre_weights = []
    pre_bias = []
    
    m_w = np.zeros([784, 10], dtype=np.float32)
    m_b = np.zeros([10], dtype=np.float32)
    data_num = np.asarray([len(sample) for sample in federated_train_data_divide])

    s=sorted([i for i in range(NUM_AGENT)])
    l=permutations(s)
    all_orders = []
    for x in l:
        all_orders.append(list(x))
    
    agent_shapley_sum = [0 for i in range(NUM_AGENT)]

    for round_num in range(50):
        local_models = federated_train(model, learning_rate, federated_train_data)
        print("learning rate: ", learning_rate)
        # print(local_models[0][0])#第0个agent的weights矩阵
        # print(local_models[0][1])#第0个agent的bias矩阵
        
        group_shapley_value = []
        agent_shapley = []
        if round_num == 0:
            pre_weights = np.array([local_models[local_model_index][0] for local_model_index in range(len(local_models))])
            pre_bias = np.array([local_models[local_model_index][1] for local_model_index in range(len(local_models))])
        else:
            gradient_weight = []
            gradient_bias = []
            print(len(pre_weights), len(pre_weights[0]))
            for k in range(NUM_AGENT):
                gradient_weight.append(np.divide(np.subtract(pre_weights[k], local_models[k][0]), learning_rate))
                gradient_bias.append(np.divide(np.subtract(pre_bias[k], local_models[k][1]), learning_rate))
            
            pre_weights = np.array([local_models[local_model_index][0] for local_model_index in range(len(local_models))])
            pre_bias = np.array([local_models[local_model_index][1] for local_model_index in range(len(local_models))])

            for s in all_sets:
                group_shapley_value.append(
                    train_with_gradient_and_valuation(s, gradient_weight, gradient_bias, learning_rate, DISTRIBUTION_TYPE,
                                                    [model['weights'], model['bias']], data_num))
                print(str(s) + "\t" + str(group_shapley_value[len(group_shapley_value) - 1]))
            
            agent_shapley = []
            for index in range(NUM_AGENT):
                shapley = 0.0
                for order in all_orders:
                    pos = order.index(index)
                    pre_list = list(order[:pos])
                    edge_list = list(order[:pos+1])
                    pre_list_index = remove_list_indexed(index, pre_list, all_sets)
                    #print(order, pre_list_index, pre_list, edge_list, group_shapley_value[pre_list_index], group_shapley_value[shapley_list_indexed(edge_list, all_sets)])
                    if pre_list_index != -1:
                        shapley += (group_shapley_value[shapley_list_indexed(edge_list, all_sets)] - group_shapley_value[
                            pre_list_index]) / len(all_orders)
                agent_shapley.append(shapley)

            for i, ag_s in enumerate(agent_shapley):
                agent_shapley_sum[i] += ag_s
                print(ag_s)

            print(agent_shapley_sum)

        m_w = np.zeros([784, 10], dtype=np.float32)
        m_b = np.zeros([10], dtype=np.float32)
        for local_model_index in range(len(local_models)):
            m_w = np.add(np.multiply(local_models[local_model_index][0], 1 / NUM_AGENT), m_w)
            m_b = np.add(np.multiply(local_models[local_model_index][1], 1 / NUM_AGENT), m_b)
        model = {
                'weights': m_w,
                'bias': m_b
        }


        learning_rate = learning_rate * 0.9
        loss = federated_eval(model, federated_train_data)
        print('round {}, loss={}'.format(round_num, loss))
        print(time.time() - start_time)

    print(agent_shapley_sum)
    print("end_time", time.time() - start_time)