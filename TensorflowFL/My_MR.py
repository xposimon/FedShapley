from __future__ import absolute_import, division, print_function
import tensorflow_federated as tff
import tensorflow.compat.v1 as tf
import numpy as np
import time, os
import sys
from scipy.special import comb, perm
from itertools import permutations
from sklearn_extra.cluster import KMedoids

tf.enable_v2_behavior()
tf.enable_eager_execution()

NUM_AGENT = 5
NUM_CLUSTER = 5
SHARE_ROUND= 20
BATCH_SIZE = 100
ITER_ROUND = 50

def checkRange(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0

        if x[i] > 1:
            x[i] = 1
    return x

def handle_readfile(x):
    return float(x.strip("[").strip("]").strip())

def get_data_for_federated_agents(x, y, num, weight=False, noiseX=False):

    samples_len = len(x)
    output_sequence = []
    Classes = []

    if weight:
        if len(weight) != NUM_AGENT:
            raise RuntimeError("Weights not corret")
        for w in weight:
            if w <= 0:
                raise RuntimeError("Weights not corret")

        piece_num = int(samples_len/sum(weight))
        left=sum(weight[:num])
        right = left+num+1
    else:
        piece_num = samples_len/NUM_AGENT
        left = num
        right = num+1

    all_samples = [i for i in range(int(left*piece_num), int(right*piece_num))]

    for i in range(0, len(all_samples), BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        output_sequence.append({
            'x': np.array([x[i].flatten() / SCALE for i in batch_samples],
                        dtype=np.float32),
            'y': np.array([y[i] for i in batch_samples], dtype=np.int32)})

    if noiseX:
        if num >= 90:
            ratio = 0.3
            sum_agent = len(all_samples)
            x_dim =  len(output_sequence[0]['x'][0])
            noisepart = x_dim/2
            print(noisepart)
            #noise = ratio * np.array([(-1)**k if k <= noisepart else 0 for k in range(x_dim)])

            directory = os.path.dirname(__file__)
            with open(os.path.join(directory, "noise.out"), "r") as f:
                content = f.read()

            noise = content.split(",")
            noise = np.array(list(map(handle_readfile, noise)))

            for i in range(0, sum_agent):
                # Deterministic noise
                #output_sequence[int(i/BATCH_SIZE)]['x'][i % BATCH_SIZE] = checkRange(np.add(output_sequence[int(i/BATCH_SIZE)]['x'][i % BATCH_SIZE], noise))
                output_sequence[int(i/BATCH_SIZE)]['x'][i % BATCH_SIZE] = checkRange(ratio*np.random.randn(x_dim))

    return output_sequence

# Data reading
######################################################################################################################

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x = np.asarray(train_x.reshape(len(train_x), -1), dtype=np.float32)
train_y = np.asarray(train_y, dtype=np.int32)

# TODO fast debug
# train_x = train_x[:2000]
# train_y = train_y[:2000]

SCALE = np.float32(max(np.max(train_x), np.max(test_x)))

federated_train_data = [get_data_for_federated_agents(train_x, train_y, d) for d in range(NUM_AGENT)]
for i in range(NUM_AGENT):
    print(len(federated_train_data[i]),federated_train_data[i][0]['x'].shape)


with open(os.path.join(os.path.dirname(__file__), "initial_model_parameters.txt"), "r") as f_ini_p:
    para_lines = f_ini_p.readlines()

w_paras = para_lines[0].split("\t")
w_paras = [float(i) for i in w_paras]
b_paras = para_lines[1].split("\t")
b_paras = [float(i) for i in b_paras]

para_num = len(federated_train_data[0][0]['x'][0])
class_num = len(np.unique(test_y))

# para_num = 28*28
# class_num = 10

test_x = np.asarray(test_x.reshape(len(test_x), para_num), dtype=np.float32)
test_x = np.divide(test_x, SCALE)
test_y = np.asarray(test_y, dtype=np.int32)

w_initial = np.asarray(w_paras, dtype=np.float32).reshape([para_num, class_num])
b_initial = np.asarray(b_paras, dtype=np.float32).reshape([class_num])

model = {
    'weights': w_initial,
    'bias': b_initial
}

######################################################################################################################

BATCH_TYPE = tff.NamedTupleType([
    ('x', tff.TensorType(tf.float32, [None, para_num])),
    ('y', tff.TensorType(tf.int32, [None]))])

MODEL_TYPE = tff.NamedTupleType([
    ('weights', tff.TensorType(tf.float32, [para_num, class_num])),
    ('bias', tff.TensorType(tf.float32, [class_num]))])

@tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
def batch_loss(model, batch):
    predicted_y = tf.nn.softmax(tf.matmul(batch.x, model.weights) + model.bias)
    return -tf.reduce_mean(tf.reduce_sum(
        tf.one_hot(batch.y, 10) * tf.log(predicted_y), axis=[1]))

@tff.tf_computation(MODEL_TYPE, BATCH_TYPE, tf.float32)
def batch_train(initial_model, batch, learning_rate):
    # Define a group of model variables and set them to `initial_model`.
    model_vars = tff.utils.create_variables('v', MODEL_TYPE)
    train_model = tff.utils.assign(model_vars, initial_model)

    # Perform `SHARE_ROUND` step of gradient descent using loss from `batch_loss`.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    with tf.control_dependencies([train_model]):
        for i in range(SHARE_ROUND):
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

    return tff.sequence_reduce(all_batches, initial_model, batch_fn)


@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_eval(model, all_batches):
    #
    return tff.sequence_sum(
        tff.sequence_map(
            tff.federated_computation(
                lambda b: batch_loss(model, b), BATCH_TYPE),
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
def tf_federated_train(model, learning_rate, data):

    return tff.federated_map(
        local_train,
        [tff.federated_broadcast(model),
         tff.federated_broadcast(learning_rate),
         data])

def tf_train_with_gradient_and_valuation(agent_list, grad, bi, g_m, datanum):
    model_g = {
        'weights': g_m[0],
        'bias': g_m[1]
    }

    gradient_w = np.zeros([para_num, class_num], dtype=np.float32)
    gradient_b = np.zeros([class_num], dtype=np.float32)
    #print(agent_list, len(grad), agent_shapley, local_model_index)
    
    data_sum = 0
    for i in agent_list:
        data_sum += datanum[i]
    agents_w = [0 for _ in range(NUM_AGENT)]
    for i in agent_list:
        agents_w[i] = datanum[i] / data_sum
    print(agents_w)
    for j in agent_list:
        gradient_w = np.add(np.multiply(grad[j], agents_w[j]), gradient_w)
        gradient_b = np.add(np.multiply(bi[j], agents_w[j]), gradient_b)
    
    model_g['weights'] = np.subtract(model_g['weights'], gradient_w)
    model_g['bias'] = np.subtract(model_g['bias'], gradient_b)

    m = np.dot(test_x, np.asarray(model_g['weights']))
    test_result = m + np.asarray(model_g['bias'])
    y = tf.nn.softmax(test_result)
    correct_prediction = tf.equal(tf.argmax(y, 1), test_y)
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

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'del_low':
            del_low=True
            print("del low")
            removed_agents = []
        elif sys.argv[1] == 'del_high':
            del_high=True
    else:
        del_low = False
        del_high = False


    total_time = time.time()

    cur_agent_num = NUM_AGENT
    learning_rate = 0.1
    pre_weights = []
    pre_bias = []
    
    m_w = np.zeros([para_num, class_num], dtype=np.float32)
    m_b = np.zeros([class_num], dtype=np.float32)

    agent_shapley_sum = [0 for i in range(NUM_AGENT)]

    for round_num in range(ITER_ROUND):
        start_time = time.time()
        ## For tf
        local_models = tf_federated_train(model, learning_rate, federated_train_data)
        end_time = time.time()
        print("local train time:", (end_time-start_time)/len(federated_train_data))
        print("learning rate: ", learning_rate)
        
        start_time = time.time()
        if cur_agent_num > NUM_CLUSTER:
            func_outputs = [] 
            for i in range(cur_agent_num):
                
                m = np.dot(test_x, np.asarray(local_models[i][0]))
                test_result = m + np.asarray(local_models[i][1])
                func_outputs.append(test_result.reshape(-1))

            func_outputs = np.asarray(func_outputs)
            print(func_outputs.shape)
            kmedoids = KMedoids(metric="cosine", n_clusters=NUM_CLUSTER, random_state=0).fit(func_outputs)
            cluster_tags = kmedoids.labels_

            k = 0
            while k < max(list(cluster_tags))+1:
                if k not in cluster_tags:
                    for j in range(len(cluster_tags)):
                        if k < cluster_tags[j]:
                            cluster_tags[j] -= 1
                else:
                    k += 1

            cluster_cnt = [list(cluster_tags).count(i) for i in range(NUM_CLUSTER)]
            agents_select = [[] for i in range(max(list(cluster_tags))+1)]
            print(agents_select, cluster_tags)
            for i in range(cur_agent_num):
                agents_select[cluster_tags[i]].append(i)
        else:
            agents_select = [[i] for i in range(cur_agent_num)]
            cluster_tags = [i for i in range(cur_agent_num)]
            cluster_cnt = [1 for i in range(cur_agent_num)]

        print(agents_select)

        all_sets = PowerSetsBinary([i for i in range(len(agents_select))])
        s=sorted([i for i in range(len(agents_select))])

        l=permutations(s)
        all_orders = []
        for x in l:
            all_orders.append(list(x))

        group_shapley_value = []
        agent_shapley = []
        
        gradient_weight = []
        gradient_bias = []
        data_num = np.asarray([0 for i in range(len(agents_select))])
        for i in range(cur_agent_num):
            data_num[cluster_tags[i]] += len(federated_train_data[i])
        
        for k in range(len(agents_select)):
            avg_w = np.zeros([para_num, class_num], dtype=np.float32)
            avg_b = np.zeros([class_num], dtype=np.float32)
            for j in agents_select[k]:
                avg_w = np.add(local_models[j][0], avg_w)
                avg_b = np.add(local_models[j][1], avg_b)
            
            avg_w = np.divide(avg_w, len(agents_select[k]))
            avg_b = np.divide(avg_b, len(agents_select[k]))
            tmp_w_gra = np.subtract(model['weights'], avg_w)
            tmp_b_gra = np.subtract(model['bias'], avg_b)

            gradient_weight.append(tmp_w_gra.copy())
            gradient_bias.append(tmp_b_gra.copy())
        
        for s in all_sets:
            group_shapley_value.append(
                tf_train_with_gradient_and_valuation(s, gradient_weight, gradient_bias,
                                                [model['weights'], model['bias']], data_num))
            print(s, [agents_select[s[i]] for i in range(len(s))], str(group_shapley_value[-1]))
        
        cluster_shapley = []
        for index in range(len(agents_select)):
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
            cluster_shapley.append(shapley)

        for i in range(cur_agent_num):
            tag = cluster_tags[i]
            agent_shapley_sum[i] += (cluster_shapley[tag] / cluster_cnt[tag])

        if del_low==True:
            to_remove_list = []
            for tag in range(len(agents_select)):
                if cluster_shapley[tag] < 0:
                    # Remove all in the cluster
                    tmp = []
                    for idx in agents_select[tag]:
                        print(idx, agent_shapley_sum)
                        tmp.append((idx, agent_shapley_sum[idx]))
                        to_remove_list.append(idx)
                        
                        print("[+] Remove %d"%(idx), cluster_shapley[tag])
                        # print(removed_agents)
                    removed_agents.append((NUM_AGENT-cur_agent_num, tmp))
                    cur_agent_num -= len(agents_select[tag])

            if len(to_remove_list) > 0:
                
                agent_shapley_sum = [j for i,j in enumerate(agent_shapley_sum) if i not in to_remove_list]
                federated_train_data = [j for i,j in enumerate(federated_train_data) if i not in to_remove_list]

                if cur_agent_num > NUM_CLUSTER:
                    all_sets = PowerSetsBinary([i for i in range(NUM_CLUSTER)])
                    s=sorted([i for i in range(NUM_CLUSTER)])
                else:
                    all_sets = PowerSetsBinary([i for i in range(cur_agent_num)])
                    s=sorted([i for i in range(cur_agent_num)])

                l=permutations(s)
                all_orders = []
                for x in l:
                    all_orders.append(list(x))

        if del_low==True:
            # Dirty way to print shapley values
            tmp = agent_shapley_sum[:]
            for offset, agents in removed_agents[::-1]:
                t_agents = sorted(agents, key=lambda x: x[0])
                for i, v in t_agents:
                    tmp = tmp[:i] + [v] + tmp[i:]
            print(tmp)
        else:
            print(agent_shapley_sum)
        
        m_w = np.zeros([para_num, class_num], dtype=np.float32)
        m_b = np.zeros([class_num], dtype=np.float32)
        print(agents_select)
        for local_model_index in range(cur_agent_num):
            m_w = np.add(np.multiply(local_models[local_model_index][0], 1/cur_agent_num), m_w)
            m_b = np.add(np.multiply(local_models[local_model_index][1], 1/cur_agent_num), m_b)
        model = {
                'weights': m_w,
                'bias': m_b
        }

        learning_rate = learning_rate * 0.9
        loss = federated_eval(model, federated_train_data)
        print('round {}, loss={}'.format(round_num, loss))
        end_time = time.time()
        print("server time:", end_time-start_time)

        if cur_agent_num < 2*NUM_CLUSTER:
            if group_shapley_value[-1] < group_shapley_value[0]:
                print("[+] Train finish")
                break
            else:
                del_low = "Finish"    

    #print(agent_shapley_sum)
    print("total_time:", time.time() - total_time)


    if del_low == "Finish" or del_low==True:
        for offset, agents in removed_agents[::-1]:
            t_agents = sorted(agents, key=lambda x: x[0])
            for i, v in t_agents:
                agent_shapley_sum = agent_shapley_sum[:i] + [v] + agent_shapley_sum[i:]
    
    print("Final shapley values:", agent_shapley_sum)