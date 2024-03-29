import json
import numpy as np
import os
from PIL import Image
from math import ceil

def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)

"""
def batch_data_multiple_iters(data, batch_size, num_iters):
    data_x = data['x']
    data_y = data['y']

    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    idx = 0

    for i in range(num_iters):
        if idx+batch_size >= len(data_x):
            idx = 0
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)
        batched_x = data_x[idx: idx+batch_size]
        batched_y = data_y[idx: idx+batch_size]
        idx += batch_size
        yield (batched_x, batched_y)
"""
def batch_data_multiple_iters(data, batch_size, num_iters):
    data_x = data['x']
    data_y = data['y']
    data_size = data_y.shape[0]

    random_idx = np.arange(data_size)
    np.random.shuffle(random_idx)
    # Shuffle the features and labels
    data_x, data_y = data_x[random_idx], data_y[random_idx]
    max_iter = ceil(data_size / batch_size)

    for iter in range(num_iters):
        round_step = (iter+1) % max_iter # round_step: 1, 2, ..., max_iter-1, 0
        if round_step == 0:
            # Exceed 1 epoch
            x_part1, y_part1 = data_x[(max_iter-1)*batch_size: data_size], \
                data_y[(max_iter-1)*batch_size: data_size]
            # Shuffle dataset before we get the next part
            np.random.shuffle(random_idx)
            data_x, data_y = data_x[random_idx], data_y[random_idx]
            x_part2, y_part2 = data_x[0: max_iter*batch_size%data_size], \
                data_y[0: max_iter*batch_size%data_size]

            batched_x = np.vstack([x_part1, x_part2])
            batched_y = np.hstack([y_part1, y_part2])  
        else:
            batched_x = data_x[(round_step-1)*batch_size: round_step*batch_size]
            batched_y = data_y[(round_step-1)*batch_size: round_step*batch_size]

        yield (batched_x, batched_y)

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data

def process_x(raw_x_batch):
    x_batch = [load_image(i) for i in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch

def process_y(raw_y_batch):
    return raw_y_batch

def load_image(img_name):
    IMAGE_SIZE = 84

    IMAGES_DIR = os.path.join('data', 'celeba', 'data', 'raw', 'img_align_celeba')
    img = Image.open(os.path.join(IMAGES_DIR, img_name))
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
    return np.array(img)
    
def gen_batch_celeba(data, batch_size, num_iter):
    data_x_name = data['x']
    data_y_name = data['y']

    data_x = np.asarray(process_x(data_x_name))
    data_y = np.asarray(process_y(data_y_name))

    index = len(data_y)

    for i in range(num_iter):
        index += batch_size
        if (index + batch_size > len(data_y)):
            index = 0
            np.random.seed(i + 1)
            # randomly shuffle the data after one pass of the entire training set
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)

        batched_x = data_x[index: index + batch_size]
        batched_y = data_y[index: index + batch_size]

        yield (batched_x, batched_y)

def gen_batch(data, batch_size, num_iter):

    data_x = np.array(data['x'])
    data_y = np.array(data['y'])

    index = len(data_y)

    for i in range(num_iter):
        index += batch_size
        if (index + batch_size > len(data_y)):
            index = 0
            np.random.seed(i+1)
            # randomly shuffle the data after one pass of the entire training set         
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)

        batched_x = data_x[index: index + batch_size]
        batched_y = data_y[index: index + batch_size]
        
        yield (batched_x, batched_y)

def gen_epoch(data, num_iter):
    '''
    input: the training data and number of iterations
    return: the E epoches of data to run gradient descent
    '''
    data_x = data['x']
    data_y = data['y']
    for i in range(num_iter):
        # randomly shuffle the data after each epoch
        np.random.seed(i+1)
        rng_state = np.random.get_state()
        np.random.shuffle(data_x)
        np.random.set_state(rng_state)
        np.random.shuffle(data_y)

        batched_x = data_x
        batched_y = data_y

        yield (batched_x, batched_y)

class Metrics(object):
    def __init__(self, clients, params):
        self.params = params
        num_rounds = params['num_rounds']
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}      
        self.accuracies = []
        self.train_accuracies = []

    def update(self, rnd, cid, stats):
        bytes_w, comp, bytes_r = stats
        self.bytes_written[cid][rnd] += bytes_w
        self.client_computations[cid][rnd] += comp
        self.bytes_read[cid][rnd] += bytes_r

    def write(self):
        metrics = {}
        metrics['dataset'] = self.params['dataset']
        metrics['num_rounds'] = self.params['num_rounds']
        metrics['eval_every'] = self.params['eval_every']
        metrics['learning_rate'] = self.params['learning_rate']
        metrics['mu'] = self.params['mu']
        metrics['num_epochs'] = self.params['num_epochs']
        metrics['batch_size'] = self.params['batch_size']
        metrics['accuracies'] = self.accuracies
        metrics['train_accuracies'] = self.train_accuracies
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read
        metrics_dir = os.path.join('out', self.params['dataset'], 'metrics_{}_{}_{}_{}_{}.json'.format(self.params['seed'], self.params['optimizer'], self.params['learning_rate'], self.params['num_epochs'], self.params['mu']))
	#os.mkdir(os.path.join('out', self.params['dataset']))
        if not os.path.exists(os.path.join('out', self.params['dataset'])):
            os.mkdir(os.path.join('out', self.params['dataset']))
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)
