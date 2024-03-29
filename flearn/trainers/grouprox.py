import numpy as np
from numpy.core.numeric import roll
from sklearn.utils.validation import check_random_state
from tqdm import trange, tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .fedbase import BaseFedarated
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, process_sparse_grad
from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.models.group import Group
import random
from utils.export_csv import CSVWriter
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, nan_euclidean_distances
from collections import Counter
import time

""" This Server class is customized for Group Prox """
class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Group prox to Train')
        self.group_list = [] # list of Group() instance
        self.group_ids = [] # list of group id

        # The attrs will be set in BaseFedarated.__init__(),
        # We repeat this assigement for clarification.
        self.num_group = params['num_group']
        self.prox = params['proximal']
        self.group_min_clients = params['min_clients']
        self.allow_empty = params['allow_empty']
        self.evenly = params['evenly']
        self.seed = params['seed']
        self.sklearn_seed = params['sklearn_seed']
        self.agg_lr = params['agg_lr']
        self.RAC = params['RAC'] # Randomly Assign Clients
        self.RCC = params['RCC'] # Random Cluster Center
        self.MADC = params['MADC'] # Use Mean Absolute Difference of pairwise Cossim
        self.recluster_epoch = params['recluster_epoch']
        self.max_temp = params['client_temp']
        self.temp_dict = {}

        """
        We implement THREE run mode of FedGroup: 
            1) Ours FedGroup
            2) IFCA: "An Efficient Framework for Clustered Federated Learning"
            3) FeSEM: "Multi-Center Federated Learning"
        """
        self.run_mode = 'FedGroup'
        if params['ifca'] == True:
            self.run_mode = 'IFCA'
        if params['fesem'] == True:
            self.run_mode = 'FeSEM'

        if self.prox == True:
            self.inner_opt = PerturbedGradientDescent(params['learning_rate'], params['mu'])
        else:
            self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        
        super(Server, self).__init__(params, learner, dataset)

        self.latest_model = self.client_model.get_params() # The global AVG model
        self.latest_update = self.client_model.get_params()

        self.create_groups()

        # Record the temperature of clients
        for c in self.clients: self.temp_dict[c] = self.max_temp

        self.writer = CSVWriter(params['export_filename'], 'results/'+params['dataset'], self.group_ids)

    """
    initialize the Group() instants
    """
    def create_groups(self):
        self.group_list = [Group(gid, self.client_model) for gid in range(self.num_group)] # 0,1,...,num_group
        self.group_ids = [g.get_group_id() for g in self.group_list]
        self.group_cold_start(self.RCC) # init the lastest_model of all groups

    def _get_cosine_similarity(self, m1, m2):
        flat_m1 = process_grad(m1)
        flat_m2 = process_grad(m2)
        cosine = np.dot(flat_m1, flat_m2) / (
            np.sqrt(np.sum(flat_m1**2)) * np.sqrt(np.sum(flat_m2**2)))
        return cosine

    """ measure the difference between client and group """
    def measure_difference(self, client, group, run_mode):
        # Strategy #1: angles (cosine) between two client update and group update
        # FedGroup use this.
        if run_mode == 'FedGroup':
            # FedGroup need pretrain the client
            cmodel, cupdate = self.pre_train_client(client)
            diff = self._get_cosine_similarity(cupdate, group.latest_update)
            diff = 1.0 - ((diff + 1.0) / 2.0) # scale to [0, 1] then flip
        
        # Strategy #2: Euclidean distance between client model and group model
        # FeSEM use this.
        if run_mode == 'FeSEM':
            cmodel, gmodel = process_grad(client.local_model), process_grad(group.latest_model)
            diff = np.sum((cmodel-gmodel)**2)

        # Strategy #3: Training Loss of group model
        # IFCA use this.
        if run_mode == 'IFCA':
            # The training loss of group model evaluate on client's training set
            backup_params = client.get_params()
            # Note: use group model for evaluation
            client.set_params(group.latest_model)
            _, train_loss, _ = client.train_error_and_loss()
            diff = train_loss
            # Restore paramters
            client.set_params(backup_params)

        return diff

    def get_ternary_cosine_similarity_matrix(self, w, V):
        #print(type(w), type(V))
        print('Delta w shape:', w.shape, 'Matrix V shape:', V.shape)
        w, V = w.astype(np.float32), V.astype(np.float32)
        left = np.matmul(w, V) # delta_w (dot) V
        scale = np.reciprocal(np.linalg.norm(w, axis=1, keepdims=True) * np.linalg.norm(V, axis=0, keepdims=True))
        diffs = left * scale # element-wise product
        diffs = (-diffs+1.)/2. # Normalize to [0,1]
        return diffs

    def get_assign_group(self, client, run_mode):
        diff_list = [] # tuple of (group, diff)
        for g in self.group_list:
            diff_g = self.measure_difference(client, g, run_mode)
            diff_list.append((g, diff_g)) # w/o sort

        #print("client:", client.id, "diff_list:", diff_list)
        assign_group = self.group_list[np.argmin([tup[1] for tup in diff_list])]

        return diff_list, assign_group
        

    def client_cold_start(self, client, run_mode):
        if client.group is not None:
            print("Warning: Client already has a group: {:2d}.".format(client.group))
        
        if run_mode == 'FedGroup':
            # Training is base on the global avg model
            start_model = self.client_model.get_params() # Backup the model first
            self.client_model.set_params(self.latest_model) # Set the training model to global avg model
            
            # client_model, client_update = self.pre_train_client(client) 
            diff_list, assign_group = self.get_assign_group(client, run_mode)

            # update(Init) the diff list of client
            client.update_difference(diff_list)
            
            # Only set the group attr of client, do not actually add clients to the group
            client.set_group(assign_group)
            
            # Recovery the training model
            self.client_model.set_params(start_model)
            return

        if run_mode == 'IFCA' or run_mode == 'FeSEM':
            pass # IFCA and FeSEM didn't use cold start strategy
            return
        
    """ Deal with the group cold start problem """
    def group_cold_start(self, random_centers=False):
        
        if self.run_mode == 'FedGroup':
            # Strategy #1: Randomly pre-train num_group clients as cluster centers
            # It is an optional strategy of FedGroup, named FedGroup-RCC
            if random_centers == True:
                    selected_clients = random.sample(self.clients, k=self.num_group)
                    for c, g in zip(selected_clients, self.group_list):
                        g.latest_model, g.latest_update = self.pre_train_client(c)
                        c.set_group(g)
            
            # Strategy #2: Pre-train, then clustering the directions of clients' weights
            # <FedGroup> and <FedGrouProx> use this strategy
            if random_centers == False:
                alpha = 20 ######## Pre-train Scaler ###################
                selected_clients = random.sample(self.clients, k=min(self.num_group*alpha, len(self.clients)))

                for c in selected_clients: c.clustering = True # Mark these clients as clustering client

                cluster = self.clustering_clients(selected_clients) # {Cluster ID: (cm, [c1, c2, ...])}
                # Init groups accroding to the clustering results
                for g, id in zip(self.group_list, cluster.keys()):
                    # Init the group latest update
                    g.latest_update = cluster[id][1]
                    g.latest_model = cluster[id][0]
                    # These clients do not need to be cold-started
                    # Set the "group" attr of client only, didn't add the client to group
                    for c in cluster[id][2]: c.set_group(g)

        # Strategy #3: random initialize group models as centers
        # <IFCA> and <FeSEM> use this strategy.
        if self.run_mode == 'IFCA' or self.run_mode == 'FeSEM':
            # Backup the original model params
            backup_params = self.client_model.get_params()
            # Reinitialize num_group clients models as centers models
            for idx, g in enumerate(self.group_list):
                # Change the seed of tensorflow
                new_seed = (idx + self.seed) * 888
                # Reinitialize params of model
                self.client_model.reinitialize_params(new_seed)
                new_params = self.client_model.get_params()
                g.latest_model, g.latest_update = new_params, new_params
                
            # Restore the seed of tensorflow
            tf.set_random_seed(123 + self.seed)
            # Restore the parameter of model
            self.client_model.set_params(backup_params)
            """
            # Reinitialize for insurance purposes
            new_params = self.client_model.reinitialize_params(123 + self.seed)
            # Restore the weights of model
            if np.array_equal(process_grad(backup_params), process_grad(new_params)) == True:
                print('############TRUE############')
            else:
                print('############FALSE############')
            """
        
        return

    """ Recluster the clients then refresh the group's optimize goal,
        It is a dynamic clustering strategy of FedGroup.
        Probelm of recluster strategy: the personality of group model will be weaken.
    """
    def group_recluster(self):
        if self.run_mode != 'FedGroup':
            return
        if self.RCC == True:
            print("Warning: The random cluster center strategy conflicts with dynamic clustering strategy.")
            return
        
        # Select alpha*num_group warm clients for reclustering. 
        alpha = 20
        warm_clients = [c for c in self.clients if c.is_cold() == False ]
        selected_clients = random.sample(warm_clients, k=min(self.num_group*alpha, len(warm_clients)))
        # Clear the clustering flage of warm clients
        for c in warm_clients: c.clustering, c.group = False, None
        for c in selected_clients: c.clustering = True

        # Recluster the selected clients
        cluster = self.clustering_clients(selected_clients) # {Cluster ID: (cm, [c1, c2, ...])}
        # Init groups accroding to the clustering results
        for g, id in zip(self.group_list, cluster.keys()):
            # Init the group latest update
            g.latest_update = cluster[id][1]
            g.latest_model = cluster[id][0]
            # These clients do not need to be cold-started
            # Set the "group" attr of client only, didn't add the client to group
            for c in cluster[id][2]: c.set_group(g)

        # Fresh the global AVG model
        self.refresh_global_model(self.group_list)

        # *Cold start the original warm clients for evaluation
        for c in warm_clients:
            if c.is_cold() == True:
                self.client_cold_start(c, run_mode='FedGroup')
        return

    def reassign_warm_clients(self):
        warm_clients = [c for c in self.clients if c.is_cold() == False]
        for c in warm_clients:
            c.group = None
            self.client_cold_start(c, self.run_mode)
        return

    """ Clustering clients by Clustering Algorithm """
    def clustering_clients(self, clients, n_clusters=None, max_iter=20):
        if n_clusters is None: n_clusters = self.num_group
        # Pre-train these clients first
        csolns, cupdates = {}, {}
        
        # The updates for clustering must be calculated upon a same model
        # We use the global auxiliary(AVG) model as start point
        self.client_model.set_params(self.latest_model)

        # Record the execution time
        start_time = time.time()
        for c in clients:
            csolns[c], cupdates[c] = self.pre_train_client(c)
        print("Pre-training takes {}s seconds".format(time.time()-start_time))

        update_array = [process_grad(update) for update in cupdates.values()]
        delta_w = np.vstack(update_array) # shape=(n_clients, n_params)
        
        # Record the execution time
        start_time = time.time()
        # Decomposed the directions of updates to num_group of directional vectors
        svd = TruncatedSVD(n_components=self.num_group, random_state=self.sklearn_seed)
        decomp_updates = svd.fit_transform(delta_w.T) # shape=(n_params, n_groups)
        print("SVD takes {}s seconds".format(time.time()-start_time))
        n_components = decomp_updates.shape[-1]

        # Record the execution time of EDC calculation
        start_time = time.time()
        decomposed_cossim_matrix = cosine_similarity(delta_w, decomp_updates.T) # shape=(n_clients, n_clients)

        ''' There is no need to normalize the data-driven measure because it is a dissimilarity measure
        # Normialize it to dissimilarity [0,1]
        decomposed_dissim_matrix = (1.0 - decomposed_cossim_matrix) / 2.0
        EDC = decomposed_dissim_matrix
        '''
        #EDC = self._calculate_data_driven_measure(decomposed_cossim_matrix, correction=False)
        print("EDC Matrix calculation takes {}s seconds".format(time.time()-start_time))
        
        # Test the excution time of full cosine dissimilarity
        start_time = time.time()
        full_cossim_matrix = cosine_similarity(delta_w) # shape=(n_clients, n_clients)
        '''
        # Normialize cossim to [0,1]
        full_dissim_matrix = (1.0 - full_cossim_matrix) / 2.0
        '''
        MADC = self._calculate_data_driven_measure(full_cossim_matrix, correction=True) # shape=(n_clients, n_clients)
        #MADC = full_dissim_matrix
        print("MADC Matrix calculation takes {}s seconds".format(time.time()-start_time))

        '''Apply RBF kernel to EDC or MADC
        gamma=0.2
        if self.MADC == True:
            affinity_matrix = np.exp(- MADC ** 2 / (2. * gamma ** 2))
        else: # Use EDC as default
            affinity_matrix = np.exp(- EDC ** 2 / (2. * gamma ** 2))
        '''
        # Record the execution time
        start_time = time.time()
        if self.MADC == True:
            affinity_matrix = MADC
            #affinity_matrix = (1.0 - full_cossim_matrix) / 2.0
            #result = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward').fit(full_cossim_matrix)
            result = AgglomerativeClustering(n_clusters, affinity='precomputed', linkage='complete').fit(affinity_matrix)
        else: # Use EDC as default
            affinity_matrix = decomposed_cossim_matrix
            #result = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward').fit(decomposed_cossim_matrix)
            #result = AgglomerativeClustering(n_clusters, affinity='precomputed', linkage='average').fit(EDC)
            result = KMeans(n_clusters, random_state=self.sklearn_seed, max_iter=max_iter).fit(affinity_matrix)
        #print('EDC', EDC[0][:10], '\nMADC', MADC[0][:10], '\naffinity', affinity_matrix[0][:10])
        #result = SpectralClustering(n_clusters, random_state=self.sklearn_seed, n_init=max_iter, affinity='precomputed').fit(affinity_matrix)

        print("Clustering takes {}s seconds".format(time.time()-start_time))
        print('Clustering Results:', Counter(result.labels_))
        #print('Clustering Inertia:', result.inertia_)

        cluster = {} # {Cluster ID: (avg_soln, avg_update, [c1, c2, ...])}
        cluster2clients = [[] for _ in range(n_clusters)] # [[c1, c2,...], [c3, c4,...], ...]
        for idx, cluster_id in enumerate(result.labels_):
            #print(idx, cluster_id, len(cluster2clients), n_clusters) # debug
            cluster2clients[cluster_id].append(clients[idx])
        for cluster_id, client_list in enumerate(cluster2clients):
            # calculate the means of cluster
            # All client have equal weight
            average_csolns = [(1, csolns[c]) for c in client_list]
            average_updates = [(1, cupdates[c]) for c in client_list]
            if average_csolns:
                # Update the cluster means
                cluster[cluster_id] = (self.aggregate(average_csolns), self.aggregate(average_updates), client_list)
            else:
                print("Error, cluster is empty")

        return cluster

    # Measure the discrepancy between group model and global model
    def measure_group_diffs(self):
        diffs = np.zeros(len(self.group_list))
        for idx, g in enumerate(self.group_list):
            # direction
            #diff = self.measure_difference(...)
            # square root
            model_a = process_grad(self.latest_model)
            model_b = process_grad(g.latest_model)
            diff = np.sum((model_a-model_b)**2)**0.5
            diffs[idx] = diff
        diffs = diffs + [np.sum(diffs)] # Append the sum(discrepancies) to the end
        return diffs

    # Measure the discrepancy between group model and client model
    def measure_client_group_diffs(self):
        average_group_diffs = np.zeros(len(self.group_list))
        total_group_diff = 0.0
        number_clients = [len(g.get_client_ids()) for g in self.group_list]
        for idx, g in enumerate(self.group_list):
            diff = 0.0
            if number_clients[idx] > 0:
                model_g = process_grad(g.latest_model)
                for c in g.clients.values():
                    model_c = process_grad(c.local_model)
                    diff += np.sum((model_c-model_g)**2)**0.5
                total_group_diff += diff
                average_group_diffs[idx] = diff / float(number_clients[idx])
                g.latest_diff = average_group_diffs[idx]
            else:
                average_group_diffs[idx] = 0 # The group is empty, the discrepancy is ZERO
        average_total_diff = total_group_diff / sum(number_clients)
        average_diffs = np.append([average_total_diff], average_group_diffs) # Append the sum of average (discrepancies) to the head
        
        return average_diffs

    """ Pre-train the client 1 epoch and return weights,
        Train the client upon the global AVG model.
    """
    def pre_train_client(self, client):
        start_model = self.client_model.get_params() # Backup the start model
        self.client_model.set_params(self.latest_model) # Set the run model to the global AVG model
        if self.prox == True:
            # Set the value of vstart to be the same as the client model to remove the proximal term
            self.inner_opt.set_params(self.latest_model, self.client_model)
        # Pretrain 1 epoch
        #soln, stat = client.solve_inner() # Pre-train the client only one epoch
        # or Pretrain 20 iterations
        soln, stat = client.solve_iters(50)

        ws = soln[1] # weights of model
        updates = [w1-w0 for w0, w1 in zip(self.latest_model, ws)]

        self.client_model.set_params(start_model) # Recovery the model
        return ws, updates

    def get_not_empty_groups(self):
        not_empty_groups = [g for g in self.group_list if not g.is_empty()]
        return not_empty_groups

    def group_test(self):
        backup_model = self.latest_model # Backup the global model
        results = []
        tot_num_client = 0
        for g in self.group_list:
            c_list = []
            for c in self.clients:
                if c.group == g:
                    c_list.append(c)
            tot_num_client += len(c_list)
            num_samples = []
            tot_correct = []
            self.client_model.set_params(g.latest_model)
            for c in c_list:
                ct, ns = c.test()
                tot_correct.append(ct*1.0)
                num_samples.append(ns)
            ids = [c.id for c in c_list]
            results.append((ids, g, num_samples, tot_correct))
        self.client_model.set_params(backup_model) # Recovery the model
        return tot_num_client, results

    def group_train_error_and_loss(self):
        backup_model = self.latest_model # Backup the global model
        results = []
        for g in self.group_list:
            c_list = []
            for c in self.clients:
                if c.group == g:
                    c_list.append(c)
            num_samples = []
            tot_correct = []
            losses = []
            self.client_model.set_params(g.latest_model)
            for c in c_list:
                ct, cl, ns = c.train_error_and_loss() 
                tot_correct.append(ct*1.0)
                num_samples.append(ns)
                losses.append(cl*1.0)
            ids = [c.id for c in c_list]
            results.append((ids, g, num_samples, tot_correct, losses))
        self.client_model.set_params(backup_model) # Recovery the model
        return results

    """Main Train Function
    """
    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))

        # Clients cold start, pre-train all clients
        start_time = time.time()
        for c in self.clients:
                if c.is_cold() == True:
                    self.client_cold_start(c, self.run_mode)
        print("Cold start clients takes {}s seconds".format(time.time()-start_time))

        for i in range(self.num_rounds):

            # Random select clients
            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            np.random.seed(i)  # make sure that the stragglers are the same for FedProx and FedAvg
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1 - self.drop_percent)), replace=False)
            
            # Clear all group, the group attr of client is retrained
            for g in self.group_list: g.clear_clients()
            
            # Reshcedule selected clients to groups, add client to group's client list
            if self.run_mode == 'FedGroup':
                # Cold start the newcomer
                for c in selected_clients:
                    if c.is_cold() == True:
                        self.client_cold_start(c, self.run_mode)
                # Reschedule the group
                self.reschedule_groups(selected_clients, self.allow_empty, self.evenly, self.RAC)
            else: # IFCA and FeSEM need rescheduling client in each round
                start_time = time.time()
                if self.run_mode == 'IFCA':
                    self.IFCA_reschedule_group(selected_clients)
                if self.run_mode == 'FeSEM':
                    self.FeSEM_reschedule_group(selected_clients)
                print("Scheduling clients takes {}s seconds".format(time.time()-start_time))

            # Get not empty groups
            handling_groups = self.get_not_empty_groups()

            for g in self.group_list:
                if g in handling_groups:
                    print("Group {}, clients {}".format(g.get_group_id(), g.get_client_ids()))
                else:
                    print("Group {} is empty.".format(g.get_group_id()))

            # Freeze these groups before training
            for g in handling_groups:
                g.freeze()
            
            # Evalute group model before training
            if i % self.eval_every == 0:
                """
                stats = self.test() # have set the latest model for all clients
                # Test on training data, it's redundancy
                stats_train = self.train_error_and_loss()
                """
                num_test_client, group_stats = self.group_test()
                group_stats_train = self.group_train_error_and_loss()
                test_tp, test_tot = 0, 0
                train_tp, train_tot = 0, 0
                train_loss_list, number_samples_list = [], []
                for stats, stats_train in zip(group_stats, group_stats_train):
                    tqdm.write('Group {}'.format(stats[1].id))
                    test_tp += np.sum(stats[3])
                    test_tot += np.sum(stats[2])
                    test_acc = np.sum(stats[3])*1.0/np.sum(stats[2])
                    tqdm.write('At round {} accuracy: {}'.format(i, test_acc))  # testing accuracy
                    
                    train_tp += np.sum(stats_train[3])
                    train_tot += np.sum(stats_train[2])
                    train_loss_list += stats_train[4]
                    number_samples_list += stats_train[2]
                    
                    train_acc = np.sum(stats_train[3])*1.0/np.sum(stats_train[2])
                    tqdm.write('At round {} training accuracy: {}'.format(i, train_acc)) # train accuracy
                    train_loss = np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])
                    tqdm.write('At round {} training loss: {}'.format(i, train_loss))
                    
                    # Write results to csv file
                    self.writer.write_stats(i, stats[1].id, test_acc,
                        train_acc, train_loss, len(stats[1].get_client_ids()))
                
                mean_test_acc = test_tp*1.0 / test_tot
                mean_train_acc = train_tp*1.0 / train_tot
                mean_train_loss = np.dot(train_loss_list, number_samples_list)*1.0/np.sum(number_samples_list)
                self.writer.write_means(mean_test_acc, mean_train_acc, mean_train_loss)
                print('At round {} mean test accuracy: {} mean train accuracy: {} mean train loss: {} \
                    number of test client: {}'.format(i, mean_test_acc, mean_train_acc, mean_train_loss, num_test_client))
                #diffs = self.measure_group_diffs()
                diffs = self.measure_client_group_diffs()
                print("The client-group discrepancy are:", diffs)
                # The diffs in the first round may not make sense.
                self.writer.write_diffs(diffs)

            # Broadcast the global model to clients(groups)
            # self.client_model.set_params(self.latest_model)
            
            # Train each group sequentially
            start_time = time.time()
            for g in handling_groups:
                # Backup the origin model
                print("Begin group {:2d} training".format(g.get_group_id()))
                # Each group train group_epochs round
                for _ in range(g.group_epochs):
                    if self.prox == True:
                        # Update the optimizer, the vstar is latest_model of this group
                        self.inner_opt.set_params(g.latest_model, self.client_model)
                    # Set the global model to the group model
                    self.client_model.set_params(g.latest_model)
                    
                    """ Begin group training, call the train() function of Group object,
                        return the update vector of client.
                    """
                    cmodels, cupdates = g.train()
                    # TODO: After end of the training of client, update the diff list of client

            print("Training groups takes {}s seconds".format(time.time()-start_time))
            
            # Aggregate groups model and update the global (latest) model 
            # Note: IFCA and FeSEM do not implement inter-group aggregation (agg_lr=0)
            self.aggregate_groups(self.group_list, agg_lr=self.agg_lr)
            
            # Refresh the global model and global delta weights (latest_update)
            self.refresh_global_model(self.group_list)

            ##########  Dynamic Strategy Code Start ##########
            # Recluster group, dynamic strategy
            if self.recluster_epoch:
                if i > 0 and i % self.recluster_epoch == 0:
                    print(f"***** Recluster groups in epoch {i} ******")
                    self.group_recluster()

            # Fresh the temperature of client
            if self.max_temp:
                self.refresh_client_temperature(cmodels)
            ##########  Dynamic Strategy Code End ##########

        # Close the writer and end the training
        self.writer.close()

    # Use for matain the global AVG model and global latest update
    def refresh_global_model(self, groups):
        start_model = self.latest_model 
        # Aggregate the groups model
        gsolns = []
        for g in groups:
            gsolns.append((1.0, g.latest_model)) # (n_k, soln)
        new_model = self.aggregate(gsolns)
        self.latest_update = [w1-w0 for w0, w1 in zip(start_model, new_model)]
        self.latest_model = new_model

        return

    def refresh_client_temperature(self, cmodels):
        self.temp_dict
        # Strategy1: Discrepancy-based
        diffs = self.measure_client_group_diffs()
        avg_total_diff = diffs[0]
        avg_group_diff = {g:diffs[idx+1] for idx, g in enumerate(self.group_list)}
        for c, model in cmodels.items():
            mean_diff = avg_group_diff[c.group]
            model_g = process_grad(c.group.latest_model)
            model_c = process_grad(model)
            client_diff = np.sum((model_c-model_g)**2)**0.5
            
            if client_diff > mean_diff:
                # This client has large discrepancy
                self.temp_dict[c] = self.temp_dict[c] - 1
            if self.temp_dict[c] == 0:
                # Redo the cold start
                old_group = c.group
                c.group = None
                self.client_cold_start(c, run_mode='FedGroup')
                self.temp_dict[c] = self.max_temp
                if old_group != c.group:
                    print(f'Client {c.id} migrates from Group {old_group.id} to Group {c.group.id}!')

    def aggregate_groups(self, groups, agg_lr):
        gsolns = [(sum(g.num_samples), g.latest_model) for g in groups]
        group_num = len(gsolns)
        # Calculate the scale of group models
        gscale = [0]*group_num
        for i, (_, gsoln) in enumerate(gsolns):
            for v in gsoln:
                gscale[i] += np.sum(v.astype(np.float64)**2)
            gscale[i] = gscale[i]**0.5
        # Aggregate the models of each group separately
        for idx, g in enumerate(groups):
            base = [0]*len(gsolns[idx][1])
            weights = [agg_lr*(1.0/scale) for scale in gscale]
            weights[idx] = 1 # The weight of the main group is 1
            total_weights = sum(weights)
            for j, (_, gsoln) in enumerate(gsolns):
                for k, v in enumerate(gsoln):
                    base[k] += weights[j]*v.astype(np.float64)
            averaged_soln = [v / total_weights for v in base]
            # Note: The latest_update accumulated from last fedavg training
            inter_aggregation_update = [w1-w0 for w0, w1 in zip(g.latest_model, averaged_soln)]
            g.latest_update = [up0+up1 for up0, up1 in zip(g.latest_update, inter_aggregation_update)]
            g.latest_model = averaged_soln

        return
  
    """ Reschedule function of FedGroup, assign selected client to group according to some addtional options.
    """
    def reschedule_groups(self, selected_clients, allow_empty=False, evenly=False, randomly=False):
        
        # deprecated
        def _get_even_per_group_num(selected_clients_num, group_num):
            per_group_num = np.array([selected_clients_num // group_num] * group_num)
            remain = selected_clients_num - sum(per_group_num)
            random_groups = random.sample(range(group_num), remain)
            per_group_num[random_groups] += 1 # plus the remain
            return per_group_num

        selected_clients = selected_clients.tolist() # convert numpy array to list

        if randomly==True and evenly==False:
            for c in selected_clients:
                if c.is_cold() == False:
                    if c.clustering == False:
                        # Randomly assgin client
                        random.choice(self.group_list).add_client(c)
                    else:
                        # This client is clustering client.
                        c.group.add_client(c)
                else:
                    print('Warnning: A newcomer is no pre-trained.')
            return
            
        if randomly==True and evenly==True:
            """
            # Randomly assgin client, but each group is even
            per_group_num = _get_even_per_group_num(len(selected_clients), len(self.group_list))
            for g, max in zip(self.group_list, per_group_num): g.max_clients = max
            head_idx, tail_idx = 0, 0
            for group_num, g in zip(per_group_num, self.group_list):
                tail_idx += group_num
                g.add_clients(selected_clients[head_idx, tail_idx])
                head_idx = tail_idx
            """
            print("Experimental setting is invalid.")
            return 

        if randomly==False and allow_empty==True:
            # Allocate clients to their first rank groups, some groups may be empty
            for c in selected_clients:
                if c.is_cold() != True:
                    first_rank_group = c.group
                    first_rank_group.add_client(c)
            return
        
        if randomly==False and evenly==True:
            """ Strategy #1: Calculate the number of clients in each group (evenly) """
            selected_clients_num = len(selected_clients)
            group_num = len(self.group_list)
            per_group_num = np.array([selected_clients_num // group_num] * group_num)
            remain = selected_clients_num - sum(per_group_num)
            random_groups = random.sample(range(group_num), remain)
            per_group_num[random_groups] += 1 # plus the remain

            for g, max in zip(self.group_list, per_group_num):
                g.max_clients = max

            """ Allocate clients to make the client num of each group evenly """
            for c in selected_clients:
                if c.is_cold() != True:
                    first_rank_group = c.group
                    if not first_rank_group.is_full():
                        first_rank_group.add_client(c)
                    else:
                        # The first rank group is full, choose next group
                        diff_list = c.difference
                        # Sort diff_list
                        diff_list = sorted(diff_list, key=lambda tup: tup[1])
                        for (group, diff) in diff_list:
                            if not group.is_full():
                                group.add_client(c)
                                break
            return

        if randomly==False and evenly==False:
            """ Strategy #2: Allocate clients to meet the minimum client requirements """
            for g in self.group_list: g.min_clients = self.group_min_clients
            # First ensure that each group has at least self.min_clients clients.
            diff_list, assigned_clients = [], []
            for c in selected_clients:
                diff_list += [(c, g, diff) for g, diff in c.difference]
            diff_list = sorted(diff_list, key=lambda tup: tup[2])
            for c, g, diff in diff_list:
                if len(g.client_ids) < g.min_clients and c not in assigned_clients:
                    g.add_client(c)
                    assigned_clients.append(c)
                               
            # Then add the remaining clients to their first rank group
            for c in selected_clients:
                if c not in assigned_clients:
                    first_rank_group = c.group
                    if c.id not in first_rank_group.client_ids:
                        first_rank_group.add_client(c)
            return

        return

    """ Reschedule function of IFCA, assign selected client according to training loss
    """
    def IFCA_reschedule_group(self, selected_clients):
        for c in selected_clients:
            # IFCA assign client to group with minium training loss
            diff_list, assign_group = self.get_assign_group(c, run_mode='IFCA')
            c.set_group(assign_group)
            c.update_difference(diff_list)
            # Add client to group's client list
            assign_group.add_client(c)
        return

    """ Similar to IFCA, get_assign_group() can handle well
    """
    def FeSEM_reschedule_group(self, selected_clients):
        for c in selected_clients:
            # IFCA assign client to group with minium training loss
            diff_list, assign_group = self.get_assign_group(c, run_mode='FeSEM')
            c.set_group(assign_group)
            c.update_difference(diff_list)
            # Add client to group's client list
            assign_group.add_client(c)
        return

    def _calculate_data_driven_measure(self, pm, correction=False):
        ''' calculate the data-driven measure such as MADD'''
        # Input: pm-> proximity matrix; Output: dm-> data-driven distance matrix
        # pm.shape=(n_clients, n_dims), dm.shape=(n_clients, n_clients)
        n_clients, n_dims = pm.shape[0], pm.shape[1]
        dm = np.zeros(shape=(n_clients, n_clients))
        
        """ Too Slow, and misunderstanding MADD. Deprecated
        for i in range(n_clients):
            for j in range(i+1, n_clients):
                for k in range(n_clients):
                    if k !=i and k != j:
                        dm[i,j] = dm[j,i] = abs(np.sum((pm[i]-pm[k])**2)**0.5 - \
                            np.sum((pm[j]-pm[k])**2)**0.5)
        """
        # Fast version
        '''1, Get the repeated proximity matrix.
            We write Row1 = d11, d12, d13, ... ; and Row2 = d21, d22, d23, ...
            [   Row1    ]   [   Row2    ]       [   Rown    ]
            |   Row1    |   |   Row2    |       |   Rown    |
            |   ...     |   |   ...     |       |   ...     |
            [   Row1    ],  [   Row2    ], ..., [   Rown    ]
        '''
        row_pm_matrix = np.repeat(pm[:,np.newaxis,:], n_clients, axis=1)
        #print('row_pm', row_pm_matrix[0][0][:5], row_pm_matrix[0][1][:5])

        # Get the repeated colum proximity matrix
        '''
            [   Row1    ]   [   Row1    ]       [   Row1    ]
            |   Row2    |   |   Row2    |       |   Row2    |
            |   ...     |   |   ...     |       |   ...     |
            [   Rown    ],  [   Rown    ], ..., [   Rown    ]
        '''
        col_pm_matrix = np.tile(pm, (n_clients, 1, 1))
        #print('col_pm', col_pm_matrix[0][0][:5], col_pm_matrix[0][1][:5])
        
        # Calculate the absolute difference of two disstance matrix, It is 'abs(||u-z|| - ||v-z||)' in MADD.
        # d(1,2) = ||w1-z|| - ||w2-z||, shape=(n_clients,); d(x,x) always equal 0
        '''
            [   d(1,1)  ]   [   d(1,2)  ]       [   d(1,n)  ]
            |   d(2,1)  |   |   d(2,2)  |       |   d(2,n)  |
            |   ...     |   |   ...     |       |   ...     |
            [   d(n,1)  ],  [   d(n,2)  ], ..., [   d(n,n)  ]
        '''
        absdiff_pm_matrix = np.abs(col_pm_matrix - row_pm_matrix) # shape=(n_clients, n_clients, n_clients)
        # Calculate the sum of absolute differences
        if correction == True:
            # We should mask these values like sim(1,2), sim(2,1) in d(1,2)
            mask = np.zeros(shape=(n_clients, n_clients))
            np.fill_diagonal(mask, 1) # Mask all diag
            mask = np.repeat(mask[np.newaxis,:,:], n_clients, axis=0)
            for idx in range(mask.shape[-1]):
                #mask[idx,idx,:] = 1 # Mask all row d(1,1), d(2,2)...; Actually d(1,1)=d(2,2)=0
                mask[idx,:,idx] = 1 # Mask all 0->n colum for 0->n diff matrix,
            dm = np.sum(np.ma.array(absdiff_pm_matrix, mask=mask), axis=-1) / (n_dims-2.0)
        else:
            dm = np.sum(absdiff_pm_matrix, axis=-1) / (n_dims)
        #print('absdiff_pm_matrix', absdiff_pm_matrix[0][0][:5])

        return dm # shape=(n_clients, n_clients)

    def test_ternary_cosine_similariy(self, alpha=20):
        ''' compare the ternary similarity and cosine similarity '''
        def _calculate_cosine_distance(v1, v2):
            cosine = np.dot(v1, v2) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))
            return cosine

        # Pre-train all clients
        csolns, cupdates = {}, {}
        for c in self.clients:
            csolns[c], cupdates[c] = self.pre_train_client(c)

        # random selecte alpha * m clients to calculate the direction matrix V
        n_clients = len(self.clients)
        selected_clients = random.sample(self.clients, k=min(self.num_group*alpha, n_clients))
        clustering_update_array = [process_grad(cupdates[c]) for c in selected_clients]
        clustering_update_array = np.vstack(clustering_update_array).T # shape=(n_params, n_clients)
        
        # We decomposed the update vectors to numer_group components.
        svd = TruncatedSVD(n_components=self.num_group, random_state=self.sklearn_seed)
        decomp_updates = svd.fit_transform(clustering_update_array) # shape=(n_params, n_groups)
        n_components = decomp_updates.shape[-1]

        """
        # calculate the ternary similarity matrix for all clients
        ternary_cossim = []
        update_array = [process_grad(cupdates[c]) for c in self.clients]
        delta_w = np.vstack(update_array) # shape=(n_clients, n_params)
        ternary_cossim = self.get_ternary_cosine_similarity_matrix(delta_w, decomp_updates)
        """
        """
        # calculate the tranditional pairwise cosine similarity matrix for all clients
        old_cossim = cosine_similarity(delta_w)
        old_cossim = (1.0 - old_cossim) / 2.0 # Normalize
        """

        # Calculate the data-driven decomposed cosine dissimilarity (EDC) for all clients
        update_array = [process_grad(cupdates[c]) for c in self.clients]
        delta_w = np.vstack(update_array) # shape=(n_clients, n_params)
        decomposed_cossim_matrix = cosine_similarity(delta_w, decomp_updates.T) # Shape = (n_clients, n_groups)
        print("Cossim_matrix shape:", decomposed_cossim_matrix.shape)
        # Normalize cossim to dissim
        #decomposed_dissim_matrix = (1.0 - decomposed_cossim_matrix) / 2.0
        #EDC = self._calculate_data_driven_measure(decomposed_cossim_matrix, correction=False)
        EDC = euclidean_distances(decomposed_cossim_matrix)

        # Calculate the data-driven full cosine similarity for all clients
        full_cossim_matrix = cosine_similarity(delta_w)
        # Normalize
        #full_dissim_matrix = (1.0 - full_cossim_matrix) / 2.0
        MADC = self._calculate_data_driven_measure(full_cossim_matrix, correction=True)

        # Print the shape of distance matries, make sure equal
        #print(EDC.shape, MADC.shape) # shape=(n_clients, n_clients)

        iu = np.triu_indices(n_clients)
        x, y = EDC[iu], MADC[iu]
        mesh_points = np.vstack((x,y)).T

        #print(x.shape, y.shape)
        np.savetxt("cossim.csv", mesh_points, delimiter="\t")
        return x, y