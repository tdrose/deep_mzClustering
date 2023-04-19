import torch
import numpy as np
import torch.nn.functional as functional
from random import sample

from .CAE import CAE
from .cnnClust import cnnClust
from .pseudo_labeling import pseudo_labeling, run_knn


class Clustering(object):
    def __init__(self, images, label_path=None, num_cluster=7, height=40, width=40, lr=0.0001,
                 batch_size=128, knn=True, k=10, use_gpu=True):
        super(Clustering, self).__init__()
        # self.spec_path = spec_path
        self.label_path = label_path
        self.num_cluster = num_cluster
        self.height = height
        self.width = width
        self.lr = lr
        self.batch_size = batch_size
        self.KNN = knn
        self.k = k
        self.knn_adj = None
        self.loss_func = torch.nn.MSELoss()
        self.image_data = images
        self.use_gpu = use_gpu
        self.image_label = None

        self.device = torch.device("cuda" if use_gpu else "cpu")

        # self.spec = np.genfromtxt(self.spec_path, delimiter=' ')
        # self.image_data = np.reshape(self.spec, (-1, self.height, self.width, 1))
        
        self.sampleN = len(self.image_data)
        
        if self.label_path:
            self.label = np.genfromtxt(self.label_path, delimiter=' ')
            self.image_label = np.asarray(self.label, dtype=np.int32)

        # image normalization
        for i in range(0, self.sampleN):
            current_min = np.min(self.image_data[i, ::])
            current_max = np.max(self.image_data[i, ::])
            self.image_data[i, ::] = (current_max - self.image_data[i, ::]) / (current_max - current_min)

        if knn:
            self.knn_adj = run_knn(self.image_data.reshape((self.image_data.shape[0], -1)),
                                   k=self.k)

    @staticmethod
    def get_batch(train_image, batch_size, train_label=None):
        sample_id = sample(range(len(train_image)), batch_size)
        # index = [[]]
        # index[0] = [x for x in range(batch_size)]
        # index.append(sample_id)
        batch_image = train_image[sample_id, ]
        if train_label is None:
            batch_label = None
        else:
            batch_label = train_label[sample_id, ]
        return batch_image, batch_label, sample_id

    @staticmethod
    def get_batch_sequential(train_image, train_label, batch_size, i):
        if i < len(train_image)//batch_size:
            batch_image = train_image[(batch_size*i):(batch_size*(i+1)), :]
            batch_label = train_label[(batch_size*i):(batch_size*(i+1))]
        else:
            batch_image = train_image[(batch_size*i):len(train_image), :]
            batch_label = train_label[(batch_size*i):len(train_image)]
        return batch_image, batch_label

    def train(self):
        
        cae = CAE(train_mode=True, height=self.height, width=self.width).to(self.device)
        clust = cnnClust(num_clust=self.num_cluster, height=self.height, width=self.width).to(self.device)
        
        model_params = list(cae.parameters()) + list(clust.parameters())
        optimizer = torch.optim.RMSprop(params=model_params, lr=0.001, weight_decay=0)
        # torch.optim.Adam(model_params, lr=lr)

        uu = 98
        ll = 46
        loss_list = list()

        random_seed = 1224
        torch.manual_seed(random_seed)
        if self.use_gpu:
            torch.cuda.manual_seed(random_seed)
            torch.backends.cudnn.deterministic = True

        # Pretraining of CAE only
        for epoch in range(0, 11):
            losses = list()
            for it in range(501):

                train_x, train_y, index = self.get_batch(self.image_data, self.batch_size,
                                                         train_label=self.image_label)

                train_x = torch.Tensor(train_x).to(self.device)
                train_x = train_x.reshape((-1, 1, self.height, self.width))
                optimizer.zero_grad()
                x_p = cae(train_x)

                loss = self.loss_func(x_p, train_x)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            print('Pretraining Epoch: {} Loss: {:.6f}'.format(
                      epoch, sum(losses)/len(losses)))

        # Todo: Why is optimizer initialized a second time?
        optimizer = torch.optim.RMSprop(params=model_params, lr=0.01, weight_decay=0.0)

        # Full model training
        for epoch in range(0, 11):

            losses = list()
            losses2 = list()

            train_x, train_y, index = self.get_batch(self.image_data, self.batch_size,
                                                     train_label=self.image_label)

            train_x = torch.Tensor(train_x).to(self.device)
            train_x = train_x.reshape((-1, 1, self.height, self.width))

            x_p = cae(train_x)
            features = clust(x_p)
            # Normalization of clustering features
            features = functional.normalize(features, p=2, dim=-1)
            # Another normalization !?
            features = features / features.norm(dim=1)[:, None]
            # Similarity as defined in formula 2 of the paper
            sim_mat = torch.matmul(features, torch.transpose(features, 0, 1))

            for it in range(31):

                train_x, train_y, index = self.get_batch(self.image_data, self.batch_size,
                                                         train_label=self.image_label)

                train_x = torch.Tensor(train_x).to(self.device)
                train_x = train_x.reshape((-1, 1, self.height, self.width))

                optimizer.zero_grad()
                x_p = cae(train_x)

                loss1 = self.loss_func(x_p, train_x)

                features = clust(x_p)
                # Feature normalization
                features = functional.normalize(features, p=2, dim=-1)
                features = features / features.norm(dim=1)[:, None]
                # Similarity computation as defined in formula 2 of the paper
                sim_mat = torch.matmul(features, torch.transpose(features, 0, 1))

                sim_numpy = sim_mat.cpu().detach().numpy()
                # Get all sim values from the batch excluding the diagonal
                # Todo: why not complete batch size?
                tmp2 = [sim_numpy[i][j] for i in range(0, self.batch_size-1)
                        for j in range(self.batch_size-1) if i != j]
                # Compute upper and lower percentiles according to uu & ll
                ub = np.percentile(tmp2, uu)
                lb = np.percentile(tmp2, ll)

                # Todo: Figure out pseudo labeling function
                # What it should do:
                # 2. compute A -> 1 if x_i element KNN(X_j) else A'
                # 3. compute pos_loc&neg_loc -> matrices (0|1) indicating whether they belong to same/different cluster
                # However knn should be with x instead of y (see formula 4)
                pos_loc, neg_loc = pseudo_labeling(ub=ub, lb=lb, sim=sim_numpy, index=index, knn=self.KNN,
                                                   knn_adj=self.knn_adj)
                pos_loc = pos_loc.to(self.device)
                neg_loc = neg_loc.to(self.device)

                pos_entropy = torch.mul(-torch.log(torch.clip(sim_mat, 1e-10, 1)), pos_loc)
                neg_entropy = torch.mul(-torch.log(torch.clip(1-sim_mat, 1e-10, 1)), neg_loc)

                loss2 = pos_entropy.sum()/pos_loc.sum() + neg_entropy.sum()/neg_loc.sum()

                loss = 1000*loss1 + loss2

                losses.append(loss1.item())
                losses2.append(loss2.item())
                loss.backward()
                optimizer.step()
                loss_list.append(sum(losses)/len(losses))

            uu = uu - 1
            ll = ll + 4
            print('Training Epoch: {} Loss: {:.6f}'.format(
                epoch, sum(losses) / len(losses)))
        return cae, clust

    def inference(self, cae, clust):
        with torch.no_grad():
            pred_label = list()

            test_x = torch.Tensor(self.image_data).to(self.device)
            test_x = test_x.reshape((-1, 1, self.height, self.width))

            x_p = cae(test_x)
            psuedo_label = clust(x_p)

            psuedo_label = torch.argmax(psuedo_label, dim=1)
            pred_label.extend(psuedo_label.cpu().detach().numpy())
            pred_label = np.array(pred_label)

            return pred_label

            # for it in range(len(self.image_data)//self.batch_size):
            #
            #     train_x, train_y = self.get_batch_sequential(self.image_data, self.image_label, self.sampleN, it)
            #
            #     train_x = torch.Tensor(train_x).to(self.device)
            #     train_x = train_x.reshape((-1, 1, self.height, self.width))
            #
            #     x_p = cae(train_x)
            #     psuedo_label = clust(x_p)
            #
            #     psuedo_label = torch.argmax(psuedo_label, dim=1)
            #     pred_label.extend(psuedo_label.cpu().detach().numpy())
            #
            # pred_label = np.array(pred_label)
            # acc = clustering_acc(train_y, pred_label)
            #
            # nmi = NMI(train_y, pred_label)
            # ari = ARI(train_y, pred_label)
            # print('testing NMI, ARI, ACC is %f, %f, %f.' % (nmi, ari, acc))

            # return nmi, ari, acc, pred_label

    # def tsne_viz(self, pred_label):
    #     x = np.reshape(self.image_data, (-1, self.height*self.width))
    #     tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    #     # t0 = time()
    #     x_tsne = tsne.fit_transform(x)
    #     print('plot embedding')
    #     plt.figure(figsize=(5, 3.5))
    #     sns.scatterplot(
    #         x=x_tsne[:, 0], y=x_tsne[:, 1],
    #         hue=pred_label,
    #         palette=sns.color_palette("hls", self.num_cluster),
    #         legend="full"
    #     )
    #     plt.xlabel('TSNE 1')
    #     plt.ylabel('TSNE 2')
