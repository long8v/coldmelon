import imageio
import numpy as np
import torch
import time
from torch.nn import functional as F
from torchvision.utils import make_grid
from sklearn.preprocessing import OneHotEncoder

EPS = 1e-12
class Trainer():
    def _idcg(self, l):
        return sum((1.0 / np.log(i + 2) for i in range(l)))

    def __init__(self, model, optimizer, nsongs, print_loss_every=50, record_loss_every=5,
                 use_cuda=False):
        """
        Class to handle training of model.

        Parameters
        ----------
        model : jointvae.models.VAE instance

        optimizer : torch.optim.Optimizer instance

        print_loss_every : int
            Frequency with which loss is printed during training.

        record_loss_every : int
            Frequency with which loss is recorded during training.

        use_cuda : bool
            If True moves model and training to GPU.
        """
        self.model = model
        self.optimizer = optimizer
        self.print_loss_every = print_loss_every
        self.record_loss_every = record_loss_every
        self.use_cuda = use_cuda
        self.nsongs = nsongs
        self._idcgs = [self._idcg(i) for i in range(251)]

        if self.use_cuda:
            self.model.cuda()

        # Initialize attributes
        self.num_steps = 0
        self.batch_size = None
        self.losses = {'total_loss': [],
                       'ndcg': []}


    def train(self, data_loader, epochs=10, save_training_gif=None):
        """
        Trains the model.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader

        epochs : int
            Number of epochs to train the model for.

        save_training_gif : None or tuple (string, Visualizer instance)
            If not None, will use visualizer object to create image of samples
            after every epoch and will save gif of these at location specified
            by string. Note that string should end with '.gif'.
        """
        if save_training_gif is not None:
            training_progress_images = []

        self.batch_size = data_loader.batch_size
        self.model.train()
        for epoch in range(epochs):
            epoch_train_start = time.time()
            mean_epoch_loss, mean_epoch_ndcg, mean_epoch_loss_recon, mean_epoch_loss_embed = self._train_epoch(data_loader)
            print('Epoch: {} Average loss: {:.2f} Average ndcg: {:.2f} Average recon_loss: {:.2f} Average embed_loss: {:.2f} Training time: {:.2f}'.format(epoch + 1,
                                                          mean_epoch_loss, mean_epoch_ndcg, mean_epoch_loss_recon, mean_epoch_loss_embed,
                                                          time.time() - epoch_train_start ))
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': self.batch_size * self.model.num_pixels * mean_epoch_loss,
                        'time': time.time() - epoch_train_start
                        }, "./output_model_epoch_"+str(epoch)+".pth")

                                                          
    def _train_epoch(self, data_loader):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
        """
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        epoch_loss = 0.
        epoch_ndcg = 0.
        epoch_loss_recon = 0.
        epoch_loss_embed = 0.
        print_every_loss = 0.
        print_every_ndcg = 0.
        print_every_loss_recon = 0.
        print_every_loss_embed = 0.
        # Keeps track of loss to print every
        # self.print_loss_every
        for batch_idx, datas in enumerate(data_loader):
            # data, label = datas['input'], datas['label']
            # data = data.to(device)
            # label = label.to(device)
            iter_loss, ndcg_loss, iter_loss_recon, iter_loss_embed = self._train_iteration(datas)
            epoch_loss += iter_loss
            epoch_ndcg += ndcg_loss
            epoch_loss_recon += iter_loss_recon
            epoch_loss_embed += iter_loss_embed
            print_every_loss += iter_loss
            print_every_ndcg += ndcg_loss
            print_every_loss_recon += iter_loss_recon
            print_every_loss_embed += iter_loss_embed
            # Print loss info every self.print_loss_every iteration
            if batch_idx % self.print_loss_every == 0:
                if batch_idx == 0:
                    mean_loss = print_every_loss
                    mean_ndcg = print_every_ndcg
                    mean_loss_recon = print_every_loss_recon
                    mean_loss_embed = print_every_loss_embed
                else:
                    mean_loss = print_every_loss / self.print_loss_every
                    mean_ndcg = print_every_ndcg / self.print_loss_every
                    mean_loss_recon = print_every_loss_recon / self.print_loss_every
                    mean_loss_embed = print_every_loss_embed / self.print_loss_every
                print('{}/{}\tLoss: {:.9f}\tndcg: {:.9f} recon_loss: {:.9f} embed_loss: {:.9f}'.format(batch_idx * len(datas['input']),
                                                  len(data_loader.dataset),
                                                  mean_loss, mean_ndcg, mean_loss_recon, mean_loss_embed))
                print_every_loss = 0.
                print_every_ndcg = 0.
                print_every_loss_recon = 0.
                print_every_loss_embed = 0.
        # Return mean epoch loss
        return epoch_loss / len(data_loader.dataset), epoch_ndcg / len(data_loader.dataset), epoch_loss_recon / len(data_loader.dataset), epoch_loss_embed / len(data_loader.dataset)

    def _train_iteration(self, datas):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            A batch of data. Shape (N, C, H, W)
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_steps += 1

        data, label = datas['input'], datas['label']
        if self.use_cuda:
            data = data.to(device)
            # label = label.cuda()
        self.optimizer.zero_grad()
        #print("ww:", data.shape, label.shape)
        recon_batch, song_embeddings_100, song_embeddings_10_100, song_embeddings_0_10 = self.model(data)
        recon_batch = recon_batch.view(len(data),-1)
        #print("ww1:", recon_batch.shape)
        sorted, indices = torch.sort(recon_batch, descending=True)
        output_recon = torch.stack(list(map(lambda a: (indices.squeeze() == a).nonzero().squeeze(), range(200))))
        loss_recon = self._loss_function(label, recon_batch)
        loss_embed = self._loss_function_embed(song_embeddings_100, song_embeddings_10_100, song_embeddings_0_10)
        loss = loss_recon + loss_embed
        #print(label.shape, output_recon.shape, recon_batch.shape, indices.shape)
        #print(label, output_recon[:14])
        ndcg_loss = self._ndcg(label.reshape(-1),output_recon.cpu().detach().numpy())
        loss.backward()
        self.optimizer.step()

        train_loss = loss.item()
        return train_loss, ndcg_loss, loss_recon, loss_embed

    def _ndcg(self, gt, rec):
        dcg = 0.0
        #print(gt.shape,rec.shape)
        for i, r in enumerate(rec):
            if r in gt:
                dcg += 1.0 / np.log(i + 2)

        return dcg / self._idcgs[len(gt)]

    def _loss_function(self, data, recon_data):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_songs = self.nsongs
        # print(n_songs)
        enc = OneHotEncoder(handle_unknown='ignore')
        X = np.array(range(n_songs))
        X = X.reshape(len(X),-1)
        enc.fit(X)
        new_data = []
        for d in data:
            new_data.append(torch.FloatTensor(3*enc.transform(d.reshape(-1,1)).toarray().sum(axis=0)))
        new_data = torch.stack(new_data).cuda()
        # data = data.view(-1).cuda()
        total_loss = F.binary_cross_entropy_with_logits(recon_data, new_data)
        # print(total_loss)

        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['total_loss'].append(total_loss.item())

        # To avoid large losses normalise by number of pixels
        return total_loss

    def _loss_function_multi_margin(self, data, recon_data):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_songs = self.nsongs
        # print(n_songs)
        enc = OneHotEncoder(handle_unknown='ignore')
        X = np.array(range(n_songs))
        X = X.reshape(len(X),-1)
        enc.fit(X)
        #new_data = []
        # for d in data:
        #     torch.FloatTensor(enc.transform(d.reshape(-1,1)).toarray()
            # new_data.append(torch.FloatTensor(enc.transform(d.reshape(-1,1)).toarray().sum(axis=0)))
        # new_data = torch.stack(new_data).cuda()
        data = data.view(-1).cuda()
        # print(recon_data.shape, data.shape)
        total_loss = 0.
        for d in data:
            total_loss += F.multi_margin_loss(recon_data, d)
        #total_loss = F.binary_cross_entropy(recon_data, new_data)
        # print(total_loss)

        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['total_loss'].append(total_loss.item())

        # To avoid large losses normalise by number of pixels
        return total_loss / data.shape[0]

    def _loss_function_embed(self, song_embeddings_100, song_embeddings_10_100, song_embeddings_0_10):
        song_embedding_100_mean = []
        song_embedding_10_100_mean = []
        song_embedding_0_10_mean = []
        for song_embedding_100 in song_embeddings_100:
            mean_100 = song_embedding_100.mean(dim=0)
            for _ in range(song_embedding_100.shape[0]):
                song_embedding_100_mean.append(mean_100)
            song_embedding_100_mean = torch.stack(song_embedding_100_mean)
            loss_100 = F.mse_loss(song_embedding_100, song_embedding_100_mean)#, reduction = 'sum')

        for song_embedding_10_100 in song_embeddings_10_100:
            mean_10_100 = song_embedding_10_100.mean(dim=0)
            for _ in range(song_embedding_10_100.shape[0]):
                song_embedding_10_100_mean.append(mean_10_100)
            song_embedding_10_100_mean = torch.stack(song_embedding_10_100_mean)
            loss_10_100 = F.mse_loss(song_embedding_10_100, song_embedding_10_100_mean)#, reduction = 'sum') / song_embedding_10_100.shape[0]

        
        for song_embedding_0_10 in song_embeddings_0_10:
            mean_0_10 = song_embedding_0_10.mean(dim=0)
            for _ in range(song_embedding_0_10.shape[0]):
                song_embedding_0_10_mean.append(mean_0_10)
            song_embedding_0_10_mean = torch.stack(song_embedding_0_10_mean)
            loss_0_10 = F.mse_loss(song_embedding_0_10, song_embedding_0_10_mean)#, reduction = 'sum') / song_embedding_0_10.shape[0]

        return loss_100+loss_10_100+loss_0_10