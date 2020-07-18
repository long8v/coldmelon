import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

class MLP(nn.Module):

    def __init__(self, n_tags, n_songs, layers=[1000, 200, 1000], dropout=False, use_cuda = False):
        """
        Simple Feedforward network with Embeddings for users and items
        """
        super().__init__()
        assert (layers[0] % 2 == 0), "layers[0] must be an even number"
        self.__alias__ = "MLP {}".format(layers)
        self.__dropout__ = dropout
        self.num_pixels = n_songs

        # user and item embedding layers
        embedding_dim = layers[0]
        # self.tag_embedding = torch.nn.Embedding(n_tags, embedding_dim)
        self.song_embedding = torch.nn.Embedding(n_songs, embedding_dim)

        # list of weight matrices
        self.fc_layers = torch.nn.ModuleList()
        # hidden dense layers
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        # final prediction layer
        self.output_layer = torch.nn.Linear(layers[-1], n_songs)
        # self.output_layer = torch.nn.DataParallel(self.output_layer)

    def forward(self, feed_dict):
        # tags = feed_dict['tags_id']
        songs = feed_dict
        ratings = []
        #x = torch.cat([tag_embedding, song_embedding], 1)
        for d in songs:
            d = d[d!=self.num_pixels]
            song_embedding = self.song_embedding(d)
            song_embedding = song_embedding.mean(dim=0)
            for idx, _ in enumerate(range(len(self.fc_layers))):
                song_embedding = self.fc_layers[idx](song_embedding)
                song_embedding = F.leaky_relu(song_embedding)
                song_embedding = F.dropout(song_embedding,  p=self.__dropout__, training=self.training)
            logit = self.output_layer(song_embedding)
            rating = torch.sigmoid(logit)
            ratings.append(rating)
        ratings = torch.stack(ratings).cuda()
        return ratings

    def predict(self, feed_dict):
        # return the score, inputs and outputs are numpy arrays
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if type(feed_dict) != type(None):
            feed_dict = torch.from_numpy(feed_dict).to(dtype=torch.long, device=device)
        output_scores = self.forward(feed_dict)
        sorted, indices = torch.sort(output_scores,descending=True)
        output = torch.stack(list(map(lambda a: (indices.squeeze() == a).nonzero().squeeze(), range(100+feed_dict.shape[1]))))

        return output.cpu().detach().numpy()

    def get_alias(self):
        return self.__alias__



class MLP2(nn.Module):

    def __init__(self, n_tags, n_songs, layers=[1024, 512, 256, 128, 256, 512], dropout=False, use_cuda = False):
        """
        Simple Feedforward network with Embeddings for users and items
        """
        super().__init__()
        assert (layers[0] % 2 == 0), "layers[0] must be an even number"
        self.__alias__ = "MLP {}".format(layers)
        self.__dropout__ = dropout
        self.num_pixels = n_songs

        # user and item embedding layers
        embedding_dim = layers[0]
        # self.tag_embedding = torch.nn.Embedding(n_tags, embedding_dim)
        self.song_embedding = torch.nn.Embedding(n_songs, embedding_dim)

        # list of weight matrices
        self.fc_layers = torch.nn.ModuleList()
        # hidden dense layers
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        # final prediction layer
        self.output_layer = torch.nn.Linear(layers[-1], n_songs)
        # self.output_layer = torch.nn.DataParallel(self.output_layer)
    '''
        encoder_layers = [
            nn.Conv2d(self.img_size[0], 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (4, 4), stride=2, padding=1),
            nn.ReLU()
        ]
        encoder_layers += [
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (4, 4), stride=2, padding=1),
            nn.ReLU()
        ]

        self.img_to_features = nn.Sequential(*encoder_layers)
    '''

    def forward(self, feed_dict):
        # tags = feed_dict['tags_id']
        songs = feed_dict
        ratings = []
        #x = torch.cat([tag_embedding, song_embedding], 1)
        for d in songs:
            d = d[d!=self.num_pixels]
            song_embedding = self.song_embedding(d)
            song_embedding = song_embedding.mean(dim=0)
            for idx, _ in enumerate(range(len(self.fc_layers))):
                song_embedding = self.fc_layers[idx](song_embedding)
                if idx == 1 or idx == 3 or idx == 5 or idx == 8 or idx == 10 or idx == 12 or idx == 14:
                    tmp = song_embedding
                # if idx == 1:
                #     tmp1 = x
                if idx == 3 or idx == 5 or idx == 7 or idx == 10 or idx == 12 or idx == 14 or idx == 16:
                    song_embedding = song_embedding + tmp
                # if idx == len(self.fc_layers)-1:
                #     x = x + tmp0
                song_embedding = F.leaky_relu(song_embedding)
                song_embedding = F.dropout(song_embedding,  p=self.__dropout__, training=self.training)
            logit = self.output_layer(song_embedding)
            rating = torch.sigmoid(logit)
            ratings.append(rating)
        ratings = torch.stack(ratings).cuda()
        return ratings

    def predict(self, feed_dict):
        # return the score, inputs and outputs are numpy arrays
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if type(feed_dict) != type(None):
            feed_dict = torch.from_numpy(feed_dict).to(dtype=torch.long, device=device)
        output_scores = self.forward(feed_dict)
        sorted, indices = torch.sort(output_scores,descending=True)
        output = torch.stack(list(map(lambda a: (indices.squeeze() == a).nonzero().squeeze(), range(100+feed_dict.shape[1]))))

        return output.cpu().detach().numpy()

    def get_alias(self):
        return self.__alias__

class MLP3(nn.Module):

    def __init__(self, n_tags, n_songs, embed_dim, dropout=False, use_cuda = False):
        """
        Simple Feedforward network with Embeddings for users and items
        """
        super().__init__()
        # assert (layers[0] % 2 == 0), "layers[0] must be an even number"
        self.__alias__ = "MLP {}".format(embed_dim)
        self.__dropout__ = dropout
        self.num_pixels = n_songs

        # user and item embedding layers
        embedding_dim = embed_dim
        # self.tag_embedding = torch.nn.Embedding(n_tags, embedding_dim)
        self.song_embedding = torch.nn.Embedding(n_songs, embedding_dim)

        # list of weight matrices
        # self.fc_layers = torch.nn.ModuleList()
        # # hidden dense layers
        # for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
        #     self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        # final prediction layer
        self.output_layer = torch.nn.Linear(embed_dim, n_songs)
        # self.output_layer = torch.nn.DataParallel(self.output_layer)
    '''
        encoder_layers = [
            nn.Conv2d(self.img_size[0], 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (4, 4), stride=2, padding=1),
            nn.ReLU()
        ]
        encoder_layers += [
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (4, 4), stride=2, padding=1),
            nn.ReLU()
        ]

        self.img_to_features = nn.Sequential(*encoder_layers)
    '''

    def forward(self, feed_dict):
        # tags = feed_dict['tags_id']
        songs = feed_dict
        # tag_embedding = self.tag_embedding(tags)
        # print("1 : ",songs.shape)
        # print("song shape1: ", songs.shape)
        ratings = []
        for d in songs:
            d = d[d!=self.num_pixels]
            song_embedding = self.song_embedding(d)
            song_embedding = song_embedding.mean(dim=0)
            logit = self.output_layer(song_embedding)
            rating = torch.sigmoid(logit)
            ratings.append(rating)
        ratings = torch.stack(ratings).cuda()
        return ratings

    # def predict(self, feed_dict):
    #     # return the score, inputs and outputs are numpy arrays
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     for key in feed_dict:
    #         if type(feed_dict) != type(None):
    #             feed_dict = torch.from_numpy(
    #                 feed_dict).to(dtype=torch.long, device=device)
    #     output_scores = self.forward(feed_dict)
    #     sorted, indices = torch.sort(output_scores,descending=True)
    #     output = torch.stack(list(map(lambda a: (indices == a).nonzero().squeeze(), range(100))))

    #     return output.cpu().detach().numpy()
    def predict(self, feed_dict):
        # return the score, inputs and outputs are numpy arrays
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if type(feed_dict) != type(None):
            feed_dict = torch.from_numpy(feed_dict).to(dtype=torch.long, device=device)
        output_scores = self.forward(feed_dict)
        sorted, indices = torch.sort(output_scores,descending=True)
        output = torch.stack(list(map(lambda a: (indices == a).nonzero().squeeze(), range(100))))

        return output.cpu().detach().numpy()

    def get_alias(self):
        return self.__alias__

class MLP4(nn.Module):

    def __init__(self, n_tags, n_songs, layers=[1024, 512, 256, 128, 256, 512], dropout=False, use_cuda = False):
        """
        Simple Feedforward network with Embeddings for users and items
        """
        super().__init__()
        assert (layers[0] % 2 == 0), "layers[0] must be an even number"
        self.__alias__ = "MLP {}".format(layers)
        self.__dropout__ = dropout
        self.num_pixels = n_songs

        # user and item embedding layers
        embedding_dim = layers[0]
        # self.tag_embedding = torch.nn.Embedding(n_tags, embedding_dim)
        self.song_embedding = torch.nn.Embedding(n_songs, embedding_dim)

        # list of weight matrices
        self.fc_layers = torch.nn.ModuleList()
        # hidden dense layers
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        # final prediction layer
        self.output_layer = torch.nn.Linear(layers[-1], n_songs)
        # self.output_layer = torch.nn.DataParallel(self.output_layer)


    def forward(self, feed_dict):
        # tags = feed_dict['tags_id']
        songs = feed_dict
        ratings = []
        #x = torch.cat([tag_embedding, song_embedding], 1)
        for d in songs:
            d = d[d!=self.num_pixels]
            song_embedding = self.song_embedding(d)
            song_embedding = song_embedding.mean(dim=0)
            for idx, _ in enumerate(range(len(self.fc_layers))):
                song_embedding = self.fc_layers[idx](song_embedding)
                if idx == 1 or idx == 3 or idx == 5 or idx == 8 or idx == 10 or idx == 12 or idx == 14:
                    tmp = song_embedding
                # if idx == 1:
                #     tmp1 = x
                if idx == 3 or idx == 5 or idx == 7 or idx == 10 or idx == 12 or idx == 14 or idx == 16:
                    song_embedding = song_embedding + tmp
                # if idx == len(self.fc_layers)-1:
                #     x = x + tmp0
                song_embedding = F.leaky_relu(song_embedding)
                song_embedding = F.dropout(song_embedding,  p=self.__dropout__, training=self.training)
            logit = self.output_layer(song_embedding)
            rating = torch.sigmoid(logit)
            ratings.append(rating)
        ratings = torch.stack(ratings).cuda()
        return ratings

    def predict(self, feed_dict):
        # return the score, inputs and outputs are numpy arrays
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if type(feed_dict) != type(None):
            feed_dict = torch.from_numpy(feed_dict).to(dtype=torch.long, device=device)
        output_scores = self.forward(feed_dict)
        sorted, indices = torch.sort(output_scores,descending=True)
        output = torch.stack(list(map(lambda a: (indices.squeeze() == a).nonzero().squeeze(), range(100+feed_dict.shape[1]))))

        return output.cpu().detach().numpy()

    def get_alias(self):
        return self.__alias__

class MLP5(nn.Module):

    def __init__(self, n_tags, n_songs_100, n_songs_10_100, n_songs_0_10, layers=[1000, 200, 1000], dropout=False, use_cuda = False):
        """
        Simple Feedforward network with Embeddings for users and items
        """
        super().__init__()
        assert (layers[0] % 2 == 0), "layers[0] must be an even number"
        self.__alias__ = "MLP {}".format(layers)
        self.__dropout__ = dropout
        self.num_pixels = n_songs_100 + n_songs_10_100 + n_songs_0_10

        # user and item embedding layers
        self.embedding_dim_100 = int(layers[0]/2)
        self.embedding_dim_10_100 = int(layers[0]/4)
        self.embedding_dim_0_10 = layers[0]- int(layers[0]/2) - int(layers[0]/4)
        # self.tag_embedding = torch.nn.Embedding(n_tags, embedding_dim)
        self.song_embedding_100 = torch.nn.Embedding(n_songs_100, self.embedding_dim_100)
        self.song_embedding_10_100 = torch.nn.Embedding(n_songs_10_100, self.embedding_dim_10_100)
        self.song_embedding_0_10 = torch.nn.Embedding(n_songs_0_10, self.embedding_dim_0_10)

        # list of weight matrices
        self.fc_layers = torch.nn.ModuleList()
        # hidden dense layers
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        # final prediction layer
        self.output_layer = torch.nn.Linear(layers[-1], self.num_pixels)
        # self.output_layer = torch.nn.DataParallel(self.output_layer)

    def forward(self, feed_dict):
        # tags = feed_dict['tags_id']
        songs = feed_dict
        ratings = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for d in songs:
            tmp = []
            d_100 = d[d[:,0]==0][:,1]
            if len(d_100) != 0:
                tmp.append(self.song_embedding_100(d_100).mean(dim=0))
            else:
                tmp.append(torch.FloatTensor(np.zeros(self.embedding_dim_100)).to(device))
            d_10_100 = d[d[:,0]==1][:,1]
            if len(d_10_100) != 0:
                tmp.append(self.song_embedding_10_100(d_10_100).mean(dim=0))
            else:
                tmp.append(torch.FloatTensor(np.zeros(self.embedding_dim_10_100)).to(device))
            d_0_10 = d[d[:,0]==2][:,1]
            if len(d_0_10) != 0:
                tmp.append(self.song_embedding_0_10(d_0_10).mean(dim=0))
            else:
                tmp.append(torch.FloatTensor(np.zeros(self.embedding_dim_0_10)).to(device))


            #print(d_100.shape, d_10_100.shape, d_0_10.shape, len(tmp))
            #print(d_100, d_10_100, d_0_10)
            song_embedding = torch.cat(tmp)
            #print(song_embedding.shape)
            for idx, _ in enumerate(range(len(self.fc_layers))):
                song_embedding = self.fc_layers[idx](song_embedding)
                song_embedding = F.relu(song_embedding)
                song_embedding = F.dropout(song_embedding,  p=self.__dropout__, training=self.training)
            logit = self.output_layer(song_embedding)
            rating = torch.sigmoid(logit)
            ratings.append(rating)
        ratings = torch.stack(ratings).cuda()
        return ratings

    def predict(self, feed_dict):
        # return the score, inputs and outputs are numpy arrays
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #if type(feed_dict) != type(None):
        #feed_dict = torch.LongTensor(feed_dict).cuda()
        output_scores = self.forward(feed_dict)
        sorted, indices = torch.sort(output_scores,descending=True)
        output = torch.stack(list(map(lambda a: (indices.squeeze() == a).nonzero().squeeze(), range(200))))

        return output.cpu().detach().numpy()

    def get_alias(self):
        return self.__alias__

class MLP5(nn.Module):

    def __init__(self, n_tags, n_songs_100, n_songs_10_100, n_songs_0_10, layers=[1000, 200, 1000], dropout=False, use_cuda = False):
        """
        Simple Feedforward network with Embeddings for users and items
        """
        super().__init__()
        assert (layers[0] % 2 == 0), "layers[0] must be an even number"
        self.__alias__ = "MLP {}".format(layers)
        self.__dropout__ = dropout
        self.num_pixels = n_songs_100 + n_songs_10_100 + n_songs_0_10

        # user and item embedding layers
        self.embedding_dim_100 = int(layers[0]/2)
        self.embedding_dim_10_100 = int(layers[0]/4)
        self.embedding_dim_0_10 = layers[0]- int(layers[0]/2) - int(layers[0]/4)
        # self.tag_embedding = torch.nn.Embedding(n_tags, embedding_dim)
        self.song_embedding_100 = torch.nn.Embedding(n_songs_100, self.embedding_dim_100)
        self.song_embedding_10_100 = torch.nn.Embedding(n_songs_10_100, self.embedding_dim_10_100)
        self.song_embedding_0_10 = torch.nn.Embedding(n_songs_0_10, self.embedding_dim_0_10)

        # list of weight matrices
        self.fc_layers = torch.nn.ModuleList()
        # hidden dense layers
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        # final prediction layer
        self.output_layer = torch.nn.Linear(layers[-1], self.num_pixels)
        # self.output_layer = torch.nn.DataParallel(self.output_layer)

    def forward(self, feed_dict):
        # tags = feed_dict['tags_id']
        songs = feed_dict
        ratings = []
        song_embeddings_100 = []
        song_embeddings_10_100 = []
        song_embeddings_0_10 = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        zeros_100 = torch.FloatTensor(np.zeros(self.embedding_dim_100)).to(device)
        zeros_10_100 = torch.FloatTensor(np.zeros(self.embedding_dim_10_100)).to(device)
        zeros_0_10 = torch.FloatTensor(np.zeros(self.embedding_dim_0_10)).to(device)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for d in songs:
            tmp = []
            d_100 = d[d[:,0]==0][:,1]
            if len(d_100) != 0:
                embed_100 = self.song_embedding_100(d_100)
                song_embeddings_100.append(embed_100)
                tmp.append(embed_100.mean(dim=0))
            else:
                tmp.append(zeros_100)
                song_embeddings_100.append(zeros_100)

            d_10_100 = d[d[:,0]==1][:,1]
            if len(d_10_100) != 0:
                embed_10_100 = self.song_embedding_10_100(d_10_100)
                song_embeddings_10_100.append(embed_10_100)
                tmp.append(embed_10_100.mean(dim=0))
            else:
                tmp.append(zeros_10_100)
                song_embeddings_10_100.append(zeros_10_100)

            d_0_10 = d[d[:,0]==2][:,1]
            if len(d_0_10) != 0:
                embed_0_10 = self.song_embedding_0_10(d_0_10)
                song_embeddings_0_10.append(embed_0_10)
                tmp.append(embed_0_10.mean(dim=0))
            else:
                tmp.append(zeros_0_10)
                song_embeddings_0_10.append(zeros_0_10)

            song_embedding = torch.cat(tmp)
            # print(song_embedding.shape)
            for idx, _ in enumerate(range(len(self.fc_layers))):
                song_embedding = self.fc_layers[idx](song_embedding)
                song_embedding = F.relu(song_embedding)
                song_embedding = F.dropout(song_embedding,  p=self.__dropout__, training=self.training)
            logit = self.output_layer(song_embedding)
            # rating = torch.sigmoid(logit)
            ratings.append(logit)
            # song_embeddings.append(song_embedding)
        ratings = torch.stack(ratings).cuda()
        # song_embeddings = torch.stack(song_embeddings).cuda()
        return ratings, song_embeddings_100, song_embeddings_10_100, song_embeddings_0_10

    def predict(self, feed_dict):
        # return the score, inputs and outputs are numpy arrays
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #if type(feed_dict) != type(None):
        #feed_dict = torch.LongTensor(feed_dict).cuda()
        output_scores, song_embeddings_100, song_embeddings_10_100, song_embeddings_0_10 = self.forward(feed_dict)
        sorted, indices = torch.sort(output_scores,descending=True)
        output = torch.stack(list(map(lambda a: (indices.squeeze() == a).nonzero().squeeze(), range(200))))

        return output.cpu().detach().numpy()

    def get_alias(self):
        return self.__alias__
