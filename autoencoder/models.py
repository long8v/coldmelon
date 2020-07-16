import torch
from torch import nn, optim
from torch.nn import functional as F

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
        self.tag_embedding = torch.nn.Embedding(n_tags, embedding_dim)
        self.song_embedding = torch.nn.Embedding(n_songs, embedding_dim)

        # list of weight matrices
        self.fc_layers = torch.nn.ModuleList()
        # hidden dense layers
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        # final prediction layer
        self.output_layer = torch.nn.Linear(layers[-1], n_songs)
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
        song_embedding = self.song_embedding(songs)
        tag_embedding = tag_embedding.mean(dim=0)
        song_embedding = song_embedding.mean(dim=0)
        # concatenate user and item embeddings to form input      
        #x = torch.cat([tag_embedding, song_embedding], 1)
        x = song_embedding
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = F.relu(x)
            x = F.dropout(x,  p=self.__dropout__, training=self.training)
        logit = self.output_layer(x)
        rating = torch.sigmoid(logit)
        return rating

    def predict(self, feed_dict):
        # return the score, inputs and outputs are numpy arrays
        for key in feed_dict:
            if type(feed_dict[key]) != type(None):
                feed_dict[key] = torch.from_numpy(
                    feed_dict[key]).to(dtype=torch.long, device=device)
        output_scores = self.forward(feed_dict)

        return output_scores.cpu().detach().numpy()

    def get_alias(self):
        return self.__alias__