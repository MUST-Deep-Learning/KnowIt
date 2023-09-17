import numpy as np

from data.prepared_dataset import PreparedDataset

class PreparedClassificationDataset(PreparedDataset):

    def __init__(self, name: str,
                 split_method: str,
                 split_portions: tuple,
                 scaling_method: str,
                 sampling_method: str,
                 seed: int,
                 scaling_mode: str,
                 batch_size: int,
                 shuffle_train: bool):


        super().__init__(name, split_method, split_portions,
                         scaling_method, sampling_method, seed,
                         scaling_mode, batch_size, shuffle_train)

        self.get_classes()


    def get_classes(self):

        the_data = self.get_the_data()
        self.class_set = set()
        for i in self.instances:
            for s in the_data[i]:
                unique_entries = np.unique(s['y'][~np.isnan(s['y'])], axis=0)
                unique_entries = [u for u in unique_entries]
                self.class_set.update(set(unique_entries))
        self.class_set = list(self.class_set)


# ---------------------DEBUGGING----------------------------------------------------------------------------------------

# import torch
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False  # if true => very slow
# train_device = torch.device('cpu')
# torch.seed()
#
# DS = PreparedClassificationDataset(name='penguin_dummy', split_method='instance-random',
#                      split_portions=(0.6, 0.2, 0.2),
#                      scaling_method='z-norm', sampling_method='random',
#                      seed=1, scaling_mode='data_feature',
#                      batch_size=32, shuffle_train=True)
#
# torch.manual_seed(DS.seed)
#
# train_loader = DS.get_dataloader('train')
# valid_loader = DS.get_dataloader('valid')
#
# from torch.optim import SGD, Adam, RAdam
# from torch.nn import MSELoss, CrossEntropyLoss
# from model.custom_model_TEMPLATE import Model
# in_dim = int(np.prod(DS.in_shape))
# out_dim = int(np.prod(DS.out_shape))
# my_mlp = Model(in_dim, out_dim, 'classification',
#                width=512, depth=5, dropout=0.3, batchnorm=False)
# my_mlp.to(train_device)
# optimizer = Adam(my_mlp.parameters(), lr=0.0001)
# loss_fn = MSELoss()
# max_epoch = 500
#
# epoch_losses = []
# for epoch in range(max_epoch):
#     train_batch_losses = []
#     for batch_id, batch in enumerate(train_loader):
#         x = batch['x']
#         y = batch['y']
#         x = x.view(DS.batch_size, in_dim)
#         x = x.to(train_device)
#         y = y.to(train_device)
#         optimizer.zero_grad()
#         h_n = my_mlp.forward(x)
#         loss = loss_fn(h_n, y)
#         loss.backward()
#         optimizer.step()
#         train_batch_losses.append(loss.item())
#     train_batch_losses = np.array(train_batch_losses)
#     train_epoch_loss = (np.mean(train_batch_losses), np.std(train_batch_losses))
#
#     my_mlp.eval()
#     valid_batch_losses = []
#     for batch_id, batch in enumerate(valid_loader):
#         x = batch['x']
#         y = batch['y']
#         x = x.view(x.shape[0], in_dim)
#         x = x.to(train_device)
#         y = y.to(train_device)
#         h_n = my_mlp.forward(x)
#         loss = loss_fn(h_n, y)
#         valid_batch_losses.append(loss.item())
#     valid_batch_losses = np.array(valid_batch_losses)
#     valid_epoch_loss = (np.mean(valid_batch_losses), np.std(valid_batch_losses))
#     my_mlp.train()
#
#     print(str(epoch) + '.   Train: ' + str(train_epoch_loss[0]) + ', Valid: ' + str(valid_epoch_loss[0]))
#     epoch_losses.append([train_epoch_loss[0], valid_epoch_loss[0]])
#
# epoch_losses = np.array(epoch_losses)
# import matplotlib.pyplot as plt
#
# plt.plot(epoch_losses[:, 0], label='train MSE', color='green', alpha=0.7)
# plt.plot(epoch_losses[:, 1], label='valid MSE', color='red', alpha=0.7)
# plt.legend()
# plt.xlabel('Epoch')
# plt.show()
# plt.close()
#
#
# exit(101)

# ---------------------DEBUGGING----------------------------------------------------------------------------------------