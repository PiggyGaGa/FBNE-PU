import torch
import torch.nn as nn
from torch.optim import Adam
from PNModel import PNModel
from sklearn import metrics
import numpy as np

class BPPN(object):
    def __init__(self, train_loader, test_loader, input_dim, output_dim, args):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        torch.manual_seed(args.seed)
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.model = PNModel(input_dim, output_dim).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=0.005)

    def train(self, epoch):
        self.model.train()
        tr_loss = 0
        temp_data=0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.type(torch.float))

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, target.type(torch.long))
            tr_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            temp_data += len(data)
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, temp_data, len(self.train_loader.dataset),
                           100. * temp_data / len(self.train_loader.dataset), loss.item()))
        print("Train loss: ", tr_loss)

    def test(self):
        self.model.eval()
        y_target = None
        y_proba=None
        y_pred=None
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                target = target.long()
                output = self.model(data)
                proba = output.detach().numpy()[:, 1].reshape(-1)
                pred = output.data.max(1)[1]
                if y_pred is None:
                    y_pred=pred
                else:
                    y_pred=np.hstack((y_pred,pred))
                if y_target is None:
                    y_target = target.detach().numpy().reshape(-1)
                else:
                    y_target = np.hstack((y_target, target.detach().numpy().reshape(-1)))

                if y_proba is None:
                    y_proba = proba
                else:
                    y_proba = np.hstack((y_proba, proba))
            acc=metrics.accuracy_score(y_target,y_pred)
            precision=metrics.precision_score(y_target,y_pred)
            recall=metrics.recall_score(y_target,y_pred)
            f1=metrics.f1_score(y_target,y_pred)

        print("pre:{},rec:{},f1:{},acc:{}".format(precision, recall, f1, acc))
        return precision, recall, f1, acc

    def run_train(self):
        for epoch in range(1, self.args.num_train_epochs + 1):
            self.train(epoch)

    def run_test(self):
        return self.test()

