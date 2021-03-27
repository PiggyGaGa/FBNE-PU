import torch
from torch.optim import Adam

from loss import PULoss
from PUModel import PUModel


class NNPU(object):
    def __init__(self, train_loader, input_dim, output_dim, prior, args):
        self.train_loader = train_loader
        self.prior = prior
        self.args = args
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        torch.manual_seed(args.seed)
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.model = PUModel(input_dim, output_dim).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=0.005)

    def train(self, epoch):
        self.model.train()
        tr_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.type(torch.float))
            loss_fct = PULoss(prior=self.prior,nnPU=True)
            loss = loss_fct(output.view(-1), target.type(torch.float))
            tr_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))

        return tr_loss


    def run_train(self):
        if not self.args.do_train and not self.args.do_eval:
            raise ValueError("At least one of `do_train` or `do_eval` must be True.")
        if self.args.do_train:
            for epoch in range(1, self.args.num_train_epochs + 1):
                self.train(epoch)
        elif self.args.do_eval:
            # self.test()
            print("not do train")
        return self.model

