import torch
from torch import nn, optim
import torch.nn.functional as F
from torchsummary import summary
from torchvision import datasets, transforms
from torch.autograd import Variable

NUM_PSEUDOINPUTS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.5
NUM_EPOCHS = 5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # pseudoinput layer
        self.ps_layer = nn.Linear(NUM_PSEUDOINPUTS, 784)
        
        # encoder layers  
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.conv2 = nn.Conv2d(3, 6, 3)
        self.fc1 = nn.Linear(150, 84)
        self.fc2 = nn.Linear(84, 10)
        self.fc_mu = nn.Linear(10, 2)
        self.fc_logvar = nn.Linear(10, 2)
        
        # decoder layers
        self.fc4 = nn.Linear(2, 400)
        self.fc5 = nn.Linear(400, 784)
        # USE TRANSPOSE CONV. LAYERS INSTEAD??
        
    def get_ps(self):
        i = torch.eye(NUM_PSEUDOINPUTS)
        ps = F.sigmoid(self.ps_layer(i))
        return ps.view(NUM_PSEUDOINPUTS, 1, 28, 28)
    
    def encode(self, x):        
        # moves through encoding layers
        x = F.sigmoid(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2) # makes it [3, 13, 13]
        x = F.sigmoid(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2) # makes it [6, 5, 5]
        x = x.view(-1, num_flat_features(x)) # makes it [150]
        x = F.sigmoid(self.fc1(x)) # makes it [84]
        x = F.sigmoid(self.fc2(x)) # makes it [10]
        mu = F.sigmoid(self.fc_mu(x)) # makes it [2]
        logvar = F.sigmoid(self.fc_logvar(x)) # makes it [2]        
        # returns mu and logvar
        return mu, logvar
        
    def reparam(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps * std + mu
        else:
            return mu
     
    def decode(self, z):
        x = F.leaky_relu(self.fc4(z)) # makes it [400]
        x = F.leaky_relu(self.fc5(x)) # makes it [400]
        return F.sigmoid(x)
    
    def forward(self, x):        
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar        

    def loss_function(self, recon_x, x, mu, logvar):
        # reconstruction error
        recon_error = F.binary_cross_entropy(recon_x, x.view(-1, num_flat_features(x)))
        
        # KL divergence
        eps = torch.randn_like(mu)
        std_dev = torch.exp(2 * logvar)
        z_sample = mu + eps * std_dev
        logpdf_posterior = diagonal_normal_logpdf(z_sample, mu, logvar)
        
        pseudoinputs = self.get_ps()
        ps_mu, ps_logvar = self.encode(pseudoinputs)
        z_sample = z_sample.view(-1, 1, 2)
        z_sample = z_sample.repeat(1, NUM_PSEUDOINPUTS, 1)
        ps_mu = ps_mu.repeat(z_sample.size()[0], 1, 1)
        ps_logvar = ps_logvar.repeat(z_sample.size()[0], 1, 1)
        logpdf_prior = diagonal_normal_logpdf(z_sample, ps_mu, ps_logvar)
        logpdf_prior = logpdf_prior.mean(1)
        
        kldiv_loss = torch.mean(logpdf_posterior - logpdf_prior)
        
        # return sum of reconstruction error and kl divergence
        if not self.training:
            print('Reconstruction Loss: {}, KL Divergence Loss {}'.format(recon_error.item(), kldiv_loss.item()))
        return recon_error + kldiv_loss

def diagonal_normal_logpdf(x, mu, logvar):
    return -0.5 * (logvar + torch.pow(x - mu, 2) / torch.exp(logvar))
        
def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_loader):
        recon_batch, mu, logvar = model(data)
        optimizer.zero_grad()
        loss = model.loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        if batch_id % 10 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                .format(
                    epoch, batch_id * len(data), len(train_loader.dataset),
                    100. * batch_id / len(train_loader), loss.item()
                )
            )


def test(model, test_loader):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            output, mu, logvar = model(data)
            test_loss += model.loss_function(output, data, mu, logvar).item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))

def main():
    train_loader = torch.utils.data.DataLoader(
		datasets.MNIST(
            'data', train=True, download=True,
			transform=transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.1307,), (0.3081,))
			])
        ), batch_size=BATCH_SIZE, shuffle=True
    )
        
    test_loader = torch.utils.data.DataLoader(
		datasets.MNIST(
            'data', train=True,
            transform=transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.1307,), (0.3081,))
			])
        ), batch_size=BATCH_SIZE, shuffle=True
    )
    
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    
    summary(model, (1, 28, 28))
    
    for epoch in range(NUM_EPOCHS):
        train(model, train_loader, optimizer, epoch + 1)
        test(model, test_loader)
        
    torch.save(model.state_dict(), "mnist_nn.pt")
        
    
if __name__ == '__main__':
    main()