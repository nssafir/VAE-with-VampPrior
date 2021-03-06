import torch
from torch import nn, optim
import torch.nn.functional as F
from torchsummary import summary
from torchvision import datasets, transforms
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        
        # encoder layers
        super(Net, self).__init__()       
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.conv2 = nn.Conv2d(3, 6, 3)
        self.fc1 = nn.Linear(150, 84)
        self.fc2 = nn.Linear(84, 10)
        self.fc3 = nn.Linear(10, 2)
        
        # decoder layers
        self.fc4 = nn.Linear(2, 400)
        self.fc5 = nn.Linear(400, 784)
        self.sig = nn.Sigmoid()
        
    def encode(self, x):        
        # moves through encoding layers
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # makes it [3, 13, 13]
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # makes it [6, 5, 5]
        x = x.view(-1, self.num_flat_features(x)) # makes it [150]
        x = F.relu(self.fc1(x)) # makes it [84]
        x = F.relu(self.fc2(x)) # makes it [10]
        mu = F.relu(self.fc3(x)) # makes it [2]
        logvar = F.relu(self.fc3(x)) # makes it [2]        
        # returns mu and logvar
        return mu, logvar
        
    def reparam(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
     
    def decode(self, z):
        x = F.relu(self.fc4(z)) # makes it [400]
        x = F.relu(self.fc5(x)) # makes it [400]
        return self.sig(x)
            
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def loss_function(recon_x, x, mu, logvar):
    # reconstruction error
    recon_error = F.binary_cross_entropy(recon_x, x.view(-1, 784))
    # kl divergence
    kld_error = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_error /= 64 * 784
    # return sum of reconstruction error and kl divergence
    return recon_error + kld_error
    


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_loader):
        recon_batch, mu, logvar = model(data)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        if batch_id % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_id * len(data), len(train_loader.dataset),
				100. * batch_id / len(train_loader), loss.item()))


def test(model, test_loader):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            output, mu, logvar = model(data)
            test_loss += loss_function(output, data, mu, logvar).item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))

def main():
    train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('data', train=True, download=True,
					transform=transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,))
					])),
		batch_size=64, shuffle=True)
        
    test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('data', train=True, transform=transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,))
					])),
		batch_size=64, shuffle=True)
    
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    #summary(model, (1, 28, 28))
    
    for epoch in range(5):
        train(model, train_loader, optimizer, epoch + 1)
        test(model, test_loader)
        
    torch.save(model.state_dict(), "mnist_nn.pt")
        
    
if __name__ == '__main__':
    main()