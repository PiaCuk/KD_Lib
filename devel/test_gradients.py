import torch as th
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        logits = self.fc(x)
        return logits


def ensemble_targets(logits_list, j):
    # Calculate ensemble target given a list of logits, omitting the j'th element
    num_logits = len(logits_list)
    target = th.zeros(logits_list[j].shape)
    for i, logits in enumerate(logits_list):
        if i != j:
            target += (1 / (num_logits - 1)) * \
                F.softmax(logits, dim=-1)
    return target


def ensemble_targets2(logits_list, j):
    logits_list = logits_list[:j] + logits_list[j+1:]
    logits = th.softmax(th.stack(logits_list), dim=-1)
    target = logits.mean(dim=0)
    # print("Ensemble target: {}".format(target))
    return target


# for repeatability...
th.manual_seed(0)
th.use_deterministic_algorithms(True)

# generate some dummy data
input_dim = 4
num_classes = 3
num_points = 64
X = th.randn((num_points, input_dim))
Y = th.randint(high=num_classes, size=(num_points,))

# generate networks
num_networks = 3
nets = [Network(input_dim, num_classes) for j in range(num_networks)]
net_optimizers = []
for n in nets:
    net_optimizers.append(th.optim.SGD(n.parameters(), lr=1e-4))

# Make deep copies of the networks with initial parameters to use later
nets_copy = [deepcopy(net) for net in nets]

# Deep mutual learning - as presented in Zhang et al. (2018)
print('Now running DML')
# Forward passes to compute logits
net_logits = [n(X) for n in nets]

for i, n1 in enumerate(nets):
    ce_loss = F.cross_entropy(net_logits[i], Y)
    kl_loss = 0.0
    for j, n2 in enumerate(nets):
        if i != j:
            # Here it is crucial to detach net_logits[j], since we do not want to backpropagate through network j in this iteration, only i!
            kl_loss += (1 / (num_networks - 1)) * F.kl_div(
                th.log_softmax(net_logits[i], dim=-1),
                th.softmax(net_logits[j].detach(), dim=-1),
                reduction='batchmean', log_target=False)
    """Notes regarding PyTorch's implementation of KL divergence
    - using reduction 'batchmean' will divide by batch size the summed result, leading to the usual KL divergence result
    - The arguments are swapped from usual mathematical definition of the KL div, so D(p, q) is obtained by kl_div(q, p)
    - The first argument should be log probabilities, so from logits z, use th.log_softmax(z) to get first arg
    - Second argument should be plain probabilities, so from logits z, use th.softmax(z) to get second arg
    - In this method, set log_target to False.
    - You can use log probs for the second input, then set log_target True
    """

    net_optimizers[i].zero_grad()
    loss = ce_loss + kl_loss
    loss.backward()
    net_optimizers[i].step()
    print("CE Loss {}, KL Loss {}".format(ce_loss, kl_loss))
    for name, p in n1.named_parameters():
        print("Network {}, parameter: {}, gradient: {}".format(i, name, p.grad))

    # update prediction of network i for future iterations
    net_logits[i] = n1(X)


# Deep mutual learning with ensemble teacher (DML_e)
print('Now running DML_e')
nets = [deepcopy(net) for net in nets_copy]

net_logits = [n(X) for n in nets]

for i, n1 in enumerate(nets):
    ce_loss = F.cross_entropy(net_logits[i], Y)
    # Calculate ensemble target
    tgt = ensemble_targets(net_logits, i)
    # use plain probs for second arg of kl_div, so log_target is False
    # detach tgt to not backprop through networks other than i
    kl_loss = F.kl_div(th.log_softmax(net_logits[i], dim=-1), tgt.detach(),
                       reduction='batchmean', log_target=False)

    net_optimizers[i].zero_grad()
    loss = ce_loss + kl_loss
    loss.backward()
    net_optimizers[i].step()
    print("CE Loss {}, KL Loss {}".format(ce_loss, kl_loss))
    for name, p in n1.named_parameters():
        print("Network {}, parameter: {}, gradient: {}".format(i, name, p.grad))

    # update prediction of network i for future iterations
    net_logits[i] = n1(X)

# Deep mutual learning with ensemble teacher (DML_e)
print('Now running DML_e - second variant')
nets = [deepcopy(net) for net in nets_copy]

net_logits = [n(X) for n in nets]

for i, n1 in enumerate(nets):
    ce_loss = F.cross_entropy(net_logits[i], Y)
    # Calculate ensemble target
    tgt = ensemble_targets2(net_logits, i)
    # use plain probs for second arg of kl_div, so log_target is False
    # detach tgt to not backprop through networks other than i
    kl_loss = F.kl_div(th.log_softmax(net_logits[i], dim=-1), tgt.detach(),
                       reduction='batchmean', log_target=False)

    net_optimizers[i].zero_grad()
    loss = ce_loss + kl_loss
    loss.backward()
    net_optimizers[i].step()
    print("CE Loss {}, KL Loss {}".format(ce_loss, kl_loss))
    for name, p in n1.named_parameters():
        print("Network {}, parameter: {}, gradient: {}".format(i, name, p.grad))

    # update prediction of network i for future iterations
    net_logits[i] = n1(X)
