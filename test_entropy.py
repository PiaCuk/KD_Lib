import torch as th
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

bs = 32 # batch size
ns = 3 # number of students
nc = 10 # number of classes

# sample some fake data to simulate output. 
# Key: classes dimension is last, number of dims before that arbitrary
logits = th.randn((bs, ns, nc))

# d represents (bs, ns) categorical distributions...
d = Categorical(logits=logits)
print(d.batch_shape)

entropy = d.entropy()
print(entropy) # ...(bs, ns) array w/ Shannon entropies of each distribution
# N.B.: entropy is in nats (taking natural log in definition of Shannon entropy). If necessary, can be converted to bits (log2) by dividing by log(2):
entropy_bits = entropy / th.log(th.as_tensor(2.0))


# just to verify that result matches what we get from applying definition
probabilities = F.softmax(logits, dim=-1) # softmax over the classes dim
e = -(probabilities * th.log(probabilities)).sum(dim=-1)
print(entropy - e) # close enough! (~1e-7)

# verify if my calculation is correct
my_probabilities = th.softmax(logits, dim=-1)
my_entropy = -(my_probabilities * my_probabilities.log()).sum(dim=-1)
print(entropy - my_entropy)

# take mean of batch to log to tensorboard
print(my_entropy.mean(dim=0))