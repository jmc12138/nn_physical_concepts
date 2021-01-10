from scinet import *
import scinet.ed_quantum as edq

edq.create_data(1, 5, 100000, 'qubit_example');

td, vd, ts, vs, proj = dl.load(5, 'qubit_example')

# Create network object
net = nn.Network(5, 2)

# Print initial reconstruction loss (depends on initialization)
net.run(vd, net.recon_loss)

# Train
net.train(500, 256, 0.001, td, vd,test_step = 10)

# Check progress. It is recommended to use Tensorboard instead for this.
net.run(vd, net.recon_loss)

# More training
#net.train(50, 256, 0.001, td, vd)

# Check progress. It is recommended to use Tensorboard instead for this.
#net.run(vd, net.recon_loss)