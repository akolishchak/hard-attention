# Hard Attention

An image is scanned via small patches (glimpses) to perform classification. Inspired by “Recurrent Models of Visual Attention” http://arxiv.org/abs/1406.6247 (RAM), however different model is used.

The differences:
1) No glimpse network, glimpse is a single patch (4x4), instead a collection of multiple resolution patches as in RAM
2) Actions are discrete: move up, up-right, right, down-right, down, down-left, left from the current location. RAM actions is exact location of patches.
3) Additional network is used to identify initial location from where glimpses start. The location network receives downsized image as input.
![alt tag](drawing.png)
Action and Initial Location networks are trained by REINFORCE. The reward with class probability value as a reward.

## Results of training on MNIST
20 glimpses of 4x4 size with action step size 2, 10 epochs
train – 97.48%, test – 95.56%

Interesting enough that in many cases actions learned trajectories that reproduce the form of digits:

original image|overlapped by saccade trajectory
---|---
![alt tag](samples/actions_1.jmg)|![alt tag](samples/actions_1-5.jmg)
![alt tag](samples/actions_4.jmg)|![alt tag](samples/actions_4-1.jmg)
![alt tag](samples/actions_20.jmg)|![alt tag](samples/actions_20-9.jmg)
![alt tag](samples/actions_26.jmg)|![alt tag](samples/actions_26-2.jmg)
![alt tag](samples/actions_48.jmg)|![alt tag](samples/actions_48-5.jmg)
![alt tag](samples/actions_53.jmg)|![alt tag](samples/actions_53-7.jmg)
![alt tag](samples/actions_57.jmg)|![alt tag](samples/actions_57-10.jmg)
![alt tag](samples/actions_77.jmg)|![alt tag](samples/actions_77-2.jmg)
![alt tag](samples/actions_85.jmg)|![alt tag](samples/actions_85-7.jmg)
![alt tag](samples/actions_90.jmg)|![alt tag](samples/actions_90-4.jmg)
![alt tag](samples/actions_99.jmg)|![alt tag](samples/actions_99-3.jmg)
![alt tag](samples/actions_102.jmg)|![alt tag](samples/actions_102-7.jmg)
![alt tag](samples/actions_104.jmg)|![alt tag](samples/actions_104-7.jmg)
![alt tag](samples/actions_106.jmg)|![alt tag](samples/actions_106-1.jmg)
![alt tag](samples/actions_124.jmg)|![alt tag](samples/actions_124-7.jmg)
![alt tag](samples/actions_131.jmg)|![alt tag](samples/actions_131-3.jmg)
![alt tag](samples/actions_135.jmg)|![alt tag](samples/actions_135-1.jmg)
![alt tag](samples/actions_137.jmg)|![alt tag](samples/actions_137-3.jmg)
![alt tag](samples/actions_188.jmg)|![alt tag](samples/actions_188-2.jmg)
![alt tag](samples/actions_217.jmg)|![alt tag](samples/actions_217-10.jmg)
![alt tag](samples/actions_218.jmg)|![alt tag](samples/actions_218-4.jmg)
![alt tag](samples/actions_239.jmg)|![alt tag](samples/actions_239-6.jmg)
![alt tag](samples/actions_315.jmg)|![alt tag](samples/actions_315-4.jmg)
---

## The model parameters
### RNN
LSTM, one layer of 512 cells

### Action Network
1 hidden layer with 512 neurons + soft max + stochastic multinomial module

### Initial Location Network
one layer, 4 x downsized input, soft max + stochastic multinomial module that outputs location coordinates

### Regularization
All networks use dropout 0.7. The dropout is essential for network stability and allows finding of better action policies.

