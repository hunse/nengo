"""
This example demonstrates how to create an integrator in neurons.
  The function an integrator implements can be written in the
  following control theoretic equation:

    a_dot(t) = A * a(t) + B * input(t)

  The NEF equivalent equation for this integrator is:

    a_dot(t) = a(t) + tau * input(t)

  where tau is the recurrent time constant.

Network diagram:

                    .----.
                    v    |
     [Input] ----> (A) --'


Network behaviour:
  A = tau * Input + Input

"""
import nengo

from nengo.objects import Node, Ensemble, Connection

model = nengo.Model('Integrator')

# Our ensemble consists of 100 leaky integrate-and-fire neurons,
# representing a one-dimensional signal
A = Ensemble('A', nengo.LIF(100), dimensions=1)
print "test"

# Create a piecewise step function for input
input = Node('Piecewise input', nengo.helpers.piecewise({0:0,0.2:1,1:0,2:-2,3:0,4:1,5:0}))

# Connect the population to itself
tau = 0.1
Connection(A, A, transform=[[1]], filter=tau) #The same time constant as recurrent to make it more 'ideal'

# Connect the input
model.connect(input, A, transform=[[tau]], filter=tau) #The same time constant as recurrent to make it more 'ideal'

model.probe('Piecewise input')
model.probe('Integrator', filter=0.01) #10ms filter

# Create our simulator
sim = model.simulator()
# Run it for 6 seconds
sim.run(6)


import matplotlib.pyplot as plt

# Plot the decoded output of the ensemble
t = sim.data(model.t) #Get the time steps
plt.plot(t, sim.data('A'), label="A output")
plt.plot(t, sim.data('Input'), 'k', label="Input")
plt.legend()
