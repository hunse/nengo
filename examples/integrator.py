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
import nengo.helpers

from nengo.objects import Node, Ensemble, DecodedConnection, Connection, Probe

model = nengo.Model(label='Integrator')

with model:
    # Our ensemble consists of 100 leaky integrate-and-fire neurons,
    # representing a one-dimensional signal
    A = Ensemble(nengo.LIF(100), dimensions=1, label='Integrator')
    
    # Create a piecewise step function for input
    input = Node(nengo.helpers.piecewise({0:0,0.2:1,1:0,2:-2,3:0,4:1,5:0}), label='Piecewise input')
    
    # Connect the population to itself
    tau = 0.1
    DecodedConnection(A, A, transform=[[1]], filter=tau) #A long time constant for stability
    
    # Connect the input
    Connection(input, A, transform=[[tau]], filter=tau) #The same time constant as recurrent to make it more 'ideal'
    
    # Add probes
    p1 = Probe(input, 'output')
    p2 = Probe(A, 'decoded_output', filter=0.01)
    p3 = Probe(model.t, 'output')


# Create our simulator
sim = model.simulator()
# Run it for 6 seconds
sim.run(6)


import matplotlib.pyplot as plt

# Plot the decoded output of the ensemble
t = sim.data(p3) #Get the time steps
plt.plot(t, sim.data(p1), label="Input")
plt.plot(t, sim.data(p2), 'k', label="Integrator output")
plt.legend()

plt.show()
