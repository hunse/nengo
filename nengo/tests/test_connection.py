import logging

import numpy as np

import nengo
from nengo.builder import ShapeMismatch
from nengo.objects import Ensemble, Node
import nengo.old_api as nef
from nengo.tests.helpers import Plotter, rmse, SimulatorTestCase, unittest
from nengo.helpers import piecewise

logger = logging.getLogger(__name__)

class TestConnection(SimulatorTestCase):

    def test_node_to_neurons(self):
        name = 'node_to_neurons'
        N = 30

        m = nengo.Model(name, seed=123)
        a = m.make_ensemble('A', nengo.LIF(N), dimensions=1)
        m.make_node('in', output=np.sin)
        m.make_node('inh', piecewise({0:0, 2.5:1}))
        m.connect('in', 'A')
        con = m.connect('inh', a.neurons, transform=[[-2.5]]*N)

        m.probe('in')
        m.probe('A', filter=0.1)
        m.probe('inh')

        sim = m.simulator(sim_class=self.Simulator)
        sim.run(5.0)
        t = sim.data(m.t)
        ideal = np.sin(t)
        ideal[t >= 2.5] = 0

        with Plotter(self.Simulator) as plt:
            plt.plot(t, sim.data('in'), label='Input')
            plt.plot(t, sim.data('A'), label='Neuron approx, filter=0.1')
            plt.plot(t, sim.data('inh'), label='Inhib signal')
            plt.plot(t, ideal, label='Ideal output')
            plt.legend(loc=0, prop={'size':10})
            plt.savefig('test_connection.test_' + name + '.pdf')
            plt.close()

        self.assertTrue(np.allclose(sim.data('A')[-10:], 0, atol=.1, rtol=.01))

    def test_ensemble_to_neurons(self):
        name = 'ensemble_to_neurons'
        N = 30

        m = nengo.Model(name, seed=123)
        a = m.make_ensemble('A', nengo.LIF(N), dimensions=1)
        b = m.make_ensemble('B', nengo.LIF(N), dimensions=1)
        m.make_node('in', output=np.sin)
        m.make_node('inh', piecewise({0:0,2.5:1}))
        m.connect('in', 'A')
        m.connect('inh', 'B')
        con = m.connect('B', a.neurons, transform=[[-2.5]]*N)

        m.probe('in')
        m.probe('A', filter=0.1)
        m.probe('B', filter=0.1)
        m.probe('inh')

        sim = m.simulator(sim_class=self.Simulator)
        sim.run(5.0)
        t = sim.data(m.t)
        ideal = np.sin(t)
        ideal[t >= 2.5] = 0

        with Plotter(self.Simulator) as plt:
            plt.plot(t, sim.data('in'), label='Input')
            plt.plot(t, sim.data('A'), label='Neuron approx, pstc=0.1')
            plt.plot(
                t, sim.data('B'), label='Neuron approx of inhib sig, pstc=0.1')
            plt.plot(t, sim.data('inh'), label='Inhib signal')
            plt.plot(t, ideal, label='Ideal output')
            plt.legend(loc=0, prop={'size':10})
            plt.savefig('test_connection.test_' + name + '.pdf')
            plt.close()

        self.assertTrue(np.allclose(sim.data('A')[-10:], 0, atol=.1, rtol=.01))
        self.assertTrue(np.alllclose(sim.data('B')[-10:], 1, atol=.1, rtol=.01))

    def test_neurons_to_ensemble(self):
        name = 'neurons_to_ensemble'
        N = 10

        m = nengo.Model(name, seed=123)
        a = m.make_ensemble('A', nengo.LIF(N * 2), dimensions=2)
        b = m.make_ensemble('B', nengo.LIF(N * 3), dimensions=3)
        c = m.make_ensemble('C', nengo.LIF(N), dimensions=N*2)
        m.connect(a.neurons, b, decoders=-10 * np.ones((N*2, 3)))
        m.connect(a.neurons, c)

        m.probe('A', filter=0.01)
        m.probe('B', filter=0.01)
        m.probe('C', filter=0.01)

        sim = m.simulator(sim_class=self.Simulator)
        sim.run(5.0)
        t = sim.data(m.t)

        with Plotter(self.Simulator) as plt:
            plt.plot(t, sim.data('A'), label='A')
            plt.plot(t, sim.data('B'), label='B')
            plt.savefig('test_connection.test_' + name + '.pdf')
            plt.close()

        self.assertTrue(np.all(sim.data('B')[-10:] < -0.5))

    def test_neurons_to_node(self):
        name = 'neurons_to_node'
        N = 30

        m = nengo.Model(name, seed=123)
        a = m.make_ensemble('A', nengo.LIF(N), dimensions=1)
        out = m.add(Node('out', lambda x: x, dimensions=N))
        m.connect(a.neurons, out)

        #m.probe('A.spikes')
        #m.probe(out)

        sim = m.simulator(sim_class=self.Simulator)
        sim.run(5.0)
        t = sim.data(m.t)

        with Plotter(self.Simulator) as plt:
            plt.plot(t, sim.data(a.neurons), label='A neurons')
            plt.plot(t, sim.data(out), label='Node output')
            plt.legend(loc=0, prop={'size':10})
            plt.savefig('test_connection.test_' + name + '.pdf')
            plt.close()

        self.assertTrue(np.allclose(sim.data(a.neurons), sim.data(out)))

    def test_neurons_to_neurons(self):
        name = 'neurons_to_neurons'
        N1, N2 = 30, 50

        m = nengo.Model(name, seed=123)
        a = m.make_ensemble('A', nengo.LIF(N1), dimensions=1)
        b = m.make_ensemble('B', nengo.LIF(N2), dimensions=1)
        inp = m.make_node('in', output=1)
        m.connect(inp, a)
        m.connect(a.neurons, b.neurons, transform=-1 * np.ones((N1, N2)))

        m.probe('in')
        m.probe('A', filter=0.1)
        m.probe('B', filter=0.1)

        sim = m.simulator(sim_class=self.Simulator)
        sim.run(5.0)
        t = sim.data(m.t)

        with Plotter(self.Simulator) as plt:
            plt.plot(t, sim.data('in'), label='Input')
            plt.plot(t, sim.data('A'), label='A, represents input')
            plt.plot(t, sim.data('B'), label='B, should be 0')
            plt.legend(loc=0, prop={'size':10})
            plt.savefig('test_connection.test_' + name + '.pdf')
            plt.close()

        self.assertTrue(np.allclose(sim.data('A')[-10:], 1, atol=.1, rtol=.01))
        self.assertTrue(np.allclose(sim.data('B')[-10:], 0, atol=.1, rtol=.01))

if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
