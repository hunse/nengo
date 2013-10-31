import numpy as np

import nengo
import nengo.old_api as nef
from nengo.tests.helpers import SimulatorTestCase, unittest


class TestModelBuild(unittest.TestCase):
    def test_build(self):
        m = nengo.Model('test_build', seed=123)
        m.make_node('in', output=1)
        m.make_ensemble('A', nengo.LIF(40), 1)
        m.make_ensemble('B', nengo.LIF(20), 1)
        m.connect('in', 'A')
        m.connect('A', 'B', function=lambda x: x ** 2)
        m.probe('in')
        m.probe('A', filter=0.01)
        m.probe('B', filter=0.01)

        mcopy = m.simulator(dt=0.001).model
        self.assertItemsEqual(m.objs.keys(), mcopy.objs.keys())

        def compare_objs(orig, copy, attrs):
            for attr in attrs:
                self.assertEqual(getattr(orig, attr), getattr(copy, attr))
            for attr in ('connections_in', 'connections_out'):
                self.assertEqual(len(getattr(orig, attr)),
                                 len(getattr(copy, attr)))
                for o_c, c_c in zip(getattr(orig, attr), getattr(copy, attr)):
                    compare_connections(o_c, c_c)
            for p_o, p_c in zip(orig.probes.values(), copy.probes.values()):
                self.assertEqual(len(p_o), len(p_c))

        def compare_connections(orig, copy):
            self.assertEqual(orig.filter, copy.filter)
            self.assertEqual(orig.transform, copy.transform)
            for p_o, p_c in zip(orig.probes.values(), copy.probes.values()):
                self.assertEqual(len(p_o), len(p_c))

        compare_objs(m.get('in'), mcopy.get('in'), ('output',))

        ens_attrs = ('name', 'dimensions', 'radius')
        compare_objs(m.get('A'), mcopy.get('A'), ens_attrs)
        compare_objs(m.get('B'), mcopy.get('B'), ens_attrs)

    def test_seeding(self):
        """Test that setting the model seed fixes everything"""

        ### TODO: this really just checks random parameters in ensembles.
        ###   Are there other objects with random parameters that should be
        ###   tested? (Perhaps initial weights of learned connections)

        m = nengo.Model('test_seeding')
        m.make_node('in', output=1)
        m.make_ensemble('A', nengo.LIF(40), 1)
        m.make_ensemble('B', nengo.LIF(20), 1)
        m.connect('in', 'A')
        m.connect('A', 'B', function=lambda x: x ** 2)
        m.probe('in')
        m.probe('A', filter=0.01)
        m.probe('B', filter=0.01)

        m.seed = 872
        m1 = m.simulator(dt=0.001).model
        m2 = m.simulator(dt=0.001).model
        m.seed = 873
        m3 = m.simulator(dt=0.001).model

        def compare_objs(obj1, obj2, attrs, equal=True):
            for attr in attrs:
                check = (np.all(getattr(obj1, attr) == getattr(obj2, attr))
                         if equal else
                         np.any(getattr(obj1, attr) != getattr(obj2, attr)))
                if not check:
                    print getattr(obj1, attr)
                    print getattr(obj2, attr)
                self.assertTrue(check)

        ens_attrs = ('encoders', 'max_rates', 'intercepts')
        A = [mi.get('A') for mi in [m1, m2, m3]]
        B = [mi.get('B') for mi in [m1, m2, m3]]
        compare_objs(A[0], A[1], ens_attrs)
        compare_objs(B[0], B[1], ens_attrs)
        compare_objs(A[0], A[2], ens_attrs, equal=False)
        compare_objs(B[0], B[2], ens_attrs, equal=False)

        neur_attrs = ('gain', 'bias')
        compare_objs(A[0].neurons, A[1].neurons, neur_attrs)
        compare_objs(B[0].neurons, B[1].neurons, neur_attrs)
        compare_objs(A[0].neurons, A[2].neurons, neur_attrs, equal=False)
        compare_objs(B[0].neurons, B[2].neurons, neur_attrs, equal=False)


class TestModel(SimulatorTestCase):

    def test_counters(self):
        m = nengo.Model('test_counters', seed=123)
        m.probe(m.steps)
        sim = m.simulator(dt=0.001, sim_class=self.Simulator)
        sim.run(0.003)
        self.assertTrue(np.allclose(sim.data(m.t).flatten(),
                                    [0.00, .001, .002]))
        self.assertTrue(np.allclose(sim.data(m.steps).flatten(), [0, 1, 2]))

    def test_get(self):
        m = nengo.Model('test_get')
        ens = m.make_ensemble('e', nengo.LIF(50), 1)
        self.assertTrue(id(ens) == id(m.get(ens)))
        self.assertTrue(id(ens) == id(m.get('e')))

        sim = m.simulator(dt=0.001)
        self.assertTrue(id(sim.get('e')) == id(sim.model.get('e')))
        self.assertTrue(id(sim.get(ens)) == id(sim.model.get(ens)))
        self.assertTrue(id(ens) != id(sim.get('e')))
        self.assertTrue(id(ens) != id(sim.get(ens)))
        self.assertTrue(id(ens) != id(sim.model.get('e')))
        self.assertTrue(id(ens) != id(sim.model.get(ens)))

        self.assertTrue(m.get('r') is None)
        self.assertTrue(m.get(3) is None)
        self.assertTrue(m.get(sim.model.get('e')) is None)


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
