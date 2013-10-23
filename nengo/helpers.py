from __future__ import absolute_import

import copy

import numpy as np

from . import decoders

def tuning_curves(sim_ens):
    eval_points = np.array(sim_ens.eval_points)
    eval_points.sort(axis=0)
    activities = sim_ens.activities(eval_points)
    return sim_ens.eval_points, activities


def encoders():
    @staticmethod
    def _process_encoders(encoders, neurons, dims, n_ensembles):
        if encoders is None:
            encoders = [None for _ in xrange(n_ensembles)]
        elif len(encoders) == dims:
            if np.asarray(encoders).ndim == 1:
                encoders = [np.array(encoders) for _ in xrange(n_ensembles)]
        elif len(encoders) == neurons:
            if len(encoders[0]) != dims:
                msg = ("len(encoders[0]) should match dimensions_per_ensemble. "
                       "Currently %d, %d" % (len(encoders[0]) != dims))
                raise core.ShapeMismatch(msg)
            encoders = [np.array(encoders) for _ in xrange(n_ensembles)]
        elif len(encoders) != n_ensembles:
            msg = ("len(encoders) should match n_ensembles. "
                   "Currently %d, %d" % (len(encoders) != n_ensembles))
            raise core.ShapeMismatch(msg)
        return encoders


def transform(pre_dims, post_dims,
              weight=1.0, index_pre=None, index_post=None):
    """Helper function used to create a ``pre_dims`` by ``post_dims``
    linear transformation matrix.

    Parameters
    ----------
    pre_dims, post_dims : int
        The numbers of presynaptic and postsynaptic dimensions.
    weight : float, optional
        The weight value to use in the transform.

        All values in the transform are either 0 or ``weight``.

        **Default**: 1.0
    index_pre, index_post : iterable of int
        Determines which values are non-zero, and indicates which
        dimensions of the pre-synaptic ensemble should be routed to which
        dimensions of the post-synaptic ensemble.

    Returns
    -------
    transform : 2D matrix of floats
        A two-dimensional transform matrix performing the requested routing.

    Examples
    --------

      # Sends the first two dims of pre to the first two dims of post
      >>> gen_transform(pre_dims=2, post_dims=3,
                        index_pre=[0, 1], index_post=[0, 1])
      [[1, 0], [0, 1], [0, 0]]

    """
    t = [[0 for pre in xrange(pre_dims)] for post in xrange(post_dims)]
    if index_pre is None:
        index_pre = range(pre_dims)
    elif isinstance(index_pre, int):
        index_pre = [index_pre]

    if index_post is None:
        index_post = range(post_dims)
    elif isinstance(index_post, int):
        index_post = [index_post]

    for i in xrange(min(len(index_pre), len(index_post))):  # was max
        pre = index_pre[i]  # [i % len(index_pre)]
        post = index_post[i]  # [i % len(index_post)]
        t[post][pre] = weight
    return t


def weights(pre_neurons, post_neurons, function):
    """Helper function used to create a ``pre_neurons`` by ``post_neurons``
    connection weight matrix.

    Parameters
    ----------
    pre_neurons, post_neurons : int
        The numbers of presynaptic and postsynaptic neurons.
    function : function
        A function that generates weights.

        If it accepts no arguments, it will be called to
        generate each individual weight (useful
        to great random weights, for example).
        If it accepts two arguments, it will be given the
        ``pre`` and ``post`` index in the weight matrix.

    Returns
    -------
    weights : 2D matrix of floats
        A two-dimensional connection weight matrix.

    Examples
    --------

      >>> gen_weights(2, 2, random.random)
      [[0.6281625119511959, 0.48560016153108376], [0.9639779858394248, 0.4768136917985597]]

      >>> def product(pre, post):
      ...     return pre * post
      >>> gen_weights(3, 3, product)
      [[0, 0, 0], [0, 1, 2], [0, 2, 4]]

    """
    argspec = inspect.getargspec(func)
    if len(argspec[0]) == 0:
        return [[func() for pre in xrange(pre_neurons)
                 for post in xrange(post_neurons)]]
    elif len(argspec[0]) == 2:
        return [[func(pre, post) for pre in xrange(pre_neurons)
                 for post in xrange(post_neurons)]]



### Helper functions for creating inputs

def piecewise(data):
    """Create a piecewise constant function from a dictionary.

    Given an input of data={0:0, 0.5:1, 0.75:-1, 1:0} this will generate a
    function that returns 0 up until t=0.5, then outputs a 1 until t=0.75,
    then a -1 until t=1, and then returns 0 after that.  This is meant as a
    shortcut for::

        def function(t):
            if t<0.5: return 0
            elif t<0.75: return 1
            elif t<1: return -1
            else: return 0

    The keys in the dictionary must be times (floats or ints).  The values in
    the data dictionary can be floats, lists, or functions that return
    floats or lists.  All lists must be of the same length.

    For times before the first specified time, it will default to zero (of
    the correct length).  This means the above example can be simplified to::

        piecewise({0.5:1, 0.75:-1, 1:0})

    Parameters
    ----------
    data : dict
        The values to change to.  Keys are the beginning time for the value.
        Values can be int, float, list, or functions that return those.

    Returns
    -------
    function:
        A function that takes a variable t and returns the corresponding
        value from the dictionary.

    Examples
    --------

      >>> func = piecewise({0.5:1, 0.75:-1, 1:0})
      >>> func(0.2)
      [0]
      >>> func(0.58)
      [1]

      >>> func = piecewise({0.5:[1,0], 0.75:[0,1]})
      >>> func(0.2)
      [0,0]
      >>> func(0.58)
      [1,0]
      >>> func(100)
      [0,1]

      >>> import math
      >>> func = piecewise({0:math.sin, 0.5:math.cos})
      >>> func(0.499)
      [0.47854771647582706]
      >>> func(0.5)
      [0.8775825618903728]

    """

    # first, sort the data (to simplify finding the right element when calling
    #  the function)
    ordered_data = []
    output_length = None  # the dimensionality of the returned values
    for time, output in sorted(data.items()):
        if not isinstance(time, (float, int)):
            raise TypeError('Keys must be times (floats or ints), not "%s"'%`time`)

        # handle ints and floats by turning them into a list
        if isinstance(output, (float, int)):
            output = [output]

        # figure out the length of this item
        if callable(output):
            value = output(0.0)
            if isinstance(value, (float, int)):
                length = 1
            else:
                length = len(value)
        else:
            length = len(output)

        # make sure this is the same length as previous items
        if output_length is None:
            output_length = length
        elif output_length != length:
            raise Exception('invalid data for piecewise function ' +
                            '(time %4g has %d items instead of %d)'%
                            (time,length,output_length))

        # add it to the ordered list
        row = (time, output)
        ordered_data.append(row)

    # set the value to zero for t befoer the first given time
    initial_value = [0]*output_length

    # build the function to return
    def piecewise_function(t, data=ordered_data, start=initial_value):
        value = start   # start at zero

        # find the correct output value
        for time, output in ordered_data:
            if value is None or time<=t:
                value = output
            else:
                break

        # if it's a function, call it
        if callable(value):
            value = value(t)
            # force the result to be a list
            if isinstance(value, (int, float)):
                value = [value]
        return value
    return piecewise_function


def sorted_neurons(ensemble, iterations=100, seed=None):
    '''Sort neurons in an ensemble by encoder and intercept.

    Parameters
    ----------
    simulator: nengo.simulator.Simulator
        The simulator that runs the ensemble

    ensemble: nengo.objects.Ensemble
        The population of neurons to be sorted

    iterations: int
        The number of times to iterate during the sort

    seed: float
        A random number seed

    Returns
    -------
    indices: nparray
        An array with sorted indices into the neurons in the ensemble

    Examples
    --------

    You can use this to generate an array of sorted indices for plotting. This
    can be done after collecting the data. E.g.

    >>>indices = sorted_neurons(simulator, 'My neurons')
    >>>plt.figure()
    >>>rasterplot(sim.data('My neurons.spikes')[:,indices])

    Algorithm
    ---------

    The algorithm is for each encoder in the initial set, randomly
    pick another encoder and check to see if swapping those two
    encoders would reduce the average difference between the
    encoders and their neighbours.  Difference is measured as the
    dot product.  Each encoder has four neighbours (N, S, E, W),
    except for the ones on the edges which have fewer (no wrapping).
    This algorithm is repeated `iterations` times, so a total of
    `iterations*N` swaps are considered.
    '''

    def score(encoders, index, rows, cols=1):
        """Helper function to compute similarity for one encoder.

        :param array encoders: the encoders
        :param integer index: the encoder to compute for
        :param integer rows: the width of the 2d grid
        :param integer cols: the height of the 2d grid
        """
        i = index % cols   # find the 2d location of the indexth element
        j = index / cols

        sim = 0     # total of dot products
        count = 0   # number of neighbours
        if i>0: # if we're not at the left edge, do the WEST comparison
            sim += np.dot(encoders[j*cols+i], encoders[j*cols+i-1])
            count += 1
        if i<cols-1:  # if we're not at the right edge, do EAST
            sim += np.dot(encoders[j*cols+i], encoders[j*cols+i+1])
            count += 1
        if j>0:   # if we're not at the top edge, do NORTH
            sim += np.dot(encoders[j*cols+i], encoders[(j-1)*cols+i])
            count += 1
        if j<rows-1:  # if we're not at the bottom edge, do SOUTH
            sim += np.dot(encoders[j*cols+i], encoders[(j+1)*cols+i])
            count += 1
        return sim/count

    #Normalize all the neurons
    encoders = np.array(ensemble.encoders)
    for i in np.arange(encoders.shape[0]):
        encoders[i,:]=encoders[i,:]/np.linalg.norm(encoders[i,:])

    #Make an array with the starting order of the neurons
    N = encoders.shape[0]
    indices = np.arange(N)
    rng = np.random.RandomState(seed)

    for k in range(iterations):
        target = rng.randint(0, N, N)  # pick random swap targets
        for i in range(N):
            j = target[i]
            if i != j:  # if not swapping with yourself
                # compute similarity score how we are (unswapped)
                sim1 = score(encoders, i, N) + score(encoders,
                                                          j, N)
                # swap the encoder
                encoders[[i,j],:] = encoders[[j,i],:]
                indices[[i,j]] = indices[[j,i]]
                # compute similarity score how we are (swapped)
                sim2 = score(encoders, i, N) + score(encoders,
                                                          j, N)

                # if we were better unswapped
                if sim1 > sim2:
                    # swap them back
                    encoders[[i,j],:] = encoders[[j,i],:]
                    indices[[i,j]] = indices[[j,i]]

    return indices


def white_noise(step, high, rms=0.5, seed=None, dimensions=None):
    """Generate white noise inputs

    Parameters
    ----------
    step : float
        The step size of different frequencies to generate

    high : float
        The highest frequency to generate (should be a multiple of step)

    rms : float
        The RMS power of the signal

    seed : int or None
        Random number seed

    dimensions : int or None
        The number of different random signals to generate.  The resulting
        function will return an array of length `dimensions` for every
        point in time.  If `dimensions` is None, the resulting function will
        just return a float for each point in time.

    Returns
    -------
    function:
        A function that takes a variable t and returns the value of the
        randomly generated signal.  This value is a float if `dimensions` is
        None; otherwise it is a list of length `dimensions`.
    """
    rng = np.random.RandomState(seed)

    if dimensions is not None:
        signals = [white_noise(step, high, rms=rms, seed=rng.randint(0x7ffffff))
                    for i in range(dimensions)]
        def white_noise_function(t, signals=signals):
            return [signal(t) for signal in signals]
        return white_noise_function

    N = int(float(high) / step)                     # number of samples
    frequencies = np.arange(1, N + 1) * step * 2 * np.pi   # frequency of each
    amplitude = rng.uniform(0, 1, N)                # amplitude for each sample
    phase = rng.uniform(0, 2*np.pi, N)              # phase of each sample

    # compute the rms of the signal
    rawRMS = np.sqrt(np.sum(amplitude**2)/2)
    # rescale
    amplitude = amplitude * rms / rawRMS

    # create a function that computes the bases and weights them by amplitude
    def white_noise_function(t, f=frequencies, a=amplitude, p=phase):
        return np.dot(a, np.sin((f*t)+p))

    return white_noise_function
