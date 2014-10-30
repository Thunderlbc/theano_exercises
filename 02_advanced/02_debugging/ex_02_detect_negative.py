# Fill in the TODOs and run the script to see if your implementation works.
import numpy as np
import theano
from theano import function
from theano import tensor as T
from theano.compile import mode


class NegativeVariableError(Exception):
    pass


def get_neg_detection_mode():
    """
    Returns a theano Mode that detects if any negative value occurs in the
    evaluation of a theano function.
    This mode should raise a NegativeVariableError if it ever detects any
    variable having a negative value during the execution of the theano
    function.
    """
    class NegDetectionMode(mode):

        def __init__(self):
            def flatten(l):
                if isinstance(l, (list, tuple)):
                    rval = []
                    for elem in l:
                        if isinstance(elem, (list, tuple)):
                            rval.extend(flatten(l))
                        else:
                            rval.append(elem)
                    else:
                        return rval

            def do_check_on(var, nd, f):
                if var.min < 0:
                    raise NegativeVariableError()

            def neg_check(i, node, fn):
                inputs = fn.inputs
                for x in flatten(inputs):
                    do_check_on(x, node, fn)
                fn()
                outputs = fn.outputs
                for j, x in enumerate(flatten(outputs)):
                    do_check_on(x, node, fn)

            wrap_linker = theano.gof.WrapLinkerMany([theano.gof.OpWiseCLinker()], [neg_check])
            super(NegDetectionMode, self).__init__(wrap_linker, optimizer='fast_run')
    return NegDetectionMode()


if __name__ == "__main__":
    x = T.scalar()
    x.name = 'x'
    y = T.nnet.sigmoid(x)
    y.name = 'y'
    z = - y
    z.name = 'z'
    mode = get_neg_detection_mode()
    f = function([x], z, mode=mode)
    caught = False
    try:
        f(0.)
    except NegativeVariableError:
        caught = True
    if not caught:
        print "You failed to catch a negative value."
        quit(-1)
    f = function([x], y, mode=mode)
    y1 = f(0.)
    f = function([x], y)
    assert np.allclose(f(0.), y1)
    print "SUCCESS!"
