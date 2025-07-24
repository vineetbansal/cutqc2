from cutqc2.core.cut_circuit import CutCircuit

"""
The coefficients and kets used for each term in the expansion of the trace
operators for each of the Pauli bases are (eq. 2 in paper).
The "0" is a placeholder initializer, which results in no substitution.

     0   1   +   i
0    1     
I    1   1
X   -1  -1   2
Y   -1  -1       2
Z    1  -1

"""


def test_zeros():
    initializations = ["zero", "zero", "zero"]
    coeffs_kets = CutCircuit.get_initializations(initializations)

    # There are no Pauli bases - we simply get what we passed in.
    assert len(coeffs_kets) == 1
    assert (1, ("zero", "zero", "zero")) in coeffs_kets


def testI():
    initializations = ["I", "zero"]
    coeffs_kets = CutCircuit.get_initializations(initializations)

    # important - don't rely on the order in which the coefficients and kets
    # are returned, but simply assert they're there.
    assert len(coeffs_kets) == 2
    # kets should populate at the location of the Pauli basis
    assert (1, ("zero", "zero")) in coeffs_kets
    assert (1, ("one", "zero")) in coeffs_kets


def testX():
    initializations = ["zero", "X", "zero", "zero"]
    coeffs_kets = CutCircuit.get_initializations(initializations)

    # important - don't rely on the order in which the coefficients and kets
    # are returned, but simply assert they're there.
    assert len(coeffs_kets) == 3
    # kets should populate at the location of the Pauli basis
    assert (-1, ("zero", "one", "zero", "zero")) in coeffs_kets
    assert (-1, ("zero", "zero", "zero", "zero")) in coeffs_kets
    assert (2, ("zero", "plus", "zero", "zero")) in coeffs_kets


def testY():
    initializations = ["zero", "Y", "zero"]
    coeffs_kets = CutCircuit.get_initializations(initializations)

    # important - don't rely on the order in which the coefficients and kets
    # are returned, but simply assert they're there.
    assert len(coeffs_kets) == 3
    # kets should populate at the location of the Pauli basis
    assert (-1, ("zero", "one", "zero")) in coeffs_kets
    assert (-1, ("zero", "zero", "zero")) in coeffs_kets
    assert (2, ("zero", "plusI", "zero")) in coeffs_kets


def testZ():
    initializations = ["zero", "zero", "Z", "zero"]
    coeffs_kets = CutCircuit.get_initializations(initializations)

    # important - don't rely on the order in which the coefficients and kets
    # are returned, but simply assert they're there.
    assert len(coeffs_kets) == 2
    # kets should populate at the location of the Pauli basis
    assert (1, ("zero", "zero", "zero", "zero")) in coeffs_kets
    assert (-1, ("zero", "zero", "one", "zero")) in coeffs_kets


def testIzeroX():
    # The input bases can contain multiple Pauli bases
    initializations = ["I", "zero", "X"]
    coeffs_kets = CutCircuit.get_initializations(initializations)

    # important - don't rely on the order in which the coefficients and kets
    # are returned, but simply assert they're there.
    assert len(coeffs_kets) == 6
    # kets should populate at the location of the Pauli bases, and the
    # coefficients multiplied together.
    assert (2, ("zero", "zero", "plus")) in coeffs_kets
    assert (-1, ("zero", "zero", "zero")) in coeffs_kets
    assert (-1, ("zero", "zero", "one")) in coeffs_kets
    assert (2, ("one", "zero", "plus")) in coeffs_kets
    assert (-1, ("one", "zero", "zero")) in coeffs_kets
    assert (-1, ("one", "zero", "one")) in coeffs_kets
