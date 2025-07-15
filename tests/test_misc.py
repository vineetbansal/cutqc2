from cutqc2.cutqc.cutqc.evaluator import measure_state

"""
sigma, effective_state = measure_state(<unmeasured_state>, <measurement_bases>)
  unmeasured_state is the n-bit state from cut circuit
  measurement_bases is the n-length list of measurement bases, LSB to MSB
    (top-wire to bottom-wire).
  sigma denotes the sign (+1, -1) in Eq. 3 of the paper
  effective_state is the state we wish to attribute the measurement to.
  This is the first n-1 bits of the n-bit unmeasured state.
"""


# If all measurement bases are "comp", then sigma=1 and effective_state=unmeasured state.
def test_measure_state0():
    sigma, effective_state = measure_state(0b000, ["comp", "comp", "comp"])
    assert sigma == 1
    assert effective_state == 0


def test_measure_state1():
    sigma, effective_state = measure_state(0b001, ["comp", "comp", "comp"])
    assert sigma == 1
    assert effective_state == 1


def test_measure_state2():
    sigma, effective_state = measure_state(0b010, ["comp", "comp", "comp"])
    assert sigma == 1
    assert effective_state == 2


def test_measure_state3():
    # Eq. 3 in paper:
    #   xx0, xx1 -> +xx  if M_last = I
    # So we have result = (+1, 0b10) = (1, 2)
    sigma, effective_state = measure_state(0b100, ["I", "comp", "comp"])
    assert sigma == 1
    assert effective_state == 2


def test_measure_state4():
    # Eq. 3 in paper:
    #   xx1 -> -xx  if M_last != I
    # So we have result = (-1, 0b110) = (-1, 6)
    sigma, effective_state = measure_state(0b1101, ["Z", "comp", "comp", "comp"])
    assert sigma == -1
    assert effective_state == 6
