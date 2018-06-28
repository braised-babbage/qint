import numpy as np


def with_head_position(fn, *positions):
    """Given a unary or binary operator acting on positions 0 and 1,
    returns the corresponding operator acting on the specified positions."""

    swap = np.swapaxes

    if len(positions) == 0:
        return fn
    elif len(positions) == 1:
        p = positions[0]
        return lambda s: swap(fn(swap(s, 0, p)), 0, p)
    elif len(positions) == 2:
        p0,p1 = positions
        
        def swapped(s):
            _s = swap(swap(s, 0, p0), 1, p1)
            val = fn(_s)
            return swap(swap(val, 1, p1), 0, p0)
        return swapped
            

def apply1(gate, pos, state):
    """Applies the unary gate to the state at the given position (qubit).
    Returns the updated state."""
    assert(0 <= pos and pos < state.ndim)

    def update(s):
        return np.dot(gate, s)

    return with_head_position(update, pos)(state)


def apply2(gate, pos0, pos1, state):
    """Applies the binary gate to the state at the given positions (qubits).
    Returns the updated state."""
    assert(0 <= pos0 and pos0 < state.ndim)
    assert(0 <= pos1 and pos1 < state.ndim)
    assert(pos0 != pos1)
    
    def update(s):
        reshaped = np.reshape(s, (4, -1))
        updated = np.dot(gate, reshaped)
        return np.reshape(updated, s.shape)

    return with_head_position(update, pos0, pos1)(state)
    

def pure(bitstring):
    """Construct a pure quantum state concentrated on a given bitstring."""
    assert(all(b in '01' for b in bitstring))
    bits = tuple(int(b) for b in bitstring)
    state = np.zeros([2]*len(bits))
    state[bits] = 1
    return state


def normalized(nparray):
    """Returns a normalized version of nparray."""
    return nparray / np.linalg.norm(npstate)


def collapse(state, pos, val):
    """Collapses the state at the given position to the specified value."""
    assert(val == 0 or val == 1)
    assert(0 <= pos and pos < state.ndim)

    def update(s):
        collapsed = state.copy()
        collapsed[1-val,] = 0
        return normalized(collapsed)
    
    return with_head_position(update)(state)


def probabilities(state, pos):
    """Return a probability distribution on bits, for the given state
    at the given distribution."""
    state = np.swapaxes(state, 0, pos)
    squares = state * np.conj(state)
    return normalized(np.sum(np.reshape(squares, (2,-1)), axis=1))


def measure(state, pos):
    """Measures the state at the given position. Returns an outcome and 
    a collapsed state."""
    p1 = probabilities(state, pos)[1]
    outcome = np.random.binomial(1,p1)
    return outcome, collapse(state, pos, outcome)
    
    
def unary_gate(entries):
    """Construct a unitary single-qubit gate with specified 
    entries (up to a normalization factor)."""
    mat = np.reshape(np.array(entries), (2,2))
    assert(mat.size == 4)
    scale = np.linalg.norm(mat[0,])
    return mat / scale


def controlled_gate(U):
    """Construct a controlled binary gate from a unary gate U."""
    I = np.eye(2)
    zero = np.zeros((2,2))
    return np.vstack([np.hstack([I, zero]),
                      np.hstack([zero, U])])


I = np.eye(2)
H = unary_gate([[1,1],
                [1,-1]])
NOT = unary_gate([[0, 1],
                  [1, 0]])
CNOT = controlled_gate(NOT)
