import numpy as np

class eigenspectrum(object):
    """
    A class to process and filter the eigenspectrum (eigenvalues and eigenfunctions)
    obtained from the QEP solver.
    """
    def __init__(self, ev, ef):
        ''' ev = eigenvalues, ef = eigenfuncions '''
        
        self.ev = ev
        self.ef = ef
    
    def pick_first_BZ(self):
        """
        Initializes the eigenspectrum object.

        Args:
            ev (list): A list of complex eigenvalues (kx).
            ef (list): A list of corresponding ngsolve.GridFunction eigenfunctions.
        """
        
        temp1 = []
        temp2 = []
        for i in range(len(self.ev)):
            if round(self.ev[i].real, 3) > -round(np.pi, 3) and round(self.ev[i].real, 3) <= round(np.pi, 3):
                temp1.append(self.ev[i])
                temp2.append(self.ef[i])

        temp1 = np.array(temp1, dtype=complex)
        self.ev_BZ = temp1
        self.ef_BZ = temp2