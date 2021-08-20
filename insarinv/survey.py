from SimPEG.survey import BaseTimeRx, BaseSurvey, BaseSrc
import properties
import numpy as np
import scipy.sparse as sp


####################################################
# Survey
####################################################


class Survey(BaseSurvey):
    """
    Time domain electromagnetic survey
    """

    source_list = properties.List(
        "A list of sources for the survey",
        properties.Instance("A SimPEG source", BaseSrc),
        default=[],
    )

    def __init__(self, source_list=None, **kwargs):
        super(Survey, self).__init__(source_list, **kwargs)

####################################################
# Source
####################################################


class Src(BaseSrc):

    def __init__(self, location, receiver_list=None, **kwargs):
        if receiver_list is not None:
            kwargs["receiver_list"] = receiver_list
        super(BaseSrc, self).__init__(location=location, **kwargs)


####################################################
# Receiver
####################################################

class Rx(BaseTimeRx):
    """
    Time domain receiver base class

    :param numpy.ndarray locations: receiver locations (ie. :code:`np.r_[x,y,z]`)
    :param numpy.ndarray times: times
    :param string orientation: receiver orientation 'x', 'y' or 'z'
    """

    _P = None

    def __init__(self, times, orientation=None, **kwargs):
        super().__init__(times=times, **kwargs)


    def getP(self, time_mesh):
        if self._P is None:
            P = self.getTimeP(time_mesh)
            n_data = self.times.size
            tmp = np.arange(n_data-1) + 1
            I = np.r_[tmp, tmp].astype(int)
            J = np.r_[tmp, np.zeros(n_data-1)].astype(int)
            data = np.r_[np.ones(n_data-1), -np.ones(n_data-1)]
            Prel = sp.coo_matrix((data, (I, J)), shape=(n_data, n_data))
            self._P = Prel * P
        return self._P

    @property
    def nD(self):
        return self.times.size

    def eval(self, time_mesh, f):
        P = self.getP(time_mesh)
        return P * f['b_t']
