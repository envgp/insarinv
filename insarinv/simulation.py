from SimPEG.simulation import BaseSimulation 
from SimPEG import utils, props, Data
import numpy as np
from discretize import TensorMesh
import scipy.sparse as sp
from scipy.signal import medfilt

class InSARSimulation1D(BaseSimulation):
    """docstring for InSARSimulation1D"""

    Sske, SskeMap, SskeDeriv = props.Invertible(
        "Specific skeleton storage of clay (elastic) [m-1]", 
        default=6.39*1e-6
    )

    Sskv, SskvMap, SskvDeriv = props.Invertible(
        "Specific skeleton storage of clay (virgin or inelastic) [m-1]",
        default=3.5*1e-4
    )

    Kv, KvMap, KvDeriv = props.Invertible(
        "Verical hydraulic conductivity [m/s]", 
        default=4.6 *1e-4/31536000
    )

    h_min0, h_min0Map, h_min0Deriv = props.Invertible(
        "Initial preconsolidation head of interbedded clays [m]"
    )

    b_equiv, b_equivMap, b_equivDeriv = props.Invertible(
        "Equivalent thickness of interbedded clays [m]"
    )    

    n_equiv, n_equivMap, n_equivDeriv = props.Invertible(
        "Equivalent number of interbedded clays [dimensionless]"
    )

    h_aquifer, h_aquiferMap, h_aquiferDeriv = props.Invertible(
        "Head of the aquifer (m)"
    )
    
    Sska, SskaMap, SskaDeriv = props.Invertible(
        "Specific skeleton storage of aquifer (elastic)[m-1]", 
        default=1.19*1e-5
    )

    Ssw, SswMap, SswDeriv = props.Invertible(
        "Specific stroage of water [m-1]", 
        default=1.5e-6
    )        

    b_aquifer, b_aquiferMap, b_aquiferDeriv = props.Invertible(
        "Thickness of the aquifer [m]"
    )

    nz = props.Integer(
        "# of vertical cells",
        default=20,
    )

    time_simulation = props.Array(
        "time (s)"
    )

    time_channels = props.Array(
        "time_channels (s)"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @property
    def clay_thickness_equiv(self):
        return self.n_equiv * self.b_equiv

    @property
    def mesh(self):
        if getattr(self, "_mesh", None) is None:
            hz = np.ones(self.nz)*self.b_equiv/2/self.nz
            self._mesh = TensorMesh([hz], x0='N')    
            self._mesh.set_cell_gradient_BC([['dirichlet', 'neumann']])        
        return self._mesh

    @property
    def dt(self):
        return self.time_simulation[1]-self.time_simulation[0]

    @property
    def dz(self):
        return self.mesh.hx[0]

    @property
    def n_time(self):
        return self.time_simulation.size

    @property
    def time_mesh(self):
        if getattr(self, "_time_mesh", None) is None:
            self._time_mesh = TensorMesh([np.diff(self.time_simulation)], x0=[self.time_simulation[0]])
        return self._time_mesh

    @property
    def L(self):
        if getattr(self, "_L", None) is None:
            Grad = self.mesh.cell_gradient
            Div = self.mesh.face_divergence
            self._L = Div@Grad
        return self._L
    
    @property
    def DivB(self):
        if getattr(self, "_DivB", None) is None:
            B = self.mesh.cell_gradient_BC
            Div = self.mesh.face_divergence
            self._DivB = Div@B 
        return self._DivB
    
    @property
    def I(self):
        if getattr(self, "_I", None) is None:
            self._I = utils.speye(self.mesh.n_cells)
        return self._I        

    def getA(self, c):
        """
            Get a system matrix A
        """
        # c = Kv/Ssk (Diffusivity)
        C = utils.sdiag(c)
        A = (C*(self.L)-1/self.dt*self.I)
        return A

    def getRHS(self, c):
        """
            Get a system matrix A
        """
            # c = Kv/Ssk (Diffusivity)
        C = utils.sdiag(c)
        A = (C*(self.L)-1/self.dt*self.I)
        return A


    def fields(self, m):
        
        if m is not None:
            self.model = m
        
        # TODO: Expand this to handle many interbedded clay beds
        # by extending the dimensions
        # y: # of insar soundings
        # z: # of clay beds per sounding
        
        Sske = self.Sske
        Sskv = self.Sskv
        Kv = self.Kv
        Sska = self.Sska
        b_aquifer = self.b_aquifer
        b_equiv = self.b_equiv
        n_equiv = self.n_equiv

        h_min0 = self.h_min0
        nz = self.nz
        dz = self.dz
        n_time = self.n_time
        dt = self.dt
        I = self.I
        n_equiv = self.n_equiv
        mesh = self.mesh

        h = np.ones((n_time, mesh.n_cells), order='C')
        hmin = np.ones((n_time, mesh.n_cells), order='C')
        
        # No need to solve for initial condition as long as Kv is constant
        # h_aquifer should be the size of n_time x n_sounding x n_layer
        # then needs to be surjected to an array with the size of n_cells
        # h[0,:] = Ptocc * h_aquifer[0,:] 

        h[0,:] = self.h_aquifer[0]
        
        # Similarly, Ptocc can be used to populate Kv, Sske, Sskv, h_min0
        e = np.ones(2, dtype=float)
        Ssk_tmp = np.zeros(mesh.n_cells, dtype=float)
        hmin0 = np.ones(mesh.n_cells, dtype=float) * h_min0
        hmin1 = np.ones(mesh.n_cells, dtype=float) * h_min0
        hmin[0,:] = h_min0

        INDS = np.zeros((n_time, mesh.n_cells), dtype=bool, order='C')
        INDSMIN = np.ones((n_time, mesh.nC), dtype=int, order='C') * -1

        # Time loop
        for i_time in range(n_time-1):
            
            h0 = h[i_time,:]
            # Find indicies where head values are 
            # lower than the preconsolidation head
            inds = h[i_time,:]<=hmin1
            bc = self.h_aquifer[i_time+1] * e

            # This part is needed to be changed
            # Sske and Sskv are variable y and z
            # Ssk_tmp[:] = Ptocc*Sske
            # Ssk_tmp[inds] = (Ptocc*Sskv)[inds]

            Ssk_tmp[:] = Sske
            Ssk_tmp[inds] = Sskv

            # Create the diffusivity, c
            c = Kv/Ssk_tmp
            C = utils.sdiag(c)

            # Solve a system: Ah = rhs
            A = (C@(self.L)-1/dt*I)
            rhs = -C@(self.DivB@bc) - h0/dt    
            Ainv = self.Solver(A)        
            h1 = Ainv*rhs    
            h[i_time+1,:] = h1

            # Update the preconsolidation head and required indices for 
            # the data projection.

            head_tmp = np.vstack((hmin0, h[:i_time+1,:]))
            hmin1 = head_tmp.min(axis=0)
            hmin[i_time+1,:] = hmin1    
            INDS[i_time+1, inds] = True
            ind_tmp = np.argmin(head_tmp, axis=0) - 1
            INDSMIN[i_time+1,:] =  ind_tmp
        
        # Below requires some thoughts for the expansion. 
        # But, mostly book keeping. 

        # For evaluating permanent subsidence from clays
        tmp = utils.mkvc(INDSMIN + np.arange(nz) * n_time)
        inds_tmp = utils.mkvc(INDSMIN) !=-1

        I = np.arange(h.size)[inds_tmp]
        J = tmp[inds_tmp]
        data = np.ones(inds_tmp.sum())
        n = h.size
        Pmin = sp.coo_matrix((data, (I, J)), shape=(n,n)).tocsr()

        data = np.ones(n)
        I = np.arange(n_time).repeat(nz)
        J = np.array([np.arange(nz) * n_time + ii for ii in range(n_time)]).ravel()
        Psum = sp.coo_matrix((data, (I, J)), shape=(n_time,n)).tocsr()

        emin_0 = (~inds_tmp).astype(float)
        Cv = 2*Sskv * n_equiv * dz
        
        # For evaluating elastic subsidence from clays
        inds_e = np.argwhere(utils.mkvc(INDS)==True).ravel()

        I = np.r_[np.arange(n), np.arange(n-1)+1]
        J = np.r_[np.arange(n), np.arange(n-1)]
        e = np.ones(n_time)
        e[0] = 0
        diag = np.tile(e, (nz))
        e_1 = -np.ones(n_time)
        e_1[0] = 0
        lower = np.r_[-np.ones(n_time-1), np.tile(e_1, (nz-1))]
        data = np.r_[diag, lower]
        data[np.in1d(I, inds_e)] = 0
        Pdiff = sp.coo_matrix((data, (I, J)), shape=(n,n))
        Ce = 2 * Sske * n_equiv * dz
        
        # For evaluating elastic subsidence from clays
        I = np.hstack([np.arange(n_time-ii) + ii for ii in range(n_time)])
        J = np.hstack([np.ones(n_time-ii)*ii for ii in range(n_time)])
        data = np.ones_like(I)
        # Pcumsum = sp.coo_matrix((data, (I, J)), shape=(n_time,n_time)).tocsr()

        Ca = -Sska * b_aquifer
        e = np.ones(n_time-1)
        I = np.r_[np.arange(n_time-1) + 1, np.arange(n_time-1) + 1]
        J = np.r_[np.arange(n_time-1) + 1, np.zeros(n_time-1)]
        data = np.r_[-1*np.ones(n_time-1), np.ones(n_time-1)]
        Pa = sp.coo_matrix((data, (I, J)), shape=(n_time,n_time)).tocsr()
        # TODO: change P matricies such that it can handle band-limited delattra
        # Psum, Pcumsum, Pa
        # Calculate subsidence of each of the three sources
        b_v = Cv * (Psum @ (Pmin @ utils.mkvc(h) + (emin_0-1) * h_min0))
        # b_v = Cv * (Psum @ (Pmin @ utils.mkvc(h) - h_min0))
        # there seems correlated error with time steps when Ssk is changing
        b_v = np.r_[0, np.cumsum(medfilt(np.diff(b_v)))]
        # b_e = Ce * (Pcumsum @ (Psum @ (Pdiff @ utils.mkvc(h))))
        b_e = Ce * (Psum @ (utils.mkvc(h) - Pmin @ utils.mkvc(h)))
        b_a = Ca * (Pa @ self.h_aquifer)

        b_t = b_v + b_e + b_a        
                
        f = {}
        f['Psum'] = Psum
        f['Pmin'] = Pmin
        # f['Pcumsum'] = Pcumsum
        f['Pdiff'] = Pdiff
        f['Pa'] = Pa
        f['emin_0'] = emin_0

        f['h'] = h
        f['hmin'] = hmin
        f['b_t'] = b_t
        f['b_v'] = b_v
        f['b_e'] = b_e
        f['b_a'] = b_a

        for prop in self._clear_on_update:
            delattr(self, prop)
        return f
    
    def dpred(self, m=None, f=None):
        """
        dpred(m, f=None)
        Create the projected data from a model.
        The fields, f, (if provided) will be used for the predicted data
        instead of recalculating the fields (which may be expensive!).

        .. math::

            d_\\text{pred} = P(f(m))

        Where P is a projection of the fields onto the data space.
        """
        if self.survey is None:
            raise AttributeError(
                "The survey has not yet been set and is required to compute "
                "data. Please set the survey for the simulation: "
                "simulation.survey = survey"
            )

        if f is None:
            if m is None:
                m = self.model

            f = self.fields(m)

        data = Data(self.survey)
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                data[src, rx] = rx.eval(self.time_mesh, f)
        return utils.mkvc(data)

    # Use the perturbation method, central difference. 
    # Cost is the two forward simulations
    # Perturbation factor is hard-coded as 0.01.
    # Need to test see if they can pass the order tests. 

    def get_J_Sskv(self, m, f=None, adjoint=False):
        Sskv_current = self.Sskv
        dSskv = 0.01 * Sskv_current
        self.Sskv = Sskv_current - dSskv
        f1 = self.fields([])
        self.Sskv = Sskv_current + dSskv
        f2 = self.fields([])
        self.Sskv = Sskv_current
        J_Sskv = (f2['b_t']-f1['b_t']) / (2*dSskv)
        return J_Sskv.reshape([-1,1])

    def get_J_Sske(self, m, f=None, adjoint=False):
        Sske_current = self.Sske
        dSske = 0.01 * Sske_current
        self.Sske = Sske_current - dSske
        f1 = self.fields([])
        self.Sske = Sske_current + dSske
        f2 = self.fields([])
        self.Sske = Sske_current
        J_Sske = (f2['b_t']-f1['b_t']) / (2*dSske)        
        return J_Sske.reshape([-1,1])

    def get_J_Kv(self, m, f=None, adjoint=False):
        Kv_current = self.Kv
        dKv = 0.01 * Kv_current
        self.Kv = Kv_current - dKv
        f1 = self.fields([])
        self.Kv = Kv_current + dKv
        f2 = self.fields([])
        self.Kv = Kv_current
        J_Kv = (f2['b_t']-f1['b_t']) / (2*dKv)        
        return J_Kv.reshape([-1,1])

    def get_J_b_equiv(self, m, f=None, adjoint=False):
        b_equiv_current = self.b_equiv
        db_equiv = 0.01 * b_equiv_current
        self.b_equiv = b_equiv_current - db_equiv
        f1 = self.fields([])
        self.b_equiv = b_equiv_current + db_equiv
        f2 = self.fields([])
        self.b_equiv = b_equiv_current
        J_b_equiv = (f2['b_t']-f1['b_t']) / (2*db_equiv)        
        return J_b_equiv.reshape([-1,1])

    def get_J_h_min0(self, m, f=None, adjoint=False):
        h_min0_current = self.h_min0
        dh_min0 = 0.01 * h_min0_current
        self.h_min0 = h_min0_current - dh_min0
        f1 = self.fields([])
        self.h_min0 = h_min0_current + dh_min0
        f2 = self.fields([])
        self.h_min0 = h_min0_current
        J_h_min0 =  (f2['b_t']-f1['b_t']) / (2*dh_min0)     
        return J_h_min0.reshape([-1,1])

        # test below idea -> it is not working, so reality is a bit more complicated.
        # return Cv * (Psum @ (emin_0-1))

    def get_J_n_equiv(self, m, f=None, adjoint=False):        
        if f is None:
            f = self.fields(m)
        J_n_equiv = f['b_v']/self.n_equiv + f['b_e']/self.n_equiv
        return J_n_equiv.reshape([-1,1])
    
    def get_J_Sska(self, m, f=None, adjoint=False):
        if f is None:
            f = self.fields(m)
        Pa = f['Pa']
        J_Sska = self.b_aquifer * (Pa * self.h_aquifer)
        return J_Sska.reshape([-1,1])

    def get_J_b_aquifer(self, m, f=None, adjoint=False):
        Pa = f['Pa']
        J_b_aquifer = self.Sska * (Pa @ self.h_aquifer)
        return J_b_aquifer.reshape([-1,1])

    def get_J_h_aquifer(self, m, f=None, adjoint=False):
        raise Exception("Not implemented yet!")
        if f is None:
            f = self.fields(m, f=f)
        return self.get_J_h_aquifer_diff(m, f=f)

    def get_J_h_aquifer_diff(self, m, f=None, adjoint=False):        
        h_aquifer_current = self.h_aquifer.copy()
        J_h_aquifer = np.zeros((self.n_time, self.h_aquifer.size), dtype=float)
        for ii in range(len(self.h_aquifer)):
            h_aquifer_ii = h_aquifer_current[ii]
            dh_aquifer_ii = 0.01 * h_aquifer_ii
            self.h_aquifer[ii] = h_aquifer_ii - dh_aquifer_ii
            f1 = self.fields([])
            self.h_aquifer[ii] = h_aquifer_ii + dh_aquifer_ii
            f2 = self.fields([])
            J_h_aquifer[:,ii] =  (f2['b_t']-f1['b_t']) / (2*dh_aquifer_ii)     
        self.h_aquifer = h_aquifer_current
        return J_h_aquifer

    def get_J(self, m, f=None):
        pass

    def Jvec(self, m, v, f=None):
        # Output is nD x 1 vector
        Jvec = np.zeros(survey.nD, dtype=float)
        Jvec += self.get_J_Sskv(m, f=f) @ (self.SskvDeriv @ v)
        Jvec += self.get_J_Sske(m, f=f) @ (self.SskeDeriv @ v)
        Jvec += self.get_J_Kv(m, f=f) @ (self.KvDeriv @ v)
        Jvec += self.get_J_h_min0(m, f=f) @ (self.h_min0Deriv @ v)
        Jvec += self.get_J_n_equiv(m, f=f) @ (self.n_equivDeriv @ v)
        Jvec += self.get_J_b_equiv(m, f=f) @ (self.b_equivDeriv @ v)
        Jvec += self.get_J_Sska(m, f=f) @ (self.SskaDeriv @ v)
        Jvec += self.get_J_b_aquifer(m, f=f) @ (self.b_aquiferDeriv @ v)
        # May be this one could be optional for not to store J?
        Jvec += self.get_J_h_aquifer(m, f=f) @ (self.h_aquiferDeriv @ v)
        return Jvec
    
    def Jtvec(self, m, v, f=None):
        # Output is M x 1 vector
        Jtvec = np.zeros(m.size, dtype=float)
        Jtvec += self.SskvDeriv.T * (self.get_J_Sskv(m, f=f, adjoint=True) @ v)
        Jtvec += self.SskeDeriv.T * (self.get_J_Sske(m, f=f, adjoint=True) @ v)
        Jtvec += self.KvDeriv.T * (self.get_J_Kv(m, f=f, adjoint=True) @ v)
        Jtvec += self.h_min0Deriv.T * (self.get_J_h_min0(m, f=f, adjoint=True) @ v)
        Jtvec += self.n_equivDeriv.T * (self.get_J_n_equiv(m, f=f, adjoint=True) @ v)
        Jtvec += self.b_equivDeriv.T * (self.get_J_b_equiv(m, f=f, adjoint=True) @ v)
        Jtvec += self.SskaDeriv.T * (self.get_J_Sska(m, f=f, adjoint=True) @ v)
        Jtvec += self.b_aquiferDeriv.T * (self.get_J_b_aquifer(m, f=f, adjoint=True) @ v)
        # May be this one could be optional for not to store J?
        Jtvec += self.h_aquiferDeriv.T * (self.get_J_h_aquifer(m, f=f, adjoint=True) @ v)     
        return Jtvec

    @property
    def _clear_on_update(self):
        """
        These matrices are deleted if there is an update to the conductivity
        model
        """
        return [
            "_L",
            "_DivB",
            "_mesh",
        ]