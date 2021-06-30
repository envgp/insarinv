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

    verbose = False

    _J_Sske = None
    _J_Sskv = None
    _J_Sska = None
    _J_Kv = None
    _J_h_min0 = None
    _J_b_equiv = None
    _J_n_equiv = None
    _J_h_aquifer = None
    _J_b_aquifer = None
    fd_factor = 0.01


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @property
    def clay_thickness_equiv(self):
        return self.n_equiv * self.b_equiv

    @property
    def mesh(self):
        if getattr(self, "_mesh", None) is None:
            if self.verbose:
                print (">> construct mesh")
            hz = np.ones(self.nz)*self.b_equiv/2/self.nz
            self._mesh = TensorMesh([hz], x0='N')    
            self._mesh.set_cell_gradient_BC([['dirichlet', 'neumann']])        
        return self._mesh

    def update_mesh(self, b_equiv):
        hz = np.ones(self.nz)*b_equiv/2/self.nz
        self._mesh = TensorMesh([hz], x0='N')
        self._mesh.set_cell_gradient_BC([['dirichlet', 'neumann']])

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
            self._time_mesh = TensorMesh(
                [np.diff(self.time_simulation)], 
                x0=[self.time_simulation[0]]
            )
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

    # def fields(self, m):
        
    #     if m is not None:
    #         self.model = m
        
    #     # TODO: Expand this to handle many interbedded clay beds
    #     # by extending the dimensions
    #     # y: # of insar soundings
    #     # z: # of clay beds per sounding
        
    #     Sske = np.asscalar(self.Sske.ravel())
    #     Sskv = np.asscalar(self.Sskv.ravel())
    #     Kv = np.asscalar(self.Kv.ravel())
    #     Sska = np.asscalar(self.Sska.ravel())
    #     b_aquifer = np.asscalar(self.b_aquifer.ravel())
    #     b_equiv = np.asscalar(self.b_equiv.ravel())
    #     n_equiv = np.asscalar(self.n_equiv.ravel())
    #     h_min0 = np.asscalar(self.h_min0.ravel())

    #     nz = self.nz
    #     dz = self.dz
    #     n_time = self.n_time
    #     dt = self.dt
    #     I = self.I
    #     mesh = self.mesh

    #     h = np.ones((n_time, mesh.n_cells), order='C')
    #     hmin = np.ones((n_time, mesh.n_cells), order='C')
        
    #     # No need to solve for initial condition as long as Kv is constant
    #     # h_aquifer should be the size of n_time x n_sounding x n_layer
    #     # then needs to be surjected to an array with the size of n_cells
    #     # h[0,:] = Ptocc * h_aquifer[0,:] 

    #     h[0,:] = self.h_aquifer[0]
        
    #     # Similarly, Ptocc can be used to populate Kv, Sske, Sskv, h_min0
    #     e = np.ones(2, dtype=float)
    #     Ssk_tmp = np.zeros(mesh.n_cells, dtype=float)
    #     hmin0 = np.ones(mesh.n_cells, dtype=float) * h_min0
    #     hmin1 = np.ones(mesh.n_cells, dtype=float) * h_min0
    #     hmin[0,:] = h_min0

    #     INDS = np.zeros((n_time, mesh.n_cells), dtype=bool, order='C')
    #     INDSMIN = np.ones((n_time, mesh.nC), dtype=int, order='C') * -1
    #     c_matrix = np.zeros((n_time, mesh.n_cells), dtype=float, order='C')
    #     # Time loop
    #     for i_time in range(n_time-1):
            
    #         h0 = h[i_time,:]
    #         # Find indicies where head values are 
    #         # lower than the preconsolidation head
    #         inds = h[i_time,:]<=hmin1
    #         bc = self.h_aquifer[i_time+1] * e

    #         # This part is needed to be changed
    #         # Sske and Sskv are variable y and z
    #         # Ssk_tmp[:] = Ptocc*Sske
    #         # Ssk_tmp[inds] = (Ptocc*Sskv)[inds]

    #         Ssk_tmp[:] = Sske
    #         Ssk_tmp[inds] = Sskv

    #         # Create the diffusivity, c
    #         c = Kv/Ssk_tmp
    #         C = utils.sdiag(c)
    #         c_matrix[i_time+1, :] = c
    #         # Solve a system: Ah = rhs
    #         A = (C@(self.L)-1/dt*I)
    #         rhs = -C@(self.DivB@bc) - h0/dt    
    #         Ainv = self.Solver(A)        
    #         h1 = Ainv*rhs    
    #         h[i_time+1,:] = h1

    #         # Update the preconsolidation head and required indices for 
    #         # the data projection.

    #         head_tmp = np.vstack((hmin0, h[:i_time+1,:]))
    #         hmin1 = head_tmp.min(axis=0)
    #         hmin[i_time+1,:] = hmin1    
    #         INDS[i_time+1, inds] = True
    #         ind_tmp = np.argmin(head_tmp, axis=0) - 1
    #         INDSMIN[i_time+1,:] =  ind_tmp
        
    #     # Below requires some thoughts for the expansion. 
    #     # But, mostly book keeping. 

    #     # For evaluating permanent subsidence from clays
    #     tmp = utils.mkvc(INDSMIN + np.arange(nz) * n_time)
    #     inds_tmp = utils.mkvc(INDSMIN) !=-1

    #     I = np.arange(h.size)[inds_tmp]
    #     J = tmp[inds_tmp]
    #     data = np.ones(inds_tmp.sum())
    #     n = h.size
    #     Pmin = sp.coo_matrix((data, (I, J)), shape=(n,n)).tocsr()

    #     data = np.ones(n)
    #     I = np.arange(n_time).repeat(nz)
    #     J = np.array([np.arange(nz) * n_time + ii for ii in range(n_time)]).ravel()
    #     Psum = sp.coo_matrix((data, (I, J)), shape=(n_time,n)).tocsr()

    #     emin_0 = (~inds_tmp).astype(float)
    #     Cv = (2*Sskv * n_equiv * dz)
        
    #     # For evaluating elastic subsidence from clays
    #     inds_e = np.argwhere(utils.mkvc(INDS)==True).ravel()

    #     I = np.r_[np.arange(n), np.arange(n-1)+1]
    #     J = np.r_[np.arange(n), np.arange(n-1)]
    #     e = np.ones(n_time)
    #     e[0] = 0
    #     diag = np.tile(e, (nz))
    #     e_1 = -np.ones(n_time)
    #     e_1[0] = 0
    #     lower = np.r_[-np.ones(n_time-1), np.tile(e_1, (nz-1))]
    #     data = np.r_[diag, lower]
    #     data[np.in1d(I, inds_e)] = 0
    #     Pdiff = sp.coo_matrix((data, (I, J)), shape=(n,n))
    #     Ce = (2 * Sske * n_equiv * dz)
        
    #     # For evaluating elastic subsidence from clays
    #     I = np.hstack([np.arange(n_time-ii) + ii for ii in range(n_time)])
    #     J = np.hstack([np.ones(n_time-ii)*ii for ii in range(n_time)])
    #     data = np.ones_like(I)
    #     # Pcumsum = sp.coo_matrix((data, (I, J)), shape=(n_time,n_time)).tocsr()

    #     Ca = (-Sska * b_aquifer) * np.ones(n_time)
    #     e = np.ones(n_time-1)
    #     I = np.r_[np.arange(n_time-1) + 1, np.arange(n_time-1) + 1]
    #     J = np.r_[np.arange(n_time-1) + 1, np.zeros(n_time-1)]
    #     data = np.r_[-1*np.ones(n_time-1), np.ones(n_time-1)]
    #     Pa = sp.coo_matrix((data, (I, J)), shape=(n_time,n_time)).tocsr()
    #     b_v = Cv * (Psum @ (Pmin @ utils.mkvc(h) + (emin_0-1) * h_min0))
    #     # b_v = Cv * (Psum @ (Pmin @ utils.mkvc(h) - h_min0))
    #     # there seems correlated error with time steps when Ssk is changing
    #     # median filter
    #     b_v = np.r_[0, np.cumsum(medfilt(np.diff(b_v)))]
    #     # b_e = Ce * (Pcumsum @ (Psum @ (Pdiff @ utils.mkvc(h))))
    #     b_e = Ce * (Psum @ (utils.mkvc(h) - Pmin @ utils.mkvc(h)))
    #     b_a = Ca * (Pa @ self.h_aquifer)

    #     Pf = (Cv * Psum @ Pmin + Ce * Psum@(utils.speye(h.size)-Pmin))

    #     b_t = b_v + b_e + b_a        
                
    #     f = {}
    #     f['Psum'] = Psum
    #     f['Pmin'] = Pmin
    #     # f['Pcumsum'] = Pcumsum
    #     f['Pdiff'] = Pdiff
    #     f['Pa'] = Pa
    #     f['emin_0'] = emin_0

    #     f['h'] = h
    #     f['hmin'] = hmin
    #     f['b_t'] = b_t
    #     f['b_v'] = b_v
    #     f['b_e'] = b_e
    #     f['b_a'] = b_a
    #     f['Pf'] = Pf
    #     f['c_matrix'] = c_matrix

    #     for prop in self._clear_on_update:
    #         delattr(self, prop)
    #     return f

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
                "self.survey = survey"
            )

        if f is None:
            if m is None:
                m = self.model

            f = self.fields(m)

        data = Data(self.survey)
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                data[src, rx] = rx.eval(self.time_mesh, f)
        # print (m, self.Sske, np.linalg.norm(utils.mkvc(data)), 'kang')
        return utils.mkvc(data)

    def fields(self, m):

        if m is not None:
            self.model = m
         
        # TODO: Expand this to handle many interbedded clay beds
        # by extending the dimensions
        # y: # of insar soundings
        # z: # of clay beds per sounding
        Sske = np.asscalar(self.Sske.ravel())
        Sskv = np.asscalar(self.Sskv.ravel())
        Kv = np.asscalar(self.Kv.ravel())
        Sska = np.asscalar(self.Sska.ravel())
        b_aquifer = np.asscalar(self.b_aquifer.ravel())
        b_equiv = np.asscalar(self.b_equiv.ravel())
        n_equiv = np.asscalar(self.n_equiv.ravel())
        h_min0 = np.asscalar(self.h_min0.ravel())

        nz = self.nz
        dz = self.dz
        n_time = self.n_time
        dt = self.dt
        I = self.I
        mesh = self.mesh

        h = np.ones((mesh.n_cells, n_time), order='F')
        hmin = np.ones((mesh.n_cells, n_time), order='F')

        # No need to solve for initial condition as long as Kv is constant
        # h_aquifer should be the size of n_time x n_sounding x n_layer
        # then needs to be surjected to an array with the size of n_cells
        # h[0,:] = Ptocc * h_aquifer[0,:] 

        h[:, 0] = self.h_aquifer[0]

        # Similarly, Ptocc can be used to populate Kv, Sske, Sskv, h_min0
        e = np.ones(2, dtype=float)
        Ssk_tmp = np.zeros(mesh.n_cells, dtype=float)
        hmin0 = np.ones(mesh.n_cells, dtype=float) * h_min0
        hmin1 = np.ones(mesh.n_cells, dtype=float) * h_min0
        hmin[:, 0] = h_min0

        INDS = np.zeros((mesh.n_cells, n_time), dtype=bool, order='F')
        INDSMIN = np.ones((mesh.n_cells, n_time, ), dtype=int, order='F') * -1
        c_matrix = np.zeros((mesh.n_cells, n_time), dtype=float, order='F')
        # Time loop
        for i_time in range(n_time-1):

            h0 = h[:, i_time]
            # Find indicies where head values are 
            # lower than the preconsolidation head
            inds = h[:, i_time]<=hmin1
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
            c_matrix[:, i_time+1] = c
            # Solve a system: Ah = rhs
            A = (C@(self.L)-1/dt*I)
            rhs = -C@(self.DivB@bc) - h0/dt    
            Ainv = self.Solver(A)        
            h1 = Ainv*rhs    
            h[::,i_time+1] = h1

            # Update the preconsolidation head and required indices for 
            # the data projection.

            head_tmp = np.hstack((hmin0.reshape([-1,1]), h[:, :i_time+1]))
            hmin1 = head_tmp.min(axis=1)
            hmin[:, i_time+1] = hmin1    
            INDS[inds, i_time+1] = True
            ind_tmp = np.argmin(head_tmp, axis=1) - 1
            INDSMIN[:, i_time+1] =  ind_tmp

            n = h.size

        i_inds = utils.mkvc(np.tile(np.arange(nz).reshape([-1,1]),(1,n_time)))
        j_inds = utils.mkvc(INDSMIN)
        inds_active = j_inds != -1
        I = np.arange(n)[inds_active]
        J = utils.sub2ind((nz, n_time), np.c_[i_inds[inds_active], j_inds[inds_active]])

        # Below requires some thoughts for the expansion. 
        # But, mostly book keeping. 

        # For evaluating permanent subsidence from clays
        i_inds_min = utils.mkvc(np.tile(np.arange(nz).reshape([-1,1]),(1,n_time)))
        j_inds_min = utils.mkvc(INDSMIN)
        inds_active = j_inds_min != -1
        I = np.arange(n)[inds_active]
        J = utils.sub2ind((nz, n_time), np.c_[i_inds_min[inds_active], j_inds_min[inds_active]])

        data = np.ones(inds_active.sum())
        Pmin = sp.coo_matrix((data, (I, J)), shape=(n,n)).tocsr()

        data = np.ones(n)
        I = np.arange(n_time).repeat(nz)
        J = np.array([np.arange(nz) + ii*nz for ii in range(n_time)]).ravel()
        Psum = sp.coo_matrix((data, (I, J)), shape=(n_time,n)).tocsr()
        emin_0 = (~inds_active).astype(float)

        Cv = (2*Sskv * n_equiv * dz)
        Ce = (2 * Sske * n_equiv * dz)

        I = np.hstack([np.arange(n_time-ii) + ii for ii in range(n_time)])
        J = np.hstack([np.ones(n_time-ii)*ii for ii in range(n_time)])
        data = np.ones_like(I)
        Pcumsum = sp.coo_matrix((data, (I, J)), shape=(n_time,n_time)).tocsr()

        Ca = (Sska * b_aquifer) * np.ones(n_time)
        e = np.ones(n_time-1)
        I = np.r_[np.arange(n_time-1) + 1, np.arange(n_time-1) + 1]
        J = np.r_[np.arange(n_time-1) + 1, np.zeros(n_time-1)]
        data = np.r_[np.ones(n_time-1), -1*np.ones(n_time-1)]
        Pa = sp.coo_matrix((data, (I, J)), shape=(n_time,n_time)).tocsr()
        b_v = Cv * (Pa) @ (Psum @ (Pmin @ utils.mkvc(h) + (emin_0-1) * h_min0))
        # there seems correlated error with time steps when Ssk is changing
        # median filter
        # b_v = np.r_[0, np.cumsum(medfilt(np.diff(b_v)))]
        b_e = Ce * (Pa) @ (Psum @ (utils.mkvc(h) - Pmin @ utils.mkvc(h)))
        b_a = Ca * (Pa @ self.h_aquifer)

        Pf = (Cv * Pa @ Psum @ Pmin + Ce * Pa @ Psum @ (utils.speye(h.size)-Pmin))

        b_t = b_v + b_e + b_a        

        f = {}
        f['Psum'] = Psum
        f['Pmin'] = Pmin
        # f['Pcumsum'] = Pcumsum
        # f['Pdiff'] = Pdiff
        f['Pa'] = Pa
        f['emin_0'] = emin_0

        f['h'] = h
        f['hmin'] = hmin
        f['b_t'] = b_t
        f['b_v'] = b_v
        f['b_e'] = b_e
        f['b_a'] = b_a
        f['Pf'] = Pf
        f['c_matrix'] = c_matrix

        for prop in self._clear_on_update:
            delattr(self, prop)
        return f

    # Use the perturbation method, central difference. 
    # Cost is the two forward simulations
    # Perturbation factor is hard-coded as 0.01.
    # Need to test see if they can pass the order tests. 
        
    def perturb_model(self, name, sign=1, factor=0.01, floor=0.):
        model_dict ={
            "Sske": self.Sske,
            "Sskv": self.Sskv,
            "Kv": self.Kv,
            "Sska": self.Sska,
            "b_aquifer": self.b_aquifer,
            "b_equiv": self.b_equiv,
            "n_equiv": self.n_equiv,
            "h_min0": self.h_min0,
        }
        model_dict[name] = model_dict[name] + model_dict[name]*sign*factor
         # + sign*floor
        return model_dict

    def get_J_Sske(self, m, f=None):
        
        if self._J_Sske is not None:
            return self._J_Sske
        else:        
            # assume exp map
            dm = (self.SskeMap.maps[1].P*m)*0.01
            m_pert = m - self.SskeMap.maps[1].P.T @ (dm)
            d1 = self.dpred(m_pert)
            m_pert = m + self.SskeMap.maps[1].P.T @ (dm)
            d2 = self.dpred(m_pert)
            self.model = m
            J_log_Sske = (d2-d1) / (2*dm)
            self._J_Sske =  J_log_Sske / self.Sske.ravel()
            self._J_Sske = self._J_Sske.reshape([-1,1])
        return self._J_Sske

    def get_J_Sskv(self, m, f=None):
        
        if self._J_Sskv is not None:
            return self._J_Sskv
        else:        
            dm = (self.SskvMap.maps[1].P*m)*0.01
            m_pert = m - self.SskvMap.maps[1].P.T @ (dm)
            d1 = self.dpred(m_pert)
            m_pert = m + self.SskvMap.maps[1].P.T @ (dm)
            d2 = self.dpred(m_pert)
            self.model = m
            J_log_Sskv = (d2-d1) / (2*dm)
            self._J_Sskv = J_log_Sskv / self.Sskv.ravel()
            self._J_Sskv = self._J_Sskv.reshape([-1,1])
        return self._J_Sskv

    def get_J_Kv(self, m, f=None):
        
        if self._J_Kv is not None:
            return self._J_Kv
        else:        
            
            dm = (self.KvMap.maps[1].P*m)*0.01
            m_pert = m - self.KvMap.maps[1].P.T @ (dm)
            d1 = self.dpred(m_pert)
            m_pert = m + self.KvMap.maps[1].P.T @ (dm)
            d2 = self.dpred(m_pert)
            self.model = m
            J_log_Kv = (d2-d1) / (2*dm)
            self._J_Kv = J_log_Kv / self.Kv.ravel()
            self._J_Kv = self._J_Kv.reshape([-1,1])
        return self._J_Kv

    def get_J_b_equiv(self, m, f=None):
        
        if self._J_b_equiv is not None:
            return self._J_b_equiv
        else:        
            self.model = m
            dm = (self.b_equivMap.P*m)*0.01
            m_pert = m - self.b_equivMap.P.T @ (dm)
            d1 = self.dpred(m_pert)
            m_pert = m + self.b_equivMap.P.T @ (dm)
            d2 = self.dpred(m_pert)
            self._J_b_equiv = (d2-d1) / (2*dm)
            self._J_b_equiv = self._J_b_equiv.reshape([-1,1])
        return self._J_b_equiv

    def get_J_h_min0(self, m, f=None):
        
        if self._J_h_min0 is not None:
            return self._J_h_min0
        else:        
            self.model = m
            dm = (self.h_min0Map.P*m)*0.01 + 5.
            m_pert = m - self.h_min0Map.P.T @ (dm)
            d1 = self.dpred(m_pert)
            m_pert = m + self.h_min0Map.P.T @ (dm)
            d2 = self.dpred(m_pert)
            self._J_h_min0 = (d2-d1) / (2*dm)
            self._J_h_min0 = self._J_h_min0.reshape([-1,1])
        return self._J_h_min0

    def get_J_n_equiv(self, m, f=None):      
        if self._J_n_equiv is not None:
            return self._J_n_equiv
        else:        
            self.model = m
            if f is None:
                f = self.fields(m)
            tmp = f['b_v']/self.n_equiv + f['b_e']/self.n_equiv
            self._J_n_equiv = []
            for src in self.survey.source_list:
                for rx in src.receiver_list:
                    Prx = rx.getP(self.time_mesh)
                    self._J_n_equiv.append(Prx * tmp)
            self._J_n_equiv = np.hstack(self._J_n_equiv).reshape([-1,1])
            return self._J_n_equiv
    
    def get_J_Sska(self, m, f=None):
        if self._J_Sska is not None:
            return self._J_Sska
        else:        
            self.model = m
            if f is None:
                f = self.fields(m)

            Pa = f['Pa']
            tmp = self.b_aquifer * (Pa @ self.h_aquifer)
            self._J_Sska = []
            for src in self.survey.source_list:
                for rx in src.receiver_list:
                    Prx = rx.getP(self.time_mesh)
                    self._J_Sska.append(Prx * tmp)
            self._J_Sska = np.hstack(self._J_Sska).reshape([-1,1])
            return self._J_Sska

    def get_J_b_aquifer(self, m, f=None):
        if self._J_b_aquifer is not None:
            return self._J_b_aquifer
        else:                
            self.model = m
            if f is None:
                f = self.fields(m)
            Pa = f['Pa']
            tmp = self.Sska * (Pa @ self.h_aquifer)
            self._J_b_aquifer = []
            for src in self.surjectedvey.source_list:
                for rx in src.receiver_list:
                    Prx = rx.getP(self.time_mesh)
                    self._J_b_aquifer.append(Prx * tmp)
            self._J_b_aquifer = np.hstack(self._J_b_aquifer).reshape([-1,1])
            return self._J_b_aquifer        
   
    def getA(self, c):
        """
            Get a system matrix A
        """
        # c = Kv/Ssk (Diffusivity)
        C = utils.sdiag(c)
        A = (C@(self.L)-1/self.dt*self.I)
        return A

    def getB(self):
        """
            Get a system matrix A
        """
        return self.I / self.dt
    
    def get_drhsdh_aquifer(self, c, i_time):
        if i_time == 0:
            print (i_time, 'kang')
            return -1/self.dt * np.ones((self.nz, 1))
        else:
            C = utils.sdiag(c)
            # extract only the bottom boundary (dirichlet)
            drhsdh_aquifer = (-C @ self.DivB).toarray()[:,0].reshape([-1,1])
            return drhsdh_aquifer


    def get_J_h_aquifer(self, m, f=None):
        """
            G_rhs^T @ A^T @ P^T
        """
        if self._J_h_aquifer is not None:
            return self._J_h_aquifer
        else:

            if f is None:
                f = self.fields(m)
            
            Pf = f['Pf']
            Pa = f['Pa']
            PT = []
            PaT= []
            for src in self.survey.source_list:
                for rx in src.receiver_list:
                    Prx = rx.getP(self.time_mesh)
                    PT.append( (Prx @ Pf).toarray().T )
                    PaT.append( (Prx @ Pa).toarray().T)

            PT = np.hstack(PT)
            PaT = np.hstack(PaT) * self.b_aquifer * (self.Sska)
            # Start backward propagation:
            Y = np.zeros(PT.shape, dtype=float)
            # assume the size of h_aquifer is n_time
            self._J_h_aquifer = np.zeros((self.survey.nD, self.n_time), dtype=float)
            SOL = np.zeros((self.nz, self.survey.nD))
            for k in range(self.n_time-1,0,-1):
                Pk = PT[k*self.nz:(k+1)*self.nz, :]
                if k == self.n_time-1:
                    RHS = Pk
                else:
                    Bk = self.getB()
                    RHS = Pk - Bk @ SOL
                c = f['c_matrix'][:, k]
                AT = self.getA(c)
                ATinv = self.Solver(AT)
                SOL = ATinv * RHS
                drhsdh_aquiferTk = self.get_drhsdh_aquifer(c, k).T
                self._J_h_aquifer[:,k] = drhsdh_aquiferTk @ (SOL)
            # Handle the initial condition
            Pk = PT[:self.nz, :]
            self._J_h_aquifer[:,0] = self.get_drhsdh_aquifer([], 0).T @ SOL + Pk.sum(axis=0)
            self._J_h_aquifer += PaT.T

            return self._J_h_aquifer

    def Jvec(self, m, v, f=None):
        # Output is nD x 1 vector
        Jvec = np.zeros(self.survey.nD, dtype=float)
        if self.SskvMap is not None:
            Jvec += self.get_J_Sskv(m, f=f) @ (self.SskvDeriv @ v)
        if self.SskeMap is not None:
            Jvec += self.get_J_Sske(m, f=f) @ (self.SskeDeriv @ v)
        if self.KvMap is not None:
            Jvec += self.get_J_Kv(m, f=f) @ (self.KvDeriv @ v)
        if self.h_min0Map is not None:
            Jvec += self.get_J_h_min0(m, f=f) @ (self.h_min0Deriv @ v)
        if self.n_equivMap is not None:            
            Jvec += self.get_J_n_equiv(m, f=f) @ (self.n_equivDeriv @ v)
        if self.b_equivMap is not None:            
            Jvec += self.get_J_b_equiv(m, f=f) @ (self.b_equivDeriv @ v)
        if self.SskaMap is not None:            
            Jvec += self.get_J_Sska(m, f=f) @ (self.SskaDeriv @ v)
        if self.b_aquiferMap is not None:            
            Jvec += self.get_J_b_aquifer(m, f=f) @ (self.b_aquiferDeriv @ v)
        if self.h_aquiferMap is not None:                        
            Jvec += self.get_J_h_aquifer(m, f=f) @ (self.h_aquiferDeriv @ v)
        return Jvec
    
    def Jtvec(self, m, v, f=None):
        # Output is M x 1 vector
        Jtvec = np.zeros(m.size, dtype=float)
        if self.SskvMap is not None:
            Jtvec += self.SskvDeriv.T * (self.get_J_Sskv(m, f=f).T @ v)
        if self.SskeMap is not None:
            Jtvec += self.SskeDeriv.T * (self.get_J_Sske(m, f=f).T @ v)
        if self.KvMap is not None:
            Jtvec += self.KvDeriv.T * (self.get_J_Kv(m, f=f).T @ v)
        if self.h_min0Map is not None:
            Jtvec += self.h_min0Deriv.T * (self.get_J_h_min0(m, f=f).T @ v)
        if self.n_equivMap is not None:            
            Jtvec += self.n_equivDeriv.T * (self.get_J_n_equiv(m, f=f).T @ v)
        if self.b_equivMap is not None:            
            Jtvec += self.b_equivDeriv.T * (self.get_J_b_equiv(m, f=f).T @ v)
        if self.SskaMap is not None:            
            Jtvec += self.SskaDeriv.T * (self.get_J_Sska(m, f=f).T @ v)
        if self.b_aquiferMap is not None:            
            Jtvec += self.b_aquiferDeriv.T * (self.get_J_b_aquifer(m, f=f).T @ v)     
        if self.h_aquiferMap is not None:                        
            Jtvec += self.h_aquiferDeriv.T * (self.get_J_h_aquifer(m, f=f).T @ v)     
               
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