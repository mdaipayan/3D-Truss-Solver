import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import warnings
class Node:
    def __init__(self, id, x, y, z, rx=0, ry=0, rz=0):
        self.id = id
        self.user_id = id
        self.x = x
        self.y = y
        self.z = z
        
        # Support Conditions (True if restrained)
        self.rx = bool(rx)
        self.ry = bool(ry)
        self.rz = bool(rz)
        
        # Reaction Forces
        self.rx_val = 0.0
        self.ry_val = 0.0
        self.rz_val = 0.0
        
        # 3 DOFs per node in a Space Truss: [X, Y, Z]
        self.dofs = [3 * id - 3, 3 * id - 2, 3 * id - 1]

class Member:
    def __init__(self, id, node_i, node_j, E, A):
        self.id = id
        self.node_i = node_i
        self.node_j = node_j
        self.E = E
        self.A = A
        self.internal_force = 0.0
        
        # 1. 3D Kinematics (Length)
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z
        self.L = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if self.L == 0:
            raise ValueError(f"Member {self.id} has zero length.")
            
        # 2. Direction Cosines (l, m, n)
        self.l = dx / self.L
        self.m = dy / self.L
        self.n = dz / self.L
        
        # 3. Transformation Vector (T)
        self.T_vector = np.array([-self.l, -self.m, -self.n, self.l, self.m, self.n])
        
        # 4. Element Stiffness Matrix in Global Coordinates (6x6)
        # k = (EA/L) * [T^T * T]
        self.k_global_matrix = (self.E * self.A / self.L) * np.outer(self.T_vector, self.T_vector)
        
        # Map element DOFs to system DOFs
        self.dofs = self.node_i.dofs + self.node_j.dofs
        self.u_local = None

    def calculate_force(self):
        """Calculates axial force. Positive = Tension, Negative = Compression."""
        if self.u_local is not None:
            # F = (EA/L) * dot(T, u_local)
            self.internal_force = (self.E * self.A / self.L) * np.dot(self.T_vector, self.u_local)
        return self.internal_force

class TrussSystem:
    def __init__(self):
        self.nodes = []
        self.members = []
        self.loads = {}  # Dictionary of {dof_index: force_value}
        
        # State variables to support the pedagogical "Glass-Box" UI
        self.K_global = None
        self.F_global = None
        self.free_dofs = []
        self.K_reduced = None
        self.F_reduced = None
        self.U_global = None
        
    def solve(self):
        num_dofs = 3 * len(self.nodes)
        self.F_global = np.zeros(num_dofs)
        
        # 1. Assemble Global Stiffness Matrix using COO (Coordinate) Format
        # This is the fastest way to build sparse matrices dynamically.
        row_indices = []
        col_indices = []
        data_values = []
        
        for member in self.members:
            for i in range(6):
                for j in range(6):
                    row_indices.append(member.dofs[i])
                    col_indices.append(member.dofs[j])
                    data_values.append(member.k_global_matrix[i, j])
                    
        # Create sparse matrix and convert to Compressed Sparse Column (CSC) format for fast math
        self.K_global = coo_matrix((data_values, (row_indices, col_indices)), shape=(num_dofs, num_dofs)).tocsc()
        
        # 2. Assemble Load Vector
        for dof, force in self.loads.items():
            self.F_global[dof] += force
            
        # 3. Apply Boundary Conditions (Matrix Partitioning)
        restrained_dofs = []
        for node in self.nodes:
            if node.rx: restrained_dofs.append(node.dofs[0])
            if node.ry: restrained_dofs.append(node.dofs[1])
            if node.rz: restrained_dofs.append(node.dofs[2])
            
        self.free_dofs = [i for i in range(num_dofs) if i not in restrained_dofs]
        
        # Isolate the Free-Free matrix components
        # (Slicing by index lists is highly optimized in SciPy's CSC format)
        self.K_reduced = self.K_global[self.free_dofs, :][:, self.free_dofs]
        self.F_reduced = self.F_global[self.free_dofs]
        
        # 4. Mathematical Bulletproofing & Solving
        if self.K_reduced.shape[0] > 0:
            
            # Instead of calculating the extremely slow Condition Number, 
            # we catch SciPy's low-level matrix warnings to detect mechanisms instantly.
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                
                # Fast Sparse Solver (U_f = K_ff^-1 * F_f)
                U_reduced = spsolve(self.K_reduced, self.F_reduced)
                
                # If spsolve throws a warning or returns NaNs/massive numbers, it is a mechanism
                if len(w) > 0 or np.any(np.isnan(U_reduced)) or np.max(np.abs(U_reduced)) > 1e6:
                    raise ValueError("Structure is unstable (mechanism detected). Check boundary conditions and member connectivity.")
        else:
            U_reduced = np.array([])
            
        # Reconstruct full global displacement vector
        self.U_global = np.zeros(num_dofs)
        for idx, dof in enumerate(self.free_dofs):
            self.U_global[dof] = U_reduced[idx]
            
        # 5. Calculate Support Reactions (R = K * U - F)
        # Sparse matrix-vector multiplication is handled perfectly by the .dot() operator
        R_global = self.K_global.dot(self.U_global) - self.F_global
        for node in self.nodes:
            node.rx_val = R_global[node.dofs[0]] if node.rx else 0.0
            node.ry_val = R_global[node.dofs[1]] if node.ry else 0.0
            node.rz_val = R_global[node.dofs[2]] if node.rz else 0.0
            
        # 6. Extract Member Forces & Local Kinematics
        for member in self.members:
            member.u_local = np.array([self.U_global[dof] for dof in member.dofs])
            member.calculate_force()
