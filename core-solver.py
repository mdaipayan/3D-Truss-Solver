import numpy as np

def solve_3d_truss(nodes, members, supports, loads):
    """
    Core engine for 3D Space Truss Analysis using Direct Stiffness Method.
    
    nodes: dict of {node_id: (x, y, z)}
    members: list of dicts [{'id': 1, 'n1': 1, 'n2': 2, 'E': 200e9, 'A': 0.005}, ...]
    supports: dict of {node_id: (restrain_X, restrain_Y, restrain_Z)} # True if restrained
    loads: dict of {node_id: (Fx, Fy, Fz)}
    """
    num_nodes = len(nodes)
    num_dof = 3 * num_nodes
    
    # 1. Initialize Global Stiffness Matrix (K) and Force Vector (F)
    K_global = np.zeros((num_dof, num_dof))
    F_global = np.zeros(num_dof)
    
    # 2. Assemble Global Stiffness Matrix
    for member in members:
        n1, n2 = member['n1'], member['n2']
        x1, y1, z1 = nodes[n1]
        x2, y2, z2 = nodes[n2]
        E, A = member['E'], member['A']
        
        # Calculate 3D Length
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        
        # Calculate Direction Cosines
        l = (x2 - x1) / L
        m = (y2 - y1) / L
        n = (z2 - z1) / L
        
        # 6x6 Element Stiffness Matrix
        k_el = (E * A / L) * np.array([
            [ l**2,  l*m,  l*n, -l**2, -l*m, -l*n],
            [ l*m,  m**2,  m*n, -l*m, -m**2, -m*n],
            [ l*n,  m*n,  n**2, -l*n, -m*n, -n**2],
            [-l**2, -l*m, -l*n,  l**2,  l*m,  l*n],
            [-l*m, -m**2, -m*n,  l*m,  m**2,  m*n],
            [-l*n, -m*n, -n**2,  l*n,  m*n,  n**2]
        ])
        
        # Map local DOFs to global DOFs
        dof_indices = [
            3*(n1-1), 3*(n1-1)+1, 3*(n1-1)+2,  # Node 1 X, Y, Z
            3*(n2-1), 3*(n2-1)+1, 3*(n2-1)+2   # Node 2 X, Y, Z
        ]
        
        # Add to Global Matrix
        for i in range(6):
            for j in range(6):
                K_global[dof_indices[i], dof_indices[j]] += k_el[i, j]
                
    # 3. Apply Loads
    for node_id, (fx, fy, fz) in loads.items():
        F_global[3*(node_id-1)] = fx
        F_global[3*(node_id-1)+1] = fy
        F_global[3*(node_id-1)+2] = fz
        
    # 4. Apply Boundary Conditions (Penalty Method or Matrix Reduction)
    # Using matrix reduction (striking out rows/columns for restrained DOFs)
    free_dofs = []
    for node_id in range(1, num_nodes + 1):
        if node_id in supports:
            rx, ry, rz = supports[node_id]
            if not rx: free_dofs.append(3*(node_id-1))
            if not ry: free_dofs.append(3*(node_id-1)+1)
            if not rz: free_dofs.append(3*(node_id-1)+2)
        else:
            free_dofs.extend([3*(node_id-1), 3*(node_id-1)+1, 3*(node_id-1)+2])
            
    # Reduce K and F matrices
    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    F_reduced = F_global[free_dofs]
    
    # 5. Solve for Displacements
    U_reduced = np.linalg.solve(K_reduced, F_reduced)
    
    # Reconstruct full displacement vector
    U_global = np.zeros(num_dof)
    for i, dof in enumerate(free_dofs):
        U_global[dof] = U_reduced[i]
        
    # 6. Calculate Member Forces and Reactions
    member_forces = []
    for member in members:
        n1, n2 = member['n1'], member['n2']
        x1, y1, z1 = nodes[n1]
        x2, y2, z2 = nodes[n2]
        E, A = member['E'], member['A']
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        l, m, n = (x2 - x1)/L, (y2 - y1)/L, (z2 - z1)/L
        
        # Transformation vector
        T = np.array([-l, -m, -n, l, m, n])
        
        # Element displacements
        u_el = np.array([
            U_global[3*(n1-1)], U_global[3*(n1-1)+1], U_global[3*(n1-1)+2],
            U_global[3*(n2-1)], U_global[3*(n2-1)+1], U_global[3*(n2-1)+2]
        ])
        
        # Axial Force (Positive = Tension, Negative = Compression)
        force = (E * A / L) * np.dot(T, u_el)
        member_forces.append({'member_id': member['id'], 'force': force})
        
    # Calculate Reactions: R = K * U - F
    Reactions = np.dot(K_global, U_global) - F_global
    
    return U_global, member_forces, Reactions
