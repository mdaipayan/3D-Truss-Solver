import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core_solver import TrussSystem, Node, Member
import datetime
import os
from visualizer import draw_undeformed_geometry, draw_results_fbd
from optimiser import TrussOptimizerGA

st.set_page_config(page_title="Professional Truss Suite (3D)", layout="wide")
st.title("🏗️ Professional Space Truss Analysis Developed by D Mandal")

st.sidebar.header("⚙️ Display Settings")
st.sidebar.info("The solver engine calculates using base SI units (Newtons, meters). Use this setting to scale the visual output on the diagrams.")

force_display = st.sidebar.selectbox(
    "Force Display Unit", 
    options=["Newtons (N)", "Kilonewtons (kN)", "Meganewtons (MN)"], 
    index=1
)

unit_map = {
    "Newtons (N)": (1.0, "N"), 
    "Kilonewtons (kN)": (1000.0, "kN"), 
    "Meganewtons (MN)": (1000000.0, "MN")
}
current_scale, current_unit = unit_map[force_display]

fig = go.Figure()

def clear_results():
    if 'solved_truss' in st.session_state:
        del st.session_state['solved_truss']
    if 'report_data' in st.session_state:
        del st.session_state['report_data']

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Input Data")
    
    st.info("💡 **Benchmark Model:** Load a 22-Node, 69-Member 3D Transmission Tower to validate spatial kinematics and asymmetric wind/cable loading.")
    if st.button("🗼 Load 3D High-Voltage Tower Benchmark"):
        
        # 22-Node Tapered Transmission Tower
        # Base at Z=0, First tier at Z=10, Cross-arms at Z=20 and Z=26, Peaks at Z=32
        st.session_state['nodes_data'] = pd.DataFrame([
            # ---- Level 0: Base (Fully Restrained) ----
            [4.0, 4.0, 0.0, 1, 1, 1],   # Node 1
            [4.0, -4.0, 0.0, 1, 1, 1],  # Node 2
            [-4.0, -4.0, 0.0, 1, 1, 1], # Node 3
            [-4.0, 4.0, 0.0, 1, 1, 1],  # Node 4
            
            # ---- Level 1: Lower Tapered Body ----
            [3.0, 3.0, 10.0, 0, 0, 0],   # Node 5
            [3.0, -3.0, 10.0, 0, 0, 0],  # Node 6
            [-3.0, -3.0, 10.0, 0, 0, 0], # Node 7
            [-3.0, 3.0, 10.0, 0, 0, 0],  # Node 8
            
            # ---- Level 2: Lower Cross-Arm Body ----
            [2.0, 2.0, 20.0, 0, 0, 0],   # Node 9
            [2.0, -2.0, 20.0, 0, 0, 0],  # Node 10
            [-2.0, -2.0, 20.0, 0, 0, 0], # Node 11
            [-2.0, 2.0, 20.0, 0, 0, 0],  # Node 12
            # Lower Cross-Arm Tips (Extending along Y-axis)
            [0.0, 7.0, 20.0, 0, 0, 0],   # Node 13 (+Y Arm)
            [0.0, -7.0, 20.0, 0, 0, 0],  # Node 14 (-Y Arm)
            
            # ---- Level 3: Upper Cross-Arm Body ----
            [1.5, 1.5, 26.0, 0, 0, 0],   # Node 15
            [1.5, -1.5, 26.0, 0, 0, 0],  # Node 16
            [-1.5, -1.5, 26.0, 0, 0, 0], # Node 17
            [-1.5, 1.5, 26.0, 0, 0, 0],  # Node 18
            # Upper Cross-Arm Tips
            [0.0, 6.0, 26.0, 0, 0, 0],   # Node 19 (+Y Arm)
            [0.0, -6.0, 26.0, 0, 0, 0],  # Node 20 (-Y Arm)
            
            # ---- Level 4: Shield Wire Peaks ----
            [0.0, 1.5, 32.0, 0, 0, 0],   # Node 21
            [0.0, -1.5, 32.0, 0, 0, 0]   # Node 22
        ], columns=["X", "Y", "Z", "Restrain_X", "Restrain_Y", "Restrain_Z"])
        
        # 69 Members (Standard Steel Sections)
        E_steel = 2e11  # 200 GPa
        A_leg = 0.006   # Main legs
        A_brace = 0.003 # Bracing and arms
        
        st.session_state['members_data'] = pd.DataFrame([
            # 1. MAIN VERTICAL LEGS
            [1,5,A_leg,E_steel], [2,6,A_leg,E_steel], [3,7,A_leg,E_steel], [4,8,A_leg,E_steel], # L0-L1
            [5,9,A_leg,E_steel], [6,10,A_leg,E_steel], [7,11,A_leg,E_steel], [8,12,A_leg,E_steel], # L1-L2
            [9,15,A_leg,E_steel], [10,16,A_leg,E_steel], [11,17,A_leg,E_steel], [12,18,A_leg,E_steel], # L2-L3
            
            # 2. HORIZONTAL RINGS
            [5,6,A_brace,E_steel], [6,7,A_brace,E_steel], [7,8,A_brace,E_steel], [8,5,A_brace,E_steel], # L1
            [9,10,A_brace,E_steel], [10,11,A_brace,E_steel], [11,12,A_brace,E_steel], [12,9,A_brace,E_steel], # L2
            [15,16,A_brace,E_steel], [16,17,A_brace,E_steel], [17,18,A_brace,E_steel], [18,15,A_brace,E_steel], # L3
            
            # 3. X-BRACING (FACES)
            # Base to L1
            [1,6,A_brace,E_steel], [2,5,A_brace,E_steel], [2,7,A_brace,E_steel], [3,6,A_brace,E_steel],
            [3,8,A_brace,E_steel], [4,7,A_brace,E_steel], [4,5,A_brace,E_steel], [1,8,A_brace,E_steel],
            # L1 to L2
            [5,10,A_brace,E_steel], [6,9,A_brace,E_steel], [6,11,A_brace,E_steel], [7,10,A_brace,E_steel],
            [7,12,A_brace,E_steel], [8,11,A_brace,E_steel], [8,9,A_brace,E_steel], [5,12,A_brace,E_steel],
            # L2 to L3
            [9,16,A_brace,E_steel], [10,15,A_brace,E_steel], [10,17,A_brace,E_steel], [11,16,A_brace,E_steel],
            [11,18,A_brace,E_steel], [12,17,A_brace,E_steel], [12,15,A_brace,E_steel], [9,18,A_brace,E_steel],
            
            # 4. LOWER CROSS-ARMS (Tips N13, N14)
            [13,9,A_brace,E_steel], [13,12,A_brace,E_steel], # +Y Arm horizontal ties
            [13,15,A_brace,E_steel], [13,18,A_brace,E_steel], # +Y Arm upward suspension struts
            [14,10,A_brace,E_steel], [14,11,A_brace,E_steel], # -Y Arm horizontal ties
            [14,16,A_brace,E_steel], [14,17,A_brace,E_steel], # -Y Arm upward suspension struts
            
            # 5. UPPER CROSS-ARMS (Tips N19, N20)
            [19,15,A_brace,E_steel], [19,18,A_brace,E_steel], # +Y Arm horizontal ties
            [19,21,A_brace,E_steel], # +Y Arm upward strut to peak
            [20,16,A_brace,E_steel], [20,17,A_brace,E_steel], # -Y Arm horizontal ties
            [20,22,A_brace,E_steel], # -Y Arm upward strut to peak
            
            # 6. TOP PEAKS
            [15,21,A_leg,E_steel], [18,21,A_leg,E_steel], # +Y Peak Legs
            [16,22,A_leg,E_steel], [17,22,A_leg,E_steel], # -Y Peak Legs
            [21,22,A_brace,E_steel], # Tie between peaks

           # 7. HORIZONTAL PLAN BRACING (Crucial for 3D torsional stability)
            [5,7,A_brace,E_steel], [6,8,A_brace,E_steel],     # L1 Diagonals
            [9,11,A_brace,E_steel], [10,12,A_brace,E_steel],   # L2 Diagonals
            [15,17,A_brace,E_steel], [16,18,A_brace,E_steel],  # 
            
            # 8. TOP PEAK X-BRACING (Prevents Windshield-Wiper Mechanism)
            [15,22,A_brace,E_steel], [16,21,A_brace,E_steel]
            
        ], columns=["Node_I", "Node_J", "Area(sq.m)", "E (N/sq.m)"])
        
        # Simulated Combined Loading 
        # Fx = Wind Load on one face
        # Fz = Heavy downward pull from conductor cables
        st.session_state['loads_data'] = pd.DataFrame([
            [13, 10000.0, 0.0, -80000.0],  # Lower Arm 1 (Wind + Heavy Cable)
            [14, 10000.0, 0.0, -80000.0],  # Lower Arm 2 (Wind + Heavy Cable)
            [19, 8000.0, 0.0, -60000.0],   # Upper Arm 1
            [20, 8000.0, 0.0, -60000.0],   # Upper Arm 2
            [21, 5000.0, 0.0, -20000.0],   # Peak 1 (Shield Wire)
            [22, 5000.0, 0.0, -20000.0]    # Peak 2 (Shield Wire)
        ], columns=["Node_ID", "Force_X (N)", "Force_Y (N)", "Force_Z (N)"])
        
        clear_results()
    st.info("💡 **Academic Benchmark:** Load the Classic 25-Bar Space Truss to validate optimization algorithms against published literature.")
    if st.button("🧊 Load 25-Bar Space Truss Benchmark"):
        
        # Unit Conversions (Literature is in Imperial, Solver is in SI)
        in2m = 0.0254
        lb2N = 4.448
        
        # 10 Nodes
        st.session_state['nodes_data'] = pd.DataFrame([
            # ---- Level 2: Top Ridge ----
            [-37.5*in2m, 0.0, 200.0*in2m, 0, 0, 0],        # Node 1
            [37.5*in2m, 0.0, 200.0*in2m, 0, 0, 0],         # Node 2
            # ---- Level 1: Mid Body ----
            [-37.5*in2m, 37.5*in2m, 100.0*in2m, 0, 0, 0],  # Node 3
            [37.5*in2m, 37.5*in2m, 100.0*in2m, 0, 0, 0],   # Node 4
            [37.5*in2m, -37.5*in2m, 100.0*in2m, 0, 0, 0],  # Node 5
            [-37.5*in2m, -37.5*in2m, 100.0*in2m, 0, 0, 0], # Node 6
            # ---- Level 0: Base (Fully Restrained) ----
            [-100.0*in2m, 100.0*in2m, 0.0, 1, 1, 1],       # Node 7
            [100.0*in2m, 100.0*in2m, 0.0, 1, 1, 1],        # Node 8
            [100.0*in2m, -100.0*in2m, 0.0, 1, 1, 1],       # Node 9
            [-100.0*in2m, -100.0*in2m, 0.0, 1, 1, 1]       # Node 10
        ], columns=["X", "Y", "Z", "Restrain_X", "Restrain_Y", "Restrain_Z"])

        # E = 10,000 ksi (Aluminum) = 68.95 GPa
        E_alu = 6.895e10 
        A_init = 0.001 # 1000 mm^2 (Placeholder before optimization)
        
        # 25 Members (Grouped symmetrically to reduce 25 variables down to 8)
        st.session_state['members_data'] = pd.DataFrame([
            [1,2, A_init, E_alu],                                                                      # Group 1 (A1)
            [1,4, A_init, E_alu], [2,3, A_init, E_alu], [1,5, A_init, E_alu], [2,6, A_init, E_alu],    # Group 2 (A2)
            [2,5, A_init, E_alu], [2,4, A_init, E_alu], [1,3, A_init, E_alu], [1,6, A_init, E_alu],    # Group 3 (A3)
            [3,6, A_init, E_alu], [4,5, A_init, E_alu],                                                # Group 4 (A4)
            [3,4, A_init, E_alu], [5,6, A_init, E_alu],                                                # Group 5 (A5)
            [3,10, A_init, E_alu], [6,7, A_init, E_alu], [4,9, A_init, E_alu], [5,8, A_init, E_alu],   # Group 6 (A6)
            [3,8, A_init, E_alu], [4,7, A_init, E_alu], [6,9, A_init, E_alu], [5,10, A_init, E_alu],   # Group 7 (A7)
            [3,7, A_init, E_alu], [4,8, A_init, E_alu], [5,9, A_init, E_alu], [6,10, A_init, E_alu]    # Group 8 (A8)
        ], columns=["Node_I", "Node_J", "Area(sq.m)", "E (N/sq.m)"])

        # Benchmark Multi-Directional Loading
        st.session_state['loads_data'] = pd.DataFrame([
            [1, 1000.0*lb2N, -10000.0*lb2N, -10000.0*lb2N],
            [2, 0.0,         -10000.0*lb2N, -10000.0*lb2N],
            [3, 500.0*lb2N,  0.0,           0.0],
            [6, 600.0*lb2N,  0.0,           0.0]
        ], columns=["Node_ID", "Force_X (N)", "Force_Y (N)", "Force_Z (N)"])
        
        clear_results()
    st.info("💡 **Multi-Load Benchmark:** Load the 72-Bar Space Truss to test optimization algorithms against multiple simultaneous load conditions.")
    if st.button("🏢 Load 72-Bar Space Truss Benchmark"):
        
        in2m = 0.0254
        kip2N = 4448.22
        E_alu = 6.895e10  # 10,000 ksi
        A_init = 0.001
        
        # 1. Generate 20 Nodes (4 stories + base)
        nodes = []
        for z in [0, 60, 120, 180, 240]:
            z_m = z * in2m
            w = 60 * in2m # 120-inch wide base (from -60 to +60)
            restrain = 1 if z == 0 else 0
            
            nodes.extend([
                [-w, w, z_m, restrain, restrain, restrain],  # Node 1
                [w, w, z_m, restrain, restrain, restrain],   # Node 2
                [w, -w, z_m, restrain, restrain, restrain],  # Node 3
                [-w, -w, z_m, restrain, restrain, restrain]  # Node 4
            ])
        st.session_state['nodes_data'] = pd.DataFrame(nodes, columns=["X", "Y", "Z", "Restrain_X", "Restrain_Y", "Restrain_Z"])
        
        # 2. Generate 72 Members Algorithmically
        members = []
        for i in range(4): # Loop through the 4 stories
            b = i * 4 + 1  # Base nodes for this story (1-based index)
            t = b + 4      # Top nodes for this story (1-based index)
            
            # Verticals (Groups 1, 5, 9, 13)
            members.extend([[b, t, A_init, E_alu], [b+1, t+1, A_init, E_alu], [b+2, t+2, A_init, E_alu], [b+3, t+3, A_init, E_alu]])
            
            # Horizontals (Groups 2, 6, 10, 14)
            members.extend([[t, t+1, A_init, E_alu], [t+1, t+2, A_init, E_alu], [t+2, t+3, A_init, E_alu], [t+3, t, A_init, E_alu]])
            
            # Face X-Bracing (Groups 3, 7, 11, 15)
            members.extend([[b, t+1, A_init, E_alu], [b+1, t, A_init, E_alu], 
                            [b+1, t+2, A_init, E_alu], [b+2, t+1, A_init, E_alu],
                            [b+2, t+3, A_init, E_alu], [b+3, t+2, A_init, E_alu],
                            [b+3, t, A_init, E_alu], [b, t+3, A_init, E_alu]])
                            
            # Plan Diagonals / Top Face (Groups 4, 8, 12, 16)
            members.extend([[t, t+2, A_init, E_alu], [t+1, t+3, A_init, E_alu]])
            
        st.session_state['members_data'] = pd.DataFrame(members, columns=["Node_I", "Node_J", "Area(sq.m)", "E (N/sq.m)"])

        # 3. Load Case 1: Asymmetric Wind + Gravity (Often the controlling case)
        st.session_state['loads_data'] = pd.DataFrame([
            [17, 5.0*kip2N, 5.0*kip2N, -5.0*kip2N] # Top corner hit by wind
        ], columns=["Node_ID", "Force_X (N)", "Force_Y (N)", "Force_Z (N)"])

        # Define 16 symmetric groups for the 72-bar truss
        # Group 0: M1-M4, Group 1: M5-M8, etc.
        groups = {}
        for i in range(16):
            groups[i] = [i*4 + 1, i*4 + 2, i*4 + 3, i*4 + 4]
        st.session_state['member_groups'] = groups
        
        clear_results()    
    if 'nodes_data' not in st.session_state:
        st.session_state['nodes_data'] = pd.DataFrame(columns=["X", "Y", "Z", "Restrain_X", "Restrain_Y", "Restrain_Z"])
        st.session_state['members_data'] = pd.DataFrame(columns=["Node_I", "Node_J", "Area(sq.m)", "E (N/sq.m)"])
        st.session_state['loads_data'] = pd.DataFrame(columns=["Node_ID", "Force_X (N)", "Force_Y (N)", "Force_Z (N)"])

    st.subheader("Nodes")
    node_df = st.data_editor(st.session_state['nodes_data'], num_rows="dynamic", key="nodes", on_change=clear_results)

    st.subheader("Members")
    member_df = st.data_editor(st.session_state['members_data'], num_rows="dynamic", key="members", on_change=clear_results)

    st.subheader("Nodal Loads")
    load_df = st.data_editor(st.session_state['loads_data'], num_rows="dynamic", key="loads", on_change=clear_results)
    
    if st.button("Calculate Results"):
        try:
            ts = TrussSystem()
            node_map = {}
            valid_node_count = 0
            
            # 1. Parse Nodes
            for i, row in node_df.iterrows():
                if pd.isna(row.get('X')) or pd.isna(row.get('Y')) or pd.isna(row.get('Z')): continue
                valid_node_count += 1
                rx = int(row.get('Restrain_X', 0)) if not pd.isna(row.get('Restrain_X')) else 0
                ry = int(row.get('Restrain_Y', 0)) if not pd.isna(row.get('Restrain_Y')) else 0
                rz = int(row.get('Restrain_Z', 0)) if not pd.isna(row.get('Restrain_Z')) else 0
                
                n = Node(valid_node_count, float(row['X']), float(row['Y']), float(row['Z']), rx, ry, rz)
                n.user_id = i + 1 
                ts.nodes.append(n)
                node_map[i + 1] = n 
                
            # 2. Parse Members
            for i, row in member_df.iterrows():
                if pd.isna(row.get('Node_I')) or pd.isna(row.get('Node_J')): continue
                ni_val, nj_val = int(row['Node_I']), int(row['Node_J'])
                
                if ni_val not in node_map or nj_val not in node_map:
                    raise ValueError(f"Member M{i+1} references an invalid Node ID.")
                    
                E = float(row.get('E (N/sq.m)', 2e11)) if not pd.isna(row.get('E (N/sq.m)')) else 2e11
                A = float(row.get('Area(sq.m)', 0.01)) if not pd.isna(row.get('Area(sq.m)')) else 0.01
                ts.members.append(Member(i+1, node_map[ni_val], node_map[nj_val], E, A))
                
            # 3. Parse Loads
            for i, row in load_df.iterrows():
                if pd.isna(row.get('Node_ID')): continue
                node_id_val = int(row['Node_ID'])
                
                if node_id_val not in node_map:
                    raise ValueError(f"Load at row {i+1} references an invalid Node ID.")
                    
                target_node = node_map[node_id_val]
                fx = float(row.get('Force_X (N)', 0)) if not pd.isna(row.get('Force_X (N)')) else 0.0
                fy = float(row.get('Force_Y (N)', 0)) if not pd.isna(row.get('Force_Y (N)')) else 0.0
                fz = float(row.get('Force_Z (N)', 0)) if not pd.isna(row.get('Force_Z (N)')) else 0.0
                
                dof_x, dof_y, dof_z = target_node.dofs[0], target_node.dofs[1], target_node.dofs[2]
                
                ts.loads[dof_x] = ts.loads.get(dof_x, 0.0) + fx
                ts.loads[dof_y] = ts.loads.get(dof_y, 0.0) + fy
                ts.loads[dof_z] = ts.loads.get(dof_z, 0.0) + fz
            
            if not ts.nodes or not ts.members:
                raise ValueError("Incomplete model: Please define at least two valid nodes and one member.")
                
            ts.solve()
            st.session_state['solved_truss'] = ts
            st.success("Analysis Complete!")
        except Exception as e:
            st.error(f"Error: {e}")

with col2:
    st.header("2. 3D Model Visualization")
    tab1, tab2 = st.tabs(["🏗️ Undeformed Geometry", "📊 Structural Forces (Results)"])

    with tab1:
        if node_df.empty:
            st.info("👈 Start adding nodes in the Input Table (or click 'Load Benchmark Data') to build your geometry.")
        else:
            fig_base, node_errors, member_errors, load_errors = draw_undeformed_geometry(node_df, member_df, load_df, scale_factor=current_scale, unit_label=current_unit)
            
            if node_errors: st.warning(f"⚠️ Geometry Warning: Invalid data at Node row(s): {', '.join(node_errors)}.")
            if member_errors: st.warning(f"⚠️ Connectivity Warning: Cannot draw M{', M'.join(member_errors)}.")
            
            st.session_state['base_fig'] = fig_base 
            st.plotly_chart(fig_base, width="stretch")

    with tab2:
        if 'solved_truss' in st.session_state:
            ts = st.session_state['solved_truss']
            fig_res = draw_results_fbd(ts, scale_factor=current_scale, unit_label=current_unit)
            st.session_state['current_fig'] = fig_res 
            st.plotly_chart(fig_res, width="stretch")
        else:
            st.info("👈 Input loads and click 'Calculate Results' to view the force diagram.")

# ---------------------------------------------------------
# NEW SECTION: THE "GLASS BOX" PEDAGOGICAL EXPLORER (3D)
# ---------------------------------------------------------
if 'solved_truss' in st.session_state:
    st.markdown("---")
    st.header("🎓 Educational Glass-Box: 3D DSM Intermediate Steps")
    
    ts = st.session_state['solved_truss']
    gb_tab1, gb_tab2, gb_tab3 = st.tabs(["📐 1. 3D Kinematics & Stiffness", "🧩 2. Global Assembly", "🚀 3. Displacements & Internal Forces"])
    
    with gb_tab1:
        st.subheader("Local Element Formulation (3D)")
        if ts.members: 
            mbr_opts = [f"Member {m.id}" for m in ts.members]
            sel_mbr = st.selectbox("Select Member to inspect kinematics and stiffness:", mbr_opts, key="gb_tab1")
            selected_id = int(sel_mbr.split(" ")[1])
            m = next((m for m in ts.members if m.id == selected_id), None)
            
            colA, colB = st.columns([1, 2])
            with colA:
                st.markdown("**Member Kinematics**")
                st.write(f"- **Length ($L$):** `{m.L:.4f} m`")
                st.write(f"- **Dir. Cosine X ($l$):** `{m.l:.4f}`")
                st.write(f"- **Dir. Cosine Y ($m$):** `{m.m:.4f}`")
                st.write(f"- **Dir. Cosine Z ($n$):** `{m.n:.4f}`")
                
                st.markdown("**Transformation Vector ($T$):**")
                st.dataframe(pd.DataFrame([m.T_vector], columns=["-l", "-m", "-n", "l", "m", "n"]).style.format("{:.4f}"))
            
            with colB:
                st.markdown("**6x6 Global Element Stiffness Matrix ($k_{global}$)**")
                df_k = pd.DataFrame(m.k_global_matrix)
                st.dataframe(df_k.style.format("{:.2e}"))

    with gb_tab2:
        st.subheader("System Partitioning & Assembly")
        colC, colD = st.columns(2)
        with colC:
            st.markdown("**Degree of Freedom (DOF) Mapping**")
            st.write(f"- **Free DOFs ($f$):** `{ts.free_dofs}`")
            st.write(f"- **Active Load Vector ($F_f$)**")
            st.dataframe(pd.DataFrame(ts.F_reduced, columns=["Force"]).style.format("{:.2e}"))

        with colD:
            with st.expander("View Full Unpartitioned Global Matrix ($K_{global}$)", expanded=True):
                # Convert sparse matrix back to dense temporarily for UI rendering
                st.dataframe(pd.DataFrame(ts.K_global.toarray()).style.format("{:.2e}"))
            with st.expander("View Reduced Stiffness Matrix ($K_{ff}$)", expanded=False):
                st.dataframe(pd.DataFrame(ts.K_reduced.toarray()).style.format("{:.2e}"))

    with gb_tab3:
        st.subheader("Solving the System & Extracting Forces")
        colE, colF = st.columns(2)
        with colE:
            st.markdown("**1. Global Displacement Vector ($U_{global}$)**")
            if hasattr(ts, 'U_global') and ts.U_global is not None:
                st.dataframe(pd.DataFrame(ts.U_global, columns=["Displacement (m)"]).style.format("{:.6e}"))
                
        with colF:
            st.markdown("**2. Internal Force Extraction**")
            if ts.members:
                sel_mbr_force = st.selectbox("Select Member to view Force Extraction:", mbr_opts, key="gb_tab3")
                selected_id = int(sel_mbr_force.split(" ")[1])
                m = next((m for m in ts.members if m.id == selected_id), None)
                
                if m and hasattr(m, 'u_local') and m.u_local is not None:
                    st.latex(r"F_{axial} = \frac{EA}{L} \cdot (T \cdot u_{local})")
                    st.markdown("**Local Displacements ($u_{local}$):**")
                    st.dataframe(pd.DataFrame([m.u_local], columns=["u_ix", "u_iy", "u_iz", "u_jx", "u_jy", "u_jz"]).style.format("{:.6e}"))
                    st.success(f"**Calculated Axial Force:** {m.internal_force:.2f} N")
# ---------------------------------------------------------
# NEW SECTION: STRUCTURAL OPTIMIZATION (GENETIC ALGORITHM)
# ---------------------------------------------------------
st.markdown("---")
st.header("🧬 AI-Driven Structural Optimization")
st.info("Optimize any arbitrary truss geometry. If no symmetry groups are defined, the algorithm will automatically optimize each member independently.")

# Advanced UI: Generalizing for any material and geometry
col_ga1, col_ga2, col_ga3, col_ga4 = st.columns(4)
with col_ga1:
    pop_size = st.number_input("Population Size", min_value=10, max_value=500, value=50, step=10)
    generations = st.number_input("Generations", min_value=10, max_value=1000, value=100, step=10)
with col_ga2:
    target_stress = st.number_input("Yield Stress Limit (MPa)", value=250.0) # Default to Structural Steel
    material_density = st.number_input("Material Density (kg/m³)", value=7850.0) # Default to Steel
with col_ga3:
    min_a = st.number_input("Min Area (cm²)", value=1.0) 
with col_ga4:
    max_a = st.number_input("Max Area (cm²)", value=50.0) 

if st.button("🚀 Run Genetic Algorithm Optimization"):
    if 'solved_truss' not in st.session_state or not st.session_state['solved_truss'].members:
        st.warning("⚠️ Please load a model and calculate initial results before optimizing.")
    else:
        ts = st.session_state['solved_truss']
        
        # --- AUTO-GROUPING LOGIC ---
        if 'member_groups' in st.session_state:
            groups = st.session_state['member_groups']
            st.toast(f"⚙️ Using {len(groups)} predefined symmetry groups.", icon="✅")
        else:
            # If no groups exist, map every member to its own independent variable
            groups = {i: [m.id] for i, m in enumerate(ts.members)}
            st.toast(f"⚙️ No symmetry groups found. Optimizing all {len(ts.members)} members independently.", icon="⚠️")
        
        with st.spinner(f"Evolving optimal design over {generations} generations... This requires heavy matrix computation."):
            
            # Initialize and run the generalized optimizer
            optimizer = TrussOptimizerGA(
                base_truss=ts, 
                member_groups=groups,
                pop_size=pop_size,
                generations=generations,
                yield_stress=target_stress * 1e6,     # Convert MPa to Pa
                density=material_density,             # kg/m^3
                min_area=min_a / 10000.0,             # Convert cm^2 to m^2
                max_area=max_a / 10000.0              # Convert cm^2 to m^2
            )
            
            best_chromosome, best_fitness, convergence = optimizer.run()
            
        st.success(f"✅ Optimization Complete! Minimum Safe Structural Weight: **{best_fitness:.2f} kg**")
        
        # ---------------------------------------------------------
        # THE LAYMAN'S EDUCATIONAL WALKTHROUGH
        # ---------------------------------------------------------
        with st.expander("🎓 How did the AI find this design? (Step-by-Step Walkthrough)", expanded=True):
            st.markdown("""
            **The Genetic Algorithm mimics biological evolution.** Instead of trying every single combination of steel bars (which would take billions of years), the AI breeds the best designs over time. Here is what just happened in the background:
            """)
            
            step1, step2, step3 = st.columns(3)
            with step1:
                st.markdown("### 🧬 1. Initialization")
                st.write(f"The AI randomly generated a 'population' of **{pop_size} completely different truss designs**.")
                st.write("*Some were way too heavy (expensive), and some were too thin and collapsed under the load.*")
            
            with step2:
                st.markdown("### ⚖️ 2. Evaluation")
                st.write("The matrix solver tested every single design against the loads.")
                st.write(f"*It assigned a 'Fitness Score' based on weight, adding a massive penalty if the stress exceeded **{target_stress} MPa**.*")
            
            with step3:
                st.markdown("### 🏆 3. Selection")
                st.write("Survival of the fittest! The AI threw away the worst designs and kept the strongest, lightest ones to be 'parents'.")
                
            st.markdown("---")
            
            step4, step5, step6 = st.columns(3)
            with step4:
                st.markdown("### 🔀 4. Crossover")
                st.write("The parent designs swapped 'genes' (cross-sectional areas).")
                st.write("*For example, a child took the thick base legs of Parent A and the thin cross-bracing of Parent B.*")
                
            with step5:
                st.markdown("### 🎲 5. Mutation")
                st.write("To prevent the designs from getting stuck, the AI randomly mutated (tweaked) a few steel bars by +/- 10%.")
                st.write("*This injected fresh ideas into the gene pool.*")
                
            with step6:
                st.markdown("### 🔄 6. Evolution")
                st.write(f"This entire process repeated for **{generations} generations**.")
                st.write(f"*The graph below shows how the 'fittest' design lost weight over time until it couldn't get any lighter without breaking.*")

                
        # --- Plotly Convergence Graph ---
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            y=convergence, 
            mode='lines',
            name='Best Weight',
            line=dict(color='#FF4B4B', width=3)
        ))
        
        fig_conv.update_layout(
            title="Genetic Algorithm Convergence History",
            xaxis_title="Generation (Evolutionary Step)",
            yaxis_title="Total Structural Weight (kg)",
            template="plotly_white",
            hovermode="x",
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        st.plotly_chart(fig_conv, width="stretch")
        
        # --- Display Optimised Variables ---
        st.subheader("📊 Optimized Cross-Sectional Areas")
        opt_data = []
        for group_idx, opt_area in enumerate(best_chromosome):
            members_in_group = ", ".join([f"M{m}" for m in groups[group_idx]])
            
            # Truncate long member lists for the UI if optimising independently
            if len(members_in_group) > 50:
                members_in_group = members_in_group[:47] + "..."
                
            opt_data.append([f"Var {group_idx + 1}", members_in_group, f"{opt_area * 10000:.4f}"])
            
        df_opt = pd.DataFrame(opt_data, columns=["Design Variable", "Assigned Members", "Optimized Area (cm²)"])
        st.dataframe(df_opt, use_container_width=True)
            
       
