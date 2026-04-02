import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core_solver import TrussSystem, Node, Member
import datetime
import os
from visualizer import draw_undeformed_geometry, draw_results_fbd

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
            [15,17,A_brace,E_steel], [16,18,A_brace,E_steel]   # L3 Diagonals
            
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
                st.dataframe(pd.DataFrame(ts.K_global).style.format("{:.2e}"))
            with st.expander("View Reduced Stiffness Matrix ($K_{ff}$)", expanded=False):
                st.dataframe(pd.DataFrame(ts.K_reduced).style.format("{:.2e}"))

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
