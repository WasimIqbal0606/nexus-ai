
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
import random
import math
import json
import time
from datetime import datetime, timedelta
import altair as alt
from streamlit_option_menu import option_menu

# Configure API URL (can be updated for ngrok)
API_URL = "http://127.0.0.1:8000"  # Default local URL

# Function to make API requests
def api_request(endpoint, method="GET", data=None):
    """Make requests to the FastAPI backend"""
    url = f"{API_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)

        if response.status_code in (200, 201):
            return response.json()
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

# Set page configuration
st.set_page_config(
    page_title="Quantum Task Management System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
        background: linear-gradient(90deg, #4338CA, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #F1F5F9;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .card-title {
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.5rem;
    }
    .card-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F1F5F9;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE;
        border-bottom: 2px solid #3B82F6;
    }
    .task-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .task-title {
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .task-status {
        font-size: 0.8rem;
        padding: 3px 8px;
        border-radius: 12px;
        display: inline-block;
        margin-right: 5px;
    }
    .status-PENDING {
        background-color: #DBEAFE;
        color: #1E40AF;
    }
    .status-IN_PROGRESS {
        background-color: #FEF3C7;
        color: #92400E;
    }
    .status-COMPLETED {
        background-color: #DCFCE7;
        color: #166534;
    }
    .status-BLOCKED {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    .quantum-badge {
        background: linear-gradient(90deg, #4338CA, #3B82F6);
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-left: 5px;
    }
    .sidebar-header {
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 1rem;
    }
    .small-text {
        font-size: 0.8rem;
        color: #64748B;
    }
    /* Custom meter styling */
    .quantum-meter {
        width: 100%;
        height: 8px;
        background-color: #E2E8F0;
        border-radius: 4px;
        overflow: hidden;
        margin: 8px 0;
    }
    .quantum-meter-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #4338CA, #3B82F6);
    }
    /* Custom tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: #0F172A;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    /* Quantum animation */
    @keyframes quantum-pulse {
        0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); }
        100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
    }
    .quantum-pulse {
        animation: quantum-pulse 2s infinite;
    }
    /* Dashboard grid layout */
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 15px;
        margin-top: 15px;
    }
    /* Tag styling */
    .task-tag {
        background-color: #E0E7FF;
        color: #4338CA;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 10px;
        margin-right: 5px;
        display: inline-block;
    }
    /* Assignee avatar */
    .avatar-circle {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: linear-gradient(45deg, #4338CA, #3B82F6);
        display: inline-flex;
        justify-content: center;
        align-items: center;
        color: white;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 5px;
    }
    /* Quantum matrix display */
    .quantum-matrix {
        font-family: monospace;
        background-color: #1E293B;
        color: #38BDF8;
        border-radius: 5px;
        padding: 10px;
        overflow-x: auto;
        white-space: pre;
    }
    /* Task detail panel */
    .task-detail-panel {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    /* ML Insights card */
    .ml-insight-card {
        background-color: #F8FAFC;
        border-left: 3px solid #8B5CF6;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'tasks' not in st.session_state:
    st.session_state.tasks = []
if 'show_create_task' not in st.session_state:
    st.session_state.show_create_task = False
if 'selected_task' not in st.session_state:
    st.session_state.selected_task = None
if 'task_filter' not in st.session_state:
    st.session_state.task_filter = "all"
if 'run_simulation' not in st.session_state:
    st.session_state.run_simulation = False
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'run_optimization' not in st.session_state:
    st.session_state.run_optimization = False
if 'optimization_result' not in st.session_state:
    st.session_state.optimization_result = None
if 'system_metrics' not in st.session_state:
    st.session_state.system_metrics = None

# Helper functions
def fetch_tasks():
    """Fetch tasks from the API"""
    response = api_request("/tasks")
    if response:
        st.session_state.tasks = response
        return response
    return []

def fetch_metrics():
    """Fetch system metrics from the API"""
    response = api_request("/metrics")
    if response:
        st.session_state.system_metrics = response
        return response
    return None

def create_task(task_data):
    """Create a new task via API"""
    response = api_request("/tasks", method="POST", data=task_data)
    if response:
        st.success(f"Created task '{task_data['title']}'")
        fetch_tasks()
        return response
    return None

def update_task(task_id, task_data):
    """Update a task via API"""
    response = api_request(f"/tasks/{task_id}", method="PUT", data=task_data)
    if response:
        st.success(f"Updated task '{task_data.get('title', 'Unknown')}'")
        fetch_tasks()
        return response
    return None

def delete_task(task_id):
    """Delete a task via API"""
    response = api_request(f"/tasks/{task_id}", method="DELETE")
    if response:
        st.success(f"Deleted task #{task_id}")
        fetch_tasks()
        return response
    return None

def create_entanglement(entanglement_data):
    """Create a new entanglement between tasks"""
    response = api_request("/entanglements", method="POST", data=entanglement_data)
    if response:
        st.success(f"Created entanglement between tasks #{entanglement_data['task_id_1']} and #{entanglement_data['task_id_2']}")
        fetch_tasks()
        return response
    return None

def run_quantum_simulation(task_ids, steps=5):
    """Run a quantum simulation on selected tasks"""
    simulation_data = {
        "task_ids": task_ids,
        "simulation_steps": steps,
        "decoherence_rate": 0.05,
        "measurement_type": "projective"
    }
    response = api_request("/quantum-simulation", method="POST", data=simulation_data)
    if response:
        return response
    return None

def optimize_assignments():
    """Run the task assignment optimization algorithm"""
    response = api_request("/optimize-assignments", method="POST", data={})
    if response:
        return response
    return None

def search_tasks(query, use_quantum=False):
    """Search for tasks using the API"""
    search_data = {
        "query": query,
        "limit": 10,
        "use_quantum": use_quantum
    }
    response = api_request("/search", method="POST", data=search_data)
    if response:
        return response
    return []

def generate_bloch_sphere(state_vector):
    """Generate a Bloch sphere visualization for a quantum state"""
    # Convert state_vector to numpy array if needed
    if not isinstance(state_vector, np.ndarray):
        state_vector = np.array(state_vector)

    # Normalize state vector if needed
    if np.linalg.norm(state_vector) != 0:
        state_vector = state_vector / np.linalg.norm(state_vector)

    # Calculate theta and phi for spherical coordinates
    theta = np.arccos(state_vector[0]) if len(state_vector) > 0 else 0
    phi = np.arctan2(state_vector[2], state_vector[1]) if len(state_vector) > 2 else 0

    # Create a sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Create the figure
    fig = go.Figure()

    # Add the sphere
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.2, colorscale='Blues', showscale=False))

    # Add the axes
    axis_length = 1.2
    axis_points = [-axis_length, axis_length]
    zeros = [0, 0]

    # X-axis
    fig.add_trace(go.Scatter3d(x=axis_points, y=zeros, z=zeros, mode='lines', line=dict(color='red', width=4), name='X'))

    # Y-axis
    fig.add_trace(go.Scatter3d(x=zeros, y=axis_points, z=zeros, mode='lines', line=dict(color='green', width=4), name='Y'))

    # Z-axis
    fig.add_trace(go.Scatter3d(x=zeros, y=zeros, z=axis_points, mode='lines', line=dict(color='blue', width=4), name='Z'))

    # Add the state vector
    state_x = np.sin(theta) * np.cos(phi)
    state_y = np.sin(theta) * np.sin(phi)
    state_z = np.cos(theta)

    fig.add_trace(go.Scatter3d(
        x=[0, state_x],
        y=[0, state_y],
        z=[0, state_z],
        mode='lines+markers',
        line=dict(color='purple', width=6),
        marker=dict(size=[0, 8], color='purple'),
        name='State'
    ))

    # Update layout
    fig.update_layout(
        title='Quantum State Visualization',
        height=500,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
    )

    return fig

def create_entanglement_network(tasks):
    """Create a network graph of task entanglements"""
    G = nx.Graph()

    # Add nodes
    for task in tasks:
        G.add_node(task['id'],
                   title=task['title'],
                   state=task['state'],
                   entropy=task.get('entropy', 0.5),
                   probability=task.get('probability_distribution', {}).get(task['state'], 0.5))

    # Add edges for entanglements
    added_edges = set()
    for task in tasks:
        for entangled_id in task.get('entangled_tasks', []):
            # Create a unique edge identifier (always use smaller id first)
            edge_id = tuple(sorted([task['id'], entangled_id]))

            if edge_id not in added_edges and task['id'] != entangled_id:
                # Find entanglement data
                entanglement_data = None
                for e in api_request("/entanglements") or []:
                    if (e['task_id_1'] == task['id'] and e['task_id_2'] == entangled_id) or \
                       (e['task_id_1'] == entangled_id and e['task_id_2'] == task['id']):
                        entanglement_data = e
                        break

                if entanglement_data:
                    G.add_edge(task['id'], entangled_id,
                               weight=entanglement_data.get('strength', 0.5),
                               type=entanglement_data.get('entanglement_type', 'standard'))
                else:
                    # Default values if entanglement details not found
                    G.add_edge(task['id'], entangled_id, weight=0.5, type='standard')

                added_edges.add(edge_id)

    # Convert to plotly
    edge_x = []
    edge_y = []
    edge_text = []

    pos = nx.spring_layout(G, seed=42)

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

        # Edge attributes
        edge_type = edge[2].get('type', '')
        edge_weight = edge[2].get('weight', 0.5)
        edge_text.append(f"Type: {edge_type}<br>Strength: {edge_weight:.2f}")

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []

    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)

        # Node attributes
        node_title = node[1].get('title', '')
        node_state = node[1].get('state', '')
        node_entropy = node[1].get('entropy', 0)
        node_probability = node[1].get('probability', 0)

        node_text.append(f"ID: {node[0]}<br>Title: {node_title}<br>State: {node_state}<br>Entropy: {node_entropy:.2f}")

        # Size based on entropy
        node_size.append(30 + node_entropy * 50)

        # Color based on state
        if node_state == 'PENDING':
            node_color.append('#3B82F6')  # Blue
        elif node_state == 'IN_PROGRESS':
            node_color.append('#F59E0B')  # Amber
        elif node_state == 'COMPLETED':
            node_color.append('#10B981')  # Green
        elif node_state == 'BLOCKED':
            node_color.append('#EF4444')  # Red
        else:
            node_color.append('#6B7280')  # Gray

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=3, color='#CBD5E1'),
        hoverinfo='text',
        text=edge_text,
        mode='lines')

    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line=dict(width=2, color='white')))

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='Task Entanglement Network',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600,
                    plot_bgcolor='#F8FAFC'))

    return fig

def render_task_card(task):
    """Render a task card with quantum properties"""
    # Set status class
    status_class = f"status-{task['state']}"

    # Format tags
    tags_html = ""
    for tag in task.get('tags', []):
        tags_html += f'<span class="task-tag">{tag}</span>'

    # Create assignee avatar
    if task.get('assignee'):
        initials = ''.join([name[0] for name in task['assignee'].split() if name])
        avatar_html = f'<div class="avatar-circle">{initials}</div> {task["assignee"]}'
    else:
        avatar_html = '<span class="text-gray-400">Unassigned</span>'

    # Format due date
    due_date_text = ""
    if task.get('due_date'):
        try:
            due_date = datetime.fromisoformat(task['due_date'].replace('Z', '+00:00'))
            today = datetime.now()
            days_until_due = (due_date.date() - today.date()).days

            if days_until_due < 0:
                due_text = f'<span style="color: #EF4444; font-weight: 600;">Overdue by {abs(days_until_due)} days</span>'
            elif days_until_due == 0:
                due_text = '<span style="color: #F59E0B; font-weight: 600;">Due today</span>'
            else:
                due_text = f'<span style="color: #64748B;">Due in {days_until_due} days</span>'
            due_date_text = f'<div class="text-xs text-gray-500">Due Date:</div><div>{due_text}</div>'
        except Exception:
            due_date_text = f'<div class="text-xs text-gray-500">Due Date:</div><div>{task["due_date"]}</div>'

    # Get quantum properties
    entropy = task.get('entropy', 0.5)

    # Get coherence and probability from quantum_state or use defaults
    coherence = 0.5
    probability = 0.5

    if task.get('quantum_state'):
        if 'fidelity' in task['quantum_state']:
            coherence = task['quantum_state']['fidelity']
        if 'eigenvalues' in task['quantum_state']:
            # Use the largest eigenvalue as a probability measure
            eigenvalues = task['quantum_state']['eigenvalues']
            if eigenvalues:
                probability = max(eigenvalues)

    # If we have a probability distribution, use the value for the current state
    if task.get('probability_distribution') and task['state'] in task['probability_distribution']:
        probability = task['probability_distribution'][task['state']]

    # Create quantum meters
    coherence_width = int(coherence * 100)
    entropy_width = int(entropy * 100)
    probability_width = int(probability * 100)

    html = f'''
    <div class="task-card">
        <div class="flex justify-between items-start mb-2">
            <div class="task-title">{task['title']}</div>
            <span class="task-status {status_class}">{task['state'].replace('_', ' ').title()}</span>
        </div>
        <div class="text-sm text-gray-600 mb-2">{task['description'][:100]}{'...' if len(task['description']) > 100 else ''}</div>
        <div class="flex flex-wrap gap-1 mb-2">
            {tags_html}
        </div>
        <div class="grid grid-cols-2 gap-4 mb-2">
            <div>
                <div class="text-xs text-gray-500">Assignee:</div>
                <div class="flex items-center">{avatar_html}</div>
            </div>
            <div>
                {due_date_text}
            </div>
        </div>
        <div class="mt-3">
            <div class="flex justify-between items-center text-xs mb-1">
                <span>Coherence</span>
                <span>{coherence:.2f}</span>
            </div>
            <div class="quantum-meter">
                <div class="quantum-meter-fill" style="width: {coherence_width}%"></div>
            </div>

            <div class="flex justify-between items-center text-xs mb-1">
                <span>Entropy</span>
                <span>{entropy:.2f}</span>
            </div>
            <div class="quantum-meter">
                <div class="quantum-meter-fill" style="width: {entropy_width}%"></div>
            </div>

            <div class="flex justify-between items-center text-xs mb-1">
                <span>Probability</span>
                <span>{probability:.2f}</span>
            </div>
            <div class="quantum-meter">
                <div class="quantum-meter-fill" style="width: {probability_width}%"></div>
            </div>
        </div>

        <div class="mt-3">
            <div class="text-xs text-gray-500">Entangled with:</div>
            <div class="text-sm">
                {', '.join([f"Task #{id}" for id in task.get('entangled_tasks', [])]) if task.get('entangled_tasks') else 'None'}
            </div>
        </div>
    </div>
    '''

    return html

def generate_random_quantum_matrix():
    """Generate a random quantum state matrix for visualization"""
    # Create a random 2x2 complex matrix
    real_part = np.random.rand(2, 2)
    imag_part = np.random.rand(2, 2)
    matrix = real_part + 1j * imag_part

    # Make it Hermitian (to represent a valid quantum operator)
    matrix = 0.5 * (matrix + matrix.conj().T)

    # Format for display
    formatted = ""
    for i in range(2):
        row = "| "
        for j in range(2):
            val = matrix[i, j]
            row += f"{val.real:.2f} + {val.imag:.2f}i "
        row += "|\n"
        formatted += row

    return formatted

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "Quantum Tasks",
        ["Dashboard", "Task List", "Simulation", "Optimization", "ML Insights"],
        icons=["house", "list-task", "braces-asterisk", "graph-up-arrow", "cpu"],
        menu_icon="atom",
        default_index=0,
    )

    st.markdown("### Quantum Metrics")

    # Fetch current metrics
    metrics = fetch_metrics()

    if metrics:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("System Entropy", f"{metrics.get('total_entropy', 0):.2f}")
            st.metric("Quantum Coherence", f"{metrics.get('quantum_coherence', 0):.2f}")
        with col2:
            st.metric("Completion Rate", f"{metrics.get('completion_rate', 0)*100:.1f}%")

            # Count tasks by status
            tasks = st.session_state.tasks
            if not tasks:
                fetch_tasks()
                tasks = st.session_state.tasks

            completed = sum(1 for task in tasks if task.get('state') == 'COMPLETED')
            st.metric("Tasks Completed", f"{completed}/{len(tasks)}")
    else:
        st.info("Connect to backend to view metrics")

    st.markdown("### Quantum Matrix")
    st.markdown(f'<div class="quantum-matrix">{generate_random_quantum_matrix()}</div>', unsafe_allow_html=True)

    if selected == "Task List":
        st.markdown("### Filters")
        status_filter = st.selectbox(
            "Status",
            ["all", "PENDING", "IN_PROGRESS", "COMPLETED", "BLOCKED"],
            index=0
        )
        st.session_state.task_filter = status_filter

        if st.button("New Task", use_container_width=True):
            st.session_state.show_create_task = True

    # API Configuration section
    with st.expander("API Configuration"):
        api_url = st.text_input("API URL", value=API_URL)
        if st.button("Update API URL"):
            API_URL = api_url
            st.success(f"API URL updated to {API_URL}")

        if st.button("Test Connection"):
            try:
                response = api_request("/")
                if response:
                    st.success("Connection successful!")
                    st.json(response)
                else:
                    st.error("Failed to connect to API")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

# Main content area
# Dashboard
if selected == "Dashboard":
    st.markdown('<h1 class="main-header">Quantum Task Management Dashboard</h1>', unsafe_allow_html=True)

    # Fetch latest data
    tasks = fetch_tasks()
    metrics = fetch_metrics()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Active Tasks</div>', unsafe_allow_html=True)
        active_count = sum(1 for task in tasks if task.get('state') in ['PENDING', 'IN_PROGRESS'])
        st.markdown(f'<div class="card-value">{active_count}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">System Coherence</div>', unsafe_allow_html=True)
        system_coherence = metrics.get('quantum_coherence', 0) if metrics else 0
        st.markdown(f'<div class="card-value">{system_coherence:.2f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Average Entropy</div>', unsafe_allow_html=True)
        avg_entropy = metrics.get('total_entropy', 0) / max(1, metrics.get('task_count', 1)) if metrics else 0
        st.markdown(f'<div class="card-value">{avg_entropy:.2f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Cognitive Load</div>', unsafe_allow_html=True)
        cognitive_load = metrics.get('average_cognitive_load', 0) if metrics else 0
        st.markdown(f'<div class="card-value">{cognitive_load:.2f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Create two columns layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Show task entanglement network
        st.subheader("Task Entanglement Network")
        with st.spinner("Generating entanglement network..."):
            network_fig = create_entanglement_network(tasks)
            st.plotly_chart(network_fig, use_container_width=True)

        # Recent tasks
        st.subheader("Recent Tasks")

        # Sort tasks by updated_at (most recent first)
        sorted_tasks = sorted(tasks, key=lambda x: x.get('updated_at', ''), reverse=True)

        # Display the 5 most recent tasks
        for task in sorted_tasks[:5]:
            st.markdown(render_task_card(task), unsafe_allow_html=True)

    with col2:
        # Task status distribution
        st.subheader("Task Status Distribution")

        # Count tasks by status
        status_counts = {}
        for task in tasks:
            status = task.get('state', 'PENDING')
            if status in status_counts:
                status_counts[status] += 1
            else:
                status_counts[status] = 1

        # Create pie chart
        status_df = pd.DataFrame({
            'Status': list(status_counts.keys()),
            'Count': list(status_counts.values())
        })

        status_colors = {
            'PENDING': '#3B82F6',
            'IN_PROGRESS': '#F59E0B',
            'COMPLETED': '#10B981',
            'BLOCKED': '#EF4444'
        }

        fig = px.pie(
            status_df,
            names='Status',
            values='Count',
            color='Status',
            color_discrete_map=status_colors,
            hole=0.4
        )

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            legend_title_text=''
        )

        st.plotly_chart(fig, use_container_width=True)

        # Task priority distribution
        st.subheader("Task Priority Distribution")

        # Count tasks by priority
        priority_counts = {}
        for task in tasks:
            priority = task.get('priority', 1)
            if priority in priority_counts:
                priority_counts[priority] += 1
            else:
                priority_counts[priority] = 1

        # Create bar chart
        priority_df = pd.DataFrame({
            'Priority': list(priority_counts.keys()),
            'Count': list(priority_counts.values())
        })

        fig = px.bar(
            priority_df,
            x='Priority',
            y='Count',
            color='Count',
            color_continuous_scale='blues',
            labels={'Count': 'Number of Tasks', 'Priority': 'Priority Level'}
        )

        fig.update_layout(
            xaxis=dict(tickmode='linear', dtick=1),
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Quick search
        st.subheader("Quick Search")
        search_query = st.text_input("Search tasks")
        use_quantum = st.checkbox("Use quantum search algorithm")

        if search_query:
            with st.spinner("Searching..."):
                search_results = search_tasks(search_query, use_quantum)

                if search_results:
                    st.success(f"Found {len(search_results)} results")
                    for result in search_results:
                        task = result.get('task')
                        if task:
                            st.markdown(f"**[{task['title']}]** (ID: {task['id']}) - Score: {result.get('relevance_score', 0):.2f}")
                else:
                    st.warning("No results found")

# Task List
elif selected == "Task List":
    st.markdown('<h1 class="main-header">Quantum Task List</h1>', unsafe_allow_html=True)

    # Fetch tasks
    if not st.session_state.tasks:
        with st.spinner("Loading tasks..."):
            fetch_tasks()

    # Create new task form
    if st.session_state.show_create_task:
        with st.expander("Create New Task", expanded=True):
            with st.form("new_task_form"):
                col1, col2 = st.columns(2)

                with col1:
                    title = st.text_input("Task Title")
                    description = st.text_area("Description")
                    assignee = st.text_input("Assignee")
                    priority = st.slider("Priority", 1, 5, 3)

                with col2:
                    state = st.selectbox("Status", ["PENDING", "IN_PROGRESS", "COMPLETED", "BLOCKED"])
                    due_date = st.date_input("Due Date")
                    tags = st.text_input("Tags (comma separated)")

                submit = st.form_submit_button("Create Task")

                if submit:
                    if not title:
                        st.error("Title is required")
                    else:
                        # Prepare data
                        tag_list = [tag.strip() for tag in tags.split(',')] if tags else []
                        task_data = {
                            "title": title,
                            "description": description,
                            "assignee": assignee if assignee else None,
                            "due_date": due_date.isoformat() if due_date else None,
                            "state": state,
                            "tags": tag_list,
                            "priority": priority
                        }

                        # Create task via API
                        with st.spinner("Creating task..."):
                            result = create_task(task_data)
                            if result:
                                st.session_state.show_create_task = False
                                st.experimental_rerun()

    # Filter tasks based on selected filter
    tasks = st.session_state.tasks
    if st.session_state.task_filter != "all":
        tasks = [task for task in tasks if task.get('state') == st.session_state.task_filter]

    # Create tabs
    tabs = st.tabs(["Card View", "Table View", "Quantum View"])

    with tabs[0]:  # Card View
        # Create a 3-column layout for task cards
        cols = st.columns(3)

        # Display tasks in cards
        for i, task in enumerate(tasks):
            with cols[i % 3]:
                # Render task card
                st.markdown(render_task_card(task), unsafe_allow_html=True)

                # Add a button to view details
                if st.button(f"View Details #{task['id']}", key=f"view_{task['id']}"):
                    st.session_state.selected_task = task

    with tabs[1]:  # Table View
        # Convert to DataFrame for table view
        if tasks:
            tasks_df = pd.DataFrame(tasks)

            # Select columns to display
            if not tasks_df.empty:
                # Check which columns are available
                available_cols = tasks_df.columns
                display_cols = []

                for col in ["id", "title", "state", "priority", "assignee", "due_date", "created_at", "updated_at"]:
                    if col in available_cols:
                        display_cols.append(col)

                if display_cols:
                    st.dataframe(tasks_df[display_cols], use_container_width=True)
                else:
                    st.dataframe(tasks_df, use_container_width=True)
        else:
            st.info("No tasks found. Create a new task to get started.")

    with tabs[2]:  # Quantum View
        st.subheader("Quantum State Visualization")

        if tasks:
            # Create selector for task
            task_options = [f"#{task['id']} - {task['title']}" for task in tasks]
            selected_task_option = st.selectbox("Select a task to visualize", task_options)

            # Extract task ID from selection
            task_id = selected_task_option.split(' - ')[0][1:]

            # Find the selected task
            selected_task = next((task for task in tasks if str(task['id']) == task_id), None)

            if selected_task:
                col1, col2 = st.columns([3, 2])

                with col1:
                    # Get coherence, entropy, and probability for Bloch sphere
                    entropy = selected_task.get('entropy', 0.5)

                    # Get coherence and probability from quantum_state or use defaults
                    coherence = 0.5
                    probability = 0.5

                    if selected_task.get('quantum_state'):
                        if 'fidelity' in selected_task['quantum_state']:
                            coherence = selected_task['quantum_state']['fidelity']
                        if 'eigenvalues' in selected_task['quantum_state']:
                            # Use the largest eigenvalue as a probability measure
                            eigenvalues = selected_task['quantum_state']['eigenvalues']
                            if eigenvalues:
                                probability = max(eigenvalues)

                    # If we have a probability distribution, use the value for the current state
                    if selected_task.get('probability_distribution') and selected_task['state'] in selected_task['probability_distribution']:
                        probability = selected_task['probability_distribution'][selected_task['state']]

                    # Create state vector for Bloch sphere
                    state_vector = [coherence, entropy, 1-probability]

                    # Generate and display Bloch sphere
                    bloch_fig = generate_bloch_sphere(state_vector)
                    st.plotly_chart(bloch_fig, use_container_width=True)

                with col2:
                    st.markdown("### Quantum Properties")

                    # Display quantum properties
                    st.markdown(f"**Coherence:** {coherence:.2f}")
                    st.markdown(f"**Entropy:** {entropy:.2f}")
                    st.markdown(f"**Probability:** {probability:.2f}")

                    # Display entanglements
                    st.markdown("### Entanglements")

                    entangled_tasks = selected_task.get('entangled_tasks', [])
                    if entangled_tasks:
                        for entangled_id in entangled_tasks:
                            entangled_task = next((t for t in st.session_state.tasks if t['id'] == entangled_id), None)
                            if entangled_task:
                                st.markdown(f"- **Task #{entangled_id}:** {entangled_task['title']}")

                                # Try to find entanglement details
                                entanglement_details = None
                                for e in api_request("/entanglements") or []:
                                    if (e['task_id_1'] == selected_task['id'] and e['task_id_2'] == entangled_id) or \
                                       (e['task_id_1'] == entangled_id and e['task_id_2'] == selected_task['id']):
                                        entanglement_details = e
                                        break

                                if entanglement_details:
                                    st.markdown(f"  Type: {entanglement_details.get('entanglement_type', 'standard')}, Strength: {entanglement_details.get('strength', 0.5):.2f}")
                    else:
                        st.markdown("No entanglements detected.")

                    # Display quantum matrix representation
                    st.markdown("### Quantum Matrix Representation")

                    # Check if we have eigenvalues in the quantum state
                    if selected_task.get('quantum_state') and 'eigenvalues' in selected_task['quantum_state']:
                        eigenvalues = selected_task['quantum_state']['eigenvalues']

                        # Create a simple 2x2 matrix based on eigenvalues
                        if len(eigenvalues) >= 2:
                            matrix = np.array([
                                [eigenvalues[0], 0],
                                [0, eigenvalues[1]]
                            ])
                        else:
                            # Default matrix if not enough eigenvalues
                            matrix = np.array([
                                [coherence, np.sqrt(entropy * probability)],
                                [np.sqrt(entropy * probability), 1 - coherence]
                            ])
                    else:
                        # Create a simple 2x2 matrix based on task properties
                        matrix = np.array([
                            [coherence, np.sqrt(entropy * probability)],
                            [np.sqrt(entropy * probability), 1 - coherence]
                        ])

                    # Format for display
                    matrix_text = ""
                    for i in range(2):
                        row = "| "
                        for j in range(2):
                            row += f"{matrix[i, j]:.2f} "
                        row += "|\n"
                        matrix_text += row

                    st.markdown(f'<div class="quantum-matrix">{matrix_text}</div>', unsafe_allow_html=True)
        else:
            st.info("No tasks found. Create a new task to visualize quantum states.")

    # Task Detail Panel if a task is selected
    if st.session_state.selected_task:
        with st.expander("Task Details", expanded=True):
            task = st.session_state.selected_task

            # Create a two-column layout
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"## {task['title']}")

                # Create status badge
                status_colors = {
                    'PENDING': 'blue',
                    'IN_PROGRESS': 'orange',
                    'COMPLETED': 'green',
                    'BLOCKED': 'red'
                }
                status_color = status_colors.get(task.get('state', 'PENDING'), 'gray')
                st.markdown(f"<span style='background-color: {status_color}; color: white; padding: 3px 8px; border-radius: 3px;'>{task.get('state', 'PENDING').replace('_', ' ').title()}</span>", unsafe_allow_html=True)

                st.markdown("### Description")
                st.write(task.get('description', 'No description'))

                # Task metadata
                col_meta1, col_meta2 = st.columns(2)
                with col_meta1:
                    st.markdown("### Details")
                    st.markdown(f"**ID:** {task['id']}")
                    st.markdown(f"**Priority:** {task.get('priority', 1)}")
                    st.markdown(f"**Assignee:** {task.get('assignee', 'Unassigned')}")

                    created_at = datetime.fromisoformat(task['created_at'].replace('Z', '+00:00')) if 'created_at' in task else None
                    if created_at:
                        st.markdown(f"**Created:** {created_at.strftime('%Y-%m-%d %H:%M')}")

                    due_date = task.get('due_date')
                    if due_date:
                        st.markdown(f"**Due Date:** {due_date}")

                with col_meta2:
                    st.markdown("### Quantum Properties")

                    # Get quantum properties
                    entropy = task.get('entropy', 0.5)

                    # Get coherence and probability
                    coherence = 0.5
                    probability = 0.5

                    if task.get('quantum_state'):
                        if 'fidelity' in task['quantum_state']:
                            coherence = task['quantum_state']['fidelity']
                        if task.get('probability_distribution') and task['state'] in task['probability_distribution']:
                            probability = task['probability_distribution'][task['state']]

                    # Create progress bars for quantum properties
                    coherence_progress = coherence * 100
                    entropy_progress = entropy * 100
                    probability_progress = probability * 100

                    st.markdown("**Coherence:**")
                    st.progress(coherence_progress / 100)
                    st.markdown(f"{coherence_progress:.1f}%")

                    st.markdown("**Entropy:**")
                    st.progress(entropy_progress / 100)
                    st.markdown(f"{entropy_progress:.1f}%")

                    st.markdown("**Probability:**")
                    st.progress(probability_progress / 100)
                    st.markdown(f"{probability_progress:.1f}%")

                # Tags
                st.markdown("### Tags")
                tag_html = ""
                for tag in task.get('tags', []):
                    tag_html += f'<span style="background-color: #E0E7FF; color: #4338CA; padding: 2px 8px; border-radius: 10px; margin-right: 5px; font-size: 0.8rem;">{tag}</span>'
                st.markdown(tag_html, unsafe_allow_html=True)

                # Edit task section
                with st.form(f"edit_task_{task['id']}"):
                    st.subheader("Edit Task")

                    edit_col1, edit_col2 = st.columns(2)

                    with edit_col1:
                        new_title = st.text_input("Title", value=task.get('title', ''))
                        new_description = st.text_area("Description", value=task.get('description', ''))
                        new_assignee = st.text_input("Assignee", value=task.get('assignee', ''))

                    with edit_col2:
                        new_state = st.selectbox("Status", ["PENDING", "IN_PROGRESS", "COMPLETED", "BLOCKED"], index=["PENDING", "IN_PROGRESS", "COMPLETED", "BLOCKED"].index(task.get('state', 'PENDING')))

                        # Convert ISO date string to date object for input
                        due_date_value = None
                        if task.get('due_date'):
                            try:
                                due_date_value = datetime.fromisoformat(task['due_date'].replace('Z', '+00:00')).date()
                            except:
                                pass

                        new_due_date = st.date_input("Due Date", value=due_date_value if due_date_value else None)
                        new_priority = st.slider("Priority", 1, 5, task.get('priority', 1))

                        # Format tags for editing
                        current_tags = ', '.join(task.get('tags', []))
                        new_tags = st.text_input("Tags (comma separated)", value=current_tags)

                    update_button = st.form_submit_button("Update Task")

                    if update_button:
                        # Prepare update data
                        tag_list = [tag.strip() for tag in new_tags.split(',')] if new_tags else []
                        update_data = {
                            "title": new_title,
                            "description": new_description,
                            "assignee": new_assignee if new_assignee else None,
                            "due_date": new_due_date.isoformat() if new_due_date else None,
                            "state": new_state,
                            "tags": tag_list,
                            "priority": new_priority
                        }

                        # Update task via API
                        with st.spinner("Updating task..."):
                            result = update_task(task['id'], update_data)
                            if result:
                                st.session_state.selected_task = result
                                st.success("Task updated!")

                # Add entanglement section
                st.subheader("Create Entanglement")

                new_entanglement_col1, new_entanglement_col2 = st.columns(2)

                with new_entanglement_col1:
                    # Get other tasks to entangle with
                    other_tasks = [t for t in st.session_state.tasks if t['id'] != task['id']]
                    other_task_options = [f"#{t['id']} - {t['title']}" for t in other_tasks]

                    if other_task_options:
                        selected_task_to_entangle = st.selectbox("Task to entangle with", other_task_options)
                        other_task_id = selected_task_to_entangle.split(' - ')[0][1:]
                    else:
                        st.info("No other tasks available for entanglement")
                        other_task_id = None

                with new_entanglement_col2:
                    entanglement_type = st.selectbox("Entanglement Type", ["standard", "CNOT", "SWAP"])
                    entanglement_strength = st.slider("Entanglement Strength", 0.1, 1.0, 0.5, step=0.1)

                if st.button("Create Entanglement") and other_task_id:
                    # Check if entanglement already exists
                    existing_entanglements = api_request("/entanglements") or []
                    already_entangled = any(
                        (e['task_id_1'] == task['id'] and e['task_id_2'] == other_task_id) or
                        (e['task_id_1'] == other_task_id and e['task_id_2'] == task['id'])
                        for e in existing_entanglements
                    )

                    if already_entangled:
                        st.warning("These tasks are already entangled!")
                    else:
                        # Create entanglement via API
                        entanglement_data = {
                            "task_id_1": task['id'],
                            "task_id_2": other_task_id,
                            "strength": entanglement_strength,
                            "entanglement_type": entanglement_type
                        }

                        with st.spinner("Creating entanglement..."):
                            result = create_entanglement(entanglement_data)
                            if result:
                                st.success("Entanglement created!")
                                # Refresh task data
                                fetch_tasks()
                                # Update selected task
                                st.session_state.selected_task = next((t for t in st.session_state.tasks if t['id'] == task['id']), None)

                # Delete task button
                if st.button("Delete Task", key=f"delete_{task['id']}"):
                    if st.checkbox("Confirm deletion", key=f"confirm_delete_{task['id']}"):
                        with st.spinner("Deleting task..."):
                            result = delete_task(task['id'])
                            if result:
                                st.session_state.selected_task = None
                                st.success("Task deleted!")
                                st.experimental_rerun()

            with col2:
                # Visualization of the task's quantum state
                st.markdown("### Quantum State")

                # Get coherence, entropy, and probability for Bloch sphere
                entropy = task.get('entropy', 0.5)

                # Get coherence and probability from quantum_state or use defaults
                coherence = 0.5
                probability = 0.5

                if task.get('quantum_state'):
                    if 'fidelity' in task['quantum_state']:
                        coherence = task['quantum_state']['fidelity']
                    if 'eigenvalues' in task['quantum_state']:
                        # Use the largest eigenvalue as a probability measure
                        eigenvalues = task['quantum_state']['eigenvalues']
                        if eigenvalues:
                            probability = max(eigenvalues)

                # If we have a probability distribution, use the value for the current state
                if task.get('probability_distribution') and task['state'] in task['probability_distribution']:
                    probability = task['probability_distribution'][task['state']]

                # Create state vector for Bloch sphere
                state_vector = [coherence, entropy, 1-probability]

                # Generate Bloch sphere
                bloch_fig = generate_bloch_sphere(state_vector)
                st.plotly_chart(bloch_fig, use_container_width=True)

                # Entanglements
                st.markdown("### Entanglements")

                entangled_tasks = task.get('entangled_tasks', [])
                if entangled_tasks:
                    for entangled_id in entangled_tasks:
                        entangled_task = next((t for t in st.session_state.tasks if t['id'] == entangled_id), None)
                        if entangled_task:
                            st.markdown(f"**Task #{entangled_id}:** {entangled_task['title']}")

                            # Try to find entanglement details
                            entanglement_details = None
                            for e in api_request("/entanglements") or []:
                                if (e['task_id_1'] == task['id'] and e['task_id_2'] == entangled_id) or \
                                   (e['task_id_1'] == entangled_id and e['task_id_2'] == task['id']):
                                    entanglement_details = e
                                    break

                            if entanglement_details:
                                st.markdown(f"**Type:** {entanglement_details.get('entanglement_type', 'standard')}")

                                # Display entanglement strength
                                st.markdown("**Strength:**")
                                st.progress(entanglement_details.get('strength', 0.5))

                                # Add button to delete entanglement
                                if st.button(f"Delete Entanglement #{entanglement_details['id']}", key=f"delete_entanglement_{entanglement_details['id']}"):
                                    if api_request(f"/entanglements/{entanglement_details['id']}", method="DELETE"):
                                        st.success("Entanglement deleted!")
                                        # Refresh task data
                                        fetch_tasks()
                                        # Update selected task
                                        st.session_state.selected_task = next((t for t in st.session_state.tasks if t['id'] == task['id']), None)
                                        st.experimental_rerun()
                else:
                    st.markdown("No entanglements detected")

                # Close button
                if st.button("Close Details", key=f"close_{task['id']}"):
                    st.session_state.selected_task = None
                    st.experimental_rerun()

# Simulation
elif selected == "Simulation":
    st.markdown('<h1 class="main-header">Quantum Task Simulation</h1>', unsafe_allow_html=True)

    # Fetch tasks if not already loaded
    if not st.session_state.tasks:
        with st.spinner("Loading tasks..."):
            fetch_tasks()

    # Simulation configuration
    st.markdown("### Simulation Parameters")

    col1, col2 = st.columns(2)

    with col1:
        simulation_steps = st.slider("Simulation Steps", 5, 50, 20)
        decoherence_rate = st.slider("Decoherence Rate", 0.0, 1.0, 0.05, step=0.01)

    with col2:
        # Select tasks to simulate
        task_options = [f"#{task['id']} - {task['title']}" for task in st.session_state.tasks]
        selected_tasks = st.multiselect("Select Tasks to Simulate", task_options)

        # Extract task IDs
        selected_task_ids = [task_option.split(' - ')[0][1:] for task_option in selected_tasks]

        measurement_type = st.selectbox("Measurement Type", ["projective", "POVM", "weak"])

    # Run simulation button
    if st.button("Run Quantum Simulation") or st.session_state.run_simulation:
        if not selected_task_ids:
            st.warning("Please select at least one task to simulate")
        else:
            st.session_state.run_simulation = True

            # Show a spinner while calculating
            with st.spinner("Running quantum simulation..."):
                # Call the API to run the simulation
                simulation_data = {
                    "task_ids": selected_task_ids,
                    "simulation_steps": simulation_steps,
                    "decoherence_rate": decoherence_rate,
                    "measurement_type": measurement_type
                }

                st.session_state.simulation_results = api_request("/quantum-simulation", method="POST", data=simulation_data)

            if st.session_state.simulation_results:
                st.success("Simulation completed!")
            else:
                st.error("Simulation failed")

    # Display simulation results
    if st.session_state.simulation_results:
        results = st.session_state.simulation_results

        # Check for error
        if 'error' in results:
            st.error(f"Simulation error: {results['error']}")
        else:
            st.markdown("### Simulation Results")

            # Create tabs for different views
            sim_tabs = st.tabs(["Simulation Steps", "Final Results", "Entanglement Analysis"])

            with sim_tabs[0]:  # Simulation Steps
                st.subheader("Quantum State Evolution")

                # Display each simulation step
                step_selector = st.slider("Select Step", 0, len(results.get('simulation_steps', [])) - 1, 0)

                # Get the selected step
                if 'simulation_steps' in results and step_selector < len(results['simulation_steps']):
                    step = results['simulation_steps'][step_selector]

                    st.markdown(f"### Step {step.get('step', step_selector)} - Operation: {step.get('operation', 'Unknown')}")

                    # Display fidelity
                    st.markdown(f"**System Fidelity:** {step.get('fidelity', 0):.4f}")

                    # Display task states
                    task_states = step.get('task_states', {})

                    # Create a grid for task states
                    cols = st.columns(min(3, len(task_states)))

                    for i, (task_id, state) in enumerate(task_states.items()):
                        with cols[i % len(cols)]:
                            st.markdown(f"#### Task #{task_id}")
                            st.markdown(f"**Title:** {state.get('task_title', 'Unknown')}")

                            # Show probabilities
                            st.markdown("**State Probabilities:**")

                            # Pending probability
                            pending_prob = state.get('pending_prob', 0) * 100
                            st.markdown(f"Pending: {pending_prob:.1f}%")
                            st.progress(pending_prob / 100)

                            # Completed probability
                            completed_prob = state.get('completed_prob', 0) * 100
                            st.markdown(f"Completed: {completed_prob:.1f}%")
                            st.progress(completed_prob / 100)

                            # Coherence
                            coherence = state.get('coherence', 0)
                            st.markdown(f"Coherence: {coherence:.2f}")

            with sim_tabs[1]:  # Final Results
                st.subheader("Final Measurement Results")

                if 'final_results' in results:
                    final_results = results['final_results']

                    # Display measurement outcomes
                    st.markdown("### Measurement Outcomes")

                    if 'measurement_outcomes' in final_results:
                        outcomes = final_results['measurement_outcomes']

                        # Create a table for outcomes
                        outcome_data = []

                        for task_id, outcome in outcomes.items():
                            outcome_data.append({
                                "Task ID": task_id,
                                "Task": outcome.get('task_title', 'Unknown'),
                                "Outcome": outcome.get('outcome', 'Unknown')
                            })

                        if outcome_data:
                            outcome_df = pd.DataFrame(outcome_data)
                            st.dataframe(outcome_df, use_container_width=True)

                            # Create a bar chart of outcomes
                            outcome_counts = outcome_df['Outcome'].value_counts().reset_index()
                            outcome_counts.columns = ['Outcome', 'Count']

                            fig = px.bar(
                                outcome_counts,
                                x='Outcome',
                                y='Count',
                                color='Outcome',
                                title='Measurement Outcomes Distribution'
                            )

                            st.plotly_chart(fig, use_container_width=True)

                    # Display final state properties
                    if 'final_state' in final_results:
                        st.markdown("### Final Quantum States")
                        st.json(final_results['final_state'])

            with sim_tabs[2]:  # Entanglement Analysis
                st.subheader("Entanglement Analysis")

                if 'final_results' in results and 'entanglement_measure' in results['final_results']:
                    entanglement_measures = results['final_results']['entanglement_measure']

                    # Create a table for entanglement measures
                    entanglement_data = []

                    for pair, data in entanglement_measures.items():
                        # Extract task IDs from the pair string
                        task_ids = pair.split('-')
                        if len(task_ids) == 2:
                            entanglement_data.append({
                                "Task 1": f"#{task_ids[0]} - {data.get('task1_title', 'Unknown')}",
                                "Task 2": f"#{task_ids[1]} - {data.get('task2_title', 'Unknown')}",
                                "Concurrence": data.get('concurrence', 0)
                            })

                    if entanglement_data:
                        entanglement_df = pd.DataFrame(entanglement_data)

                        # Sort by concurrence (highest first)
                        entanglement_df = entanglement_df.sort_values('Concurrence', ascending=False)

                        st.dataframe(entanglement_df, use_container_width=True)

                        # Create a bar chart of entanglement measures
                        fig = px.bar(
                            entanglement_df,
                            x='Concurrence',
                            y=[f"{row['Task 1']} & {row['Task 2']}" for _, row in entanglement_df.iterrows()],
                            orientation='h',
                            title='Entanglement Measures',
                            labels={'y': 'Task Pair', 'x': 'Concurrence (Entanglement Measure)'}
                        )

                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})

                        st.plotly_chart(fig, use_container_width=True)

                        # Display insight
                        max_entanglement = entanglement_df['Concurrence'].max() if not entanglement_df.empty else 0

                        if max_entanglement > 0.7:
                            st.info("""
                            **Strong Entanglement Detected**

                            The simulation detected strong quantum entanglement between tasks. This suggests that
                            these tasks are highly interdependent and should be managed together for optimal results.
                            """)
                        elif max_entanglement > 0.3:
                            st.info("""
                            **Moderate Entanglement Detected**

                            The simulation detected moderate quantum entanglement between tasks. Consider coordinating
                            work on these tasks to maintain system coherence.
                            """)
                        else:
                            st.info("""
                            **Weak Entanglement Detected**

                            The tasks in this simulation show relatively weak quantum entanglement. They can be
                            managed independently with minimal effect on each other.
                            """)
                else:
                    st.info("No entanglement data available from the simulation.")

# Optimization
elif selected == "Optimization":
    st.markdown('<h1 class="main-header">Quantum Task Optimization</h1>', unsafe_allow_html=True)

    # Fetch tasks if not already loaded
    if not st.session_state.tasks:
        with st.spinner("Loading tasks..."):
            fetch_tasks()

    # Optimization options
    st.markdown("### Optimization Parameters")

    optimization_type = st.selectbox(
        "Optimization Type",
        ["Task Assignment", "Deadline Optimization", "Resource Allocation"],
        index=0
    )

    # Run optimization button
    if st.button("Run Quantum Optimization") or st.session_state.run_optimization:
        st.session_state.run_optimization = True

        # Show a spinner while calculating
        with st.spinner("Running quantum-inspired optimization algorithm..."):
            # Call the API to run the optimization
            if optimization_type == "Task Assignment":
                st.session_state.optimization_result = optimize_assignments()

        if st.session_state.optimization_result:
            st.success("Optimization completed!")
        else:
            st.error("Optimization failed")

    # Display optimization results
    if st.session_state.optimization_result:
        result = st.session_state.optimization_result

        st.markdown("### Optimization Results")

        # Create tabs for different views
        opt_tabs = st.tabs(["Recommendations", "Analysis", "Implementation"])

        with opt_tabs[0]:  # Recommendations
            st.subheader("Task Assignment Recommendations")

            if 'recommendations' in result:
                recommendations = result['recommendations']

                if recommendations:
                    # Create a table of recommendations
                    st.markdown(f"Found {len(recommendations)} recommended changes:")

                    # Convert to DataFrame for display
                    rec_data = []

                    for rec in recommendations:
                        rec_data.append({
                            "Task ID": rec.get('task_id', 'Unknown'),
                            "Task": rec.get('task_title', 'Unknown'),
                            "From": rec.get('from_assignee', 'Unassigned'),
                            "To": rec.get('to_assignee', 'Unassigned'),
                            "Load Improvement": rec.get('load_improvement', 0)
                        })

                    rec_df = pd.DataFrame(rec_data)
                    st.dataframe(rec_df, use_container_width=True)

                    # Create a visual representation of changes
                    if not rec_df.empty:
                        fig = px.bar(
                            rec_df,
                            y='Task',
                            x='Load Improvement',
                            orientation='h',
                            color='Load Improvement',
                            color_continuous_scale='blues',
                            title='Recommended Task Reassignments',
                            labels={'Load Improvement': 'Cognitive Load Improvement', 'Task': 'Task'}
                        )

                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})

                        st.plotly_chart(fig, use_container_width=True)

                    # Add button to apply recommendations
                    if st.button("Apply All Recommendations"):
                        # For each recommendation, update the task assignee
                        success_count = 0

                        for rec in recommendations:
                            task_id = rec.get('task_id')
                            new_assignee = rec.get('to_assignee')

                            if task_id and new_assignee is not None:
                                # Update task via API
                                result = update_task(task_id, {"assignee": new_assignee})
                                if result:
                                    success_count += 1

                        if success_count > 0:
                            st.success(f"Applied {success_count} recommendations successfully!")
                            # Reset optimization state and refresh tasks
                            st.session_state.run_optimization = False
                            fetch_tasks()
                            st.experimental_rerun()
                        else:
                            st.error("Failed to apply recommendations")
                else:
                    st.info("No recommendations found. Current assignments are optimal.")
            else:
                st.info("No recommendations available in the optimization results.")

        with opt_tabs[1]:  # Analysis
            st.subheader("Optimization Analysis")

            # Extract metrics from the result
            total_recommendations = result.get('total_recommendations', 0)
            st.markdown(f"Total recommendations: **{total_recommendations}**")

            # If there are load metrics in the recommendations, visualize them
            if 'recommendations' in result:
                recommendations = result['recommendations']

                if recommendations:
                    # Look for load improvement values
                    load_improvements = [rec.get('load_improvement', 0) for rec in recommendations]
                    avg_improvement = sum(load_improvements) / len(load_improvements) if load_improvements else 0

                    st.markdown(f"Average load improvement per task: **{avg_improvement:.2f}**")

                    # Create a histogram of improvement values
                    if load_improvements:
                        fig = px.histogram(
                            x=load_improvements,
                            nbins=10,
                            title='Distribution of Load Improvements',
                            labels={'x': 'Load Improvement', 'y': 'Count'}
                        )

                        st.plotly_chart(fig, use_container_width=True)

            # Add general analysis text
            st.markdown("""
            ### Key Findings

            The quantum optimization algorithm analyzed the current task assignments and
            identified opportunities for improvement based on:

            1. **Workload Balance** - Ensuring team members have equitable cognitive loads
            2. **Entanglement Preservation** - Keeping entangled tasks assigned to the same person
            3. **Expertise Matching** - Aligning tasks with team member skills
            4. **Quantum Coherence** - Maximizing overall system coherence

            The algorithm used simulated annealing with quantum-inspired perturbations to
            find a near-optimal solution to this NP-hard assignment problem.
            """)

        with opt_tabs[2]:  # Implementation
            st.subheader("Implementation Plan")

            if 'recommendations' in result:
                recommendations = result['recommendations']

                if recommendations:
                    # Group recommendations by assignee
                    assignee_changes = {}

                    for rec in recommendations:
                        to_assignee = rec.get('to_assignee', 'Unassigned')

                        if to_assignee not in assignee_changes:
                            assignee_changes[to_assignee] = []

                        assignee_changes[to_assignee].append({
                            "task_id": rec.get('task_id', 'Unknown'),
                            "task_title": rec.get('task_title', 'Unknown'),
                            "from_assignee": rec.get('from_assignee', 'Unassigned')
                        })

                    # Display changes by assignee
                    for assignee, changes in assignee_changes.items():
                        with st.expander(f"{assignee} - {len(changes)} new tasks"):
                            for change in changes:
                                st.markdown(f"- **{change['task_title']}** (from {change['from_assignee']})")

                            # Allow implementing just this assignee's changes
                            if st.button(f"Apply changes for {assignee}", key=f"apply_{assignee}"):
                                success_count = 0

                                for change in changes:
                                    task_id = change.get('task_id')

                                    if task_id:
                                        # Update task via API
                                        result = update_task(task_id, {"assignee": assignee})
                                        if result:
                                            success_count += 1

                                if success_count > 0:
                                    st.success(f"Applied {success_count} changes for {assignee}!")
                                    # Reset optimization state and refresh tasks
                                    st.session_state.run_optimization = False
                                    fetch_tasks()
                                    st.experimental_rerun()
                                else:
                                    st.error(f"Failed to apply changes for {assignee}")

                    # Provide implementation notes
                    st.markdown("""
                    ### Implementation Notes

                    For optimal results, consider these factors when implementing the recommendations:

                    1. **Prioritize high-impact changes** - Focus on changes with the largest improvement values
                    2. **Consider team context** - Account for factors not captured in the optimization model
                    3. **Phase implementation** - Implement changes gradually to minimize disruption
                    4. **Monitor outcomes** - Track system coherence and entropy after changes
                    """)
                else:
                    st.info("No recommendations to implement. Current assignments are optimal.")
            else:
                st.info("No implementation plan available.")

# ML Insights
elif selected == "ML Insights":
    st.markdown('<h1 class="main-header">ML-Powered Quantum Insights</h1>', unsafe_allow_html=True)

    # Fetch tasks if not already loaded
    if not st.session_state.tasks:
        with st.spinner("Loading tasks..."):
            fetch_tasks()

    # Create tabs for different types of insights
    insight_tabs = st.tabs(["Task Analysis", "Quantum Patterns", "Cognitive Load", "Predictions"])

    with insight_tabs[0]:  # Task Analysis
        st.subheader("ML Task Classification")

        # Group tasks by ML category if available
        tasks_by_category = {}

        for task in st.session_state.tasks:
            category = task.get('category', 'Uncategorized')

            if category not in tasks_by_category:
                tasks_by_category[category] = []

            tasks_by_category[category].append(task)

        # Display tasks by category
        for category, tasks in tasks_by_category.items():
            with st.expander(f"{category} ({len(tasks)} tasks)"):
                for task in tasks:
                    st.markdown(f"**{task['title']}** (ID: {task['id']})")

                    # Display ML summary if available
                    if task.get('ml_summary'):
                        st.markdown(f"*{task['ml_summary']}*")

                    # Display state and other key info
                    st.markdown(f"State: {task['state']} | Priority: {task.get('priority', 1)} | Entropy: {task.get('entropy', 0):.2f}")
                    st.markdown("---")

        # Display categorization insights
        st.subheader("Task Distribution")

        # Create pie chart of tasks by category
        category_counts = {category: len(tasks) for category, tasks in tasks_by_category.items()}

        category_df = pd.DataFrame({
            'Category': list(category_counts.keys()),
            'Count': list(category_counts.values())
        })

        fig = px.pie(
            category_df,
            names='Category',
            values='Count',
            title='Tasks by ML Category'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display sample ML insights
        st.subheader("ML-Generated Task Insights")

        # Example insights
        insights = [
            "**Development tasks** have the highest average entropy, suggesting more uncertainty in implementation.",
            "**Documentation tasks** show the highest completion probability based on quantum state analysis.",
            "Tasks with ML category 'Integration' have stronger quantum entanglement patterns than others.",
            "Combining task text analysis with quantum properties improves task outcome prediction by 32%."
        ]

        for insight in insights:
            st.markdown(f'<div class="ml-insight-card">{insight}</div>', unsafe_allow_html=True)

    with insight_tabs[1]:  # Quantum Patterns
        st.subheader("Quantum State Patterns")

        # Create a 3D scatter plot of tasks in quantum space (entropy, coherence, probability)
        task_data = []

        for task in st.session_state.tasks:
            # Get quantum properties
            entropy = task.get('entropy', 0.5)

            # Get coherence from quantum_state if available
            coherence = 0.5
            if task.get('quantum_state') and 'fidelity' in task['quantum_state']:
                coherence = task['quantum_state']['fidelity']

            # Get probability for the current state
            probability = 0.5
            if task.get('probability_distribution') and task['state'] in task['probability_distribution']:
                probability = task['probability_distribution'][task['state']]

            task_data.append({
                'id': task['id'],
                'title': task['title'],
                'state': task['state'],
                'entropy': entropy,
                'coherence': coherence,
                'probability': probability,
                'category': task.get('category', 'Uncategorized')
            })

        if task_data:
            task_df = pd.DataFrame(task_data)

            fig = px.scatter_3d(
                task_df,
                x='coherence',
                y='entropy',
                z='probability',
                color='state',
                symbol='category',
                text='title',
                hover_name='title',
                title='Tasks in Quantum State Space',
                labels={
                    'coherence': 'Coherence',
                    'entropy': 'Entropy',
                    'probability': 'Completion Probability'
                }
            )

            fig.update_layout(
                height=700,
                scene=dict(
                    xaxis_title='Coherence',
                    yaxis_title='Entropy',
                    zaxis_title='Probability'
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add explanation
            st.markdown("""
            This 3D visualization shows each task positioned in quantum state space according to its
            coherence, entropy, and probability values. The clustering of tasks reveals quantum patterns:

            - **High Coherence, Low Entropy Region**: Tasks that are well-defined and stable
            - **High Entropy, Low Coherence Region**: Tasks with high uncertainty and ambiguity
            - **High Probability Region**: Tasks most likely to be completed successfully
            """)

            # Show correlation analysis
            st.subheader("Quantum Property Correlations")

            # Calculate correlations
            corr = task_df[['entropy', 'coherence', 'probability']].corr()

            # Display correlation matrix
            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale='blues',
                title='Correlation Matrix of Quantum Properties'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add insight based on correlations
            coherence_entropy_corr = corr.loc['coherence', 'entropy']

            if coherence_entropy_corr < -0.5:
                st.info("""
                **Strong Negative Correlation Detected**

                The strong negative correlation between coherence and entropy indicates your task system
                follows quantum principles well - high coherence tasks have low entropy and vice versa.
                This suggests well-defined tasks with clear requirements.
                """)
            elif coherence_entropy_corr < 0:
                st.info("""
                **Moderate Negative Correlation Detected**

                The moderate relationship between coherence and entropy suggests some tasks may benefit
                from additional refinement to better align with quantum patterns.
                """)
            else:
                st.info("""
                **Unusual Correlation Pattern Detected**

                The positive correlation between coherence and entropy indicates potential issues with
                task definitions that may require attention.
                """)
        else:
            st.info("Not enough task data available for quantum pattern analysis.")

    with insight_tabs[2]:  # Cognitive Load
        st.subheader("Cognitive Load Analysis")

        # Group tasks by assignee
        assignee_tasks = {}

        for task in st.session_state.tasks:
            assignee = task.get('assignee', 'Unassigned')

            if assignee not in assignee_tasks:
                assignee_tasks[assignee] = []

            assignee_tasks[assignee].append(task)

        # Calculate cognitive load for each assignee
        cognitive_loads = {}

        for assignee, tasks in assignee_tasks.items():
            # Calculate total load and weighted load
            total_load = len(tasks)

            # Weighted by entropy and priority
            weighted_load = sum(task.get('priority', 1) * task.get('entropy', 0.5) for task in tasks)

            cognitive_loads[assignee] = {
                'total_tasks': total_load,
                'weighted_load': weighted_load,
                'avg_entropy': sum(task.get('entropy', 0.5) for task in tasks) / max(1, len(tasks)),
                'avg_priority': sum(task.get('priority', 1) for task in tasks) / max(1, len(tasks))
            }

        # Create a DataFrame for visualization
        load_data = []

        for assignee, data in cognitive_loads.items():
            load_data.append({
                'Assignee': assignee,
                'Task Count': data['total_tasks'],
                'Weighted Load': data['weighted_load'],
                'Avg Entropy': data['avg_entropy'],
                'Avg Priority': data['avg_priority']
            })

        if load_data:
            load_df = pd.DataFrame(load_data)

            # Create a scatter plot of cognitive load
            fig = px.scatter(
                load_df,
                x='Task Count',
                y='Weighted Load',
                size='Weighted Load',
                color='Avg Entropy',
                hover_name='Assignee',
                text='Assignee',
                color_continuous_scale=['green', 'yellow', 'red'],
                title='Team Cognitive Load Distribution'
            )

            fig.update_layout(height=500)

            st.plotly_chart(fig, use_container_width=True)

            # Create a bar chart of cognitive load by assignee
            fig = px.bar(
                load_df,
                x='Assignee',
                y='Weighted Load',
                color='Avg Entropy',
                hover_data=['Task Count', 'Avg Priority'],
                color_continuous_scale=['green', 'yellow', 'red'],
                title='Cognitive Load by Assignee'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Identify potential cognitive overload
            overload_threshold = 10.0
            overloaded_assignees = load_df[load_df['Weighted Load'] > overload_threshold]

            if not overloaded_assignees.empty:
                st.warning(f"""
                **Potential Cognitive Overload Detected**

                The following team members may be experiencing cognitive overload:

                {', '.join(overloaded_assignees['Assignee'].tolist())}

                Consider redistributing tasks or providing additional support.
                """)

            # Display detailed breakdown for each assignee
            for assignee, tasks in assignee_tasks.items():
                if assignee != 'Unassigned':  # Skip unassigned tasks for this view
                    with st.expander(f"{assignee} - {len(tasks)} tasks"):
                        # Show tasks sorted by cognitive load (priority * entropy)
                        task_data = []

                        for task in tasks:
                            entropy = task.get('entropy', 0.5)
                            priority = task.get('priority', 1)
                            cognitive_load = priority * entropy

                            task_data.append({
                                'ID': task['id'],
                                'Title': task['title'],
                                'State': task['state'],
                                'Priority': priority,
                                'Entropy': entropy,
                                'Cognitive Load': cognitive_load
                            })

                        if task_data:
                            task_df = pd.DataFrame(task_data)

                            # Sort by cognitive load (descending)
                            task_df = task_df.sort_values('Cognitive Load', ascending=False)

                            st.dataframe(task_df, use_container_width=True)

                            # Add recommendations if load is high
                            assignee_load = cognitive_loads[assignee]['weighted_load']
                            if assignee_load > overload_threshold:
                                highest_load_task = task_df.iloc[0]
                                st.info(f"""
                                **Load Balancing Recommendation:**
                                Consider reassigning "{highest_load_task['Title']}" (Cognitive Load: {highest_load_task['Cognitive Load']:.2f})
                                to reduce {assignee}'s workload.
                                """)
        else:
            st.info("No assignee data available for cognitive load analysis.")

    with insight_tabs[3]:  # Predictions
        st.subheader("ML Predictions")

        # Create a prediction of task completion timeline
        st.markdown("### Task Completion Timeline Prediction")

        # Generate a simple prediction based on current tasks
        pending_tasks = [task for task in st.session_state.tasks if task['state'] in ['PENDING', 'IN_PROGRESS']]

        if pending_tasks:
            # Sort by predicted completion time (use entropy and probability)
            for task in pending_tasks:
                entropy = task.get('entropy', 0.5)

                # Get probability for completion
                probability = 0.5
                if task.get('probability_distribution') and 'COMPLETED' in task['probability_distribution']:
                    probability = task['probability_distribution']['COMPLETED']

                # Calculate expected days to completion
                # Higher entropy and lower probability = longer time
                expected_days = int((entropy * 10) / max(0.1, probability))
                task['expected_days'] = expected_days

            # Sort by expected days
            pending_tasks.sort(key=lambda x: x.get('expected_days', 999))

            # Create a timeline
            timeline_data = []

            today = datetime.now().date()

            for task in pending_tasks:
                completion_date = today + timedelta(days=task.get('expected_days', 7))

                timeline_data.append({
                    'Task': f"#{task['id']} - {task['title']}",
                    'Expected Completion': completion_date,
                    'Days': task.get('expected_days', 7),
                    'State': task['state'],
                    'Priority': task.get('priority', 1)
                })

            timeline_df = pd.DataFrame(timeline_data)

            # Create a Gantt chart
            fig = px.timeline(
                timeline_df,
                x_start=today,
                x_end='Expected Completion',
                y='Task',
                color='State',
                hover_data=['Days', 'Priority'],
                title='Predicted Task Completion Timeline',
                labels={'Task': 'Task', 'Expected Completion': 'Expected Completion Date'}
            )

            fig.update_layout(height=500)

            st.plotly_chart(fig, use_container_width=True)

            # Add prediction insights
            st.markdown("### Completion Predictions")

            # Count tasks by expected completion timeframe
            this_week = sum(1 for task in pending_tasks if task.get('expected_days', 7) <= 7)
            next_week = sum(1 for task in pending_tasks if 7 < task.get('expected_days', 7) <= 14)
            later = sum(1 for task in pending_tasks if task.get('expected_days', 7) > 14)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("This Week", this_week)

            with col2:
                st.metric("Next Week", next_week)

            with col3:
                st.metric("Later", later)

            # Add completion rate prediction
            weekly_rate = this_week / max(1, len(pending_tasks))
            completion_weeks = len(pending_tasks) / max(1, this_week)

            st.markdown(f"""
            ### Project Completion Prediction

            Based on the current task states, entropy, and probability distributions:

            - Expected completion rate: **{weekly_rate:.0%}** of tasks per week
            - Estimated time to complete all current tasks: **{completion_weeks:.1f} weeks**
            - Critical path tasks: Tasks with high entropy and high priority
            """)

            # Show critical path tasks
            critical_tasks = [task for task in pending_tasks
                              if task.get('entropy', 0) > 0.6 and task.get('priority', 0) >= 4]

            if critical_tasks:
                st.markdown("### Critical Path Tasks")

                for task in critical_tasks:
                    st.markdown(f'<div class="ml-insight-card">{task["title"]} (Priority: {task.get("priority", 1)}, Entropy: {task.get("entropy", 0):.2f})</div>', unsafe_allow_html=True)
        else:
            st.info("No pending tasks available for completion prediction.")
