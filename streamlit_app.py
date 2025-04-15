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
            # Create a unique edge identifier
            edge_id = tuple(sorted([task['id'], entangled_id]))
            if edge_id not in added_edges:
                G.add_edge(task['id'], entangled_id, weight=0.7)  # Default weight
                added_edges.add(edge_id)

    return G

def draw_entanglement_network(tasks):
    """Draw the entanglement network using NetworkX and Matplotlib"""
    G = create_entanglement_network(tasks)
    
    if len(G.nodes) == 0:
        st.info("No tasks or entanglements to visualize.")
        return None
    
    # Create color map based on task states
    state_colors = {
        'PENDING': '#DBEAFE',
        'IN_PROGRESS': '#FEF3C7',
        'COMPLETED': '#DCFCE7',
        'BLOCKED': '#FEE2E2'
    }
    
    node_colors = []
    for node in G.nodes:
        state = G.nodes[node].get('state', 'PENDING')
        node_colors.append(state_colors.get(state, '#DBEAFE'))
    
    # Create size map based on task entropy
    node_sizes = []
    for node in G.nodes:
        entropy = G.nodes[node].get('entropy', 0.5)
        node_sizes.append(300 + entropy * 500)  # Scale entropy to node size
    
    # Set up plot
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, edgecolors='white')
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='#3B82F6')
    
    # Draw labels
    labels = {node: G.nodes[node].get('title', '')[:10] + '...' for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')
    
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf()

def generate_ai_task():
    """Generate a task using AI model"""
    st.info("Generating task with Groq LLM...")
    
    topic = st.session_state.get('ai_task_topic', 'General Task')
    context = st.session_state.get('ai_task_context', '')
    
    # Call the AI generation endpoint
    response = api_request("/generate-task", method="POST", data={
        "topic": topic,
        "context": context
    })
    
    if response:
        # Fill the form with generated data
        st.session_state.new_task_title = response.get('title', '')
        st.session_state.new_task_description = response.get('description', '')
        st.session_state.new_task_priority = response.get('priority', 3)
        st.session_state.new_task_tags = ','.join(response.get('tags', []))
        st.success("Task generated successfully! Review and submit the form.")
    else:
        st.error("Failed to generate task. Please try again or create manually.")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/quantum-computing.png", width=80)
    st.markdown("<h1 class='main-header'>Quantum Tasks</h1>", unsafe_allow_html=True)
    
    # Navigation
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Tasks", "Entanglements", "Simulations", "Optimization", "Settings"],
        icons=["house", "list-task", "link", "lightning-charge", "diagram-3", "gear"],
        default_index=0,
    )
    
    # System status
    if st.session_state.system_metrics:
        st.markdown("<p class='sidebar-header'>System Metrics</p>", unsafe_allow_html=True)
        
        metrics = st.session_state.system_metrics
        
        # Display as progress bars
        st.markdown(f"**Tasks:** {metrics['task_count']}")
        
        st.markdown("**System Entropy:**")
        entropy = metrics['total_entropy']
        st.markdown(f"""
        <div class='quantum-meter'>
            <div class='quantum-meter-fill' style='width: {entropy * 100}%'></div>
        </div>
        <p class='small-text'>{entropy:.2f}</p>
        """, unsafe_allow_html=True)
        
        st.markdown("**Quantum Coherence:**")
        coherence = metrics['quantum_coherence']
        st.markdown(f"""
        <div class='quantum-meter'>
            <div class='quantum-meter-fill' style='width: {coherence * 100}%'></div>
        </div>
        <p class='small-text'>{coherence:.2f}</p>
        """, unsafe_allow_html=True)
        
        st.markdown("**Entanglement Density:**")
        density = metrics['entanglement_density']
        st.markdown(f"""
        <div class='quantum-meter'>
            <div class='quantum-meter-fill' style='width: {density * 100}%'></div>
        </div>
        <p class='small-text'>{density:.2f}</p>
        """, unsafe_allow_html=True)
    
    # Fetch tasks on sidebar load (do this once)
    if 'sidebar_loaded' not in st.session_state:
        st.session_state.sidebar_loaded = True
        fetch_tasks()
        fetch_metrics()

# Main content area based on navigation
if selected == "Dashboard":
    st.markdown("<h1 class='main-header'>Quantum Task Dashboard</h1>", unsafe_allow_html=True)
    
    # Quick stats in columns
    col1, col2, col3, col4 = st.columns(4)
    
    tasks = st.session_state.tasks
    metrics = st.session_state.system_metrics or {}
    
    # Calculate stats if tasks exist
    task_count = len(tasks)
    pending_count = sum(1 for task in tasks if task.get('state') == 'PENDING')
    completed_count = sum(1 for task in tasks if task.get('state') == 'COMPLETED')
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<p class='card-title'>Total Tasks</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='card-value'>{task_count}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<p class='card-title'>Pending Tasks</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='card-value'>{pending_count}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<p class='card-title'>Completed Tasks</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='card-value'>{completed_count}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<p class='card-title'>Quantum Coherence</p>", unsafe_allow_html=True)
        coherence = metrics.get('quantum_coherence', 0)
        st.markdown(f"<p class='card-value'>{coherence:.2f}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Two columns for main dashboard content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Task Entanglement Network")
        if tasks:
            network_fig = draw_entanglement_network(tasks)
            if network_fig:
                st.pyplot(network_fig)
            
        else:
            st.info("No tasks available. Create some tasks to visualize the entanglement network.")
            
    with col2:
        st.subheader("Task Status Distribution")
        if tasks:
            # Count tasks by state
            states = ['PENDING', 'IN_PROGRESS', 'COMPLETED', 'BLOCKED']
            state_counts = {state: sum(1 for task in tasks if task.get('state') == state) for state in states}
            
            # Create DataFrame
            df = pd.DataFrame({
                'Status': list(state_counts.keys()),
                'Count': list(state_counts.values())
            })
            
            # Create chart
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('Status', sort=states),
                y='Count',
                color=alt.Color('Status', scale=alt.Scale(
                    domain=states,
                    range=['#DBEAFE', '#FEF3C7', '#DCFCE7', '#FEE2E2']
                ))
            ).properties(height=250)
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No tasks available.")
        
        # High entropy tasks
        st.subheader("High Entropy Tasks")
        if tasks:
            # Sort tasks by entropy
            sorted_tasks = sorted(tasks, key=lambda x: x.get('entropy', 0), reverse=True)
            high_entropy_tasks = sorted_tasks[:3]
            
            for task in high_entropy_tasks:
                st.markdown(f"""
                <div class='task-card'>
                    <p class='task-title'>{task.get('title')}</p>
                    <span class='task-status status-{task.get('state')}'>{task.get('state')}</span>
                    <span class='small-text'>Entropy: {task.get('entropy', 0):.2f}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No tasks available.")

    # Recently updated tasks
    st.subheader("Recently Updated Tasks")
    if tasks:
        # Sort tasks by updated_at
        recent_tasks = sorted(tasks, key=lambda x: x.get('updated_at', ''), reverse=True)[:5]
        
        # Display in a table
        task_data = []
        for task in recent_tasks:
            # Parse updated_at to datetime
            updated_at = datetime.fromisoformat(task.get('updated_at').replace('Z', '+00:00'))
            task_data.append({
                'ID': task.get('id')[:8],
                'Title': task.get('title'),
                'Status': task.get('state'),
                'Updated': updated_at.strftime('%Y-%m-%d %H:%M'),
                'Assignee': task.get('assignee') or 'Unassigned'
            })
        
        df = pd.DataFrame(task_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No tasks available.")

elif selected == "Tasks":
    st.markdown("<h1 class='main-header'>Quantum Task Management</h1>", unsafe_allow_html=True)
    
    # Tabs for task views
    tab1, tab2, tab3 = st.tabs(["All Tasks", "Create Task", "Search Tasks"])
    
    with tab1:
        # Filter controls
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            filter_option = st.selectbox(
                "Filter by Status",
                ["All", "Pending", "In Progress", "Completed", "Blocked"],
                key="task_filter_status"
            )
        
        with col2:
            sort_option = st.selectbox(
                "Sort by",
                ["Updated (Newest)", "Updated (Oldest)", "Priority (High to Low)", "Entropy (High to Low)"],
                key="task_sort"
            )
        
        with col3:
            refresh_button = st.button("Refresh", key="refresh_tasks")
            if refresh_button:
                fetch_tasks()
        
        # Tasks list
        tasks = st.session_state.tasks
        if tasks:
            # Apply filters
            if filter_option != "All":
                filter_state = filter_option.upper().replace(" ", "_")
                tasks = [task for task in tasks if task.get('state') == filter_state]
            
            # Apply sorting
            if sort_option == "Updated (Newest)":
                tasks = sorted(tasks, key=lambda x: x.get('updated_at', ''), reverse=True)
            elif sort_option == "Updated (Oldest)":
                tasks = sorted(tasks, key=lambda x: x.get('updated_at', ''))
            elif sort_option == "Priority (High to Low)":
                tasks = sorted(tasks, key=lambda x: x.get('priority', 0), reverse=True)
            elif sort_option == "Entropy (High to Low)":
                tasks = sorted(tasks, key=lambda x: x.get('entropy', 0), reverse=True)
            
            # Display tasks
            for task in tasks:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class='task-card'>
                        <p class='task-title'>{task.get('title')}
                            <span class='task-status status-{task.get('state')}'>{task.get('state')}</span>
                            <span class='small-text'>Entropy: {task.get('entropy', 0):.2f}</span>
                        </p>
                        <p>{task.get('description')[:100]}{'...' if len(task.get('description', '')) > 100 else ''}</p>
                        <p class='small-text'>
                    """, unsafe_allow_html=True)
                    
                    # Display tags
                    for tag in task.get('tags', []):
                        st.markdown(f"<span class='task-tag'>{tag}</span>", unsafe_allow_html=True)
                    
                    st.markdown("</p></div>", unsafe_allow_html=True)
                
                with col2:
                    # Task actions
                    view_button = st.button("View", key=f"view_{task.get('id')}")
                    if view_button:
                        st.session_state.selected_task = task
                        st.experimental_rerun()
                    
                    edit_button = st.button("Edit", key=f"edit_{task.get('id')}")
                    if edit_button:
                        st.session_state.edit_task = task
                        st.experimental_rerun()
                    
                    delete_button = st.button("Delete", key=f"delete_{task.get('id')}")
                    if delete_button:
                        if delete_task(task.get('id')):
                            st.success(f"Deleted task: {task.get('title')}")
                            fetch_tasks()
                            st.experimental_rerun()
            
            # Task detail view
            if st.session_state.selected_task:
                task = st.session_state.selected_task
                
                with st.expander("Task Details", expanded=True):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.subheader(task.get('title'))
                        st.markdown(f"<span class='task-status status-{task.get('state')}'>{task.get('state')}</span>", unsafe_allow_html=True)
                        st.write(task.get('description'))
                        
                        st.markdown("**Tags:**")
                        for tag in task.get('tags', []):
                            st.markdown(f"<span class='task-tag'>{tag}</span>", unsafe_allow_html=True)
                        
                        if task.get('assignee'):
                            st.markdown(f"**Assignee:** {task.get('assignee')}")
                        
                        if task.get('due_date'):
                            st.markdown(f"**Due Date:** {task.get('due_date')}")
                        
                        st.markdown(f"**Priority:** {task.get('priority')}/5")
                        
                        # ML Insights
                        if task.get('ml_summary') or task.get('category'):
                            st.markdown("**AI Insights:**")
                            st.markdown(f"""
                            <div class='ml-insight-card'>
                                {task.get('ml_summary', '')}
                                <br><span class='small-text'>Category: {task.get('category', 'Unknown')}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**Quantum State:**")
                        st.markdown(f"Entropy: {task.get('entropy', 0):.2f}")
                        
                        # Display probability distribution
                        prob_dist = task.get('probability_distribution', {})
                        if prob_dist:
                            data = []
                            for state, prob in prob_dist.items():
                                data.append({"State": state, "Probability": prob})
                                
                            df = pd.DataFrame(data)
                            fig = px.bar(df, x="State", y="Probability", text="Probability")
                            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display entangled tasks
                        entangled_ids = task.get('entangled_tasks', [])
                        if entangled_ids:
                            st.markdown("**Entangled With:**")
                            for entangled_id in entangled_ids:
                                # Find the task
                                for t in st.session_state.tasks:
                                    if t.get('id') == entangled_id:
                                        st.markdown(f"- {t.get('title')} ({t.get('state')})")
                                        break
                    
                    # Close button
                    if st.button("Close Task Details"):
                        st.session_state.selected_task = None
                        st.experimental_rerun()
        else:
            st.info("No tasks available. Create a new task to get started.")
    
    with tab2:
        st.subheader("Create New Task")
        
        # AI Task Generation
        ai_generate_expander = st.expander("Generate with AI", expanded=False)
        with ai_generate_expander:
            st.markdown("Let the LLM generate a task for you based on a topic and context.")
            
            st.text_input("Topic", key="ai_task_topic", placeholder="E.g., Frontend, Backend, Documentation")
            st.text_area("Context", key="ai_task_context", placeholder="Additional context for the task (optional)")
            
            if st.button("Generate Task"):
                generate_ai_task()
        
        # Create task form
        with st.form("create_task_form"):
            title = st.text_input("Title", key="new_task_title")
            description = st.text_area("Description", key="new_task_description")
            
            col1, col2 = st.columns(2)
            with col1:
                assignee = st.text_input("Assignee (optional)", key="new_task_assignee")
                priority = st.slider("Priority", 1, 5, 3, key="new_task_priority")
            
            with col2:
                tags_input = st.text_input("Tags (comma-separated)", key="new_task_tags")
                due_date = st.date_input("Due Date (optional)", value=None, key="new_task_due_date")
            
            submit_button = st.form_submit_button("Create Task")
            
            if submit_button:
                # Validate
                if not title:
                    st.error("Title is required")
                elif not description:
                    st.error("Description is required")
                else:
                    # Process tags
                    tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
                    
                    # Create task data
                    task_data = {
                        "title": title,
                        "description": description,
                        "assignee": assignee if assignee else None,
                        "priority": priority,
                        "tags": tags
                    }
                    
                    # Add due date if provided
                    if due_date:
                        task_data["due_date"] = datetime.combine(due_date, datetime.min.time()).isoformat()
                    
                    # Create task
                    if create_task(task_data):
                        # Clear form
                        st.session_state.new_task_title = ""
                        st.session_state.new_task_description = ""
                        st.session_state.new_task_assignee = ""
                        st.session_state.new_task_tags = ""
                        st.session_state.new_task_priority = 3
                        st.session_state.new_task_due_date = None
                        
                        st.success("Task created successfully!")
    
    with tab3:
        st.subheader("Search Tasks")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input("Search Query", key="search_query")
        
        with col2:
            use_quantum = st.checkbox("Use Quantum Search", value=False, key="use_quantum_search")
        
        if st.button("Search", key="search_button"):
            if search_query:
                with st.spinner("Searching..."):
                    search_results = search_tasks(search_query, use_quantum)
                    
                if search_results:
                    st.success(f"Found {len(search_results)} results")
                    
                    for task in search_results:
                        st.markdown(f"""
                        <div class='task-card'>
                            <p class='task-title'>{task.get('title')}
                                <span class='task-status status-{task.get('state')}'>{task.get('state')}</span>
                            </p>
                            <p>{task.get('description')[:100]}{'...' if len(task.get('description', '')) > 100 else ''}</p>
                            <p class='small-text'>
                        """, unsafe_allow_html=True)
                        
                        # Display tags
                        for tag in task.get('tags', []):
                            st.markdown(f"<span class='task-tag'>{tag}</span>", unsafe_allow_html=True)
                        
                        st.markdown("</p></div>", unsafe_allow_html=True)
                else:
                    st.info("No matching tasks found.")
            else:
                st.warning("Please enter a search query.")

elif selected == "Entanglements":
    st.markdown("<h1 class='main-header'>Quantum Entanglements</h1>", unsafe_allow_html=True)
    
    # Tabs for entanglement views
    tab1, tab2, tab3 = st.tabs(["Visualization", "Create Entanglement", "Entanglement List"])
    
    with tab1:
        st.subheader("Task Entanglement Network")
        
        tasks = st.session_state.tasks
        if tasks:
            network_fig = draw_entanglement_network(tasks)
            if network_fig:
                st.pyplot(network_fig)
                
                # Add explanation
                st.markdown("""
                **Network Legend:**
                - **Node Size:** Represents task entropy (larger = higher entropy)
                - **Node Color:** Represents task state (blue = pending, yellow = in progress, green = completed, red = blocked)
                - **Edge:** Represents quantum entanglement between tasks
                
                Tasks that are entangled influence each other's quantum states. When one task changes state, it affects the probability distribution of its entangled partners.
                """)
            else:
                st.info("No entanglements to visualize. Create some entanglements first.")
        else:
            st.info("No tasks available. Create some tasks to visualize the entanglement network.")
    
    with tab2:
        st.subheader("Create New Entanglement")
        
        tasks = st.session_state.tasks
        if len(tasks) >= 2:
            with st.form("create_entanglement_form"):
                # Get task options for select boxes
                task_options = {task.get('id'): f"{task.get('title')} ({task.get('state')})" for task in tasks}
                
                # Convert to list of tuples for selectbox
                task_options_list = list(task_options.items())
                
                # Select tasks
                col1, col2 = st.columns(2)
                
                with col1:
                    task_id_1 = st.selectbox(
                        "Task 1",
                        options=[item[0] for item in task_options_list],
                        format_func=lambda x: task_options.get(x, x),
                        key="entanglement_task_1"
                    )
                
                with col2:
                    # Filter out the first task
                    remaining_options = [item for item in task_options_list if item[0] != task_id_1]
                    task_id_2 = st.selectbox(
                        "Task 2",
                        options=[item[0] for item in remaining_options],
                        format_func=lambda x: task_options.get(x, x),
                        key="entanglement_task_2"
                    )
                
                # Entanglement properties
                st.markdown("### Entanglement Properties")
                entanglement_strength = st.slider("Entanglement Strength", 0.1, 1.0, 0.7, 0.1, key="entanglement_strength")
                
                entanglement_type = st.selectbox(
                    "Entanglement Type",
                    options=["standard", "SWAP", "CNOT"],
                    key="entanglement_type"
                )
                
                st.markdown("""
                **Entanglement Types:**
                - **Standard:** General entanglement with bi-directional effects
                - **SWAP:** Swap-like entanglement that exchanges quantum properties
                - **CNOT:** Controlled-NOT like behavior where one task controls the other
                """)
                
                submit_button = st.form_submit_button("Create Entanglement")
                
                if submit_button:
                    # Create entanglement data
                    entanglement_data = {
                        "task_id_1": task_id_1,
                        "task_id_2": task_id_2,
                        "strength": entanglement_strength,
                        "entanglement_type": entanglement_type
                    }
                    
                    # Create entanglement
                    if create_entanglement(entanglement_data):
                        st.success("Entanglement created successfully!")
        else:
            st.warning("You need at least 2 tasks to create an entanglement.")
    
    with tab3:
        # Fetch entanglements
        entanglements_data = api_request("/entanglements")
        
        if entanglements_data:
            st.subheader(f"All Entanglements ({len(entanglements_data)})")
            
            # Create a lookup dict for task titles
            task_dict = {task.get('id'): task.get('title') for task in st.session_state.tasks}
            
            for entanglement in entanglements_data:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    task1_title = task_dict.get(entanglement.get('task_id_1'), 'Unknown Task')
                    task2_title = task_dict.get(entanglement.get('task_id_2'), 'Unknown Task')
                    
                    st.markdown(f"""
                    <div class='task-card'>
                        <p class='task-title'>{task1_title} ⟷ {task2_title}</p>
                        <p>Type: {entanglement.get('entanglement_type')}</p>
                        <p>Strength: {entanglement.get('strength')}</p>
                        <p class='small-text'>Created: {entanglement.get('created_at')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button("Delete", key=f"delete_entanglement_{entanglement.get('id')}"):
                        response = api_request(f"/entanglements/{entanglement.get('id')}", method="DELETE")
                        if response:
                            st.success("Entanglement deleted!")
                            fetch_tasks()
                            st.experimental_rerun()
        else:
            st.info("No entanglements available. Create an entanglement to get started.")

elif selected == "Simulations":
    st.markdown("<h1 class='main-header'>Quantum Simulations</h1>", unsafe_allow_html=True)
    
    tasks = st.session_state.tasks
    
    if tasks:
        # Select tasks for simulation
        st.subheader("Run Quantum Simulation")
        
        with st.form("simulation_form"):
            # Multi-select for tasks
            task_options = {task.get('id'): f"{task.get('title')} ({task.get('state')})" for task in tasks}
            selected_task_ids = st.multiselect(
                "Select Tasks to Simulate",
                options=list(task_options.keys()),
                format_func=lambda x: task_options.get(x, x),
                key="simulation_tasks"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                simulation_steps = st.slider("Simulation Steps", 2, 10, 5, key="simulation_steps")
            
            with col2:
                decoherence_rate = st.slider("Decoherence Rate", 0.0, 0.2, 0.05, 0.01, key="decoherence_rate")
            
            submit_button = st.form_submit_button("Run Simulation")
            
            if submit_button:
                if selected_task_ids:
                    with st.spinner("Running quantum simulation..."):
                        # Call the simulation endpoint
                        simulation_data = {
                            "task_ids": selected_task_ids,
                            "simulation_steps": simulation_steps,
                            "decoherence_rate": decoherence_rate,
                            "measurement_type": "projective"
                        }
                        
                        simulation_results = api_request("/quantum-simulation", method="POST", data=simulation_data)
                        
                        if simulation_results:
                            st.session_state.simulation_results = simulation_results
                            st.success("Simulation completed successfully!")
                else:
                    st.warning("Please select at least one task to simulate.")
        
        # Display simulation results
        if st.session_state.simulation_results:
            simulation = st.session_state.simulation_results
            
            st.markdown("### Simulation Results")
            
            # Display tasks included in simulation
            task_data = simulation.get('tasks', [])
            if task_data:
                st.markdown("**Tasks in Simulation:**")
                for task in task_data:
                    st.markdown(f"- {task.get('title')} (State: {task.get('state')}, Entropy: {task.get('entropy', 0):.2f})")
            
            # Entanglement Matrix
            entanglement_matrix = simulation.get('entanglement_matrix', [])
            if entanglement_matrix:
                st.markdown("**Entanglement Matrix:**")
                
                # Convert to DataFrame
                df = pd.DataFrame(entanglement_matrix)
                
                # Use task titles as column names if available
                if task_data:
                    task_titles = [task.get('title', f"Task {i+1}")[:10] for i, task in enumerate(task_data)]
                    df.columns = task_titles
                    df.index = task_titles
                
                st.dataframe(df, use_container_width=True)
            
            # Simulation Steps
            steps_data = simulation.get('simulation_steps', [])
            if steps_data:
                st.markdown("**Simulation Evolution:**")
                
                # Create tabs for each step
                step_tabs = st.tabs([f"Step {i+1}" for i in range(len(steps_data))])
                
                for i, (tab, step) in enumerate(zip(step_tabs, steps_data)):
                    with tab:
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            # For each task, show probability distribution changes
                            for task_id, task_step_data in step.items():
                                # Find task title
                                task_title = next((task.get('title') for task in task_data if task.get('id') == task_id), f"Task {task_id}")
                                
                                st.markdown(f"**{task_title}**")
                                st.markdown(f"Entropy: {task_step_data.get('entropy', 0):.2f}")
                                
                                # Probability distribution
                                prob_dist = task_step_data.get('probability_distribution', {})
                                if prob_dist:
                                    data = []
                                    for state, prob in prob_dist.items():
                                        data.append({"State": state, "Probability": prob})
                                        
                                    df = pd.DataFrame(data)
                                    fig = px.bar(df, x="State", y="Probability", text="Probability")
                                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                                    fig.update_layout(height=250)
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Display quantum state visualization for one task
                            if task_data:
                                selected_task_id = task_data[0].get('id')
                                if selected_task_id in step:
                                    quantum_state = step[selected_task_id].get('quantum_state', {})
                                    if quantum_state and 'visualization_data' in quantum_state:
                                        # Create a quantum state visualization
                                        vis_data = quantum_state.get('visualization_data', [])
                                        
                                        if vis_data:
                                            # Create a simple Bloch sphere-like visualization
                                            if len(vis_data) >= 3:
                                                # Use the first three probabilities to create a 3D vector
                                                vec = np.array([
                                                    np.sqrt(vis_data[0]) if len(vis_data) > 0 else 0,
                                                    np.sqrt(vis_data[1]) if len(vis_data) > 1 else 0,
                                                    np.sqrt(vis_data[2]) if len(vis_data) > 2 else 0
                                                ])
                                                
                                                # Normalize
                                                if np.linalg.norm(vec) > 0:
                                                    vec = vec / np.linalg.norm(vec)
                                                
                                                bloch_fig = generate_bloch_sphere(vec)
                                                st.plotly_chart(bloch_fig, use_container_width=True)
                                            else:
                                                st.info("Insufficient data for Bloch sphere visualization")
                                    else:
                                        st.info("No quantum state visualization data available")
    else:
        st.info("No tasks available. Create some tasks to run simulations.")

elif selected == "Optimization":
    st.markdown("<h1 class='main-header'>Quantum Optimization</h1>", unsafe_allow_html=True)
    
    # Task Assignment Optimization
    st.subheader("Task Assignment Optimization")
    
    st.markdown("""
    This feature uses quantum-inspired optimization algorithms to find the optimal assignment
    of tasks to team members, considering factors like:
    
    - Task priority
    - Task entropy (uncertainty)
    - Due dates
    - Team member workload
    - Task dependencies and entanglements
    """)
    
    if st.button("Run Optimization Algorithm"):
        with st.spinner("Running quantum-inspired optimization..."):
            optimization_result = optimize_assignments()
            
            if optimization_result:
                st.session_state.optimization_result = optimization_result
                st.success("Optimization completed successfully!")
    
    # Display optimization results
    if st.session_state.optimization_result:
        result = st.session_state.optimization_result
        
        st.markdown("### Optimization Results")
        st.markdown(f"**Optimization Score:** {result.get('optimization_score', 0):.3f}")
        st.markdown(f"**Tasks Assigned:** {result.get('task_count', 0)}")
        
        # Assignments
        assignments = result.get('assignments', {})
        if assignments:
            st.markdown("**Recommended Task Assignments:**")
            
            # Get task details
            task_dict = {task.get('id'): task for task in st.session_state.tasks}
            
            # Group tasks by assignee
            assignee_tasks = {}
            for task_id, assignee in assignments.items():
                if assignee not in assignee_tasks:
                    assignee_tasks[assignee] = []
                
                # Get task details
                task = task_dict.get(task_id, {'id': task_id, 'title': 'Unknown Task', 'priority': 0})
                assignee_tasks[assignee].append(task)
            
            # Display grouped by assignee
            for assignee, tasks in assignee_tasks.items():
                with st.expander(f"{assignee} ({len(tasks)} tasks)", expanded=True):
                    for task in tasks:
                        st.markdown(f"""
                        <div class='task-card'>
                            <p class='task-title'>{task.get('title')}
                                <span class='task-status status-{task.get('state', 'PENDING')}'>{task.get('state', 'PENDING')}</span>
                            </p>
                            <p>Priority: {task.get('priority', 0)}/5</p>
                            <p class='small-text'>Entropy: {task.get('entropy', 0):.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add apply button
                    if st.button(f"Apply Assignments for {assignee}", key=f"apply_{assignee}"):
                        # Create tasks to update
                        for task in tasks:
                            task_id = task.get('id')
                            update_data = {"assignee": assignee}
                            update_task(task_id, update_data)
                        
                        st.success(f"Assignments for {assignee} have been applied!")
                        fetch_tasks()
        else:
            st.info("No assignments recommended. Try adding more tasks or team members.")

elif selected == "Settings":
    st.markdown("<h1 class='main-header'>System Settings</h1>", unsafe_allow_html=True)
    
    st.write("Configure the quantum task management system.")
    
    # API Configuration
    st.subheader("API Configuration")
    
    api_url = st.text_input("API URL", value=API_URL)
    if st.button("Save API URL"):
        API_URL = api_url
        st.success(f"API URL updated to {API_URL}")
    
    # Groq API Key
    st.subheader("Groq LLM Configuration")
    
    groq_api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key to enable advanced LLM features")
    if st.button("Save Groq API Key"):
        st.success("Groq API key saved!")
    
    # System Info
    st.subheader("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Tasks Count", len(st.session_state.tasks))
        
        # Check API connection
        if st.button("Test API Connection"):
            response = api_request("/")
            if response:
                st.success("API connection successful!")
                st.json(response)
            else:
                st.error("API connection failed. Check URL and server status.")
    
    with col2:
        metrics = st.session_state.system_metrics or {}
        st.metric("System Entropy", f"{metrics.get('total_entropy', 0):.2f}")
        st.metric("Quantum Coherence", f"{metrics.get('quantum_coherence', 0):.2f}")

# Fetch data regularly
if st.button("Refresh Data", key="global_refresh"):
    fetch_tasks()
    fetch_metrics()
    st.success("Data refreshed!")

# Footer
st.markdown("""
<div style="text-align:center; margin-top:30px; padding:10px; font-size:0.8rem; color:#64748B;">
    Neuromorphic Quantum Cognitive Task System 🧠✨💻
</div>
""", unsafe_allow_html=True)
