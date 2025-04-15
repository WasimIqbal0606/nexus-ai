import streamlit as st
import streamlit.components.v1 as components

def load_css():
    """Load custom CSS for styling."""
    st.markdown("""
    <style>
    /* Main app styling with modern look */
    .main {
        background: linear-gradient(135deg, #0a192f 0%, #112240 100%);
        color: #e6f1ff;
    }
    
    /* Header styling */
    .main h1, .main h2, .main h3 {
        background: linear-gradient(120deg, #64ffda, #00bfa5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: 1px;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from {
            text-shadow: 0 0 5px rgba(100, 255, 218, 0.3);
        }
        to {
            text-shadow: 0 0 20px rgba(100, 255, 218, 0.7);
        }
    }
    
    /* Task card styling */
    .task-card {
        background: rgba(16, 33, 65, 0.8);
        border-radius: 12px;
        border-left: 4px solid #64ffda;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .task-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
    }
    
    /* Priority indicators */
    .priority-high {
        color: #ff5252;
        font-weight: bold;
    }
    
    .priority-medium {
        color: #ffab40;
    }
    
    .priority-low {
        color: #69f0ae;
    }
    
    /* Status badges */
    .status-badge {
        padding: 4px 10px;
        border-radius: 50px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-pending {
        background: rgba(255, 171, 64, 0.15);
        color: #ffab40;
        border: 1px solid #ffab40;
    }
    
    .status-in-progress {
        background: rgba(33, 150, 243, 0.15);
        color: #29b6f6;
        border: 1px solid #29b6f6;
    }
    
    .status-completed {
        background: rgba(105, 240, 174, 0.15);
        color: #69f0ae;
        border: 1px solid #69f0ae;
    }
    
    .status-blocked {
        background: rgba(255, 82, 82, 0.15);
        color: #ff5252;
        border: 1px solid #ff5252;
    }
    
    /* Quantum visualization styling */
    .quantum-visualization {
        background: rgba(16, 33, 65, 0.5);
        border-radius: 8px;
        padding: 12px;
        border: 1px solid rgba(100, 255, 218, 0.3);
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        color: white;
        font-weight: bold;
        background: linear-gradient(90deg, #0072ff, #00c6ff);
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 114, 255, 0.4);
    }
    
    /* Form field styling */
    input, textarea {
        background-color: rgba(16, 33, 65, 0.6) !important;
        color: #e6f1ff !important;
        border: 1px solid rgba(100, 255, 218, 0.3) !important;
        border-radius: 8px !important;
    }
    
    input:focus, textarea:focus {
        border: 1px solid #64ffda !important;
        box-shadow: 0 0 0 2px rgba(100, 255, 218, 0.3) !important;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0a192f 0%, #091429 100%);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #64ffda;
    }
    </style>
    """, unsafe_allow_html=True)

def animated_title():
    """Display an animated title using JavaScript animation."""
    components.html(
        """
        <script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
        <style>
        #particles-js {
            position: relative;
            width: 100%;
            height: 120px;
            background-color: transparent;
            margin-bottom: 20px;
        }
        .title-container {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 100%;
            text-align: center;
        }
        .app-title {
            font-family: 'Arial', sans-serif;
            font-size: 44px;
            font-weight: 800;
            background: linear-gradient(90deg, #64ffda, #0072ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0px 2px 4px rgba(0,0,0,0.3);
            animation: fadeInUp 1.5s ease-out;
        }
        .app-subtitle {
            font-family: 'Arial', sans-serif;
            font-size: 18px;
            color: #ccd6f6;
            margin-top: 10px;
            animation: fadeInUp 1.5s ease-out 0.3s both;
            letter-spacing: 1px;
        }
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        </style>
        
        <div id="particles-js">
            <div class="title-container">
                <div class="app-title">Quantum Nexus</div>
                <div class="app-subtitle">Advanced Task Management System</div>
            </div>
        </div>
        
        <script>
        particlesJS('particles-js', {
            "particles": {
                "number": {
                    "value": 80,
                    "density": {
                        "enable": true,
                        "value_area": 800
                    }
                },
                "color": {
                    "value": "#64ffda"
                },
                "shape": {
                    "type": "circle",
                    "stroke": {
                        "width": 0,
                        "color": "#000000"
                    },
                },
                "opacity": {
                    "value": 0.5,
                    "random": true,
                },
                "size": {
                    "value": 3,
                    "random": true,
                },
                "line_linked": {
                    "enable": true,
                    "distance": 150,
                    "color": "#29b6f6",
                    "opacity": 0.4,
                    "width": 1
                },
                "move": {
                    "enable": true,
                    "speed": 2,
                    "direction": "none",
                    "random": false,
                    "straight": false,
                    "out_mode": "out",
                    "bounce": false,
                }
            },
            "interactivity": {
                "detect_on": "canvas",
                "events": {
                    "onhover": {
                        "enable": true,
                        "mode": "grab"
                    },
                    "onclick": {
                        "enable": true,
                        "mode": "push"
                    },
                    "resize": true
                },
                "modes": {
                    "grab": {
                        "distance": 140,
                        "line_linked": {
                            "opacity": 1
                        }
                    },
                    "push": {
                        "particles_nb": 4
                    }
                }
            },
            "retina_detect": true
        });
        </script>
        """,
        height=150,
    )

def animated_task_card(task):
    """Render a task card with animations."""
    # Get status and priority classes
    status_class = f"status-{task['state'].lower()}"
    
    priority_class = "priority-low"
    if task.get('priority', 1) >= 4:
        priority_class = "priority-high"
    elif task.get('priority', 1) >= 2:
        priority_class = "priority-medium"
    
    # Calculate probability values for visualization
    prob_values = task.get('probability_distribution', {})
    prob_pending = prob_values.get('PENDING', 0) * 100
    prob_in_progress = prob_values.get('IN_PROGRESS', 0) * 100
    prob_completed = prob_values.get('COMPLETED', 0) * 100
    prob_blocked = prob_values.get('BLOCKED', 0) * 100
    
    entropy = task.get('entropy', 0.5)
    
    # Format dates
    created_at = task.get('created_at')
    due_date = task.get('due_date')
    
    created_at_str = ""
    due_date_str = ""
    
    if created_at:
        try:
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            created_at_str = created_at.strftime("%Y-%m-%d")
        except:
            created_at_str = str(created_at)
    
    if due_date:
        try:
            if isinstance(due_date, str):
                due_date = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
            due_date_str = due_date.strftime("%Y-%m-%d")
        except:
            due_date_str = str(due_date)
    
    html = f"""
    <div class="task-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px;">
            <div>
                <h3 style="margin: 0; font-size: 18px;">{task['title']}</h3>
                <div style="display: flex; gap: 10px; margin-top: 6px;">
                    <span class="status-badge {status_class}">{task['state']}</span>
                    <span class="{priority_class}">Priority: {task.get('priority', 1)}</span>
                </div>
            </div>
            <div style="text-align: right;">
                {f'<div style="font-size: 12px; color: #a8b2d1; margin-bottom: 5px;">Due: {due_date_str}</div>' if due_date_str else ''}
                <div style="font-size: 12px; color: #a8b2d1;">Created: {created_at_str}</div>
            </div>
        </div>
        
        <p style="color: #a8b2d1; margin-bottom: 12px;">{task['description']}</p>
        
        <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px;">
            {' '.join([f'<span style="background: rgba(100, 255, 218, 0.15); color: #64ffda; padding: 3px 8px; border-radius: 50px; font-size: 11px;">{tag}</span>' for tag in task.get('tags', [])])}
        </div>
        
        <div class="quantum-visualization">
            <div style="font-size: 12px; margin-bottom: 8px; display: flex; justify-content: space-between;">
                <span>Quantum State Probabilities</span>
                <span>Entropy: {entropy:.2f}</span>
            </div>
            <div style="display: flex; width: 100%; height: 24px; border-radius: 4px; overflow: hidden;">
                <div style="width: {prob_pending}%; background-color: #ffab40; height: 100%;" title="PENDING: {prob_pending:.1f}%"></div>
                <div style="width: {prob_in_progress}%; background-color: #29b6f6; height: 100%;" title="IN_PROGRESS: {prob_in_progress:.1f}%"></div>
                <div style="width: {prob_completed}%; background-color: #69f0ae; height: 100%;" title="COMPLETED: {prob_completed:.1f}%"></div>
                <div style="width: {prob_blocked}%; background-color: #ff5252; height: 100%;" title="BLOCKED: {prob_blocked:.1f}%"></div>
            </div>
        </div>
    </div>
    """
    
    return html

def task_list_animation(tasks):
    """Display animated task list with fade-in and staggered animations."""
    if not tasks:
        st.info("No tasks found. Create a new task to get started.")
        return
    
    html_tasks = []
    for i, task in enumerate(tasks):
        html_tasks.append(animated_task_card(task))
    
    components.html(
        f"""
        <style>
        .tasks-container {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        .task-card {{
            opacity: 0;
            transform: translateY(20px);
            animation: fadeInUp 0.5s ease-out forwards;
            animation-delay: calc(0.1s * {i} + 0.1s);
        }}
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        </style>
        
        <div class="tasks-container">
            {''.join(html_tasks)}
        </div>
        """,
        height=300 + (len(tasks) * 220),
    )

def animated_entanglement_network(vis_data):
    """Create a more visually appealing entanglement network visualization."""
    components.html(
        f"""
        <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/vis-network.min.js"></script>
        <style>
        #mynetwork {{
            width: 100%;
            height: 500px;
            border: 1px solid rgba(100, 255, 218, 0.3);
            border-radius: 12px;
            background-color: rgba(16, 33, 65, 0.3);
            overflow: hidden;
        }}
        </style>
        
        <div id="mynetwork"></div>
        
        <script>
            // Parse the data
            var nodes = {vis_data['nodes']};
            var edges = {vis_data['edges']};
            
            // Create a network
            var container = document.getElementById('mynetwork');
            
            // Customize the nodes
            for (var i = 0; i < nodes.length; i++) {{
                // Set colors based on state
                var color, fontColor;
                
                switch(nodes[i].state) {{
                    case 'PENDING':
                        color = 'rgba(255, 171, 64, 0.8)';
                        fontColor = '#ffffff';
                        break;
                    case 'IN_PROGRESS':
                        color = 'rgba(33, 150, 243, 0.8)';
                        fontColor = '#ffffff';
                        break;
                    case 'COMPLETED':
                        color = 'rgba(105, 240, 174, 0.8)';
                        fontColor = '#ffffff';
                        break;
                    case 'BLOCKED':
                        color = 'rgba(255, 82, 82, 0.8)';
                        fontColor = '#ffffff';
                        break;
                    default:
                        color = 'rgba(100, 255, 218, 0.8)';
                        fontColor = '#ffffff';
                }}
                
                // Add custom node styling
                nodes[i].shape = 'dot';
                nodes[i].size = 15 + (nodes[i].entropy * 15); // Size based on entropy
                nodes[i].color = {{
                    background: color,
                    border: 'rgba(255, 255, 255, 0.5)',
                    highlight: {{
                        background: color,
                        border: '#ffffff'
                    }}
                }};
                nodes[i].font = {{
                    color: fontColor,
                    size: 14,
                    face: 'Arial'
                }};
                nodes[i].shadow = {{
                    enabled: true,
                    color: 'rgba(0, 0, 0, 0.4)',
                    size: 5,
                    x: 1,
                    y: 1
                }};
            }}
            
            // Customize the edges
            for (var i = 0; i < edges.length; i++) {{
                // Set width based on strength
                var strength = edges[i].strength || 0.5;
                
                // Add custom edge styling
                edges[i].width = 1 + (strength * 5);
                edges[i].color = {{
                    color: 'rgba(100, 255, 218, 0.6)',
                    highlight: 'rgba(100, 255, 218, 1)'
                }};
                edges[i].arrows = {{
                    to: {{
                        enabled: false
                    }}
                }};
                edges[i].smooth = {{
                    enabled: true,
                    type: 'continuous'
                }};
                edges[i].shadow = {{
                    enabled: true,
                    color: 'rgba(0, 0, 0, 0.2)',
                    size: 3,
                    x: 0,
                    y: 0
                }};
                
                // Add some curvature
                edges[i].physics = true;
            }}
            
            // Network options
            var options = {{
                physics: {{
                    enabled: true,
                    barnesHut: {{
                        gravitationalConstant: -3000,
                        centralGravity: 0.3,
                        springLength: 150,
                        springConstant: 0.04,
                        damping: 0.09
                    }},
                    stabilization: {{
                        enabled: true,
                        iterations: 200,
                        updateInterval: 25
                    }}
                }},
                interaction: {{
                    navigationButtons: false,
                    hover: true,
                    multiselect: true,
                    keyboard: false,
                    tooltipDelay: 200
                }},
                nodes: {{
                    shape: 'dot'
                }}
            }};
            
            // Create the network
            var data = {{
                nodes: new vis.DataSet(nodes),
                edges: new vis.DataSet(edges)
            }};
            
            var network = new vis.Network(container, data, options);
            
            // Animation effect when loaded
            network.on("stabilizationProgress", function(params) {{
                // Update progress if needed
            }});
            
            network.on("stabilizationIterationsDone", function() {{
                // Animation completed
                network.setOptions({{ physics: {{ enabled: true }} }});
            }});
        </script>
        """,
        height=520,
    )

def bloch_sphere_animation(state_vector):
    """Display an animated 3D Bloch sphere visualization."""
    components.html(
        f"""
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
        #bloch-sphere {{
            width: 100%;
            height: 400px;
            border-radius: 12px;
            overflow: hidden;
        }}
        </style>
        
        <div id="bloch-sphere"></div>
        
        <script>
            // Parse the state vector data
            var state = {json.dumps(state_vector)};
            
            // Calculate Bloch sphere coordinates
            function stateToBloch(state) {{
                // Compute theta (polar angle, 0 to pi)
                var theta = 2 * Math.acos(Math.abs(state[0]));
                
                // Compute phi (azimuthal angle, 0 to 2pi)
                var phi;
                if (state[0] == 0) {{
                    phi = 0;
                }} else {{
                    phi = Math.atan2(state[1].imag, state[1].real);
                }}
                
                // Convert to Cartesian coordinates
                var x = Math.sin(theta) * Math.cos(phi);
                var y = Math.sin(theta) * Math.sin(phi);
                var z = Math.cos(theta);
                
                return {{x: x, y: y, z: z}};
            }}
            
            var blochCoords = stateToBloch(state);
            
            // Create the Bloch sphere
            var data = [
                // Unit sphere (transparent)
                {{
                    type: 'mesh3d',
                    x: Array(20).fill().map((_, i) => 
                        Array(20).fill().map((_, j) => 
                            Math.cos(j/19 * Math.PI * 2) * Math.sin(i/19 * Math.PI)
                        )
                    ).flat(),
                    y: Array(20).fill().map((_, i) => 
                        Array(20).fill().map((_, j) => 
                            Math.sin(j/19 * Math.PI * 2) * Math.sin(i/19 * Math.PI)
                        )
                    ).flat(),
                    z: Array(20).fill().map((_, i) => 
                        Array(20).fill().map((_, j) => 
                            Math.cos(i/19 * Math.PI)
                        )
                    ).flat(),
                    opacity: 0.15,
                    color: '#64ffda',
                    hoverinfo: 'none'
                }},
                
                // Coordinate axes (x, y, z)
                {{
                    type: 'scatter3d',
                    x: [-1.2, 1.2],
                    y: [0, 0],
                    z: [0, 0],
                    mode: 'lines',
                    line: {{
                        color: 'red',
                        width: 3
                    }},
                    hoverinfo: 'none',
                    showlegend: false
                }},
                {{
                    type: 'scatter3d',
                    x: [0, 0],
                    y: [-1.2, 1.2],
                    z: [0, 0],
                    mode: 'lines',
                    line: {{
                        color: 'green',
                        width: 3
                    }},
                    hoverinfo: 'none',
                    showlegend: false
                }},
                {{
                    type: 'scatter3d',
                    x: [0, 0],
                    y: [0, 0],
                    z: [-1.2, 1.2],
                    mode: 'lines',
                    line: {{
                        color: 'blue',
                        width: 3
                    }},
                    hoverinfo: 'none',
                    showlegend: false
                }},
                
                // State vector point
                {{
                    type: 'scatter3d',
                    x: [blochCoords.x],
                    y: [blochCoords.y],
                    z: [blochCoords.z],
                    mode: 'markers',
                    marker: {{
                        size: 10,
                        color: '#ff5252',
                        symbol: 'circle',
                        line: {{
                            color: '#ffffff',
                            width: 2
                        }}
                    }},
                    hoverinfo: 'text',
                    hovertext: 'Quantum State',
                    showlegend: false
                }},
                
                // Line from origin to state point
                {{
                    type: 'scatter3d',
                    x: [0, blochCoords.x],
                    y: [0, blochCoords.y],
                    z: [0, blochCoords.z],
                    mode: 'lines',
                    line: {{
                        color: '#ff5252',
                        width: 4,
                        dash: 'solid'
                    }},
                    hoverinfo: 'none',
                    showlegend: false
                }}
            ];
            
            var layout = {{
                title: {{
                    text: 'Quantum State Visualization',
                    font: {{
                        family: 'Arial',
                        size: 20,
                        color: '#e6f1ff'
                    }}
                }},
                autosize: true,
                showlegend: false,
                margin: {{
                    l: 10,
                    r: 10,
                    b: 10,
                    t: 50,
                    pad: 0
                }},
                scene: {{
                    xaxis: {{
                        title: 'X',
                        range: [-1.2, 1.2],
                        gridcolor: 'rgba(255, 255, 255, 0.1)',
                        zerolinecolor: 'rgba(255, 255, 255, 0.3)'
                    }},
                    yaxis: {{
                        title: 'Y',
                        range: [-1.2, 1.2],
                        gridcolor: 'rgba(255, 255, 255, 0.1)',
                        zerolinecolor: 'rgba(255, 255, 255, 0.3)'
                    }},
                    zaxis: {{
                        title: 'Z',
                        range: [-1.2, 1.2],
                        gridcolor: 'rgba(255, 255, 255, 0.1)',
                        zerolinecolor: 'rgba(255, 255, 255, 0.3)'
                    }},
                    aspectratio: {{
                        x: 1, y: 1, z: 1
                    }},
                    camera: {{
                        eye: {{
                            x: 1.5, 
                            y: 1.5, 
                            z: 1.5
                        }},
                        up: {{
                            x: 0, 
                            y: 0, 
                            z: 1
                        }}
                    }},
                    annotations: [
                        {{
                            showarrow: false,
                            x: 1.1,
                            y: 0,
                            z: 0,
                            text: "X",
                            font: {{
                                color: 'red',
                                size: 14
                            }}
                        }},
                        {{
                            showarrow: false,
                            x: 0,
                            y: 1.1,
                            z: 0,
                            text: "Y",
                            font: {{
                                color: 'green',
                                size: 14
                            }}
                        }},
                        {{
                            showarrow: false,
                            x: 0,
                            y: 0,
                            z: 1.1,
                            text: "Z",
                            font: {{
                                color: 'blue',
                                size: 14
                            }}
                        }}
                    ]
                }},
                paper_bgcolor: 'rgba(16, 33, 65, 0.0)',
                plot_bgcolor: 'rgba(16, 33, 65, 0.0)',
                font: {{
                    family: 'Arial',
                    size: 12,
                    color: '#e6f1ff'
                }}
            }};
            
            var config = {{
                responsive: true,
                displayModeBar: true
            }};
            
            Plotly.newPlot('bloch-sphere', data, layout, config);
            
            // Add animation
            function animate() {{
                // Rotate the point around the z-axis
                var time = Date.now() * 0.001;
                var newX = blochCoords.x * Math.cos(time) - blochCoords.y * Math.sin(time);
                var newY = blochCoords.x * Math.sin(time) + blochCoords.y * Math.cos(time);
                
                var update = {{
                    x: [[newX], [0, newX]],
                    y: [[newY], [0, newY]]
                }};
                
                Plotly.update('bloch-sphere', update, {{}}, [3, 4]);
                requestAnimationFrame(animate);
            }}
            
            animate();
        </script>
        """,
        height=420,
    )