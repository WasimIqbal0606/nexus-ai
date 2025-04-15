import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import json
import math
import random

def get_color_by_state(state):
    """Get color for task state"""
    color_map = {
        'PENDING': '#4299e1',     # Blue
        'IN_PROGRESS': '#f6ad55', # Orange
        'COMPLETED': '#68d391',   # Green
        'BLOCKED': '#fc8181'      # Red
    }
    return color_map.get(state, '#4299e1')

def quantum_simulation_visualizer(simulation_data, height=700):
    """Create an advanced quantum simulation visualization with interactive animations"""
    
    if not simulation_data:
        return None
    
    # Extract simulation steps data
    steps = simulation_data.get('simulation_steps', [])
    tasks = simulation_data.get('tasks', [])
    entanglement_matrix = simulation_data.get('entanglement_matrix', [])
    
    if not steps or not tasks:
        return None
    
    # Prepare task data
    task_data = {}
    for task in tasks:
        task_id = task.get('id')
        task_data[task_id] = {
            'title': task.get('title', 'Unknown Task'),
            'initial_state': task.get('state', 'PENDING'),
            'color': get_color_by_state(task.get('state', 'PENDING')),
            'priority': task.get('priority', 1)
        }
    
    # Prepare step data
    vis_steps = []
    for i, step in enumerate(steps):
        step_data = {'step': i, 'tasks': {}}
        for task_id, task_step_data in step.items():
            # Skip if task not in our task list
            if task_id not in task_data:
                continue
                
            entropy = task_step_data.get('entropy', 0.5)
            state = task_step_data.get('state', task_data[task_id]['initial_state'])
            prob_dist = task_step_data.get('probability_distribution', {})
            
            # Default probability distribution if not available
            if not prob_dist:
                prob_dist = {'PENDING': 0.25, 'IN_PROGRESS': 0.25, 'COMPLETED': 0.25, 'BLOCKED': 0.25}
            
            # Get quantum state visualization data if available
            quantum_state = task_step_data.get('quantum_state', {})
            vis_data = quantum_state.get('visualization_data', [0.25, 0.25, 0.25, 0.25])
            
            # Create step data for this task
            step_data['tasks'][task_id] = {
                'title': task_data[task_id]['title'],
                'state': state,
                'entropy': entropy,
                'color': get_color_by_state(state),
                'probability_distribution': prob_dist,
                'bloch_vector': vis_data[:3] if len(vis_data) >= 3 else [0, 0, 0]
            }
        
        vis_steps.append(step_data)
    
    # Prepare for HTML/JS visualization
    steps_json = json.dumps(vis_steps)
    tasks_json = json.dumps(task_data)
    entanglement_json = json.dumps(entanglement_matrix)
    
    # D3.js and Three.js for visualizations
    d3_src = "https://d3js.org/d3.v7.min.js"
    three_src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
    
    # Create the HTML/JS visualization
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="{d3_src}"></script>
        <script src="{three_src}"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: sans-serif;
                background: transparent;
                color: #f8fafc;
            }}
            #quantum-simulation {{
                width: 100%;
                height: {height}px;
                position: relative;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            }}
            .control-panel {{
                position: absolute;
                top: 20px;
                left: 20px;
                width: 180px;
                background: rgba(15, 23, 42, 0.8);
                padding: 15px;
                border-radius: 8px;
                z-index: 100;
                backdrop-filter: blur(4px);
                border: 1px solid rgba(100, 116, 139, 0.2);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            .panel-title {{
                margin: 0 0 15px 0;
                color: #e2e8f0;
                font-size: 16px;
                font-weight: 600;
                text-align: center;
            }}
            .progress-tracker {{
                width: 100%;
                height: 5px;
                background: rgba(100, 116, 139, 0.2);
                margin: 10px 0;
                border-radius: 3px;
                overflow: hidden;
            }}
            .progress-fill {{
                height: 100%;
                width: 0%;
                background: linear-gradient(90deg, #4338CA, #3B82F6);
                transition: width 0.3s ease;
            }}
            .control-button {{
                background: rgba(59, 130, 246, 0.2);
                border: 1px solid rgba(59, 130, 246, 0.5);
                color: #e2e8f0;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                font-size: 18px;
                transition: all 0.2s ease;
                margin: 0 5px;
            }}
            .control-button:hover {{
                background: rgba(59, 130, 246, 0.4);
            }}
            .control-button:active {{
                transform: scale(0.95);
            }}
            .controls {{
                display: flex;
                justify-content: center;
                margin-top: 15px;
            }}
            .step-counter {{
                text-align: center;
                margin: 10px 0;
                color: #e2e8f0;
                font-size: 14px;
            }}
            .vizcontainer {{
                display: flex;
                position: absolute;
                top: 20px;
                left: 220px;
                right: 20px;
                bottom: 20px;
            }}
            .task-panel {{
                width: 300px;
                height: 100%;
                background: rgba(15, 23, 42, 0.7);
                border-radius: 8px;
                padding: 15px;
                overflow-y: auto;
                backdrop-filter: blur(4px);
                border: 1px solid rgba(100, 116, 139, 0.2);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-right: 15px;
            }}
            .vizspace {{
                flex: 1;
                position: relative;
                background: rgba(15, 23, 42, 0.5);
                border-radius: 8px;
                overflow: hidden;
                padding: 15px;
                backdrop-filter: blur(4px);
                border: 1px solid rgba(100, 116, 139, 0.2);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                display: flex;
                flex-direction: column;
            }}
            .visualization-container {{
                flex: 1;
                position: relative;
                overflow: hidden;
            }}
            #bloch-sphere {{
                width: 100%;
                height: 50%;
                position: relative;
            }}
            #network-viz {{
                width: 100%;
                height: 50%;
                position: relative;
            }}
            .task-card {{
                background: rgba(30, 41, 59, 0.6);
                border-radius: 6px;
                margin-bottom: 10px;
                padding: 12px;
                border-left: 4px solid #3B82F6;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }}
            .task-card.active {{
                box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
                transform: translateX(5px);
            }}
            .task-card-title {{
                font-weight: 600;
                margin: 0 0 5px 0;
                font-size: 14px;
                color: #e2e8f0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .task-state {{
                padding: 3px 6px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 500;
            }}
            .task-entropy {{
                margin: 10px 0 5px 0;
                font-size: 12px;
                color: #cbd5e1;
            }}
            .entropy-bar {{
                height: 4px;
                background: rgba(100, 116, 139, 0.2);
                margin-top: 3px;
                border-radius: 2px;
                overflow: hidden;
            }}
            .entropy-fill {{
                height: 100%;
                background: linear-gradient(90deg, #4338CA, #3B82F6);
                width: 0%;
                transition: width 0.5s ease;
            }}
            .probabilities {{
                display: flex;
                margin-top: 8px;
                height: 20px;
                border-radius: 4px;
                overflow: hidden;
            }}
            .prob-segment {{
                height: 100%;
                transition: width 0.5s ease;
                position: relative;
            }}
            .prob-label {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-size: 10px;
                white-space: nowrap;
                color: rgba(255, 255, 255, 0.9);
                font-weight: 600;
                text-shadow: 0 0 3px rgba(0, 0, 0, 0.6);
                opacity: 0;
                transition: opacity 0.3s ease;
            }}
            .prob-segment:hover .prob-label {{
                opacity: 1;
            }}
            .quantum-pulse {{
                position: absolute;
                top: 0;
                right: 0;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #3B82F6;
                opacity: 0.8;
            }}
            @keyframes pulse {{
                0% {{
                    box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.8);
                }}
                70% {{
                    box-shadow: 0 0 0 5px rgba(59, 130, 246, 0);
                }}
                100% {{
                    box-shadow: 0 0 0 0 rgba(59, 130, 246, 0);
                }}
            }}
            .quantum-particles {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 0;
            }}
            .particle {{
                position: absolute;
                width: 2px;
                height: 2px;
                border-radius: 50%;
                background: rgba(59, 130, 246, 0.7);
                opacity: 0.5;
                animation: float 15s infinite linear;
            }}
            @keyframes float {{
                0% {{ transform: translateY(0) translateX(0); opacity: 0; }}
                10% {{ opacity: 0.5; }}
                90% {{ opacity: 0.5; }}
                100% {{ transform: translateY(-100px) translateX(50px); opacity: 0; }}
            }}
            /* Tabs styling */
            .viz-tabs {{
                display: flex;
                margin-bottom: 10px;
            }}
            .viz-tab {{
                padding: 8px 15px;
                cursor: pointer;
                border-radius: 6px 6px 0 0;
                background: rgba(30, 41, 59, 0.4);
                margin-right: 3px;
                font-size: 13px;
                transition: all 0.2s ease;
            }}
            .viz-tab.active {{
                background: rgba(59, 130, 246, 0.2);
                border-bottom: 2px solid #3B82F6;
            }}
            .viz-tab:hover:not(.active) {{
                background: rgba(30, 41, 59, 0.7);
            }}
            .viz-panel {{
                display: none;
                height: calc(100% - 40px);
            }}
            .viz-panel.active {{
                display: block;
            }}
            .matrix-viz {{
                width: 100%;
                padding: 15px;
                box-sizing: border-box;
            }}
            /* Tooltip */
            .tooltip {{
                position: absolute;
                background: rgba(15, 23, 42, 0.9);
                border: 1px solid #3B82F6;
                border-radius: 6px;
                padding: 10px;
                font-size: 12px;
                pointer-events: none;
                z-index: 1000;
                opacity: 0;
                transition: opacity 0.2s;
                max-width: 220px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
            }}
            #label-placeholder {{
                margin-top: 5px;
                font-size: 13px;
                text-align: center;
                color: #94a3b8;
            }}
            /* Animations */
            .animation-container {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 50;
                opacity: 0;
                transition: opacity 0.3s;
            }}
            .show-animation {{
                opacity: 1;
            }}
        </style>
    </head>
    <body>
        <div id="quantum-simulation">
            <div class="quantum-particles" id="particles"></div>
            
            <!-- Control Panel -->
            <div class="control-panel">
                <h3 class="panel-title">Quantum Simulation</h3>
                <div class="progress-tracker">
                    <div class="progress-fill" id="progress-bar"></div>
                </div>
                <div class="step-counter">
                    Step <span id="step-counter">1</span> of <span id="step-total">5</span>
                </div>
                <div class="controls">
                    <div class="control-button" id="prev-btn">⟨</div>
                    <div class="control-button" id="play-btn">▶</div>
                    <div class="control-button" id="next-btn">⟩</div>
                </div>
            </div>
            
            <!-- Visualization Space -->
            <div class="vizcontainer">
                <!-- Task Panel -->
                <div class="task-panel" id="task-panel">
                    <!-- Task cards will be rendered here -->
                </div>
                
                <!-- Visualization Space -->
                <div class="vizspace">
                    <!-- Tabs -->
                    <div class="viz-tabs">
                        <div class="viz-tab active" data-tab="bloch">Quantum States</div>
                        <div class="viz-tab" data-tab="network">Entanglement Network</div>
                        <div class="viz-tab" data-tab="matrix">Entanglement Matrix</div>
                    </div>
                    
                    <!-- Visualization Panels -->
                    <div class="visualization-container">
                        <!-- Bloch Sphere Viz Panel -->
                        <div class="viz-panel active" id="bloch-panel">
                            <div id="bloch-sphere"></div>
                            <div id="label-placeholder">Select a task to see its quantum state visualization</div>
                        </div>
                        
                        <!-- Network Viz Panel -->
                        <div class="viz-panel" id="network-panel">
                            <div id="network-viz"></div>
                        </div>
                        
                        <!-- Matrix Viz Panel -->
                        <div class="viz-panel" id="matrix-panel">
                            <div class="matrix-viz" id="matrix-viz"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Tooltip -->
            <div class="tooltip" id="tooltip"></div>
            
            <!-- Animation container for transitions -->
            <div class="animation-container" id="animation-container"></div>
        </div>
        
        <script>
            // Simulation data
            const simulationSteps = {steps_json};
            const tasksData = {tasks_json};
            const entanglementMatrix = {entanglement_json};
            
            // State
            let currentStep = 0;
            let isPlaying = false;
            let playInterval;
            let selectedTaskId = null;
            let blochSphere;
            let network;
            
            // Elements
            const prevBtn = document.getElementById('prev-btn');
            const playBtn = document.getElementById('play-btn');
            const nextBtn = document.getElementById('next-btn');
            const progressBar = document.getElementById('progress-bar');
            const stepCounter = document.getElementById('step-counter');
            const stepTotal = document.getElementById('step-total');
            const taskPanel = document.getElementById('task-panel');
            const tooltip = document.getElementById('tooltip');
            const blochPanel = document.getElementById('bloch-panel');
            const networkPanel = document.getElementById('network-panel');
            const matrixPanel = document.getElementById('matrix-panel');
            const labelPlaceholder = document.getElementById('label-placeholder');
            const animationContainer = document.getElementById('animation-container');
            
            // Initialize particles
            function createParticles() {{
                const particlesContainer = document.getElementById('particles');
                for (let i = 0; i < 50; i++) {{
                    const particle = document.createElement('div');
                    particle.className = 'particle';
                    particle.style.left = Math.random() * 100 + '%';
                    particle.style.top = Math.random() * 100 + '%';
                    particle.style.animationDelay = (Math.random() * 10) + 's';
                    particlesContainer.appendChild(particle);
                }}
            }}
            
            // Initialize tabs
            function initTabs() {{
                const tabs = document.querySelectorAll('.viz-tab');
                tabs.forEach(tab => {{
                    tab.addEventListener('click', () => {{
                        // Update active tab
                        tabs.forEach(t => t.classList.remove('active'));
                        tab.classList.add('active');
                        
                        // Update active panel
                        const panels = document.querySelectorAll('.viz-panel');
                        panels.forEach(p => p.classList.remove('active'));
                        const panelId = tab.getAttribute('data-tab') + '-panel';
                        document.getElementById(panelId).classList.add('active');
                    }});
                }});
            }}
            
            // Update UI for current step
            function updateUI() {{
                // Update progress
                const progress = (currentStep / (simulationSteps.length - 1)) * 100;
                progressBar.style.width = `${{progress}}%`;
                stepCounter.textContent = currentStep + 1;
                stepTotal.textContent = simulationSteps.length;
                
                // Get current step data
                const stepData = simulationSteps[currentStep];
                
                // Update task cards
                updateTaskCards(stepData);
                
                // Update visualizations
                if (selectedTaskId) {{
                    updateBlochSphere(selectedTaskId, stepData);
                }}
                updateNetworkVisualization(stepData);
                updateMatrixVisualization();
            }}
            
            // Initialize task cards
            function initTaskCards() {{
                taskPanel.innerHTML = '';
                const firstStep = simulationSteps[0];
                
                for (const [taskId, task] of Object.entries(firstStep.tasks)) {{
                    // Create task card
                    const card = document.createElement('div');
                    card.className = 'task-card';
                    card.id = `task-card-${{taskId}}`;
                    card.style.borderLeftColor = task.color;
                    
                    // Pulse indicator
                    const pulse = document.createElement('div');
                    pulse.className = 'quantum-pulse';
                    pulse.style.animation = 'pulse 2s infinite';
                    card.appendChild(pulse);
                    
                    // Title
                    const title = document.createElement('div');
                    title.className = 'task-card-title';
                    title.innerHTML = `
                        <span>${{task.title}}</span>
                        <span class="task-state" style="background: ${{task.color}}20; color: ${{task.color}}">
                            ${{task.state}}
                        </span>
                    `;
                    card.appendChild(title);
                    
                    // Entropy
                    const entropy = document.createElement('div');
                    entropy.className = 'task-entropy';
                    entropy.innerHTML = `
                        Entropy: <span id="entropy-value-${{taskId}}">${{task.entropy.toFixed(2)}}</span>
                        <div class="entropy-bar">
                            <div class="entropy-fill" id="entropy-fill-${{taskId}}" style="width: ${{task.entropy * 100}}%"></div>
                        </div>
                    `;
                    card.appendChild(entropy);
                    
                    // Probability distribution
                    const probs = document.createElement('div');
                    probs.className = 'probabilities';
                    probs.id = `probs-${{taskId}}`;
                    
                    // Add probability segments
                    for (const [state, prob] of Object.entries(task.probability_distribution)) {{
                        const segment = document.createElement('div');
                        segment.className = 'prob-segment';
                        segment.style.width = `${{prob * 100}}%`;
                        segment.style.background = getColorByState(state);
                        
                        const label = document.createElement('div');
                        label.className = 'prob-label';
                        label.textContent = `${{state}}: ${{(prob * 100).toFixed(0)}}%`;
                        segment.appendChild(label);
                        
                        probs.appendChild(segment);
                    }}
                    
                    card.appendChild(probs);
                    
                    // Add click event
                    card.addEventListener('click', () => {{
                        selectTask(taskId);
                    }});
                    
                    // Add hover event for tooltip
                    card.addEventListener('mouseover', event => {{
                        showTooltip(event, task, taskId);
                    }});
                    
                    card.addEventListener('mousemove', event => {{
                        positionTooltip(event);
                    }});
                    
                    card.addEventListener('mouseout', () => {{
                        hideTooltip();
                    }});
                    
                    taskPanel.appendChild(card);
                }}
                
                // Select first task by default
                if (Object.keys(firstStep.tasks).length > 0) {{
                    selectTask(Object.keys(firstStep.tasks)[0]);
                }}
            }}
            
            // Update task cards for current step
            function updateTaskCards(stepData) {{
                for (const [taskId, task] of Object.entries(stepData.tasks)) {{
                    const card = document.getElementById(`task-card-${{taskId}}`);
                    if (!card) continue;
                    
                    // Update state
                    const stateEl = card.querySelector('.task-state');
                    stateEl.textContent = task.state;
                    stateEl.style.background = `${{task.color}}20`;
                    stateEl.style.color = task.color;
                    
                    // Update card border
                    card.style.borderLeftColor = task.color;
                    
                    // Update entropy with animation
                    const entropyValue = card.querySelector(`#entropy-value-${{taskId}}`);
                    const entropyFill = card.querySelector(`#entropy-fill-${{taskId}}`);
                    
                    // Animate entropy change
                    animateValue(entropyValue, parseFloat(entropyValue.textContent), task.entropy, 500);
                    entropyFill.style.width = `${{task.entropy * 100}}%`;
                    
                    // Update probability distribution
                    const probs = card.querySelector(`#probs-${{taskId}}`);
                    probs.innerHTML = '';
                    
                    // Add probability segments with animation
                    for (const [state, prob] of Object.entries(task.probability_distribution)) {{
                        const segment = document.createElement('div');
                        segment.className = 'prob-segment';
                        segment.style.width = `${{prob * 100}}%`;
                        segment.style.background = getColorByState(state);
                        
                        const label = document.createElement('div');
                        label.className = 'prob-label';
                        label.textContent = `${{state}}: ${{(prob * 100).toFixed(0)}}%`;
                        segment.appendChild(label);
                        
                        probs.appendChild(segment);
                    }}
                }}
            }}
            
            // Select a task for detailed visualization
            function selectTask(taskId) {{
                // Update selection
                selectedTaskId = taskId;
                
                // Update UI to show selected task
                document.querySelectorAll('.task-card').forEach(card => {{
                    card.classList.remove('active');
                }});
                
                const selectedCard = document.getElementById(`task-card-${{taskId}}`);
                if (selectedCard) {{
                    selectedCard.classList.add('active');
                }}
                
                // Hide placeholder text
                labelPlaceholder.style.display = 'none';
                
                // Update visualizations
                const stepData = simulationSteps[currentStep];
                updateBlochSphere(taskId, stepData);
            }}
            
            // Show tooltip
            function showTooltip(event, task, taskId) {{
                const stateProbabilities = Object.entries(task.probability_distribution)
                    .map(([state, prob]) => `<div style="display: flex; justify-content: space-between; margin: 3px 0;">
                        <span style="color: ${{getColorByState(state)}}">${{state}}</span>
                        <span>${{(prob * 100).toFixed(1)}}%</span>
                    </div>`)
                    .join('');
                
                tooltip.innerHTML = `
                    <div style="font-weight: 600; margin-bottom: 5px; color: #e2e8f0;">${{task.title}}</div>
                    <div style="margin-bottom: 5px;">
                        <span style="font-weight: 500;">State:</span> 
                        <span style="color: ${{task.color}};">${{task.state}}</span>
                    </div>
                    <div style="margin-bottom: 5px;">
                        <span style="font-weight: 500;">Entropy:</span> ${{task.entropy.toFixed(2)}}
                    </div>
                    <div style="font-weight: 500; margin-bottom: 2px;">Probabilities:</div>
                    <div style="border-top: 1px solid rgba(100, 116, 139, 0.3); padding-top: 3px;">
                        ${{stateProbabilities}}
                    </div>
                `;
                
                tooltip.style.opacity = 1;
                positionTooltip(event);
            }}
            
            // Position tooltip
            function positionTooltip(event) {{
                tooltip.style.left = (event.pageX + 10) + 'px';
                tooltip.style.top = (event.pageY - 20) + 'px';
            }}
            
            // Hide tooltip
            function hideTooltip() {{
                tooltip.style.opacity = 0;
            }}
            
            // Get color for task state
            function getColorByState(state) {{
                const colorMap = {{
                    'PENDING': '#4299e1',     // Blue
                    'IN_PROGRESS': '#f6ad55', // Orange
                    'COMPLETED': '#68d391',   // Green
                    'BLOCKED': '#fc8181'      // Red
                }};
                return colorMap[state] || '#4299e1';
            }}
            
            // Initialize Three.js Bloch sphere
            function initBlochSphere() {{
                const container = document.getElementById('bloch-sphere');
                
                // Scene
                const scene = new THREE.Scene();
                
                // Camera
                const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
                camera.position.z = 2.5;
                
                // Renderer
                const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
                renderer.setSize(container.clientWidth, container.clientHeight);
                renderer.setClearColor(0x000000, 0);
                container.appendChild(renderer.domElement);
                
                // Sphere (Bloch sphere)
                const sphereGeometry = new THREE.SphereGeometry(1, 32, 32);
                const sphereMaterial = new THREE.MeshBasicMaterial({{
                    color: 0x3B82F6,
                    wireframe: true,
                    transparent: true,
                    opacity: 0.3
                }});
                const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
                scene.add(sphere);
                
                // Axes
                const axisLength = 1.2;
                
                // X axis (red)
                const xAxisGeometry = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(-axisLength, 0, 0),
                    new THREE.Vector3(axisLength, 0, 0)
                ]);
                const xAxisMaterial = new THREE.LineBasicMaterial({{ color: 0xff0000 }});
                const xAxis = new THREE.Line(xAxisGeometry, xAxisMaterial);
                scene.add(xAxis);
                
                // X axis label
                const xLabelDiv = document.createElement('div');
                xLabelDiv.textContent = 'X';
                xLabelDiv.style.position = 'absolute';
                xLabelDiv.style.color = '#ff0000';
                xLabelDiv.style.padding = '2px';
                xLabelDiv.style.fontWeight = '600';
                container.appendChild(xLabelDiv);
                
                // Y axis (green)
                const yAxisGeometry = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(0, -axisLength, 0),
                    new THREE.Vector3(0, axisLength, 0)
                ]);
                const yAxisMaterial = new THREE.LineBasicMaterial({{ color: 0x00ff00 }});
                const yAxis = new THREE.Line(yAxisGeometry, yAxisMaterial);
                scene.add(yAxis);
                
                // Y axis label
                const yLabelDiv = document.createElement('div');
                yLabelDiv.textContent = 'Y';
                yLabelDiv.style.position = 'absolute';
                yLabelDiv.style.color = '#00ff00';
                yLabelDiv.style.padding = '2px';
                yLabelDiv.style.fontWeight = '600';
                container.appendChild(yLabelDiv);
                
                // Z axis (blue)
                const zAxisGeometry = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(0, 0, -axisLength),
                    new THREE.Vector3(0, 0, axisLength)
                ]);
                const zAxisMaterial = new THREE.LineBasicMaterial({{ color: 0x0000ff }});
                const zAxis = new THREE.Line(zAxisGeometry, zAxisMaterial);
                scene.add(zAxis);
                
                // Z axis label
                const zLabelDiv = document.createElement('div');
                zLabelDiv.textContent = 'Z';
                zLabelDiv.style.position = 'absolute';
                zLabelDiv.style.color = '#0000ff';
                zLabelDiv.style.padding = '2px';
                zLabelDiv.style.fontWeight = '600';
                container.appendChild(zLabelDiv);
                
                // State vector
                const vectorGeometry = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(0, 0, 0),
                    new THREE.Vector3(0, 0, 1) // Will be updated dynamically
                ]);
                const vectorMaterial = new THREE.LineBasicMaterial({{ color: 0xffffff, linewidth: 3 }});
                const stateVector = new THREE.Line(vectorGeometry, vectorMaterial);
                scene.add(stateVector);
                
                // Vector endpoint
                const endpointGeometry = new THREE.SphereGeometry(0.05, 16, 16);
                const endpointMaterial = new THREE.MeshBasicMaterial({{ color: 0xffffff }});
                const endpoint = new THREE.Mesh(endpointGeometry, endpointMaterial);
                scene.add(endpoint);
                
                // Rotation animation
                let rotationSpeed = 0.005;
                
                // Handle window resize
                window.addEventListener('resize', () => {{
                    const width = container.clientWidth;
                    const height = container.clientHeight;
                    camera.aspect = width / height;
                    camera.updateProjectionMatrix();
                    renderer.setSize(width, height);
                }});
                
                // Animation loop
                function animate() {{
                    requestAnimationFrame(animate);
                    
                    // Rotate sphere
                    sphere.rotation.y += rotationSpeed;
                    sphere.rotation.z += rotationSpeed * 0.5;
                    
                    // Update axis labels positions
                    const width = container.clientWidth;
                    const height = container.clientHeight;
                    
                    // Project 3D points to 2D
                    const vector = new THREE.Vector3();
                    
                    // X axis label
                    vector.set(axisLength + 0.1, 0, 0);
                    vector.project(camera);
                    xLabelDiv.style.left = ((vector.x * 0.5 + 0.5) * width) + 'px';
                    xLabelDiv.style.top = ((-vector.y * 0.5 + 0.5) * height) + 'px';
                    
                    // Y axis label
                    vector.set(0, axisLength + 0.1, 0);
                    vector.project(camera);
                    yLabelDiv.style.left = ((vector.x * 0.5 + 0.5) * width) + 'px';
                    yLabelDiv.style.top = ((-vector.y * 0.5 + 0.5) * height) + 'px';
                    
                    // Z axis label
                    vector.set(0, 0, axisLength + 0.1);
                    vector.project(camera);
                    zLabelDiv.style.left = ((vector.x * 0.5 + 0.5) * width) + 'px';
                    zLabelDiv.style.top = ((-vector.y * 0.5 + 0.5) * height) + 'px';
                    
                    renderer.render(scene, camera);
                }}
                
                // Start animation
                animate();
                
                // Define update method for the Bloch sphere
                const updateVector = (x, y, z, color) => {{
                    // Normalize the vector
                    const length = Math.sqrt(x*x + y*y + z*z);
                    const nx = length > 0 ? x/length : 0;
                    const ny = length > 0 ? y/length : 0;
                    const nz = length > 0 ? z/length : 0;
                    
                    // Update vector line
                    const points = stateVector.geometry.attributes.position.array;
                    points[3] = nx;
                    points[4] = ny;
                    points[5] = nz;
                    stateVector.geometry.attributes.position.needsUpdate = true;
                    
                    // Update endpoint position
                    endpoint.position.set(nx, ny, nz);
                    
                    // Update colors
                    vectorMaterial.color.set(color);
                    endpointMaterial.color.set(color);
                }};
                
                return {{ updateVector }};
            }}
            
            // Update Bloch sphere visualization
            function updateBlochSphere(taskId, stepData) {{
                const task = stepData.tasks[taskId];
                if (!task) return;
                
                // Extract Bloch vector
                const blochVector = task.bloch_vector;
                
                // Calculate vector components
                let x = 0, y = 0, z = 0;
                
                if (blochVector && blochVector.length >= 3) {{
                    [x, y, z] = blochVector;
                }}
                
                // Update Bloch sphere vector
                blochSphere.updateVector(x, y, z, task.color);
            }}
            
            // Initialize network visualization
            function initNetworkVisualization() {{
                const container = document.getElementById('network-viz');
                
                // Create SVG
                const width = container.clientWidth;
                const height = container.clientHeight;
                
                const svg = d3.select(container)
                    .append('svg')
                    .attr('width', width)
                    .attr('height', height);
                
                // Define force simulation
                const simulation = d3.forceSimulation()
                    .force('charge', d3.forceManyBody().strength(-300))
                    .force('center', d3.forceCenter(width / 2, height / 2))
                    .force('collision', d3.forceCollide().radius(d => d.size + 5))
                    .force('link', d3.forceLink().id(d => d.id).distance(100));
                
                // Create groups for links and nodes
                const linkGroup = svg.append('g').attr('class', 'links');
                const nodeGroup = svg.append('g').attr('class', 'nodes');
                const labelGroup = svg.append('g').attr('class', 'labels');
                
                // First step data
                const firstStep = simulationSteps[0];
                
                // Create nodes and links data
                const nodes = [];
                const nodeIds = new Set();
                
                for (const [taskId, task] of Object.entries(firstStep.tasks)) {{
                    nodes.push({{
                        id: taskId,
                        name: task.title,
                        color: task.color,
                        state: task.state,
                        entropy: task.entropy,
                        size: 10 + (task.entropy * 15)
                    }});
                    nodeIds.add(taskId);
                }}
                
                // Create links from entanglement matrix
                const links = [];
                
                if (entanglementMatrix && entanglementMatrix.length > 0) {{
                    for (let i = 0; i < entanglementMatrix.length; i++) {{
                        for (let j = i + 1; j < entanglementMatrix[i].length; j++) {{
                            const strength = entanglementMatrix[i][j];
                            if (strength > 0) {{
                                // Get task IDs from node indices
                                const source = Array.from(nodeIds)[i];
                                const target = Array.from(nodeIds)[j];
                                
                                links.push({{
                                    source,
                                    target,
                                    strength
                                }});
                            }}
                        }}
                    }}
                }}
                
                // Draw links
                const link = linkGroup.selectAll('line')
                    .data(links)
                    .enter()
                    .append('line')
                    .attr('stroke', '#4299e1')
                    .attr('stroke-opacity', 0.6)
                    .attr('stroke-width', d => d.strength * 3);
                
                // Draw nodes
                const node = nodeGroup.selectAll('circle')
                    .data(nodes)
                    .enter()
                    .append('circle')
                    .attr('r', d => d.size)
                    .attr('fill', d => d.color)
                    .attr('stroke', '#ffffff')
                    .attr('stroke-width', 1.5)
                    .call(d3.drag()
                        .on('start', dragStarted)
                        .on('drag', dragging)
                        .on('end', dragEnded));
                
                // Add labels
                const label = labelGroup.selectAll('text')
                    .data(nodes)
                    .enter()
                    .append('text')
                    .text(d => d.name.length > 15 ? d.name.substring(0, 15) + '...' : d.name)
                    .attr('font-size', 10)
                    .attr('fill', 'white')
                    .attr('text-anchor', 'middle')
                    .attr('dy', -15);
                
                // Handle simulation ticks
                simulation.nodes(nodes)
                    .on('tick', () => {{
                        link
                            .attr('x1', d => d.source.x)
                            .attr('y1', d => d.source.y)
                            .attr('x2', d => d.target.x)
                            .attr('y2', d => d.target.y);
                        
                        node
                            .attr('cx', d => d.x)
                            .attr('cy', d => d.y);
                        
                        label
                            .attr('x', d => d.x)
                            .attr('y', d => d.y);
                    }});
                
                simulation.force('link').links(links);
                
                // Drag handlers
                function dragStarted(event, d) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }}
                
                function dragging(event, d) {{
                    d.fx = event.x;
                    d.fy = event.y;
                }}
                
                function dragEnded(event, d) {{
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }}
                
                // Update method for network
                const updateNetwork = (stepData) => {{
                    // Update nodes
                    for (const [taskId, task] of Object.entries(stepData.tasks)) {{
                        const nodeIndex = nodes.findIndex(n => n.id === taskId);
                        if (nodeIndex >= 0) {{
                            nodes[nodeIndex].color = task.color;
                            nodes[nodeIndex].state = task.state;
                            nodes[nodeIndex].entropy = task.entropy;
                            nodes[nodeIndex].size = 10 + (task.entropy * 15);
                        }}
                    }}
                    
                    // Update visual elements
                    node.transition().duration(500)
                        .attr('r', d => d.size)
                        .attr('fill', d => d.color);
                }};
                
                return {{ updateNetwork }};
            }}
            
            // Update network visualization
            function updateNetworkVisualization(stepData) {{
                network.updateNetwork(stepData);
            }}
            
            // Initialize matrix visualization
            function initMatrixVisualization() {{
                const container = document.getElementById('matrix-viz');
                
                // Create matrix visualization
                const margin = {{top: 20, right: 20, bottom: 20, left: 20}};
                const width = container.clientWidth - margin.left - margin.right;
                const height = container.clientHeight - margin.top - margin.bottom;
                
                const svg = d3.select(container)
                    .append('svg')
                    .attr('width', width + margin.left + margin.right)
                    .attr('height', height + margin.top + margin.bottom)
                    .append('g')
                    .attr('transform', `translate(${{margin.left}}, ${{margin.top}})`);
                
                // Get task names
                const taskIds = Object.keys(simulationSteps[0].tasks);
                const taskNames = taskIds.map(id => simulationSteps[0].tasks[id].title);
                
                // Create scales
                const x = d3.scaleBand()
                    .range([0, width])
                    .domain(taskNames)
                    .padding(0.05);
                
                const y = d3.scaleBand()
                    .range([0, height])
                    .domain(taskNames)
                    .padding(0.05);
                
                // Add X axis
                svg.append('g')
                    .attr('class', 'x-axis')
                    .attr('transform', `translate(0, ${{height}})`)
                    .call(d3.axisBottom(x).tickSize(0))
                    .selectAll('text')
                    .attr('transform', 'translate(-10,0)rotate(-45)')
                    .style('text-anchor', 'end')
                    .style('font-size', '10px')
                    .style('fill', '#cbd5e1');
                
                // Add Y axis
                svg.append('g')
                    .attr('class', 'y-axis')
                    .call(d3.axisLeft(y).tickSize(0))
                    .selectAll('text')
                    .style('font-size', '10px')
                    .style('fill', '#cbd5e1');
                
                // Remove axis lines
                svg.selectAll('.domain').style('stroke', 'none');
                
                // Create color scale
                const color = d3.scaleLinear()
                    .range(['#e2e8f0', '#3B82F6'])
                    .domain([0, 1]);
                
                // Create matrix data
                const matrixData = [];
                
                for (let i = 0; i < taskIds.length; i++) {{
                    for (let j = 0; j < taskIds.length; j++) {{
                        const strength = i === j ? 1 : 
                                       (entanglementMatrix && entanglementMatrix[i] && 
                                        entanglementMatrix[i][j]) || 0;
                        
                        matrixData.push({{
                            x: taskNames[j],
                            y: taskNames[i],
                            strength: strength,
                            xId: taskIds[j],
                            yId: taskIds[i]
                        }});
                    }}
                }}
                
                // Add matrix cells
                svg.selectAll('rect')
                    .data(matrixData)
                    .enter()
                    .append('rect')
                    .attr('x', d => x(d.x))
                    .attr('y', d => y(d.y))
                    .attr('width', x.bandwidth())
                    .attr('height', y.bandwidth())
                    .style('fill', d => color(d.strength))
                    .style('stroke', '#0f172a')
                    .style('opacity', 0.8);
                
                // Add matrix labels
                svg.selectAll('.matrix-value')
                    .data(matrixData.filter(d => d.strength > 0))
                    .enter()
                    .append('text')
                    .attr('class', 'matrix-value')
                    .attr('x', d => x(d.x) + x.bandwidth() / 2)
                    .attr('y', d => y(d.y) + y.bandwidth() / 2)
                    .attr('text-anchor', 'middle')
                    .attr('dominant-baseline', 'middle')
                    .text(d => d.strength.toFixed(1))
                    .style('font-size', '10px')
                    .style('fill', d => d.strength > 0.5 ? 'white' : '#334155');
            }}
            
            // Update matrix visualization
            function updateMatrixVisualization() {{
                // Matrix doesn't change during steps in this version
                // Could be enhanced in future versions with dynamic entanglement
            }}
            
            // Animate value changes
            function animateValue(element, start, end, duration) {{
                let startTimestamp = null;
                const step = timestamp => {{
                    if (!startTimestamp) startTimestamp = timestamp;
                    const progress = Math.min((timestamp - startTimestamp) / duration, 1);
                    const value = start + progress * (end - start);
                    element.textContent = value.toFixed(2);
                    if (progress < 1) {{
                        window.requestAnimationFrame(step);
                    }}
                }};
                window.requestAnimationFrame(step);
            }}
            
            // Step control functions
            function goToNextStep() {{
                if (currentStep < simulationSteps.length - 1) {{
                    currentStep++;
                    
                    // Show transition animation
                    showStepAnimation();
                    
                    // Update UI
                    updateUI();
                }} else if (isPlaying) {{
                    // Stop playback when reaching the end
                    stopPlayback();
                }}
            }}
            
            function goToPrevStep() {{
                if (currentStep > 0) {{
                    currentStep--;
                    
                    // Show transition animation
                    showStepAnimation();
                    
                    // Update UI
                    updateUI();
                }}
            }}
            
            // Start playback
            function startPlayback() {{
                if (!isPlaying) {{
                    isPlaying = true;
                    playBtn.textContent = '⏸︎';
                    
                    // Set interval for automatic progression
                    playInterval = setInterval(() => {{
                        goToNextStep();
                        
                        // Stop at the end
                        if (currentStep >= simulationSteps.length - 1) {{
                            stopPlayback();
                        }}
                    }}, 2000);
                }}
            }}
            
            // Stop playback
            function stopPlayback() {{
                if (isPlaying) {{
                    isPlaying = false;
                    playBtn.textContent = '▶';
                    clearInterval(playInterval);
                }}
            }}
            
            // Toggle playback
            function togglePlayback() {{
                if (isPlaying) {{
                    stopPlayback();
                }} else {{
                    startPlayback();
                }}
            }}
            
            // Step transition animation
            function showStepAnimation() {{
                animationContainer.innerHTML = '';
                animationContainer.classList.add('show-animation');
                
                // Create particles for animation
                for (let i = 0; i < 50; i++) {{
                    const particle = document.createElement('div');
                    particle.style.position = 'absolute';
                    particle.style.width = '3px';
                    particle.style.height = '3px';
                    particle.style.background = '#3B82F6';
                    particle.style.borderRadius = '50%';
                    particle.style.left = `${{Math.random() * 100}}%`;
                    particle.style.top = `${{Math.random() * 100}}%`;
                    particle.style.boxShadow = '0 0 5px #3B82F6';
                    
                    // Random animation
                    const duration = 300 + Math.random() * 500;
                    const distance = 20 + Math.random() * 50;
                    const angle = Math.random() * Math.PI * 2;
                    const dx = Math.cos(angle) * distance;
                    const dy = Math.sin(angle) * distance;
                    
                    particle.animate(
                        [
                            {{ opacity: 0, transform: 'scale(0)' }},
                            {{ opacity: 1, transform: 'scale(1) translate(0, 0)' }},
                            {{ opacity: 0, transform: `scale(0) translate(${{dx}}px, ${{dy}}px)` }}
                        ],
                        {{
                            duration: duration,
                            easing: 'ease-out'
                        }}
                    );
                    
                    animationContainer.appendChild(particle);
                }}
                
                // Remove animation after it completes
                setTimeout(() => {{
                    animationContainer.classList.remove('show-animation');
                }}, 800);
            }}
            
            // Initialize visualization
            function init() {{
                // Create particles
                createParticles();
                
                // Initialize tabs
                initTabs();
                
                // Initialize task cards
                initTaskCards();
                
                // Initialize Bloch sphere
                blochSphere = initBlochSphere();
                
                // Initialize network visualization
                network = initNetworkVisualization();
                
                // Initialize matrix visualization
                initMatrixVisualization();
                
                // Set up UI
                stepTotal.textContent = simulationSteps.length;
                
                // Button event listeners
                prevBtn.addEventListener('click', () => {{
                    stopPlayback();
                    goToPrevStep();
                }});
                
                nextBtn.addEventListener('click', () => {{
                    stopPlayback();
                    goToNextStep();
                }});
                
                playBtn.addEventListener('click', togglePlayback);
                
                // Initial UI update
                updateUI();
            }}
            
            // Initialize when page loads
            window.addEventListener('load', init);
        </script>
    </body>
    </html>
    """
    
    # Display the visualization
    components.html(html_content, height=height, scrolling=False)
    
    return None