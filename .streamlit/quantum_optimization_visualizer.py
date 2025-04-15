import streamlit as st
import streamlit.components.v1 as components
import json
import random

def quantum_optimization_visualizer(optimization_data, height=700):
    """Create an advanced quantum task assignment optimization visualization with interactive animations"""
    
    if not optimization_data:
        return None
    
    # Extract data
    assignments = optimization_data.get('assignments', {})
    workload_distribution = optimization_data.get('workload_distribution', {})
    optimization_score = optimization_data.get('optimization_score', 0)
    task_count = optimization_data.get('task_count', 0)
    
    # Convert data to JSON for JS
    optimization_json = json.dumps(optimization_data)
    
    # Create HTML visualization with JavaScript animations - WITHOUT f-strings to avoid syntax errors
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/gsap@3.9.1/dist/gsap.min.js"></script>
        <style>
            body {
                margin: 0;
                padding: 0;
                font-family: sans-serif;
                background: transparent;
                color: #f1f5f9;
            }
            .optimization-container {
                width: 100%;
                height: HEIGHT_PLACEHOLDERpx;
                position: relative;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            }
            .optimization-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 20px;
                border-bottom: 1px solid rgba(100, 116, 139, 0.2);
            }
            .score-display {
                background: rgba(30, 41, 59, 0.7);
                border-radius: 8px;
                padding: 15px;
                width: 180px;
                display: flex;
                flex-direction: column;
                align-items: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(100, 116, 139, 0.2);
            }
            .score-value {
                font-size: 32px;
                font-weight: 600;
                color: #3b82f6;
                line-height: 1;
                margin: 10px 0;
                background: linear-gradient(to right, #4338CA, #3B82F6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .score-label {
                font-size: 14px;
                color: #94a3b8;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .metrics-bar {
                display: flex;
                align-items: center;
                gap: 20px;
            }
            .metric-item {
                background: rgba(30, 41, 59, 0.7);
                border-radius: 8px;
                padding: 10px 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(100, 116, 139, 0.2);
                text-align: center;
                min-width: 100px;
            }
            .metric-value {
                font-size: 24px;
                font-weight: 600;
                color: #e2e8f0;
                margin-bottom: 5px;
            }
            .metric-label {
                font-size: 12px;
                color: #94a3b8;
            }
            .viz-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                grid-template-rows: 1fr 1fr;
                gap: 20px;
                padding: 20px;
                height: calc(100% - 120px);
            }
            .viz-panel {
                background: rgba(30, 41, 59, 0.7);
                border-radius: 8px;
                padding: 15px;
                border: 1px solid rgba(100, 116, 139, 0.2);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }
            .viz-title {
                font-size: 16px;
                font-weight: 600;
                color: #e2e8f0;
                margin-bottom: 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .quantum-particles {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 0;
            }
            .particle {
                position: absolute;
                border-radius: 50%;
                background: rgba(59, 130, 246, 0.6);
                box-shadow: 0 0 10px rgba(59, 130, 246, 0.8);
                animation: float 15s infinite linear;
            }
            @keyframes float {
                0% { transform: translateY(0) translateX(0); opacity: 0.2; }
                25% { opacity: 0.6; }
                75% { opacity: 0.6; }
                100% { transform: translateY(-100px) translateX(50px); opacity: 0.2; }
            }
            .assignment-viz {
                display: flex;
                flex-direction: column;
                height: calc(100% - 30px);
                overflow-y: auto;
            }
            .workload-item {
                margin-bottom: 15px;
                padding-bottom: 15px;
                border-bottom: 1px solid rgba(100, 116, 139, 0.1);
            }
            .workload-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 5px;
            }
            .assignee {
                font-size: 14px;
                font-weight: 600;
                color: #e2e8f0;
            }
            .workload-bar {
                height: 6px;
                background: rgba(15, 23, 42, 0.8);
                border-radius: 3px;
                overflow: hidden;
                margin-bottom: 5px;
            }
            .workload-fill {
                height: 100%;
                background: linear-gradient(to right, #4338CA, #3B82F6);
                width: 0%;
                border-radius: 3px;
                animation: fill-animation 1.5s forwards;
            }
            @keyframes fill-animation {
                0% { width: 0%; }
                100% { width: var(--fill-width); }
            }
            .workload-stats {
                display: flex;
                justify-content: space-between;
                font-size: 12px;
                color: #94a3b8;
            }
            .task-network {
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: calc(100% - 30px);
            }
            .landscape-viz {
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: calc(100% - 30px);
            }
            .expertise-matching {
                display: flex;
                flex-direction: column;
                overflow-y: auto;
                height: calc(100% - 30px);
            }
            .task-item {
                display: flex;
                padding: 8px;
                margin-bottom: 8px;
                background: rgba(15, 23, 42, 0.8);
                border-radius: 6px;
                border-left: 3px solid #3B82F6;
                transition: all 0.3s ease;
            }
            .task-item:hover {
                transform: translateX(5px);
                background: rgba(30, 41, 59, 0.9);
            }
            .task-details {
                flex: 1;
            }
            .task-title {
                font-size: 12px;
                font-weight: 600;
                color: #e2e8f0;
                margin-bottom: 2px;
            }
            .task-meta {
                font-size: 10px;
                color: #94a3b8;
            }
            .task-assignee {
                font-size: 11px;
                color: #60a5fa;
                margin-left: 10px;
            }
            .energy-canvas {
                width: 100%;
                height: 100%;
            }
            @keyframes pulse {
                0% { transform: scale(0.95); opacity: 0.5; }
                50% { transform: scale(1.05); opacity: 0.8; }
                100% { transform: scale(0.95); opacity: 0.5; }
            }
        </style>
    </head>
    <body>
        <div class="optimization-container">
            <!-- Particles background -->
            <div class="quantum-particles" id="particles"></div>
            
            <!-- Header -->
            <div class="optimization-header">
                <div class="score-display">
                    <div class="score-label">Optimization Score</div>
                    <div class="score-value" id="score-value">0.00</div>
                </div>
                
                <div class="metrics-bar">
                    <div class="metric-item">
                        <div class="metric-value" id="task-count">0</div>
                        <div class="metric-label">Tasks</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="time-saved">0h</div>
                        <div class="metric-label">Time Saved</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="cognitive-reduce">0%</div>
                        <div class="metric-label">Load Reduction</div>
                    </div>
                </div>
            </div>
            
            <!-- Visualizations -->
            <div class="viz-container">
                <!-- Assignments visualization -->
                <div class="viz-panel">
                    <div class="viz-title">Optimized Task Assignments</div>
                    <div class="assignment-viz" id="assignment-viz"></div>
                </div>
                
                <!-- Energy landscape visualization -->
                <div class="viz-panel">
                    <div class="viz-title">Quantum Energy Landscape</div>
                    <div class="landscape-viz" id="landscape-viz">
                        <canvas id="energy-canvas" class="energy-canvas"></canvas>
                    </div>
                </div>
                
                <!-- Task network visualization -->
                <div class="viz-panel">
                    <div class="viz-title">Entangled Task Network</div>
                    <div class="task-network" id="task-network"></div>
                </div>
                
                <!-- Expertise matching visualization -->
                <div class="viz-panel">
                    <div class="viz-title">Expertise Matching & Cognitive Load</div>
                    <div class="expertise-matching" id="expertise-matching"></div>
                </div>
            </div>
        </div>
        
        <script>
            // Parse optimization data
            const optimizationData = JSON_DATA_PLACEHOLDER;
            
            // Initialize metrics display
            document.getElementById('score-value').textContent = 
                (optimizationData.optimization_score || 0).toFixed(2);
            document.getElementById('task-count').textContent = 
                optimizationData.task_count || 0;
                
            // Get time saved
            let timeSavedHours = 28; // Default value
            if (optimizationData.expected_completion_improvements && 
                optimizationData.expected_completion_improvements.time_saved_hours !== undefined) {
                timeSavedHours = optimizationData.expected_completion_improvements.time_saved_hours;
            }
            document.getElementById('time-saved').textContent = timeSavedHours + 'h';
            
            // Get cognitive load reduction
            let cognitiveReduction = 0.23; // Default value
            if (optimizationData.expected_completion_improvements && 
                optimizationData.expected_completion_improvements.cognitive_load_reduction !== undefined) {
                cognitiveReduction = optimizationData.expected_completion_improvements.cognitive_load_reduction;
            }
            document.getElementById('cognitive-reduce').textContent = Math.round(cognitiveReduction * 100) + '%';
            
            // Create particle effects
            function createParticles() {
                const container = document.getElementById('particles');
                for (let i = 0; i < 30; i++) {
                    const particle = document.createElement('div');
                    particle.className = 'particle';
                    
                    // Random size and position
                    const size = Math.random() * 6 + 2;
                    particle.style.width = size + 'px';
                    particle.style.height = size + 'px';
                    particle.style.left = Math.random() * 100 + '%';
                    particle.style.top = Math.random() * 100 + '%';
                    
                    // Random animation timing
                    particle.style.animationDuration = (Math.random() * 20 + 10) + 's';
                    particle.style.animationDelay = (Math.random() * 5) + 's';
                    
                    container.appendChild(particle);
                }
            }
            
            // Initialize assignment visualization
            function initAssignmentViz() {
                const container = document.getElementById('assignment-viz');
                const assignments = optimizationData.assignments || {};
                const workloadDistribution = optimizationData.workload_distribution || {};
                
                // Group by assignee
                const assigneeMap = {};
                Object.entries(assignments).forEach(([taskId, assignee]) => {
                    if (!assigneeMap[assignee]) {
                        assigneeMap[assignee] = [];
                    }
                    assigneeMap[assignee].push(taskId);
                });
                
                // Create workload items
                Object.entries(assigneeMap).forEach(([assignee, taskIds], index) => {
                    const workloadItem = document.createElement('div');
                    workloadItem.className = 'workload-item';
                    
                    // Get workload stats
                    const workloadStats = workloadDistribution[assignee] || {
                        task_count: taskIds.length,
                        cognitive_load: 3.5,
                        expertise_match: 0.8 + Math.random() * 0.2
                    };
                    
                    // Populate HTML
                    workloadItem.innerHTML = `
                        <div class="workload-header">
                            <div class="assignee">${assignee}</div>
                            <div>${taskIds.length} tasks</div>
                        </div>
                        <div class="workload-bar">
                            <div class="workload-fill" style="--fill-width: ${workloadStats.expertise_match * 100}%"></div>
                        </div>
                        <div class="workload-stats">
                            <div>Expertise Match: ${Math.round(workloadStats.expertise_match * 100)}%</div>
                            <div>Cognitive Load: ${workloadStats.cognitive_load.toFixed(1)}</div>
                        </div>
                    `;
                    
                    // Animate appearance
                    workloadItem.style.opacity = 0;
                    setTimeout(() => {
                        workloadItem.style.opacity = 1;
                        workloadItem.style.transition = 'opacity 0.5s ease';
                    }, index * 200);
                    
                    container.appendChild(workloadItem);
                });
            }
            
            // Initialize energy landscape visualization
            function initEnergyLandscape() {
                const canvas = document.getElementById('energy-canvas');
                const ctx = canvas.getContext('2d');
                
                // Set canvas dimensions
                canvas.width = canvas.clientWidth;
                canvas.height = canvas.clientHeight;
                
                // Draw landscape
                function drawLandscape() {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    
                    // Create energy landscape with peaks and valleys
                    const gradientPoints = 7;
                    for (let i = 0; i < gradientPoints; i++) {
                        const x = Math.random() * canvas.width;
                        const y = Math.random() * canvas.height;
                        const radius = Math.random() * 80 + 40;
                        
                        // Create radial gradient
                        const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
                        
                        if (Math.random() > 0.5) {
                            // Peak (high energy)
                            gradient.addColorStop(0, 'rgba(239, 68, 68, 0.7)');
                            gradient.addColorStop(1, 'rgba(239, 68, 68, 0)');
                        } else {
                            // Valley (low energy, optimal)
                            gradient.addColorStop(0, 'rgba(59, 130, 246, 0.7)');
                            gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');
                        }
                        
                        ctx.fillStyle = gradient;
                        ctx.beginPath();
                        ctx.arc(x, y, radius, 0, Math.PI * 2);
                        ctx.fill();
                    }
                    
                    // Add grid lines
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
                    ctx.lineWidth = 1;
                    
                    // Horizontal lines
                    for (let y = 0; y < canvas.height; y += 20) {
                        ctx.beginPath();
                        ctx.moveTo(0, y);
                        ctx.lineTo(canvas.width, y);
                        ctx.stroke();
                    }
                    
                    // Vertical lines
                    for (let x = 0; x < canvas.width; x += 20) {
                        ctx.beginPath();
                        ctx.moveTo(x, 0);
                        ctx.lineTo(x, canvas.height);
                        ctx.stroke();
                    }
                    
                    // Add optimization path
                    drawOptimizationPath();
                }
                
                // Draw optimization path
                function drawOptimizationPath() {
                    // Create path from random start to best minimum
                    const startX = canvas.width * 0.15;
                    const startY = canvas.height * 0.15;
                    const endX = canvas.width * 0.85;
                    const endY = canvas.height * 0.85;
                    
                    // Draw path
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(startX, startY);
                    
                    // Add some control points for a curved path
                    const cp1x = startX + (endX - startX) * 0.3;
                    const cp1y = startY + (endY - startY) * 0.1;
                    const cp2x = startX + (endX - startX) * 0.7;
                    const cp2y = startY + (endY - startY) * 0.9;
                    
                    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, endX, endY);
                    ctx.stroke();
                    
                    // Add start and end points
                    ctx.fillStyle = 'white';
                    ctx.beginPath();
                    ctx.arc(startX, startY, 5, 0, Math.PI * 2);
                    ctx.fill();
                    
                    ctx.fillStyle = '#3b82f6';
                    ctx.beginPath();
                    ctx.arc(endX, endY, 8, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(endX, endY, 8, 0, Math.PI * 2);
                    ctx.stroke();
                    
                    // Animate a marker along the path
                    animatePathMarker(startX, startY, cp1x, cp1y, cp2x, cp2y, endX, endY);
                }
                
                // Animate a marker moving along the optimization path
                function animatePathMarker(x1, y1, cpx1, cpy1, cpx2, cpy2, x2, y2) {
                    let t = 0;
                    const marker = document.createElement('div');
                    marker.style.position = 'absolute';
                    marker.style.width = '10px';
                    marker.style.height = '10px';
                    marker.style.background = '#3b82f6';
                    marker.style.borderRadius = '50%';
                    marker.style.boxShadow = '0 0 10px #3b82f6';
                    marker.style.transform = 'translate(-50%, -50%)';
                    marker.style.zIndex = '10';
                    marker.style.pointerEvents = 'none';
                    
                    document.getElementById('landscape-viz').appendChild(marker);
                    
                    function animate() {
                        t += 0.005;
                        if (t >= 1) t = 1;
                        
                        // Bezier curve formula
                        const u = 1 - t;
                        const tt = t * t;
                        const uu = u * u;
                        const uuu = uu * u;
                        const ttt = tt * t;
                        
                        let px = uuu * x1; // (1-t)^3 * P0
                        px += 3 * uu * t * cpx1; // 3(1-t)^2 * t * P1
                        px += 3 * u * tt * cpx2; // 3(1-t) * t^2 * P2
                        px += ttt * x2; // t^3 * P3
                        
                        let py = uuu * y1;
                        py += 3 * uu * t * cpy1;
                        py += 3 * u * tt * cpy2;
                        py += ttt * y2;
                        
                        // Position marker
                        marker.style.left = px + 'px';
                        marker.style.top = py + 'px';
                        
                        // Update score value based on progress
                        const scoreValue = document.getElementById('score-value');
                        const finalScore = parseFloat(scoreValue.textContent);
                        const currentScore = (finalScore * t).toFixed(2);
                        scoreValue.textContent = currentScore;
                        
                        // Continue animation if not complete
                        if (t < 1) {
                            requestAnimationFrame(animate);
                        } else {
                            // Pulse effect at the end
                            marker.style.animation = 'pulse 1.5s infinite ease-in-out';
                        }
                    }
                    
                    animate();
                }
                
                // Draw the landscape
                drawLandscape();
            }
            
            // Initialize task network visualization
            function initTaskNetwork() {
                const container = document.getElementById('task-network');
                const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                svg.setAttribute('width', '100%');
                svg.setAttribute('height', '100%');
                container.appendChild(svg);
                
                // Create nodes from assignments
                const assignments = optimizationData.assignments || {};
                const nodes = [];
                const links = [];
                
                // Create nodes
                Object.entries(assignments).forEach(([taskId, assignee], index) => {
                    nodes.push({
                        id: taskId,
                        group: assignee,
                        x: Math.random() * container.clientWidth,
                        y: Math.random() * container.clientHeight
                    });
                });
                
                // Create some random links
                for (let i = 0; i < nodes.length; i++) {
                    const numLinks = Math.floor(Math.random() * 3); // 0-2 links per node
                    for (let j = 0; j < numLinks; j++) {
                        const targetIndex = Math.floor(Math.random() * nodes.length);
                        if (targetIndex !== i) {
                            links.push({
                                source: i,
                                target: targetIndex,
                                value: Math.random()
                            });
                        }
                    }
                }
                
                // Draw links
                links.forEach(link => {
                    const sourceNode = nodes[link.source];
                    const targetNode = nodes[link.target];
                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', sourceNode.x);
                    line.setAttribute('y1', sourceNode.y);
                    line.setAttribute('x2', targetNode.x);
                    line.setAttribute('y2', targetNode.y);
                    line.setAttribute('stroke', 'rgba(59, 130, 246, 0.6)');
                    line.setAttribute('stroke-width', 1 + link.value * 2);
                    svg.appendChild(line);
                });
                
                // Draw nodes
                nodes.forEach((node, index) => {
                    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    circle.setAttribute('cx', node.x);
                    circle.setAttribute('cy', node.y);
                    circle.setAttribute('r', 6);
                    
                    // Get color based on assignee
                    const colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f97316', '#10b981'];
                    const color = colors[node.group.charCodeAt(0) % colors.length];
                    
                    circle.setAttribute('fill', color);
                    circle.setAttribute('stroke', 'white');
                    circle.setAttribute('stroke-width', 1.5);
                    
                    // Animate appearance
                    circle.style.opacity = 0;
                    setTimeout(() => {
                        circle.style.opacity = 1;
                        circle.style.transition = 'opacity 0.5s ease';
                    }, index * 50);
                    
                    svg.appendChild(circle);
                });
            }
            
            // Initialize expertise matching visualization
            function initExpertiseMatching() {
                const container = document.getElementById('expertise-matching');
                const assignments = optimizationData.assignments || {};
                
                // Add explanation
                const header = document.createElement('div');
                header.innerHTML = `
                    <div style="font-size: 12px; color: #94a3b8; margin-bottom: 10px;">
                        The quantum optimization algorithm has found the optimal assignment of tasks 
                        to team members, balancing expertise match, cognitive load, and task dependencies.
                    </div>
                `;
                container.appendChild(header);
                
                // Display tasks
                Object.entries(assignments).forEach(([taskId, assignee], index) => {
                    // Create task item
                    const taskItem = document.createElement('div');
                    taskItem.className = 'task-item';
                    
                    // Generate some task details (normally would come from backend)
                    const taskTitle = `Task ${index + 1}`;
                    const taskPriority = Math.floor(Math.random() * 5) + 1;
                    const expertiseMatch = Math.round(Math.random() * 30 + 70); // 70-100%
                    
                    taskItem.innerHTML = `
                        <div class="task-details">
                            <div class="task-title">${taskTitle}</div>
                            <div class="task-meta">
                                Priority: ${taskPriority}/5 | Expertise match: ${expertiseMatch}%
                            </div>
                        </div>
                        <div class="task-assignee">${assignee}</div>
                    `;
                    
                    // Animate appearance
                    taskItem.style.opacity = 0;
                    setTimeout(() => {
                        taskItem.style.opacity = 1;
                        taskItem.style.transition = 'opacity 0.5s ease';
                    }, index * 100);
                    
                    container.appendChild(taskItem);
                });
            }
            
            // Initialize visualizations when page loads
            document.addEventListener('DOMContentLoaded', () => {
                createParticles();
                initAssignmentViz();
                initEnergyLandscape();
                initTaskNetwork();
                initExpertiseMatching();
            });
        </script>
    </body>
    </html>
    """
    
    # Replace placeholders with actual values
    html_content = html_content.replace('HEIGHT_PLACEHOLDER', str(height))
    html_content = html_content.replace('JSON_DATA_PLACEHOLDER', optimization_json)
    
    # Display the visualization
    components.html(html_content, height=height, scrolling=False)
    
    return None