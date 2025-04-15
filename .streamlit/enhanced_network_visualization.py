import streamlit as st
import streamlit.components.v1 as components
import json
import random
import math
import numpy as np

def generate_network_data(tasks):
    """Generate network data from tasks for D3 visualization"""
    nodes = []
    links = []
    used_ids = set()
    
    for task in tasks:
        task_id = task.get('id')
        if task_id in used_ids:
            continue
            
        used_ids.add(task_id)
        
        # Extract relevant data
        task_title = task.get('title', 'Unknown Task')
        task_state = task.get('state', 'PENDING')
        task_entropy = task.get('entropy', 0.5)
        task_priority = task.get('priority', 1)
        task_assignee = task.get('assignee', None)
        
        # Add node
        nodes.append({
            'id': task_id,
            'title': task_title,
            'state': task_state,
            'entropy': task_entropy,
            'priority': task_priority,
            'assignee': task_assignee,
            'size': 10 + (task_entropy * 15) + (task_priority * 3)  # Size based on entropy and priority
        })
        
        # Add links for entangled tasks
        for entangled_id in task.get('entangled_tasks', []):
            if task_id != entangled_id:  # Avoid self-links
                links.append({
                    'source': task_id,
                    'target': entangled_id,
                    'value': 0.5 + random.random() * 0.5  # Random strength between 0.5 and 1.0
                })
    
    return {'nodes': nodes, 'links': links}

def quantum_entanglement_network(tasks, height=600, width=None):
    """Create an enhanced entanglement network visualization with D3"""
    if not tasks or len(tasks) == 0:
        return None
        
    # Generate network data
    network_data = generate_network_data(tasks)
    
    # Convert to JSON for JavaScript
    network_json = json.dumps(network_data)
    
    # Calculate width if not provided
    if not width:
        width = "100%"
    
    # Create HTML/JavaScript visualization
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                overflow: hidden;
                background: transparent;
                font-family: sans-serif;
            }}
            .network-container {{
                position: relative;
                width: 100%;
                height: {height}px;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            }}
            .svg-container {{
                width: 100%;
                height: 100%;
            }}
            .links line {{
                stroke-opacity: 0.6;
                stroke-width: 2px;
                stroke-dasharray: 5;
                animation: dash 30s linear infinite;
            }}
            @keyframes dash {{
                to {{ stroke-dashoffset: -1000; }}
            }}
            .nodes circle {{
                stroke: #fff;
                stroke-width: 1.5px;
                transition: all 0.3s ease;
            }}
            .nodes circle:hover {{
                stroke: #fff;
                stroke-width: 3px;
                filter: drop-shadow(0 0 8px rgba(255,255,255,0.8));
            }}
            .node-labels {{
                pointer-events: none;
                font-size: 10px;
                fill: #fff;
                text-shadow: 0 1px 3px rgba(0,0,0,0.8);
            }}
            .legend {{
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(15, 23, 42, 0.8);
                padding: 10px;
                border-radius: 8px;
                color: white;
                font-size: 12px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(4px);
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                margin: 5px 0;
            }}
            .legend-color {{
                width: 15px;
                height: 15px;
                border-radius: 50%;
                margin-right: 8px;
            }}
            .tooltip {{
                position: absolute;
                background: rgba(15, 23, 42, 0.9);
                border: 1px solid #3B82F6;
                border-radius: 6px;
                padding: 10px;
                font-size: 12px;
                pointer-events: none;
                z-index: 10;
                opacity: 0;
                transition: opacity 0.2s;
                color: white;
                max-width: 220px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
            }}
            .tooltip-field {{
                margin: 3px 0;
            }}
            .tooltip-label {{
                font-weight: bold;
                color: #94a3b8;
            }}
            .background-particles {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
            }}
            .particle {{
                position: absolute;
                border-radius: 50%;
                background: rgba(59, 130, 246, 0.3);
                pointer-events: none;
                animation: float 15s infinite linear;
            }}
            @keyframes float {{
                0% {{ transform: translateY(0) translateX(0); opacity: 0; }}
                10% {{ opacity: 0.3; }}
                90% {{ opacity: 0.3; }}
                100% {{ transform: translateY(-100px) translateX(50px); opacity: 0; }}
            }}
            .pulse-ring {{
                position: absolute;
                border-radius: 50%;
                animation: pulse-animation 3s infinite ease-out;
            }}
            @keyframes pulse-animation {{
                0% {{ transform: scale(0.1); opacity: 0; }}
                50% {{ opacity: 0.2; }}
                100% {{ transform: scale(3); opacity: 0; }}
            }}
            .highlight-overlay {{
                position: absolute;
                pointer-events: none;
                z-index: 1;
                width: 100%;
                height: 100%;
                left: 0;
                top: 0;
            }}
        </style>
    </head>
    <body>
        <div class="network-container">
            <!-- Background effects -->
            <div class="background-particles" id="particles"></div>
            <div class="highlight-overlay" id="highlight-overlay"></div>
            
            <!-- Network visualization -->
            <div class="svg-container" id="network"></div>
            
            <!-- Legend -->
            <div class="legend">
                <div style="font-weight: bold; margin-bottom: 8px;">Task States</div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #4299e1;"></div>
                    <div>PENDING</div>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #f6ad55;"></div>
                    <div>IN PROGRESS</div>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #68d391;"></div>
                    <div>COMPLETED</div>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #fc8181;"></div>
                    <div>BLOCKED</div>
                </div>
                <div style="border-top: 1px solid rgba(255,255,255,0.1); margin: 8px 0;"></div>
                <div style="font-size: 10px; color: #94a3b8;">
                    Node size indicates entropy & priority
                </div>
            </div>
            
            <!-- Tooltip -->
            <div class="tooltip" id="tooltip"></div>
        </div>
        
        <script>
            // Parse the network data
            const networkData = {network_json};
            
            // Color mapping for states
            const stateColors = {{
                'PENDING': '#4299e1',     // Blue
                'IN_PROGRESS': '#f6ad55', // Orange
                'COMPLETED': '#68d391',   // Green
                'BLOCKED': '#fc8181'      // Red
            }};
            
            // Color palette for links
            const linkColorScale = d3.scaleLinear()
                .domain([0, 1])
                .range(['#60a5fa', '#818cf8']);
            
            // Create background particles
            function createParticles() {{
                const container = document.getElementById('particles');
                for (let i = 0; i < 50; i++) {{
                    const particle = document.createElement('div');
                    particle.className = 'particle';
                    
                    // Random size
                    const size = Math.random() * 4 + 1;
                    particle.style.width = size + 'px';
                    particle.style.height = size + 'px';
                    
                    // Random position
                    particle.style.left = Math.random() * 100 + '%';
                    particle.style.top = Math.random() * 100 + '%';
                    
                    // Animation delay
                    particle.style.animationDelay = Math.random() * 5 + 's';
                    
                    container.appendChild(particle);
                }}
                
                // Add a few pulse rings
                for (let i = 0; i < 3; i++) {{
                    const ring = document.createElement('div');
                    ring.className = 'pulse-ring';
                    
                    // Random position
                    ring.style.left = (30 + Math.random() * 40) + '%';
                    ring.style.top = (30 + Math.random() * 40) + '%';
                    
                    // Size
                    const size = 10 + Math.random() * 20;
                    ring.style.width = size + 'px';
                    ring.style.height = size + 'px';
                    
                    // Color
                    ring.style.border = '1px solid rgba(59, 130, 246, 0.5)';
                    
                    // Animation delay
                    ring.style.animationDelay = Math.random() * 2 + 's';
                    
                    container.appendChild(ring);
                }}
            }}
            
            // Show tooltip
            function showTooltip(event, d) {{
                const tooltip = document.getElementById('tooltip');
                
                // Format task information
                const stateBadge = `<span style="padding: 2px 6px; background: ${{stateColors[d.state]}}30; color: ${{stateColors[d.state]}}; border-radius: 4px; font-size: 10px;">${{d.state}}</span>`;
                
                tooltip.innerHTML = `
                    <div style="font-weight: bold; margin-bottom: 5px;">${{d.title}}</div>
                    <div class="tooltip-field">
                        <span class="tooltip-label">State:</span> ${{stateBadge}}
                    </div>
                    <div class="tooltip-field">
                        <span class="tooltip-label">Priority:</span> ${{d.priority}}/5
                    </div>
                    <div class="tooltip-field">
                        <span class="tooltip-label">Entropy:</span> ${{d.entropy.toFixed(2)}}
                    </div>
                ` + (d.assignee ? `
                    <div class="tooltip-field">
                        <span class="tooltip-label">Assigned to:</span> ${{d.assignee}}
                    </div>` : '') + `
                    <div class="tooltip-field">
                        <span class="tooltip-label">Connections:</span> ${{d.connections || 0}}
                    </div>
                `;
                
                // Position tooltip
                tooltip.style.left = (event.pageX + 10) + 'px';
                tooltip.style.top = (event.pageY - 10) + 'px';
                
                // Show tooltip
                tooltip.style.opacity = 1;
            }}
            
            // Hide tooltip
            function hideTooltip() {{
                document.getElementById('tooltip').style.opacity = 0;
            }}
            
            // Move tooltip
            function moveTooltip(event) {{
                const tooltip = document.getElementById('tooltip');
                tooltip.style.left = (event.pageX + 10) + 'px';
                tooltip.style.top = (event.pageY - 10) + 'px';
            }}
            
            // Create network visualization
            function createNetwork() {{
                const container = document.getElementById('network');
                const width = container.clientWidth;
                const height = container.clientHeight;
                
                // Create SVG
                const svg = d3.select(container)
                    .append('svg')
                    .attr('width', width)
                    .attr('height', height);
                
                // Add a subtle grid background
                const gridSize = 30;
                const grid = svg.append('g')
                    .attr('class', 'grid');
                
                // Create horizontal grid lines
                for (let y = 0; y < height; y += gridSize) {{
                    grid.append('line')
                        .attr('x1', 0)
                        .attr('y1', y)
                        .attr('x2', width)
                        .attr('y2', y)
                        .attr('stroke', 'rgba(255, 255, 255, 0.03)')
                        .attr('stroke-width', 1);
                }}
                
                // Create vertical grid lines
                for (let x = 0; x < width; x += gridSize) {{
                    grid.append('line')
                        .attr('x1', x)
                        .attr('y1', 0)
                        .attr('x2', x)
                        .attr('y2', height)
                        .attr('stroke', 'rgba(255, 255, 255, 0.03)')
                        .attr('stroke-width', 1);
                }}
                
                // Set up link and node groups
                const linkGroup = svg.append('g').attr('class', 'links');
                const nodeGroup = svg.append('g').attr('class', 'nodes');
                const labelGroup = svg.append('g').attr('class', 'node-labels');
                
                // Count connections for each node
                const connectionCounts = {{}};
                networkData.links.forEach(link => {{
                    if (!connectionCounts[link.source]) connectionCounts[link.source] = 0;
                    if (!connectionCounts[link.target]) connectionCounts[link.target] = 0;
                    connectionCounts[link.source]++;
                    connectionCounts[link.target]++;
                }});
                
                // Add connection count to nodes
                networkData.nodes.forEach(node => {{
                    node.connections = connectionCounts[node.id] || 0;
                }});
                
                // Create force simulation
                const simulation = d3.forceSimulation(networkData.nodes)
                    .force('link', d3.forceLink(networkData.links)
                        .id(d => d.id)
                        .distance(d => 100 - (d.value * 30))) // Links with higher value pull nodes closer
                    .force('charge', d3.forceManyBody().strength(-200))
                    .force('center', d3.forceCenter(width / 2, height / 2))
                    .force('collision', d3.forceCollide().radius(d => d.size + 5));
                
                // Create links
                const link = linkGroup.selectAll('line')
                    .data(networkData.links)
                    .enter().append('line')
                    .attr('stroke', d => linkColorScale(d.value))
                    .attr('stroke-width', d => 1 + d.value * 2);
                
                // Create nodes
                const node = nodeGroup.selectAll('circle')
                    .data(networkData.nodes)
                    .enter().append('circle')
                    .attr('r', d => d.size)
                    .attr('fill', d => stateColors[d.state] || '#4299e1')
                    .attr('stroke', 'white')
                    .attr('stroke-width', 1.5)
                    .call(d3.drag()
                        .on('start', dragStarted)
                        .on('drag', dragging)
                        .on('end', dragEnded));
                
                // Add drop shadow effect to nodes
                const defs = svg.append('defs');
                const filter = defs.append('filter')
                    .attr('id', 'drop-shadow')
                    .attr('x', '-50%')
                    .attr('y', '-50%')
                    .attr('width', '200%')
                    .attr('height', '200%');
                
                filter.append('feGaussianBlur')
                    .attr('in', 'SourceAlpha')
                    .attr('stdDeviation', 3);
                
                filter.append('feOffset')
                    .attr('dx', 0)
                    .attr('dy', 0);
                
                filter.append('feComponentTransfer')
                    .append('feFuncA')
                    .attr('type', 'linear')
                    .attr('slope', 0.2);
                
                const feMerge = filter.append('feMerge');
                feMerge.append('feMergeNode');
                feMerge.append('feMergeNode').attr('in', 'SourceGraphic');
                
                node.style('filter', 'url(#drop-shadow)');
                
                // Add labels
                const label = labelGroup.selectAll('text')
                    .data(networkData.nodes)
                    .enter().append('text')
                    .text(d => d.title.length > 12 ? d.title.substring(0, 10) + '...' : d.title)
                    .attr('dy', -15)
                    .attr('text-anchor', 'middle')
                    .style('font-size', '10px')
                    .style('fill', 'white')
                    .style('pointer-events', 'none')
                    .style('text-shadow', '0 1px 3px rgba(0,0,0,0.8)');
                
                // Add interaction to nodes
                node.on('mouseover', function(event, d) {{
                    // Highlight the node
                    d3.select(this)
                        .transition()
                        .duration(300)
                        .attr('r', d.size * 1.2);
                    
                    // Show tooltip
                    showTooltip(event, d);
                    
                    // Highlight related links and nodes
                    const connectedNodes = new Set();
                    link.style('stroke-opacity', l => {{
                        if (l.source.id === d.id || l.target.id === d.id) {{
                            if (l.source.id === d.id) connectedNodes.add(l.target.id);
                            else connectedNodes.add(l.source.id);
                            return 1;
                        }}
                        return 0.1;
                    }});
                    
                    node.style('opacity', n => {{
                        return n.id === d.id || connectedNodes.has(n.id) ? 1 : 0.2;
                    }});
                    
                    label.style('opacity', n => {{
                        return n.id === d.id || connectedNodes.has(n.id) ? 1 : 0.2;
                    }});
                }})
                .on('mousemove', moveTooltip)
                .on('mouseout', function(event, d) {{
                    // Reset node size
                    d3.select(this)
                        .transition()
                        .duration(300)
                        .attr('r', d.size);
                    
                    // Hide tooltip
                    hideTooltip();
                    
                    // Reset link and node opacity
                    link.style('stroke-opacity', 0.6);
                    node.style('opacity', 1);
                    label.style('opacity', 1);
                }})
                .on('click', function(event, d) {{
                    // Highlight the node with a pulsing effect
                    const highlightOverlay = document.getElementById('highlight-overlay');
                    highlightOverlay.innerHTML = '';
                    
                    const rect = this.getBoundingClientRect();
                    const containerRect = document.querySelector('.network-container').getBoundingClientRect();
                    
                    const pulseDiv = document.createElement('div');
                    pulseDiv.style.position = 'absolute';
                    pulseDiv.style.width = '30px';
                    pulseDiv.style.height = '30px';
                    pulseDiv.style.borderRadius = '50%';
                    pulseDiv.style.background = 'radial-gradient(circle, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0) 70%)';
                    pulseDiv.style.left = (rect.left - containerRect.left + rect.width/2 - 15) + 'px';
                    pulseDiv.style.top = (rect.top - containerRect.top + rect.height/2 - 15) + 'px';
                    pulseDiv.style.animation = 'pulse-animation 2s infinite';
                    
                    highlightOverlay.appendChild(pulseDiv);
                }});
                
                // Update positions on tick
                simulation.on('tick', () => {{
                    // Update link positions
                    link
                        .attr('x1', d => d.source.x)
                        .attr('y1', d => d.source.y)
                        .attr('x2', d => d.target.x)
                        .attr('y2', d => d.target.y);
                    
                    // Update node positions (constrained to view)
                    node
                        .attr('cx', d => d.x = Math.max(d.size, Math.min(width - d.size, d.x)))
                        .attr('cy', d => d.y = Math.max(d.size, Math.min(height - d.size, d.y)));
                    
                    // Update label positions
                    label
                        .attr('x', d => d.x)
                        .attr('y', d => d.y);
                }});
                
                // Drag functions
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
            }}
            
            // Initialize everything when window loads
            window.addEventListener('load', () => {{
                createParticles();
                createNetwork();
            }});
        </script>
    </body>
    </html>
    """
    
    # Render the HTML
    components.html(html, height=height, width=width)
    
    return None

def animated_entanglement_network(tasks, height=600, width=None):
    """Wrapper for quantum entanglement network visualization"""
    return quantum_entanglement_network(tasks, height, width)