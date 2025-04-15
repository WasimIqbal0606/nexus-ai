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
    node_map = {}
    
    # Create nodes
    for i, task in enumerate(tasks):
        task_id = task.get('id')
        entropy = task.get('entropy', 0.5)
        state = task.get('state', 'PENDING')
        
        # Map state to color
        color_map = {
            'PENDING': '#4299e1',     # Blue
            'IN_PROGRESS': '#f6ad55', # Orange
            'COMPLETED': '#68d391',   # Green
            'BLOCKED': '#fc8181'      # Red
        }
        
        # Calculate node size based on entropy and priority
        size = 10 + (entropy * 15) + (task.get('priority', 1) * 3)
        
        node = {
            'id': task_id,
            'name': task.get('title', f'Task {i+1}'),
            'group': list(color_map.keys()).index(state) if state in color_map else 0,
            'size': size,
            'color': color_map.get(state, '#4299e1'),
            'entropy': entropy,
            'state': state,
            'priority': task.get('priority', 1)
        }
        
        nodes.append(node)
        node_map[task_id] = len(nodes) - 1
    
    # Create links from entangled_tasks
    for task in tasks:
        source_id = task.get('id')
        entangled_tasks = task.get('entangled_tasks', [])
        
        for target_id in entangled_tasks:
            # Only add link if both nodes exist and avoid duplicates
            if source_id in node_map and target_id in node_map:
                source_idx = node_map[source_id]
                target_idx = node_map[target_id]
                
                # Check if this link already exists (in either direction)
                link_exists = False
                for link in links:
                    if (link['source'] == source_idx and link['target'] == target_idx) or \
                       (link['source'] == target_idx and link['target'] == source_idx):
                        link_exists = True
                        break
                
                if not link_exists:
                    # Find the entanglement strength if available
                    strength = 0.8  # Default strength
                    
                    # Calculate link width based on entanglement strength
                    width = 1 + (strength * 4)
                    
                    link = {
                        'source': source_idx,
                        'target': target_idx,
                        'value': strength,
                        'width': width
                    }
                    links.append(link)
    
    return {'nodes': nodes, 'links': links}

def quantum_entanglement_network(tasks, height=600, width=None):
    """Create an enhanced entanglement network visualization with D3"""
    
    # Generate network data
    network_data = generate_network_data(tasks)
    
    # Convert to JSON
    network_json = json.dumps(network_data)
    
    # Custom CSS and D3 visualization
    custom_css = """
    <style>
        .node {
            transition: r 0.3s ease-in-out, stroke-width 0.3s ease;
        }
        .node:hover {
            stroke-width: 3px;
            filter: brightness(1.2) drop-shadow(0 0 5px rgba(255, 255, 255, 0.7));
        }
        .link {
            transition: stroke-width 0.3s ease, stroke-opacity 0.3s ease;
        }
        .link:hover {
            stroke-opacity: 1;
        }
        .node-label {
            font-family: sans-serif;
            pointer-events: none;
            text-shadow: 0 1px 2px rgba(0,0,0,0.8), 0 0 5px rgba(0,0,0,0.6);
        }
        .quantum-glow {
            filter: drop-shadow(0 0 3px rgba(79, 209, 234, 0.8));
            animation: pulse 4s infinite alternate;
        }
        .entanglement-line {
            stroke-dasharray: 5;
            animation: dash 20s linear infinite;
        }
        @keyframes pulse {
            0% { filter: drop-shadow(0 0 2px rgba(79, 209, 234, 0.5)); }
            50% { filter: drop-shadow(0 0 6px rgba(79, 209, 234, 0.9)); }
            100% { filter: drop-shadow(0 0 3px rgba(79, 209, 234, 0.5)); }
        }
        @keyframes dash {
            to { stroke-dashoffset: -1000; }
        }
        .node-tooltip {
            position: absolute;
            background: rgba(30, 41, 59, 0.9);
            border: 1px solid #4299e1;
            border-radius: 6px;
            padding: 10px;
            color: white;
            font-family: sans-serif;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(4px);
        }
        .node-tooltip h4 {
            margin: 0 0 5px 0;
            font-size: 14px;
            color: #4299e1;
        }
        .node-tooltip p {
            margin: 3px 0;
        }
        .node-tooltip .entropy-bar {
            height: 4px;
            width: 100%;
            background: #2c3e50;
            margin-top: 5px;
            border-radius: 2px;
            overflow: hidden;
        }
        .node-tooltip .entropy-fill {
            height: 100%;
            background: linear-gradient(90deg, #4338CA, #3B82F6);
        }
        .quantum-particles {
            position: absolute;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 0;
        }
        .particle {
            position: absolute;
            width: 2px;
            height: 2px;
            border-radius: 50%;
            background: #4299e1;
            opacity: 0.7;
            animation: float 8s infinite linear;
        }
        @keyframes float {
            0% {
                transform: translateY(0) translateX(0);
                opacity: 0;
            }
            10% {
                opacity: 0.7;
            }
            90% {
                opacity: 0.7;
            }
            100% {
                transform: translateY(-100px) translateX(50px);
                opacity: 0;
            }
        }
    </style>
    """
    
    # D3.js version 7
    d3_src = "https://d3js.org/d3.v7.min.js"
    
    # HTML template with D3 visualization
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="{d3_src}"></script>
        {custom_css}
    </head>
    <body>
        <div id="quantum-network" style="width:100%; height:{height}px; position:relative; overflow:hidden; background: linear-gradient(to bottom, #0f172a, #1e293b);">
            <div class="quantum-particles" id="particles"></div>
            <div id="tooltip" class="node-tooltip"></div>
            <svg width="100%" height="100%"></svg>
        </div>
        
        <script>
        // Network data from Python
        const graph = {network_json};
        
        // Set up the D3 visualization
        const svg = d3.select("#quantum-network svg");
        const width = svg.node().getBoundingClientRect().width;
        const height = {height};
        
        // Create quantum particle effect
        function createParticles() {{
            const particlesContainer = document.getElementById('particles');
            for (let i = 0; i < 50; i++) {{
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * width + 'px';
                particle.style.top = Math.random() * height + 'px';
                particle.style.animationDelay = (Math.random() * 8) + 's';
                particle.style.animationDuration = (Math.random() * 5 + 8) + 's';
                particlesContainer.appendChild(particle);
            }}
        }}
        createParticles();
        
        // Create a tooltip
        const tooltip = d3.select("#tooltip");
        
        // Set up the force simulation
        const simulation = d3.forceSimulation(graph.nodes)
            .force("link", d3.forceLink(graph.links).id(d => d.id).distance(120))
            .force("charge", d3.forceManyBody().strength(-400))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collide", d3.forceCollide().radius(d => d.size + 5).iterations(3));
        
        // Create the links
        const link = svg.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(graph.links)
            .enter().append("line")
            .attr("class", "link entanglement-line")
            .attr("stroke", "#4299e1")
            .attr("stroke-width", d => d.width || 1)
            .attr("stroke-opacity", 0.6)
            .attr("stroke-linecap", "round");
        
        // Add pulsing entanglement orbs in the middle of each link
        const linkOrbs = svg.append("g")
            .attr("class", "link-orbs")
            .selectAll("circle")
            .data(graph.links)
            .enter().append("circle")
            .attr("r", 3)
            .attr("class", "quantum-glow")
            .attr("fill", "rgba(79, 209, 234, 0.8)");
            
        // Create the nodes
        const node = svg.append("g")
            .attr("class", "nodes")
            .selectAll("circle")
            .data(graph.nodes)
            .enter().append("circle")
            .attr("class", "node quantum-glow")
            .attr("r", d => d.size)
            .attr("fill", d => d.color)
            .attr("stroke", "#ffffff")
            .attr("stroke-width", 1.5)
            .attr("cursor", "pointer")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        // Add labels to the nodes
        const label = svg.append("g")
            .attr("class", "labels")
            .selectAll("text")
            .data(graph.nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .attr("dy", 4)
            .attr("text-anchor", "middle")
            .attr("fill", "white")
            .text(d => d.name.length > 15 ? d.name.substring(0, 15) + "..." : d.name)
            .attr("font-size", d => 10 + (d.size / 10));
        
        // Add tooltips
        node.on("mouseover", function(event, d) {{
            // Show tooltip
            tooltip.style("opacity", 1)
                .html(`
                    <h4>${{d.name}}</h4>
                    <p>State: <span style="color:${{d.color}}">${{d.state}}</span></p>
                    <p>Priority: ${{d.priority}}/5</p>
                    <p>Entropy: ${{d.entropy.toFixed(2)}}</p>
                    <div class="entropy-bar">
                        <div class="entropy-fill" style="width: ${{d.entropy * 100}}%"></div>
                    </div>
                `)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
            
            // Highlight connected links and nodes
            const connectedNodeIds = new Set();
            link.style("stroke-opacity", l => {{
                if (l.source.id === d.id || l.target.id === d.id) {{
                    if (l.source.id === d.id) connectedNodeIds.add(l.target.id);
                    if (l.target.id === d.id) connectedNodeIds.add(l.source.id);
                    return 1;
                }} else {{
                    return 0.1;
                }}
            }})
            .style("stroke-width", l => {{
                if (l.source.id === d.id || l.target.id === d.id) {{
                    return (l.width || 1) * 2;
                }} else {{
                    return l.width || 1;
                }}
            }});
            
            linkOrbs.style("opacity", l => {{
                return (l.source.id === d.id || l.target.id === d.id) ? 1 : 0.1;
            }});
            
            node.style("opacity", n => {{
                return n.id === d.id || connectedNodeIds.has(n.id) ? 1 : 0.3;
            }});
            
            label.style("opacity", n => {{
                return n.id === d.id || connectedNodeIds.has(n.id) ? 1 : 0.3;
            }});
            
            // Animate the highlighted node
            d3.select(this)
                .transition()
                .duration(300)
                .attr("r", d.size * 1.2);
        }})
        .on("mouseout", function(event, d) {{
            // Hide tooltip
            tooltip.style("opacity", 0);
            
            // Reset highlights
            link.style("stroke-opacity", 0.6)
                .style("stroke-width", l => l.width || 1);
                
            linkOrbs.style("opacity", 1);
            
            node.style("opacity", 1);
            label.style("opacity", 1);
            
            // Reset the node size
            d3.select(this)
                .transition()
                .duration(300)
                .attr("r", d.size);
        }});
        
        // Set up simulation tick handler
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            linkOrbs
                .attr("cx", d => (d.source.x + d.target.x) / 2)
                .attr("cy", d => (d.source.y + d.target.y) / 2);
            
            node
                .attr("cx", d => d.x = Math.max(d.size, Math.min(width - d.size, d.x)))
                .attr("cy", d => d.y = Math.max(d.size, Math.min(height - d.size, d.y)));
            
            label
                .attr("x", d => d.x)
                .attr("y", d => d.y - d.size - 5);
        }});
        
        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
        
        // Add zoom capabilities
        const zoom = d3.zoom()
            .scaleExtent([0.5, 3])
            .on("zoom", (event) => {{
                svg.selectAll("g").attr("transform", event.transform);
            }});
        
        svg.call(zoom);
        
        // Create animated wave effect in the background
        const defs = svg.append("defs");
        
        // Create gradient
        const gradient = defs.append("linearGradient")
            .attr("id", "wave-gradient")
            .attr("x1", "0%")
            .attr("y1", "0%")
            .attr("x2", "100%")
            .attr("y2", "0%");
            
        gradient.append("stop")
            .attr("offset", "0%")
            .attr("stop-color", "rgba(66, 153, 225, 0.1)");
            
        gradient.append("stop")
            .attr("offset", "50%")
            .attr("stop-color", "rgba(66, 153, 225, 0.3)");
            
        gradient.append("stop")
            .attr("offset", "100%")
            .attr("stop-color", "rgba(66, 153, 225, 0.1)");
        
        // Add waves
        for (let i = 0; i < 3; i++) {{
            const freq = 0.5 + (i * 0.15);
            const amplitude = 20 - (i * 5);
            const speed = 0.1 - (i * 0.02);
            
            defs.append("path")
                .attr("id", `wave-path-${{i}}`)
                .attr("d", createWavePath(width, height, freq, amplitude))
                .attr("fill", "none");
            
            // Animate the wave
            const waveAnimation = defs.append("animate")
                .attr("xlink:href", `#wave-path-${{i}}`)
                .attr("attributeName", "d")
                .attr("dur", `${{15 + i * 5}}s`)
                .attr("repeatCount", "indefinite");
            
            // Set animation values with JavaScript
            setInterval(() => {{
                const t = Date.now() * speed;
                waveAnimation.attr("values", createWavePath(width, height, freq, amplitude, t) +
                                            ";" +
                                            createWavePath(width, height, freq, amplitude, t + Math.PI) +
                                            ";" +
                                            createWavePath(width, height, freq, amplitude, t + Math.PI * 2));
            }}, 50);
            
            // Create wave visual
            svg.append("use")
                .attr("href", `#wave-path-${{i}}`)
                .attr("stroke", "url(#wave-gradient)")
                .attr("stroke-width", 1.5)
                .attr("opacity", 0.5 - (i * 0.1))
                .attr("transform", `translate(0, ${{height - 50 + (i * 30)}})`);
        }}
        
        // Function to create wave path
        function createWavePath(width, height, frequency, amplitude, offset = 0) {{
            let path = `M0,${{height / 2}}`;
            
            for (let x = 0; x <= width; x += 10) {{
                const y = Math.sin((x * frequency + offset) / 50) * amplitude;
                path += ` L${{x}},${{height / 2 + y}}`;
            }}
            
            return path;
        }}
        
        </script>
    </body>
    </html>
    """
    
    # Display the visualization using Streamlit's HTML component
    components.html(html_template, height=height, width=width or "100%", scrolling=False)
    
    return None

def animated_entanglement_network(tasks, height=600):
    """Wrapper for quantum entanglement network visualization"""
    if not tasks:
        st.info("No tasks available to visualize entanglement network.")
        return None
    
    st.markdown("<h3 style='text-align: center;'>Quantum Task Entanglement Network</h3>", unsafe_allow_html=True)
    
    # Display legend
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="background-color: #4299e1; width: 15px; height: 15px; border-radius: 50%; margin-right: 5px;"></div>
            <span>PENDING</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="background-color: #f6ad55; width: 15px; height: 15px; border-radius: 50%; margin-right: 5px;"></div>
            <span>IN_PROGRESS</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="background-color: #68d391; width: 15px; height: 15px; border-radius: 50%; margin-right: 5px;"></div>
            <span>COMPLETED</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="background-color: #fc8181; width: 15px; height: 15px; border-radius: 50%; margin-right: 5px;"></div>
            <span>BLOCKED</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.5); border-radius: 8px; padding: 10px; margin-bottom: 15px; font-size: 0.9em;">
        <p style="margin-bottom: 5px;"><strong>Network Information:</strong></p>
        <ul style="margin: 0; padding-left: 20px;">
            <li>Node size represents task entropy (uncertainty) and priority</li>
            <li>Lines represent quantum entanglements between tasks</li>
            <li>Hover over nodes for detailed information</li>
            <li>Drag nodes to reorganize the network</li>
            <li>Use mouse wheel to zoom in/out</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Render the network visualization
    return quantum_entanglement_network(tasks, height=height)