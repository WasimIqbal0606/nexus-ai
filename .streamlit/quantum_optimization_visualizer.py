import streamlit as st
import streamlit.components.v1 as components
import json
import random
import math

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
    
    # Create HTML visualization with JavaScript animations
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/gsap@3.9.1/dist/gsap.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: sans-serif;
                background: transparent;
                color: #f1f5f9;
                overflow: hidden;
            }}
            #optimization-container {{
                width: 100%;
                height: {height}px;
                position: relative;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            }}
            .optimization-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 20px;
                border-bottom: 1px solid rgba(100, 116, 139, 0.2);
            }}
            .score-display {{
                background: rgba(30, 41, 59, 0.7);
                border-radius: 8px;
                padding: 15px;
                width: 180px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(100, 116, 139, 0.2);
                text-align: center;
            }}
            .score-value {{
                font-size: 32px;
                font-weight: 600;
                color: #3b82f6;
                line-height: 1;
                margin: 10px 0;
                background: linear-gradient(to right, #4338CA, #3B82F6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                position: relative;
            }}
            .score-value::after {{
                content: "";
                position: absolute;
                top: -10px;
                left: -10px;
                right: -10px;
                bottom: -10px;
                background: radial-gradient(circle, rgba(59, 130, 246, 0.4) 0%, rgba(59, 130, 246, 0) 70%);
                opacity: 0.5;
                border-radius: 50%;
                z-index: -1;
                animation: pulse 2s infinite ease-in-out;
            }}
            @keyframes pulse {{
                0% {{ transform: scale(0.95); opacity: 0.5; }}
                50% {{ transform: scale(1.05); opacity: 0.8; }}
                100% {{ transform: scale(0.95); opacity: 0.5; }}
            }}
            .score-label {{
                font-size: 14px;
                color: #94a3b8;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .metrics-bar {{
                display: flex;
                align-items: center;
                gap: 20px;
            }}
            .metric-item {{
                background: rgba(30, 41, 59, 0.7);
                border-radius: 8px;
                padding: 10px 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(100, 116, 139, 0.2);
                text-align: center;
                min-width: 100px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: 600;
                color: #e2e8f0;
                margin-bottom: 5px;
            }}
            .metric-label {{
                font-size: 12px;
                color: #94a3b8;
            }}
            .main-visualization {{
                display: flex;
                position: absolute;
                top: 100px;
                left: 0;
                right: 0;
                bottom: 0;
                padding: 20px;
            }}
            .controls-panel {{
                width: 220px;
                background: rgba(30, 41, 59, 0.7);
                border-radius: 8px;
                padding: 15px;
                margin-right: 20px;
                backdrop-filter: blur(4px);
                border: 1px solid rgba(100, 116, 139, 0.2);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                display: flex;
                flex-direction: column;
            }}
            .viz-space {{
                flex: 1;
                display: flex;
                flex-direction: column;
                gap: 20px;
            }}
            .viz-row {{
                display: flex;
                flex: 1;
                gap: 20px;
            }}
            .viz-panel {{
                flex: 1;
                background: rgba(30, 41, 59, 0.7);
                border-radius: 8px;
                padding: 15px;
                backdrop-filter: blur(4px);
                border: 1px solid rgba(100, 116, 139, 0.2);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                position: relative;
                overflow: hidden;
            }}
            .viz-title {{
                font-size: 16px;
                font-weight: 600;
                color: #e2e8f0;
                margin-bottom: 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .viz-title .info-icon {{
                font-size: 14px;
                color: #94a3b8;
                cursor: pointer;
                transition: color 0.2s;
            }}
            .viz-title .info-icon:hover {{
                color: #3b82f6;
            }}
            .control-btn {{
                background: rgba(59, 130, 246, 0.2);
                border: 1px solid rgba(59, 130, 246, 0.5);
                color: #e2e8f0;
                border-radius: 6px;
                padding: 8px 12px;
                cursor: pointer;
                transition: all 0.2s;
                font-size: 14px;
                margin-bottom: 10px;
                text-align: center;
            }}
            .control-btn:hover {{
                background: rgba(59, 130, 246, 0.4);
            }}
            .control-btn:active {{
                transform: scale(0.98);
            }}
            .control-section {{
                margin-bottom: 20px;
            }}
            .control-section-title {{
                font-size: 14px;
                font-weight: 600;
                color: #e2e8f0;
                margin: 0 0 10px 0;
                padding-bottom: 5px;
                border-bottom: 1px solid rgba(100, 116, 139, 0.2);
            }}
            .control-item {{
                margin-bottom: 15px;
            }}
            .control-label {{
                font-size: 12px;
                color: #94a3b8;
                margin-bottom: 5px;
            }}
            .selector {{
                width: 100%;
                background: rgba(15, 23, 42, 0.8);
                border: 1px solid rgba(100, 116, 139, 0.2);
                color: #e2e8f0;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
                cursor: pointer;
            }}
            .slider-container {{
                display: flex;
                flex-direction: column;
            }}
            .slider-value {{
                font-size: 12px;
                color: #94a3b8;
                text-align: right;
                margin-bottom: 5px;
            }}
            .slider {{
                -webkit-appearance: none;
                width: 100%;
                height: 6px;
                border-radius: 3px;  
                background: rgba(15, 23, 42, 0.8);
                outline: none;
                transition: opacity 0.2s;
            }}
            .slider::-webkit-slider-thumb {{
                -webkit-appearance: none;
                appearance: none;
                width: 16px;
                height: 16px;
                border-radius: 50%; 
                background: #3b82f6;
                cursor: pointer;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
            }}
            .slider::-moz-range-thumb {{
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background: #3b82f6;
                cursor: pointer;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
            }}
            /* Assignment visualization */
            .workload-item {{
                margin-bottom: 15px;
                padding-bottom: 15px;
                border-bottom: 1px solid rgba(100, 116, 139, 0.1);
            }}
            .workload-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 5px;
            }}
            .assignee {{
                font-size: 14px;
                font-weight: 600;
                color: #e2e8f0;
                display: flex;
                align-items: center;
            }}
            .avatar {{
                width: 24px;
                height: 24px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                color: white;
                font-size: 12px;
                margin-right: 8px;
            }}
            .task-count {{
                font-size: 12px;
                color: #94a3b8;
                background: rgba(15, 23, 42, 0.8);
                padding: 2px 8px;
                border-radius: 10px;
            }}
            .workload-bar {{
                height: 6px;
                background: rgba(15, 23, 42, 0.8);
                border-radius: 3px;
                overflow: hidden;
                margin-bottom: 5px;
            }}
            .workload-fill {{
                height: 100%;
                background: linear-gradient(to right, #4338CA, #3B82F6);
                width: 0%;
                border-radius: 3px;
                position: relative;
                box-shadow: 0 0 5px rgba(59, 130, 246, 0.5);
                opacity: 0;
                transform: translateX(-100%);
                animation: none;
            }}
            @keyframes fill-animation {{
                0% {{ transform: translateX(-100%); opacity: 0; }}
                100% {{ transform: translateX(0); opacity: 1; }}
            }}
            .workload-stats {{
                display: flex;
                justify-content: space-between;
                font-size: 12px;
                color: #94a3b8;
            }}
            .stats-item {{
                display: flex;
                align-items: center;
                gap: 5px;
            }}
            .stats-icon {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
            }}
            .energy-landscape {{
                width: 100%;
                height: 100%;
                position: relative;
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            .energy-canvas {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
            }}
            .convergence-path {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 1;
            }}
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
            /* Network visualization */
            .links line {{
                stroke-opacity: 0.6;
            }}
            .nodes circle {{
                stroke: #fff;
                stroke-width: 1.5px;
                transition: all 0.3s ease;
            }}
            .task-item {{
                display: flex;
                padding: 8px;
                margin-bottom: 8px;
                background: rgba(15, 23, 42, 0.8);
                border-radius: 6px;
                border-left: 3px solid #3B82F6;
                cursor: pointer;
                transition: all 0.2s;
            }}
            .task-item:hover {{
                transform: translateX(5px);
                background: rgba(30, 41, 59, 0.9);
            }}
            .task-item.selected {{
                background: rgba(59, 130, 246, 0.2);
                transform: translateX(5px);
            }}
            .task-icon {{
                width: 24px;
                height: 24px;
                border-radius: 4px;
                margin-right: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                color: white;
            }}
            .task-details {{
                flex: 1;
            }}
            .task-title {{
                font-size: 12px;
                font-weight: 600;
                color: #e2e8f0;
                margin-bottom: 2px;
            }}
            .task-meta {{
                font-size: 10px;
                color: #94a3b8;
                display: flex;
                align-items: center;
                gap: 5px;
            }}
            .task-meta-item {{
                padding: 1px 5px;
                background: rgba(15, 23, 42, 0.8);
                border-radius: 3px;
            }}
            .task-actions {{
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                font-size: 10px;
            }}
            .task-assignee {{
                background: rgba(59, 130, 246, 0.2);
                color: #e2e8f0;
                padding: 1px 5px;
                border-radius: 3px;
                text-align: center;
                margin-top: 2px;
                font-size: 10px;
            }}
            /* Animations */
            .quantum-particles {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 0;
                overflow: hidden;
            }}
            .particle {{
                position: absolute;
                border-radius: 50%;
                opacity: 0.7;
                pointer-events: none;
            }}
            @keyframes floating {{
                0% {{ transform: translate(0, 0) rotate(0deg); opacity: 0.2; }}
                25% {{ opacity: 0.5; }}
                50% {{ transform: translate(0, -15px) rotate(180deg); opacity: 0.7; }}
                75% {{ opacity: 0.5; }}
                100% {{ transform: translate(0, 0) rotate(360deg); opacity: 0.2; }}
            }}
            /* Pulse Effects */
            .pulse-ring {{
                position: absolute;
                border-radius: 50%;
                animation: pulse-animation 3s infinite ease-out;
            }}
            @keyframes pulse-animation {{
                0% {{ transform: scale(0.1); opacity: 0; }}
                50% {{ opacity: 0.3; }}
                100% {{ transform: scale(2); opacity: 0; }}
            }}
            .energy-node {{
                position: absolute;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: #3B82F6;
                transform: translate(-50%, -50%);
                z-index: 2;
                box-shadow: 0 0 5px rgba(59, 130, 246, 0.8);
            }}
            .energy-label {{
                position: absolute;
                font-size: 10px;
                color: #e2e8f0;
                transform: translate(-50%, -150%);
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8);
                pointer-events: none;
                z-index: 3;
            }}
            .entanglement-line {{
                stroke-dasharray: 5;
                animation: dash 30s linear infinite;
            }}
            @keyframes dash {{
                to {{ stroke-dashoffset: -1000; }}
            }}
            /* Animated wave */
            .wave {{
                position: absolute;
                width: 100%;
                height: 5px;
                background: linear-gradient(to right, rgba(59, 130, 246, 0), rgba(59, 130, 246, 0.5), rgba(59, 130, 246, 0));
                opacity: 0.5;
                border-radius: 50%;
                transform-origin: 50% 50%;
                transform: scaleX(0.1);
                animation: wave 3s infinite ease-out;
            }}
            @keyframes wave {{
                0% {{ transform: scaleX(0.1); opacity: 0; }}
                50% {{ opacity: 0.5; }}
                100% {{ transform: scaleX(3); opacity: 0; }}
            }}
            /* Loading spinner */
            .spinner {{
                width: 40px;
                height: 40px;
                position: absolute;
                top: 50%;
                left: 50%;
                margin-top: -20px;
                margin-left: -20px;
                border-radius: 50%;
                border: 3px solid rgba(59, 130, 246, 0.1);
                border-top-color: #3B82F6;
                animation: spin 1s infinite linear;
            }}
            @keyframes spin {{
                100% {{ transform: rotate(360deg); }}
            }}
            /* GSAP Timeline */
            .timeline-marker {{
                position: absolute;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #3B82F6;
                transform: translate(-50%, -50%);
                z-index: 3;
            }}
            /* Label animation */
            @keyframes labelPop {{
                0% {{ transform: translateY(10px); opacity: 0; }}
                100% {{ transform: translateY(0); opacity: 1; }}
            }}
            /* Glass panel effect */
            .glass-panel {{
                backdrop-filter: blur(4px);
                -webkit-backdrop-filter: blur(4px);
                background: rgba(15, 23, 42, 0.6);
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                padding: 15px;
            }}
        </style>
    </head>
    <body>
        <div id="optimization-container">
            <!-- Quantum particles background -->
            <div class="quantum-particles" id="particles"></div>
            
            <!-- Main header -->
            <div class="optimization-header">
                <div class="score-display">
                    <div class="score-label">Optimization Score</div>
                    <div class="score-value" id="score-value">0.00</div>
                </div>
                
                <div class="metrics-bar">
                    <div class="metric-item">
                        <div class="metric-value" id="task-count">0</div>
                        <div class="metric-label">Tasks Optimized</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="iteration-count">0</div>
                        <div class="metric-label">Iterations</div>
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
            
            <!-- Main visualization area -->
            <div class="main-visualization">
                <!-- Controls panel -->
                <div class="controls-panel">
                    <!-- Control section: Animation -->
                    <div class="control-section">
                        <div class="control-section-title">Animation Controls</div>
                        <div class="control-btn" id="play-annealing">Run Quantum Annealing</div>
                        <div class="control-btn" id="reset-annealing">Reset Simulation</div>
                    </div>
                    
                    <!-- Control section: Display Options -->
                    <div class="control-section">
                        <div class="control-section-title">Display Options</div>
                        
                        <div class="control-item">
                            <div class="control-label">View Mode</div>
                            <select class="selector" id="view-mode">
                                <option value="3d">3D Energy Landscape</option>
                                <option value="2d">2D Contour Map</option>
                                <option value="network">Network View</option>
                            </select>
                        </div>
                        
                        <div class="control-item">
                            <div class="control-label">Color Scheme</div>
                            <select class="selector" id="color-scheme">
                                <option value="quantum">Quantum Gradient</option>
                                <option value="heat">Heat Map</option>
                                <option value="spectral">Spectral</option>
                            </select>
                        </div>
                        
                        <div class="control-item slider-container">
                            <div class="control-label">Animation Speed</div>
                            <div class="slider-value" id="speed-value">1.0x</div>
                            <input type="range" min="0.5" max="2" value="1" step="0.1" class="slider" id="speed-slider">
                        </div>
                        
                        <div class="control-item slider-container">
                            <div class="control-label">Detail Level</div>
                            <div class="slider-value" id="detail-value">Medium</div>
                            <input type="range" min="1" max="3" value="2" step="1" class="slider" id="detail-slider">
                        </div>
                    </div>
                    
                    <!-- Control section: Algorithm Parameters -->
                    <div class="control-section">
                        <div class="control-section-title">Quantum Parameters</div>
                        
                        <div class="control-item slider-container">
                            <div class="control-label">Temperature</div>
                            <div class="slider-value" id="temp-value">1.0</div>
                            <input type="range" min="0.1" max="2" value="1" step="0.1" class="slider" id="temp-slider">
                        </div>
                        
                        <div class="control-item slider-container">
                            <div class="control-label">Annealing Steps</div>
                            <div class="slider-value" id="steps-value">1000</div>
                            <input type="range" min="100" max="2000" value="1000" step="100" class="slider" id="steps-slider">
                        </div>
                    </div>
                </div>
                
                <!-- Visualization space -->
                <div class="viz-space">
                    <!-- Top row: Assignment and Energy Landscape -->
                    <div class="viz-row">
                        <!-- Assignment visualization -->
                        <div class="viz-panel">
                            <div class="viz-title">
                                Optimized Task Assignments
                                <span class="info-icon" title="Shows the optimized task assignments for each team member">ⓘ</span>
                            </div>
                            <div id="assignment-viz" style="height: 100%; overflow-y: auto;"></div>
                        </div>
                        
                        <!-- Energy landscape visualization -->
                        <div class="viz-panel">
                            <div class="viz-title">
                                Quantum Energy Landscape
                                <span class="info-icon" title="Shows the optimization energy landscape with local minima">ⓘ</span>
                            </div>
                            <div class="energy-landscape" id="energy-landscape">
                                <canvas class="energy-canvas" id="energy-canvas"></canvas>
                                <svg class="convergence-path" id="convergence-path"></svg>
                                <!-- Loading spinner shown during initialization -->
                                <div class="spinner" id="energy-spinner"></div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Bottom row: Tasks and Workload Distribution -->
                    <div class="viz-row">
                        <!-- Tasks visualization -->
                        <div class="viz-panel">
                            <div class="viz-title">
                                Entangled Task Network
                                <span class="info-icon" title="Shows the tasks and their quantum entanglements">ⓘ</span>
                            </div>
                            <div id="task-network" style="width: 100%; height: 100%;"></div>
                        </div>
                        
                        <!-- Workload distribution visualization -->
                        <div class="viz-panel">
                            <div class="viz-title">
                                Expertise Matching & Cognitive Load
                                <span class="info-icon" title="Shows the matching between team members and task requirements">ⓘ</span>
                            </div>
                            <div id="expertise-matching" style="width: 100%; height: 100%;"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Tooltip for various elements -->
            <div class="tooltip" id="tooltip"></div>
        </div>
        
        <script>
            // Parse the optimization data from Python
            const optimizationData = {optimization_json};
            
            // DOM Elements
            const scoreValue = document.getElementById('score-value');
            const taskCount = document.getElementById('task-count');
            const iterationCount = document.getElementById('iteration-count');
            const timeSaved = document.getElementById('time-saved');
            const cognitiveReduce = document.getElementById('cognitive-reduce');
            const assignmentViz = document.getElementById('assignment-viz');
            const energyLandscape = document.getElementById('energy-landscape');
            const energyCanvas = document.getElementById('energy-canvas');
            const convergencePath = document.getElementById('convergence-path');
            const energySpinner = document.getElementById('energy-spinner');
            const taskNetwork = document.getElementById('task-network');
            const expertiseMatching = document.getElementById('expertise-matching');
            const tooltip = document.getElementById('tooltip');
            const particles = document.getElementById('particles');
            
            // Control elements
            const playAnnealingBtn = document.getElementById('play-annealing');
            const resetAnnealingBtn = document.getElementById('reset-annealing');
            const viewModeSelect = document.getElementById('view-mode');
            const colorSchemeSelect = document.getElementById('color-scheme');
            const speedSlider = document.getElementById('speed-slider');
            const speedValue = document.getElementById('speed-value');
            const detailSlider = document.getElementById('detail-slider');
            const detailValue = document.getElementById('detail-value');
            const tempSlider = document.getElementById('temp-slider');
            const tempValue = document.getElementById('temp-value');
            const stepsSlider = document.getElementById('steps-slider');
            const stepsValue = document.getElementById('steps-value');
            
            // Animation state
            let isAnnealingRunning = false;
            let annealingAnimation = null;
            let currentStep = 0;
            let energyLandscapeData = null;
            let convergencePathData = null;
            let taskNetworkSimulation = null;
            
            // Colors for states, priorities, and other visual elements
            const colors = {
                states: {
                    'PENDING': '#4299e1',     // Blue
                    'IN_PROGRESS': '#f6ad55', // Orange
                    'COMPLETED': '#68d391',   // Green
                    'BLOCKED': '#fc8181'      // Red
                },
                priorities: [
                    '#94a3b8', // Priority 1 (lowest)
                    '#64748b',
                    '#475569',
                    '#334155',
                    '#1e293b'  // Priority 5 (highest)
                ],
                avatarBg: [
                    '#3b82f6', // Blue
                    '#8b5cf6', // Purple
                    '#ec4899', // Pink
                    '#f97316', // Orange
                    '#10b981'  // Green
                ]
            };
            
            // Initialize the visualization when the page loads
            window.addEventListener('load', () => {
                // Create quantum particles background
                createParticles();
                
                // Initialize various visualizations
                initializeAssignmentVisualization();
                initializeEnergyLandscape();
                initializeTaskNetwork();
                initializeExpertiseMatching();
                
                // Set initial values in UI
                updateMetricsDisplay();
                
                // Set up event listeners for controls
                setupEventListeners();
            });
            
            // Create background particles
            function createParticles() {
                particles.innerHTML = '';
                
                // Add quantum particles
                for (let i = 0; i < 50; i++) {
                    const particle = document.createElement('div');
                    particle.className = 'particle';
                    
                    // Random particle size
                    const size = Math.random() * 4 + 2;
                    particle.style.width = `${size}px`;
                    particle.style.height = `${size}px`;
                    
                    // Random position
                    particle.style.left = `${Math.random() * 100}%`;
                    particle.style.top = `${Math.random() * 100}%`;
                    
                    // Random color - quantum theme
                    const hue = 210 + Math.random() * 60; // Blue to purple range
                    particle.style.backgroundColor = `hsla(${hue}, 70%, 60%, 0.8)`;
                    
                    // Animation
                    particle.style.animation = `floating ${5 + Math.random() * 10}s infinite ease-in-out`;
                    particle.style.animationDelay = `${Math.random() * 5}s`;
                    
                    particles.appendChild(particle);
                }
                
                // Add pulse rings
                for (let i = 0; i < 3; i++) {
                    const ring = document.createElement('div');
                    ring.className = 'pulse-ring';
                    
                    // Position in random place
                    ring.style.left = `${30 + Math.random() * 40}%`;
                    ring.style.top = `${30 + Math.random() * 40}%`;
                    
                    // Size and color
                    const size = 20 + Math.random() * 30;
                    ring.style.width = `${size}px`;
                    ring.style.height = `${size}px`;
                    
                    // Random color - quantum theme
                    const hue = 210 + Math.random() * 60; // Blue to purple range
                    ring.style.border = `2px solid hsla(${hue}, 70%, 60%, 0.3)`;
                    
                    // Animation timing
                    ring.style.animationDuration = `${5 + Math.random() * 5}s`;
                    ring.style.animationDelay = `${Math.random() * 2}s`;
                    
                    particles.appendChild(ring);
                }
                
                // Add waves
                for (let i = 0; i < 2; i++) {
                    const wave = document.createElement('div');
                    wave.className = 'wave';
                    
                    // Position near middle
                    wave.style.left = `${40 + Math.random() * 20}%`;
                    wave.style.top = `${40 + Math.random() * 20}%`;
                    
                    // Animation timing
                    wave.style.animationDuration = `${6 + Math.random() * 4}s`;
                    wave.style.animationDelay = `${Math.random() * 3}s`;
                    
                    particles.appendChild(wave);
                }
            }
            
            // Initialize the assignment visualization
            function initializeAssignmentVisualization() {
                assignmentViz.innerHTML = '';
                
                // Get assignment data
                const assignments = optimizationData.assignments || {};
                const workloadDistribution = optimizationData.workload_distribution || {};
                
                // Group tasks by assignee
                const assigneeMap = {};
                
                // Build assignee map
                Object.entries(assignments).forEach(([taskId, assignee]) => {
                    if (!assigneeMap[assignee]) {
                        assigneeMap[assignee] = [];
                    }
                    assigneeMap[assignee].push(taskId);
                });
                
                // Create workload items for each assignee
                Object.entries(assigneeMap).forEach(([assignee, taskIds], index) => {
                    const workloadItem = document.createElement('div');
                    workloadItem.className = 'workload-item';
                    
                    // Get workload stats if available
                    const workloadStats = workloadDistribution[assignee] || {
                        task_count: taskIds.length,
                        total_priority: 0,
                        entropy_sum: 0,
                        cognitive_load: 0,
                        expertise_match: 0.8 + Math.random() * 0.2 // Fallback
                    };
                    
                    // Create avatar with first letter
                    const avatarColor = colors.avatarBg[index % colors.avatarBg.length];
                    const avatarLetter = assignee.charAt(0).toUpperCase();
                    
                    workloadItem.innerHTML = `
                        <div class="workload-header">
                            <div class="assignee">
                                <div class="avatar" style="background-color: ${avatarColor};">${avatarLetter}</div>
                                ${assignee}
                            </div>
                            <div class="task-count">${taskIds.length} tasks</div>
                        </div>
                        <div class="workload-bar">
                            <div class="workload-fill" style="width: ${workloadStats.expertise_match * 100}%;"></div>
                        </div>
                        <div class="workload-stats">
                            <div class="stats-item">
                                <div class="stats-icon" style="background-color: #3b82f6;"></div>
                                Expertise Match: ${Math.round(workloadStats.expertise_match * 100)}%
                            </div>
                            <div class="stats-item">
                                <div class="stats-icon" style="background-color: #f97316;"></div>
                                Load: ${workloadStats.cognitive_load.toFixed(1)}
                            </div>
                        </div>
                    `;
                    
                    assignmentViz.appendChild(workloadItem);
                    
                    // Animate the workload bar fills with GSAP
                    const workloadFill = workloadItem.querySelector('.workload-fill');
                    gsap.fromTo(workloadFill, 
                        { opacity: 0, translateX: '-100%' },
                        { 
                            opacity: 1, 
                            translateX: '0%', 
                            duration: 1, 
                            delay: index * 0.2,
                            ease: "power2.out"
                        }
                    );
                });
            }
            
            // Initialize energy landscape visualization
            function initializeEnergyLandscape() {
                // Set canvas size
                const width = energyCanvas.clientWidth;
                const height = energyCanvas.clientHeight;
                energyCanvas.width = width;
                energyCanvas.height = height;
                
                // Get context
                const ctx = energyCanvas.getContext('2d');
                
                // Generate energy landscape data
                generateEnergyLandscapeData(width, height);
                
                // Render the energy landscape
                renderEnergyLandscape(ctx, width, height);
                
                // Initialize convergence path SVG
                initializeConvergencePath(width, height);
                
                // Hide the spinner
                energySpinner.style.display = 'none';
            }
            
            // Generate energy landscape data
            function generateEnergyLandscapeData(width, height) {
                // Create a grid of energy values
                const gridSize = 50;
                const data = [];
                
                // Create some random peaks and valleys
                const peakCount = 5 + Math.floor(Math.random() * 5);
                const peaks = [];
                
                for (let i = 0; i < peakCount; i++) {
                    peaks.push({
                        x: Math.random() * width,
                        y: Math.random() * height,
                        height: Math.random() * 0.8 + 0.2, // 0.2 to 1.0
                        width: Math.random() * 50 + 50, // 50 to 100
                        type: Math.random() > 0.6 ? 'peak' : 'valley' // 60% valleys, 40% peaks
                    });
                }
                
                // Generate grid points
                for (let x = 0; x < width; x += gridSize) {
                    for (let y = 0; y < height; y += gridSize) {
                        // Calculate energy value based on distance to peaks/valleys
                        let energy = 0.5; // Base energy level
                        
                        for (const peak of peaks) {
                            const dx = x - peak.x;
                            const dy = y - peak.y;
                            const distance = Math.sqrt(dx * dx + dy * dy);
                            const influence = Math.exp(-distance / peak.width) * peak.height;
                            
                            if (peak.type === 'peak') {
                                energy += influence;
                            } else {
                                energy -= influence;
                            }
                        }
                        
                        // Clamp energy between 0 and 1
                        energy = Math.max(0, Math.min(1, energy));
                        
                        data.push({ x, y, energy });
                    }
                }
                
                // Save data for later use
                energyLandscapeData = {
                    grid: data,
                    peaks,
                    minima: peaks.filter(p => p.type === 'valley').map(p => ({ x: p.x, y: p.y }))
                };
                
                // Generate a convergence path
                generateConvergencePath(width, height);
            }
            
            // Generate a path for the optimization convergence
            function generateConvergencePath(width, height) {
                // Start from a random point
                const startX = width * 0.2 + Math.random() * width * 0.6;
                const startY = height * 0.2 + Math.random() * height * 0.6;
                
                // Find the closest minimum from the energy landscape
                let closestMinimum = null;
                let closestDistance = Infinity;
                
                for (const minimum of energyLandscapeData.minima) {
                    const dx = minimum.x - startX;
                    const dy = minimum.y - startY;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    
                    if (distance < closestDistance) {
                        closestDistance = distance;
                        closestMinimum = minimum;
                    }
                }
                
                // Generate a path with some randomness but trending toward the minimum
                const pathPoints = [];
                let currentX = startX;
                let currentY = startY;
                
                // Number of steps in the path
                const steps = 100 + Math.floor(Math.random() * 50);
                
                for (let i = 0; i < steps; i++) {
                    // Add the current point
                    pathPoints.push({ x: currentX, y: currentY });
                    
                    // Calculate progress ratio
                    const progress = i / steps;
                    
                    // As we progress, reduce randomness and increase pull toward minimum
                    const randomFactor = 1 - progress;
                    const minimumPull = progress;
                    
                    // Calculate direction toward minimum
                    const dx = closestMinimum.x - currentX;
                    const dy = closestMinimum.y - currentY;
                    
                    // Add some random movement
                    const randomX = (Math.random() * 2 - 1) * 20 * randomFactor;
                    const randomY = (Math.random() * 2 - 1) * 20 * randomFactor;
                    
                    // Calculate new position with randomness and pull toward minimum
                    currentX += dx * 0.1 * minimumPull + randomX;
                    currentY += dy * 0.1 * minimumPull + randomY;
                    
                    // Add some oscillations as we get closer
                    if (progress > 0.7) {
                        const oscillation = Math.sin(progress * 20) * 10 * (1 - progress);
                        currentX += oscillation;
                        currentY += oscillation;
                    }
                }
                
                // Save the convergence path
                convergencePathData = pathPoints;
            }
            
            // Render the energy landscape
            function renderEnergyLandscape(ctx, width, height) {
                // Clear the canvas
                ctx.clearRect(0, 0, width, height);
                
                // Color scheme based on user selection
                const colorScheme = colorSchemeSelect.value;
                
                // Draw the energy landscape
                if (viewModeSelect.value === '2d') {
                    // 2D contour map
                    renderContourMap(ctx, width, height, colorScheme);
                } else {
                    // 3D landscape (default)
                    render3DLandscape(ctx, width, height, colorScheme);
                }
            }
            
            // Render a 3D-like energy landscape
            function render3DLandscape(ctx, width, height, colorScheme) {
                // Sort grid points by y-coordinate for proper rendering
                const sortedGrid = [...energyLandscapeData.grid].sort((a, b) => a.y - b.y);
                
                // Draw each grid point as a 3D bar
                for (const point of sortedGrid) {
                    const barHeight = point.energy * 50; // Scale for visibility
                    const color = getColorForEnergy(point.energy, colorScheme);
                    
                    // Draw 3D-like bar
                    ctx.beginPath();
                    ctx.rect(point.x, point.y - barHeight, 10, barHeight);
                    ctx.fillStyle = color;
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
                    ctx.lineWidth = 1;
                    ctx.fill();
                    ctx.stroke();
                    
                    // Add highlight for 3D effect
                    ctx.beginPath();
                    ctx.rect(point.x, point.y - barHeight, 10, 2);
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
                    ctx.fill();
                }
                
                // Mark minima
                for (const minimum of energyLandscapeData.minima) {
                    ctx.beginPath();
                    ctx.arc(minimum.x, minimum.y, 8, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(59, 130, 246, 0.7)';
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 2;
                    ctx.fill();
                    ctx.stroke();
                }
            }
            
            // Render a 2D contour map
            function renderContourMap(ctx, width, height, colorScheme) {
                // Create an image data to render the contour map
                const imageData = ctx.createImageData(width, height);
                
                // Interpolate grid data to fill the entire image
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        // Find the closest grid points
                        let closestDistance = Infinity;
                        let weightedEnergy = 0;
                        let totalWeight = 0;
                        
                        for (const point of energyLandscapeData.grid) {
                            const dx = x - point.x;
                            const dy = y - point.y;
                            const distance = Math.sqrt(dx * dx + dy * dy);
                            
                            if (distance < 50) { // Influence radius
                                const weight = 1 / (1 + distance);
                                weightedEnergy += point.energy * weight;
                                totalWeight += weight;
                            }
                        }
                        
                        // Calculate interpolated energy
                        const energy = totalWeight > 0 ? weightedEnergy / totalWeight : 0.5;
                        
                        // Get color based on energy value
                        const color = getColorForEnergy(energy, colorScheme);
                        
                        // Convert hex to RGB
                        const r = parseInt(color.substring(1, 3), 16);
                        const g = parseInt(color.substring(3, 5), 16);
                        const b = parseInt(color.substring(5, 7), 16);
                        
                        // Set pixel color
                        const index = (y * width + x) * 4;
                        imageData.data[index] = r;
                        imageData.data[index + 1] = g;
                        imageData.data[index + 2] = b;
                        imageData.data[index + 3] = 255; // Alpha
                    }
                }
                
                // Draw the image data
                ctx.putImageData(imageData, 0, 0);
                
                // Draw contour lines
                ctx.beginPath();
                for (let i = 0; i < 10; i++) {
                    const energy = i / 10;
                    
                    // Draw contour line for this energy level
                    for (const point of energyLandscapeData.grid) {
                        if (Math.abs(point.energy - energy) < 0.05) {
                            ctx.moveTo(point.x, point.y);
                            ctx.arc(point.x, point.y, 1, 0, Math.PI * 2);
                        }
                    }
                }
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
                ctx.stroke();
                
                // Mark minima
                for (const minimum of energyLandscapeData.minima) {
                    ctx.beginPath();
                    ctx.arc(minimum.x, minimum.y, 8, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(59, 130, 246, 0.7)';
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 2;
                    ctx.fill();
                    ctx.stroke();
                }
            }
            
            // Get color for energy value based on color scheme
            function getColorForEnergy(energy, scheme) {
                // Clamp energy between 0 and 1
                energy = Math.max(0, Math.min(1, energy));
                
                if (scheme === 'heat') {
                    // Heat map: blue (low) to red (high)
                    const r = Math.floor(energy * 255);
                    const g = Math.floor((1 - Math.abs(energy - 0.5) * 2) * 255);
                    const b = Math.floor((1 - energy) * 255);
                    return `rgb(${r}, ${g}, ${b})`;
                } else if (scheme === 'spectral') {
                    // Spectral: rainbow colors
                    const hue = (1 - energy) * 240; // 240 (blue) to 0 (red)
                    return `hsl(${hue}, 80%, 60%)`;
                } else {
                    // Quantum: blue/purple gradient (default)
                    const h = 270 - energy * 60; // 270 (purple) to 210 (blue)
                    const s = 70 + energy * 20; // 70% to 90% saturation
                    const l = 40 + energy * 30; // 40% to 70% lightness
                    return `hsl(${h}, ${s}%, ${l}%)`;
                }
            }
            
            // Initialize the SVG for convergence path
            function initializeConvergencePath(width, height) {
                // Create SVG
                const svg = d3.select(convergencePath)
                    .attr('width', width)
                    .attr('height', height);
                
                // Clear existing content
                svg.selectAll('*').remove();
                
                // Create path generator
                const lineGenerator = d3.line()
                    .x(d => d.x)
                    .y(d => d.y)
                    .curve(d3.curveBasis);
                
                // Add path
                svg.append('path')
                    .datum(convergencePathData)
                    .attr('fill', 'none')
                    .attr('stroke', 'rgba(59, 130, 246, 0.8)')
                    .attr('stroke-width', 2)
                    .attr('stroke-dasharray', '5,5')
                    .attr('d', lineGenerator)
                    .attr('opacity', 0); // Initially hidden
                
                // Add marker that will move along the path
                svg.append('circle')
                    .attr('class', 'path-marker')
                    .attr('r', 5)
                    .attr('fill', '#3B82F6')
                    .attr('filter', 'drop-shadow(0 0 3px rgba(59, 130, 246, 0.8))')
                    .attr('opacity', 0); // Initially hidden
            }
            
            // Run the quantum annealing animation
            function runAnnealingAnimation() {
                // Check if already running
                if (isAnnealingRunning) return;
                
                isAnnealingRunning = true;
                currentStep = 0;
                
                // Get the SVG and path
                const svg = d3.select(convergencePath);
                const path = svg.select('path');
                const marker = svg.select('.path-marker');
                
                // Show the path with animation
                path.transition()
                    .duration(1000)
                    .attr('opacity', 0.8);
                
                // Show the marker
                marker.attr('opacity', 1);
                
                // Animate the marker along the path
                const pathNode = path.node();
                const pathLength = pathNode.getTotalLength();
                
                // Duration based on speed slider
                const speed = parseFloat(speedSlider.value);
                const duration = 10000 / speed; // 10 seconds at speed 1.0
                
                // Update score as animation progresses
                const startScore = 0;
                const finalScore = optimizationData.optimization_score || 0.92;
                
                // GSAP timeline
                const timeline = gsap.timeline({
                    onComplete: () => {
                        isAnnealingRunning = false;
                        playAnnealingBtn.textContent = 'Run Quantum Annealing';
                        
                        // Show final score
                        scoreValue.textContent = finalScore.toFixed(2);
                        
                        // Add particles for completed effect
                        addCompletionParticles();
                    }
                });
                
                // Animate score counting up
                timeline.to(scoreValue, {
                    innerHTML: finalScore.toFixed(2),
                    duration: duration / 1000,
                    ease: "power2.out",
                    snap: { innerHTML: 0.01 }
                });
                
                // Animate marker along path
                timeline.to(marker.node(), {
                    duration: duration / 1000,
                    ease: "power1.inOut",
                    motionPath: {
                        path: pathNode,
                        align: pathNode,
                        alignOrigin: [0.5, 0.5]
                    },
                    onUpdate: function() {
                        // Update step counter
                        currentStep = Math.floor(this.progress() * convergencePathData.length);
                        
                        // Update iterations counter
                        const iterations = Math.floor(this.progress() * parseInt(stepsSlider.value));
                        iterationCount.textContent = iterations;
                    }
                }, 0); // Start at the same time as score animation
                
                // Animate task cards
                const taskCards = document.querySelectorAll('.task-item');
                taskCards.forEach(card => {
                    timeline.to(card, {
                        duration: 0.5,
                        backgroundColor: 'rgba(59, 130, 246, 0.2)',
                        x: 5,
                        yoyo: true,
                        repeat: 1,
                        ease: "power1.inOut"
                    }, Math.random() * 5); // Random start time
                });
                
                // Save animation timeline
                annealingAnimation = timeline;
                
                // Update button text
                playAnnealingBtn.textContent = 'Stop Simulation';
            }
            
            // Stop the annealing animation
            function stopAnnealingAnimation() {
                if (!isAnnealingRunning) return;
                
                isAnnealingRunning = false;
                
                // Stop the animation
                if (annealingAnimation) {
                    annealingAnimation.kill();
                }
                
                // Reset UI
                playAnnealingBtn.textContent = 'Run Quantum Annealing';
            }
            
            // Reset the annealing simulation
            function resetAnnealingSimulation() {
                // Stop any running animation
                stopAnnealingAnimation();
                
                // Reset markers and paths
                const svg = d3.select(convergencePath);
                svg.select('path').attr('opacity', 0);
                svg.select('.path-marker').attr('opacity', 0);
                
                // Reset counters
                scoreValue.textContent = '0.00';
                iterationCount.textContent = '0';
                currentStep = 0;
                
                // Re-initialize visualizations
                initializeAssignmentVisualization();
                initializeEnergyLandscape();
                initializeTaskNetwork();
                initializeExpertiseMatching();
            }
            
            // Add particles for completion effect
            function addCompletionParticles() {
                // Create explosion of particles
                for (let i = 0; i < 30; i++) {
                    const particle = document.createElement('div');
                    particle.className = 'particle';
                    
                    // Random size
                    const size = Math.random() * 6 + 3;
                    particle.style.width = `${size}px`;
                    particle.style.height = `${size}px`;
                    
                    // Position at score
                    const rect = scoreValue.getBoundingClientRect();
                    const containerRect = document.getElementById('optimization-container').getBoundingClientRect();
                    
                    const left = rect.left - containerRect.left + rect.width / 2;
                    const top = rect.top - containerRect.top + rect.height / 2;
                    
                    particle.style.left = `${left}px`;
                    particle.style.top = `${top}px`;
                    
                    // Random color - celebration colors
                    const hue = Math.random() * 360;
                    particle.style.backgroundColor = `hsla(${hue}, 70%, 60%, 0.8)`;
                    
                    // Add to container
                    particles.appendChild(particle);
                    
                    // Animate with GSAP
                    gsap.to(particle, {
                        x: Math.random() * 200 - 100,
                        y: Math.random() * 200 - 100,
                        opacity: 0,
                        duration: 1.5,
                        ease: "power2.out",
                        onComplete: () => {
                            particle.remove();
                        }
                    });
                }
            }
            
            // Initialize the task network visualization
            function initializeTaskNetwork() {
                taskNetwork.innerHTML = '';
                
                // Create SVG
                const width = taskNetwork.clientWidth;
                const height = taskNetwork.clientHeight;
                
                const svg = d3.select(taskNetwork)
                    .append('svg')
                    .attr('width', width)
                    .attr('height', height);
                
                // Create nodes from assignments
                const assignments = optimizationData.assignments || {};
                const nodes = [];
                const nodeMap = new Map();
                
                // Create nodes for each task
                Object.entries(assignments).forEach(([taskId, assignee], index) => {
                    const node = {
                        id: taskId,
                        group: assignee,
                        // Generate some fake task data for visualization
                        title: `Task ${index + 1}`,
                        state: ['PENDING', 'IN_PROGRESS', 'COMPLETED', 'BLOCKED'][Math.floor(Math.random() * 4)],
                        priority: Math.floor(Math.random() * 5) + 1,
                        entropy: Math.random().toFixed(2)
                    };
                    
                    nodes.push(node);
                    nodeMap.set(taskId, node);
                });
                
                // Create links between entangled tasks
                const links = [];
                
                // Create some random links for visualization
                for (let i = 0; i < nodes.length; i++) {
                    const numLinks = Math.floor(Math.random() * 3); // 0-2 links per node
                    
                    for (let j = 0; j < numLinks; j++) {
                        // Pick a random target node that's not the current node
                        let targetIndex;
                        do {
                            targetIndex = Math.floor(Math.random() * nodes.length);
                        } while (targetIndex === i);
                        
                        links.push({
                            source: nodes[i].id,
                            target: nodes[targetIndex].id,
                            value: Math.random().toFixed(2)
                        });
                    }
                }
                
                // Create a force simulation
                const simulation = d3.forceSimulation(nodes)
                    .force('link', d3.forceLink(links).id(d => d.id).distance(100))
                    .force('charge', d3.forceManyBody().strength(-300))
                    .force('center', d3.forceCenter(width / 2, height / 2))
                    .force('collide', d3.forceCollide().radius(30));
                
                // Add links
                const link = svg.append('g')
                    .attr('class', 'links')
                    .selectAll('line')
                    .data(links)
                    .enter().append('line')
                    .attr('stroke-width', d => Math.sqrt(d.value) * 2)
                    .attr('stroke', '#3B82F6')
                    .attr('class', 'entanglement-line');
                
                // Add nodes
                const node = svg.append('g')
                    .attr('class', 'nodes')
                    .selectAll('circle')
                    .data(nodes)
                    .enter().append('circle')
                    .attr('r', 10)
                    .attr('fill', d => colors.states[d.state])
                    .call(d3.drag()
                        .on('start', dragstarted)
                        .on('drag', dragging)
                        .on('end', dragended));
                
                // Add labels
                const label = svg.append('g')
                    .attr('class', 'labels')
                    .selectAll('text')
                    .data(nodes)
                    .enter().append('text')
                    .text(d => d.title)
                    .attr('font-size', 10)
                    .attr('dx', 12)
                    .attr('dy', 4)
                    .style('fill', 'white');
                
                // Update positions on tick
                simulation.on('tick', () => {
                    link
                        .attr('x1', d => d.source.x)
                        .attr('y1', d => d.source.y)
                        .attr('x2', d => d.target.x)
                        .attr('y2', d => d.target.y);
                    
                    node
                        .attr('cx', d => d.x = Math.max(10, Math.min(width - 10, d.x)))
                        .attr('cy', d => d.y = Math.max(10, Math.min(height - 10, d.y)));
                    
                    label
                        .attr('x', d => d.x)
                        .attr('y', d => d.y);
                });
                
                // Drag functions
                function dragstarted(event, d) {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }
                
                function dragging(event, d) {
                    d.fx = event.x;
                    d.fy = event.y;
                }
                
                function dragended(event, d) {
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }
                
                // Add hover effects
                node.on('mouseover', function(event, d) {
                    // Highlight the node
                    d3.select(this)
                        .transition()
                        .duration(200)
                        .attr('r', 15);
                    
                    // Show tooltip
                    tooltip.style.opacity = 1;
                    tooltip.innerHTML = `
                        <div style="font-weight: 600; margin-bottom: 5px;">${d.title}</div>
                        <div>State: <span style="color: ${colors.states[d.state]}">${d.state}</span></div>
                        <div>Priority: ${d.priority}</div>
                        <div>Entropy: ${d.entropy}</div>
                        <div>Assigned to: ${d.group}</div>
                    `;
                    tooltip.style.left = (event.pageX + 10) + 'px';
                    tooltip.style.top = (event.pageY - 10) + 'px';
                    
                    // Highlight connected links and nodes
                    link.style('stroke-opacity', l => 
                        l.source.id === d.id || l.target.id === d.id ? 1 : 0.1
                    );
                    
                    node.style('opacity', n => 
                        n.id === d.id || links.some(l => 
                            (l.source.id === d.id && l.target.id === n.id) || 
                            (l.target.id === d.id && l.source.id === n.id)
                        ) ? 1 : 0.3
                    );
                    
                    label.style('opacity', n => 
                        n.id === d.id || links.some(l => 
                            (l.source.id === d.id && l.target.id === n.id) || 
                            (l.target.id === d.id && l.source.id === n.id)
                        ) ? 1 : 0.3
                    );
                })
                .on('mousemove', function(event) {
                    tooltip.style.left = (event.pageX + 10) + 'px';
                    tooltip.style.top = (event.pageY - 10) + 'px';
                })
                .on('mouseout', function() {
                    // Reset node size
                    d3.select(this)
                        .transition()
                        .duration(200)
                        .attr('r', 10);
                    
                    // Hide tooltip
                    tooltip.style.opacity = 0;
                    
                    // Reset link and node opacity
                    link.style('stroke-opacity', 0.6);
                    node.style('opacity', 1);
                    label.style('opacity', 1);
                });
                
                // Save the simulation for later updates
                taskNetworkSimulation = simulation;
            }
            
            // Initialize the expertise matching visualization
            function initializeExpertiseMatching() {
                expertiseMatching.innerHTML = '';
                
                // Create a container
                const container = document.createElement('div');
                container.className = 'glass-panel';
                container.style.height = '100%';
                container.style.overflow = 'auto';
                
                // Create a header
                const header = document.createElement('div');
                header.style.marginBottom = '15px';
                header.innerHTML = `
                    <div style="font-weight: 600; margin-bottom: 5px; color: #e2e8f0;">Optimal Task Assignments</div>
                    <div style="font-size: 12px; color: #94a3b8; margin-bottom: 10px;">
                        The quantum optimization algorithm has found the optimal assignment of tasks to team members,
                        balancing expertise match, cognitive load, and task dependencies.
                    </div>
                `;
                container.appendChild(header);
                
                // Create task list
                const tasks = document.createElement('div');
                tasks.className = 'task-list';
                
                // Add tasks
                const assignments = optimizationData.assignments || {};
                Object.entries(assignments).forEach(([taskId, assignee], index) => {
                    // Create task item
                    const taskItem = document.createElement('div');
                    taskItem.className = 'task-item';
                    taskItem.id = `task-item-${taskId}`;
                    
                    // Generate random task data
                    const task = {
                        id: taskId,
                        title: `Task ${index + 1}`,
                        state: ['PENDING', 'IN_PROGRESS', 'COMPLETED', 'BLOCKED'][Math.floor(Math.random() * 4)],
                        priority: Math.floor(Math.random() * 5) + 1,
                        entropy: Math.random().toFixed(2),
                        assignee
                    };
                    
                    // Task icon
                    const taskIcon = document.createElement('div');
                    taskIcon.className = 'task-icon';
                    taskIcon.style.backgroundColor = colors.states[task.state];
                    taskIcon.textContent = task.title.charAt(0);
                    
                    // Task details
                    const taskDetails = document.createElement('div');
                    taskDetails.className = 'task-details';
                    taskDetails.innerHTML = `
                        <div class="task-title">${task.title}</div>
                        <div class="task-meta">
                            <span class="task-meta-item">Priority: ${task.priority}</span>
                            <span class="task-meta-item">Entropy: ${task.entropy}</span>
                        </div>
                    `;
                    
                    // Task assignee
                    const taskAssignee = document.createElement('div');
                    taskAssignee.className = 'task-assignee';
                    taskAssignee.textContent = task.assignee;
                    
                    // Assemble task item
                    taskItem.appendChild(taskIcon);
                    taskItem.appendChild(taskDetails);
                    taskItem.appendChild(taskAssignee);
                    
                    // Add to task list
                    tasks.appendChild(taskItem);
                    
                    // Add hover effect for the task
                    taskItem.addEventListener('mouseover', () => {
                        showTaskTooltip(event, task);
                    });
                    
                    taskItem.addEventListener('mousemove', (event) => {
                        tooltip.style.left = (event.pageX + 10) + 'px';
                        tooltip.style.top = (event.pageY - 10) + 'px';
                    });
                    
                    taskItem.addEventListener('mouseout', () => {
                        tooltip.style.opacity = 0;
                    });
                    
                    // Add click selection
                    taskItem.addEventListener('click', () => {
                        document.querySelectorAll('.task-item').forEach(item => {
                            item.classList.remove('selected');
                        });
                        taskItem.classList.add('selected');
                    });
                    
                    // Animate task appearance
                    gsap.fromTo(taskItem, 
                        { opacity: 0, x: -20 }, 
                        { opacity: 1, x: 0, duration: 0.5, delay: index * 0.1 }
                    );
                });
                
                container.appendChild(tasks);
                expertiseMatching.appendChild(container);
                
                // Add optimization results
                const resultsContainer = document.createElement('div');
                resultsContainer.className = 'glass-panel';
                resultsContainer.style.marginTop = '15px';
                resultsContainer.style.padding = '10px';
                
                resultsContainer.innerHTML = `
                    <div style="font-weight: 600; margin-bottom: 5px; color: #e2e8f0;">Optimization Results</div>
                    <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                        <div>
                            <div style="font-size: 12px; color: #94a3b8;">Time Saved</div>
                            <div style="font-size: 16px; color: #3b82f6; font-weight: 600;">${optimizationData.expected_completion_improvements?.time_saved_hours || 28}h</div>
                        </div>
                        <div>
                            <div style="font-size: 12px; color: #94a3b8;">Quality Improved</div>
                            <div style="font-size: 16px; color: #3b82f6; font-weight: 600;">${(optimizationData.expected_completion_improvements?.quality_improvement || 0.15) * 100}%</div>
                        </div>
                        <div>
                            <div style="font-size: 12px; color: #94a3b8;">Cognitive Load</div>
                            <div style="font-size: 16px; color: #3b82f6; font-weight: 600;">-${(optimizationData.expected_completion_improvements?.cognitive_load_reduction || 0.23) * 100}%</div>
                        </div>
                    </div>
                `;
                
                container.appendChild(resultsContainer);
            }
            
            // Show tooltip for task
            function showTaskTooltip(event, task) {
                tooltip.style.opacity = 1;
                tooltip.innerHTML = `
                    <div style="font-weight: 600; margin-bottom: 5px;">${task.title}</div>
                    <div>State: <span style="color: ${colors.states[task.state]}">${task.state}</span></div>
                    <div>Priority: ${task.priority}</div>
                    <div>Entropy: ${task.entropy}</div>
                    <div style="margin-top: 5px;">Assigned to: <span style="color: #3b82f6;">${task.assignee}</span></div>
                    <div style="margin-top: 5px; font-size: 10px; color: #94a3b8;">
                        Expertise match: ${Math.round(Math.random() * 30 + 70)}%
                    </div>
                `;
                tooltip.style.left = (event.pageX + 10) + 'px';
                tooltip.style.top = (event.pageY - 10) + 'px';
            }
            
            // Update metrics display
            function updateMetricsDisplay() {
                const score = optimizationData.optimization_score || 0;
                const tasks = optimizationData.task_count || 0;
                const iterations = parseInt(stepsSlider.value) || 1000;
                const timeSaved = optimizationData.expected_completion_improvements?.time_saved_hours || 28;
                const cognitiveReduction = optimizationData.expected_completion_improvements?.cognitive_load_reduction || 0.23;
                
                // Update elements
                scoreValue.textContent = '0.00'; // Start at 0, will be animated
                taskCount.textContent = tasks;
                iterationCount.textContent = '0'; // Start at 0, will be animated
                timeSaved.textContent = `${timeSaved}h`;
                cognitiveReduce.textContent = `${Math.round(cognitiveReduction * 100)}%`;
            }
            
            // Set up event listeners for controls
            function setupEventListeners() {
                // Play/Stop annealing button
                playAnnealingBtn.addEventListener('click', () => {
                    if (isAnnealingRunning) {
                        stopAnnealingAnimation();
                    } else {
                        runAnnealingAnimation();
                    }
                });
                
                // Reset annealing button
                resetAnnealingBtn.addEventListener('click', resetAnnealingSimulation);
                
                // View mode selector
                viewModeSelect.addEventListener('change', () => {
                    // Redraw the energy landscape
                    const width = energyCanvas.clientWidth;
                    const height = energyCanvas.clientHeight;
                    const ctx = energyCanvas.getContext('2d');
                    renderEnergyLandscape(ctx, width, height);
                });
                
                // Color scheme selector
                colorSchemeSelect.addEventListener('change', () => {
                    // Redraw the energy landscape
                    const width = energyCanvas.clientWidth;
                    const height = energyCanvas.clientHeight;
                    const ctx = energyCanvas.getContext('2d');
                    renderEnergyLandscape(ctx, width, height);
                });
                
                // Speed slider
                speedSlider.addEventListener('input', () => {
                    speedValue.textContent = `${speedSlider.value}x`;
                });
                
                // Detail slider
                detailSlider.addEventListener('input', () => {
                    const detailLevel = parseInt(detailSlider.value);
                    const detailLabels = ['Low', 'Medium', 'High'];
                    detailValue.textContent = detailLabels[detailLevel - 1];
                });
                
                // Temperature slider
                tempSlider.addEventListener('input', () => {
                    tempValue.textContent = tempSlider.value;
                });
                
                // Steps slider
                stepsSlider.addEventListener('input', () => {
                    stepsValue.textContent = stepsSlider.value;
                });
                
                // Tooltip for info icons
                document.querySelectorAll('.info-icon').forEach(icon => {
                    icon.addEventListener('mouseover', (event) => {
                        tooltip.style.opacity = 1;
                        tooltip.textContent = icon.getAttribute('title');
                        tooltip.style.left = (event.pageX + 10) + 'px';
                        tooltip.style.top = (event.pageY - 10) + 'px';
                    });
                    
                    icon.addEventListener('mousemove', (event) => {
                        tooltip.style.left = (event.pageX + 10) + 'px';
                        tooltip.style.top = (event.pageY - 10) + 'px';
                    });
                    
                    icon.addEventListener('mouseout', () => {
                        tooltip.style.opacity = 0;
                    });
                });
                
                // Handle window resize
                window.addEventListener('resize', () => {
                    // Redraw the energy landscape
                    const width = energyCanvas.clientWidth;
                    const height = energyCanvas.clientHeight;
                    energyCanvas.width = width;
                    energyCanvas.height = height;
                    const ctx = energyCanvas.getContext('2d');
                    renderEnergyLandscape(ctx, width, height);
                    
                    // Resize convergence path SVG
                    d3.select(convergencePath)
                        .attr('width', width)
                        .attr('height', height);
                });
            }
            
            // Run initial animation when loaded
            setTimeout(() => {
                runAnnealingAnimation();
            }, 1000);
        </script>
    </body>
    </html>
    """
    
    # Display the visualization
    components.html(html_content, height=height, scrolling=False)
    
    return None