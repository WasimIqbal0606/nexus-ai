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
    """Display an advanced, vibrant, and shining animated title using JavaScript animation."""
    components.html(
        """
        <script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500&display=swap');
        
        #particles-js {
            position: relative;
            width: 100%;
            height: 180px;
            background-color: transparent;
            background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);
            margin-bottom: 20px;
            overflow: hidden;
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }
        
        .title-container {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 100%;
            text-align: center;
            z-index: 10;
        }
        
        .app-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 55px;
            font-weight: 900;
            position: relative;
            background: linear-gradient(
                92deg,
                #ff00cc, #3393ff, #00ffff, #ffff00, #ff9966, #ff00cc
            );
            background-size: 1000% 1000%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 
                0 0 15px rgba(255, 0, 204, 0.7),
                0 0 25px rgba(51, 147, 255, 0.5),
                0 0 35px rgba(0, 255, 255, 0.3);
            animation: ultra_shimmer 7s linear infinite, pulse_bright 2s ease-in-out infinite alternate;
            letter-spacing: 3px;
            filter: drop-shadow(0 0 15px rgba(255, 0, 204, 0.9));
            transform-style: preserve-3d;
            perspective: 500px;
        }
        
        .app-title::before {
            content: attr(data-text);
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                92deg,
                #ff00cc, #3393ff, #00ffff, #ffff00, #ff9966, #ff00cc
            );
            background-size: 1000% 1000%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            opacity: 0.5;
            filter: blur(8px);
            animation: ultra_shimmer 7s linear infinite reverse;
            z-index: -1;
        }
        
        .app-title::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(
                circle at 50% 50%,
                rgba(255, 255, 255, 0.8) 0%,
                rgba(255, 255, 255, 0) 70%
            );
            filter: blur(5px);
            mix-blend-mode: overlay;
            pointer-events: none;
            animation: pulse_overlay 3s ease-in-out infinite alternate;
            z-index: 1;
        }
        
        @keyframes ultra_shimmer {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        @keyframes pulse_bright {
            0% { filter: drop-shadow(0 0 15px rgba(255, 0, 204, 0.9)); }
            50% { filter: drop-shadow(0 0 25px rgba(51, 147, 255, 0.9)); }
            100% { filter: drop-shadow(0 0 20px rgba(0, 255, 255, 0.9)); }
        }
        
        @keyframes pulse_overlay {
            0% { opacity: 0.1; }
            100% { opacity: 0.4; }
        }
        
        .quantum-letter {
            display: inline-block;
            position: relative;
            animation: float_3d 0.2s ease-in-out infinite alternate, 
                       color_cycle 4s linear infinite;
            transform-style: preserve-3d;
            text-shadow: 0 0 15px currentColor;
        }
        
        @keyframes float_3d {
            0% { transform: translateY(0) rotateY(0deg) scale(1); }
            100% { transform: translateY(-6px) rotateY(10deg) scale(1.08); }
        }
        
        @keyframes color_cycle {
            0% { color: #ff00cc; }
            20% { color: #3393ff; }
            40% { color: #00ffff; }
            60% { color: #ffff00; }
            80% { color: #ff9966; }
            100% { color: #ff00cc; }
        }
        
        .quantum-letter:nth-child(1) { animation-delay: 0.1s; }
        .quantum-letter:nth-child(2) { animation-delay: 0.2s; }
        .quantum-letter:nth-child(3) { animation-delay: 0.3s; }
        .quantum-letter:nth-child(4) { animation-delay: 0.4s; }
        .quantum-letter:nth-child(5) { animation-delay: 0.5s; }
        .quantum-letter:nth-child(6) { animation-delay: 0.6s; }
        .quantum-letter:nth-child(7) { animation-delay: 0.7s; }
        .quantum-letter:nth-child(8) { animation-delay: 0.8s; }
        .quantum-letter:nth-child(9) { animation-delay: 0.9s; }
        .quantum-letter:nth-child(10) { animation-delay: 1.0s; }
        .quantum-letter:nth-child(11) { animation-delay: 1.1s; }
        .quantum-letter:nth-child(12) { animation-delay: 1.2s; }
        .quantum-letter:nth-child(13) { animation-delay: 1.3s; }
        
        .app-subtitle {
            font-family: 'Rajdhani', sans-serif;
            font-size: 20px;
            color: #7fdbca;
            margin-top: 10px;
            animation: fadeInUp 1.5s ease-out 0.5s both;
            letter-spacing: 3px;
            text-transform: uppercase;
            opacity: 0.9;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
        }
        
        .app-subtitle::before,
        .app-subtitle::after {
            content: '⚛';
            display: inline-block;
            margin: 0 10px;
            animation: rotate360 4s linear infinite;
            color: #64ffda;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes shimmer {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }
        
        @keyframes glow {
            from {
                text-shadow: 
                    0 0 10px rgba(32, 255, 223, 0.6),
                    0 0 20px rgba(32, 255, 223, 0.4),
                    0 0 30px rgba(32, 255, 223, 0.2);
            }
            to {
                text-shadow: 
                    0 0 20px rgba(32, 255, 223, 0.8),
                    0 0 30px rgba(32, 255, 223, 0.6),
                    0 0 40px rgba(32, 255, 223, 0.4);
            }
        }
        
        @keyframes vibrate {
            from { transform: translateY(0) scale(1); }
            to { transform: translateY(-2px) scale(1.05); }
        }
        
        @keyframes rotate360 {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .shine-effect {
            position: absolute;
            top: 0;
            left: -100%;
            width: 50%;
            height: 100%;
            background: linear-gradient(
                to right,
                rgba(255, 255, 255, 0) 0%,
                rgba(255, 255, 255, 0.3) 50%,
                rgba(255, 255, 255, 0) 100%
            );
            transform: skewX(-25deg);
            animation: shine 3s infinite;
        }
        
        @keyframes shine {
            0% { left: -100%; }
            20% { left: 100%; }
            100% { left: 100%; }
        }
        
        /* Star background */
        .stars {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1;
        }
        
        .star {
            position: absolute;
            background-color: white;
            border-radius: 50%;
            animation: twinkle linear infinite;
        }
        
        @keyframes twinkle {
            0% { opacity: 0.2; }
            50% { opacity: 1; }
            100% { opacity: 0.2; }
        }
        </style>
        
        <div id="particles-js">
            <div class="stars" id="stars"></div>
            <div class="shine-effect"></div>
            <div class="title-container">
                <div class="app-title">
                    <span class="quantum-letter">Q</span>
                    <span class="quantum-letter">u</span>
                    <span class="quantum-letter">a</span>
                    <span class="quantum-letter">n</span>
                    <span class="quantum-letter">t</span>
                    <span class="quantum-letter">u</span>
                    <span class="quantum-letter">m</span>
                    <span class="quantum-letter">&nbsp;</span>
                    <span class="quantum-letter">N</span>
                    <span class="quantum-letter">e</span>
                    <span class="quantum-letter">x</span>
                    <span class="quantum-letter">u</span>
                    <span class="quantum-letter">s</span>
                </div>
                <div class="app-subtitle">Advanced Task Management System</div>
            </div>
        </div>
        
        <script>
        // Create randomly positioned stars
        const starsContainer = document.getElementById('stars');
        const starCount = 100;
        
        for (let i = 0; i < starCount; i++) {
            const star = document.createElement('div');
            star.className = 'star';
            
            // Random position
            const x = Math.random() * 100;
            const y = Math.random() * 100;
            
            // Random size (0.5px to 2px)
            const size = Math.random() * 1.5 + 0.5;
            
            // Random animation duration (1-5s)
            const duration = Math.random() * 4 + 1;
            
            star.style.left = `${x}%`;
            star.style.top = `${y}%`;
            star.style.width = `${size}px`;
            star.style.height = `${size}px`;
            star.style.animationDuration = `${duration}s`;
            
            starsContainer.appendChild(star);
        }
        
        // Configure particles
        particlesJS('particles-js', {
            "particles": {
                "number": {
                    "value": 100,
                    "density": {
                        "enable": true,
                        "value_area": 800
                    }
                },
                "color": {
                    "value": ["#64ffda", "#00bcd4", "#7986cb", "#9c27b0", "#f06292"]
                },
                "shape": {
                    "type": ["circle", "triangle", "polygon"],
                    "stroke": {
                        "width": 0,
                        "color": "#000000"
                    },
                    "polygon": {
                        "nb_sides": 6
                    }
                },
                "opacity": {
                    "value": 0.6,
                    "random": true,
                    "anim": {
                        "enable": true,
                        "speed": 1,
                        "opacity_min": 0.1,
                        "sync": false
                    }
                },
                "size": {
                    "value": 3,
                    "random": true,
                    "anim": {
                        "enable": true,
                        "speed": 2,
                        "size_min": 0.1,
                        "sync": false
                    }
                },
                "line_linked": {
                    "enable": true,
                    "distance": 120,
                    "color": "#64ffda",
                    "opacity": 0.5,
                    "width": 1
                },
                "move": {
                    "enable": true,
                    "speed": 3,
                    "direction": "none",
                    "random": true,
                    "straight": false,
                    "out_mode": "out",
                    "bounce": false,
                    "attract": {
                        "enable": true,
                        "rotateX": 600,
                        "rotateY": 1200
                    }
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
                        "particles_nb": 6
                    }
                }
            },
            "retina_detect": true
        });
        </script>
        """,
        height=200,
    )

def animated_task_card(task):
    """Render a task card with advanced animations and visual effects."""
    import random
    import datetime as dt
    
    # Get status and priority classes
    task_state = task['state'].lower()
    status_class = f"status-{task_state}"
    
    # Create more vibrant color schemes based on state
    color_schemes = {
        "pending": {
            "gradient": "linear-gradient(135deg, #FFD700, #FFA500)",
            "glow": "0 0 15px rgba(255, 215, 0, 0.7)",
            "accent": "#FFD700",
            "border": "#FFA500",
            "text": "#FFFFFF"
        },
        "in_progress": {
            "gradient": "linear-gradient(135deg, #00BFFF, #1E90FF)",
            "glow": "0 0 15px rgba(0, 191, 255, 0.7)",
            "accent": "#00BFFF",
            "border": "#1E90FF",
            "text": "#FFFFFF"
        },
        "completed": {
            "gradient": "linear-gradient(135deg, #32CD32, #008000)",
            "glow": "0 0 15px rgba(50, 205, 50, 0.7)",
            "accent": "#32CD32",
            "border": "#008000",
            "text": "#FFFFFF"
        },
        "blocked": {
            "gradient": "linear-gradient(135deg, #FF4500, #B22222)",
            "glow": "0 0 15px rgba(255, 69, 0, 0.7)",
            "accent": "#FF4500",
            "border": "#B22222",
            "text": "#FFFFFF"
        }
    }
    
    # Get color scheme based on state
    colors = color_schemes.get(task_state, color_schemes["pending"])
    
    # Determine priority and its visual representation
    priority = task.get('priority', 1)
    priority_class = "priority-low"
    priority_stars = "★"
    
    if priority >= 4:
        priority_class = "priority-high"
        priority_stars = "★★★★★"
    elif priority >= 3:
        priority_class = "priority-high"
        priority_stars = "★★★★☆"
    elif priority >= 2:
        priority_class = "priority-medium"
        priority_stars = "★★★☆☆"
    else:
        priority_stars = "★★☆☆☆"
    
    # Calculate quantum probability values for visualization
    prob_values = task.get('probability_distribution', {})
    prob_pending = prob_values.get('PENDING', 0.25) * 100
    prob_in_progress = prob_values.get('IN_PROGRESS', 0.25) * 100
    prob_completed = prob_values.get('COMPLETED', 0.25) * 100
    prob_blocked = prob_values.get('BLOCKED', 0.25) * 100
    
    entropy = task.get('entropy', 0.5)
    
    # Generate particle count based on entropy (more entropy = more particles)
    particle_count = int(entropy * 50) + 10
    
    # Format dates
    created_at = task.get('created_at')
    due_date = task.get('due_date')
    
    created_at_str = ""
    due_date_str = ""
    
    if created_at:
        try:
            if isinstance(created_at, str):
                created_at = dt.datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            created_at_str = created_at.strftime("%Y-%m-%d")
        except:
            created_at_str = str(created_at)
    
    if due_date:
        try:
            if isinstance(due_date, str):
                due_date = dt.datetime.fromisoformat(due_date.replace('Z', '+00:00'))
            due_date_str = due_date.strftime("%Y-%m-%d")
        except:
            due_date_str = str(due_date)
    
    # Calculate urgency for due tasks
    urgency_class = ""
    urgency_animation = ""
    urgency_label = ""
    
    if due_date:
        try:
            now = dt.datetime.now()
            if isinstance(due_date, str):
                due_date = dt.datetime.fromisoformat(due_date.replace('Z', '+00:00'))
            
            days_remaining = (due_date - now).days
            
            if days_remaining < 0:
                urgency_class = "overdue-task"
                urgency_animation = "pulse-red 1.5s infinite"
                urgency_label = f"<span class='overdue-label'>OVERDUE by {abs(days_remaining)} days!</span>"
            elif days_remaining == 0:
                urgency_class = "due-today-task"
                urgency_animation = "pulse-yellow 2s infinite"
                urgency_label = "<span class='due-today-label'>DUE TODAY!</span>"
            elif days_remaining <= 2:
                urgency_class = "soon-due-task"
                urgency_animation = "pulse-yellow 3s infinite"
                urgency_label = "<span class='soon-due-label'>DUE SOON</span>"
        except:
            pass
    
    # Generate a random ID for this task card's particle container
    particle_id = f"particles-{task['id']}"
    
    # Get tags with random vibrant colors for each
    tag_colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#FBAD4B", "#9D65C9", 
        "#5D87E1", "#58C7F3", "#4CD964", "#FF5E3A", "#C644FC",
        "#FF9500", "#5856D6", "#C86EDF", "#00C7BE", "#5AC8FA"
    ]
    
    tags_html = ""
    if task.get('tags'):
        for i, tag in enumerate(task.get('tags', [])):
            color_index = i % len(tag_colors)
            tag_color = tag_colors[color_index]
            tags_html += f"""
            <span class="task-tag" style="background: {tag_color}20; color: {tag_color}; 
                animation: tag-pop 0.5s ease-out {i*0.1}s both;">
                {tag}
            </span>
            """
    
    # Generate the assignee avatar if available
    assignee_html = ""
    if task.get('assignee'):
        initials = ''.join([name[0].upper() for name in task.get('assignee', 'U').split(' ')])[:2]
        assignee_html = f"""
        <div class="assignee-avatar" title="Assigned to: {task.get('assignee')}">
            {initials}
        </div>
        """
    
    html = f"""
    <style>
    @keyframes float-up-down {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-5px); }}
    }}
    
    @keyframes shine-across {{
        0% {{ left: -100%; }}
        20%, 100% {{ left: 100%; }}
    }}
    
    @keyframes pulse-red {{
        0%, 100% {{ box-shadow: 0 0 15px rgba(255, 69, 0, 0.7); }}
        50% {{ box-shadow: 0 0 25px rgba(255, 69, 0, 0.9); }}
    }}
    
    @keyframes pulse-yellow {{
        0%, 100% {{ box-shadow: 0 0 10px rgba(255, 215, 0, 0.5); }}
        50% {{ box-shadow: 0 0 20px rgba(255, 215, 0, 0.8); }}
    }}
    
    @keyframes tag-pop {{
        0% {{ transform: scale(0.8); opacity: 0; }}
        70% {{ transform: scale(1.1); }}
        100% {{ transform: scale(1); opacity: 1; }}
    }}
    
    @keyframes rotate-glow {{
        0% {{ transform: rotate(0deg); filter: hue-rotate(0deg); }}
        100% {{ transform: rotate(360deg); filter: hue-rotate(360deg); }}
    }}
    
    .task-card-enhanced {{
        background: rgba(16, 33, 65, 0.7);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
        border-left: 5px solid {colors['border']};
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3), {colors['glow']};
        animation: float-up-down 5s ease-in-out infinite;
        transition: all 0.3s ease;
        z-index: 1;
    }}
    
    .task-card-enhanced:hover {{
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4), {colors['glow']};
        z-index: 10;
    }}
    
    .card-header {{
        position: relative;
        margin-bottom: 15px;
        padding-bottom: 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .task-title {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 22px;
        font-weight: bold;
        color: white;
        margin: 0;
        padding: 0;
        letter-spacing: 0.5px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        display: inline-block;
    }}
    
    .task-title::after {{
        content: '';
        display: block;
        width: 0;
        height: 2px;
        background: {colors['accent']};
        transition: width 0.3s ease;
        margin-top: 3px;
    }}
    
    .task-card-enhanced:hover .task-title::after {{
        width: 100%;
    }}
    
    .shine-line {{
        position: absolute;
        top: 0;
        left: -100%;
        width: 50%;
        height: 100%;
        background: linear-gradient(
            to right,
            rgba(255, 255, 255, 0) 0%,
            rgba(255, 255, 255, 0.2) 50%,
            rgba(255, 255, 255, 0) 100%
        );
        transform: skewX(-25deg);
        animation: shine-across 5s infinite;
        animation-delay: {random.random() * 5}s;
    }}
    
    .status-badge-enhanced {{
        background: {colors['gradient']};
        color: {colors['text']};
        font-size: 11px;
        font-weight: bold;
        padding: 5px 12px;
        border-radius: 50px;
        display: inline-block;
        margin-right: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.2);
        animation: float-up-down 3s ease-in-out infinite;
    }}
    
    .priority-stars {{
        color: gold;
        font-size: 14px;
        text-shadow: 0 0 5px rgba(255, 215, 0, 0.7);
        letter-spacing: 2px;
        animation: float-up-down 3s ease-in-out infinite;
        animation-delay: 0.2s;
    }}
    
    .task-description {{
        color: rgba(255, 255, 255, 0.8);
        font-size: 14px;
        line-height: 1.5;
        margin-bottom: 15px;
        padding: 10px;
        border-radius: 8px;
        background: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(5px);
        border-left: 2px solid {colors['accent']};
    }}
    
    .task-tag {{
        font-size: 11px;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 50px;
        display: inline-block;
        margin-right: 8px;
        margin-bottom: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }}
    
    .task-tag:hover {{
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
    }}
    
    .quantum-visualization-enhanced {{
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        padding: 15px;
        margin-top: 15px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .quantum-visualization-enhanced::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(
            circle at center,
            rgba(32, 156, 238, 0.1) 0%,
            rgba(32, 156, 238, 0) 70%
        );
        animation: rotate-glow 10s linear infinite;
        z-index: -1;
    }}
    
    .quantum-title {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        color: rgba(255, 255, 255, 0.9);
        font-size: 13px;
        font-weight: 500;
    }}
    
    .quantum-title svg {{
        margin-right: 5px;
        animation: float-up-down 2s ease-in-out infinite;
    }}
    
    .entropy-value {{
        background: rgba(0, 0, 0, 0.3);
        padding: 3px 8px;
        border-radius: 4px;
        font-family: monospace;
        color: #64ffda;
        border: 1px solid rgba(100, 255, 218, 0.3);
    }}
    
    .probability-bar {{
        height: 8px;
        border-radius: 4px;
        transition: all 0.5s ease;
    }}
    
    .probability-container {{
        height: 30px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    
    .probability-labels {{
        display: flex;
        justify-content: space-between;
        margin-top: 5px;
        font-size: 10px;
        color: rgba(255, 255, 255, 0.7);
    }}
    
    .assignee-avatar {{
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: {colors['gradient']};
        color: white;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 12px;
        font-weight: bold;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.3);
        position: absolute;
        top: 15px;
        right: 15px;
        cursor: pointer;
        transition: all 0.3s ease;
        z-index: 10;
    }}
    
    .assignee-avatar:hover {{
        transform: scale(1.2);
    }}
    
    .task-dates {{
        font-size: 11px;
        color: rgba(255, 255, 255, 0.6);
        display: flex;
        justify-content: space-between;
        margin-top: 15px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .date-label {{
        font-weight: 500;
        color: {colors['accent']};
        margin-right: 5px;
    }}
    
    .overdue-label, .due-today-label, .soon-due-label {{
        background-color: #FF4500;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: bold;
        margin-left: 10px;
        animation: pulse-red 1.5s infinite;
    }}
    
    .due-today-label {{
        background-color: #FFD700;
        color: #000;
        animation: pulse-yellow 2s infinite;
    }}
    
    .soon-due-label {{
        background-color: #FFA500;
        color: #000;
    }}
    
    #{particle_id} {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        overflow: hidden;
        border-radius: 12px;
    }}
    
    .particle {{
        position: absolute;
        background-color: {colors['accent']};
        border-radius: 50%;
        opacity: 0.5;
        animation: float-particle linear infinite;
    }}
    
    @keyframes float-particle {{
        0% {{ transform: translateY(0) rotate(0deg); opacity: 0; }}
        10% {{ opacity: 0.5; }}
        90% {{ opacity: 0.5; }}
        100% {{ transform: translateY(-100px) rotate(360deg); opacity: 0; }}
    }}
    </style>
    
    <div class="task-card-enhanced {urgency_class}" style="animation: float-up-down 5s ease-in-out infinite{', ' + urgency_animation if urgency_animation else ''}">
        <div id="{particle_id}"></div>
        <div class="shine-line"></div>
        
        <div class="card-header">
            {assignee_html}
            <div class="task-title">{task['title']}</div>
            
            <div style="margin-top: 12px;">
                <span class="status-badge-enhanced">{task['state']}</span>
                <span class="priority-stars" title="Priority: {priority}/5">{priority_stars}</span>
                {urgency_label}
            </div>
        </div>
        
        <div class="task-description">{task['description']}</div>
        
        <div style="display: flex; flex-wrap: wrap; margin-bottom: 5px;">
            {tags_html}
        </div>
        
        <div class="quantum-visualization-enhanced">
            <div class="quantum-title">
                <div>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 20C7.59 20 4 16.41 4 12C4 7.59 7.59 4 12 4C16.41 4 20 7.59 20 12C20 16.41 16.41 20 12 20Z" fill="#64ffda"/>
                        <path d="M12 17C14.7614 17 17 14.7614 17 12C17 9.23858 14.7614 7 12 7C9.23858 7 7 9.23858 7 12C7 14.7614 9.23858 17 12 17Z" fill="#64ffda"/>
                    </svg>
                    Quantum States
                </div>
                <span class="entropy-value">{entropy:.2f} ψ</span>
            </div>
            
            <div class="probability-container">
                <div style="display: flex; width: 100%; height: 8px; border-radius: 4px; overflow: hidden; background: rgba(0,0,0,0.3);">
                    <div class="probability-bar" style="width: {prob_pending}%; background-color: #FFD700; height: 100%; position: relative;" title="PENDING: {prob_pending:.1f}%"></div>
                    <div class="probability-bar" style="width: {prob_in_progress}%; background-color: #00BFFF; height: 100%; position: relative;" title="IN_PROGRESS: {prob_in_progress:.1f}%"></div>
                    <div class="probability-bar" style="width: {prob_completed}%; background-color: #32CD32; height: 100%; position: relative;" title="COMPLETED: {prob_completed:.1f}%"></div>
                    <div class="probability-bar" style="width: {prob_blocked}%; background-color: #FF4500; height: 100%; position: relative;" title="BLOCKED: {prob_blocked:.1f}%"></div>
                </div>
                <div class="probability-labels">
                    <span>Pending</span>
                    <span>In Progress</span>
                    <span>Completed</span>
                    <span>Blocked</span>
                </div>
            </div>
        </div>
        
        <div class="task-dates">
            <span><span class="date-label">Created:</span> {created_at_str}</span>
            {f'<span><span class="date-label">Due:</span> {due_date_str}</span>' if due_date_str else ''}
        </div>
    </div>
    
    <script>
    // Create particles effect based on task state and entropy
    (function() {{
        const container = document.getElementById('{particle_id}');
        const particleCount = {particle_count};
        
        for (let i = 0; i < particleCount; i++) {{
            const particle = document.createElement('div');
            particle.className = 'particle';
            
            // Random position
            const x = Math.random() * 100;
            const y = Math.random() * 100 + 50;  // Start from bottom half
            
            // Random size (0.5px to 3px)
            const size = Math.random() * 2.5 + 0.5;
            
            // Random animation duration (3-8s)
            const duration = Math.random() * 5 + 3;
            
            // Random delay
            const delay = Math.random() * 5;
            
            // Apply styles
            particle.style.left = `${{x}}%`;
            particle.style.bottom = `${{-10}}%`;  // Start below
            particle.style.width = `${{size}}px`;
            particle.style.height = `${{size}}px`;
            particle.style.opacity = (Math.random() * 0.5) + 0.1;
            particle.style.animation = `float-particle ${{duration}}s linear infinite`;
            particle.style.animationDelay = `${{delay}}s`;
            
            container.appendChild(particle);
        }}
    }})();
    </script>
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