from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union, Tuple
import numpy as np
import networkx as nx
from scipy import optimize, sparse
from datetime import datetime, timedelta
import uuid
import random
import math
import os
import asyncio
import concurrent.futures
from functools import lru_cache
import logging
import uvicorn
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# LangChain imports - consolidated block
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import CommaSeparatedListOutputParser
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain.chains import LLMChain
    from langchain_community.embeddings import HuggingFaceEmbeddings, FakeEmbeddings
    
    # Vector database - consistent community imports
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document

    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain successfully imported")
except ImportError:
    logger.warning("LangChain libraries not available. Running with limited functionality.")
    LANGCHAIN_AVAILABLE = False

# Quantum simulation
try:
    import pennylane as qml
    QUANTUM_LIBS_AVAILABLE = True
except ImportError:
    logger.warning("Quantum libraries not available. Running with limited functionality.")
    QUANTUM_LIBS_AVAILABLE = False

# --- Configure CPU optimization ---
# Determine optimal thread count for the system
CPU_COUNT = os.cpu_count() or 4
EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT)

# --- LangChain and LLM Setup ---
llm_cache = {}

def get_llm(model_name="llama3-8b-8192"):
    """Get a Groq LLM instance with caching"""
    global llm_cache
    
    if not LANGCHAIN_AVAILABLE:
        logger.warning(f"LangChain not available. Cannot load {model_name} model.")
        return None
    
    # Return cached model if available
    if model_name in llm_cache:
        return llm_cache[model_name]
    
    # Configure the Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    
    if not groq_api_key:
        logger.warning("GROQ_API_KEY not found in environment variables")
        return None
    
    try:
        # Initialize the Groq chat model
        llm = ChatGroq(
            api_key=groq_api_key,
            model_name=model_name,
            temperature=0.3,
            max_tokens=1024
        )
        
        llm_cache[model_name] = llm
        logger.info(f"Successfully loaded {model_name} model")
        return llm
    except Exception as e:
        logger.error(f"Error loading LLM {model_name}: {str(e)}")
        return None

# Initialize embeddings - use HuggingFace for local embedding or fall back to fake
try:
    embeddings_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        cache_folder="./embeddings_cache"
    )
    logger.info("Successfully loaded HuggingFace embeddings model")
except Exception as e:
    logger.warning(f"Falling back to simpler embeddings: {str(e)}")
    embeddings_model = FakeEmbeddings(size=384)

# --- Models ---
class TaskState:
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"

class TaskCreate(BaseModel):
    title: str
    description: str
    assignee: Optional[str] = None
    due_date: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    priority: int = 1  # 1-5

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    assignee: Optional[str] = None
    due_date: Optional[datetime] = None
    state: Optional[str] = None
    tags: Optional[List[str]] = None
    priority: Optional[int] = None  # 1-5

class ComplexValue(BaseModel):
    real: float
    imag: float

class QuantumState(BaseModel):
    """Represents a quantum state for a task"""
    amplitudes: Dict[str, ComplexValue]
    fidelity: float
    coherence_time: float

class Task(BaseModel):
    id: str
    title: str
    description: str
    assignee: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    due_date: Optional[datetime] = None
    state: str = TaskState.PENDING
    tags: List[str] = []
    priority: int = 1  # 1-5
    entropy: float = 1.0  # 0-1, quantum-inspired uncertainty
    probability_distribution: Dict[str, float] = {}  # Quantum state probabilities
    embedding: Optional[List[float]] = None
    entangled_tasks: List[str] = []  # IDs of entangled tasks
    quantum_state: Optional[Dict[str, Any]] = None  # Advanced quantum state information
    category: Optional[str] = None  # LLM-predicted category
    ml_summary: Optional[str] = None  # LLM-generated summary or priority notes

class EntanglementCreate(BaseModel):
    task_id_1: str
    task_id_2: str
    strength: float = 1.0  # 0-1, strength of entanglement
    entanglement_type: str = "standard"  # standard, SWAP, CNOT, etc.

class EntanglementUpdate(BaseModel):
    strength: float
    entanglement_type: Optional[str] = None

class Entanglement(BaseModel):
    id: str
    task_id_1: str
    task_id_2: str
    strength: float
    entanglement_type: str = "standard"
    created_at: datetime
    updated_at: datetime

class SearchQuery(BaseModel):
    query: str
    limit: int = 10
    use_quantum: bool = False

class SystemMetrics(BaseModel):
    total_entropy: float
    task_count: int
    completion_rate: float
    average_cognitive_load: float
    entanglement_density: float
    quantum_coherence: float = 0.0

class QuantumSimulationRequest(BaseModel):
    task_ids: List[str]
    simulation_steps: int = 5
    decoherence_rate: float = 0.05
    measurement_type: str = "projective"  # projective, POVM, weak

# --- Database setup ---
from sqlitedict import SqliteDict
from diskcache import Cache

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Initialize persistent storage
tasks_db = SqliteDict('./data/tasks.sqlite', autocommit=True)
entanglements_db = SqliteDict('./data/entanglements.sqlite', autocommit=True)
cache = Cache('./data/cache')

# Convert to regular dictionaries for compatibility with existing code
tasks = {k: v for k, v in tasks_db.items()} if tasks_db else {}
entanglements = {k: v for k, v in entanglements_db.items()} if entanglements_db else {}
task_graph = nx.Graph()

# Vector store for semantic search
vector_store = None

# Helper function to save tasks
def save_task(task_id, task_data):
    """Save task to persistent storage"""
    tasks[task_id] = task_data  # Update in-memory cache
    tasks_db[task_id] = task_data  # Update persistent storage
    
# Helper function to save entanglements
def save_entanglement(entanglement_id, entanglement_data):
    """Save entanglement to persistent storage"""
    entanglements[entanglement_id] = entanglement_data  # Update in-memory cache
    entanglements_db[entanglement_id] = entanglement_data  # Update persistent storage

# --- Advanced Quantum-inspired utility functions ---
@lru_cache(maxsize=128)
def calculate_task_embedding(task_text):
    """Calculate embedding for task text using HuggingFace model or fallback."""
    try:
        # Get embedding from model
        if embeddings_model:
            embedding = embeddings_model.embed_query(task_text)
            return embedding
        else:
            # Create a simple fallback embedding
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(max_features=384)
            vectorizer.fit([task_text])
            vector = vectorizer.transform([task_text]).toarray()[0]

            # Normalize and pad/truncate to proper dimension
            if len(vector) > 0:
                vector = vector / (np.linalg.norm(vector) + 1e-6)
            if len(vector) > 384:
                vector = vector[:384]
            else:
                vector = np.pad(vector, (0, 384 - len(vector)))

            return vector.tolist()
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        # Create a random embedding as fallback
        embedding = np.random.randn(384)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()

def classify_task(task_text):
    """Classify task using LLM API or fallback rules."""
    try:
        llm = get_llm()
        if llm:
            # Create classification prompt
            prompt = PromptTemplate.from_template(
                """You are a task classification expert. Classify the following task into one of these categories:
                HIGH_PRIORITY, BUG_FIX, FEATURE, IMPLEMENTATION, DEVELOPMENT, RESEARCH, DESIGN, GENERAL_TASK
                
                Task: {task_text}
                
                Category:"""
            )
            
            # Create chain
            chain = LLMChain(llm=llm, prompt=prompt)
            
            # Run chain
            result = chain.run(task_text=task_text).strip()
            
            # Clean the result
            valid_categories = [
                "HIGH_PRIORITY", "BUG_FIX", "FEATURE", "IMPLEMENTATION", 
                "DEVELOPMENT", "RESEARCH", "DESIGN", "GENERAL_TASK"
            ]
            
            # Find the closest match
            for category in valid_categories:
                if category in result:
                    return category
            
            return "GENERAL_TASK"
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")

    # Simple rule-based fallback
    keywords = {
        "urgent": "HIGH_PRIORITY",
        "important": "HIGH_PRIORITY",
        "critical": "HIGH_PRIORITY",
        "bug": "BUG_FIX",
        "fix": "BUG_FIX",
        "feature": "FEATURE",
        "implement": "IMPLEMENTATION",
        "develop": "DEVELOPMENT",
        "research": "RESEARCH",
        "design": "DESIGN"
    }

    task_text_lower = task_text.lower()
    for keyword, category in keywords.items():
        if keyword in task_text_lower:
            return category

    return "GENERAL_TASK"

def generate_task_summary(task_text, max_length=50):
    """Generate a summary for a task using LLM or simple extraction."""
    try:
        llm = get_llm()
        if llm:
            # Create summary prompt
            prompt = PromptTemplate.from_template(
                """Summarize the following task in a concise way (maximum {max_length} characters):
                
                Task: {task_text}
                
                Summary:"""
            )
            
            # Create chain
            chain = LLMChain(llm=llm, prompt=prompt)
            
            # Run chain with parameters
            summary = chain.run(task_text=task_text, max_length=max_length).strip()
            
            # Ensure length constraint
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
                
            return summary
    except Exception as e:
        logger.error(f"Summary generation error: {str(e)}")

    # Simple fallback - extract first sentence
    first_sentence = task_text.split('.')[0]
    if len(first_sentence) > max_length:
        return first_sentence[:max_length] + "..."
    return first_sentence

def answer_task_question(question, context):
    """Answer a question about a task using LLM or simple extraction."""
    try:
        llm = get_llm()
        if llm:
            # Create QA prompt
            prompt = PromptTemplate.from_template(
                """Answer the following question based on the provided context:
                
                Context: {context}
                
                Question: {question}
                
                Answer:"""
            )
            
            # Create chain
            chain = LLMChain(llm=llm, prompt=prompt)
            
            # Run chain
            answer = chain.run(question=question, context=context).strip()
            
            return {
                "answer": answer,
                "score": 0.9,  # Default confidence score
            }
    except Exception as e:
        logger.error(f"QA error: {str(e)}")

    # Simple fallback - look for question keywords in context
    question_lower = question.lower()
    context_lower = context.lower()

    # Extract question focus
    question_words = ["what", "when", "where", "who", "how", "why"]
    focus_word = next((word for word in question_words if word in question_lower), None)

    if focus_word and focus_word in question_lower:
        # Get the portion of the question after the focus word
        focus_index = question_lower.index(focus_word) + len(focus_word)
        question_focus = question_lower[focus_index:].strip()
        
        # Look for related words in context
        focus_words = [word for word in question_focus.split() if len(word) > 3]
        
        for word in focus_words:
            if word in context_lower:
                start_idx = context_lower.index(word)
                # Extract a window around the word
                window_start = max(0, start_idx - 50)
                window_end = min(len(context_lower), start_idx + 50)
                return {
                    "answer": context[window_start:window_end] + "...",
                    "score": 0.5
                }
    
    return {
        "answer": "I couldn't find a specific answer to that question in the context.",
        "score": 0.1
    }

def extract_tags_from_text(text):
    """Extract relevant tags from text using LLM or keyword analysis."""
    try:
        llm = get_llm()
        if llm:
            # Create tag extraction prompt
            prompt = PromptTemplate.from_template(
                """Extract 2-5 relevant tags from the following task description. Return them as a comma-separated list:
                
                Task: {text}
                
                Tags:"""
            )
            
            # Create parser
            output_parser = CommaSeparatedListOutputParser()
            
            # Create chain
            chain = LLMChain(llm=llm, prompt=prompt)
            
            # Run chain
            result = chain.run(text=text).strip()
            
            # Parse tags
            try:
                tags = output_parser.parse(result)
                # Clean and limit tags
                tags = [tag.strip().lower() for tag in tags if tag.strip()]
                return tags[:5]  # Limit to 5 tags
            except Exception as e:
                logger.error(f"Tag parsing error: {str(e)}")
                # Fall through to keyword extraction
        
    except Exception as e:
        logger.error(f"Tag extraction error: {str(e)}")
    
    # Fallback to simple keyword extraction
    common_tags = [
        "frontend", "backend", "bug", "feature", "documentation", 
        "ui", "ux", "api", "database", "testing", "security",
        "performance", "refactor", "cleanup", "research"
    ]
    
    text_lower = text.lower()
    found_tags = []
    
    for tag in common_tags:
        if tag in text_lower:
            found_tags.append(tag)
    
    # Add priority tag if keywords suggest it
    priority_keywords = ["urgent", "critical", "important", "priority", "asap"]
    if any(keyword in text_lower for keyword in priority_keywords):
        found_tags.append("high-priority")
    
    return found_tags[:5]  # Limit to 5 tags

def simulate_quantum_state(task_count):
    """Simulate a quantum state for visualization purposes"""
    if QUANTUM_LIBS_AVAILABLE:
        try:
            # Create a simple quantum circuit with PennyLane
            n_qubits = min(task_count, 5)  # Limit to 5 qubits for performance
            dev = qml.device("default.qubit", wires=n_qubits)
            
            @qml.qnode(dev)
            def quantum_circuit():
                # Initialize qubits
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                
                # Add some entanglement
                for i in range(n_qubits-1):
                    qml.CNOT(wires=[i, i+1])
                
                # Add some rotation based on task properties
                for i in range(n_qubits):
                    qml.RX(np.pi * random.random(), wires=i)
                    qml.RY(np.pi * random.random(), wires=i)
                
                # Return state
                return qml.state()
            
            # Execute circuit
            state = quantum_circuit()
            
            # Convert to proper format
            if isinstance(state, np.ndarray):
                state_list = state.tolist()
                # Convert complex numbers to dict representation
                state_dict = {}
                for i, amp in enumerate(state_list):
                    if isinstance(amp, complex):
                        state_dict[str(i)] = {"real": float(amp.real), "imag": float(amp.imag)}
                    else:
                        state_dict[str(i)] = {"real": float(amp), "imag": 0.0}
                
                return {
                    "state_vector": state_dict,
                    "n_qubits": n_qubits,
                    "fidelity": random.uniform(0.85, 0.99),
                    "visualization_data": [abs(amp)**2 for amp in state_list[:8]]  # First 8 probabilities
                }
        except Exception as e:
            logger.error(f"Quantum simulation error: {str(e)}")
    
    # Fallback to classical simulation
    n_states = 2 ** min(task_count, 3)
    probs = [random.random() for _ in range(n_states)]
    total = sum(probs)
    probs = [p/total for p in probs]
    
    state_dict = {}
    for i in range(n_states):
        theta = random.uniform(0, np.pi)
        phi = random.uniform(0, 2*np.pi)
        amp = np.sqrt(probs[i]) * np.exp(1j * phi)
        state_dict[str(i)] = {"real": float(amp.real), "imag": float(amp.imag)}
    
    return {
        "state_vector": state_dict,
        "n_qubits": min(task_count, 3),
        "fidelity": random.uniform(0.85, 0.99),
        "visualization_data": probs
    }

def create_initial_probability_distribution():
    """Create a probability distribution across task states"""
    states = [TaskState.PENDING, TaskState.IN_PROGRESS, TaskState.COMPLETED, TaskState.BLOCKED]
    probs = [random.random() for _ in states]
    total = sum(probs)
    return {state: probs[i]/total for i, state in enumerate(states)}

def update_task_entropy(task_id):
    """Update entropy (uncertainty) for a task based on its properties"""
    if task_id not in tasks:
        return
    
    task = tasks[task_id]
    
    # Base entropy calculation
    due_date_factor = 0.0
    if task.due_date:
        days_remaining = (task.due_date - datetime.now()).days
        if days_remaining < 0:
            due_date_factor = 1.0  # Overdue tasks have max entropy
        else:
            due_date_factor = max(0.0, 1.0 - (days_remaining / 14.0))  # Higher entropy as due date approaches
    
    # Priority factor (higher priority = higher entropy)
    priority_factor = task.priority / 5.0 if task.priority is not None else 0.5
    
    # State factor (completed tasks have low entropy)
    state_factor = 0.0 if task.state == TaskState.COMPLETED else (
        0.3 if task.state == TaskState.BLOCKED else (
            0.7 if task.state == TaskState.IN_PROGRESS else 1.0
        )
    )
    
    # Tags factor (more tags = more complexity)
    tags_factor = min(1.0, len(task.tags) / 10.0)
    
    # Calculate final entropy (0-1 scale)
    task.entropy = round(
        0.3 * due_date_factor + 
        0.3 * priority_factor + 
        0.3 * state_factor +
        0.1 * tags_factor,
        3
    )
    
    # Add slight randomization to simulate quantum effects
    task.entropy = min(1.0, max(0.0, task.entropy + random.uniform(-0.05, 0.05)))
    
    # Update probability distribution for the task
    total = 0
    for state in [TaskState.PENDING, TaskState.IN_PROGRESS, TaskState.COMPLETED, TaskState.BLOCKED]:
        if state == task.state:
            # Current state has highest probability
            task.probability_distribution[state] = 0.6 + (random.random() * 0.2)
        else:
            # Other states have lower probabilities
            task.probability_distribution[state] = random.random() * 0.4
        total += task.probability_distribution[state]
    
    # Normalize to ensure sum = 1
    for state in task.probability_distribution:
        task.probability_distribution[state] /= total
        task.probability_distribution[state] = round(task.probability_distribution[state], 3)
    
    # Simulate quantum state as an extended property
    task.quantum_state = {
        "amplitudes": {
            "basis_state_0": {"real": math.sqrt(1 - task.entropy), "imag": 0},
            "basis_state_1": {"real": 0, "imag": math.sqrt(task.entropy)}
        },
        "superposition": task.entropy > 0.3,
        "coherence_time": (1 - task.entropy) * 10,  # 0-10 scale
    }

    # Save updated task
    tasks[task_id] = task
    
def get_entanglement_network():
    """Get the network of task entanglements"""
    G = nx.Graph()
    
    # Add nodes
    for task_id, task in tasks.items():
        G.add_node(task_id, 
                  title=task.title,
                  state=task.state,
                  entropy=task.entropy)
    
    # Add edges
    for entanglement_id, entanglement in entanglements.items():
        G.add_edge(
            entanglement.task_id_1,
            entanglement.task_id_2,
            weight=entanglement.strength,
            type=entanglement.entanglement_type
        )
    
    return G

def propagate_entanglement_effects(source_task_id):
    """Propagate changes through entangled tasks"""
    if source_task_id not in tasks:
        return
    
    # Get the entanglement network
    G = get_entanglement_network()
    
    # Find all neighbors (directly entangled tasks)
    if source_task_id in G:
        neighbors = list(G.neighbors(source_task_id))
        source_task = tasks[source_task_id]
        
        # Propagate effects to neighbors based on entanglement strength
        for neighbor_id in neighbors:
            if neighbor_id in tasks:
                # Get edge data (entanglement properties)
                edge_data = G.get_edge_data(source_task_id, neighbor_id)
                if edge_data:
                    entanglement_strength = edge_data.get('weight', 0.5)
                    
                    # Apply entanglement effects
                    neighbor_task = tasks[neighbor_id]
                    
                    # Update entropy (partial transfer of uncertainty)
                    entropy_change = (source_task.entropy - neighbor_task.entropy) * entanglement_strength * 0.3
                    neighbor_task.entropy = min(1.0, max(0.0, neighbor_task.entropy + entropy_change))
                    
                    # Update probability distribution (quantum-inspired state influence)
                    for state in neighbor_task.probability_distribution:
                        if state in source_task.probability_distribution:
                            # Blend probabilities based on entanglement strength
                            neighbor_prob = neighbor_task.probability_distribution[state]
                            source_prob = source_task.probability_distribution[state]
                            blended_prob = (neighbor_prob * (1 - entanglement_strength * 0.3) + 
                                         source_prob * entanglement_strength * 0.3)
                            neighbor_task.probability_distribution[state] = blended_prob
                    
                    # Normalize the probability distribution
                    total = sum(neighbor_task.probability_distribution.values())
                    for state in neighbor_task.probability_distribution:
                        neighbor_task.probability_distribution[state] /= total
                    
                    # Save updated task
                    tasks[neighbor_id] = neighbor_task

def calculate_system_metrics():
    """Calculate overall system metrics"""
    task_values = list(tasks.values())
    task_count = len(task_values)
    
    if task_count == 0:
        return SystemMetrics(
            total_entropy=0.0,
            task_count=0,
            completion_rate=0.0,
            average_cognitive_load=0.0,
            entanglement_density=0.0,
            quantum_coherence=0.0
        )
    
    # Calculate total entropy
    total_entropy = sum(task.entropy for task in task_values) / task_count
    
    # Calculate completion rate
    completed_tasks = sum(1 for task in task_values if task.state == TaskState.COMPLETED)
    completion_rate = completed_tasks / task_count if task_count > 0 else 0
    
    # Calculate cognitive load (based on task priority and entropy)
    cognitive_load = sum((task.priority / 5) * task.entropy for task in task_values) / task_count
    
    # Calculate entanglement density
    entanglement_count = len(entanglements)
    max_possible_entanglements = (task_count * (task_count - 1)) / 2 if task_count > 1 else 1
    entanglement_density = entanglement_count / max_possible_entanglements if max_possible_entanglements > 0 else 0
    
    # Calculate quantum coherence (inverse of average entropy)
    quantum_coherence = 1.0 - total_entropy
    
    return SystemMetrics(
        total_entropy=round(total_entropy, 3),
        task_count=task_count,
        completion_rate=round(completion_rate, 3),
        average_cognitive_load=round(cognitive_load, 3),
        entanglement_density=round(entanglement_density, 3),
        quantum_coherence=round(quantum_coherence, 3)
    )

# --- FastAPI Application ---
app = FastAPI(
    title="Quantum Task Management API",
    description="A quantum-inspired API for task management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- API Routes ---
@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "operational",
        "quantum_coherence": round(random.random(), 3),
        "api_version": "1.0.0",
        "task_count": len(tasks),
        "entanglement_count": len(entanglements)
    }

@app.get("/tasks", response_model=List[Task])
async def get_tasks():
    """Get all tasks"""
    return list(tasks.values())

@app.post("/tasks", response_model=Task, status_code=201)
async def create_task_api(task_data: TaskCreate, background_tasks: BackgroundTasks):
    """Create a new task"""
    task_id = str(uuid.uuid4())
    now = datetime.now()
    
    # Generate embedding
    task_text = f"{task_data.title} {task_data.description}"
    embedding = calculate_task_embedding(task_text)
    
    # Extract tags if not provided
    tags = task_data.tags
    if not tags:
        tags = extract_tags_from_text(task_text)
    
    # Create initial probability distribution
    prob_dist = create_initial_probability_distribution()
    
    task = Task(
        id=task_id,
        title=task_data.title,
        description=task_data.description,
        assignee=task_data.assignee,
        created_at=now,
        updated_at=now,
        due_date=task_data.due_date,
        state=TaskState.PENDING,
        tags=tags,
        priority=task_data.priority,
        embedding=embedding,
        entangled_tasks=[],
        probability_distribution=prob_dist
    )
    
    # Run ML tasks in background
    def background_ml_tasks():
        try:
            # Classify task
            task.category = classify_task(task_text)
            
            # Generate summary
            task.ml_summary = generate_task_summary(task_text)
            
            # Update entropy and quantum state
            update_task_entropy(task_id)
            
            # Save updated task
            tasks[task_id] = task
        except Exception as e:
            logger.error(f"Background ML tasks error: {str(e)}")
    
    # Store task first to make it available immediately
    tasks[task_id] = task
    
    # Add to graph
    task_graph.add_node(task_id, title=task.title, state=task.state)
    
    # Run background tasks
    background_tasks.add_task(background_ml_tasks)
    
    return task

@app.get("/tasks/{task_id}", response_model=Task)
async def get_task(task_id: str):
    """Get a specific task by ID"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

@app.put("/tasks/{task_id}", response_model=Task)
async def update_task_api(task_id: str, task_update: TaskUpdate, background_tasks: BackgroundTasks):
    """Update a task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    update_data = task_update.dict(exclude_unset=True)
    
    # Track if state changed
    old_state = task.state
    state_changed = "state" in update_data and update_data["state"] != old_state
    
    # Apply updates
    for key, value in update_data.items():
        setattr(task, key, value)
    
    # Update timestamp
    task.updated_at = datetime.now()
    
    # If a significant field changed, update embedding and entropy
    if ("title" in update_data or "description" in update_data or state_changed):
        # Schedule background updates
        def background_update():
            try:
                # Update embedding if title or description changed
                if "title" in update_data or "description" in update_data:
                    task_text = f"{task.title} {task.description}"
                    task.embedding = calculate_task_embedding(task_text)
                    
                    # Update ML fields
                    task.category = classify_task(task_text)
                    task.ml_summary = generate_task_summary(task_text)
                
                # Update entropy and quantum state
                update_task_entropy(task_id)
                
                # If state changed, propagate effects through entanglements
                if state_changed:
                    propagate_entanglement_effects(task_id)
                
                # Save updated task
                tasks[task_id] = task
                
                # Update graph
                if task_id in task_graph:
                    task_graph.nodes[task_id]['state'] = task.state
            except Exception as e:
                logger.error(f"Background update error: {str(e)}")
        
        background_tasks.add_task(background_update)
    
    # Update task immediately to reflect changes
    tasks[task_id] = task
    
    return task

@app.delete("/tasks/{task_id}", response_model=Dict[str, str])
async def delete_task_api(task_id: str):
    """Delete a task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Remove task
    del tasks[task_id]
    
    # Remove from graph
    if task_id in task_graph:
        task_graph.remove_node(task_id)
    
    # Remove related entanglements
    entanglements_to_remove = []
    for ent_id, ent in entanglements.items():
        if ent.task_id_1 == task_id or ent.task_id_2 == task_id:
            entanglements_to_remove.append(ent_id)
    
    for ent_id in entanglements_to_remove:
        del entanglements[ent_id]
    
    return {"status": "success", "message": f"Task {task_id} deleted"}

@app.post("/entanglements", response_model=Entanglement, status_code=201)
async def create_entanglement_api(entanglement_data: EntanglementCreate):
    """Create a new entanglement between tasks"""
    # Validate tasks exist
    if entanglement_data.task_id_1 not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {entanglement_data.task_id_1} not found")
    if entanglement_data.task_id_2 not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {entanglement_data.task_id_2} not found")
    
    # Check if entanglement already exists
    for ent in entanglements.values():
        if ((ent.task_id_1 == entanglement_data.task_id_1 and ent.task_id_2 == entanglement_data.task_id_2) or
            (ent.task_id_1 == entanglement_data.task_id_2 and ent.task_id_2 == entanglement_data.task_id_1)):
            raise HTTPException(status_code=400, detail="Entanglement already exists")
    
    # Create entanglement
    entanglement_id = str(uuid.uuid4())
    now = datetime.now()
    
    entanglement = Entanglement(
        id=entanglement_id,
        task_id_1=entanglement_data.task_id_1,
        task_id_2=entanglement_data.task_id_2,
        strength=entanglement_data.strength,
        entanglement_type=entanglement_data.entanglement_type,
        created_at=now,
        updated_at=now
    )
    
    # Store entanglement
    entanglements[entanglement_id] = entanglement
    
    # Update task entangled_tasks lists
    task1 = tasks[entanglement_data.task_id_1]
    task2 = tasks[entanglement_data.task_id_2]
    
    if entanglement_data.task_id_2 not in task1.entangled_tasks:
        task1.entangled_tasks.append(entanglement_data.task_id_2)
    
    if entanglement_data.task_id_1 not in task2.entangled_tasks:
        task2.entangled_tasks.append(entanglement_data.task_id_1)
    
    # Update graph
    task_graph.add_edge(
        entanglement_data.task_id_1,
        entanglement_data.task_id_2,
        weight=entanglement_data.strength,
        type=entanglement_data.entanglement_type
    )
    
    # Save updated tasks
    tasks[entanglement_data.task_id_1] = task1
    tasks[entanglement_data.task_id_2] = task2
    
    # Apply initial entanglement effects
    propagate_entanglement_effects(entanglement_data.task_id_1)
    
    return entanglement

@app.get("/entanglements", response_model=List[Entanglement])
async def get_entanglements():
    """Get all entanglements"""
    return list(entanglements.values())

@app.get("/entanglements/{entanglement_id}", response_model=Entanglement)
async def get_entanglement(entanglement_id: str):
    """Get a specific entanglement by ID"""
    if entanglement_id not in entanglements:
        raise HTTPException(status_code=404, detail="Entanglement not found")
    return entanglements[entanglement_id]

@app.delete("/entanglements/{entanglement_id}", response_model=Dict[str, str])
async def delete_entanglement_api(entanglement_id: str):
    """Delete an entanglement"""
    if entanglement_id not in entanglements:
        raise HTTPException(status_code=404, detail="Entanglement not found")
    
    # Get entanglement data
    entanglement = entanglements[entanglement_id]
    
    # Remove entanglement
    del entanglements[entanglement_id]
    
    # Update task entangled_tasks lists
    if entanglement.task_id_1 in tasks and entanglement.task_id_2 in tasks[entanglement.task_id_1].entangled_tasks:
        tasks[entanglement.task_id_1].entangled_tasks.remove(entanglement.task_id_2)
    
    if entanglement.task_id_2 in tasks and entanglement.task_id_1 in tasks[entanglement.task_id_2].entangled_tasks:
        tasks[entanglement.task_id_2].entangled_tasks.remove(entanglement.task_id_1)
    
    # Update graph
    if task_graph.has_edge(entanglement.task_id_1, entanglement.task_id_2):
        task_graph.remove_edge(entanglement.task_id_1, entanglement.task_id_2)
    
    return {"status": "success", "message": f"Entanglement {entanglement_id} deleted"}

@app.post("/search", response_model=List[Task])
async def search_tasks_api(search_query: SearchQuery):
    """Search for tasks using vector similarity or text matching"""
    if not tasks:
        return []
    
    results = []
    
    # If we have embeddings model available, use vector search
    if embeddings_model and search_query.query.strip():
        try:
            # Get embedding for query
            query_embedding = embeddings_model.embed_query(search_query.query)
            
            # Calculate similarities with all tasks
            similarities = []
            for task_id, task in tasks.items():
                if task.embedding:
                    # Calculate cosine similarity
                    task_embedding = np.array(task.embedding)
                    query_embedding_array = np.array(query_embedding)
                    
                    similarity = np.dot(task_embedding, query_embedding_array) / (
                        np.linalg.norm(task_embedding) * np.linalg.norm(query_embedding_array)
                    )
                    
                    similarities.append((task, similarity))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top results
            results = [task for task, _ in similarities[:search_query.limit]]
            
        except Exception as e:
            logger.error(f"Vector search error: {str(e)}")
            # Fall through to text search
    
    # If vector search failed or we don't have embeddings, use text search
    if not results:
        query_lower = search_query.query.lower()
        for task in tasks.values():
            if (query_lower in task.title.lower() or 
                query_lower in task.description.lower() or
                any(query_lower in tag.lower() for tag in task.tags)):
                results.append(task)
        
        # Limit results
        results = results[:search_query.limit]
    
    # Apply quantum-inspired randomization if requested
    if search_query.use_quantum and results:
        # Add slight randomization to results order
        randomization_factor = 0.2  # How much to randomize (0-1)
        results_count = len(results)
        
        # Only apply to results with more than 1 item
        if results_count > 1:
            # Calculate swap probability based on entropy
            for i in range(results_count - 1):
                for j in range(i+1, results_count):
                    # Higher entropy = higher swap probability
                    swap_prob = (results[i].entropy + results[j].entropy) * randomization_factor / 2
                    if random.random() < swap_prob:
                        results[i], results[j] = results[j], results[i]
    
    return results

@app.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics():
    """Get system-wide metrics"""
    return calculate_system_metrics()

@app.post("/quantum-simulation", response_model=Dict[str, Any])
async def quantum_simulation_api(simulation_request: QuantumSimulationRequest):
    """Run a quantum simulation on selected tasks"""
    # Validate tasks exist
    for task_id in simulation_request.task_ids:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Set up simulation parameters
    n_steps = simulation_request.simulation_steps
    decoherence_rate = simulation_request.decoherence_rate
    
    # Generate quantum simulation results
    simulation_results = {
        "tasks": [],
        "entanglement_matrix": [],
        "simulation_steps": []
    }
    
    # Add task data
    for task_id in simulation_request.task_ids:
        task = tasks[task_id]
        simulation_results["tasks"].append({
            "id": task.id,
            "title": task.title,
            "state": task.state,
            "entropy": task.entropy,
            "probability_distribution": task.probability_distribution
        })
    
    # Generate entanglement matrix
    task_ids = simulation_request.task_ids
    n_tasks = len(task_ids)
    entanglement_matrix = np.zeros((n_tasks, n_tasks))
    
    for i, task_id1 in enumerate(task_ids):
        for j, task_id2 in enumerate(task_ids):
            if i != j:
                # Check if tasks are entangled
                for ent in entanglements.values():
                    if ((ent.task_id_1 == task_id1 and ent.task_id_2 == task_id2) or
                        (ent.task_id_1 == task_id2 and ent.task_id_2 == task_id1)):
                        entanglement_matrix[i][j] = ent.strength
                        break
    
    simulation_results["entanglement_matrix"] = entanglement_matrix.tolist()
    
    # Run simulation steps
    for step in range(n_steps):
        # Calculate new state for each task
        step_results = {}
        
        for i, task_id in enumerate(task_ids):
            task = tasks[task_id]
            
            # Start with current probabilities
            new_probs = task.probability_distribution.copy()
            
            # Apply entanglement effects
            for j, other_id in enumerate(task_ids):
                if i != j and entanglement_matrix[i][j] > 0:
                    other_task = tasks[other_id]
                    strength = entanglement_matrix[i][j]
                    
                    # Blend probabilities based on entanglement strength
                    for state in new_probs:
                        if state in other_task.probability_distribution:
                            new_probs[state] = (new_probs[state] * (1 - strength * 0.2) + 
                                              other_task.probability_distribution[state] * strength * 0.2)
            
            # Apply decoherence (move toward classical distribution)
            if step > 0:
                for state in new_probs:
                    if state == task.state:
                        # Current state gets higher probability
                        classical_prob = 0.8
                    else:
                        # Other states share remaining probability
                        classical_prob = 0.2 / (len(new_probs) - 1)
                    
                    # Blend with classical distribution based on decoherence rate
                    new_probs[state] = (new_probs[state] * (1 - decoherence_rate) + 
                                      classical_prob * decoherence_rate)
            
            # Calculate new entropy
            min_entropy = 0.1 if task.state != TaskState.COMPLETED else 0.01
            max_entropy = 0.9 if task.state != TaskState.COMPLETED else 0.3
            
            # More uniform distribution = higher entropy
            values = list(new_probs.values())
            entropy = -(sum(p * math.log(p + 1e-10) for p in values) / math.log(len(values)))
            entropy = min(max_entropy, max(min_entropy, entropy))
            
            # Store results
            step_results[task_id] = {
                "probability_distribution": {k: round(v, 3) for k, v in new_probs.items()},
                "entropy": round(entropy, 3),
                "quantum_state": simulate_quantum_state(n_tasks)
            }
        
        simulation_results["simulation_steps"].append(step_results)
    
    return simulation_results

@app.post("/optimize-assignments", response_model=Dict[str, Any])
async def optimize_assignments_api():
    """Run task assignment optimization algorithm"""
    if not tasks:
        return {"assignments": {}, "optimization_score": 0}
    
    # Get assignees and unassigned tasks
    assignees = set()
    unassigned_tasks = []
    
    for task in tasks.values():
        if task.assignee:
            assignees.add(task.assignee)
        elif task.state != TaskState.COMPLETED:
            unassigned_tasks.append(task)
    
    # If no assignees or no unassigned tasks, return early
    if not assignees or not unassigned_tasks:
        return {"assignments": {}, "optimization_score": 0}
    
    # Convert to lists for indexing
    assignee_list = list(assignees)
    
    # Create cost matrix for Hungarian algorithm
    cost_matrix = np.zeros((len(unassigned_tasks), len(assignee_list)))
    
    # Populate cost matrix
    for i, task in enumerate(unassigned_tasks):
        for j, assignee in enumerate(assignee_list):
            # Calculate cost (lower is better)
            # Factors: priority, entropy, due date proximity
            cost = 0
            
            # Priority cost (higher priority = lower cost)
            priority_cost = 5 - task.priority  # 0-4 range (5 is highest priority)
            
            # Entropy cost (higher entropy = higher cost)
            entropy_cost = task.entropy * 5  # 0-5 range
            
            # Due date cost
            due_date_cost = 0
            if task.due_date:
                days_remaining = (task.due_date - datetime.now()).days
                if days_remaining < 0:
                    due_date_cost = 0  # Overdue tasks are urgent (low cost)
                else:
                    due_date_cost = min(5, days_remaining / 3)  # 0-5 range
            else:
                due_date_cost = 3  # No due date is medium priority
            
            # Calculate weighted sum
            cost = 0.4 * priority_cost + 0.3 * entropy_cost + 0.3 * due_date_cost
            
            # Add randomization for quantum effect
            quantum_factor = random.uniform(-0.5, 0.5) * task.entropy
            cost += quantum_factor
            
            cost_matrix[i, j] = cost
    
    # Run Hungarian algorithm
    try:
        row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
        
        # Create assignments dictionary
        assignments = {}
        for i, j in zip(row_ind, col_ind):
            task_id = unassigned_tasks[i].id
            assignee = assignee_list[j]
            assignments[task_id] = assignee
        
        # Calculate optimization score (lower is better)
        total_cost = cost_matrix[row_ind, col_ind].sum()
        max_possible_cost = np.max(cost_matrix) * len(row_ind)
        optimization_score = 1 - (total_cost / max_possible_cost) if max_possible_cost > 0 else 0
        
        return {
            "assignments": assignments,
            "optimization_score": round(optimization_score, 3),
            "task_count": len(assignments)
        }
    
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        return {"assignments": {}, "optimization_score": 0, "error": str(e)}

@app.post("/generate-task", response_model=Dict[str, Any])
async def generate_task_api(data: Dict[str, Any]):
    """Generate a task using LLM"""
    topic = data.get("topic", "")
    context = data.get("context", "")
    
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")
    
    try:
        llm = get_llm()
        if not llm:
            return {
                "title": f"Task about {topic}",
                "description": "Could not generate task description due to model unavailability."
            }
        
        # Create task generation prompt
        prompt = PromptTemplate.from_template(
            """Generate a realistic task given the topic and context. Include a concise title and detailed description.
            
            Topic: {topic}
            Context: {context}
            
            Respond in the following JSON format:
            {{
                "title": "Task title here",
                "description": "Detailed task description here",
                "priority": 1-5 (1 is lowest, 5 is highest),
                "tags": ["tag1", "tag2", "tag3"]
            }}
            
            JSON response:"""
        )
        
        # Create chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Run chain
        result = chain.run(topic=topic, context=context).strip()
        
        # Parse JSON response
        try:
            # Find the JSON part of the response if there are other tokens
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_result = result[json_start:json_end]
                task_data = json.loads(json_result)
                
                # Validate required fields
                if "title" not in task_data or "description" not in task_data:
                    raise ValueError("Missing required fields in generated task")
                
                return task_data
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            logger.error(f"JSON parsing error: {str(e)}")
            # Return a basic response with the raw text
            return {
                "title": f"Task about {topic}",
                "description": result,
                "priority": 3,
                "tags": [topic]
            }
            
    except Exception as e:
        logger.error(f"Task generation error: {str(e)}")
        return {
            "title": f"Task about {topic}",
            "description": f"A task related to {topic} - {context}",
            "priority": 3,
            "tags": [topic]
        }

# Initialize some sample data
def init_sample_data():
    """Initialize some sample data for testing"""
    # Only initialize if no tasks exist
    if not tasks:
        # Create sample tasks
        sample_tasks = [
            {
                "title": "Implement quantum entanglement visualization",
                "description": "Create a 3D visualization of quantum entanglement between tasks using Three.js or similar library.",
                "assignee": "quantum_dev",
                "priority": 4,
                "tags": ["frontend", "visualization", "quantum"]
            },
            {
                "title": "Fix task entropy calculation",
                "description": "The entropy calculation for completed tasks is too high. Adjust the algorithm to reduce entropy for completed tasks.",
                "assignee": "bug_fixer",
                "priority": 3,
                "tags": ["backend", "bug", "algorithm"]
            },
            {
                "title": "Write API documentation",
                "description": "Create comprehensive API documentation for all endpoints using OpenAPI schema.",
                "assignee": None,
                "priority": 2,
                "tags": ["documentation", "api"]
            }
        ]
        
        for task_data in sample_tasks:
            task_create = TaskCreate(
                title=task_data["title"],
                description=task_data["description"],
                assignee=task_data["assignee"],
                tags=task_data["tags"],
                priority=task_data["priority"]
            )
            
            # Create task (synchronously for initialization)
            task_id = str(uuid.uuid4())
            now = datetime.now()
            
            task_text = f"{task_data['title']} {task_data['description']}"
            embedding = calculate_task_embedding(task_text)
            
            prob_dist = create_initial_probability_distribution()
            
            task = Task(
                id=task_id,
                title=task_data["title"],
                description=task_data["description"],
                assignee=task_data["assignee"],
                created_at=now,
                updated_at=now,
                due_date=None,
                state=TaskState.PENDING,
                tags=task_data["tags"],
                priority=task_data["priority"],
                embedding=embedding,
                entangled_tasks=[],
                probability_distribution=prob_dist
            )
            
            # Classify and summarize
            task.category = classify_task(task_text)
            task.ml_summary = generate_task_summary(task_text)
            
            # Update entropy
            update_task_entropy(task_id)
            
            # Store task
            tasks[task_id] = task
            
            # Add to graph
            task_graph.add_node(task_id, title=task.title, state=task.state)
        
        # Create sample entanglements between tasks if we have at least 2 tasks
        task_ids = list(tasks.keys())
        if len(task_ids) >= 2:
            entanglement_data = EntanglementCreate(
                task_id_1=task_ids[0],
                task_id_2=task_ids[1],
                strength=0.7,
                entanglement_type="standard"
            )
            
            # Create entanglement
            entanglement_id = str(uuid.uuid4())
            now = datetime.now()
            
            entanglement = Entanglement(
                id=entanglement_id,
                task_id_1=entanglement_data.task_id_1,
                task_id_2=entanglement_data.task_id_2,
                strength=entanglement_data.strength,
                entanglement_type=entanglement_data.entanglement_type,
                created_at=now,
                updated_at=now
            )
            
            # Store entanglement
            entanglements[entanglement_id] = entanglement
            
            # Update task entangled_tasks lists
            task1 = tasks[entanglement_data.task_id_1]
            task2 = tasks[entanglement_data.task_id_2]
            
            task1.entangled_tasks.append(entanglement_data.task_id_2)
            task2.entangled_tasks.append(entanglement_data.task_id_1)
            
            # Update graph
            task_graph.add_edge(
                entanglement_data.task_id_1,
                entanglement_data.task_id_2,
                weight=entanglement_data.strength,
                type=entanglement_data.entanglement_type
            )
            
            # Save updated tasks
            tasks[entanglement_data.task_id_1] = task1
            tasks[entanglement_data.task_id_2] = task2
            
            # Apply initial entanglement effects
            propagate_entanglement_effects(entanglement_data.task_id_1)

# Initialize vector store
def init_vector_store():
    """Initialize vector store for semantic search"""
    global vector_store
    
    # Only initialize if we have tasks and embeddings_model
    if tasks and embeddings_model:
        try:
            documents = []
            metadatas = []
            ids = []
            
            for task_id, task in tasks.items():
                text = f"{task.title} {task.description}"
                documents.append(text)
                metadatas.append({"task_id": task_id})
                ids.append(task_id)
            
            # Create vector store
            vector_store = Chroma.from_texts(
                texts=documents,
                embedding=embeddings_model,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info("Vector store initialized")
        except Exception as e:
            logger.error(f"Vector store initialization failed: {str(e)}")

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    """Initialize data and services on startup"""
    # Initialize sample data
    init_sample_data()
    
    # Initialize vector store
    init_vector_store()
    
    logger.info("API started successfully")

# Run the server when executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
