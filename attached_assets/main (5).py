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

# ML model imports - single consolidated block
try:
    from transformers import (
        pipeline,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForQuestionAnswering,
        AutoModelForCausalLM
    )
    from sentence_transformers import SentenceTransformer
    import torch
    from torch.multiprocessing import Pool, set_start_method
    
    # Try to set start method for torch multiprocessing
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        logger.warning("Could not set multiprocessing start method to 'spawn'")
    
    ML_LIBS_AVAILABLE = True
except ImportError:
    logger.warning("ML libraries not available. Running with limited functionality.")
    ML_LIBS_AVAILABLE = False

# Vector database - consistent community imports
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document  # Updated import for Document
    VECTOR_DB_AVAILABLE = True
except ImportError:
    logger.warning("Vector database libraries not available. Running with limited functionality.")
    VECTOR_DB_AVAILABLE = False

# Advanced quantum simulation
try:
    import qutip as qt
    from pennylane import numpy as qnp
    import pennylane as qml
    QUANTUM_LIBS_AVAILABLE = True
except ImportError:
    logger.warning("Quantum libraries not available. Running with limited functionality.")
    QUANTUM_LIBS_AVAILABLE = False

# --- Configure CPU optimization ---
# Determine optimal thread count for the system
CPU_COUNT = os.cpu_count() or 4
EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT)

# --- ML Models Management ---
ml_models = {}
ml_model_usage = {}  # Track when models were last used
MAX_MODELS_LOADED = 2  # Maximum number of models to keep in memory

def get_ml_model(model_type):
    """Lazy-load ML models when needed with memory management"""
    if not ML_LIBS_AVAILABLE:
        logger.warning(f"ML libraries not available. Cannot load {model_type} model.")
        return None

    global ml_models, ml_model_usage

    # Update usage time if model is already loaded
    if model_type in ml_models:
        ml_model_usage[model_type] = datetime.now()
        return ml_models.get(model_type)

    # Check if we need to unload a model to free memory
    if len(ml_models) >= MAX_MODELS_LOADED:
        # Find least recently used model
        lru_model = min(ml_model_usage.items(), key=lambda x: x[1])[0]
        logger.info(f"Unloading model {lru_model} to free memory")
        ml_models.pop(lru_model)
        ml_model_usage.pop(lru_model)

    # Now load the requested model
    try:
        logger.info(f"Loading ML model: {model_type}")

        if model_type == "task_classifier":
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
            ml_models[model_type] = (tokenizer, pipeline("text-classification", model=model, tokenizer=tokenizer))

        elif model_type == "task_embedding":
            # Use explicit cache dir to avoid potential path issues
            model_name = "all-MiniLM-L6-v2"
            model_path = os.path.join(os.getcwd(), "models", model_name)
            
            # Check if model exists locally, download if needed
            if not os.path.exists(model_path):
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                logger.info(f"Model not found locally. Will download to {model_path}")
                ml_models[model_type] = SentenceTransformer(model_name)
            else:
                logger.info(f"Loading model from local path: {model_path}")
                ml_models[model_type] = SentenceTransformer(model_path)

        elif model_type == "text_generator":
            # Use a smaller model by default for reliability
            tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            ml_models[model_type] = (tokenizer, model)

        elif model_type == "qa_assistant":
            tokenizer = AutoTokenizer.from_pretrained("deepset/minilm-uncased-squad2")
            model = AutoModelForQuestionAnswering.from_pretrained("deepset/minilm-uncased-squad2")
            ml_models[model_type] = (tokenizer, pipeline("question-answering", model=model, tokenizer=tokenizer))

        # Update usage time
        ml_model_usage[model_type] = datetime.now()
        logger.info(f"Successfully loaded {model_type} model")

    except Exception as e:
        logger.error(f"Error loading {model_type}: {str(e)}")
        return None

    return ml_models.get(model_type)

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
    category: Optional[str] = None  # ML-predicted category
    ml_summary: Optional[str] = None  # ML-generated summary or priority notes

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

# --- In-memory database ---
tasks = {}
entanglements = {}
task_graph = nx.Graph()

# Vector store for semantic search
embeddings_model = None
vector_store = None

# --- Advanced Quantum-inspired utility functions ---

@lru_cache(maxsize=128)
def calculate_task_embedding(task_text):
    """Calculate embedding for task text using ML model or fallback."""
    try:
        # Try to use the ML embedding model
        embedding_model = get_ml_model("task_embedding")
        if embedding_model:
            # Get embedding from sentence transformer
            embedding = embedding_model.encode(task_text)
            
            # Convert to list for serialization
            return embedding.tolist()
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
    """Classify task using ML model or fallback rules."""
    try:
        classifier = get_ml_model("task_classifier")
        if classifier and classifier[1] is not None:
            tokenizer, pipeline = classifier
            result = pipeline(task_text)
            return result[0]['label']
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
    """Generate a summary for a task using ML or simple extraction."""
    try:
        generator = get_ml_model("text_generator")
        if generator and generator[0] is not None and generator[1] is not None:
            tokenizer, model = generator
            inputs = tokenizer(f"Summarize: {task_text}", return_tensors="pt", max_length=100, truncation=True)
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=2,
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
    except Exception as e:
        logger.error(f"Summary generation error: {str(e)}")

    # Simple fallback - extract first sentence
    first_sentence = task_text.split('.')[0]
    if len(first_sentence) > max_length:
        return first_sentence[:max_length] + "..."
    return first_sentence

def answer_task_question(question, context):
    """Answer a question about a task using QA model or simple extraction."""
    try:
        qa_model = get_ml_model("qa_assistant")
        if qa_model and qa_model[1] is not None:
            tokenizer, pipeline = qa_model
            answer = pipeline(question=question, context=context)
            return answer
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
        focus = question_lower[focus_index:].strip()

        # Look for the focus in the context
        if focus in context_lower:
            focus_index = context_lower.index(focus)
            # Extract a window of text around the focus
            start = max(0, focus_index - 50)
            end = min(len(context_lower), focus_index + 100)
            answer_text = context[start:end]
            return {"answer": answer_text, "score": 0.5}

    # If no match found or no focus word identified
    return {"answer": "Information not found in the task details.", "score": 0.0}

async def update_task_entropy(task_id, decay_factor=0.95):
    """Update task entropy based on quantum decoherence principles."""
    if task_id in tasks:
        task = tasks[task_id]
        # Entropy naturally decays over time
        task.entropy *= decay_factor

        # More information reduces entropy (more views/updates)
        interaction_factor = 0.9  # Each interaction reduces entropy
        task.entropy *= interaction_factor

        # Apply noise effects
        coherence_factor = random.uniform(0.98, 1.02)  # Small fluctuations
        task.entropy *= coherence_factor

        # Ensure entropy stays within bounds
        task.entropy = max(0.1, min(1.0, task.entropy))

        # Update probability distribution
        await update_probability_distribution(task_id)

        # Update quantum state if defined
        if task.quantum_state:
            task.quantum_state["fidelity"] *= decay_factor
            # Simulate decoherence over time
            decoherence = 1.0 - (1.0 - task.entropy) * 0.5
            task.quantum_state["coherence_time"] *= decoherence

        return task.entropy
    return None

async def update_probability_distribution(task_id):
    """Update the probability distribution of task states."""
    if task_id in tasks:
        task = tasks[task_id]
        states = [TaskState.PENDING, TaskState.IN_PROGRESS, TaskState.COMPLETED, TaskState.BLOCKED]

        # Current state has higher probability
        probs = [0.1, 0.1, 0.1, 0.1]  # Base probabilities
        current_state_idx = states.index(task.state)
        probs[current_state_idx] = max(0.5, 1 - task.entropy)

        # Distribute remaining probability based on entropy
        remaining = 1.0 - probs[current_state_idx]
        for i in range(len(states)):
            if i != current_state_idx:
                probs[i] = remaining / (len(states) - 1)

        # Update task probability distribution
        task.probability_distribution = {states[i]: float(probs[i]) for i in range(len(states))}

        # Store quantum state information
        if not task.quantum_state:
            task.quantum_state = {
                "fidelity": 1.0,
                "coherence_time": 1.0,
                "eigenvalues": [0.7, 0.2, 0.05, 0.05],
                "amplitudes": {}
            }

        # Calculate amplitudes
        amplitudes = {}
        for i, state in enumerate(states):
            real_part = math.sqrt(probs[i])
            imag_part = 0.0
            amplitudes[state] = {"real": float(real_part), "imag": float(imag_part)}

        task.quantum_state["amplitudes"] = amplitudes
        tasks[task_id] = task

async def propagate_entanglement(task_id):
    """Propagate changes through entangled tasks."""
    if task_id not in tasks:
        return

    visited = set()
    to_visit = [task_id]

    while to_visit:
        current_id = to_visit.pop(0)
        if current_id in visited:
            continue

        visited.add(current_id)
        current_task = tasks[current_id]

        # Get all connected entanglements
        connected_entanglements = []
        for e_id, entanglement in entanglements.items():
            if entanglement.task_id_1 == current_id or entanglement.task_id_2 == current_id:
                connected_entanglements.append((e_id, entanglement))

        # Process entanglements
        for e_id, entanglement in connected_entanglements:
            other_id = None
            if entanglement.task_id_1 == current_id:
                other_id = entanglement.task_id_2
            elif entanglement.task_id_2 == current_id:
                other_id = entanglement.task_id_1

            if not other_id or other_id not in tasks:
                continue

            # Apply entanglement effects
            other_task = tasks[other_id]

            # Simple entanglement effects
            if entanglement.entanglement_type == "CNOT":
                # Control-NOT like operation
                if current_task.state == TaskState.COMPLETED and other_task.state == TaskState.PENDING:
                    if random.random() < entanglement.strength * 0.5:
                        other_task.state = TaskState.IN_PROGRESS
                        await update_probability_distribution(other_id)

            elif entanglement.entanglement_type == "SWAP":
                # SWAP-like operation
                if random.random() < entanglement.strength * 0.3:
                    # Swap priorities
                    current_task.priority, other_task.priority = other_task.priority, current_task.priority
                    await update_probability_distribution(current_id)
                    await update_probability_distribution(other_id)

            else:  # Standard entanglement
                # Apply influence on entropy
                entropy_shift = (current_task.entropy - other_task.entropy) * entanglement.strength * 0.3
                other_task.entropy = max(0.1, min(1.0, other_task.entropy + entropy_shift))
                await update_probability_distribution(other_id)

            # Add to visit queue if not visited
            if other_id not in visited:
                to_visit.append(other_id)

def calculate_task_similarity(task1, task2):
    """Calculate similarity between tasks."""
    if not task1.embedding or not task2.embedding:
        return 0.0

    # Standard vector similarity calculation
    try:
        embedding1 = np.array(task1.embedding)
        embedding2 = np.array(task2.embedding)

        # Normalize embeddings
        embedding1 = embedding1 / (np.linalg.norm(embedding1) + 1e-6)
        embedding2 = embedding2 / (np.linalg.norm(embedding2) + 1e-6)

        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)

        # Scale to 0-1 range
        similarity = (similarity + 1) / 2
    except Exception as e:
        logger.error(f"Similarity calculation error: {str(e)}")
        similarity = 0.0

    # Add tag-based similarity component
    common_tags = set(task1.tags).intersection(set(task2.tags))
    tag_similarity = len(common_tags) / max(1, len(set(task1.tags).union(set(task2.tags))))

    # Weighted combination
    return 0.7 * similarity + 0.3 * tag_similarity

def suggest_entanglements(task_id, threshold=0.7):
    """Suggest possible task entanglements based on similarity."""
    if task_id not in tasks:
        return []

    target_task = tasks[task_id]
    suggestions = []

    # Check similarity with each task
    for other_id, other_task in tasks.items():
        if other_id == task_id:
            continue

        # Calculate similarity
        similarity = calculate_task_similarity(target_task, other_task)

        # Check if already entangled
        already_entangled = any(
            (e.task_id_1 == task_id and e.task_id_2 == other_id) or
            (e.task_id_1 == other_id and e.task_id_2 == task_id)
            for e in entanglements.values()
        )

        if similarity >= threshold and not already_entangled:
            suggestions.append({
                "task_id": other_id,
                "title": other_task.title,
                "similarity": similarity
            })

    # Sort by similarity descending
    suggestions.sort(key=lambda x: x["similarity"], reverse=True)
    return suggestions

def optimize_task_assignment(tasks_dict):
    """Optimize task assignment to balance workload."""
    # Extract assignees and their tasks
    assignees = {}
    for task_id, task in tasks_dict.items():
        if task.assignee:
            if task.assignee not in assignees:
                assignees[task.assignee] = []
            assignees[task.assignee].append(task_id)

    if not assignees:
        return []

    # Calculate load per assignee
    cognitive_loads = {}
    for assignee, task_ids in assignees.items():
        # Sum of task complexities
        tasks_load = sum((tasks_dict[tid].entropy * tasks_dict[tid].priority) for tid in task_ids)
        # Adjustment for number of tasks
        adjustment = 1 + (0.1 * (len(task_ids) - 1))

        total_load = tasks_load * adjustment
        cognitive_loads[assignee] = total_load

    # Identify overloaded and underloaded assignees
    mean_load = sum(cognitive_loads.values()) / len(cognitive_loads)
    overloaded = [a for a, load in cognitive_loads.items() if load > mean_load * 1.2]
    underloaded = [a for a, load in cognitive_loads.items() if load < mean_load * 0.8]

    # Generate recommendations
    recommendations = []

    # For each overloaded assignee
    for over_assignee in overloaded:
        over_tasks = [tid for tid in assignees[over_assignee]]

        # Sort tasks by load contribution (priority * entropy)
        task_loads = [(tid, tasks_dict[tid].entropy * tasks_dict[tid].priority) for tid in over_tasks]
        task_loads.sort(key=lambda x: x[1], reverse=True)

        # For each underloaded assignee
        for under_assignee in underloaded:
            # Try redistributing highest load tasks
            for task_id, load in task_loads[:2]:  # Consider top 2 tasks
                # Calculate new loads
                new_over_load = cognitive_loads[over_assignee] - load
                new_under_load = cognitive_loads[under_assignee] + load

                # Only recommend if it improves balance
                if new_over_load >= new_under_load * 0.8 and new_over_load <= new_under_load * 1.2:
                    recommendations.append({
                        "task_id": task_id,
                        "task_title": tasks_dict[task_id].title,
                        "from_assignee": over_assignee,
                        "to_assignee": under_assignee,
                        "load_improvement": cognitive_loads[over_assignee] - new_over_load
                    })
                    break  # Only one recommendation per assignee pair

    # Sort by load improvement
    recommendations.sort(key=lambda x: x.get("load_improvement", 0), reverse=True)
    return recommendations

def calculate_system_metrics():
    """Calculate system-wide metrics."""
    if not tasks:
        return SystemMetrics(
            total_entropy=0,
            task_count=0,
            completion_rate=0,
            average_cognitive_load=0,
            entanglement_density=0,
            quantum_coherence=0.0
        )

    # Task count
    task_count = len(tasks)

    # Total entropy
    total_entropy = sum(task.entropy for task in tasks.values())

    # Completion rate
    completed_tasks = sum(1 for task in tasks.values() if task.state == TaskState.COMPLETED)
    completion_rate = completed_tasks / task_count if task_count > 0 else 0

    # Average cognitive load
    avg_cognitive_load = sum(task.priority * task.entropy for task in tasks.values()) / task_count if task_count > 0 else 0

    # Entanglement density
    max_possible_entanglements = task_count * (task_count - 1) / 2 if task_count > 1 else 1
    entanglement_density = len(entanglements) / max_possible_entanglements

    # Quantum coherence (simplified)
    quantum_coherence = 1.0 - (total_entropy / task_count) if task_count > 0 else 0.0

    # Create and return metrics
    return SystemMetrics(
        total_entropy=total_entropy,
        task_count=task_count,
        completion_rate=completion_rate,
        average_cognitive_load=avg_cognitive_load,
        entanglement_density=entanglement_density,
        quantum_coherence=quantum_coherence
    )

# --- Quantum Simulation Functions ---

def simulate_quantum_circuit(tasks_dict, task_ids, steps=5):
    """Run a simplified quantum circuit simulation on selected tasks."""
    if not QUANTUM_LIBS_AVAILABLE:
        # Simplified simulation without quantum libraries
        return simulate_simplified_circuit(tasks_dict, task_ids, steps)

    if not task_ids or len(task_ids) > 8:  # Limit to 8 tasks for performance
        return {"error": "Invalid number of tasks (should be 1-8)"}

    # Select tasks from dictionary
    selected_tasks = {tid: tasks_dict[tid] for tid in task_ids if tid in tasks_dict}
    if not selected_tasks:
        return {"error": "No valid tasks found"}

    try:
        # Use qutip for simulation
        import qutip as qt

        # Initial states based on task entropy
        states = []
        for tid in task_ids:
            if tid in selected_tasks:
                task = selected_tasks[tid]
                # Lower entropy -> closer to pure state
                purity = 1.0 - task.entropy
                # Initial state: weighted superposition
                state = math.sqrt(purity) * qt.basis(2, 0) + math.sqrt(1-purity) * qt.basis(2, 1)
                states.append(state)
            else:
                # Default state if task not found
                states.append(qt.basis(2, 0))

        # Create composite state
        initial_state = states[0]
        for state in states[1:]:
            initial_state = qt.tensor(initial_state, state)

        # Run simulation steps
        current_state = initial_state
        results = []

        for step in range(steps):
            # Apply random quantum operations
            op_type = random.choice(["CNOT", "SWAP", "Hadamard"])

            try:
                # Apply the selected operation
                if op_type == "Hadamard":
                    # Apply Hadamard to random qubit
                    qubit = random.randint(0, len(task_ids)-1)
                    h_gate = qt.hadamard_transform()
                    gate = qt.gate_expand_1toN(h_gate, len(task_ids), qubit)
                    current_state = gate * current_state
                elif len(task_ids) > 1:  # CNOT and SWAP need at least 2 qubits
                    if op_type == "CNOT":
                        # Apply CNOT between two random qubits
                        control = random.randint(0, len(task_ids)-1)
                        target = random.randint(0, len(task_ids)-1)
                        while target == control:
                            target = random.randint(0, len(task_ids)-1)

                        # Create CNOT matrix for these specific qubits
                        cnot = qt.cnot()
                        gate = qt.gate_expand_2toN(cnot, len(task_ids), control, target)
                        current_state = gate * current_state
                    elif op_type == "SWAP":
                        # Apply SWAP between two random qubits
                        q1 = random.randint(0, len(task_ids)-1)
                        q2 = random.randint(0, len(task_ids)-1)
                        while q2 == q1:
                            q2 = random.randint(0, len(task_ids)-1)

                        # Create SWAP matrix
                        swap = qt.swap()
                        gate = qt.gate_expand_2toN(swap, len(task_ids), q1, q2)
                        current_state = gate * current_state
            except Exception as e:
                logger.error(f"Quantum operation error: {str(e)}")
                # If operation fails, apply identity (do nothing)
                op_type = "Identity"

               # Calculate probabilities after this step
            dm = current_state * current_state.dag()

            # Extract results for this step
            step_result = {
                "step": step,
                "operation": op_type,
                "task_states": {}
            }

            # Calculate individual task states
            for i, tid in enumerate(task_ids):
                # Partial trace to get reduced density matrix for this task
                try:
                    reduced_dm = dm.ptrace(i)

                    # Calculate probabilities
                    prob_0 = float(reduced_dm[0,0])  # |0⟩ probability
                    prob_1 = float(reduced_dm[1,1])  # |1⟩ probability

                    # Coherence - off-diagonal elements
                    coherence = float(abs(reduced_dm[0,1]))

                    step_result["task_states"][tid] = {
                        "pending_prob": prob_0,
                        "completed_prob": prob_1,
                        "coherence": coherence,
                        "task_title": selected_tasks[tid].title if tid in selected_tasks else "Unknown"
                    }
                except Exception as e:
                    logger.error(f"Error calculating task state: {str(e)}")
                    step_result["task_states"][tid] = {
                        "pending_prob": 0.5,
                        "completed_prob": 0.5,
                        "coherence": 0.0,
                        "task_title": selected_tasks[tid].title if tid in selected_tasks else "Unknown",
                        "error": str(e)
                    }

            results.append(step_result)

        # Final measurement simulation
        final_result = {
            "final_state": {},
            "entanglement_measure": {},
            "measurement_outcomes": {}
        }

        # Calculate final state probabilities
        final_dm = current_state * current_state.dag()

        # Calculate pairwise entanglement between tasks
        for i, tid1 in enumerate(task_ids):
            for j, tid2 in enumerate(task_ids):
                if i < j:  # Process each pair once
                    try:
                        # Calculate simplified entanglement measure
                        sub_systems = [i, j]
                        rho_ij = final_dm.ptrace(sub_systems)
                        # Approximate concurrence
                        concurrence = 2 * abs(float(rho_ij[0,3]) if rho_ij.shape[0] > 3 else 0)

                        final_result["entanglement_measure"][f"{tid1}-{tid2}"] = {
                            "concurrence": concurrence,
                            "task1_title": selected_tasks.get(tid1, Task(id="unknown", title="Unknown", description="", created_at=datetime.now(), updated_at=datetime.now())).title,
                            "task2_title": selected_tasks.get(tid2, Task(id="unknown", title="Unknown", description="", created_at=datetime.now(), updated_at=datetime.now())).title
                        }
                    except Exception as e:
                        logger.error(f"Entanglement calculation error: {str(e)}")
                        final_result["entanglement_measure"][f"{tid1}-{tid2}"] = {
                            "error": str(e),
                            "task1_title": selected_tasks.get(tid1, Task(id="unknown", title="Unknown", description="", created_at=datetime.now(), updated_at=datetime.now())).title,
                            "task2_title": selected_tasks.get(tid2, Task(id="unknown", title="Unknown", description="", created_at=datetime.now(), updated_at=datetime.now())).title
                        }

        # Simulate measurements
        measurements = {}
        for i, tid in enumerate(task_ids):
            try:
                # Get reduced density matrix for this task
                reduced_dm = final_dm.ptrace(i)

                # Extract probabilities
                prob_0 = float(reduced_dm[0,0])  # |0⟩ probability (PENDING)
                prob_1 = float(reduced_dm[1,1])  # |1⟩ probability (COMPLETED)

                # Simulate measurement outcome
                if random.random() < prob_1:
                    outcome = "COMPLETED"
                else:
                    outcome = "PENDING"

                measurements[tid] = {
                    "outcome": outcome,
                    "pending_prob": prob_0,
                    "completed_prob": prob_1,
                    "task_title": selected_tasks.get(tid, Task(id="unknown", title="Unknown", description="", created_at=datetime.now(), updated_at=datetime.now())).title
                }
            except Exception as e:
                logger.error(f"Measurement error: {str(e)}")
                measurements[tid] = {
                    "outcome": "PENDING",
                    "error": str(e),
                    "task_title": selected_tasks.get(tid, Task(id="unknown", title="Unknown", description="", created_at=datetime.now(), updated_at=datetime.now())).title
                }

        final_result["measurement_outcomes"] = measurements

        return {
            "simulation_steps": results,
            "final_results": final_result
        }

    except Exception as e:
        logger.error(f"Quantum simulation error: {str(e)}")
        return {"error": f"Simulation failed: {str(e)}"}

def simulate_simplified_circuit(tasks_dict, task_ids, steps=5):
    """Fallback simulation when quantum libraries are not available."""
    if not task_ids or len(task_ids) > 8:
        return {"error": "Invalid number of tasks (should be 1-8)"}

    # Select tasks from dictionary
    selected_tasks = {tid: tasks_dict[tid] for tid in task_ids if tid in tasks_dict}
    if not selected_tasks:
        return {"error": "No valid tasks found"}

    # Run simplified simulation
    results = []

    # Initial state based on task entropy
    task_states = {}
    for tid in task_ids:
        if tid in selected_tasks:
            task = selected_tasks[tid]
            # Lower entropy -> higher pending probability
            pending_prob = 1.0 - task.entropy * 0.5
            completed_prob = 1.0 - pending_prob
            coherence = task.entropy * 0.3  # Simplified coherence measure

            task_states[tid] = {
                "pending_prob": pending_prob,
                "completed_prob": completed_prob,
                "coherence": coherence
            }
        else:
            # Default state if task not found
            task_states[tid] = {
                "pending_prob": 0.8,
                "completed_prob": 0.2,
                "coherence": 0.1
            }

    # Simulate steps
    current_states = task_states.copy()

    for step in range(steps):
        # Apply random operation
        op_type = random.choice(["CNOT", "SWAP", "Hadamard", "Noise"])

        # Create result for this step
        step_result = {
            "step": step,
            "operation": op_type,
            "task_states": {}
        }

        # Apply operation effects
        if op_type == "Hadamard":
            # Apply Hadamard to random task - moves towards equal superposition
            random_task = random.choice(task_ids)
            if random_task in current_states:
                # Move probabilities closer to 0.5
                state = current_states[random_task]
                state["pending_prob"] = 0.5 + (state["pending_prob"] - 0.5) * 0.3
                state["completed_prob"] = 1.0 - state["pending_prob"]
                state["coherence"] = min(1.0, state["coherence"] + 0.2)  # Increase coherence

        elif op_type == "CNOT" and len(task_ids) > 1:
            # Apply CNOT between two random tasks
            control = random.choice(task_ids)
            target = random.choice([t for t in task_ids if t != control])

            if control in current_states and target in current_states:
                # If control has high completed_prob, it influences target
                if current_states[control]["completed_prob"] > 0.7:
                    current_states[target]["completed_prob"] = (
                        current_states[target]["completed_prob"] * 0.3 +
                        current_states[control]["completed_prob"] * 0.7
                    )
                    current_states[target]["pending_prob"] = 1.0 - current_states[target]["completed_prob"]

        elif op_type == "SWAP" and len(task_ids) > 1:
            # Swap probabilities between two random tasks
            t1 = random.choice(task_ids)
            t2 = random.choice([t for t in task_ids if t != t1])

            if t1 in current_states and t2 in current_states:
                # Swap probabilities
                current_states[t1]["pending_prob"], current_states[t2]["pending_prob"] = (
                    current_states[t2]["pending_prob"], current_states[t1]["pending_prob"]
                )
                current_states[t1]["completed_prob"], current_states[t2]["completed_prob"] = (
                    current_states[t2]["completed_prob"], current_states[t1]["completed_prob"]
                )

        elif op_type == "Noise":
            # Apply random noise to all tasks
            for tid in task_ids:
                if tid in current_states:
                    noise = random.uniform(-0.1, 0.1)
                    current_states[tid]["pending_prob"] = max(0.1, min(0.9, current_states[tid]["pending_prob"] + noise))
                    current_states[tid]["completed_prob"] = 1.0 - current_states[tid]["pending_prob"]
                    current_states[tid]["coherence"] = max(0.0, current_states[tid]["coherence"] - 0.05)  # Decrease coherence

        # Copy states to result
        for tid in task_ids:
            if tid in current_states:
                state = current_states[tid].copy()
                state["task_title"] = selected_tasks[tid].title if tid in selected_tasks else "Unknown"
                step_result["task_states"][tid] = state

        results.append(step_result)

    # Final measurement simulation
    final_result = {
        "final_state": {},
        "entanglement_measure": {},
        "measurement_outcomes": {}
    }

    # Generate some simplified entanglement measures
    for i, tid1 in enumerate(task_ids):
        for j, tid2 in enumerate(task_ids):
            if i < j:  # Process each pair once
                # Calculate simplified entanglement
                if tid1 in current_states and tid2 in current_states:
                    # Coherence product as entanglement measure
                    coherence1 = current_states[tid1]["coherence"]
                    coherence2 = current_states[tid2]["coherence"]
                    concurrence = coherence1 * coherence2 * random.uniform(0.8, 1.2)

                    final_result["entanglement_measure"][f"{tid1}-{tid2}"] = {
                        "concurrence": min(1.0, concurrence),
                        "task1_title": selected_tasks.get(tid1, Task(id="unknown", title="Unknown", description="", created_at=datetime.now(), updated_at=datetime.now())).title,
                        "task2_title": selected_tasks.get(tid2, Task(id="unknown", title="Unknown", description="", created_at=datetime.now(), updated_at=datetime.now())).title
                    }

    # Simulate measurements
    measurements = {}
    for tid in task_ids:
        if tid in current_states:
            # Simulate measurement based on probabilities
            if random.random() < current_states[tid]["completed_prob"]:
                outcome = "COMPLETED"
            else:
                outcome = "PENDING"

            measurements[tid] = {
                "outcome": outcome,
                "pending_prob": current_states[tid]["pending_prob"],
                "completed_prob": current_states[tid]["completed_prob"],
                "task_title": selected_tasks.get(tid, Task(id="unknown", title="Unknown", description="", created_at=datetime.now(), updated_at=datetime.now())).title
            }

    final_result["measurement_outcomes"] = measurements

    return {
        "simulation_steps": results,
        "final_results": final_result,
        "note": "Simplified simulation (quantum libraries not available)"
    }

# --- API Endpoints ---

# Define lifespan for FastAPI app
async def lifespan(app: FastAPI):
    """App initialization and cleanup"""
    # Startup event
    logger.info("Starting application...")

    # Initialize embeddings model if vector database is available
    global embeddings_model, vector_store

    if VECTOR_DB_AVAILABLE:
        try:
            # Initialize embeddings model with explicit model path
            model_name = "all-MiniLM-L6-v2"
            model_path = os.path.join(os.getcwd(), "models", model_name)
            
            # Ensure model directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            logger.info(f"Initializing embeddings with model path: {model_path}")
            
            try:
                # First try to load from local path if it exists
                if os.path.exists(model_path):
                    logger.info(f"Loading embedding model from local path: {model_path}")
                    embeddings_model = HuggingFaceEmbeddings(model_name=model_path)
                else:
                    # Otherwise load from HuggingFace directly with explicit cache dir
                    logger.info(f"Loading embedding model from HuggingFace: {model_name}")
                    embeddings_model = HuggingFaceEmbeddings(
                        model_name=model_name,
                        cache_folder=os.path.dirname(model_path)
                    )
            except Exception as embed_err:
                logger.error(f"Error loading embedding model: {str(embed_err)}")
                # Fallback to simple embedding
                logger.info("Using fallback embedding method")
                embeddings_model = None

            # Ensure directories exist for vector store
            os.makedirs("./data/task_embeddings", exist_ok=True)

            # Initialize vector store if embeddings are available
            if embeddings_model:
                try:
                    vector_store = Chroma(
                        persist_directory="./data/task_embeddings",
                        embedding_function=embeddings_model
                    )
                    logger.info("Vector store initialized successfully!")
                except Exception as vs_err:
                    logger.error(f"Error initializing vector store: {str(vs_err)}")
                    vector_store = None
            else:
                logger.warning("Embeddings model not available. Vector search will be limited.")
        except Exception as e:
            logger.error(f"Error during startup: {str(e)}")
            logger.warning("Running without vector search capabilities.")
    else:
        logger.warning("Vector database libraries not available. Running without vector search.")

    yield

    # Shutdown event
    logger.info("Shutting down application...")

    # Persist vector store if available
    if vector_store is not None:
        try:
            vector_store.persist()
            logger.info("Vector store persisted successfully!")
        except Exception as e:
            logger.error(f"Error persisting vector store: {str(e)}")

# Initialize FastAPI app with proper lifespan
app = FastAPI(
    title="Advanced Neuromorphic Quantum-Cognitive Task System",
    description="A quantum-inspired task management system with ML capabilities",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "name": "Advanced Neuromorphic Quantum-Cognitive Task System",
        "version": "2.0.0-quantum",
        "status": "operational",
        "capabilities": {
            "ml_enabled": ML_LIBS_AVAILABLE,
            "quantum_enabled": QUANTUM_LIBS_AVAILABLE,
            "vector_search": VECTOR_DB_AVAILABLE and embeddings_model is not None
        },
        "endpoints": [
            "/tasks", "/tasks/{task_id}", "/entanglements",
            "/entanglements/{entanglement_id}", "/metrics",
            "/quantum-simulation", "/search", "/optimize-assignments"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "time": datetime.now().isoformat(),
        "system_load": {
            "tasks": len(tasks),
            "entanglements": len(entanglements),
            "memory_usage": {
                "ml_models_loaded": len(ml_models)
            }
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Get system-wide metrics."""
    return calculate_system_metrics()

@app.post("/tasks", response_model=Task)
async def create_task(task: TaskCreate, background_tasks: BackgroundTasks):
    """Create a new task with quantum state initialization."""
    task_id = str(uuid.uuid4())
    now = datetime.now()

    # Use ML to enhance task information (in background to avoid delays)
    task_text = f"{task.title} {task.description}"
    embedding = None
    category = None
    ml_summary = None

    try:
        # Generate embedding
        embedding = calculate_task_embedding(task_text)

        # Basic classification without ML model
        category = classify_task(task_text)

        # Generate simple summary without ML model
        ml_summary = task_text.split('.')[0][:50]
    except Exception as e:
        logger.error(f"Task processing error: {str(e)}")

    # Initialize quantum-inspired probability distribution
    states = [TaskState.PENDING, TaskState.IN_PROGRESS, TaskState.COMPLETED, TaskState.BLOCKED]
    probs = [0.7, 0.2, 0.05, 0.05]  # Initial probabilities

    # Create quantum state information
    quantum_state = {
        "fidelity": 1.0,
        "coherence_time": 1.0,
        "amplitudes": {
            states[i]: {"real": math.sqrt(probs[i]), "imag": 0.0}
            for i in range(len(states))
        }
    }

    # Create new task with quantum properties
    new_task = Task(
        id=task_id,
        title=task.title,
        description=task.description,
        assignee=task.assignee,
        created_at=now,
        updated_at=now,
        due_date=task.due_date,
        state=TaskState.PENDING,
        tags=list(task.tags) if task.tags else [],
        priority=task.priority,
        entropy=1.0,  # Start with maximum entropy/uncertainty
        probability_distribution={states[i]: probs[i] for i in range(len(states))},
        embedding=embedding,
        entangled_tasks=[],
        quantum_state=quantum_state,
        category=category,
        ml_summary=ml_summary
    )

    # Store task
    tasks[task_id] = new_task

    # Update graph and vector store in background
    background_tasks.add_task(update_task_graph, task_id)
    background_tasks.add_task(update_vector_store, task_id)

    # Schedule ML enhancements in background (if libraries available)
    if ML_LIBS_AVAILABLE:
        background_tasks.add_task(enhance_task_with_ml, task_id, task_text)

    return new_task

@app.get("/tasks", response_model=List[Task])
async def list_tasks(
    state: Optional[str] = None,
    assignee: Optional[str] = None,
    sort_by: str = "created_at",
    limit: int = Query(100, ge=1, le=500),
    tags: Optional[List[str]] = Query(None)
):
    """List all tasks with filtering and sorting."""
    filtered_tasks = list(tasks.values())

    # Apply filters
    if state:
        filtered_tasks = [t for t in filtered_tasks if t.state == state]
    if assignee:
        filtered_tasks = [t for t in filtered_tasks if t.assignee == assignee]
    if tags:
        filtered_tasks = [t for t in filtered_tasks if any(tag in t.tags for tag in tags)]

    # Apply sorting
    if sort_by == "priority":
        filtered_tasks.sort(key=lambda t: t.priority, reverse=True)
    elif sort_by == "due_date":
        # Sort by due_date, putting None at the end
        filtered_tasks.sort(key=lambda t: (t.due_date is None, t.due_date))
    elif sort_by == "entropy":
        filtered_tasks.sort(key=lambda t: t.entropy, reverse=True)
    else:  # default to created_at
        filtered_tasks.sort(key=lambda t: t.created_at, reverse=True)

    # Apply limit
    return filtered_tasks[:limit]

@app.get("/tasks/{task_id}", response_model=Task)
async def get_task(task_id: str, background_tasks: BackgroundTasks):
    """Get a task by ID and update its entropy."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    # Update task entropy in background
    background_tasks.add_task(update_task_entropy, task_id)

    return tasks[task_id]

@app.put("/tasks/{task_id}", response_model=Task)
async def update_task(task_id: str, task_update: TaskUpdate, background_tasks: BackgroundTasks):
    """Update a task with quantum state changes."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    update_data = task_update.dict(exclude_unset=True)

    # Update fields
    for field, value in update_data.items():
        if field != "entangled_tasks" and field != "quantum_state" and field != "embedding":
            setattr(task, field, value)

    # Update timestamp
    task.updated_at = datetime.now()

    # Update embedding if title or description changed
    if "title" in update_data or "description" in update_data:
        task_text = f"{task.title} {task.description}"
        task.embedding = calculate_task_embedding(task_text)

        # Update ML-generated fields
        task.category = classify_task(task_text)
        task.ml_summary = generate_task_summary(task_text)

        # Schedule more detailed ML processing in background
        if ML_LIBS_AVAILABLE:
            background_tasks.add_task(enhance_task_with_ml, task_id, task_text)

    # Update task quantum state and propagate entanglement effects
    background_tasks.add_task(update_task_entropy, task_id, decay_factor=0.98)
    background_tasks.add_task(propagate_entanglement, task_id)
    background_tasks.add_task(update_vector_store, task_id)

    return task

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str, background_tasks: BackgroundTasks):
    """Delete a task and update entanglements."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    # Remove task
    task = tasks.pop(task_id)

    # Remove from graph
    if task_id in task_graph:
        task_graph.remove_node(task_id)

    # Remove entanglements involving this task
    entanglements_to_remove = []
    for e_id, e in entanglements.items():
        if e.task_id_1 == task_id or e.task_id_2 == task_id:
            entanglements_to_remove.append(e_id)

    for e_id in entanglements_to_remove:
        entanglements.pop(e_id)

    # Update vector store in background
    if vector_store is not None:
        try:
            background_tasks.add_task(remove_from_vector_store, task_id)
        except Exception as e:
            logger.error(f"Error scheduling vector store update: {str(e)}")

    return {"message": "Task deleted"}

@app.post("/entanglements", response_model=Entanglement)
async def create_entanglement(entanglement: EntanglementCreate, background_tasks: BackgroundTasks):
    """Create an entanglement between two tasks."""
    # Validate both tasks exist
    if entanglement.task_id_1 not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {entanglement.task_id_1} not found")
    if entanglement.task_id_2 not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {entanglement.task_id_2} not found")

    # Check if entanglement already exists
    for e in entanglements.values():
        if ((e.task_id_1 == entanglement.task_id_1 and e.task_id_2 == entanglement.task_id_2) or
            (e.task_id_1 == entanglement.task_id_2 and e.task_id_2 == entanglement.task_id_1)):
            raise HTTPException(status_code=400, detail="Entanglement already exists")

    # Create new entanglement
    entanglement_id = str(uuid.uuid4())
    now = datetime.now()

    new_entanglement = Entanglement(
        id=entanglement_id,
        task_id_1=entanglement.task_id_1,
        task_id_2=entanglement.task_id_2,
        strength=entanglement.strength,
        entanglement_type=entanglement.entanglement_type,
        created_at=now,
        updated_at=now
    )

    # Store entanglement
    entanglements[entanglement_id] = new_entanglement

    # Update task entangled_tasks lists
    task1 = tasks[entanglement.task_id_1]
    task2 = tasks[entanglement.task_id_2]

    if entanglement.task_id_2 not in task1.entangled_tasks:
        task1.entangled_tasks.append(entanglement.task_id_2)

    if entanglement.task_id_1 not in task2.entangled_tasks:
        task2.entangled_tasks.append(entanglement.task_id_1)

    # Update the task graph
    task_graph.add_edge(
        entanglement.task_id_1,
        entanglement.task_id_2,
        weight=entanglement.strength,
        type=entanglement.entanglement_type
    )

    # Apply initial entanglement effects
    background_tasks.add_task(propagate_entanglement, entanglement.task_id_1)

    return new_entanglement

@app.get("/entanglements", response_model=List[Entanglement])
async def list_entanglements(
    task_id: Optional[str] = None,
    entanglement_type: Optional[str] = None
):
    """List all entanglements with optional filtering."""
    filtered_entanglements = list(entanglements.values())

    # Apply filters
    if task_id:
        filtered_entanglements = [
            e for e in filtered_entanglements
            if e.task_id_1 == task_id or e.task_id_2 == task_id
        ]

    if entanglement_type:
        filtered_entanglements = [
            e for e in filtered_entanglements
            if e.entanglement_type == entanglement_type
        ]

    return filtered_entanglements

@app.get("/entanglements/{entanglement_id}", response_model=Entanglement)
async def get_entanglement(entanglement_id: str):
    """Get an entanglement by ID."""
    if entanglement_id not in entanglements:
        raise HTTPException(status_code=404, detail="Entanglement not found")

    return entanglements[entanglement_id]

@app.put("/entanglements/{entanglement_id}", response_model=Entanglement)
async def update_entanglement(
    entanglement_id: str,
    entanglement_update: EntanglementUpdate,
    background_tasks: BackgroundTasks
):
    """Update an entanglement's properties."""
    if entanglement_id not in entanglements:
        raise HTTPException(status_code=404, detail="Entanglement not found")

    entanglement = entanglements[entanglement_id]

    # Update fields
    update_data = entanglement_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(entanglement, field, value)

    # Update timestamp
    entanglement.updated_at = datetime.now()

    # Update edge in graph
    if task_graph.has_edge(entanglement.task_id_1, entanglement.task_id_2):
        task_graph[entanglement.task_id_1][entanglement.task_id_2]["weight"] = entanglement.strength
        task_graph[entanglement.task_id_1][entanglement.task_id_2]["type"] = entanglement.entanglement_type

    # Propagate changes
    background_tasks.add_task(propagate_entanglement, entanglement.task_id_1)

    return entanglement

@app.delete("/entanglements/{entanglement_id}")
async def delete_entanglement(entanglement_id: str):
    """Delete an entanglement."""
    if entanglement_id not in entanglements:
        raise HTTPException(status_code=404, detail="Entanglement not found")

    entanglement = entanglements.pop(entanglement_id)

    # Remove from tasks' entangled_tasks lists
    if entanglement.task_id_1 in tasks and entanglement.task_id_2 in tasks[entanglement.task_id_1].entangled_tasks:
        tasks[entanglement.task_id_1].entangled_tasks.remove(entanglement.task_id_2)

    if entanglement.task_id_2 in tasks and entanglement.task_id_1 in tasks[entanglement.task_id_2].entangled_tasks:
        tasks[entanglement.task_id_2].entangled_tasks.remove(entanglement.task_id_1)

    # Remove edge from graph
    if task_graph.has_edge(entanglement.task_id_1, entanglement.task_id_2):
        task_graph.remove_edge(entanglement.task_id_1, entanglement.task_id_2)

    return {"message": "Entanglement deleted"}

@app.post("/search")
async def search_tasks(search_query: SearchQuery):
    """Search for tasks using vector search or direct comparison."""
    if not tasks:
        return []

    query_text = search_query.query.strip()
    if not query_text:
        return []

    # Calculate query embedding
    query_embedding = calculate_task_embedding(query_text)

    # Choose search method
    if vector_store is not None and VECTOR_DB_AVAILABLE and not search_query.use_quantum:
        # Try vector database search
        try:
            results = vector_store.similarity_search_with_score(
                query_text,
                k=search_query.limit
            )

            # Format results
            search_results = []
            for doc, score in results:
                task_id = doc.metadata.get("task_id")
                if task_id in tasks:
                    search_results.append({
                        "task": tasks[task_id],
                        "relevance_score": float(score)
                    })

            return search_results
        except Exception as e:
            logger.error(f"Vector search error: {str(e)}")
            # Fall back to direct comparison

    # Direct embedding comparison (fallback or quantum mode)
    similarities = []

    # Calculate similarity for each task
    for task_id, task in tasks.items():
        if task.embedding:
            # Calculate similarity
            try:
                task_embedding = np.array(task.embedding)
                query_embedding_np = np.array(query_embedding)

                # Normalize embeddings
                task_embedding = task_embedding / (np.linalg.norm(task_embedding) + 1e-6)
                query_embedding_np = query_embedding_np / (np.linalg.norm(query_embedding_np) + 1e-6)

                similarity = float(np.dot(task_embedding, query_embedding_np))

                # Apply quantum effects if requested
                if search_query.use_quantum:
                    # Apply task entropy as noise factor
                    quantum_factor = 1.0 - (task.entropy * 0.3)  # Higher entropy = more uncertainty

                    # Add quantum fluctuation
                    phase_factor = math.sin(random.uniform(0, math.pi))
                    quantum_similarity = similarity * quantum_factor * (1.0 + 0.2 * phase_factor)

                    # Boost relevance for tasks in specific states
                    if task.state == TaskState.IN_PROGRESS:
                        quantum_similarity *= 1.1  # Boost in-progress tasks

                    similarity = quantum_similarity

                similarities.append((task_id, similarity))
            except Exception as e:
                logger.error(f"Search similarity calculation error for task {task_id}: {str(e)}")

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Format results
    search_results = []
    for task_id, score in similarities[:search_query.limit]:
        search_results.append({
            "task": tasks[task_id],
            "relevance_score": score
        })

    return search_results

@app.post("/task-suggestions/{task_id}")
async def suggest_related_tasks(task_id: str, threshold: float = Query(0.7, ge=0.1, le=1.0)):
    """Suggest tasks for entanglement based on similarity."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    # Get suggestions
    suggestions = suggest_entanglements(task_id, threshold)

    return suggestions

@app.post("/quantum-simulation")
async def run_quantum_simulation(simulation_request: QuantumSimulationRequest):
    """Run quantum simulation on selected tasks."""
    # Validate task IDs
    invalid_ids = [tid for tid in simulation_request.task_ids if tid not in tasks]
    if invalid_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Tasks not found: {', '.join(invalid_ids)}"
        )

    # Limit simulation steps for performance
    steps = min(simulation_request.simulation_steps, 10)

    # Run the simulation
    simulation_results = simulate_quantum_circuit(
        tasks,
        simulation_request.task_ids,
        steps=steps
    )

    # Update task states based on simulation (optional)
    if (not simulation_results.get("error") and
        simulation_request.measurement_type == "projective" and
        "final_results" in simulation_results and
        "measurement_outcomes" in simulation_results["final_results"]):

        # Only apply with a low probability to avoid too much state churn
        if random.random() < 0.3:
            for task_id, outcome in simulation_results["final_results"]["measurement_outcomes"].items():
                if task_id in tasks and random.random() < 0.5:  # Only apply some measurements
                    if outcome["outcome"] == "COMPLETED" and tasks[task_id].state != TaskState.COMPLETED:
                        tasks[task_id].state = TaskState.COMPLETED
                        # Update probability distribution
                        asyncio.create_task(update_probability_distribution(task_id))

    return simulation_results

@app.post("/optimize-assignments")
async def optimize_task_assignments():
    """Optimize task assignments using intelligent methods."""
    # Run optimization algorithm
    recommendations = optimize_task_assignment(tasks)

    return {
        "recommendations": recommendations,
        "total_recommendations": len(recommendations)
    }

@app.post("/ask-question")
async def ask_task_question(task_id: str, question: str):
    """Ask a question about a task using NLP model or simple extraction."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    context = f"Task Title: {task.title}\nDescription: {task.description}\n"
    if task.ml_summary:
        context += f"Summary: {task.ml_summary}\n"
    context += f"Status: {task.state}\nPriority: {task.priority}\n"
    context += f"Due Date: {task.due_date}\nAssignee: {task.assignee or 'Unassigned'}"

    # Process question
    answer = answer_task_question(question, context)

    return {
        "task_id": task_id,
        "question": question,
        "answer": answer,
        "task_title": task.title
    }

@app.get("/system-graph")
async def get_system_graph():
    """Return a representation of the task graph for visualization."""
    # Convert networkx graph to dictionary format
    nodes = []
    for node_id in task_graph.nodes():
        if node_id in tasks:
            task = tasks[node_id]
            nodes.append({
                "id": node_id,
                "title": task.title,
                "state": task.state,
                "priority": task.priority,
                "entropy": task.entropy
            })

    edges = []
    for u, v, data in task_graph.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "weight": data.get("weight", 1.0),
            "type": data.get("type", "standard")
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "task_count": len(tasks),
        "entanglement_count": len(entanglements)
    }

# --- Helper functions for background tasks ---

async def enhance_task_with_ml(task_id, task_text):
    """Enhance task with ML-generated information in background."""
    if task_id not in tasks:
        return

    task = tasks[task_id]

    try:
        # More detailed ML processing
        if ML_LIBS_AVAILABLE:
            # Try to generate better embedding
            embedding_model = get_ml_model("task_embedding")
            if embedding_model:
                task.embedding = embedding_model.encode(task_text).tolist()

            # Try to classify with ML
            classifier = get_ml_model("task_classifier")
            if classifier and classifier[1] is not None:
                tokenizer, pipeline = classifier
                result = pipeline(task_text)
                task.category = result[0]['label']

            # Try to generate better summary
            generator = get_ml_model("text_generator")
            if generator and generator[0] is not None and generator[1] is not None:
                tokenizer, model = generator
                inputs = tokenizer(f"Summarize: {task_text}", return_tensors="pt", max_length=100, truncation=True)
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=50,
                    num_beams=2,
                    early_stopping=True
                )
                task.ml_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Update in vector store if available
        await update_vector_store(task_id)

    except Exception as e:
        logger.error(f"Error enhancing task with ML: {str(e)}")

async def update_task_graph(task_id):
    """Update task graph with a new task."""
    if task_id not in task_graph.nodes and task_id in tasks:
        task = tasks[task_id]
        # Add to graph
        task_graph.add_node(
            task_id,
            title=task.title,
            state=task.state,
            priority=task.priority,
            entropy=task.entropy
        )

async def update_vector_store(task_id):
    """Update vector store with task embedding for efficient retrieval."""
    global vector_store
    if vector_store is None or task_id not in tasks or not VECTOR_DB_AVAILABLE:
        return

    task = tasks[task_id]
    if not task.embedding:
        # Generate embedding if not exists
        task_text = f"{task.title} {task.description}"
        task.embedding = calculate_task_embedding(task_text)
        tasks[task_id] = task

    try:
        # Create document for vector store
        doc = Document(
            page_content=f"{task.title}\n{task.description}",
            metadata={
                "task_id": task_id,
                "title": task.title,
                "state": task.state
            }
        )

        # Check if document exists and handle update/insert
        try:
            # Try to get existing document
            existing_docs = vector_store.get([task_id])
            
            if existing_docs and len(existing_docs) > 0:
                # Update existing document
                vector_store.update_document(task_id, doc)
                logger.debug(f"Updated task {task_id} in vector store")
            else:
                # Add new document if not exists
                vector_store.add_documents([doc], ids=[task_id])
                logger.debug(f"Added task {task_id} to vector store")
        except Exception as e:
            logger.warning(f"Error checking document existence, attempting direct add: {str(e)}")
            # Fallback: try direct add which will replace if exists
            vector_store.add_documents([doc], ids=[task_id])

    except Exception as e:
        logger.error(f"Error updating vector store: {str(e)}")

async def remove_from_vector_store(task_id):
    """Remove a task from the vector store."""
    if vector_store is None or not VECTOR_DB_AVAILABLE:
        return

    try:
        vector_store.delete([task_id])
        logger.debug(f"Removed task {task_id} from vector store")
    except Exception as e:
        logger.error(f"Error removing task from vector store: {str(e)}")
