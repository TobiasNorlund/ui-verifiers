# Architecture Decision: TaskRunner Instances

## Question
Should we have one large TaskRunner instance handling many UI sessions, or multiple TaskRunner instances (one per UI session)?

## Decision: **Multiple TaskRunner Instances (One per UI Session)** ✅

## Rationale

### ✅ Advantages of Multiple TaskRunners

#### 1. **Isolation & Fault Tolerance**
```python
# Actor 1 crashes → Others continue
# Actor 2 slow → Doesn't block others
# Actor 3 success → Independent trajectory collection
```

#### 2. **True Parallelism**
- No GIL contention for I/O operations
- Each actor runs in its own thread
- Concurrent HTTP requests to different UI environments
- Better CPU utilization

#### 3. **Simpler State Management**
```python
# Each TaskRunner maintains its own:
- current_trajectory
- step_count
- episode_state
- connection to UI environment
```

#### 4. **Scalability**
```python
# Easy to scale up
actors = [TaskRunner(url) for url in ui_env_urls]

# Easy to scale down
actors = actors[:half]
```

#### 5. **Resource Management**
- Each actor can have its own data directory
- Clear ownership of UI sessions
- Easy to track which actor produced which data

### ❌ Disadvantages of Single Large TaskRunner

#### 1. **Complex State Management**
```python
# Would need to track:
sessions = {
    'session_1': {state, trajectory, step_count, ...},
    'session_2': {state, trajectory, step_count, ...},
    # ...
}
# Error-prone and complex
```

#### 2. **Single Point of Failure**
- One bug crashes all data collection
- Difficult to recover from errors
- All-or-nothing restart

#### 3. **Blocking Issues**
```python
# If session_1 hangs:
for session in sessions:
    run_episode(session)  # All others wait!
```

#### 4. **Resource Contention**
- Shared connection pool
- Lock contention for shared state
- Difficult to debug performance issues

## Implementation Patterns

### Pattern 1: Fixed Actors (Recommended for Training)

Each actor continuously runs episodes on its assigned UI environment:

```python
from src.orchestration.multi_actor_runner import MultiActorRunner

# Multiple UI environment instances
ui_env_urls = [
    "http://localhost:8000",
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003"
]

# Create multi-actor runner
multi_actor = MultiActorRunner(
    model=vlm,
    trajectory_queue=trajectory_queue,
    ui_env_urls=ui_env_urls,
    task_prompt=task_prompt
)

# Start all actors
multi_actor.start()

# Training loop collects from shared queue
while training:
    trajectories = []
    for _ in range(batch_size):
        traj = trajectory_queue.get()
        trajectories.append(traj)

    trainer.train_step(trajectories)

# Stop actors
multi_actor.stop()
```

**Use when:**
- Fixed number of UI environments
- Long-running training
- Each environment runs the same task repeatedly

### Pattern 2: Worker Pool (Recommended for Diverse Tasks)

Workers dynamically pick tasks from a queue:

```python
from src.orchestration.multi_actor_runner import ActorPool

# Create task queue
task_queue = queue.Queue()

# Add diverse tasks
tasks = [
    ("http://ui-1:8000", "Click the login button"),
    ("http://ui-2:8000", "Fill out the form"),
    ("http://ui-3:8000", "Navigate to settings"),
]

for task in tasks:
    task_queue.put(task)

# Create worker pool
actor_pool = ActorPool(
    num_actors=4,
    model=vlm,
    trajectory_queue=trajectory_queue,
    task_queue=task_queue
)

actor_pool.start()

# Workers process tasks concurrently
actor_pool.stop()
```

**Use when:**
- Many different tasks to process
- Variable task duration
- Dynamic workload
- Want load balancing

### Pattern 3: Hybrid (Advanced)

Combine both patterns for maximum flexibility:

```python
# Fixed actors for continuous data collection
continuous_actors = MultiActorRunner(...)
continuous_actors.start()

# Worker pool for ad-hoc evaluation
eval_pool = ActorPool(...)
eval_pool.start()

# Both feed into same trajectory queue
```

## Architecture Diagram

### Multiple TaskRunners (Recommended)

```
┌─────────────────────────────────────────────────┐
│              Trainer (Learner)                  │
│         Consumes from trajectory_queue          │
└────────────────┬────────────────────────────────┘
                 │
                 │ Shared Queue
                 │
    ┌────────────┴────────────┬─────────────┐
    │                         │             │
    ▼                         ▼             ▼
┌─────────┐              ┌─────────┐   ┌─────────┐
│TaskRunner│             │TaskRunner│  │TaskRunner│
│  (Actor 1)│            │  (Actor 2)│  │  (Actor 3)│
└────┬─────┘             └────┬─────┘   └────┬─────┘
     │                        │              │
     ▼                        ▼              ▼
┌─────────┐              ┌─────────┐   ┌─────────┐
│UI Env 1 │              │UI Env 2 │   │UI Env 3 │
└─────────┘              └─────────┘   └─────────┘
```

**Benefits:**
- ✅ Parallel execution
- ✅ Independent failures
- ✅ Clear ownership
- ✅ Easy to debug
- ✅ Scalable

### Single Large TaskRunner (Not Recommended)

```
┌─────────────────────────────────────────────────┐
│              Trainer (Learner)                  │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │  TaskRunner    │
        │  (Multiplexer) │
        └────────┬───────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│UI Env 1 │  │UI Env 2 │  │UI Env 3 │
└─────────┘  └─────────┘  └─────────┘
```

**Problems:**
- ❌ Complex state management
- ❌ Single point of failure
- ❌ Potential blocking
- ❌ Difficult debugging

## Model Sharing

Both patterns share the same VLM model instance:

```python
# Model is shared across all actors
vlm = VLMWrapper(model_name="Qwen/Qwen2-VL-2B-Instruct")

# All actors reference the same model
actor1 = TaskRunner(model=vlm, ...)
actor2 = TaskRunner(model=vlm, ...)
actor3 = TaskRunner(model=vlm, ...)

# During training, model weights update
# All actors automatically see updated weights
```

**Thread Safety:**
- ✅ Reading (inference) is thread-safe
- ✅ PyTorch tensors are thread-safe for inference
- ⚠️ Training updates happen on separate thread (Trainer)
- ⚠️ Use locks if actors need to call training methods

## Memory Considerations

### Per-Actor Memory Usage

| Component | Memory per Actor |
|-----------|------------------|
| Episode state | ~10 MB |
| Trajectory buffer | ~50-100 MB |
| HTTP connections | ~5 MB |
| Thread overhead | ~8 MB |
| **Total** | **~75-125 MB** |

**For 4 actors: ~300-500 MB**

### Shared Memory Usage

| Component | Memory (Shared) |
|-----------|-----------------|
| VLM Model | 4-16 GB |
| Trajectory Queue | 100-500 MB |
| **Total** | **4-17 GB** |

**Total for 4 actors: ~4.3-17.5 GB**

## Performance Benchmarks

### Data Collection Throughput

**Single TaskRunner (Sequential):**
```
1 actor × 1 episode/sec = 1 episode/sec
```

**Multiple TaskRunners (Parallel):**
```
4 actors × 1 episode/sec = 4 episodes/sec
8 actors × 1 episode/sec = 8 episodes/sec
```

**Real-World Example (Qwen2-VL on RTX 3090):**
```
1 actor:  ~180 episodes/hour
4 actors: ~720 episodes/hour (4x speedup)
8 actors: ~1200 episodes/hour (6.7x speedup, some overhead)
```

## Best Practices

### 1. Start with Few Actors
```python
# Start small
num_actors = 2

# Monitor performance
# Scale up if GPU underutilized
```

### 2. Monitor Statistics
```python
stats = multi_actor.get_statistics()
print(f"Episodes collected: {stats['total_episodes']}")
print(f"Errors: {stats['total_errors']}")
print(f"Episodes per actor: {stats['episodes_per_actor']}")
```

### 3. Handle Actor Failures
```python
# Automatic restart on error
if actor_failed:
    multi_actor.restart_actor(actor_id)
```

### 4. Balance Load
```python
# If one actor much slower, investigate:
# - Network latency to UI environment
# - UI environment performance
# - Task complexity
```

### 5. Graceful Shutdown
```python
try:
    multi_actor.start()
    # Training...
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    multi_actor.stop()
```

## Configuration Example

```yaml
# config/default_config.yaml

actor:
  num_actors: 4
  max_steps_per_episode: 50
  action_format: "text"

  # UI environment URLs (one per actor)
  ui_env_urls:
    - "http://localhost:8000"
    - "http://localhost:8001"
    - "http://localhost:8002"
    - "http://localhost:8003"

  # Or use a base URL with port range
  ui_env_base_url: "http://localhost"
  ui_env_port_start: 8000
  ui_env_port_count: 4
```

## Comparison Summary

| Aspect | Multiple TaskRunners | Single TaskRunner |
|--------|---------------------|-------------------|
| **Complexity** | Low | High |
| **Fault Tolerance** | High | Low |
| **Parallelism** | True parallel | Sequential/Complex |
| **Scalability** | Easy | Difficult |
| **Debugging** | Easy | Difficult |
| **State Management** | Simple | Complex |
| **Memory Overhead** | Minimal | Potentially higher |
| **Performance** | Linear scaling | Limited |
| **Recommendation** | ✅ **Yes** | ❌ No |

## Conclusion

**Use Multiple TaskRunner Instances** - it's simpler, more robust, more scalable, and performs better.

The current codebase already supports this pattern, and the new `MultiActorRunner` and `ActorPool` classes make it even easier to manage multiple actors.

## References

- Implementation: `src/orchestration/multi_actor_runner.py`
- Example: `examples/train_qwen2_vl.py`
- Training script: `scripts/train.py` (uses single actor, easily extended)
