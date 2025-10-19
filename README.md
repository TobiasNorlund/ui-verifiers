# UI Verifiers

Implements infrastructure and verifiable desktop tasks for scaling training of UI agents using RL. 

## Task: Simple Data Entry

[simple_data_entry.webm](https://github.com/user-attachments/assets/3bea7dcd-7ef9-463b-9b22-d2b49dd61f28)


## UI Verifiers Server


Launch the server locally with:

```bash
make build run
```

**Usage:**
```bash
# Show status
curl "http://localhost:8000"

# Get screenshot from a session
curl "http://localhost:8000/act?action_type=screenshot" > screen.png

# Perform an action in a session (e.g. left click in top left corner), and get screenshot after 1s
curl "http://localhost:8000/act?action_type=left_click&x=0&y=0&delay=1" > screen.png

# Get feedback/rewards from session
curl "http://localhost:8000/progress"
```
