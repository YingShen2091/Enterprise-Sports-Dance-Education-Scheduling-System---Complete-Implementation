# Enterprise Sports Dance Education Scheduling System

## Overview

A comprehensive production-grade scheduling system for sports and dance education institutions, implementing advanced algorithms from research on multi-objective optimization. This system combines neural networks with reinforcement learning to create optimal class schedules while considering multiple constraints including instructor availability, student endurance, venue capacity, and workload balance.

## Features

### Core Capabilities
- **Hybrid Scheduling Algorithm**: Combines LSTM/Transformer neural networks with Double DQN reinforcement learning
- **Multi-Objective Optimization**: Balances conflict resolution, workload distribution, resource utilization, student endurance, and schedule continuity
- **Enterprise Database Management**: SQLite-based system with connection pooling and comprehensive schema
- **Real-time and Batch Processing**: Supports multiple scheduling modes including distributed computing
- **Performance Monitoring**: Built-in metrics tracking and system monitoring
- **REST API**: Production-ready API layer for integration

### Technical Features
- Mixed precision training with gradient scaling
- Prioritized experience replay for reinforcement learning
- Attention mechanisms for improved schedule quality
- Data augmentation and preprocessing pipelines
- Comprehensive testing framework
- Model checkpointing and versioning
- ONNX export for deployment

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for accelerated training)
- 8GB+ RAM recommended

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sports-dance-scheduling.git
cd sports-dance-scheduling
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the system:
```bash
cp config.sample.yaml config.yaml
# Edit config.yaml with your settings
```

## Quick Start

### Basic Usage

```python
from sports_dance_scheduling import SystemConfiguration, HybridSchedulingModel, DatabaseManager

# Initialize configuration
config = SystemConfiguration()
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup database
db_manager = DatabaseManager('scheduling.db')

# Create and train model
model = HybridSchedulingModel(config).to(config.device)

# Generate schedule
schedule = model.generate_schedule(constraints)
```

### Running the System

```bash
# Run with default configuration
python sports_dance_scheduling.py

# Run with custom configuration
python sports_dance_scheduling.py --config custom_config.yaml

# Run in distributed mode
python sports_dance_scheduling.py --distributed --world-size 4
```

## Architecture

### System Components

```
┌─────────────────────────────────────────┐
│         REST API Layer                  │
├─────────────────────────────────────────┤
│     Hybrid Scheduling Model             │
│  ┌──────────────┬────────────────┐     │
│  │ Neural Net   │ RL Agent       │     │
│  │ (LSTM/Trans) │ (Double DQN)   │     │
│  └──────────────┴────────────────┘     │
├─────────────────────────────────────────┤
│    Multi-Objective Optimizer            │
├─────────────────────────────────────────┤
│       Database Management               │
└─────────────────────────────────────────┘
```

### Model Architectures

- **LSTM Scheduler**: Bidirectional LSTM with multi-head attention
- **Transformer Scheduler**: Full transformer encoder architecture
- **Hybrid Model**: Combines neural network with reinforcement learning
- **Ensemble**: Multiple models with voting mechanism

## Configuration

The system uses a hierarchical configuration structure:

```yaml
dataset:
  num_classes: 50
  num_instructors: 10
  num_students: 200
  num_timeslots: 30
  
model:
  architecture: hybrid_rnn_rl
  hidden_size: 256
  num_layers: 3
  attention_heads: 8
  
training:
  num_epochs: 100
  learning_rate: 0.001
  optimizer: adamw
  mixed_precision: true
```

See `config.sample.yaml` for complete configuration options.

## Performance Metrics

The system optimizes for multiple objectives:

1. **Conflict Resolution Rate**: Minimizes scheduling conflicts (target: >98%)
2. **Workload Balance**: Ensures equitable instructor workload (efficiency: >85%)
3. **Resource Utilization**: Maximizes venue usage (target: >75%)
4. **Student Endurance**: Prevents overload based on HRV metrics
5. **Schedule Continuity**: Minimizes gaps between classes

## Database Schema

The system includes comprehensive data management:

- **instructors**: Instructor profiles and availability
- **students**: Student information and performance metrics
- **classes**: Class definitions and requirements
- **venues**: Venue specifications and capacity
- **schedules**: Generated schedules with optimization scores
- **enrollments**: Student-class associations
- **performance_metrics**: System performance tracking

## API Documentation

### Endpoints

- `POST /api/schedule/generate` - Generate new schedule
- `GET /api/schedule/{id}` - Retrieve schedule
- `PUT /api/schedule/{id}/optimize` - Optimize existing schedule
- `GET /api/metrics/performance` - Get system metrics
- `POST /api/instructors` - Add instructor
- `POST /api/students` - Add student
- `POST /api/classes` - Add class

### Example Request

```bash
curl -X POST http://localhost:8000/api/schedule/generate \
  -H "Content-Type: application/json" \
  -d '{
    "semester": "2024-Spring",
    "optimization_mode": "hybrid",
    "constraints": {
      "max_conflicts": 5,
      "min_utilization": 0.7
    }
  }'
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_model.py
python -m pytest tests/test_database.py
python -m pytest tests/test_optimization.py

# Run with coverage
python -m pytest --cov=sports_dance_scheduling tests/
```

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t sports-dance-scheduler .

# Run container
docker run -p 8000:8000 -v $(pwd)/data:/app/data sports-dance-scheduler
```

### Production Deployment

1. Export model for production:
```python
deployment.save_for_production(model, 'models/production_model.pt')
```

2. Export to ONNX for inference:
```python
deployment.export_to_onnx(model, 'models/model.onnx', input_shape)
```


### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.





## Roadmap

- [ ] Web-based UI for schedule visualization
- [ ] Integration with calendar systems (Google Calendar, Outlook)
- [ ] Mobile application for students and instructors
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] Cloud deployment options (AWS, Azure, GCP)
