"""
Unit tests for the Sports Dance Education Scheduling System
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

# Import modules to test
from sports_dance_scheduling import (
    SystemConfiguration,
    DatasetConfiguration,
    ModelConfiguration,
    HybridSchedulingModel,
    DatabaseManager,
    SyntheticDataGenerator,
    DataPreprocessor,
    PerformanceEvaluator,
    MultiObjectiveOptimizer,
    EnhancedLSTMScheduler,
    TransformerScheduler,
    TrainingEngine
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def config():
    """Create test configuration"""
    config = SystemConfiguration()
    config.dataset.num_classes = 10
    config.dataset.num_instructors = 5
    config.dataset.num_students = 20
    config.dataset.num_timeslots = 15
    config.model.hidden_size = 64
    config.model.num_layers = 2
    config.device = 'cpu'
    return config


@pytest.fixture
def db_manager():
    """Create temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    manager = DatabaseManager(db_path, pool_size=2)
    yield manager
    manager.close_all()
    os.unlink(db_path)


@pytest.fixture
def sample_schedule():
    """Generate sample schedule tensor"""
    return torch.rand(10, 5, 15)  # classes x instructors x timeslots


@pytest.fixture
def model(config):
    """Create model instance for testing"""
    return HybridSchedulingModel(config).to(config.device)


# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfiguration:
    def test_config_initialization(self):
        """Test configuration initialization"""
        config = SystemConfiguration()
        assert config.dataset.num_classes == 50
        assert config.model.hidden_size == 256
        assert config.training.num_epochs == 100
    
    def test_config_save_load(self, tmp_path):
        """Test configuration save and load"""
        config = SystemConfiguration()
        config.dataset.num_classes = 25
        
        config_file = tmp_path / "test_config.yaml"
        config.save(str(config_file))
        
        loaded_config = SystemConfiguration.load(str(config_file))
        assert loaded_config.dataset.num_classes == 25
    
    def test_model_architecture_enum(self):
        """Test model architecture enumeration"""
        from sports_dance_scheduling import ModelArchitecture
        assert ModelArchitecture.LSTM.value == "lstm"
        assert ModelArchitecture.TRANSFORMER.value == "transformer"


# ============================================================================
# Database Tests
# ============================================================================

class TestDatabaseManager:
    def test_database_initialization(self, db_manager):
        """Test database initialization and schema creation"""
        # Check if tables exist
        tables = db_manager.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        table_names = [t['name'] for t in tables]
        
        assert 'instructors' in table_names
        assert 'students' in table_names
        assert 'classes' in table_names
        assert 'schedules' in table_names
    
    def test_insert_and_query(self, db_manager):
        """Test insert and query operations"""
        # Insert instructor
        result = db_manager.execute_update(
            """INSERT INTO instructors (instructor_id, name, email, specialization)
               VALUES (?, ?, ?, ?)""",
            (1, "Test Instructor", "test@test.com", "Ballet")
        )
        assert result == 1
        
        # Query instructor
        results = db_manager.execute_query(
            "SELECT * FROM instructors WHERE instructor_id = ?",
            (1,)
        )
        assert len(results) == 1
        assert results[0]['name'] == "Test Instructor"
    
    def test_bulk_insert(self, db_manager):
        """Test bulk insert operation"""
        data = [
            {'student_id': i, 'name': f'Student_{i}', 'email': f's{i}@test.com',
             'skill_level': 'Beginner', 'hrv_baseline': 60.0,
             'performance_score': 0.5, 'attendance_rate': 0.9, 'active': 1}
            for i in range(5)
        ]
        
        result = db_manager.bulk_insert('students', data)
        assert result == 5
        
        # Verify insertion
        count = db_manager.execute_query("SELECT COUNT(*) as count FROM students")[0]['count']
        assert count == 5
    
    def test_database_statistics(self, db_manager):
        """Test database statistics retrieval"""
        stats = db_manager.get_statistics()
        assert 'instructors_count' in stats
        assert 'students_count' in stats
        assert 'database_size_mb' in stats


# ============================================================================
# Data Generation Tests
# ============================================================================

class TestDataGeneration:
    def test_synthetic_data_generator(self, config):
        """Test synthetic data generation"""
        generator = SyntheticDataGenerator(config.dataset)
        
        instructors = generator.generate_instructor_data()
        assert len(instructors) == config.dataset.num_instructors
        assert all('instructor_id' in i for i in instructors)
        
        students = generator.generate_student_data()
        assert len(students) == config.dataset.num_students
        
        classes = generator.generate_class_data()
        assert len(classes) == config.dataset.num_classes
    
    def test_availability_matrix_generation(self, config):
        """Test availability matrix generation"""
        generator = SyntheticDataGenerator(config.dataset)
        availability = generator.generate_availability_matrix()
        
        assert availability.shape == (
            config.dataset.num_instructors,
            config.dataset.num_days,
            config.dataset.slots_per_day
        )
        assert availability.min() >= 0
        assert availability.max() <= 1
    
    def test_enrollment_matrix_generation(self, config):
        """Test enrollment matrix generation"""
        generator = SyntheticDataGenerator(config.dataset)
        enrollment = generator.generate_enrollment_matrix()
        
        assert enrollment.shape == (
            config.dataset.num_students,
            config.dataset.num_classes
        )
        # Each student should be enrolled in at least one class
        assert all(row.sum() >= 1 for row in enrollment)


# ============================================================================
# Model Tests
# ============================================================================

class TestModels:
    def test_lstm_scheduler_forward(self, config):
        """Test LSTM scheduler forward pass"""
        model = EnhancedLSTMScheduler(config.model)
        batch_size = 4
        seq_len = 10
        input_tensor = torch.randn(batch_size, seq_len, config.model.input_size)
        
        output, hidden = model(input_tensor)
        assert output.shape == (batch_size, seq_len, config.model.output_size)
    
    def test_transformer_scheduler_forward(self, config):
        """Test Transformer scheduler forward pass"""
        config.model.architecture = "transformer"
        model = TransformerScheduler(config.model)
        batch_size = 4
        seq_len = 10
        input_tensor = torch.randn(batch_size, seq_len, config.model.input_size)
        
        output = model(input_tensor)
        assert output.shape == (batch_size, seq_len, config.model.output_size)
    
    def test_hybrid_model_initialization(self, config):
        """Test hybrid model initialization"""
        model = HybridSchedulingModel(config)
        assert model.neural_net is not None
        assert model.rl_agent is not None
        assert model.optimizer_engine is not None
    
    @pytest.mark.parametrize("architecture", ["lstm", "transformer"])
    def test_model_gradient_flow(self, config, architecture):
        """Test gradient flow through models"""
        if architecture == "lstm":
            model = EnhancedLSTMScheduler(config.model)
        else:
            model = TransformerScheduler(config.model)
        
        input_tensor = torch.randn(2, 5, config.model.input_size, requires_grad=True)
        if architecture == "lstm":
            output, _ = model(input_tensor)
        else:
            output = model(input_tensor)
        
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        assert input_tensor.grad is not None
        assert not torch.isnan(input_tensor.grad).any()


# ============================================================================
# Optimization Tests
# ============================================================================

class TestOptimization:
    def test_multi_objective_optimizer(self, config, sample_schedule):
        """Test multi-objective optimization"""
        optimizer = MultiObjectiveOptimizer(config.training)
        
        # Test individual loss components
        conflict_loss = optimizer.compute_conflict_loss(sample_schedule)
        assert conflict_loss >= 0
        
        workload_loss = optimizer.compute_workload_loss(sample_schedule)
        assert workload_loss.item() >= 0
        
        continuity_loss = optimizer.compute_continuity_loss(sample_schedule)
        assert continuity_loss >= 0
    
    def test_total_loss_computation(self, config, sample_schedule):
        """Test total loss computation"""
        optimizer = MultiObjectiveOptimizer(config.training)
        
        intensity = torch.rand(config.dataset.num_classes)
        student_hrv = torch.rand(config.dataset.num_students) * 50 + 30
        capacity = torch.ones(config.dataset.num_venues) * 30
        
        total_loss, loss_dict = optimizer.compute_total_loss(
            sample_schedule, intensity, student_hrv, capacity
        )
        
        assert total_loss >= 0
        assert 'conflict' in loss_dict
        assert 'workload' in loss_dict
        assert 'total' in loss_dict


# ============================================================================
# Performance Evaluation Tests
# ============================================================================

class TestPerformanceEvaluation:
    def test_conflict_resolution_evaluation(self, config, sample_schedule):
        """Test conflict resolution evaluation"""
        evaluator = PerformanceEvaluator(config.dataset)
        rate = evaluator.evaluate_conflict_resolution(sample_schedule)
        
        assert 0 <= rate <= 1
    
    def test_workload_balance_evaluation(self, config, sample_schedule):
        """Test workload balance evaluation"""
        evaluator = PerformanceEvaluator(config.dataset)
        efficiency = evaluator.evaluate_workload_balance(sample_schedule)
        
        assert 0 <= efficiency <= 1
    
    def test_comprehensive_evaluation(self, config, sample_schedule):
        """Test comprehensive evaluation"""
        evaluator = PerformanceEvaluator(config.dataset)
        metrics = evaluator.comprehensive_evaluation(sample_schedule)
        
        assert 'conflict_resolution_rate' in metrics
        assert 'workload_balance_efficiency' in metrics
        assert 'continuity_score' in metrics
        assert 'utilization_rate' in metrics
        
        # All metrics should be between 0 and 1
        assert all(0 <= v <= 1 for v in metrics.values())


# ============================================================================
# Training Tests
# ============================================================================

class TestTraining:
    @pytest.mark.slow
    def test_training_engine_initialization(self, config, model, db_manager):
        """Test training engine initialization"""
        engine = TrainingEngine(model, config, db_manager)
        
        assert engine.optimizer is not None
        assert engine.scheduler is not None
        assert engine.early_stopping is not None
    
    def test_early_stopping(self):
        """Test early stopping mechanism"""
        from sports_dance_scheduling import EarlyStopping
        
        early_stopping = EarlyStopping(patience=3, delta=0.001)
        
        # Simulate decreasing loss
        losses = [1.0, 0.9, 0.8, 0.7]
        for loss in losses:
            should_stop = early_stopping(loss)
            assert not should_stop
        
        # Simulate plateau
        for _ in range(4):
            should_stop = early_stopping(0.7)
        
        assert should_stop  # Should stop after patience exceeded


# ============================================================================
# Data Preprocessing Tests
# ============================================================================

class TestDataPreprocessing:
    def test_data_preprocessor(self, config):
        """Test data preprocessing"""
        preprocessor = DataPreprocessor(config.dataset)
        
        # Test normalization
        data = np.random.randn(100, 10) * 10 + 5
        normalized = preprocessor.fit_transform(data, 'test_feature')
        
        assert np.abs(normalized.mean()) < 0.1
        assert np.abs(normalized.std() - 1.0) < 0.1
    
    def test_missing_value_handling(self, config):
        """Test missing value handling"""
        preprocessor = DataPreprocessor(config.dataset)
        
        data = np.random.randn(100, 10)
        # Add missing values
        data[::10, ::3] = np.nan
        
        cleaned = preprocessor.handle_missing_values(data, strategy='mean')
        assert not np.any(np.isnan(cleaned))
    
    def test_data_augmentation(self, config):
        """Test data augmentation"""
        preprocessor = DataPreprocessor(config.dataset)
        
        data = np.random.randn(10, 5, 5)
        augmented = preprocessor.augment_data(data, factor=2)
        
        # Should have 3x the original data (original + 2 augmentations per factor)
        assert augmented.shape[0] > data.shape[0]


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
class TestIntegration:
    def test_end_to_end_scheduling(self, config, db_manager):
        """Test end-to-end scheduling process"""
        # Generate synthetic data
        generator = SyntheticDataGenerator(config.dataset)
        instructors = generator.generate_instructor_data()
        students = generator.generate_student_data()
        classes = generator.generate_class_data()
        
        # Insert into database
        db_manager.bulk_insert('instructors', instructors)
        db_manager.bulk_insert('students', students)
        db_manager.bulk_insert('classes', classes)
        
        # Create and run model
        model = HybridSchedulingModel(config).to(config.device)
        
        # Generate schedule
        input_tensor = torch.randn(1, config.dataset.num_timeslots, config.model.input_size)
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output is not None
        assert output.shape[-1] == config.model.output_size
    
    @pytest.mark.slow
    def test_model_checkpoint_save_load(self, config, model, tmp_path):
        """Test model checkpoint saving and loading"""
        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, checkpoint_path)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        assert checkpoint['config'].dataset.num_classes == config.dataset.num_classes


# ============================================================================
# Performance Benchmarks
# ============================================================================

@pytest.mark.benchmark
class TestBenchmarks:
    def test_schedule_generation_speed(self, config, model, benchmark):
        """Benchmark schedule generation speed"""
        input_tensor = torch.randn(1, config.dataset.num_timeslots, config.model.input_size)
        
        def generate():
            with torch.no_grad():
                return model(input_tensor)
        
        result = benchmark(generate)
        assert result is not None
    
    def test_database_query_speed(self, db_manager, benchmark):
        """Benchmark database query speed"""
        # Insert test data
        for i in range(100):
            db_manager.execute_update(
                "INSERT INTO classes (class_name, class_type, intensity_level) VALUES (?, ?, ?)",
                (f"Class_{i}", "Test", i % 10 + 1)
            )
        
        def query():
            return db_manager.execute_query("SELECT * FROM classes WHERE intensity_level > 5")
        
        result = benchmark(query)
        assert len(result) > 0


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    def test_invalid_configuration(self):
        """Test handling of invalid configuration"""
        config = SystemConfiguration()
        config.dataset.num_classes = -1  # Invalid value
        
        with pytest.raises(Exception):
            # Should raise error with invalid configuration
            generator = SyntheticDataGenerator(config.dataset)
            generator.generate_class_data()
    
    def test_database_connection_error(self):
        """Test database connection error handling"""
        with pytest.raises(Exception):
            # Should fail with invalid path
            db_manager = DatabaseManager("/invalid/path/database.db")
    
    def test_model_dimension_mismatch(self, config):
        """Test model dimension mismatch handling"""
        model = EnhancedLSTMScheduler(config.model)
        
        # Wrong input dimension
        wrong_input = torch.randn(2, 5, config.model.input_size + 10)
        
        with pytest.raises(RuntimeError):
            model(wrong_input)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])