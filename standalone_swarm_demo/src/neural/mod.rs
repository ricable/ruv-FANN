//! Neural Network Core Module
//! 
//! This module contains the neural network implementations for the swarm optimization system.
//! It includes both basic neural networks and specialized ML models for RAN optimization.

use crate::models::{RANMetrics, AgentSpecialization};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod ml_model;
pub mod demand_predictor;
pub mod neural_agent;
pub mod kpi_predictor;
pub mod throughput_model;
pub mod latency_optimizer;
pub mod quality_predictor;
pub mod endc_predictor;
pub mod feature_engineering;

pub use ml_model::MLModel;
pub use demand_predictor::DemandPredictor;
pub use neural_agent::NeuralAgent;
pub use kpi_predictor::KpiPredictor;
pub use throughput_model::ThroughputModel;
pub use latency_optimizer::LatencyOptimizer;
pub use quality_predictor::QualityPredictor;
pub use endc_predictor::EndcPredictor;
pub use feature_engineering::FeatureEngineering;

/// Neural network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkConfig {
    pub input_size: usize,
    pub hidden_layers: Vec<usize>,
    pub output_size: usize,
    pub learning_rate: f32,
    pub activation_function: String,
}

impl Default for NeuralNetworkConfig {
    fn default() -> Self {
        Self {
            input_size: 8,
            hidden_layers: vec![16, 8],
            output_size: 1,
            learning_rate: 0.001,
            activation_function: "relu".to_string(),
        }
    }
}

/// Simple neural network implementation enhanced for KPI prediction
#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    pub config: NeuralNetworkConfig,
    pub weights: Vec<Vec<Vec<f32>>>,
    pub biases: Vec<Vec<f32>>,
    pub is_trained: bool,
    pub performance_metrics: PerformanceMetrics,
    pub last_training_loss: f32,
    pub training_iterations: u32,
}

/// Performance tracking for neural networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub accuracy: f32,
    pub loss: f32,
    pub validation_accuracy: f32,
    pub training_time_ms: u64,
    pub prediction_count: u64,
    pub convergence_rate: f32,
}

impl NeuralNetwork {
    pub fn new(config: NeuralNetworkConfig) -> Self {
        let mut network = Self {
            config: config.clone(),
            weights: Vec::new(),
            biases: Vec::new(),
            is_trained: false,
            performance_metrics: PerformanceMetrics {
                accuracy: 0.0,
                loss: f32::INFINITY,
                validation_accuracy: 0.0,
                training_time_ms: 0,
                prediction_count: 0,
                convergence_rate: 0.0,
            },
            last_training_loss: f32::INFINITY,
            training_iterations: 0,
        };
        
        network.initialize_weights();
        network
    }
    
    fn initialize_weights(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let mut layer_sizes = vec![self.config.input_size];
        layer_sizes.extend(&self.config.hidden_layers);
        layer_sizes.push(self.config.output_size);
        
        self.weights.clear();
        self.biases.clear();
        
        for i in 0..layer_sizes.len() - 1 {
            let rows = layer_sizes[i + 1];
            let cols = layer_sizes[i];
            
            let mut layer_weights = Vec::new();
            let mut layer_biases = Vec::new();
            
            for _ in 0..rows {
                let mut row = Vec::new();
                for _ in 0..cols {
                    row.push(rng.gen_range(-1.0..1.0));
                }
                layer_weights.push(row);
                layer_biases.push(rng.gen_range(-1.0..1.0));
            }
            
            self.weights.push(layer_weights);
            self.biases.push(layer_biases);
        }
    }
    
    pub fn forward(&self, inputs: &[f32]) -> Result<Vec<f32>, String> {
        if inputs.len() != self.config.input_size {
            return Err("Input size mismatch".to_string());
        }
        
        let mut current_layer = inputs.to_vec();
        
        for (layer_idx, (layer_weights, layer_biases)) in 
            self.weights.iter().zip(self.biases.iter()).enumerate() {
            
            let mut next_layer = Vec::new();
            
            for (neuron_weights, bias) in layer_weights.iter().zip(layer_biases.iter()) {
                let weighted_sum: f32 = neuron_weights.iter()
                    .zip(current_layer.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f32>() + bias;
                
                let activated = self.activate(weighted_sum);
                next_layer.push(activated);
            }
            
            current_layer = next_layer;
        }
        
        Ok(current_layer)
    }
    
    fn activate(&self, x: f32) -> f32 {
        match self.config.activation_function.as_str() {
            "relu" => x.max(0.0),
            "sigmoid" => 1.0 / (1.0 + (-x).exp()),
            "tanh" => x.tanh(),
            _ => x, // linear
        }
    }
    
    pub fn train(&mut self, training_data: &[(Vec<f32>, Vec<f32>)]) -> Result<(), String> {
        if training_data.is_empty() {
            return Err("Training data cannot be empty".to_string());
        }
        
        let start_time = std::time::Instant::now();
        let iterations = 100;
        let learning_rate = self.config.learning_rate;
        let mut total_loss = 0.0;
        
        for iteration in 0..iterations {
            let mut batch_loss = 0.0;
            
            for (inputs, targets) in training_data {
                // Forward pass
                let outputs = self.forward(inputs)?;
                
                // Calculate loss (mean squared error)
                let loss: f32 = outputs.iter()
                    .zip(targets.iter())
                    .map(|(o, t)| (o - t).powi(2))
                    .sum::<f32>() / outputs.len() as f32;
                
                batch_loss += loss;
                
                // Simplified backpropagation
                self.update_weights_simple(inputs, targets, &outputs, learning_rate)?;
            }
            
            let avg_loss = batch_loss / training_data.len() as f32;
            total_loss += avg_loss;
            
            // Update learning rate (decay)
            if iteration % 20 == 0 {
                self.config.learning_rate *= 0.95;
            }
        }
        
        self.last_training_loss = total_loss / iterations as f32;
        self.training_iterations += iterations;
        self.performance_metrics.training_time_ms = start_time.elapsed().as_millis() as u64;
        self.performance_metrics.loss = self.last_training_loss;
        
        // Calculate training accuracy
        let mut correct_predictions = 0;
        for (inputs, targets) in training_data {
            let outputs = self.forward(inputs)?;
            if self.is_prediction_correct(&outputs, targets) {
                correct_predictions += 1;
            }
        }
        self.performance_metrics.accuracy = correct_predictions as f32 / training_data.len() as f32;
        
        self.is_trained = true;
        Ok(())
    }
    
    fn update_weights_simple(&mut self, inputs: &[f32], targets: &[f32], outputs: &[f32], lr: f32) -> Result<(), String> {
        // Simplified weight update - in practice would use proper backpropagation
        let output_errors: Vec<f32> = outputs.iter()
            .zip(targets.iter())
            .map(|(o, t)| t - o)
            .collect();
        
        // Update output layer weights (simplified)
        if let (Some(last_weights), Some(last_biases)) = (self.weights.last_mut(), self.biases.last_mut()) {
            for (i, error) in output_errors.iter().enumerate() {
                if i < last_biases.len() {
                    last_biases[i] += lr * error;
                }
                
                if i < last_weights.len() {
                    for (j, weight) in last_weights[i].iter_mut().enumerate() {
                        if j < inputs.len() {
                            *weight += lr * error * inputs[j];
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn is_prediction_correct(&self, outputs: &[f32], targets: &[f32]) -> bool {
        if outputs.len() != targets.len() {
            return false;
        }
        
        // For regression tasks, consider prediction correct if within 10% of target
        outputs.iter()
            .zip(targets.iter())
            .all(|(o, t)| {
                if *t == 0.0 {
                    o.abs() < 0.1
                } else {
                    (o - t).abs() / t.abs() < 0.1
                }
            })
    }
    
    pub fn predict_fitness(&self, metrics: &RANMetrics) -> Result<f32, String> {
        if !self.is_trained {
            return Err("Network must be trained before prediction".to_string());
        }
        
        let inputs = vec![
            metrics.throughput / 100.0,
            metrics.latency / 50.0,
            metrics.energy_efficiency,
            metrics.interference_level,
            // Add more features as needed
        ];
        
        // Pad or truncate inputs to match expected size
        let mut padded_inputs = inputs;
        padded_inputs.resize(self.config.input_size, 0.0);
        
        let outputs = self.forward(&padded_inputs)?;
        
        if outputs.is_empty() {
            return Err("Network produced no output".to_string());
        }
        
        Ok(outputs[0])
    }
    
    /// Get performance metrics for the network
    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }
    
    /// Get training loss
    pub fn get_training_loss(&self) -> f32 {
        self.last_training_loss
    }
    
    /// Get number of training iterations completed
    pub fn get_training_iterations(&self) -> u32 {
        self.training_iterations
    }
    
    /// Update prediction count for metrics
    pub fn increment_prediction_count(&mut self) {
        self.performance_metrics.prediction_count += 1;
    }
}

/// Neural network factory for creating specialized networks
pub struct NeuralNetworkFactory;

impl NeuralNetworkFactory {
    pub fn create_for_specialization(specialization: &AgentSpecialization) -> NeuralNetwork {
        let config = match specialization {
            AgentSpecialization::ThroughputOptimizer => NeuralNetworkConfig {
                input_size: 8,
                hidden_layers: vec![16, 12, 8],
                output_size: 1,
                learning_rate: 0.01,
                activation_function: "relu".to_string(),
            },
            AgentSpecialization::LatencyMinimizer => NeuralNetworkConfig {
                input_size: 8,
                hidden_layers: vec![12, 8],
                output_size: 1,
                learning_rate: 0.005,
                activation_function: "sigmoid".to_string(),
            },
            AgentSpecialization::EnergyEfficiencyExpert => NeuralNetworkConfig {
                input_size: 8,
                hidden_layers: vec![10, 6],
                output_size: 1,
                learning_rate: 0.002,
                activation_function: "tanh".to_string(),
            },
            AgentSpecialization::InterferenceAnalyst => NeuralNetworkConfig {
                input_size: 8,
                hidden_layers: vec![20, 16, 8],
                output_size: 1,
                learning_rate: 0.01,
                activation_function: "relu".to_string(),
            },
            AgentSpecialization::GeneralPurpose => NeuralNetworkConfig::default(),
        };
        
        NeuralNetwork::new(config)
    }
}