#!/usr/bin/env python3
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Converts a trained Keras model to TFLite format and tests accuracy.

This script loads a trained Keras model, converts it to TFLite format with
quantization, and tests the accuracy of both the Keras model and TFLite model.

Example usage:
python convert_model.py --weights saved.model.keras
"""

import argparse
from pathlib import Path
from typing import Callable, Iterable, Tuple

import numpy as np
import tensorflow as tf

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def _load_representative_npz(npz_path: Path) -> Tuple[np.lib.npyio.NpzFile, np.memmap]:
    """Load the NPZ file containing mel-spectrograms for representative data."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Representative dataset '{npz_path}' was not found.")

    npz_file = np.load(npz_path, mmap_mode='r')
    if 'x' not in npz_file:
        raise KeyError(f"Expected key 'x' inside {npz_path}, but only found {list(npz_file.files)}")
    return npz_file, npz_file['x']


def build_representative_dataset(samples: np.memmap, max_examples: int = 500,
                                 random_seed: int = 1337) -> Callable[[], Iterable[Tuple[np.ndarray]]]:
    """Create a representative dataset generator for post-training quantization."""
    total = samples.shape[0]
    take = min(max_examples, total)
    rng = np.random.default_rng(seed=random_seed)
    indices = rng.choice(total, size=take, replace=False) if take < total else np.arange(total)

    def generator():
        for idx in indices:
            sample = samples[idx]
            # Ensure shape (1, time, freq) and float32 dtype for converter.
            yield [np.expand_dims(sample.astype(np.float32), axis=0)]

    return generator


def _build_lstm_concrete_function(model: tf.keras.Model, batch_size: int) -> Tuple[tf.types.experimental.ConcreteFunction, tf.TensorSpec]:
    """Create a concrete function with a fixed batch size for LSTM lowering."""
    if not model.inputs:
        raise ValueError("Model must have at least one input tensor.")

    input_tensor = model.inputs[0]
    input_shape = tf.TensorShape(input_tensor.shape)

    if input_shape.rank != 3:
        raise ValueError(
            "LiteRT RNN conversion expects a 3D input tensor (batch, time, features)."
        )

    if input_shape[1] is None or input_shape[2] is None:
        raise ValueError(
            "Model input time/frequency dimensions must be static to lower TensorList ops."
        )

    dtype = input_tensor.dtype or tf.float32
    spec = tf.TensorSpec(
    [batch_size, int(input_shape[1]), int(input_shape[2])],
        dtype=dtype,
        name=input_tensor.name.split(':')[0],
    )
    concrete_fn = tf.function(model, jit_compile=False).get_concrete_function(spec)
    return concrete_fn, spec


def convert_to_tflite(model_path: Path,
                      output_path: Path = Path('model.tflite'),
                      rep_data_path: Path = Path('all-waves.npz'),
                      quantization: str = 'dynamic',
                      representative_samples: int = 512,
                      representative_seed: int = 1337,
                      batch_size: int = 1,
                      best_weights: Path | None = Path('saved.model/best.weights.h5')) -> Tuple[tf.keras.Model, np.lib.npyio.NpzFile]:
    """Convert Keras model to LiteRT-friendly TFLite format."""
    quantization = quantization.lower()
    if quantization not in {'int8', 'float16', 'dynamic', 'float32', 'none'}:
        raise ValueError("quantization must be one of: int8, float16, dynamic, float32/none")

    if quantization == 'none':
        quantization = 'float32'

    print(f"Loading Keras model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    if best_weights is not None:
        if best_weights.exists():
            print(f"Restoring trained weights from {best_weights}...")
            model.load_weights(best_weights)
        else:
            print(f"{bcolors.WARNING}Warning:{bcolors.ENDC} Requested best weights '{best_weights}' not found. Continuing without extra weights.")
    model.summary()

    concrete_fn, input_spec = _build_lstm_concrete_function(model, batch_size)
    print(
        "Using LiteRT RNN lowering with fixed input signature"
        f" batch={input_spec.shape[0]}, time={input_spec.shape[1]}, features={input_spec.shape[2]}"
    )

    representative_npz, representative_samples_memmap = _load_representative_npz(rep_data_path)
    representative_dataset = None
    if quantization == 'int8':
        representative_dataset = build_representative_dataset(
            representative_samples_memmap,
            max_examples=representative_samples,
            random_seed=representative_seed,
        )

    print(f"Converting to LiteRT format using '{quantization}' quantization...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], model)
    converter._experimental_lower_tensor_list_ops = True  # Required to fuse TensorList ops into SequenceLSTM

    if quantization == 'float32':
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    elif quantization == 'dynamic':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    elif quantization == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.target_spec.supported_types = [tf.float16]
    else:  # int8
        if representative_dataset is None:
            raise ValueError("INT8 quantization requires a representative dataset.")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)

    print(f"LiteRT model ({quantization}) saved to {output_path}")
    return model, representative_npz


def test_keras_accuracy(model, data, num_samples=1000):
    """Test the accuracy of the Keras model."""
    print(f"\nTesting Keras model accuracy on {num_samples} samples...")
    
    dataset = tf.data.Dataset.from_tensor_slices((data['x'], data['y']))
    misses_model = 0
    count = 0
    
    for element in dataset.take(num_samples):
        mels, y = element
        input_data = np.float32(np.reshape(mels, (1, mels.shape[0], mels.shape[1])))

        true_value = int(y.numpy()[1] * 256)
        predictions_model = int(model(input_data).numpy()[0][1] * 256 + 0.5)
        
        is_correct = abs(true_value - predictions_model) <= 70
        color = bcolors.OKGREEN if is_correct else bcolors.FAIL
        
        print(f"true: {true_value:3d}, model: {color}{predictions_model:3d}{bcolors.ENDC}")
        
        count += 1
        if not is_correct:
            misses_model += 1

    accuracy = 100 - float(misses_model) / count * 100
    print(f"\n{bcolors.BOLD}Keras Model Accuracy: {accuracy:.2f}%{bcolors.ENDC}")
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Convert Keras model to TFLite and test accuracy')
    parser.add_argument(
        '--weights',
        type=Path,
        default=Path('saved.model.keras'),
        help='Path to the Keras model file (default: saved.model.keras)')
    parser.add_argument(
        '--weight',
        dest='weights',
        type=Path,
        help='Alias for --weights')
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('model.tflite'),
        help='Output path for LiteRT model (default: model.tflite)')
    parser.add_argument(
        '--test-samples',
        type=int,
        default=1000,
        help='Number of samples to test for accuracy (default: 1000)')
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip accuracy testing')
    parser.add_argument(
        '--quantization',
        choices=['int8', 'float16', 'dynamic', 'float32', 'none'],
        default='dynamic',
        help='Quantization strategy for LiteRT model (default: dynamic)')
    parser.add_argument(
        '--rep-data',
        type=Path,
        default=Path('all-waves.npz'),
        help='NPZ file containing representative dataset (default: all-waves.npz)')
    parser.add_argument(
        '--rep-samples',
        type=int,
        default=512,
        help='Number of representative samples to use for INT8 quantization (default: 512)')
    parser.add_argument(
        '--rep-seed',
        type=int,
        default=1337,
        help='Random seed when sampling representative dataset (default: 1337)')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Fixed batch size for the LiteRT input signature (default: 1)')
    parser.add_argument(
        '--best-weights',
        type=Path,
        default=Path('saved.model/best.weights.h5'),
        help='Optional weights file to load into the model before conversion (default: saved.model/best.weights.h5)')
    parser.add_argument(
        '--skip-best-weights',
        action='store_true',
        help='Do not attempt to load an external weights file before conversion')
    
    args = parser.parse_args()

    data = None
    
    try:
        model, data = convert_to_tflite(
            args.weights,
            args.output,
            args.rep_data,
            args.quantization,
            args.rep_samples,
            args.rep_seed,
            args.batch_size,
            None if args.skip_best_weights else args.best_weights,
        )
        
        if not args.skip_test:
            test_keras_accuracy(model, data, args.test_samples)
        
        print(f"\n{bcolors.OKGREEN}Conversion complete!{bcolors.ENDC}")
        print(f"TFLite model saved to: {args.output}")
        print("\nYou can now copy the model to Raspberry Pi and run it with litert_run_wav.py.")
        
    except Exception as e:
        print(f"{bcolors.FAIL}Error: {e}{bcolors.ENDC}")
        return 1
    finally:
        if data is not None:
            try:
                data.close()
            except Exception:
                pass
    
    return 0


if __name__ == '__main__':
    exit(main())
