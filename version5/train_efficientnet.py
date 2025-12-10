#!/usr/bin/env python3
"""
Train EfficientNet for ROSHN Facade Classification
===================================================
Uses transfer learning with data augmentation to achieve high accuracy
on small datasets (72 facade images: 36 modern + 36 traditional).

Features:
- Transfer learning from EfficientNetB0 (pretrained on ImageNet)
- Aggressive data augmentation (flip, rotate, zoom, brightness)
- GPU acceleration (RTX 5070)
- Early stopping & model checkpointing
"""

import os
import sys
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    logger.info(f"✓ TensorFlow {tf.__version__}")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"✓ GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            logger.info(f"  {gpu}")
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.info("Using CPU (no GPU detected)")
        
except ImportError:
    logger.error("TensorFlow not installed. Install with:")
    logger.error("  pip install tensorflow")
    sys.exit(1)


class EfficientNetFacadeClassifier:
    """
    Transfer learning classifier for facade types.
    Uses EfficientNetB0 pretrained on ImageNet.
    """
    
    def __init__(self, img_size: int = 224, learning_rate: float = 0.001):
        """
        Initialize classifier.
        
        Args:
            img_size: Input image size (224 for EfficientNetB0)
            learning_rate: Initial learning rate
        """
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.model = None
        self.class_names = ['modern', 'traditional']
        
    def build_model(self):
        """Build transfer learning model."""
        logger.info("Building EfficientNetB0 model...")
        
        # Load pretrained EfficientNetB0 (without top classification layer)
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze base model (we only train the top layers)
        base_model.trainable = False
        
        # Build classification head
        inputs = keras.Input(shape=(self.img_size, self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(2, activation='softmax')(x)  # 2 classes: modern, traditional
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        logger.info(f"✓ Model built: {self.model.count_params():,} total params")
        logger.info(f"  Trainable params: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
        
    def create_data_generators(self, train_dir: str, val_split: float = 0.2, batch_size: int = 16):
        """
        Create data generators with aggressive augmentation.
        
        Args:
            train_dir: Directory with modern/ and traditional/ subdirectories
            val_split: Validation split ratio (0.2 = 20%)
            batch_size: Batch size for training
        """
        logger.info("Creating data generators with augmentation...")
        
        # Training data augmentation (AGGRESSIVE)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,           # Rotate ±20 degrees
            width_shift_range=0.2,       # Horizontal shift
            height_shift_range=0.2,      # Vertical shift
            shear_range=0.15,            # Shear transformation
            zoom_range=0.2,              # Zoom in/out
            horizontal_flip=True,        # Flip horizontally
            brightness_range=[0.8, 1.2], # Brightness variation
            fill_mode='nearest',
            validation_split=val_split
        )
        
        # Validation data (no augmentation, only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=val_split
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        val_generator = val_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        logger.info(f"✓ Training samples: {train_generator.samples}")
        logger.info(f"✓ Validation samples: {val_generator.samples}")
        logger.info(f"✓ Classes: {train_generator.class_indices}")
        
        return train_generator, val_generator
    
    def train(self, train_dir: str, epochs: int = 30, batch_size: int = 16, 
              val_split: float = 0.2, checkpoint_path: str = './models/best_model.h5'):
        """
        Train the model.
        
        Args:
            train_dir: Directory with facade_types/{modern,traditional}
            epochs: Number of training epochs
            batch_size: Batch size
            val_split: Validation split
            checkpoint_path: Path to save best model
        """
        if self.model is None:
            self.build_model()
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators(train_dir, val_split, batch_size)
        
        # Callbacks
        callbacks = [
            # Save best model
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            )
        ]
        
        # Train
        logger.info(f"Starting training for {epochs} epochs...")
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Print final results
        final_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Final Training Accuracy: {final_acc:.2%}")
        logger.info(f"Final Validation Accuracy: {final_val_acc:.2%}")
        logger.info(f"Best model saved: {checkpoint_path}")
        logger.info(f"{'='*60}\n")
        
        return history
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Predict facade type for an image.
        
        Args:
            image_path: Path to facade image
            
        Returns:
            (prediction, confidence)
        """
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(
            image_path,
            target_size=(self.img_size, self.img_size)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        prediction = self.class_names[class_idx]
        return prediction, float(confidence)
    
    def save(self, model_path: str, metadata_path: str = None):
        """Save model and metadata."""
        self.model.save(model_path)
        logger.info(f"✓ Model saved: {model_path}")
        
        if metadata_path:
            metadata = {
                'img_size': self.img_size,
                'class_names': self.class_names,
                'learning_rate': self.learning_rate
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"✓ Metadata saved: {metadata_path}")
    
    def load(self, model_path: str, metadata_path: str = None):
        """Load model and metadata."""
        self.model = keras.models.load_model(model_path)
        logger.info(f"✓ Model loaded: {model_path}")
        
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            self.img_size = metadata['img_size']
            self.class_names = metadata['class_names']
            self.learning_rate = metadata.get('learning_rate', 0.001)
            logger.info(f"✓ Metadata loaded: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Train EfficientNet facade classifier')
    parser.add_argument('--train-dir', default='./reference_images/facade_types',
                       help='Training directory with modern/ and traditional/ subdirs')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio (0.2 = 20%%)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--output', default='./models/efficientnet_facade.h5',
                       help='Output model path')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize classifier
    classifier = EfficientNetFacadeClassifier(learning_rate=args.lr)
    
    # Train
    history = classifier.train(
        train_dir=args.train_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        checkpoint_path=args.output
    )
    
    # Save final model
    metadata_path = args.output.replace('.h5', '_metadata.pkl')
    classifier.load(args.output)  # Load best checkpoint
    classifier.save(args.output, metadata_path)
    
    logger.info("\n✓ Training complete!")
    logger.info(f"  Model: {args.output}")
    logger.info(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
