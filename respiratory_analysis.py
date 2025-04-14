import os
import argparse
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import tempfile
import warnings
import subprocess
warnings.filterwarnings('ignore')

def convert_audio_to_wav(input_path, output_path=None):
    """Convert audio file to WAV format using ffmpeg."""
    try:
        # If no output path specified, create one in the same directory
        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + '_converted.wav'
            
        print(f"Converting {input_path} to WAV format...")
        print(f"Input file exists: {os.path.exists(input_path)}")
        print(f"Input file size: {os.path.getsize(input_path)} bytes")
        
        # Use ffmpeg to convert the audio file
        ffmpeg_path = r"C:\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"
        command = [
            ffmpeg_path,
            '-i', input_path,  # Input file
            '-acodec', 'pcm_s16le',  # Output codec (standard WAV)
            '-ar', '44100',  # Sample rate
            '-ac', '1',  # Convert to mono
            '-y',  # Overwrite output file if it exists
            output_path
        ]
        
        # Run the ffmpeg command
        print(f"Running ffmpeg command: {' '.join(command)}")
        process = subprocess.run(command, capture_output=True, text=True)
        
        if process.returncode != 0:
            print(f"Error during conversion: {process.stderr}")
            return None
            
        if os.path.exists(output_path):
            print(f"Conversion successful: {output_path}")
            print(f"Output file exists: {os.path.exists(output_path)}")
            print(f"Output file size: {os.path.getsize(output_path)} bytes")
            return output_path
        else:
            print(f"Error: Output file was not created")
            return None
            
    except Exception as e:
        print(f"Error converting audio: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Full error traceback:")
        traceback.print_exc()
        return None

class RespiratoryModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(RespiratoryModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class RespiratoryDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class RespiratoryAnalysis:
    def __init__(self):
        self.sample_rate = 22050
        self.duration = 5  # seconds
        self.model = None
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_and_preprocess_audio(self, file_path):
        """Load and preprocess audio file."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} does not exist")
                return None

            # Convert audio to wav if it's not already
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext != '.wav':
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                    temp_wav_path = temp_wav.name
                    converted_path = convert_audio_to_wav(file_path, temp_wav_path)
                    if converted_path is None:
                        return None
                    file_path = converted_path

            print(f"Loading audio file: {file_path}")
            # Load audio file with librosa
            audio_data, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration, mono=True)

            # Clean up temporary file if it exists
            if file_ext != '.wav' and os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)

            if audio_data is None or len(audio_data) == 0:
                print("Error: No audio data loaded")
                return None
                
            print(f"Successfully loaded audio. Duration: {len(audio_data)/self.sample_rate:.2f} seconds")
                
            # Pad or trim audio to fixed length
            target_length = self.sample_rate * self.duration
            if len(audio_data) < target_length:
                print(f"Padding audio from {len(audio_data)} to {target_length} samples")
                audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
            else:
                print(f"Trimming audio from {len(audio_data)} to {target_length} samples")
                audio_data = audio_data[:target_length]
                
            return audio_data
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def extract_features(self, audio):
        """Extract relevant features from audio signal."""
        if audio is None:
            print("Cannot extract features: No audio data provided")
            return None
            
        try:
            # Extract MFCC features with more coefficients
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20)
            
            # Extract spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            
            # Extract spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            
            # Extract zero crossing rate
            zero_crossing = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Extract RMS energy
            rms = librosa.feature.rms(y=audio)[0]
            
            # Extract spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            
            # Combine features
            features = np.concatenate([
                mfccs.mean(axis=1),
                mfccs.std(axis=1),
                spectral_centroids.mean().reshape(1),
                spectral_rolloff.mean().reshape(1),
                zero_crossing.mean().reshape(1),
                rms.mean().reshape(1),
                spectral_bandwidth.mean().reshape(1)
            ])
            
            return features
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None

    def build_model(self, input_shape, num_classes):
        """Build the PyTorch model."""
        model = RespiratoryModel(input_shape, num_classes).to(self.device)
        return model

    def load_dataset(self, data_dir):
        """Load and prepare dataset for training."""
        features = []
        labels = []
        
        for condition in os.listdir(data_dir):
            condition_path = os.path.join(data_dir, condition)
            if os.path.isdir(condition_path):
                for audio_file in os.listdir(condition_path):
                    if audio_file.endswith('.wav'):
                        file_path = os.path.join(condition_path, audio_file)
                        audio = self.load_and_preprocess_audio(file_path)
                        if audio is not None:
                            features.append(self.extract_features(audio))
                            labels.append(condition)
        
        X = np.array(features)
        y = self.label_encoder.fit_transform(labels)
        return X, y

    def train(self, data_dir):
        """Train the model on the dataset."""
        print("Loading and preprocessing dataset...")
        X, y = self.load_dataset(data_dir)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create data loaders
        train_dataset = RespiratoryDataset(X_train, y_train)
        test_dataset = RespiratoryDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Build model
        self.model = self.build_model(X_train.shape[1], len(np.unique(y)))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Training loop
        epochs = 100  # Increased epochs
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            val_loss = val_loss / len(test_loader)
            val_acc = 100. * correct / total
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after epoch {epoch+1}')
                break
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Save model
        torch.save(self.model.state_dict(), 'respiratory_model.pth')
        # Save label encoder classes
        np.save('label_encoder_classes.npy', self.label_encoder.classes_)
        
        # Plot training history
        self.plot_training_history({
            'loss': train_losses,
            'val_loss': val_losses,
            'accuracy': train_accuracies,
            'val_accuracy': val_accuracies
        })

    def predict(self, audio_file):
        """Predict respiratory condition for a single audio file."""
        model_path = 'respiratory_model.pth'
        encoder_path = 'label_encoder_classes.npy'
        print(f"\nLooking for model files at:")
        print(f"Model path: {os.path.abspath(model_path)}")
        print(f"Encoder path: {os.path.abspath(encoder_path)}")
        
        if not os.path.exists(model_path):
            print("Error: No trained model found. Please train the model first.")
            return "Error", 0.0
        
        if not os.path.exists(encoder_path):
            print("Error: No label encoder state found. Please train the model first.")
            return "Error", 0.0
            
        # Load label encoder classes
        self.label_encoder.classes_ = np.load('label_encoder_classes.npy')
        
        # Load model if not already loaded
        if self.model is None:
            audio_data = self.load_and_preprocess_audio(audio_file)
            if audio_data is None:
                print("Error: Could not preprocess audio for model initialization")
                return "Error", 0.0
                
            input_features = self.extract_features(audio_data)
            if input_features is None:
                print("Error: Could not extract features for model initialization")
                return "Error", 0.0
                
            self.model = self.build_model(len(input_features), len(self.label_encoder.classes_))
            self.model.load_state_dict(torch.load('respiratory_model.pth'))
        
        self.model.eval()
        
        # Process audio file
        audio = self.load_and_preprocess_audio(audio_file)
        if audio is None:
            print("Error: Could not preprocess audio file")
            return "Error", 0.0
        
        # Extract features and predict
        features = self.extract_features(audio)
        if features is None:
            print("Error: Could not extract features from audio")
            return "Error", 0.0
            
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        try:
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[0][predicted_class].item()
            
            condition = self.label_encoder.inverse_transform([predicted_class])[0]
            return condition, confidence
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return "Error", 0.0

    def plot_training_history(self, history):
        """Plot training history."""
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Respiratory Sound Analysis')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                      help='Operation mode: train or predict')
    parser.add_argument('--input', help='Input audio file for prediction')
    parser.add_argument('--data_dir', default='data/raw',
                      help='Directory containing training data')
    
    args = parser.parse_args()
    
    analyzer = RespiratoryAnalysis()
    
    if args.mode == 'train':
        if not os.path.exists(args.data_dir):
            print(f"Error: Data directory '{args.data_dir}' not found.")
            return
        analyzer.train(args.data_dir)
        
    elif args.mode == 'predict':
        if not args.input:
            print("Error: Please provide input audio file for prediction")
            return
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found")
            return
            
        condition, confidence = analyzer.predict(args.input)
        print(f"\nPredicted condition: {condition}")
        print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    main() 