
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import json
import os

MAIN_FILE_READ_PATH: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Facial Emotion Recognition"
json_file_path: str = os.path.join(MAIN_FILE_READ_PATH, "config.json")

# Load the configuration file.
with open(json_file_path, "r") as f:
    config = json.load(f)

# Unpack the parameters into variables for easy acess.
train_cfg = config["training_params"]

# Paths to training and test files
MAIN_FILE_READ_PATH: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Facial Emotion Recognition"
CORRECTED_TRAINING_FILE_PATH: str = os.path.join(MAIN_FILE_READ_PATH, "/Corrected Dataset/train")
CORRECTED_TEST_FILE_PATH: str = os.path.join(MAIN_FILE_READ_PATH, "/Corrected Dataset/test")
OLD_TRAINING_FILE_PATH: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Facial Emotion Recognition/Old Dataset/train"
OLD_TEST_FILE_PATH: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Facial Emotion Recognition/Old Dataset/test"

active_training_path: str = OLD_TRAINING_FILE_PATH
active_test_path: str = OLD_TEST_FILE_PATH

#! This part is for fixing the bad dataset, doing it once is enough since it saves the new dataset to a directory.
#! Unmark this section if you want to do re-do the process.
"""
import os
from deepface import DeepFace
from PIL import Image

# Iterate over the images beforehand, to filter out the bad datas according to Deepface Model.
print("\nStarting with filtering-out process of bad data.")
emotions: list[str] = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

for emotion in emotions:
    training_read_file_path: str = OLD_TRAINING_FILE_PATH + "/" + emotion
    training_emotion_images: list = os.listdir(training_read_file_path)

    no_face_amount: int = 0
    for training_emotion_img in training_emotion_images:
        # If no face is found, just skip it.
        try: 
            training_emotion_img_path: str = training_read_file_path + "/" + training_emotion_img

            prediction = DeepFace.analyze(
                training_emotion_img_path, 
                actions=['emotion'], # Only check the emotion attribute.
                enforce_detection = True, # If a face is not recognised, raise an ValueError.
                silent = True, # Keeps the consol output clean.
            )
            predicted_emotion = prediction[0]["dominant_emotion"]

            if predicted_emotion == emotion:
                corrected_training_write_file_path: str = CORRECTED_TRAINING_FILE_PATH + "/" + predicted_emotion + "/" + training_emotion_img

                emotion_picture = Image.open(training_emotion_img_path)
                emotion_picture.save(corrected_training_write_file_path, format="JPEG")
        except ValueError:
            no_face_amount += 1

    # Repeat the same process for test data as well.
    test_read_file_path: str = OLD_TEST_FILE_PATH + "/" + emotion
    test_emotion_images: list = os.listdir(test_read_file_path)

    for test_emotion_img in test_emotion_images:
        try:
            test_emotion_img_path: str = test_read_file_path + "/" + test_emotion_img
            prediction = DeepFace.analyze(
                test_emotion_img_path, 
                actions=['emotion'], # Only check the emotion attribute.
                enforce_detection = True, # If a face is not recognised, raise an ValueError.
                silent = True, # Keeps the consol output clean.
            )
            predicted_emotion = prediction[0]["dominant_emotion"]

            if predicted_emotion == emotion:
                corrected_test_write_file_path: str = CORRECTED_TEST_FILE_PATH + "/" + predicted_emotion + "/" + test_emotion_img

                emotion_picture = Image.open(test_emotion_img_path)
                emotion_picture.save(corrected_test_write_file_path, format="JPEG")
        except ValueError:
            no_face_amount += 1
    print(f"{emotion} got filtered out.")

print("\nFiltered out the bad datas from the dataset using DeepFace Model.")
"""

# Custom Dataset for utilizing data augmentation on specific classes
class CustomDatasetLoader(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=None)
        # transform = None, because we will manually apply it.
        self.transform = transform

    def __getitem__(self, index):
        # Get the path and label for the image at the given index.
        path, target = self.samples[index]
        # Load the image from the path.
        sample = self.loader(path)

        # If a transformation is available, apply it on the image.
        if self.transform:
            sample = self.transform(sample)
        
        return sample, target

BATCH_SIZE: int = 32
"""
# Apply slight randomizations on the images to create very similar, but not same datas.
augmentation_pipeline = T.Compose(
    [ 
        # Convert all images to 1-channel grayscale.
        T.Grayscale(num_output_channels=1),
        # Flip 90 % of the images horizontally.
        T.RandomHorizontalFlip(p=0.5),
        # Rotate the images by a random angle between -15 and +15 degrees.
        T.RandomRotation(degrees=15),
        # Randomly zooms in on the image between 80 % and 120 %
        #? T.RandomAffine(degrees=0, scale=(0.8, 1.2), shear=10),
        # Apply a random perspective(shearing) transformation for % 40 of the time.
        #? T.RandomPerspective(distortion_scale=0.2, p=0.4),
        # Randomly change the brightness and contrast of the image.
        # T.ColorJitter(brightness=0.5, contrast=0.5),
        # Determine the size to 48 x 48.
        T.Resize(size=(48, 48)),
        # Always end with converting to tensor and normalizing.
        T.ToTensor(), # Converts to [0, 1] range and shape [C, H, W]
        # Randomly erase a rectangular regions in the image.
        #? T.RandomErasing(p=0.4, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        #       RandomErasing requires a Tensor to work on.
        T.Normalize(mean=(0.5,), std=(0.5,)), # Normalizes to [-1, 1] range
    ]
)
"""
# Define a seperate pipeline for testing.
standard_pipeline = T.Compose(
    [
        T.Grayscale(num_output_channels=1),
        T.Resize((48, 48)),
        T.ToTensor(),
        T.Normalize(mean=(0.5,), std=(0.5,),),
    ]
)

# Load the Dataset using ImageFolder with pipeline
try:
    train_dataset = CustomDatasetLoader(
        root = active_training_path,
        transform = standard_pipeline,
    )
except:
    print("Training dataset couldn't be loaded!")
    exit()

# bincount function can only search in tensors.
targets = torch.tensor(train_dataset.targets)

# We will create a custom sampler for training, since the dataset is unbalanced.

# Get the count of each emotion.
training_emotion_counts: torch.Tensor = torch.bincount(targets)
# Calculate weight for each emotion.
try:
    training_emotion_weights: torch.Tensor = 1.0 / training_emotion_counts
except ZeroDivisionError:
    print(f"Error: training_emotion_counts is zero, cannot divide by zero!")
    exit()
# Assign a weight to every single sample in the training dataset.
weighted_labels = training_emotion_weights[targets]

# Create WeightedRandomSampler: It will draw samples with probabilities proportional to their weights.
sampler = WeightedRandomSampler(
    weights = weighted_labels,
    num_samples = len(weighted_labels), # Draw this many samples in total per epoch.
    replacement = True, # Allows oversampling of rare emotions.
)

# Create DataLoader with the Dataset and Sampler
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = BATCH_SIZE,
    sampler = sampler, # The sampler handles the shuffling logic.
)

# Repeat the creating dataset for test files.
full_test_dataset = ImageFolder(
    root = active_test_path,
    transform = standard_pipeline
)

# We will create a validation set for dynamically adjusting learning rates while training.

# Define split sizes for test and validation.
test_subset_size: int = int(0.6 * len(full_test_dataset))
val_size: int = len(full_test_dataset) - test_subset_size

# Split the full test dataset into test and validation sets.
test_subset, val_subset = random_split(full_test_dataset, [test_subset_size, val_size])

test_loader = DataLoader(
    dataset = test_subset,
    batch_size = BATCH_SIZE,
    shuffle = False,
)
val_loader = DataLoader(
    dataset = val_subset,
    batch_size = BATCH_SIZE,
    shuffle = True,
)

print("\nData pre-processing was successful.")


