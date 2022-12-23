from pathlib import Path
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
#Media Folder (where to store samples)
SAMPLES_ROOT =  os.path.join(BASE_DIR, 'samples')
SAMPLES_URL = '/samples/'

# Model dolfer
MODELS_ROOT = os.path.join(BASE_DIR, "api", "utils", "recognition", "saved_models")
LABELS_ROOT = os.path.join(BASE_DIR, "api", "utils", "recognition", "pickles")  
