#!/usr/bin/env python3
"""
Fix DeepFace protobuf issues by patching the library to handle protobuf errors gracefully
"""

import os
import sys

# Set protobuf to use pure Python implementation before any imports
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow verbosity

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def patch_deepface():
    """Apply patches to DeepFace to fix protobuf issues"""
    try:
        import deepface
        from deepface.commons import functions
        
        # Patch the model building to handle protobuf errors
        original_build_model = deepface.DeepFace.build_model
        
        def patched_build_model(model_name):
            """Patched version that handles protobuf errors"""
            try:
                return original_build_model(model_name)
            except Exception as e:
                if "protobuf" in str(e).lower() or "RepeatedCompositeFieldContainer" in str(e):
                    print(f"Protobuf error with {model_name}, trying alternative...")
                    # Try alternative models
                    alternatives = ["VGG-Face", "OpenFace", "Dlib"]
                    for alt in alternatives:
                        if alt != model_name:
                            try:
                                print(f"Trying {alt}...")
                                return original_build_model(alt)
                            except:
                                continue
                raise e
        
        deepface.DeepFace.build_model = patched_build_model
        print("DeepFace patched successfully")
        
    except ImportError:
        print("DeepFace not installed")


def test_deepface_models():
    """Test which DeepFace models work without protobuf errors"""
    from deepface import DeepFace
    import numpy as np
    
    # Create a dummy image
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_image[30:70, 30:70] = 255  # White square as face
    
    models = ["VGG-Face", "OpenFace", "Facenet", "Facenet512", "DeepFace", "DeepID", "ArcFace", "Dlib"]
    working_models = []
    
    print("\nTesting DeepFace models:")
    print("-" * 50)
    
    for model in models:
        try:
            print(f"Testing {model}...", end=" ")
            # Try to build the model
            DeepFace.build_model(model)
            
            # Try to extract features
            result = DeepFace.represent(
                img_path=dummy_image,
                model_name=model,
                enforce_detection=False,
                detector_backend="opencv"
            )
            
            print(f"✓ Success (embedding size: {len(result[0]['embedding'])})")
            working_models.append(model)
            
        except Exception as e:
            error_msg = str(e)
            if "protobuf" in error_msg.lower():
                print("✗ Protobuf error")
            elif "RepeatedCompositeFieldContainer" in error_msg:
                print("✗ Protobuf container error")
            else:
                print(f"✗ Error: {error_msg[:50]}...")
    
    print("\nWorking models:", working_models)
    return working_models


def create_deepface_config(working_models):
    """Create a configuration file with working models"""
    config_content = f"""# DeepFace Configuration
# Auto-generated based on models that work without protobuf errors

WORKING_MODELS = {working_models}

# Recommended model (first working model)
DEFAULT_MODEL = "{working_models[0] if working_models else 'VGG-Face'}"

# Recommended detector (least problematic)
DEFAULT_DETECTOR = "opencv"

# Settings to avoid issues
ENFORCE_DETECTION = False
ALIGN = True
NORMALIZATION = "base"
"""
    
    config_path = os.path.join(os.path.dirname(__file__), "deepface_config.py")
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"\nConfiguration saved to: {config_path}")


if __name__ == "__main__":
    print("DeepFace Protobuf Fix Script")
    print("=" * 50)
    
    # Apply patches
    patch_deepface()
    
    # Test models
    working_models = test_deepface_models()
    
    # Create config
    if working_models:
        create_deepface_config(working_models)
    else:
        print("\nWARNING: No models work without errors!")
        print("Consider using a different face recognition library")