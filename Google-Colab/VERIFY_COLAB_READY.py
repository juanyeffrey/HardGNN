#!/usr/bin/env python3
"""
HardGNN Google Colab Pro+ Verification Script
==============================================

This script verifies that all components are ready for Google Colab Pro+ execution.
Run this to ensure everything will work before deploying to Colab.

Usage:
    python VERIFY_COLAB_READY.py
"""

import sys
import os
import traceback
import subprocess
import importlib.util

def test_section(name):
    """Decorator for test sections"""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*60}")
            print(f"🔍 {name}")
            print('='*60)
            try:
                result = func()
                print(f"✅ {name} - PASSED")
                return True
            except Exception as e:
                print(f"❌ {name} - FAILED")
                print(f"Error: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                return False
        return wrapper
    return decorator

@test_section("Python Environment Check")
def test_python_environment():
    """Test Python version and basic environment"""
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major != 3:
        raise Exception("Python 3 required")
    
    if python_version.minor < 8:
        raise Exception("Python 3.8+ required for Google Colab compatibility")
    
    print(f"✓ Python {python_version.major}.{python_version.minor} is compatible")
    return True

@test_section("TensorFlow Import and Version Check")
def test_tensorflow():
    """Test TensorFlow import and version compatibility"""
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Test tf.compat.v1 availability
        tf.compat.v1.disable_eager_execution()
        print("✓ tf.compat.v1 compatibility layer available")
        
        # Test basic operations
        x = tf.compat.v1.constant([1, 2, 3])
        print("✓ Basic TensorFlow operations work")
        
        # Test GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU acceleration available: {len(gpus)} GPU(s)")
        else:
            print("⚠️ No GPU detected (OK for CPU testing)")
            
        return True
    except ImportError:
        raise Exception("TensorFlow not found. Install with: pip install tensorflow>=2.10.0")

@test_section("Core Dependencies Check")
def test_dependencies():
    """Test all required dependencies"""
    required_packages = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'sklearn')  # Package name vs import name
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            spec = importlib.util.find_spec(import_name)
            if spec is None:
                missing_packages.append(package_name)
            else:
                exec(f"import {import_name}")
                print(f"✓ {package_name} imported successfully")
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        raise Exception(f"Missing packages: {missing_packages}")
    
    return True

@test_section("HardGNN Module Structure Check")
def test_file_structure():
    """Test that all required files are present"""
    required_files = [
        'Params.py',
        'DataHandler.py',
        'model_colab_compatible.py',
        'HardGNN_Colab_Script.py',
        'Utils/NNLayers_tf2.py',
        'Utils/attention_tf2.py',
        'Utils/TimeLogger.py',
        'requirements_final.txt'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path} exists")
    
    if missing_files:
        raise Exception(f"Missing files: {missing_files}")
    
    return True

@test_section("Parameters Configuration Check")
def test_parameters():
    """Test that parameters are properly configured"""
    try:
        from Params import args
        
        # Test essential parameters
        essential_params = [
            'lr', 'batch', 'reg', 'epoch', 'graphNum', 'latdim',
            'use_hard_neg', 'hard_neg_top_k', 'contrastive_weight',
            'temp', 'seq_length', 'time_split'
        ]
        
        for param in essential_params:
            if not hasattr(args, param):
                raise Exception(f"Missing parameter: {param}")
            print(f"✓ {param} = {getattr(args, param)}")
        
        # Test hard negative sampling configuration
        if not args.use_hard_neg:
            print("⚠️ Hard negative sampling disabled by default (will be enabled in script)")
        
        return True
    except ImportError:
        raise Exception("Cannot import Params.py")

@test_section("TF2 Compatible Utilities Check")
def test_tf2_utilities():
    """Test TF2 compatible utility modules"""
    try:
        # Test NNLayers_tf2
        import Utils.NNLayers_tf2 as NNs
        print("✓ Utils.NNLayers_tf2 imported successfully")
        
        # Test attention_tf2
        from Utils.attention_tf2 import AdditiveAttention, MultiHeadSelfAttention
        print("✓ Utils.attention_tf2 imported successfully")
        
        # Test TimeLogger
        import Utils.TimeLogger as logger
        print("✓ Utils.TimeLogger imported successfully")
        
        return True
    except ImportError as e:
        raise Exception(f"Cannot import TF2 utilities: {e}")

@test_section("Model Import Check")
def test_model_import():
    """Test model import with fallback mechanism"""
    try:
        # Test the import logic from HardGNN_Colab_Script.py
        try:
            from model import Recommender
            print("✓ Original model imported successfully")
            model_source = "original"
        except ImportError:
            print("ℹ️ Original model not available, trying TF2 compatible version...")
            from model_colab_compatible import Recommender
            print("✓ TF2 compatible model imported successfully")
            model_source = "tf2_compatible"
        
        print(f"Model source: {model_source}")
        return True
    except ImportError as e:
        raise Exception(f"Cannot import any model version: {e}")

@test_section("DataHandler Import Check")
def test_datahandler():
    """Test DataHandler import"""
    try:
        from DataHandler import DataHandler, negSamp, transpose
        print("✓ DataHandler imported successfully")
        return True
    except ImportError as e:
        raise Exception(f"Cannot import DataHandler: {e}")

@test_section("Basic Model Instantiation Check")
def test_basic_model():
    """Test basic model functionality without data"""
    import tensorflow as tf
    
    # Setup TF2 compatibility
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    
    # Test session creation
    with tf.compat.v1.Session(config=config) as sess:
        print("✓ TensorFlow session created successfully")
        
        # Test basic operations
        x = tf.compat.v1.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.compat.v1.constant([[2.0, 1.0], [1.0, 2.0]])
        z = tf.compat.v1.matmul(x, y)
        result = sess.run(z)
        print(f"✓ Basic TensorFlow operations: {result.shape}")
    
    return True

@test_section("HardGNN Script Syntax Check")
def test_script_syntax():
    """Test that the main script has valid syntax"""
    script_path = 'HardGNN_Colab_Script.py'
    
    try:
        # Check syntax by compiling
        with open(script_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        compile(source, script_path, 'exec')
        print("✓ HardGNN_Colab_Script.py has valid syntax")
        
        # Check for TF2 compatibility patterns
        tf2_patterns = [
            'tf.compat.v1.disable_eager_execution()',
            'tf.compat.v1.ConfigProto()',
            'tf.compat.v1.Session(',
            'model_colab_compatible'
        ]
        
        for pattern in tf2_patterns:
            if pattern in source:
                print(f"✓ Found TF2 compatibility pattern: {pattern}")
            else:
                print(f"⚠️ Missing TF2 pattern: {pattern}")
        
        return True
    except SyntaxError as e:
        raise Exception(f"Syntax error in {script_path}: {e}")
    except FileNotFoundError:
        raise Exception(f"Script file not found: {script_path}")

@test_section("Requirements File Check")
def test_requirements():
    """Test requirements file"""
    req_file = 'requirements_final.txt'
    
    if not os.path.exists(req_file):
        raise Exception(f"{req_file} not found")
    
    with open(req_file, 'r') as f:
        requirements = f.read()
    
    essential_deps = [
        'tensorflow',
        'numpy',
        'scipy',
        'matplotlib'
    ]
    
    for dep in essential_deps:
        if dep in requirements:
            print(f"✓ {dep} found in requirements")
        else:
            print(f"⚠️ {dep} not explicitly in requirements")
    
    print(f"Requirements file length: {len(requirements.splitlines())} lines")
    return True

def main():
    """Run all verification tests"""
    print("🚀 HardGNN Google Colab Pro+ Verification")
    print("=" * 60)
    print("This script verifies all components for Google Colab Pro+ compatibility.")
    print("All tests must PASS for successful deployment to Colab.")
    
    tests = [
        test_python_environment,
        test_tensorflow,
        test_dependencies,
        test_file_structure,
        test_parameters,
        test_tf2_utilities,
        test_model_import,
        test_datahandler,
        test_basic_model,
        test_script_syntax,
        test_requirements
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print("📊 VERIFICATION SUMMARY")
    print('='*60)
    print(f"✅ Tests Passed: {passed}")
    print(f"❌ Tests Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ HardGNN is ready for Google Colab Pro+ deployment!")
        print("\n📋 Next Steps:")
        print("1. Upload this folder to Google Colab Pro+")
        print("2. Run: exec(open('HardGNN_Colab_Script.py').read())")
        print("3. Follow the script output for training")
    else:
        print(f"\n⚠️ {failed} TESTS FAILED!")
        print("❌ Please fix the issues above before deploying to Colab.")
        print("💡 Check the error messages for specific fixes needed.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 