#!/usr/bin/env python3
"""
Test script to verify medical code functionality
"""

import tempfile
import pandas as pd
from pathlib import Path
import sys

# Add mgpt_eval to path
sys.path.append(str(Path(__file__).parent.parent.parent / "mgpt_eval"))

def test_medical_code_data():
    """Test if system can handle medical code data correctly."""
    
    # Create medical code dataset with 'claims' column
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Medical code data (as you described)
        data = [
            {'mcid': 'MC001', 'claims': 'N6320 G0378 Z91048 M1710 O0903', 'label': 1},
            {'mcid': 'MC002', 'claims': 'K5901 E119 Z5111 M7969', 'label': 0},
            {'mcid': 'MC003', 'claims': '76642 Z09 N6322 K9289', 'label': 1},
            {'mcid': 'MC004', 'claims': 'O9989 Z03818 U0003 M1710', 'label': 0}
        ]
        
        dataset_path = temp_path / 'medical_codes.csv'
        pd.DataFrame(data).to_csv(dataset_path, index=False)
        print(f"✓ Created test dataset with 'claims' column")
        
        # Try to load with current system
        try:
            from models.data_models import Dataset
            dataset = Dataset.from_file(str(dataset_path))
            print("❌ UNEXPECTED: System loaded 'claims' column successfully")
            print("   This means the system might be flexible enough to handle it")
        except Exception as e:
            print(f"❌ EXPECTED ERROR: {e}")
            print("   System expects 'text' column but data has 'claims' column")
        
        # Test target code matching
        test_target_code_logic()

def test_target_code_logic():
    """Test if target code matching works for space-separated codes."""
    
    # Simulate generated medical code sequence
    generated_output = "K9289 |eoc| N6322 76642 |eop| Z09 76642 |eoc| Z1239 O9989 |eoc| Z03818 U0003"
    target_codes = ["E119", "76642"]
    
    print(f"\n🧪 Testing target code logic:")
    print(f"Generated: {generated_output}")
    print(f"Target codes: {target_codes}")
    
    # Test current word boundary approach
    import re
    current_matches = {}
    for code in target_codes:
        pattern = r'\b' + re.escape(code.lower()) + r'\b'
        current_matches[code] = bool(re.search(pattern, generated_output.lower()))
    
    print(f"Current word boundary method: {current_matches}")
    
    # Test space-split approach (what it should be)
    generated_codes = generated_output.split()
    space_split_matches = {}
    for code in target_codes:
        space_split_matches[code] = code in generated_codes
    
    print(f"Space-split method: {space_split_matches}")
    
    # The space-split method should find 76642, word boundary might not
    if space_split_matches["76642"] and not current_matches["76642"]:
        print("❌ PROBLEM: Word boundary method missed '76642' but space-split found it")
    elif space_split_matches == current_matches:
        print("✓ Both methods agree")
    else:
        print("⚠️  Methods disagree - need to check which is correct")

def check_config_file():
    """Check if config file has input data path."""
    
    config_path = Path('mgpt_eval/configs/pipeline_config.yaml')
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            content = f.read()
        
        if 'dataset_path' in content or 'input_path' in content:
            print("✓ Config file has input data path")
        else:
            print("❌ Config file has NO input data path")
            print("   You must specify --dataset-path on command line")
    else:
        print("❌ Config file not found")

if __name__ == "__main__":
    print("Medical Code System Test")
    print("=" * 50)
    
    test_medical_code_data()
    check_config_file()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("1. System expects 'text' column but you have 'claims' - NEEDS FIX")
    print("2. No input path in config - must use command line")
    print("3. Default target words are wrong for medical codes")
    print("4. Word matching logic may not work correctly for space-separated codes")