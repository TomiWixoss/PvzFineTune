# -*- coding: utf-8 -*-
"""
Validate Dataset - Validate và clean training data
"""

import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pvz_ai.labeler.validator import ActionValidator


def validate_dataset(input_path: str, output_path: str = None, fix: bool = False):
    """
    Validate training dataset và optionally output cleaned version
    
    Args:
        input_path: Path to training data JSON
        output_path: Path to save cleaned data (optional)
        fix: If True, save only valid samples
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"📂 Loaded {len(samples)} samples from {input_path.name}")
    print("=" * 50)
    
    # Validate
    result = ActionValidator.validate_training_data(samples)
    
    # Print results
    print(f"\n📊 Validation Results:")
    print(f"   Total: {result['total']}")
    print(f"   Valid: {result['valid_count']} ✓")
    print(f"   Invalid: {result['invalid_count']} ✗")
    print(f"   Score: {result['score']:.1f}%")
    
    if result['errors']:
        print(f"\n❌ Errors ({len(result['errors'])}):")
        for err in result['errors'][:20]:
            print(f"   {err}")
        if len(result['errors']) > 20:
            print(f"   ... và {len(result['errors']) - 20} lỗi khác")
    
    # Save cleaned data if requested
    if fix and output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result['valid_samples'], f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved {len(result['valid_samples'])} valid samples to {output_path}")
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Validate training dataset')
    parser.add_argument('input', help='Input training data JSON')
    parser.add_argument('-o', '--output', help='Output cleaned data JSON')
    parser.add_argument('--fix', action='store_true', help='Save only valid samples')
    args = parser.parse_args()
    
    validate_dataset(args.input, args.output, args.fix)


if __name__ == "__main__":
    main()
