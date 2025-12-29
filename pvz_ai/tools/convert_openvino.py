# -*- coding: utf-8 -*-
"""
Convert PyTorch model to OpenVINO IR format
Usage: python -m pvz_ai.tools.convert_openvino <pytorch_model_path>
"""

import shutil
from pathlib import Path


def convert_to_openvino(
    pytorch_path: str,
    output_path: str = "models/gemma",
    clear_output: bool = True,
    delete_source: bool = False
):
    """
    Convert PyTorch model to OpenVINO IR format
    
    Args:
        pytorch_path: Path to PyTorch model folder
        output_path: Output path for OpenVINO model
        clear_output: Clear output folder before saving
        delete_source: Delete source PyTorch folder after conversion
    """
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer
    
    pytorch_path = Path(pytorch_path)
    output_path = Path(output_path)
    
    if not pytorch_path.exists():
        raise FileNotFoundError(f"PyTorch model not found: {pytorch_path}")
    
    # Clear output folder if exists
    if clear_output and output_path.exists():
        print(f"Clearing {output_path}...")
        shutil.rmtree(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading PyTorch model from {pytorch_path}...")
    print("Converting to OpenVINO IR format...")
    
    # Load and convert
    ov_model = OVModelForCausalLM.from_pretrained(
        str(pytorch_path),
        export=True,
        compile=False,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(pytorch_path),
        local_files_only=True,
    )
    
    # Save
    print(f"Saving to {output_path}...")
    ov_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Verify
    files = list(output_path.glob("*"))
    print(f"\n✓ OpenVINO model saved!")
    print(f"  Files: {len(files)}")
    for f in sorted(files):
        size = f.stat().st_size / 1024 / 1024
        print(f"    {f.name}: {size:.1f}MB" if size > 1 else f"    {f.name}")
    
    # Delete source folder
    if delete_source and pytorch_path.exists():
        print(f"\nDeleting source folder {pytorch_path}...")
        shutil.rmtree(pytorch_path)
        print("✓ Source folder deleted")
    
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert PyTorch to OpenVINO')
    parser.add_argument('pytorch_path', help='Path to PyTorch model')
    parser.add_argument('-o', '--output', default='models/gemma', help='Output path')
    parser.add_argument('--no-clear', action='store_true', help='Do not clear output folder')
    parser.add_argument('--delete-source', action='store_true', help='Delete source PyTorch folder after conversion')
    args = parser.parse_args()
    
    convert_to_openvino(
        args.pytorch_path,
        args.output,
        clear_output=not args.no_clear,
        delete_source=args.delete_source
    )


if __name__ == "__main__":
    main()
