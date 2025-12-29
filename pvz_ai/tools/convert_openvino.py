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
    delete_source: bool = False,
    compress_fp16: bool = True
):
    """
    Convert PyTorch model to OpenVINO IR format
    
    Args:
        pytorch_path: Path to PyTorch model folder
        output_path: Output path for OpenVINO model
        clear_output: Clear output folder before saving
        delete_source: Delete source PyTorch folder after conversion
        compress_fp16: Compress weights to FP16 (reduces size by ~50%)
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
    print(f"Converting to OpenVINO IR format (FP16={compress_fp16})...")
    
    # Load and convert with FP16 compression
    ov_model = OVModelForCausalLM.from_pretrained(
        str(pytorch_path),
        export=True,
        compile=False,
        local_files_only=True,
        load_in_8bit=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(pytorch_path),
        local_files_only=True,
    )
    
    # Save with FP16 compression
    print(f"Saving to {output_path}...")
    if compress_fp16:
        from openvino import save_model
        save_model(ov_model.model, output_path / "openvino_model.xml", compress_to_fp16=True)
        # Save config and tokenizer separately
        ov_model.config.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        # Copy generation_config if exists
        gen_config = pytorch_path / "generation_config.json"
        if gen_config.exists():
            shutil.copy(gen_config, output_path / "generation_config.json")
    else:
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
    parser.add_argument('--no-fp16', action='store_true', help='Do not compress to FP16')
    args = parser.parse_args()
    
    convert_to_openvino(
        args.pytorch_path,
        args.output,
        clear_output=not args.no_clear,
        delete_source=args.delete_source,
        compress_fp16=not args.no_fp16
    )


if __name__ == "__main__":
    main()
