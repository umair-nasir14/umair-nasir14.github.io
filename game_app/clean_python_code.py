#!/usr/bin/env python3
"""
Script to clean Python files by keeping only:
- Import statements
- Class definitions
- Function definitions
- Removes all module-level executable code
"""

import ast
import argparse
import os
import sys
from typing import List, Union

def is_git_bash():
    """Detect if running in Git Bash environment"""
    return os.environ.get('MSYSTEM') is not None or 'MINGW' in os.environ.get('MSYSTEM', '')


class CodeCleaner(ast.NodeVisitor):
    """AST visitor that extracts only imports, classes, and functions"""
    
    def __init__(self):
        self.kept_nodes = []
    
    def visit_Import(self, node: ast.Import) -> None:
        """Keep import statements"""
        self.kept_nodes.append(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Keep from...import statements"""
        self.kept_nodes.append(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Keep class definitions"""
        self.kept_nodes.append(node)
        # Don't visit children - we want the entire class as-is
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Keep function definitions"""
        self.kept_nodes.append(node)
        # Don't visit children - we want the entire function as-is
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Keep async function definitions"""
        self.kept_nodes.append(node)
        # Don't visit children - we want the entire function as-is


def clean_python_file(file_path: str, output_path: str = None, create_backup: bool = False, path_replacement: tuple = None) -> str:
    """
    Clean a Python file by keeping only imports, classes, and functions.
    
    Args:
        file_path: Path to the input Python file
        output_path: Path for the output file (if None, returns cleaned code as string)
        create_backup: If True and output_path equals file_path, create a backup first
        path_replacement: Tuple of (old_path, new_path) for string replacement
    
    Returns:
        Cleaned Python code as string
    """
    
    # Read the original file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_code = f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None
    
    # Parse the AST
    try:
        tree = ast.parse(original_code)
    except SyntaxError as e:
        print(f"Syntax error in '{file_path}': {e}")
        return None
    
    # Extract only the nodes we want to keep
    cleaner = CodeCleaner()
    cleaner.visit(tree)
    
    if not cleaner.kept_nodes:
        print(f"Warning: No imports, classes, or functions found in '{file_path}'")
        return ""
    
    # Create a new AST with only the kept nodes
    new_tree = ast.Module(body=cleaner.kept_nodes, type_ignores=[])
    
    # Convert back to Python code
    try:
        import astor
        cleaned_code = astor.to_source(new_tree)
    except ImportError:
        # Fallback: use ast.unparse (Python 3.9+)
        try:
            cleaned_code = ast.unparse(new_tree)
        except AttributeError:
            print("Error: This script requires either 'astor' package or Python 3.9+")
            print("Install astor with: pip install astor")
            return None
    
    # Apply path replacement if specified
    if path_replacement:
        old_path, new_path = path_replacement
        
        # Auto-detect Git Bash and handle path conversion
        git_bash_converted_path = None
        if is_git_bash() and old_path.startswith('/') and not old_path.startswith('//'):
            # In Git Bash, check if path was auto-converted
            if 'Program Files/Git' in old_path:
                # Path is already converted by Git Bash, use as-is
                pass
            else:
                # Predict what Git Bash would convert the path to
                possible_converted_backslash = f"C:/Program Files/Git{old_path}".replace('/', '\\')
                possible_converted_forward = f"C:/Program Files/Git{old_path}"
                
                if possible_converted_backslash in cleaned_code:
                    git_bash_converted_path = possible_converted_backslash
                elif possible_converted_forward in cleaned_code:
                    git_bash_converted_path = possible_converted_forward
        elif old_path.startswith('//'):
            # Double slash - remove one slash to get the actual path
            actual_path = old_path[1:]  # Remove first slash
            old_path = actual_path
        
        # Try to replace the original path first
        replacement_made = False
        original_code = cleaned_code
        
        # Replace both quoted versions of the path
        cleaned_code = cleaned_code.replace(f"'{old_path}'", f"r'{new_path}'")
        cleaned_code = cleaned_code.replace(f'"{old_path}"', f'r"{new_path}"')
        # Also replace unquoted versions (though less common)
        cleaned_code = cleaned_code.replace(old_path, new_path)
        
        if cleaned_code != original_code:
            replacement_made = True
            print(f"Replaced path: '{old_path}' -> r'{new_path}'")
        
        # If no replacement made and we detected Git Bash conversion, try that
        if not replacement_made and git_bash_converted_path:
            cleaned_code = cleaned_code.replace(f"'{git_bash_converted_path}'", f"r'{new_path}'")
            cleaned_code = cleaned_code.replace(f'"{git_bash_converted_path}"', f'r"{new_path}"')
            cleaned_code = cleaned_code.replace(git_bash_converted_path, new_path)
            print(f"Replaced Git Bash converted path: '{git_bash_converted_path}' -> r'{new_path}'")
            replacement_made = True
        
        if not replacement_made:
            print(f"Warning: Path '{old_path}' not found in code. No replacement made.")
    
    # Write to output file if specified
    if output_path:
        try:
            # Create backup if requested and we're overwriting the original file
            if create_backup and output_path == file_path:
                import shutil
                backup_path = f"{file_path}.bak"
                shutil.copy2(file_path, backup_path)
                print(f"Backup created: {backup_path}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_code)
            print(f"Cleaned code written to '{output_path}'")
        except Exception as e:
            print(f"Error writing to '{output_path}': {e}")
            return None
    
    return cleaned_code


def clean_directory(directory_path: str, output_directory: str = None, pattern: str = "*.py", in_place: bool = False, create_backup: bool = False, path_replacement: tuple = None):
    """
    Clean all Python files in a directory.
    
    Args:
        directory_path: Input directory path
        output_directory: Output directory path (if None, creates 'cleaned' subdirectory)
        pattern: File pattern to match (default: "*.py")
        in_place: If True, overwrite original files instead of creating new ones
        create_backup: If True, create backup files when using in_place
        path_replacement: Tuple of (old_path, new_path) for string replacement
    """
    import glob
    
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return
    
    # Set up output directory (only if not editing in-place)
    if not in_place:
        if output_directory is None:
            output_directory = os.path.join(directory_path, "cleaned")
        os.makedirs(output_directory, exist_ok=True)
    
    # Find all Python files
    search_pattern = os.path.join(directory_path, pattern)
    python_files = glob.glob(search_pattern)
    
    if not python_files:
        print(f"No Python files found matching pattern '{pattern}' in '{directory_path}'")
        return
    
    print(f"Found {len(python_files)} Python files to clean...")
    
    for file_path in python_files:
        filename = os.path.basename(file_path)
        
        if in_place:
            output_path = file_path  # Overwrite the original file
            print(f"Cleaning {filename} (in-place)...")
        else:
            output_path = os.path.join(output_directory, filename)
            print(f"Cleaning {filename}...")
        
        cleaned_code = clean_python_file(file_path, output_path, create_backup and in_place, path_replacement)
        
        if cleaned_code is None:
            print(f"Failed to clean {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean Python files by keeping only imports, classes, and functions"
    )
    parser.add_argument(
        "input", 
        help="Input Python file or directory"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file or directory (default: adds '_cleaned' suffix or 'cleaned' subdirectory)"
    )
    parser.add_argument(
        "-i", "--in-place",
        action="store_true",
        help="Edit files in-place (overwrite original files with cleaned code)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup files (adds .bak extension) when using --in-place"
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompts for in-place editing"
    )
    parser.add_argument(
        "--replace-path",
        nargs=2,
        metavar=("OLD_PATH", "NEW_PATH"),
        help="Replace path strings in the code (e.g. --replace-path '/old/path' 'C:\\new\\path')"
    )
    parser.add_argument(
        "-d", "--directory",
        action="store_true",
        help="Process entire directory instead of single file"
    )
    parser.add_argument(
        "--pattern",
        default="*.py",
        help="File pattern for directory processing (default: *.py)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview cleaned code without writing to file"
    )
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.in_place and args.output:
        print("Error: --in-place and --output cannot be used together")
        sys.exit(1)
    
    if args.in_place and args.preview:
        print("Error: --in-place and --preview cannot be used together")
        sys.exit(1)
        
    if args.backup and not args.in_place:
        print("Error: --backup can only be used with --in-place")
        sys.exit(1)
    
    # Confirmation for in-place editing
    if args.in_place and not args.yes:
        if args.directory:
            response = input(f"This will overwrite all Python files in '{args.input}'. Continue? (y/N): ")
        else:
            response = input(f"This will overwrite '{args.input}'. Continue? (y/N): ")
        
        if response.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)
    
    # Prepare path replacement tuple
    path_replacement = tuple(args.replace_path) if args.replace_path else None
    
    if args.directory:
        clean_directory(args.input, args.output, args.pattern, args.in_place, args.backup, path_replacement)
    else:
        # Single file processing
        if args.preview:
            cleaned_code = clean_python_file(args.input, None, False, path_replacement)
            if cleaned_code:
                print("=" * 50)
                print("CLEANED CODE:")
                print("=" * 50)
                print(cleaned_code)
        else:
            if args.in_place:
                output_path = args.input  # Overwrite the original file
                print(f"Cleaning {args.input} in-place...")
            else:
                output_path = args.output
                if output_path is None:
                    base, ext = os.path.splitext(args.input)
                    output_path = f"{base}_cleaned{ext}"
            
            clean_python_file(args.input, output_path, args.backup and args.in_place, path_replacement)


if __name__ == "__main__":
    main() 