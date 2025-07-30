import os
import json
import datetime
from typing import List, Dict, Any, Optional
import mimetypes

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_FILES_DIR = os.path.join(BASE_DIR, "generated_files")

# Ensure directories exist
os.makedirs(GENERATED_FILES_DIR, exist_ok=True)

def get_file_info(filename: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific file"""
    file_path = os.path.join(GENERATED_FILES_DIR, filename)
    
    if not os.path.exists(file_path):
        return None
    
    file_size = os.path.getsize(file_path)
    file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
    mime_type, _ = mimetypes.guess_type(file_path)
    
    return {
        'filename': filename,
        'file_path': file_path,
        'file_size': file_size,
        'modified_at': file_modified.isoformat(),
        'mime_type': mime_type or 'application/octet-stream',
        'type': filename.split('.')[-1] if '.' in filename else 'unknown'
    }

def list_all_generated_files() -> List[Dict[str, Any]]:
    """List all generated files with their information"""
    files = []
    
    if not os.path.exists(GENERATED_FILES_DIR):
        return files
    
    for filename in os.listdir(GENERATED_FILES_DIR):
        file_path = os.path.join(GENERATED_FILES_DIR, filename)
        if os.path.isfile(file_path):
            file_info = get_file_info(filename)
            if file_info:
                files.append(file_info)
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x['modified_at'], reverse=True)
    return files

def cleanup_old_files(days_old: int = 7) -> Dict[str, Any]:
    """Clean up files older than specified days"""
    deleted_files = []
    current_time = datetime.datetime.now()
    cutoff_time = current_time - datetime.timedelta(days=days_old)
    
    if not os.path.exists(GENERATED_FILES_DIR):
        return {
            'status': 'success',
            'deleted_files': deleted_files,
            'count': 0,
            'cutoff_date': cutoff_time.isoformat()
        }
    
    for filename in os.listdir(GENERATED_FILES_DIR):
        file_path = os.path.join(GENERATED_FILES_DIR, filename)
        if os.path.isfile(file_path):
            file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
            if file_modified < cutoff_time:
                try:
                    os.remove(file_path)
                    deleted_files.append({
                        'filename': filename,
                        'modified_at': file_modified.isoformat()
                    })
                except Exception as e:
                    print(f"Error deleting {filename}: {e}")
    
    return {
        'status': 'success',
        'deleted_files': deleted_files,
        'count': len(deleted_files),
        'cutoff_date': cutoff_time.isoformat()
    }

def display_file_card(file_info: Dict[str, Any]) -> str:
    """Generate HTML card for file display (for Streamlit compatibility)"""
    filename = file_info.get('filename', 'Unknown')
    file_size = file_info.get('file_size', 0)
    file_type = file_info.get('type', 'unknown')
    
    # Format file size
    if file_size > 1024 * 1024:
        size_str = f"{file_size / (1024 * 1024):.1f} MB"
    elif file_size > 1024:
        size_str = f"{file_size / 1024:.1f} KB"
    else:
        size_str = f"{file_size} bytes"
    
    # File icon
    if file_type == 'pdf':
        icon = "📄"
    elif file_type in ['docx', 'doc']:
        icon = "📝"
    elif file_type in ['txt', 'md']:
        icon = "📋"
    else:
        icon = "📁"
    
    return f"{icon} **{filename}** ({size_str} • {file_type.upper()})"