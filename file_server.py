"""
File serving utilities for generated documents.
Provides functionality to serve generated files through Streamlit.
"""

import os
import mimetypes
from pathlib import Path
import streamlit as st
from typing import Optional, Dict, Any
import json
import datetime

# Get the base directory and generated files directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_FILES_DIR = os.path.join(BASE_DIR, "generated_files")

def get_file_info(filename: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a generated file.
    
    Args:
        filename: Name of the file
        
    Returns:
        Dictionary with file information or None if file doesn't exist
    """
    file_path = os.path.join(GENERATED_FILES_DIR, filename)
    
    if not os.path.exists(file_path):
        return None
    
    file_size = os.path.getsize(file_path)
    file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
    
    # Get MIME type
    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type:
        mime_type = 'application/octet-stream'
    
    return {
        "filename": filename,
        "file_path": file_path,
        "file_size": file_size,
        "mime_type": mime_type,
        "modified_at": file_modified.isoformat(),
        "type": filename.split('.')[-1] if '.' in filename else "unknown"
    }

def serve_file_download(filename: str) -> bool:
    """
    Serve a file for download through Streamlit.
    
    Args:
        filename: Name of the file to serve
        
    Returns:
        True if file was served successfully, False otherwise
    """
    file_info = get_file_info(filename)
    
    if not file_info:
        st.error(f"File '{filename}' not found.")
        return False
    
    try:
        # Read file content
        with open(file_info['file_path'], 'rb') as f:
            file_data = f.read()
        
        # Create download button
        st.download_button(
            label=f"📥 Download {filename}",
            data=file_data,
            file_name=filename,
            mime=file_info['mime_type'],
            key=f"download_{filename}_{file_info['modified_at']}"
        )
        
        return True
        
    except Exception as e:
        st.error(f"Error serving file: {str(e)}")
        return False

def create_download_link(filename: str, label: str = None) -> str:
    """
    Create a download link for a generated file.
    
    Args:
        filename: Name of the file
        label: Optional custom label for the link
        
    Returns:
        HTML string for the download link
    """
    if not label:
        label = f"Download {filename}"
    
    file_info = get_file_info(filename)
    if not file_info:
        return f"<span style='color: red;'>File '{filename}' not found</span>"
    
    # Format file size
    size_mb = file_info['file_size'] / (1024 * 1024)
    if size_mb < 1:
        size_str = f"{file_info['file_size'] / 1024:.1f} KB"
    else:
        size_str = f"{size_mb:.1f} MB"
    
    return f"""
    <div style="
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        background-color: #f8f9fa;
        display: flex;
        align-items: center;
        justify-content: space-between;
    ">
        <div style="display: flex; align-items: center;">
            <span style="font-size: 20px; margin-right: 8px;">📄</span>
            <div>
                <div style="font-weight: 500; color: #1f2937;">{filename}</div>
                <div style="font-size: 12px; color: #6b7280;">{size_str} • {file_info['type'].upper()}</div>
            </div>
        </div>
        <div style="margin-left: 12px;">
            <button onclick="downloadFile('{filename}')" style="
                background-color: #3b82f6;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
            ">📥 Download</button>
        </div>
    </div>
    """

def display_file_card(filename: str, show_download: bool = True) -> None:
    """
    Display a file card with download option in Streamlit.
    
    Args:
        filename: Name of the file
        show_download: Whether to show download button
    """
    file_info = get_file_info(filename)
    
    if not file_info:
        st.error(f"File '{filename}' not found.")
        return
    
    # Format file size
    size_mb = file_info['file_size'] / (1024 * 1024)
    if size_mb < 1:
        size_str = f"{file_info['file_size'] / 1024:.1f} KB"
    else:
        size_str = f"{size_mb:.1f} MB"
    
    # Create file card
    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            # File icon based on type
            if file_info['type'] == 'pdf':
                st.markdown("📄", unsafe_allow_html=True)
            elif file_info['type'] in ['docx', 'doc']:
                st.markdown("📝", unsafe_allow_html=True)
            elif file_info['type'] in ['txt', 'md']:
                st.markdown("📋", unsafe_allow_html=True)
            else:
                st.markdown("📁", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**{filename}**")
            st.caption(f"{size_str} • {file_info['type'].upper()} • Modified: {file_info['modified_at'][:19]}")
        
        with col3:
            if show_download:
                serve_file_download(filename)

def list_all_generated_files() -> list:
    """
    List all generated files in the directory.
    
    Returns:
        List of file information dictionaries
    """
    files = []
    
    if os.path.exists(GENERATED_FILES_DIR):
        for filename in os.listdir(GENERATED_FILES_DIR):
            file_info = get_file_info(filename)
            if file_info:
                files.append(file_info)
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x['modified_at'], reverse=True)
    
    return files

def cleanup_old_files(days_old: int = 7) -> Dict[str, Any]:
    """
    Clean up files older than specified days.
    
    Args:
        days_old: Number of days old files should be to be deleted
        
    Returns:
        Dictionary with cleanup results
    """
    deleted_files = []
    current_time = datetime.datetime.now()
    cutoff_time = current_time - datetime.timedelta(days=days_old)
    
    if os.path.exists(GENERATED_FILES_DIR):
        for filename in os.listdir(GENERATED_FILES_DIR):
            file_path = os.path.join(GENERATED_FILES_DIR, filename)
            if os.path.isfile(file_path):
                file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_modified < cutoff_time:
                    try:
                        os.remove(file_path)
                        deleted_files.append({
                            "filename": filename,
                            "modified_at": file_modified.isoformat()
                        })
                    except Exception as e:
                        print(f"Error deleting file {filename}: {e}")
    
    return {
        "deleted_files": deleted_files,
        "count": len(deleted_files),
        "cutoff_date": cutoff_time.isoformat()
    }
