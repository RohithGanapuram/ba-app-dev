import streamlit as st
import sqlite3
import re
import urllib.parse
import os

import asyncio
import nest_asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

nest_asyncio.apply()


import streamlit as st
import asyncio
import json
import os
import tempfile
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import nest_asyncio
import base64
import PyPDF2
import docx
from io import BytesIO
# Import file server utilities
from file_server import display_file_card, list_all_generated_files, get_file_info, cleanup_old_files


nest_asyncio.apply()
load_dotenv()

def safe_run_async(coro):
    """Safely run async coroutine with proper event loop handling"""
    try:
        # Try to get existing event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            # Create new event loop if none exists or is closed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(coro)
    except Exception as e:
        print(f"Error in async operation: {str(e)}")
        raise

def estimate_tokens(text: str) -> int:
    """Rough estimation of tokens in text (approximately 4 characters per token)"""
    return len(text) // 4

def truncate_content_for_api(content: str, max_tokens: int = 15000) -> str:
    """Truncate content to stay within token limits while preserving important information"""
    estimated_tokens = estimate_tokens(content)

    if estimated_tokens <= max_tokens:
        return content

    # Calculate how much content we can keep
    max_chars = max_tokens * 4

    # Try to keep the beginning and end of the document
    if len(content) > max_chars:
        keep_start = max_chars // 2
        keep_end = max_chars // 2

        truncated = (
            content[:keep_start] +
            f"\n\n[... CONTENT TRUNCATED - REMOVED {len(content) - max_chars:,} CHARACTERS TO STAY WITHIN API LIMITS ...]\n\n" +
            content[-keep_end:]
        )
        return truncated

    return content

def extract_generated_files_from_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Extract information about generated files from AI response text.
    Looks for JSON responses from file generation tools.
    """
    generated_files = []

    # Look for JSON patterns that indicate file generation
    json_pattern = r'\{[^{}]*"status":\s*"success"[^{}]*"filename":[^{}]*\}'
    matches = re.findall(json_pattern, response_text)

    for match in matches:
        try:
            file_info = json.loads(match)
            if file_info.get('status') == 'success' and 'filename' in file_info:
                generated_files.append(file_info)
        except json.JSONDecodeError:
            continue

    return generated_files

def clean_message_content(content: str) -> str:
    """
    Clean message content by removing JavaScript code and other unwanted elements.
    """
    # Temporarily disable cleaning to avoid errors - just return original content
    return content

def display_generated_files(files: List[Dict[str, Any]]) -> None:
    """
    Display generated files with download buttons.
    """
    if not files:
        return

    st.markdown("### 📁 Generated Files")

    for file_info in files:
        filename = file_info.get('filename', 'unknown')
        file_size = file_info.get('file_size', 0)
        file_type = file_info.get('type', 'unknown')

        # Format file size
        if file_size > 1024 * 1024:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        elif file_size > 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size} bytes"

        # Create a container for the file
        with st.container():
            col1, col2, col3 = st.columns([1, 4, 2])

            with col1:
                # File icon
                if file_type == 'pdf':
                    st.markdown("📄")
                elif file_type in ['docx', 'doc']:
                    st.markdown("📝")
                elif file_type in ['txt', 'md']:
                    st.markdown("📋")
                else:
                    st.markdown("📁")

            with col2:
                st.markdown(f"**{filename}**")
                st.caption(f"{size_str} • {file_type.upper()}")

            with col3:
                # Check if file actually exists and provide download
                actual_file_info = get_file_info(filename)
                if actual_file_info:
                    try:
                        with open(actual_file_info['file_path'], 'rb') as f:
                            file_data = f.read()

                        st.download_button(
                            label="📥 Download",
                            data=file_data,
                            file_name=filename,
                            mime=actual_file_info['mime_type'],
                            key=f"download_{filename}_{hash(str(file_info))}"
                        )
                    except Exception:
                        st.error("Error loading file")
                else:
                    st.error("File not found")

        st.markdown("---")

def process_uploaded_file(uploaded_file):
    """Process uploaded file and extract content with improved multi-page handling"""
    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        content = ""
        page_count = 0
        extraction_info = []

        if file_ext == "pdf":
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            total_pages = len(pdf_reader.pages)

            print(f"📄 Processing PDF: {uploaded_file.name} with {total_pages} pages")

            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        content += f"\n--- PAGE {page_num} ---\n"
                        content += page_text + "\n"
                        page_count += 1
                        extraction_info.append(f"Page {page_num}: {len(page_text.split())} words")
                    else:
                        extraction_info.append(f"Page {page_num}: Empty or unreadable")
                except Exception as page_error:
                    extraction_info.append(f"Page {page_num}: Error - {str(page_error)}")
                    continue

            print(f"✅ Extracted content from {page_count}/{total_pages} pages")

        elif file_ext == "txt":
            content = uploaded_file.read().decode("utf-8")
            page_count = 1
            extraction_info.append(f"Text file: {len(content.split())} words")

        elif file_ext == "docx":
            doc = docx.Document(BytesIO(uploaded_file.read()))
            paragraph_count = 0
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():  # Only add non-empty paragraphs
                    content += paragraph.text + "\n"
                    paragraph_count += 1
            page_count = 1
            extraction_info.append(f"DOCX file: {paragraph_count} paragraphs, {len(content.split())} words")

        else:
            content = "File type not supported for content extraction"
            extraction_info.append("Unsupported file type")

        # Enhanced return with more detailed information
        result = {
            'filename': uploaded_file.name,
            'size': uploaded_file.size,
            'content': content,
            'full_content': content,  # Keep full content for context
            'word_count': len(content.split()) if content else 0,
            'file_type': file_ext,
            'page_count': page_count,
            'extraction_info': extraction_info,
            'content_preview': content[:500] + "..." if len(content) > 500 else content
        }

        print(f"📊 File processed: {result['word_count']} words, {page_count} pages/sections")
        return result

    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            'filename': uploaded_file.name,
            'size': uploaded_file.size,
            'content': error_msg,
            'full_content': error_msg,
            'word_count': 0,
            'file_type': file_ext if 'file_ext' in locals() else 'unknown',
            'page_count': 0,
            'extraction_info': [error_msg],
            'content_preview': error_msg
        }

# Page configuration
st.set_page_config(
    page_title="BroadAxis-AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simple CSS for better layout
st.markdown("""
<style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Toolbar button styling - for all toolbar buttons */
    div[data-testid="column"]:nth-child(1) .stButton > button,
    div[data-testid="column"]:nth-child(2) .stButton > button,
    div[data-testid="column"]:nth-child(4) .stButton > button {
        width: 40px !important;
        height: 40px !important;
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        background-color: #ffffff !important;
        color: #6b7280 !important;
        font-size: 16px !important;
        padding: 0 !important;
        margin: 2px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-width: 40px !important;
        max-width: 40px !important;
    }

    /* Specific styling for toolbar file uploader */
    div[data-testid="column"]:nth-child(3) .stFileUploader {
        width: 40px !important;
        height: 40px !important;
        margin: 2px !important;
    }

    div[data-testid="column"]:nth-child(3) .stFileUploader button {
        width: 40px !important;
        height: 40px !important;
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        background-color: #ffffff !important;
        color: #6b7280 !important;
        font-size: 16px !important;
        padding: 0 !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    div[data-testid="column"]:nth-child(1) .stButton > button:hover,
    div[data-testid="column"]:nth-child(2) .stButton > button:hover,
    div[data-testid="column"]:nth-child(4) .stButton > button:hover {
        background-color: #f3f4f6 !important;
        border-color: #d1d5db !important;
    }

    /* Style the file uploader to look like a button */
    div[data-testid="column"]:nth-child(3) .stFileUploader > div > div > div > button {
        width: 40px !important;
        height: 40px !important;
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        background-color: #ffffff !important;
        color: #6b7280 !important;
        font-size: 16px !important;
        padding: 0 !important;
        margin: 2px !important;
    }

    /* Hide the file uploader text and show icon */
    div[data-testid="column"]:nth-child(3) .stFileUploader > div > div > div > button::before {
        content: "📎" !important;
        font-size: 16px !important;
    }

    div[data-testid="column"]:nth-child(3) .stFileUploader > div > div > div > button > span {
        display: none !important;
    }

    /* Fix button text wrapping */
    .stButton > button {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        word-wrap: normal !important;
        word-break: normal !important;
    }

    /* Left panel button styling */
    div[data-testid="column"]:first-child .stButton > button {
        width: 100% !important;
        height: auto !important;
        min-height: 36px !important;
        padding: 8px 12px !important;
        white-space: nowrap !important;
        font-size: 14px !important;
        text-align: left !important;
        justify-content: flex-start !important;
    }

    /* Primary button (New chat) styling */
    .stButton > button[kind="primary"] {
        background-color: #3b82f6 !important;
        color: white !important;
        border: 1px solid #3b82f6 !important;
        white-space: nowrap !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #2563eb !important;
        border-color: #2563eb !important;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        padding: 16px;
        margin: 8px 0;
        border-radius: 8px;
    }
    
    .assistant-message {
        background-color: #fafafa;
        border-left: 4px solid #6b7280;
        padding: 16px;
        margin: 8px 0;
        border-radius: 8px;
    }
    
    /* Connection status */
    .connection-status {
        padding: 8px 12px;
        border-radius: 6px;
        margin: 8px 0;
        text-align: center;
        font-weight: 500;
        font-size: 12px;
    }
    
    .connected {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    .disconnected {
        background-color: #fef2f2;
        color: #dc2626;
        border: 1px solid #fecaca;
    }

    /* Simple approach - limit chat messages height and make them scrollable */
    .chat-messages {
        max-height: 400px !important;
        overflow-y: auto !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 16px !important;
        background-color: #fafafa !important;
        margin: 16px 0 !important;
    }

    /* Ensure main container doesn't grow too much */
    .main .block-container {
        max-width: 100% !important;
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }

    /* Clean file uploader styling */
    .stFileUploader {
        margin: 10px 0 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 16px !important;
        background-color: #f9fafb !important;
    }

    /* Hide the drag and drop area text */
    .stFileUploader > div > div > div > div > span {
        display: none !important;
    }

    /* Hide the file type limit text */
    .stFileUploader > div > div > div > small {
        display: none !important;
    }

    /* Style the browse button nicely */
    .stFileUploader button {
        background-color: #3b82f6 !important;
        color: white !important;
        border: none !important;
        padding: 8px 16px !important;
        border-radius: 6px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }

    .stFileUploader button:hover {
        background-color: #2563eb !important;
    }

    /* Clean up the file uploader container */
    .stFileUploader > div > div {
        border: none !important;
        background: none !important;
    }

    /* Hide drag and drop instructions */
    .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }

    /* Make the file uploader more compact */
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        min-height: 60px !important;
        padding: 10px !important;
        text-align: center !important;
    }

    /* Model selector styling */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        font-size: 14px !important;
    }

    .stSelectbox > div > div:hover {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }

    /* Model selector label styling */
    .stSelectbox label {
        font-weight: 600 !important;
        color: #374151 !important;
        margin-bottom: 4px !important;
    }

    /* Fix toolbar button sizing */
    .stButton > button {
        width: 100% !important;
        height: 40px !important;
        min-height: 40px !important;
        padding: 8px !important;
        font-size: 16px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
    }

    /* Left column styling */
    div[data-testid="column"]:first-child {
        background-color: #f8fafc !important;
        border-right: 1px solid #e5e7eb !important;
        padding: 16px !important;
        min-height: 80vh !important;
    }

    /* Right column styling */
    div[data-testid="column"]:last-child {
        background-color: #f8fafc !important;
        border-left: 1px solid #e5e7eb !important;
        padding: 16px !important;
        min-height: 80vh !important;
    }

    /* Prevent text wrapping in left panel */
    div[data-testid="column"]:first-child * {
        word-wrap: normal !important;
        word-break: normal !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /* Allow wrapping for markdown headers */
    div[data-testid="column"]:first-child h1,
    div[data-testid="column"]:first-child h2,
    div[data-testid="column"]:first-child h3,
    div[data-testid="column"]:first-child strong {
        white-space: normal !important;
    }

    /* Copy button styling - simple and clean */
    .copy-btn {
        background: transparent !important;
        border: none !important;
        color: #6b7280 !important;
        font-size: 6px !important;
        padding: 4px !important;
        cursor: pointer !important;
        border-radius: 4px !important;
        transition: all 0.2s ease !important;
        width: 28px !important;
        height: 28px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    .copy-btn:hover {
        background-color: #f3f4f6 !important;
        color: #374151 !important;
    }
</style>
""", unsafe_allow_html=True)

class MCPStreamlitInterface:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.anthropic = Anthropic()
        self.available_tools: List[Dict] = []
        self.available_prompts: List[Dict] = []
        self.available_resources: List[Dict] = []
        self.is_connected = False

    async def connect_to_server(self):
        """Connect to the MCP server"""
        try:
            import os
            import sys

            # Get the current directory where server.py is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            server_path = os.path.join(current_dir, "server.py")

            # Check if server.py exists
            if not os.path.exists(server_path):
                st.error(f"❌ Server file not found at: {server_path}")
                return False

            server_params = StdioServerParameters(
                command=sys.executable,  # Use current Python interpreter
                args=[server_path],
                env=None,
                cwd=current_dir  # Set working directory
            )
            self.server_params = server_params
            
            # Test connection and get capabilities
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Get tools
                    tools_response = await session.list_tools()
                    self.available_tools = [{
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    } for tool in tools_response.tools]
                    
                    # Get prompts
                    prompts_response = await session.list_prompts()
                    self.available_prompts = [{
                        "name": prompt.name,
                        "description": prompt.description
                    } for prompt in prompts_response.prompts]
                    
                    # Get resources
                    resources_response = await session.list_resources()
                    self.available_resources = [{
                        "name": resource.name,
                        "description": resource.description,
                        "uri": resource.uri
                    } for resource in resources_response.resources]
                    
                    self.is_connected = True
                    return True
        except Exception as e:
            self.is_connected = False
            import traceback
            error_details = traceback.format_exc()
            st.error(f"Connection failed: {str(e)}")
            st.error(f"Error details: {error_details}")
            return False

    async def process_query(self, query: str, enabled_tools: List[str], model: str = None):
        """Process user query with selected tools and model"""
        if not self.is_connected:
            return "Please connect to the server first."

        # Use provided model or default
        selected_model = model or "claude-3-7-sonnet-20250219"

        # Filter tools based on enabled ones
        filtered_tools = [tool for tool in self.available_tools if tool["name"] in enabled_tools]

        # Comprehensive system prompt for BroadAxis RFP/RFQ platform
        system_prompt = """Hello! I'm BroadAxis-AI, your intelligent assistant designed to help you excel in RFP and RFQ management. I'm here to support you with professional, efficient, and data-driven assistance.

You are BroadAxis-AI, an intelligent assistant for BroadAxis - a leading vendor platform specializing in RFP (Request for Proposal) and RFQ (Request for Quotation) management and response generation.

## CRITICAL: TOOL USAGE HIERARCHY & PRIORITY

### PRIMARY RULE: Broadaxis_knowledge_search FIRST
You MUST use the Broadaxis_knowledge_search tool for ANY questions related to BroadAxis internal information, including but not limited to:
- Company information ("What does BroadAxis do?", "Tell me about the company")
- Team structure and personnel ("Who are our team members?", "What is our organizational structure?")
- Job roles and responsibilities ("What are the job responsibilities of network engineer?", "What does a project manager do here?")
- Company capabilities and expertise ("What are our technical capabilities?", "What projects have we worked on?")
- Internal processes and procedures ("How do we handle RFPs?", "What is our project methodology?")
- Company policies and guidelines
- Past projects and case studies
- Domain expertise and specializations

### SECONDARY RULE: web_search_tool Usage
The web_search_tool should ONLY be used when:
1. The Broadaxis_knowledge_search tool returns no relevant results
2. You need external market information, competitor analysis, or industry trends
3. You need current news, regulations, or external technical information
4. The query is explicitly about external entities or general knowledge NOT related to BroadAxis

### WORKFLOW FOR INFORMATION QUERIES:
1. FIRST: Always try Broadaxis_knowledge_search for any company-related query
2. ANALYZE: Review the results from the internal knowledge base
3. ONLY IF insufficient internal information: Then consider web_search_tool for external context
4. COMBINE: Present internal knowledge as primary source, external as supplementary if needed

## CRITICAL: RFP/RFQ DOCUMENT UPLOAD WORKFLOW

### AUTOMATIC DOCUMENT ANALYSIS WORKFLOW:
When a user uploads an RFP, RFQ, or RFI document, you MUST follow this exact sequence:

1. **DOCUMENT ANALYSIS**: Immediately analyze the uploaded document and provide:
   - Executive Summary
   - Key Requirements (bullet points)
   - Project Scope & Timeline
   - Budget Information (if available)
   - Evaluation Criteria
   - Submission Deadline
   - Key Risks & Opportunities

2. **MANDATORY GO/NO-GO OFFER**: After providing the summary, you MUST always ask:
   "📊 **Would you like my recommendation for a Go/No-Go decision on this opportunity?**"

3. **GO/NO-GO ANALYSIS** (if user says yes):
   - Use Broadaxis_knowledge_search to find relevant company experience
   - Assess capability alignment (provide percentage match)
   - Identify resource requirements
   - Evaluate competitive positioning
   - Provide clear GO or NO-GO recommendation with supporting evidence

## CRITICAL: NO HALLUCINATION POLICY FOR FORM FILLING

### STRICT INFORMATION SOURCING RULES:
When users upload documents with blank spaces, forms, or questionnaires to be filled:

**ONLY USE VERIFIED INFORMATION FROM:**
1. BroadAxis internal knowledge database (via Broadaxis_knowledge_search)
2. Information explicitly provided in uploaded documents
3. Previously shared context in the current conversation

**NEVER DO:**
- Hallucinate or invent company information
- Make up project details, dates, or capabilities
- Create fictional team member names or roles
- Invent financial information or company statistics
- Fill personal information fields with made-up data

**WHEN YOU DON'T KNOW:**
- Clearly state: "I don't have this information in our knowledge base"
- Say: "This field requires input from [specific person/department]"
- Suggest: "Please provide this information or consult with [relevant team]"
- For personal information: "Personal details must be provided by the individual"

**EXAMPLE RESPONSES FOR UNKNOWN INFORMATION:**
- "❌ I don't have information about our company's founding date. Please check with management."
- "❌ Personal contact information must be provided by the team member directly."
- "❌ I cannot find specific project budget details in our knowledge base. Please consult with the finance team."
- "✅ Based on our knowledge base, I can confirm that BroadAxis specializes in [specific area]."

## YOUR ROLE & MISSION
You help BroadAxis team members efficiently process, analyze, and respond to RFPs and RFQs by leveraging company knowledge, research capabilities, and document generation tools.

## CORE CAPABILITIES & WORKFLOW

### 1. RFP/RFQ ANALYSIS & SUMMARIZATION
When users upload RFP/RFQ documents, AUTOMATICALLY provide:
- Clear, structured summaries highlighting key requirements
- Project scope, timeline, budget constraints, and evaluation criteria
- Technical requirements and compliance needs
- Potential risks and opportunities
- **ALWAYS end with Go/No-Go recommendation offer**

### 2. GO/NO-GO DECISION SUPPORT
Use the Broadaxis_knowledge_search tool to:
- Cross-reference current RFP requirements with past company experience
- Identify similar projects BroadAxis has completed successfully
- Assess company capabilities against RFP requirements (provide % match)
- Provide data-driven go/no-go recommendations with supporting evidence
- Highlight capability gaps and suggest mitigation strategies

### 3. RESPONSE DOCUMENT GENERATION
Help create professional response documents:
- **Capability Statements**: Showcase relevant company experience and expertise
- **Technical Proposals**: Address specific technical requirements
- **Questionnaire Responses**: Answer RFP questions comprehensively (NO HALLUCINATION)
- **Executive Summaries**: Create compelling overviews for decision-makers

### 4. FORM FILLING & QUESTIONNAIRE COMPLETION
When filling forms or questionnaires:
- **ONLY use verified information** from knowledge base or provided documents
- **Mark unknown fields clearly** with "Information not available - requires input"
- **Never invent or hallucinate** company details, personal information, or project data
- **Suggest appropriate sources** for missing information

### 5. DOCUMENT GENERATION & FILE CREATION
You have powerful document generation tools available:

**PDF Generation** (generate_pdf_document):
- Create professional PDF documents with proper formatting
- Support for titles, headings, bullet points, and structured content
- Perfect for proposals, capability statements, and executive summaries
- Usage: generate_pdf_document(title="Document Title", content="Content here", filename="optional_name")

**Word Document Generation** (generate_word_document):
- Create Microsoft Word documents (.docx) with professional formatting
- Support for headings, bullet points, numbered lists
- Ideal for formal proposals and response documents
- Usage: generate_word_document(title="Document Title", content="Content here", filename="optional_name")

**Text File Generation** (generate_text_file):
- Create various text-based files (txt, md, csv, json, etc.)
- Flexible for different file formats and content types
- Usage: generate_text_file(content="Content here", filename="optional_name", file_extension="txt")

**File Management Tools**:
- list_generated_files: Show all available generated files
- cleanup_old_files: Remove old files to maintain organization

**When to Use Document Generation:**
- After completing RFP analysis, offer to create summary documents
- When users request formal proposals or capability statements
- For creating structured responses to RFQ requirements
- When generating executive summaries or technical documents

**Always offer file generation** when providing substantial content that users might want to save, share, or submit.

### 6. RESEARCH & INTELLIGENCE
Leverage available tools for competitive intelligence:
- Use web_search_tool for market research and competitor analysis
- Search academic papers for technical insights and best practices
- Gather industry trends and regulatory information

## AVAILABLE TOOLS & WHEN TO USE THEM

### TIER 1 - PRIMARY INTERNAL TOOL (Use First):
- **Broadaxis_knowledge_search**:
  * ALWAYS use FIRST for any BroadAxis-related queries
  * Contains: team structure, job responsibilities, company capabilities, past projects, internal processes
  * Use for: "What are the job responsibilities of network engineer?", "Who is on our team?", "What projects have we done?"

### TIER 2 - DOCUMENT GENERATION TOOLS (Use When Creating Deliverables):
- **generate_pdf_document**: Create professional PDF documents with formatting
- **generate_word_document**: Create Microsoft Word documents (.docx)
- **generate_text_file**: Create text files in various formats (txt, md, csv, json)
- **list_generated_files**: Show all available generated files
- **cleanup_old_files**: Clean up old generated files

### TIER 3 - EXTERNAL RESEARCH TOOLS (Use Only When Internal Search Insufficient):
- **web_search_tool**:
  * Use ONLY when Broadaxis_knowledge_search doesn't provide sufficient information
  * For: external market research, competitor analysis, industry trends, current news
  * NOT for internal company information that should be in our knowledge base

- **search_papers**: Find relevant academic research and technical papers for technical insights
- **extract_info**: Get detailed information about specific research papers

### Supporting Tools:
- **Document processing**: Analyze uploaded PDFs, Word docs, and other files
- **get_alerts/get_forecast**: Weather information for project planning
- **sum**: Basic calculations for budgets and timelines

## RESPONSE GUIDELINES

### For RFP/RFQ Analysis:
1. **Structure**: Use clear headings and bullet points
2. **Prioritize**: Highlight most critical requirements first
3. **Quantify**: Include specific numbers, dates, and metrics when available
4. **Risk Assessment**: Identify potential challenges and mitigation strategies

### For Go/No-Go Recommendations:
1. **Evidence-Based**: Always use Broadaxis_knowledge_search to support recommendations
2. **Scoring**: Provide capability match scores (e.g., "85% alignment with past experience")
3. **Specific Examples**: Reference specific past projects and outcomes
4. **Clear Recommendation**: End with explicit GO or NO-GO recommendation

### For Document Generation:
1. **Professional Tone**: Maintain formal, confident business language
2. **Customization**: Tailor content to specific RFP requirements
3. **Value Proposition**: Clearly articulate BroadAxis's unique advantages
4. **Compliance**: Ensure all RFP requirements are addressed

## INTERACTION STYLE
- Be proactive in suggesting next steps
- Ask clarifying questions when requirements are unclear
- Provide actionable insights, not just information
- Maintain confidentiality and professionalism
- Focus on helping BroadAxis win more business

## EXAMPLE WORKFLOWS

**RFP Upload → Analysis → Go/No-Go → Response Creation**
1. User uploads RFP PDF
2. You provide structured summary
3. You search company knowledge for relevant experience
4. You recommend go/no-go with evidence
5. If GO, you help create response documents

Remember: Your goal is to help BroadAxis make informed decisions quickly and create winning proposals efficiently."""

        messages = [{'role': 'user', 'content': query}]

        try:
            # Create a new session for each query with better error handling
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    response = self.anthropic.messages.create(
                        max_tokens=4024,
                        model=selected_model,
                        system=system_prompt,
                        tools=filtered_tools,
                        messages=messages
                    )

                    process_query = True
                    full_response = ""

                    while process_query:
                        assistant_content = []
                        for content in response.content:
                            if content.type == 'text':
                                full_response += content.text
                                assistant_content.append(content)
                                if len(response.content) == 1:
                                    process_query = False
                            elif content.type == 'tool_use':
                                assistant_content.append(content)
                                messages.append({'role': 'assistant', 'content': assistant_content})

                                tool_id = content.id
                                tool_args = content.input
                                tool_name = content.name

                                try:
                                    # Call the tool with better error handling
                                    result = await session.call_tool(tool_name, arguments=tool_args)
                                    messages.append({
                                        "role": "user",
                                        "content": [{
                                            "type": "tool_result",
                                            "tool_use_id": tool_id,
                                            "content": result.content
                                        }]
                                    })
                                except Exception as tool_error:
                                    # Handle tool errors gracefully
                                    error_message = f"Tool '{tool_name}' failed: {str(tool_error)}"
                                    messages.append({
                                        "role": "user",
                                        "content": [{
                                            "type": "tool_result",
                                            "tool_use_id": tool_id,
                                            "content": [{"type": "text", "text": error_message}]
                                        }]
                                    })

                                response = self.anthropic.messages.create(
                                    max_tokens=2024,
                                    model=selected_model,
                                    system=system_prompt,
                                    tools=filtered_tools,
                                    messages=messages
                                )

                                if len(response.content) == 1 and response.content[0].type == "text":
                                    full_response += response.content[0].text
                                    process_query = False

                    return full_response
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return f"Error processing query: {str(e)}\n\nDetails: {error_details}"

    async def get_prompt_content(self, prompt_name: str, arguments: dict = None):
        """Get the content of a prompt with arguments"""
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Call the prompt with arguments
                    result = await session.get_prompt(prompt_name, arguments or {})

                    # Handle different response types
                    if hasattr(result, 'messages') and result.messages:
                        # If it returns messages, combine them
                        content = ""
                        for msg in result.messages:
                            if hasattr(msg, 'content'):
                                if hasattr(msg.content, 'text'):
                                    content += msg.content.text + "\n"
                                else:
                                    content += str(msg.content) + "\n"
                        return content.strip()
                    else:
                        # If it returns a simple string
                        return str(result)
        except Exception as e:
            return f"Error getting prompt: {str(e)}"

    async def get_resource_content(self, resource_uri: str):
        """Get the content of a resource"""
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Read the resource
                    result = await session.read_resource(resource_uri)

                    # Handle different response types
                    if hasattr(result, 'contents') and result.contents:
                        content = ""
                        for item in result.contents:
                            if hasattr(item, 'text'):
                                content += item.text + "\n"
                            else:
                                content += str(item) + "\n"
                        return content.strip()
                    else:
                        return str(result)
        except Exception as e:
            return f"Error getting resource: {str(e)}"

    async def process_uploaded_file(self, uploaded_file):
        """Process uploaded file using MCP tools"""
        try:
            # Read file content and encode as base64
            file_content = uploaded_file.read()
            encoded_content = base64.b64encode(file_content).decode('utf-8')

            # Get file extension
            file_extension = uploaded_file.name.lower().split('.')[-1]

            # Create a new session to call the tool
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Use MCP upload_file tool
                    result = await session.call_tool(
                        "upload_file",
                        {
                            "filename": uploaded_file.name,
                            "content": encoded_content,
                            "file_type": file_extension
                        }
                    )

                    # Parse the JSON response
                    import json
                    file_info = json.loads(result.content[0].text)

                    return file_info

        except Exception as e:
            return {
                'status': 'error',
                'filename': uploaded_file.name,
                'size': uploaded_file.size,
                'content': f"Error processing file: {str(e)}",
                'word_count': 0,
                'char_count': 0
            }

# Initialize session state
if 'mcp_interface' not in st.session_state:
    st.session_state.mcp_interface = MCPStreamlitInterface()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Temporarily disable chat history cleaning to avoid errors
# if 'chat_cleaned' not in st.session_state:
#     # Clean all existing messages
#     for message in st.session_state.chat_history:
#         if message["role"] == "assistant":
#             message["content"] = clean_message_content(message["content"])
#     st.session_state.chat_cleaned = True

if 'enabled_tools' not in st.session_state:
    st.session_state.enabled_tools = set()

if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = []

if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None

if 'show_prompts_menu' not in st.session_state:
    st.session_state.show_prompts_menu = False

if 'show_resources_menu' not in st.session_state:
    st.session_state.show_resources_menu = False

if 'selected_prompt' not in st.session_state:
    st.session_state.selected_prompt = None

if 'prompt_inputs' not in st.session_state:
    st.session_state.prompt_inputs = {}

if 'selected_resource' not in st.session_state:
    st.session_state.selected_resource = None

if 'attached_resource' not in st.session_state:
    st.session_state.attached_resource = None

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'show_file_upload' not in st.session_state:
    st.session_state.show_file_upload = False

# Initialize model selection
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "claude-3-7-sonnet-20250219"  # Default model



# Main title
st.title("🤖BroadAxis-AI")

# Create three-column layout
left_col, main_col, right_col = st.columns([2, 6, 2], gap="medium")

# Left Panel - Chat Management
with left_col:
    st.markdown("### 🤖 HR-Services")
    
    # New Chat Button
    if st.button("➕ New chat", use_container_width=True, type="primary"):
        if st.session_state.chat_history:
            # Create a better title from the first message
            first_message = st.session_state.chat_history[0]["content"]
            # Remove newlines and extra spaces, then truncate
            clean_title = " ".join(first_message.split())
            chat_title = clean_title[:25] + "..." if len(clean_title) > 25 else clean_title
            st.session_state.chat_sessions.append({
                "id": len(st.session_state.chat_sessions),
                "title": chat_title,
                "history": st.session_state.chat_history.copy()
            })

        st.session_state.chat_history = []
        st.session_state.current_session_id = None
        st.rerun()
    
    st.markdown("**💬 Chats**")
    st.markdown("**🧩 Artifacts**")
    
    # Recent chats
    if st.session_state.chat_sessions:
        st.markdown("**Recents**")
        for session in reversed(st.session_state.chat_sessions[-5:]):  # Show last 5
            if st.button(session["title"], key=f"chat_{session['id']}", use_container_width=True):
                st.session_state.chat_history = session["history"].copy()
                st.session_state.current_session_id = session["id"]
                st.rerun()
    
    # Connection status
    st.markdown("---")
    if st.session_state.mcp_interface.is_connected:
        st.markdown('<div class="connection-status connected">✅ Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="connection-status disconnected">❌ Disconnected</div>', unsafe_allow_html=True)
        if st.button("Connect", use_container_width=True):
            with st.spinner("Connecting..."):
                try:
                    success = safe_run_async(st.session_state.mcp_interface.connect_to_server())
                    if success:
                        st.rerun()
                    else:
                        st.error("Failed to connect to server. Please check the server status.")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")

# Main Content Area
with main_col:
    # Toolbar
    st.markdown("### Chat Interface")
    
    # Toolbar buttons in a single row with proper sizing
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 8])

    with col1:
        if st.button("➕", key="prompts_btn", help="Prompts", use_container_width=True):
            st.session_state.show_prompts_menu = not st.session_state.show_prompts_menu
            st.session_state.show_resources_menu = False
            st.session_state.show_file_upload = False

    with col2:
        if st.button("📄", key="resources_btn", help="Resources", use_container_width=True):
            st.session_state.show_resources_menu = not st.session_state.show_resources_menu
            st.session_state.show_prompts_menu = False
            st.session_state.show_file_upload = False

    with col3:
        if st.button("📎", key="file_attach_btn", help="Attach files", use_container_width=True):
            st.session_state.show_file_upload = not st.session_state.show_file_upload
            st.session_state.show_prompts_menu = False
            st.session_state.show_resources_menu = False

    with col4:
        if st.button("⚙️", key="settings_btn", help="Settings", use_container_width=True):
            st.info("Settings panel")

    # Model Selection Dropdown
    st.markdown("---")

    # Create a container for the model selector
    model_container = st.container()
    with model_container:
        col_model_label, col_model_select, col_model_info = st.columns([2, 4, 1])

        with col_model_label:
            st.markdown("**🤖 Model:**")

        with col_model_select:
            # Available models (updated with your requested models)
            available_models = [
                "claude-opus-4-20250514",  # Default
                "claude-sonnet-4-20250514",      # Your requested model
                "claude-3-7-sonnet-20250219"

            ]

            # Model display names for better UX
            model_display_names = {
                "claude-3-7-sonnet-20250219": "🚀 Claude 3.7 Sonnet (Default)",
                "claude-opus-4-20250514": "⭐ Claude Opus 4 (Latest)",
                "claude-sonnet-4-20250514": "💎 Claude 4 Sonnet (Latest)"
            }

            # Find current selection index
            current_index = 0
            for i, model in enumerate(available_models):
                if model == st.session_state.selected_model:
                    current_index = i
                    break

            # Create selectbox with display names
            selected_display = st.selectbox(
                "Choose AI Model",
                options=[model_display_names[model] for model in available_models],
                index=current_index,
                key="model_selector",
                label_visibility="collapsed"
            )

            # Map back to actual model name
            for model, display in model_display_names.items():
                if display == selected_display:
                    st.session_state.selected_model = model
                    break

        with col_model_info:
            # Show a small indicator for the selected model
            if "opus-4" in st.session_state.selected_model:
                st.markdown("🔥")  # Latest model indicator
            elif "3.5-sonnet" in st.session_state.selected_model:
                st.markdown("✨")  # Default model indicator
            elif "haiku" in st.session_state.selected_model:
                st.markdown("⚡")  # Fast model indicator
            else:
                st.markdown("🤖")  # General AI indicator

    st.markdown("---")

    # Show uploaded files preview with enhanced information
    if st.session_state.uploaded_files:
        st.markdown("### 📎 Attached Files")

        # Show helpful tip for multiple files
        if len(st.session_state.uploaded_files) > 1:
            st.info(f"💡 **{len(st.session_state.uploaded_files)} files uploaded.** For best results with multiple documents, try asking: 'Can you generate summaries of each document?' or ask about specific files by name.")

        for i, file_info in enumerate(st.session_state.uploaded_files):
            # Enhanced title with page count if available
            page_info = ""
            if file_info.get('page_count', 0) > 1:
                page_info = f" | {file_info['page_count']} pages"
            elif file_info.get('page_count', 0) == 1:
                page_info = f" | {file_info['page_count']} page"

            title = f"📄 {file_info['filename']} ({file_info['word_count']} words{page_info})"

            with st.expander(title, expanded=False):
                # Show extraction information if available
                if file_info.get('extraction_info'):
                    st.markdown("**📊 Extraction Details:**")
                    for info in file_info['extraction_info']:
                        st.text(f"• {info}")
                    st.markdown("---")

                col1, col2 = st.columns([3, 1])
                with col1:
                    # Use content_preview if available, otherwise fallback to truncated content
                    preview_content = file_info.get('content_preview',
                                                  file_info['content'][:1000] + ("..." if len(file_info['content']) > 1000 else ""))

                    st.text_area(
                        "File Content Preview",
                        preview_content,
                        height=200,
                        disabled=True,
                        key=f"preview_{i}"
                    )

                    # Show full content length info
                    full_length = len(file_info.get('full_content', file_info['content']))
                    if full_length > 1000:
                        st.info(f"📏 Full document: {full_length:,} characters | Preview shows first 500 characters")

                with col2:
                    if st.button("🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.uploaded_files.pop(i)
                        if 'processed_files' in st.session_state:
                            st.session_state.processed_files = []
                        st.rerun()



    # Show dropdown menus for prompts and resources
    if st.session_state.show_prompts_menu and st.session_state.mcp_interface.available_prompts:
        st.markdown("**💬 Available Prompts:**")
        for prompt in st.session_state.mcp_interface.available_prompts:
            if st.button(f"💬 {prompt['name']}", key=f"prompt_{prompt['name']}", help=prompt.get('description', '')):
                st.session_state.selected_prompt = prompt
                st.session_state.show_prompts_menu = False
                st.rerun()

    if st.session_state.show_resources_menu and st.session_state.mcp_interface.available_resources:
        st.markdown("**📄 Available Resources:**")
        for resource in st.session_state.mcp_interface.available_resources:
            if st.button(f"📄 {resource['name']}", key=f"resource_{resource['name']}", help=resource.get('description', '')):
                st.session_state.selected_resource = resource
                st.session_state.show_resources_menu = False
                st.rerun()

    # Clean file upload interface when 📎 button is clicked
    if st.session_state.show_file_upload:
        st.markdown("---")

        # Create a clean container for file upload
        with st.container():
            st.markdown("### 📎 Upload Files")
            st.markdown("Select files from your computer to upload and analyze.")

            # Clean file uploader with minimal text
            uploaded_file = st.file_uploader(
                "Browse files",
                type=['pdf', 'docx', 'doc', 'txt', 'md', 'py', 'js', 'html', 'css', 'json', 'xml', 'csv'],
                key="clean_file_upload",
                label_visibility="collapsed"
            )

            # Handle file upload
            if uploaded_file:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    file_info = process_uploaded_file(uploaded_file)
                    st.session_state.uploaded_files.append(file_info)
                    st.success(f"✅ File uploaded: {uploaded_file.name}")
                    st.session_state.show_file_upload = False
                    st.rerun()

            # Close button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Close", key="close_upload_clean", use_container_width=True):
                    st.session_state.show_file_upload = False
                    st.rerun()


    # Handle prompt template input
    if st.session_state.selected_prompt:
        prompt = st.session_state.selected_prompt
        st.markdown("### Enter prompt inputs")
        st.markdown(f"**{prompt['name']}**")
        st.markdown(prompt.get('description', ''))

        # Check if this prompt needs inputs (based on common prompt patterns)
        needs_inputs = False
        input_fields = []

        # For analyze_actor prompt, we know it needs actor_name
        if prompt['name'] == 'analyze_actor':
            needs_inputs = True
            input_fields = [{'name': 'actor_name', 'label': 'Actor_name', 'required': True}]
        elif prompt['name'] == 'review_code':
            needs_inputs = True
            input_fields = [{'name': 'code', 'label': 'Code', 'required': True}]
        elif prompt['name'] == 'debug_error':
            needs_inputs = True
            input_fields = [{'name': 'error', 'label': 'Error', 'required': True}]

        if needs_inputs:
            # Create input fields
            inputs = {}
            for field in input_fields:
                if field['name'] == 'code':
                    inputs[field['name']] = st.text_area(
                        field['label'] + ('*' if field['required'] else ''),
                        key=f"input_{field['name']}"
                    )
                else:
                    inputs[field['name']] = st.text_input(
                        field['label'] + ('*' if field['required'] else ''),
                        key=f"input_{field['name']}"
                    )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Add prompt", type="primary"):
                    # Validate required fields
                    all_filled = all(inputs[field['name']].strip() for field in input_fields if field['required'])

                    if all_filled:
                        # Get the prompt content with inputs
                        with st.spinner("Processing prompt..."):
                            try:
                                prompt_content = safe_run_async(
                                    st.session_state.mcp_interface.get_prompt_content(prompt['name'], inputs)
                                )

                                # Process with AI directly (don't show the prompt template)
                                response = safe_run_async(
                                    st.session_state.mcp_interface.process_query(prompt_content, list(st.session_state.enabled_tools), st.session_state.selected_model)
                                )
                            except Exception as e:
                                response = f"Error processing prompt: {str(e)}"

                        # Add only a clean user message showing what was requested
                        user_message = f"Analyze actor: {inputs.get('actor_name', inputs.get('code', inputs.get('error', 'N/A')))}"
                        st.session_state.chat_history.append({"role": "user", "content": user_message})

                        # Add the AI response
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        st.session_state.selected_prompt = None
                        st.rerun()
                    else:
                        st.error("Please fill in all required fields")

            with col2:
                if st.button("Cancel"):
                    st.session_state.selected_prompt = None
                    st.rerun()
        else:
            # No inputs needed, just use the prompt directly
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Add prompt", type="primary"):
                    with st.spinner("Processing prompt..."):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        prompt_content = loop.run_until_complete(
                            st.session_state.mcp_interface.get_prompt_content(prompt['name'], {})
                        )

                        # Process with AI directly
                        response = loop.run_until_complete(
                            st.session_state.mcp_interface.process_query(prompt_content, list(st.session_state.enabled_tools), st.session_state.selected_model)
                        )

                    # Add clean user message
                    user_message = f"Used prompt: {prompt['name']}"
                    st.session_state.chat_history.append({"role": "user", "content": user_message})

                    # Add AI response
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.session_state.selected_prompt = None
                    st.rerun()

            with col2:
                if st.button("Cancel"):
                    st.session_state.selected_prompt = None
                    st.rerun()

    # Handle resource attachment
    elif st.session_state.selected_resource:
        resource = st.session_state.selected_resource
        st.markdown("### Attach Resource")
        st.markdown(f"**{resource['name']}**")
        st.markdown(resource.get('description', ''))

        with st.spinner("Loading resource..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            resource_content = loop.run_until_complete(
                st.session_state.mcp_interface.get_resource_content(resource['uri'])
            )

        # Show preview of resource content
        with st.expander("Resource Content Preview", expanded=True):
            st.text(resource_content[:500] + "..." if len(resource_content) > 500 else resource_content)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Attach to chat", type="primary"):
                # Add resource content to chat input area (we'll modify this)
                st.session_state.attached_resource = {
                    'name': resource['name'],
                    'content': resource_content
                }
                st.session_state.selected_resource = None
                st.success(f"Resource '{resource['name']}' attached!")
                st.rerun()

        with col2:
            if st.button("Cancel"):
                st.session_state.selected_resource = None
                st.rerun()

    # Clean file management interface
    elif st.session_state.show_file_upload:
        st.markdown("### 📎 Attached Documents")

        if st.session_state.uploaded_files:
            for i, file_info in enumerate(st.session_state.uploaded_files):
                # Simple card for each file
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        # File info
                        file_icon = "📄" if file_info.get('file_type', '').lower() == 'pdf' else "📝" if file_info.get('file_type', '').lower() in ['docx', 'doc'] else "📋"
                        st.markdown(f"**{file_icon} {file_info.get('filename', 'Unknown')}**")
                        st.caption(f"{file_info.get('word_count', 0):,} words • {file_info.get('size', 0):,} bytes • {file_info.get('file_type', 'Unknown').upper()}")

                    with col2:
                        if st.button("👁️ Preview", key=f"preview_file_{i}"):
                            st.session_state[f"show_preview_{i}"] = not st.session_state.get(f"show_preview_{i}", False)

                    with col3:
                        if st.button("🗑️ Remove", key=f"delete_file_{i}", type="secondary"):
                            st.session_state.uploaded_files.pop(i)
                            if 'processed_files' in st.session_state:
                                st.session_state.processed_files = []
                            st.rerun()

                    # Show preview if toggled
                    if st.session_state.get(f"show_preview_{i}", False):
                        with st.expander("📄 Content Preview", expanded=True):
                            content_preview = file_info.get('content', '')[:1000]
                            st.text_area("Preview", content_preview, height=200, disabled=True, label_visibility="collapsed")

                st.divider()
        else:
            st.info("📎 No files attached. Use the 📎 button in the chat input to attach files directly to your messages.")

    # Chat area
    st.markdown("---")
    
    # Display chat history in scrollable container
    st.markdown("### 💬 Chat Messages")

    # Create a container with fixed height for chat messages
    chat_container = st.container(height=400)
    with chat_container:
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    # Assistant message with simple copy functionality
                    col1, col2 = st.columns([20, 1])

                    with col1:
                        st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)

                    with col2:
                        # Simple copy button using streamlit's built-in functionality
                        copy_key = f"copy_btn_{i}"

                        # Use streamlit's download button as a copy mechanism
                        st.download_button(
                            label="📋",
                            data=message["content"],
                            file_name=f"response_{i}.txt",
                            mime="text/plain",
                            key=copy_key,
                            help="Download response as text file"
                        )

                    # Check if this assistant message contains generated files
                    generated_files = extract_generated_files_from_response(message["content"])
                    if generated_files:
                        st.markdown("---")
                        display_generated_files(generated_files)
        else:
            st.info("Start a conversation by typing a message below or using prompts/resources from the toolbar.")
    
    # Show attached resource if any
    if st.session_state.attached_resource:
        st.markdown("### 📎 Attached Resource")
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{st.session_state.attached_resource['name']}**")
                st.text(f"{len(st.session_state.attached_resource['content'])} characters")
            with col2:
                if st.button("❌", key="remove_attachment", help="Remove attachment"):
                    st.session_state.attached_resource = None
                    st.rerun()

    # Clean ChatGPT-style input area
    if st.session_state.mcp_interface.is_connected:
        # File attachment and chat input in a clean layout
        col1, col2 = st.columns([0.1, 0.9])

        with col1:
            # Simple file uploader
            chat_uploaded_file = st.file_uploader(
                "📎",
                type=['pdf', 'docx', 'doc', 'txt', 'md', 'py', 'js', 'html', 'css', 'json', 'xml', 'csv'],
                label_visibility="collapsed",
                key="chat_input_file_upload",
                help="Attach files: PDF, DOCX, DOC, TXT, MD, PY, JS, HTML, CSS, JSON, XML, CSV (Max 200MB)"
            )

        with col2:
            user_input = st.chat_input("Message Claude...")

        # Show attached file preview in a clean card
        if chat_uploaded_file:
            st.info(f"📎 **{chat_uploaded_file.name}** ({chat_uploaded_file.size:,} bytes) - Ready to process")

        # Handle file attachment first (like ChatGPT)
        if chat_uploaded_file and chat_uploaded_file not in st.session_state.get('processed_files', []):
            with st.spinner(f"Processing {chat_uploaded_file.name}..."):
                try:
                    file_info = safe_run_async(
                        st.session_state.mcp_interface.process_uploaded_file(chat_uploaded_file)
                    )

                    if file_info.get('status') == 'success':
                        # Add to uploaded files for context
                        st.session_state.uploaded_files.append(file_info)

                        # Track processed files to avoid reprocessing
                        if 'processed_files' not in st.session_state:
                            st.session_state.processed_files = []
                        st.session_state.processed_files.append(chat_uploaded_file)

                        # Add file attachment message to chat
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": f"📎 {chat_uploaded_file.name}"
                        })

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"I can see you've attached **{chat_uploaded_file.name}** ({file_info['word_count']} words). I've processed the document and can now answer questions about it. What would you like to know?"
                        })
                        st.rerun()
                    else:
                        st.error(f"Failed to process {chat_uploaded_file.name}: {file_info.get('content', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error processing file {chat_uploaded_file.name}: {str(e)}")
                    # Still track the file to avoid repeated processing attempts
                    if 'processed_files' not in st.session_state:
                        st.session_state.processed_files = []
                    st.session_state.processed_files.append(chat_uploaded_file)

        if user_input:
            # Prepare the query with context from uploaded files and attached resources
            final_input = user_input
            context_parts = []

            # Add uploaded files context with smart content management
            if st.session_state.uploaded_files:
                context_parts.append("=== DOCUMENT CONTEXT ===")

                # Check if user is asking for summaries of multiple documents
                is_summary_request = any(word in user_input.lower() for word in ['summary', 'summaries', 'summarize', 'summarise'])

                if is_summary_request and len(st.session_state.uploaded_files) > 1:
                    # For summary requests with multiple files, process each document separately
                    context_parts.append("MULTIPLE DOCUMENTS UPLOADED - PROCESSING FOR SUMMARIES:")

                    for i, file_info in enumerate(st.session_state.uploaded_files, 1):
                        filename = file_info.get('filename', file_info.get('name', 'Unknown'))
                        content = file_info.get('full_content', file_info.get('content', ''))
                        word_count = file_info.get('word_count', 0)
                        page_count = file_info.get('page_count', 0)

                        context_parts.append(f"\n--- DOCUMENT {i}: {filename} ---")
                        if page_count > 0:
                            context_parts.append(f"Stats: {word_count:,} words, {page_count} pages")
                        else:
                            context_parts.append(f"Stats: {word_count:,} words")

                        # Truncate content to manage token usage
                        truncated_content = truncate_content_for_api(content, max_tokens=8000)  # Smaller limit per document
                        context_parts.append(f"Content:\n{truncated_content}")
                        context_parts.append("--- END OF DOCUMENT ---")
                else:
                    # For single document or non-summary requests, use normal processing
                    for file_info in st.session_state.uploaded_files:
                        filename = file_info.get('filename', file_info.get('name', 'Unknown'))
                        content = file_info.get('full_content', file_info.get('content', ''))
                        word_count = file_info.get('word_count', 0)
                        page_count = file_info.get('page_count', 0)

                        context_parts.append(f"Document: {filename}")
                        if page_count > 0:
                            context_parts.append(f"Document Stats: {word_count:,} words, {page_count} pages")
                        else:
                            context_parts.append(f"Document Stats: {word_count:,} words")

                        # Use smart truncation for large documents
                        truncated_content = truncate_content_for_api(content, max_tokens=15000)
                        context_parts.append(f"Full Content:\n{truncated_content}")
                        context_parts.append("--- END OF DOCUMENT ---")

            # Add attached resource if any
            if st.session_state.attached_resource:
                context_parts.append("=== ATTACHED RESOURCE ===")
                context_parts.append(f"Resource: {st.session_state.attached_resource['name']}")
                context_parts.append(f"Content:\n{st.session_state.attached_resource['content']}")
                context_parts.append("---")
                st.session_state.attached_resource = None  # Clear after use

            # Combine context with user query
            if context_parts:
                final_input = "\n".join(context_parts) + f"\n\n=== USER QUESTION ===\n{user_input}"

            # Show clean message in chat (like ChatGPT)
            display_message = user_input
            if chat_uploaded_file:
                display_message = f"📎 {user_input}"

            st.session_state.chat_history.append({"role": "user", "content": display_message})

            with st.spinner("Processing..."):
                try:
                    response = safe_run_async(
                        st.session_state.mcp_interface.process_query(final_input, list(st.session_state.enabled_tools), st.session_state.selected_model)
                    )
                except Exception as e:
                    error_str = str(e)
                    if "rate_limit_error" in error_str or "429" in error_str:
                        response = """🚫 **Rate Limit Exceeded**

I apologize, but we've hit the API rate limit due to the large amount of content being processed. This typically happens when:

- Multiple large documents are uploaded at once
- Documents contain a lot of text that exceeds token limits

**What you can do:**
1. **Wait a moment** and try again (rate limits reset every minute)
2. **Process fewer documents** at once
3. **Ask more specific questions** about particular documents
4. **Upload smaller documents** or break large ones into sections

**For document summaries:** Try asking about one document at a time, like "Can you summarize the first document?" or "What are the key points in [filename]?"

Please try again in a minute or two, or ask about specific documents individually."""
                    else:
                        response = f"Error processing your request: {error_str}\n\nPlease try again or contact support if the issue persists."

            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    else:
        st.info("Connect to server to start chatting")

# Right Panel - Tools
with right_col:
    st.markdown("### Available Tools")
    
    if st.session_state.mcp_interface.available_tools:
        # Tool control options
        col1, col2 = st.columns(2)
        with col1:
            disable_all = st.checkbox("Disable all tools")
        with col2:
            enable_all = st.checkbox("Enable all tools")

        # Handle enable/disable all logic
        if disable_all:
            st.session_state.enabled_tools = set()
        elif enable_all:
            st.session_state.enabled_tools = set(tool["name"] for tool in st.session_state.mcp_interface.available_tools)

        st.markdown("---")

        for tool in st.session_state.mcp_interface.available_tools:
            tool_name = tool["name"]
            is_enabled = st.checkbox(
                tool_name,
                value=tool_name in st.session_state.enabled_tools and not disable_all,
                key=f"tool_{tool_name}",
                help=tool.get("description", "No description"),
                disabled=disable_all or enable_all
            )

            if is_enabled and not disable_all and not enable_all:
                st.session_state.enabled_tools.add(tool_name)
            elif not is_enabled and not enable_all:
                st.session_state.enabled_tools.discard(tool_name)
    else:
        st.info("Connect to server to see available tools")

    # Generated Files Section
    st.markdown("---")
    st.markdown("### 📁 Generated Files")

    # Get list of generated files
    generated_files = list_all_generated_files()

    if generated_files:
        st.markdown(f"**{len(generated_files)} file(s) available**")

        # Show recent files (limit to 5 for space)
        for file_info in generated_files[:5]:
            filename = file_info['filename']
            file_size = file_info['file_size']
            file_type = file_info['type']

            # Format file size
            if file_size > 1024 * 1024:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
            elif file_size > 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size} bytes"

            with st.container():
                col1, col2 = st.columns([3, 1])

                with col1:
                    # File icon and name
                    if file_type == 'pdf':
                        icon = "📄"
                    elif file_type in ['docx', 'doc']:
                        icon = "📝"
                    elif file_type in ['txt', 'md']:
                        icon = "📋"
                    else:
                        icon = "📁"

                    st.markdown(f"{icon} **{filename}**")
                    st.caption(f"{size_str} • {file_type.upper()}")

                with col2:
                    # Download button
                    try:
                        with open(file_info['file_path'], 'rb') as f:
                            file_data = f.read()

                        st.download_button(
                            label="📥",
                            data=file_data,
                            file_name=filename,
                            mime=file_info['mime_type'],
                            key=f"sidebar_download_{filename}_{hash(file_info['modified_at'])}",
                            help=f"Download {filename}"
                        )
                    except Exception as e:
                        st.error("❌")

            st.markdown("---")

        # Show "View All" button if there are more files
        if len(generated_files) > 5:
            if st.button(f"📂 View All {len(generated_files)} Files", use_container_width=True):
                st.session_state.show_all_files = True
                st.rerun()
    else:
        st.info("No files generated yet. Ask me to create a document!")

    # Cleanup button
    if generated_files:
        st.markdown("---")
        if st.button("🗑️ Cleanup Old Files", use_container_width=True, help="Remove files older than 7 days"):
            with st.spinner("Cleaning up old files..."):
                try:
                    cleanup_result = cleanup_old_files(7)
                    count = cleanup_result.get('count', 0)
                    if count > 0:
                        st.success(f"✅ Cleaned up {count} old files")
                    else:
                        st.info("ℹ️ No old files to clean up")
                except Exception as e:
                    st.error(f"❌ Cleanup failed: {str(e)}")
                st.rerun()
