# Story Maker Agent Crew FastAPI Api

An intelligent multi-agent system that creates engaging children's storybooks with AI-generated illustrations and automatic PDF conversion. Built with [CrewAI](https://crewai.com), this project leverages multiple specialized AI agents to collaborate on story creation, illustration generation, and formatting.

## 🚀 Project Overview

This project implements a sophisticated AI-powered storybook creation system featuring:
- **Multi-Agent Collaboration** using CrewAI framework
- **Automated Story Generation** with structured 5-chapter narratives
- **AI-Generated Illustrations** using DALL-E 3 for each chapter
- **Professional PDF Output** with embedded images and formatting
- **RESTful API Interface** for easy integration and usage

## ✨ Features

- **Intelligent Story Outlining** with character development and plot structure
- **Contextual Story Writing** (100 words per chapter, 5 chapters total)
- **Custom Illustration Generation** tailored to each chapter's content
- **Professional Formatting** with markdown-to-PDF conversion
- **Asynchronous Processing** with task status tracking
- **Multiple LLM Support** (OpenAI GPT-4, Groq models)
- **Configurable Agent Behavior** through YAML configuration files

## 📋 Requirements

### System Requirements
- Python 3.10+ (< 3.13)
- 4GB+ RAM recommended
- Internet connection for AI model APIs
- 2GB+ storage space for generated content

### API Dependencies
- OpenAI API key (for GPT-4 and DALL-E 3)
- Groq API key (optional, for alternative models)

## 🛠️ Setup Instructions

### Method 1: Using UV (Recommended)

UV provides faster dependency resolution and better package management.

#### 1. Install UV
```bash
pip install uv
```

#### 2. Clone and Setup Project
```bash
git clone https://github.com/yourusername/story-maker-agent.git
cd story-maker-agent
```

#### 3. Install Dependencies with UV
```bash
# Install all dependencies
uv pip install -r requirements.txt

# Or use CrewAI CLI (if available)
crewai install
```

#### 4. Configure Environment Variables
Create a `.env` file in the root directory:

```env
# API Configuration
API_V1_STR=/api/v1
PROJECT_NAME=Story Maker Agent API

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4-turbo

# Groq Configuration (Optional)
GROQ_API_KEY=gsk-your-groq-api-key-here
GROQ_MODEL=mixtral-8x7b-32768

# Model Configuration
MODEL=gpt-4-turbo
```

### Method 2: Manual Setup

#### 1. Clone Repository
```bash
git clone https://github.com/yourusername/story-maker-agent.git
cd story-maker-agent
```

#### 2. Create Virtual Environment

**On Windows:**
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Install Additional Requirements
```bash
# Install mdpdf for PDF conversion
npm install -g mdpdf

# Or using pip (if available)
pip install mdpdf
```

#### 5. Configure Environment
Create and configure your `.env` file as shown in Method 1.

## 🎯 How to Use

### Starting the API Server

#### Development Mode (with auto-reload)
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Production Mode
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Using the API

#### 1. Generate a New Story
```bash
curl -X POST http://localhost:8000/api/v1/topic \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "A brave little mouse on an adventure"
  }'
```

**Response:**
```json
{
  "task_id": "c1a2b3d4-5678-90ab-cdef-1234567890ab",
  "status": "pending",
  "result": null,
  "error": null
}
```

#### 2. Check Task Status
```bash
curl http://localhost:8000/api/v1/task/c1a2b3d4-5678-90ab-cdef-1234567890ab
```

**Response (Processing):**
```json
{
  "task_id": "c1a2b3d4-5678-90ab-cdef-1234567890ab",
  "status": "processing",
  "result": null,
  "error": null
}
```

**Response (Completed):**
```json
{
  "task_id": "c1a2b3d4-5678-90ab-cdef-1234567890ab",
  "status": "completed",
  "result": {
    "story_title": "The Brave Little Mouse Adventure",
    "pdf_path": "app/resulted_stories/story.pdf",
    "markdown_path": "app/story_intermediary/story.md",
    "images_folder": "app/images/the_brave_little_mouse_adventure/",
    "images": [
      "app/images/the_brave_little_mouse_adventure/chapter_1.png",
      "app/images/the_brave_little_mouse_adventure/chapter_2.png",
      "app/images/the_brave_little_mouse_adventure/chapter_3.png",
      "app/images/the_brave_little_mouse_adventure/chapter_4.png",
      "app/images/the_brave_little_mouse_adventure/chapter_5.png"
    ]
  },
  "error": null
}
```
**Response (Failed)**

```json
{
  "task_id": "c1a2b3d4-5678-90ab-cdef-1234567890ab",
  "status": "failed",
  "result": null,
  "error": "Description of what went wrong"
}
```

**Response (Task not found)**
```json
{
  "detail": "Task not found"
}
```


#### 3. Access Generated Content
The generated files are stored locally in the application directories:

**PDF Location:**
```bash
# Final PDF story
app/resulted_stories/story.pdf
```

**Images Location:**
```bash
# Chapter images (organized by story topic)
app/images/{story_topic_slug}/
├── chapter_1.png
├── chapter_2.png
├── chapter_3.png
├── chapter_4.png
└── chapter_5.png
```

**Markdown Location:**
```bash
# Intermediate markdown file
app/story_intermediary/story.md
```

**Example for a story about "A Brave Little Mouse":**
```bash
# PDF
app/resulted_stories/story.pdf

# Images
app/images/a_brave_little_mouse_adventure/
├── chapter_1.png
├── chapter_2.png
├── chapter_3.png
├── chapter_4.png
└── chapter_5.png

# Markdown
app/story_intermediary/story.md
```

## 📁 File Organization

### Generated Content Structure
All generated content is stored locally within the application directory:

```
app/
├── images/                       # Generated story images
│   └── {story_topic_slug}/      # Folder named after story topic
│       ├── chapter_1.png        # Chapter 1 illustration
│       ├── chapter_2.png        # Chapter 2 illustration
│       ├── chapter_3.png        # Chapter 3 illustration
│       ├── chapter_4.png        # Chapter 4 illustration
│       └── chapter_5.png        # Chapter 5 illustration
├── story_intermediary/          # Temporary processing files
│   └── story.md                 # Generated markdown content
└── resulted_stories/            # Final PDF outputs
    └── story.pdf               # Complete storybook PDF
```

## 📁 Project Structure
The story topic gets converted to a folder-safe slug:
- **Original**: "A Brave Little Mouse Adventure"
- **Slug**: `a_brave_little_mouse_adventure`
- **Folder**: `app/images/a_brave_little_mouse_adventure/`

### Accessing Your Generated Stories
After generation completes, you can find your files at:

1. **PDF Storybook**: `app/resulted_stories/story.pdf`
2. **Individual Images**: `app/images/{your_story_slug}/chapter_X.png`
3. **Markdown Source**: `app/story_intermediary/story.md`

```
story-maker-agent/
├── .git/                          # Git repository
├── .gitignore                     # Git ignore file
├── .python-version                # Python version specification
├── README.md                      # Project documentation
├── pyproject.toml                 # Project configuration (UV/Poetry)
├── requirements.txt               # Python dependencies
├── uv.lock                        # UV lock file
├── .env                          # Environment variables (create this)
├── .venv/                        # Virtual environment
├── app/                          # Main application directory
│   ├── __init__.py
│   ├── main.py                   # FastAPI application entry point
│   ├── api/                      # API routes and endpoints
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           └── analyzer.py   # Story generation endpoints
│   ├── core/                     # Core application logic
│   │   ├── __init__.py
│   │   └── config.py            # Configuration management
│   ├── crew/                     # CrewAI agents and tasks
│   │   ├── config/
│   │   │   ├── agents.yaml      # Agent definitions
│   │   │   └── tasks.yaml       # Task definitions
│   │   ├── knowledge/
│   │   │   └── template.md      # Markdown template
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   └── custom_tool.py   # Custom tools (image gen, PDF)
│   │   └── crew.py              # Crew orchestration
│   ├── models/                   # Pydantic models
│   │   ├── __init__.py
│   │   └── models.py
│   ├── services/                 # Business logic services
│   │   ├── __init__.py
│   │   └── services.py
│   ├── images/                   # Generated story images
│   │   └── [story_slug]/
│   │       ├── chapter_1.png
│   │       ├── chapter_2.png
│   │       └── ...
│   ├── story_intermediary/       # Temporary markdown files
│   │   └── story.md
│   └── resulted_stories/         # Final PDF outputs
│       └── story.pdf
└── config/                       # Additional configuration files
```

## 🤖 Agent Architecture

### 1. Story Outliner Agent
- **Role**: Creates structured story outlines
- **Responsibility**: Develops chapter titles, characters, and plot points
- **Output**: 5-chapter outline with character descriptions

### 2. Story Writer Agent
- **Role**: Writes complete story content
- **Responsibility**: Creates 100-word chapters following the outline
- **Output**: Full manuscript with title and chapters

### 3. Image Generator Agent
- **Role**: Creates chapter illustrations
- **Responsibility**: Generates DALL-E 3 images for each chapter
- **Output**: 5 PNG images (1024x1024) with consistent style

### 4. Content Formatter Agent
- **Role**: Formats content for publication
- **Responsibility**: Creates clean markdown with embedded images
- **Output**: Publication-ready markdown file

### 5. PDF Creator Agent
- **Role**: Converts markdown to PDF
- **Responsibility**: Professional PDF generation with images
- **Output**: Final PDF storybook

## ⚙️ Configuration

### Agent Configuration (`app/crew/config/agents.yaml`)
Customize agent behavior, roles, and capabilities:

```yaml
story_outliner:
  role: Story Outliner
  goal: Develop an outline for a children's storybook
  backstory: An imaginative creator who lays the foundation
  verbose: true
  allow_delegation: false
```

### Task Configuration (`app/crew/config/tasks.yaml`)
Define task workflows and dependencies:

```yaml
task_outline:
  description: Create an outline for the children's storybook
  expected_output: A structured outline document
  agent: story_outliner
```

### Model Configuration (`app/core/config.py`)
Switch between different LLM providers:

```python
# Use OpenAI GPT-4
model="openai/gpt-4-turbo"

# Use Groq models
model="groq/llama-3.2-90b-text-preview"
```

## 🔧 Troubleshooting

### Common Issues

#### 1. API Key Errors
```bash
# Verify your .env file contains valid API keys
cat .env | grep API_KEY
```

#### 2. PDF Generation Fails
```bash
# Install mdpdf globally
npm install -g mdpdf

# Or check if mdpdf is in PATH
which mdpdf
```

#### 3. Image Generation Issues
- Ensure OpenAI API key has DALL-E 3 access
- Check API usage limits
- Verify internet connectivity

#### 4. Port Already in Use
```bash
# Kill process using port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn app.main:app --port 8001
```

#### 5. Dependencies Issues
```bash
# Reinstall requirements
pip install --force-reinstall -r requirements.txt

# Clear pip cache
pip cache purge
```

## 🎨 Customization

### Story Templates
Edit `app/crew/knowledge/template.md` to modify the output format:

```markdown
# {Story Title}

## Chapter 1: {Chapter Title}
![Chapter 1 Image](../images/{slug}/chapter_1.png)

{Chapter content...}
```

### Image Styles
Modify the image generation prompt in `app/crew/tools/custom_tool.py`:

```python
prompt = f"Image is about: {chapter_content}. Style: Watercolor illustration..."
```

### Output Directories
Change output paths in the configuration:

```python
# Modify paths in custom_tool.py
image_folder = "custom_images/"
pdf_output = "custom_pdfs/"
```

## 📊 Monitoring and Logging

### View Logs
```bash
# Application logs
tail -f app.log

# PDF conversion logs
tail -f mdpdf.log
```

### Performance Monitoring
The API includes built-in metrics for:
- Story generation time
- Image generation time
- PDF conversion time
- Success/failure rates

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables for Production
```env
# Production configuration
DEBUG=false
API_V1_STR=/api/v1
ALLOWED_HOSTS=your-domain.com

# Rate limiting
RATE_LIMIT_PER_MINUTE=10
MAX_CONCURRENT_TASKS=5
```


## 🙏 Acknowledgments

- **[CrewAI](https://crewai.com)** for the multi-agent framework
- **OpenAI** for GPT-4 and DALL-E 3 APIs
- **FastAPI** for the web framework
- **Groq** for additional LLM options

---

**Create magical stories with the power of AI! ✨📚🤖**
