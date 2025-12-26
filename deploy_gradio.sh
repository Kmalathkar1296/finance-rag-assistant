#!/bin/bash
# Automated deployment script for Finance RAG Assistant with Gradio
# Deploys to GitHub and HuggingFace Spaces

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "======================================================================"
echo "  Finance RAG Assistant - Gradio Deployment Script"
echo "======================================================================"
echo -e "${NC}"

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Step 1: Check prerequisites
echo ""
echo "Step 1: Checking prerequisites..."
echo "----------------------------------------"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python installed: $PYTHON_VERSION"
else
    print_error "Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check Git
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version)
    print_success "Git installed: $GIT_VERSION"
else
    print_error "Git not found. Please install Git"
    exit 1
fi

# Check if in correct directory
if [ ! -d "src" ]; then
    print_warning "src/ directory not found. Are you in the project root?"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 2: Get user information
echo ""
echo "Step 2: Configuration"
echo "----------------------------------------"

read -p "GitHub username: " GITHUB_USERNAME
read -p "Repository name (default: finance-rag-assistant): " REPO_NAME
REPO_NAME=${REPO_NAME:-finance-rag-assistant}

read -p "HuggingFace username: " HF_USERNAME
read -p "HuggingFace Space name (default: finance-rag-assistant): " HF_SPACE
HF_SPACE=${HF_SPACE:-finance-rag-assistant}

echo ""
print_info "Configuration:"
print_info "  GitHub: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
print_info "  HuggingFace: https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE"
echo ""

read -p "Is this correct? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    print_error "Deployment cancelled"
    exit 1
fi

# Step 3: Create requirements.txt
echo ""
echo "Step 3: Creating requirements.txt..."
echo "----------------------------------------"

cat > requirements.txt << 'EOF'
gradio==4.44.1
langchain==1.1.0
langchain-community==1.1.0
langchain-core==1.1.0
langchain-text-splitters==1.1.0
langchain-huggingface==1.1.0
langchain-chroma==1.1.0
langchain-anthropic==1.1.0
chromadb==0.5.23
sentence-transformers==3.3.1
pandas==2.2.3
numpy==2.2.0
openpyxl==3.1.5
python-dotenv==1.0.1
pydantic>=2.9.0
typing-extensions>=4.12.0
EOF

print_success "requirements.txt created"

# Step 4: Create README.md
echo ""
echo "Step 4: Creating README.md..."
echo "----------------------------------------"

cat > README.md << EOF
---
title: Finance RAG Assistant
emoji: 💰
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# 💰 Finance RAG Assistant

AI-powered payment reconciliation and financial analysis system.

## Features

- 🔍 Natural language queries for financial data
- 💸 Automatic payment discrepancy detection
- 📊 Budget variance analysis
- 💳 Expense claims management
- 📈 Comprehensive financial reporting

## Live Demo

Visit: [https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE](https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE)

## Usage

1. **Setup**: Generate sample financial data
2. **Query**: Ask questions in natural language
3. **Analyze**: Review discrepancies and insights
4. **Report**: Generate comprehensive reports

## Tech Stack

- **LangChain 1.1.0** - RAG orchestration
- **ChromaDB** - Vector database
- **Gradio** - Web interface
- **Sentence Transformers** - Embeddings

## Local Development

\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Run app
python app.py
\`\`\`

Visit http://localhost:7860

## Deployment

Deployed on HuggingFace Spaces with free tier:
- 16GB RAM
- Auto-scaling
- Public access
- No authentication required

## License

MIT License
EOF

print_success "README.md created"

# Step 5: Update .gitignore
echo ""
echo "Step 5: Updating .gitignore..."
echo "----------------------------------------"

cat >> .gitignore << 'EOF'

# Gradio specific
flagged/
gradio_cached_examples/

# HuggingFace
.huggingface/
EOF

print_success ".gitignore updated"

# Step 6: Initialize Git (if needed)
echo ""
echo "Step 6: Git setup..."
echo "----------------------------------------"

if [ ! -d ".git" ]; then
    print_info "Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit: Finance RAG Assistant with Gradio"
    print_success "Git repository initialized"
else
    print_success "Git repository already exists"
fi

# Step 7: Add GitHub remote
echo ""
echo "Step 7: Connecting to GitHub..."
echo "----------------------------------------"

# Check if remote already exists
if git remote | grep -q "origin"; then
    print_warning "Remote 'origin' already exists"
    read -p "Update remote URL? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git remote set-url origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
        print_success "Remote 'origin' updated"
    fi
else
    git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
    print_success "Remote 'origin' added"
fi

# Step 8: Push to GitHub
echo ""
echo "Step 8: Pushing to GitHub..."
echo "----------------------------------------"

print_info "Make sure you've created the repository on GitHub:"
print_info "  https://github.com/new"
echo ""
read -p "Repository created on GitHub? (Y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    git branch -M main
    
    if git push -u origin main; then
        print_success "Pushed to GitHub successfully!"
        print_info "View at: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    else
        print_warning "Push failed. You may need to authenticate or create the repo first."
        print_info "Create repo: https://github.com/new"
        print_info "Then run: git push -u origin main"
    fi
else
    print_warning "Skipping GitHub push"
fi

# Step 9: Add HuggingFace remote
echo ""
echo "Step 9: Connecting to HuggingFace..."
echo "----------------------------------------"

print_info "First, create your Space on HuggingFace:"
print_info "  1. Go to: https://huggingface.co/new-space"
print_info "  2. Name: $HF_SPACE"
print_info "  3. SDK: Gradio"
print_info "  4. Create Space"
echo ""
read -p "Space created on HuggingFace? (Y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    # Check if HF remote exists
    if git remote | grep -q "hf"; then
        print_warning "Remote 'hf' already exists"
        git remote set-url hf "https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE"
    else
        git remote add hf "https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE"
    fi
    
    print_success "HuggingFace remote added"
    
    # Step 10: Push to HuggingFace
    echo ""
    echo "Step 10: Deploying to HuggingFace..."
    echo "----------------------------------------"
    
    print_info "You'll need to authenticate with HuggingFace"
    print_info "Get your token: https://huggingface.co/settings/tokens"
    echo ""
    
    if git push hf main; then
        print_success "Deployed to HuggingFace successfully!"
        echo ""
        echo -e "${GREEN}======================================================================"
        echo "  🎉 Deployment Complete!"
        echo "======================================================================${NC}"
        echo ""
        echo -e "📱 Your app is live at:"
        echo -e "   ${BLUE}https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE${NC}"
        echo ""
        echo -e "🔗 GitHub repository:"
        echo -e "   ${BLUE}https://github.com/$GITHUB_USERNAME/$REPO_NAME${NC}"
        echo ""
        echo -e "⏱️  Building time: ~2-5 minutes"
        echo -e "📊 Monitor build: Check 'Logs' tab on HuggingFace"
        echo ""
        echo -e "${GREEN}=====================================================================${NC}"
    else
        print_error "Push to HuggingFace failed"
        print_info "Manual steps:"
        print_info "  1. Get HF token: https://huggingface.co/settings/tokens"
        print_info "  2. Run: git push https://USERNAME:TOKEN@huggingface.co/spaces/$HF_USERNAME/$HF_SPACE main"
    fi
else
    print_warning "Skipping HuggingFace deployment"
    print_info "To deploy manually later:"
    print_info "  git remote add hf https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE"
    print_info "  git push hf main"
fi

# Final notes
echo ""
echo "======================================================================"
echo "  Next Steps"
echo "======================================================================"
echo ""
echo "1. ⏰ Wait 2-5 minutes for HuggingFace to build your Space"
echo "2. 🔍 Check build status: https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE"
echo "3. 🎯 Test your app when 'Running' status appears"
echo "4. 🔑 (Optional) Add API keys in Space Settings → Secrets"
echo "5. 📱 Share your link with anyone!"
echo ""
echo "To update your app:"
echo "  git add ."
echo "  git commit -m 'Update'"
echo "  git push origin main    # Push to GitHub"
echo "  git push hf main        # Deploy to HuggingFace"
echo ""
echo "======================================================================"
