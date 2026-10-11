# 🧠 MindHaven Therapy Clinic Virtual Assistant

A conversational AI chatbot designed to provide emotional support and therapy guidance through intelligent intent classification and personalized responses.

## 📋 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)

## 🎯 Overview

**MindHaven** is an AI-powered virtual therapy assistant that leverages deep learning to understand user intents and provide empathetic, contextual responses. It's designed to be accessible, user-friendly, and supportive for individuals seeking emotional guidance.

## 🔍 Problem Statement

Mental health support is often inaccessible due to:
- **High costs** of professional therapy
- **Limited availability** of mental health professionals
- **Stigma** around seeking help
- **Time constraints** for scheduling appointments

MindHaven aims to bridge this gap by providing:
- 24/7 availability for emotional support
- Cost-effective mental health guidance
- A non-judgmental, private space to discuss feelings
- Quick access to therapeutic resources

## ✨ Features

- 🤖 **Intent Classification**: Uses neural networks to understand user intent from text input
- 💬 **Intelligent Responses**: Fetches contextual responses based on classified intents
- 🔊 **Text-to-Speech (TTS)**: Optional audio output for chatbot responses
- 🎨 **User-Friendly Interface**: Built with Streamlit for an intuitive web interface
- 📊 **Custom Training**: Train the model with your own intents and responses
- 🎭 **Personalized Interaction**: Greets users by name and maintains conversation history

## 🛠️ How It Works

### Architecture Overview

```
User Input
    ↓
Tokenization & Preprocessing
    ↓
Neural Network Model
    ↓
Intent Classification
    ↓
Intent Lookup in intents.json
    ↓
Response Generation
    ↓
User Output (Text + Optional Audio)
```

### Key Components

1. **[train.py](train.py)** - Model training and inference
   - Defines the `ChatbotModel` neural network
   - Implements intent classification logic
   - Handles response fetching from intents database

2. **[ui.py](ui.py)** - Streamlit web interface
   - User authentication and model training interface
   - Chat interaction display
   - Text-to-speech integration

3. **[intents.json](intents.json)** - Intent database
   - Defines therapy-related intents (greetings, questions, support)
   - Contains patterns users might say
   - Stores appropriate responses for each intent

4. **[best_chatbot.pth](best_chatbot.pth)** - Trained model weights
   - Pre-trained PyTorch model
   - Ready for immediate use without retraining

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/mindhaven-chatbot.git
cd mindhaven-chatbot
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; import streamlit; print('Installation successful!')"
```

## 📖 Usage

### Running the Application

```bash
streamlit run ui.py
```

The application will open in your default web browser at `http://localhost:8501`

### Training a Custom Model

1. **Enter your name** in the sidebar
2. **Specify the number of epochs** (100-30000 recommended)
3. **Click "Train Chatbot"** button
4. Wait for training to complete
5. Start chatting!

### Using Pre-trained Model

If the pre-trained `best_chatbot.pth` exists:
1. Enter your name
2. Click directly to the chat interface
3. Start conversing with the chatbot

### Features During Chat

- **Text Input**: Type your message in the chat input box
- **Audio Output**: Enable TTS to hear bot responses (if `TTS_ENABLED = True`)
- **Chat History**: All conversations are saved during the session
- **Intent Display**: See which intent was classified for your input

## 📁 Project Structure

```
mindhaven-chatbot/
├── train.py                 # Model training & inference logic
├── ui.py                    # Streamlit web interface
├── intents.json            # Intent database with patterns & responses
├── best_chatbot.pth        # Pre-trained model weights
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## 🔧 Technologies Used

| Technology | Purpose |
|-----------|---------|
| **PyTorch** | Deep learning framework for neural networks |
| **Streamlit** | Web UI framework for interactive dashboards |
| **Python** | Primary programming language |
| **NLTK/spaCy** | Natural language processing |
| **Pickle** | Model serialization |

## 📊 Model Details

### Neural Network Architecture
- **Input Layer**: Word embeddings from tokenized patterns
- **Hidden Layers**: Dense layers with ReLU activation
- **Output Layer**: Softmax activation for intent classification
- **Loss Function**: Cross-entropy loss
- **Optimizer**: Adam optimizer

### Training Parameters
- **Default Epochs**: Variable (100-30000)
- **Batch Size**: Configurable
- **Learning Rate**: Optimized for convergence
- **Validation Split**: Train-test evaluation

## 🎓 Example Intents

The chatbot understands intents like:
- **Greeting**: "Hello", "Hi", "Hey"
- **Support**: "I'm sad", "Help me", "I need support"
- **Questions**: "What is therapy?", "How can you help?"
- **Farewell**: "Goodbye", "See you", "Bye"

## 🔒 Security & Privacy

- ✅ **Local Processing**: All data processed locally
- ✅ **No Cloud Upload**: Models and data stay on your machine
- ✅ **Session-Based**: Data cleared after session ends
- ✅ **Anonymous**: No account or personal info required

## ⚙️ Configuration

Edit these settings in [ui.py](ui.py):

```python
TTS_ENABLED = True  # Enable/disable text-to-speech
```

## 🐛 Troubleshooting

### Issue: "Module not found" error
**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: Model not loading
**Solution**: Verify `best_chatbot.pth` exists in the root directory

### Issue: Streamlit won't start
**Solution**: Try running with explicit Python
```bash
python -m streamlit run ui.py
```

## 📈 Future Enhancements

- [ ] Multi-language support
- [ ] Sentiment analysis for better responses
- [ ] Integration with professional mental health resources
- [ ] Chat history export functionality
- [ ] Advanced user preference learning
- [ ] Mood tracking dashboard



