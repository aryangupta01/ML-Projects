# AI Chatbot with Neural Network

A simple yet powerful chatbot built using TensorFlow/TFLearn and NLTK for natural language processing. The bot uses a neural network to classify user intents and provide appropriate responses based on predefined patterns.

## Features

- **Intent Recognition**: Uses neural networks to classify user messages into different categories
- **Natural Language Processing**: Implements NLTK for text preprocessing and tokenization
- **Customizable Responses**: Easy to modify intents and responses through JSON configuration
- **Bag of Words Model**: Converts text input into numerical vectors for neural network processing
- **Conversation Memory**: Saves trained model for faster subsequent runs
- **Interactive Chat Interface**: Real-time conversation capability

## Technologies Used

- **Python 3.7+**
- **TensorFlow**: Deep learning framework
- **TFLearn**: High-level neural network library
- **NLTK**: Natural language processing toolkit
- **NumPy**: Numerical computing
- **Pickle**: Data serialization
- **JSON**: Configuration file format

## Project Structure

```
chatbot-project/
│
├── Chatbot.ipynb          # Main Jupyter notebook with implementation
├── intents.json           # Intent patterns and responses configuration
├── data.pickle           # Preprocessed training data (generated)
├── model.tflearn.*       # Trained model files (generated)
└── README.md             # Project documentation
```

## Setup Instructions

### Prerequisites

Make sure you have Python 3.7 or higher installed on your system.

### 1. Clone the Repository

```bash
git clone <repo-url>
cd ai-chatbot
```

### 2. Install Required Dependencies

```bash
pip install nltk numpy tensorflow tflearn pickle-mixin
```

### 3. Download NLTK Data

Run Python and execute:

```python
import nltk
nltk.download('punkt')
```

### 4. Prepare Intent Data

The `intents.json` file contains the training data. You can modify it to add new intents:

```json
{
  "tag": "your_intent_name",
  "patterns": ["user input example 1", "user input example 2"],
  "responses": ["bot response 1", "bot response 2"],
  "context_set": ""
}
```

## How to Use

### Running in Jupyter Notebook

1. Open `Chatbot.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells sequentially
3. The last cell will start the interactive chat interface
4. Type your messages and press Enter
5. Type `quit` to exit the chat

### Running as Python Script

If you want to convert the notebook to a Python script:

1. Save the notebook content as `chatbot.py`
2. Run: `python chatbot.py`

### Sample Conversation

```
Start talking with the bot (type quit to stop)!
You: Hi
Bot: Hello!

You: What is your name?
Bot: You can call me Jim.

You: How old are you?
Bot: I am 20 years old!

You: What are your hours?
Bot: We are open 7am-4pm Monday-Friday!

You: quit
```

## Customization

### Adding New Intents

1. Open `intents.json`
2. Add a new intent object with:
   - `tag`: Unique identifier for the intent
   - `patterns`: List of example user inputs
   - `responses`: List of possible bot responses
   - `context_set`: Context information (optional)

3. Delete `data.pickle` and `model.tflearn.*` files
4. Re-run the training process

### Example New Intent

```json
{
  "tag": "weather",
  "patterns": ["How's the weather?", "What's the weather like?", "Is it sunny?"],
  "responses": ["I'm sorry, I don't have access to weather data.", "You might want to check a weather app for that!"],
  "context_set": ""
}
```

### Modifying Neural Network Architecture

In the notebook, you can adjust the neural network structure:

```python
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)  # First hidden layer (8 neurons)
net = tflearn.fully_connected(net, 8)  # Second hidden layer (8 neurons)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
```

You can:
- Change the number of neurons in hidden layers
- Add more hidden layers
- Modify the number of training epochs
- Adjust the batch size

## How It Works

1. **Data Preprocessing**: 
   - Tokenizes input patterns using NLTK
   - Stems words using Lancaster Stemmer
   - Creates bag-of-words representation

2. **Neural Network Training**:
   - Uses a feedforward neural network
   - Two hidden layers with 8 neurons each
   - Softmax activation for output classification
   - Adam optimizer for training

3. **Intent Classification**:
   - Converts user input to bag-of-words vector
   - Passes through trained neural network
   - Selects intent with highest confidence (>70%)
   - Returns random response from matched intent

## Model Performance

- **Training Accuracy**: ~91% (after 1000 epochs)
- **Confidence Threshold**: 70% (responses only given if confidence > 0.7)
- **Fallback Response**: "I didn't get that, try again." for low confidence predictions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Future Enhancements

- [ ] Add context handling for multi-turn conversations
- [ ] Implement spell checking and correction
- [ ] Add support for more complex entities
- [ ] Create a web interface using Flask/Django
- [ ] Add database integration for conversation history
- [ ] Implement more sophisticated NLP techniques
- [ ] Add support for multiple languages

## Troubleshooting

### Common Issues

1. **NLTK Data Not Found**:
   ```python
   import nltk
   nltk.download('punkt')
   ```

2. **TensorFlow Version Compatibility**:
   - This project uses TensorFlow 1.x syntax
   - For TensorFlow 2.x, you might need compatibility mode

3. **Memory Issues**:
   - Reduce batch size or number of epochs
   - Use a smaller neural network architecture

## Acknowledgments

- Built using TensorFlow and TFLearn frameworks
- NLTK library for natural language processing
- Inspired by various chatbot tutorials and implementations

---

**Note**: This is a basic implementation suitable for learning purposes. For production use, consider more advanced frameworks like Rasa, Dialogflow, or custom transformer models.
