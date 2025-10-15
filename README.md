# Hotel Customer Service Chatbot with RAG Evaluation

An advanced AI-powered hotel customer service chatbot featuring RAG (Retrieval-Augmented Generation) with comprehensive quality evaluation using RAGAS metrics.

## 🌟 Features

### Core Capabilities
- **Intelligent Query Processing**: ML-based intent classification using scikit-learn
- **Named Entity Recognition**: Advanced NER using spaCy for extracting dates, names, room types, etc.
- **RAG Architecture**: LangChain-powered retrieval system with hotel-specific knowledge base
- **Context Management**: Maintains conversation history for coherent multi-turn dialogues
- **Real-time Performance Tracking**: Monitors response times, resolution rates, and confidence scores

### RAG Evaluation System
- **Faithfulness Measurement**: Validates if answers are grounded in retrieved context
- **Answer Relevancy Scoring**: Ensures responses directly address user queries
- **Context Precision**: Evaluates quality of retrieved documents
- **Context Recall**: Measures completeness of information retrieval
- **Answer Correctness**: Validates semantic accuracy against ground truth
- **Batch Evaluation**: Process multiple test cases simultaneously
- **Export Capabilities**: Save evaluation results to JSON for analysis

### Data Integration
- **Kaggle Dataset Support**: Loads real hotel booking data
- **Dynamic Knowledge Base**: Auto-generates insights from booking patterns
- **Fallback Mechanisms**: Sample data generation if external data unavailable

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install -r requirements_hotel.txt
```

### 2. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 3. Set Up Environment

Create a `.env` file in the project directory:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## 💻 Usage

### Basic Usage

```python
from hotel_chatbot_with_rag_eval import EnhancedHotelChatbot

# Initialize chatbot
chatbot = EnhancedHotelChatbot(use_kaggle_data=True)

# Process a query
response = chatbot.process_query("What time is check-in?")
print(response)
```

### Query with RAG Evaluation

```python
# Process query with evaluation
response = chatbot.process_query(
    user_query="What time is check-in?",
    ground_truth="Check-in time is 3:00 PM and check-out time is 11:00 AM.",
    evaluate_rag=True
)

# View evaluation results
chatbot.print_rag_evaluation_summary()
```

### Batch Evaluation

```python
# Define test cases
test_cases = [
    {
        "question": "What time is check-in?",
        "ground_truth": "Check-in time is 3:00 PM and check-out time is 11:00 AM."
    },
    {
        "question": "Do you allow pets?",
        "ground_truth": "Small pets under 15kg are welcome with a $50 additional fee per stay."
    },
    {
        "question": "What amenities do you offer?",
        "ground_truth": "Free WiFi, 24/7 fitness center, business center, on-site restaurant and bar, complimentary parking, and 24/7 room service."
    },
    {
        "question": "What is your cancellation policy?",
        "ground_truth": "Free cancellation is available up to 24 hours before arrival."
    },
]

# Run batch evaluation
chatbot.batch_evaluate_rag(test_cases)

# View comprehensive results
chatbot.print_rag_evaluation_summary()

# Export results to JSON
chatbot.export_rag_evaluations("evaluation_results.json")
```

### Conversation Context

```python
# Multi-turn conversation with context
session_id = "user_123"

response1 = chatbot.process_query("I need a room", session_id=session_id)
response2 = chatbot.process_query("For 3 people", session_id=session_id)
response3 = chatbot.process_query("Next Friday", session_id=session_id)
```

### Performance Metrics

```python
# Get performance statistics
metrics = chatbot.get_real_performance_metrics()
print(f"Total queries: {metrics['total_queries']}")
print(f"Avg response time: {metrics['average_response_time']:.2f}s")
print(f"Resolution accuracy: {metrics['query_resolution_accuracy_actual']:.1f}%")

# Print detailed analysis
chatbot.print_detailed_metrics()
```

### Stress Testing

```python
# Test high-volume performance
chatbot.stress_test(num_queries=100)
```

### Save Conversation History

```python
# Save all queries and responses
chatbot.save_conversation_history("chat_history.json")
```

## 📊 Understanding RAG Metrics

### Faithfulness (0.0 - 1.0)
**Measures:** Whether the answer is factually grounded in the retrieved context

- **>0.9:** Excellent - Answer fully supported by context
- **0.7-0.9:** Good - Mostly supported with minor gaps
- **0.5-0.7:** Fair - Some unsupported claims
- **<0.5:** Poor - Contains hallucinations or fabricated information

**Example:**
```
Context: "Check-in time is 3:00 PM"
Answer: "Check-in is at 3:00 PM" → Faithfulness: 1.0 ✓
Answer: "Check-in is at 2:00 PM" → Faithfulness: 0.0 ✗
```

### Answer Relevancy (0.0 - 1.0)
**Measures:** How directly the answer addresses the question

- **>0.9:** Excellent - Directly and completely answers the question
- **0.7-0.9:** Good - Addresses the question adequately
- **0.5-0.7:** Fair - Partially relevant
- **<0.5:** Poor - Off-topic or tangential

**Example:**
```
Question: "What time is check-in?"
Answer: "Check-in is at 3:00 PM" → Relevancy: 1.0 ✓
Answer: "We have a swimming pool" → Relevancy: 0.1 ✗
```

### Context Precision (0.0 - 1.0)
**Measures:** Relevance of retrieved documents to the question

- **>0.9:** Excellent - All retrieved contexts are highly relevant
- **0.7-0.9:** Good - Most contexts are relevant
- **0.5-0.7:** Fair - Some irrelevant contexts retrieved
- **<0.5:** Poor - Many irrelevant contexts

**Example:**
```
Question: "What time is check-in?"
Context 1: "Check-in time is 3:00 PM" → Relevant ✓
Context 2: "Pool hours are 9 AM - 5 PM" → Irrelevant ✗
Precision: 0.5
```

### Context Recall (0.0 - 1.0)
**Measures:** Completeness of information retrieval

*Requires ground truth for calculation*

- **>0.9:** Excellent - All necessary information retrieved
- **0.7-0.9:** Good - Most important information retrieved
- **0.5-0.7:** Fair - Some key information missing
- **<0.5:** Poor - Critical information not retrieved

### Answer Correctness (0.0 - 1.0)
**Measures:** Semantic similarity between answer and ground truth

*Requires ground truth for calculation*

- **>0.9:** Excellent - Answer matches ground truth meaning
- **0.7-0.9:** Good - Semantically similar with minor differences
- **0.5-0.7:** Fair - Partially correct
- **<0.5:** Poor - Semantically different or incorrect

## 🎯 Example Output

### RAG Evaluation Summary
```
==============================================================
RAG EVALUATION SUMMARY
==============================================================

Overall Metrics (n=5):
  • Faithfulness:       0.892 ✓
  • Answer Relevancy:   0.856 ✓
  • Context Precision:  0.743 ✓
  • Context Recall:     0.812 ✓
  • Answer Correctness: 0.778 ✓

Individual Evaluations:

  [1] Q: What time is check-in?...
      Faithfulness: 0.923
      Relevancy:    0.891
      Precision:    0.812
      Recall:       0.856
      Correctness:  0.823

  [2] Q: Do you allow pets?...
      Faithfulness: 0.867
      Relevancy:    0.834
      Precision:    0.789
      Recall:       0.801
      Correctness:  0.745

==============================================================
RECOMMENDATIONS
==============================================================
✓ RAG system performing well across all metrics!
```

### Performance Metrics
```
==============================================================
CHATBOT PERFORMANCE METRICS
==============================================================

📊 Query Processing:
   Total Queries: 50
   Avg Response Time: 2.34s
   Baseline Time: 15.0s

🚀 Performance Improvements:
   Response Time Reduction: 84.4%
   Resolution Accuracy: 92.0%
   Avg Confidence: 0.847

🎯 Intent Distribution:
   Booking Inquiry           : 15 queries
   Room Availability         : 12 queries
   Amenities                 : 10 queries
   Pricing                   : 8 queries
   Check In Out              : 5 queries

✅ Resolution Rate by Intent:
   Booking Inquiry           : 93.3% (14/15)
   Room Availability         : 91.7% (11/12)
   Amenities                 : 100.0% (10/10)
   Pricing                   : 87.5% (7/8)
   Check In Out              : 100.0% (5/5)
```

## 📁 Project Structure

```
hotel-chatbot/
│
├── hotel_chatbot_with_rag_eval.py  # Main chatbot code
├── requirements_hotel.txt           # Dependencies
├── .env                             # API keys (create this)
│
├── output/
│   ├── hotel_chat_history.json     # Conversation logs
│   └── hotel_rag_evaluations.json  # Evaluation results
│
└── README.md                        # This file
```

## 🔧 Configuration

### Chatbot Initialization

```python
chatbot = EnhancedHotelChatbot(
    use_kaggle_data=True  # Set to False to use sample data
)
```

### Retrieval Parameters

Modify in `_setup_llm()` method:

```python
self.retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}  # Number of documents to retrieve
)
```

### LLM Configuration

Modify in `_setup_llm()` method:

```python
self.llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    temperature=0  # Lower = more deterministic
)
```

## 🎨 Customization

### Adding New Intents

1. Add to `QueryIntent` enum:
```python
class QueryIntent(Enum):
    YOUR_NEW_INTENT = "your_new_intent"
```

2. Add training examples in `_generate_training_data()`:
```python
QueryIntent.YOUR_NEW_INTENT: [
    "example query 1",
    "example query 2",
]
```

3. Add fallback response in `_fallback_response()`:
```python
QueryIntent.YOUR_NEW_INTENT: "Your response here"
```

### Extending Knowledge Base

Add documents in `_create_knowledge_base()`:

```python
knowledge_items.append(
    "Your new hotel information here"
)
```

## 🐛 Troubleshooting

### "LLM initialization failed"
- Verify `OPENAI_API_KEY` is set in `.env` file
- Check API key has sufficient credits
- Ensure internet connectivity

### "spaCy model not found"
```bash
python -m spacy download en_core_web_sm
```

### "RAG evaluation failed"
- Install RAGAS: `pip install ragas==0.1.7`
- Install datasets: `pip install datasets==2.17.0`
- Check OpenAI API quota

### Low RAG Scores

**Low Faithfulness (<0.7):**
- Add more specific documents to knowledge base
- Refine system prompt to emphasize grounding
- Reduce LLM temperature

**Low Relevancy (<0.7):**
- Improve retrieval quality
- Add more focused, topic-specific documents
- Increase number of retrieved documents (k parameter)

**Low Precision (<0.7):**
- Tune retrieval similarity thresholds
- Use better embedding models
- Clean and structure documents better
- Adjust chunk size and overlap

**Low Recall (<0.7):**
- Increase k (number of documents retrieved)
- Add more comprehensive documents
- Reduce chunk size for finer granularity

## 📈 Performance Benchmarks

### Expected Metrics

**Response Performance:**
- Average response time: 2-5 seconds
- Baseline comparison: 85-90% faster than traditional systems
- Throughput: 20-50 queries/second
- Resolution accuracy: 90%+

**RAG Quality:**
- Faithfulness: >0.8
- Answer Relevancy: >0.8
- Context Precision: >0.7
- Context Recall: >0.7
- Answer Correctness: >0.75

## 🔒 Security & Privacy

- API keys stored in `.env` file (not committed to version control)
- Conversation history can be encrypted before saving
- No sensitive customer data stored by default
- All API calls use secure HTTPS

## 🚀 Deployment

### Local Development
```bash
python hotel_chatbot_with_rag_eval.py
```

### Production Considerations
- Use environment variables for API keys
- Implement rate limiting
- Add logging and monitoring
- Consider caching frequent queries
- Use production-grade vector databases
- Implement user authentication

## 📝 Best Practices

### Creating Test Cases
1. **Cover All Intents**: Test each intent category
2. **Use Realistic Queries**: Match actual customer language
3. **Provide Ground Truth**: Always include expected answers
4. **Test Edge Cases**: Ambiguous queries, typos, multi-intent

### Improving RAG Quality
1. **Iterate on Documents**: Continuously refine knowledge base
2. **Monitor Metrics**: Track scores over time
3. **A/B Testing**: Compare different prompt strategies
4. **User Feedback**: Incorporate real usage patterns

### Monitoring Performance
1. **Regular Evaluation**: Run batch tests weekly
2. **Track Trends**: Monitor metric changes
3. **Log Failures**: Analyze low-scoring queries
4. **Update Knowledge**: Keep hotel information current

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional query intents
- Multi-language support
- Sentiment analysis
- Booking system integration
- Voice interface
- More comprehensive evaluation metrics

## 🙏 Acknowledgments

- **LangChain**: RAG framework
- **RAGAS**: Evaluation metrics
- **spaCy**: NLP processing
- **OpenAI**: LLM API
- **Kaggle**: Hotel booking dataset

## 🔄 Version History

### v1.0.0 - Current
- Complete RAG implementation
- RAGAS evaluation integration
- ML-based intent classification
- Named entity recognition
- Performance tracking
- Batch evaluation
- Export capabilities

---

**Built with ❤️ for better customer service experiences**
