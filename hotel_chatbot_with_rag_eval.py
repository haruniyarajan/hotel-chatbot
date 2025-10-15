"""
Enhanced Hotel Customer Service Chatbot with RAG EVALUATION
Complete version with RAGAS metrics for evaluating response quality

Dependencies:
pip install kagglehub pandas langchain-openai langchain-community python-dotenv chromadb spacy scikit-learn ragas datasets
python -m spacy download en_core_web_sm
"""

import os
import json
import datetime
import time
import glob
import random
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

import kagglehub

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Enhanced NLP imports
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# RAG Evaluation imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from datasets import Dataset

load_dotenv()

class QueryIntent(Enum):
    BOOKING_INQUIRY = "booking_inquiry"
    CANCELLATION = "cancellation"
    CHECK_IN_OUT = "check_in_out"
    ROOM_AVAILABILITY = "room_availability"
    SPECIAL_REQUESTS = "special_requests"
    PRICING = "pricing"
    AMENITIES = "amenities"
    GENERAL_INFO = "general_info"
    COMPLAINT = "complaint"
    UNKNOWN = "unknown"

@dataclass
class CustomerQuery:
    text: str
    intent: QueryIntent
    entities: Dict[str, str]
    timestamp: str
    response_time: float
    response: str = ""
    resolved: bool = False
    escalated: bool = False
    confidence_score: float = 0.0
    
    def __post_init__(self):
        if not hasattr(self, 'timestamp') or not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()

@dataclass
class RAGEvaluation:
    """Store RAG evaluation results"""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    faithfulness_score: float = 0.0
    answer_relevancy_score: float = 0.0
    context_precision_score: float = 0.0
    context_recall_score: float = 0.0
    answer_correctness_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

class PerformanceTracker:
    """Tracks real performance metrics vs baseline"""
    
    def __init__(self):
        self.baseline_response_time = 15.0
        self.baseline_accuracy = 60.0
        self.queries_processed = []
        self.start_time = time.time()
    
    def record_query(self, query: CustomerQuery, actual_processing_time: float):
        """Record actual query performance"""
        self.queries_processed.append({
            'query': query,
            'actual_time': actual_processing_time,
            'baseline_time': self.baseline_response_time,
            'resolved': query.resolved,
            'confidence': query.confidence_score
        })
    
    def get_real_metrics(self) -> Dict:
        """Calculate actual performance improvements"""
        if not self.queries_processed:
            return {"error": "No queries processed yet"}
        
        avg_ai_time = sum(q['actual_time'] for q in self.queries_processed) / len(self.queries_processed)
        response_time_reduction = ((self.baseline_response_time - avg_ai_time) / self.baseline_response_time) * 100
        
        resolved_count = sum(1 for q in self.queries_processed if q['resolved'])
        actual_accuracy = (resolved_count / len(self.queries_processed)) * 100
        
        avg_confidence = sum(q['confidence'] for q in self.queries_processed) / len(self.queries_processed)
        
        return {
            "total_queries": len(self.queries_processed),
            "response_time_reduction_actual": response_time_reduction,
            "query_resolution_accuracy_actual": actual_accuracy,
            "average_confidence": avg_confidence,
            "average_response_time": avg_ai_time,
            "baseline_response_time": self.baseline_response_time,
            "processing_speed_improvement": f"{response_time_reduction:.1f}%"
        }

class AdvancedNLPProcessor:
    """Enhanced NLP with ML-based intent classification and NER"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✓ spaCy model loaded successfully")
        except OSError:
            print("⚠️  spaCy model not found. Run: python -m spacy download en_core_web_sm")
            print("Using basic processing without NER...")
            self.nlp = None
        
        self.training_data = self._generate_training_data()
        self.intent_classifier = self._train_intent_classifier()
    
    def _generate_training_data(self) -> Tuple[List[str], List[str]]:
        """Generate training data for intent classification"""
        training_examples = {
            QueryIntent.BOOKING_INQUIRY: [
                "I want to book a room", "Can I make a reservation", "Book hotel for tonight",
                "Reserve a room for Friday", "I need accommodation", "Looking to stay here",
                "Make a booking", "Reserve accommodation", "Book me a suite"
            ],
            QueryIntent.CANCELLATION: [
                "Cancel my booking", "I want to cancel reservation", "Need to cancel my stay",
                "Remove my booking", "Cancel room reservation", "Refund my booking",
                "Delete reservation", "I need to cancel", "Cancellation request"
            ],
            QueryIntent.CHECK_IN_OUT: [
                "What time is check-in", "When can I check out", "Early check-in possible",
                "Late checkout", "Check-in procedures", "Checkout time",
                "Check in time", "What's the checkout time", "Early arrival"
            ],
            QueryIntent.ROOM_AVAILABILITY: [
                "Do you have rooms available", "Any vacancy", "Rooms for tonight",
                "Available rooms this weekend", "Check availability", "Room options",
                "Any free rooms", "Rooms left", "Vacant rooms"
            ],
            QueryIntent.PRICING: [
                "How much does it cost", "Room rates", "Price per night", "Cost of stay",
                "Pricing information", "What are your rates", "Expensive rooms",
                "How much", "Room prices", "Cost for weekend"
            ],
            QueryIntent.AMENITIES: [
                "Do you have wifi", "Is there a pool", "Gym facilities", "Free breakfast",
                "Parking available", "Room service", "What amenities do you offer",
                "Hotel facilities", "Do you have spa", "Swimming pool"
            ],
            QueryIntent.COMPLAINT: [
                "I have a complaint", "Not satisfied with service", "Room was dirty",
                "Poor service", "Want to complain", "Unhappy with stay",
                "Terrible experience", "Not happy", "Bad service"
            ],
            QueryIntent.GENERAL_INFO: [
                "Tell me about the hotel", "Hotel information", "Where are you located",
                "Hotel address", "Contact information", "About your hotel"
            ]
        }
        
        texts = []
        labels = []
        for intent, examples in training_examples.items():
            texts.extend(examples)
            labels.extend([intent.value] * len(examples))
        
        return texts, labels
    
    def _train_intent_classifier(self) -> Pipeline:
        """Train ML-based intent classifier"""
        texts, labels = self.training_data
        
        classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        classifier.fit(texts, labels)
        
        predictions = classifier.predict(texts)
        accuracy = accuracy_score(labels, predictions)
        print(f"✓ Intent Classifier Training Accuracy: {accuracy:.3f}")
        
        return classifier
    
    def extract_intent_ml(self, query: str) -> Tuple[QueryIntent, float]:
        """Extract intent using ML classifier"""
        try:
            prediction = self.intent_classifier.predict([query])[0]
            probabilities = self.intent_classifier.predict_proba([query])[0]
            confidence = max(probabilities)
            
            for intent in QueryIntent:
                if intent.value == prediction:
                    return intent, confidence
            
            return QueryIntent.UNKNOWN, 0.5
        except Exception as e:
            print(f"⚠️  Intent classification error: {e}")
            return QueryIntent.UNKNOWN, 0.0
    
    def extract_entities_ner(self, query: str) -> Dict[str, str]:
        """Extract entities using spaCy NER with error handling"""
        entities = {}
        
        if self.nlp:
            try:
                doc = self.nlp(query)
                
                for ent in doc.ents:
                    if ent.label_ == "DATE":
                        entities['date'] = ent.text
                    elif ent.label_ == "CARDINAL":
                        entities['number'] = ent.text
                    elif ent.label_ == "PERSON":
                        entities['person'] = ent.text
                    elif ent.label_ == "GPE":
                        entities['location'] = ent.text
                
                room_types = ['standard', 'deluxe', 'suite', 'single', 'double', 'twin']
                for token in doc:
                    if token.text.lower() in room_types:
                        entities['room_type'] = token.text.lower()
                        
            except Exception as e:
                print(f"⚠️  NER processing error: {e}")
        
        # Fallback to regex
        if 'date' not in entities:
            date_patterns = [
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
                r'\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
            ]
            for pattern in date_patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    entities['date'] = matches[0]
                    break
        
        if 'number' not in entities:
            numbers = re.findall(r'\b(\d+)\b', query)
            if numbers:
                entities['number'] = numbers[0]
        
        return entities

class EnhancedHotelChatbot:
    def __init__(self, use_kaggle_data: bool = True):
        self.queries: List[CustomerQuery] = []
        self.llm = None
        self.rag_chain = None
        self.retriever = None
        self.hotel_data = None
        self.conversation_context = {}
        
        # Enhanced components
        self.nlp_processor = AdvancedNLPProcessor()
        self.performance_tracker = PerformanceTracker()
        self.rag_evaluations: List[RAGEvaluation] = []
        
        self._load_hotel_data(use_kaggle_data)
        self._setup_llm()
    
    def _load_hotel_data(self, use_kaggle_data: bool = True):
        """Load hotel bookings data from Kaggle dataset"""
        if use_kaggle_data:
            try:
                print("Loading hotel booking data from Kaggle...")
                path = kagglehub.dataset_download("thedevastator/hotel-bookings-analysis")
                print(f"Dataset downloaded to: {path}")
                
                csv_files = glob.glob(os.path.join(path, "*.csv"))
                
                if csv_files:
                    print(f"Found {len(csv_files)} CSV file(s)")
                    self.hotel_data = pd.read_csv(csv_files[0])
                    print(f"✓ Successfully loaded dataset: {len(self.hotel_data)} bookings")
                else:
                    raise FileNotFoundError("No CSV files found")
                    
            except Exception as e:
                print(f"⚠️  Error loading Kaggle dataset: {e}")
                print("Using sample data instead...")
                self.hotel_data = self._create_sample_data()
        else:
            self.hotel_data = self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create comprehensive sample hotel data"""
        random.seed(42)
        
        data = {
            'hotel': ['City Hotel'] * 500 + ['Resort Hotel'] * 500,
            'adr': [random.uniform(50, 300) for _ in range(1000)],
            'adults': [random.choice([1, 2, 3, 4]) for _ in range(1000)],
            'children': [random.choice([0, 0, 0, 1, 2]) for _ in range(1000)],
            'meal': [random.choice(['BB', 'HB', 'FB', 'SC']) for _ in range(1000)],
            'country': [random.choice(['USA', 'UK', 'Germany', 'France', 'Spain']) for _ in range(1000)],
            'reserved_room_type': [random.choice(['A', 'B', 'C', 'D', 'E']) for _ in range(1000)],
            'is_canceled': [random.choice([0, 0, 0, 1]) for _ in range(1000)],
            'lead_time': [random.randint(0, 365) for _ in range(1000)]
        }
        
        print(f"✓ Created sample dataset: {len(data['hotel'])} bookings")
        return pd.DataFrame(data)
    
    def _setup_llm(self):
        """Setup LLM with hotel knowledge"""
        try:
            self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
            documents = self._create_knowledge_base()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            
            system_prompt = (
                "You are a professional hotel customer service agent. Use the hotel data to provide accurate, "
                "helpful responses. Be concise but thorough. Base your answers strictly on the provided context. "
                "If you cannot fully resolve an issue, suggest escalation to a human agent.\n\n{context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])
            
            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
            self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
            
            print("✓ LLM and RAG chain initialized successfully")
            
        except Exception as e:
            print(f"⚠️  LLM initialization failed: {e}")
            print("Chatbot will use fallback responses...")
    
    def _create_knowledge_base(self) -> List[Document]:
        """Create comprehensive knowledge base from hotel data"""
        documents = []
        
        if self.hotel_data is not None and len(self.hotel_data) > 0:
            try:
                avg_price = self.hotel_data['adr'].mean() if 'adr' in self.hotel_data.columns else 100.0
                min_price = self.hotel_data['adr'].min() if 'adr' in self.hotel_data.columns else 50.0
                max_price = self.hotel_data['adr'].max() if 'adr' in self.hotel_data.columns else 300.0
                cancellation_rate = self.hotel_data['is_canceled'].mean() * 100 if 'is_canceled' in self.hotel_data.columns else 20.0
                occupancy_rate = 100 - cancellation_rate
                
                hotel_types = list(self.hotel_data['hotel'].unique()) if 'hotel' in self.hotel_data.columns else ['City Hotel', 'Resort Hotel']
                room_types = list(self.hotel_data['reserved_room_type'].unique()) if 'reserved_room_type' in self.hotel_data.columns else ['A', 'B', 'C', 'D']
                meal_plans = list(self.hotel_data['meal'].unique()) if 'meal' in self.hotel_data.columns else ['BB', 'HB', 'FB']
                
                knowledge_items = [
                    f"Our hotels include: {', '.join(str(h) for h in hotel_types)}",
                    f"Average room rate: ${avg_price:.2f} per night (range: ${min_price:.2f} - ${max_price:.2f})",
                    f"Current occupancy rate: {occupancy_rate:.1f}%",
                    f"Available room types: {', '.join(str(r) for r in room_types)}",
                    f"Meal plans: {', '.join(str(m) for m in meal_plans)} (BB=Bed&Breakfast, HB=Half Board, FB=Full Board, SC=Self Catering)",
                    f"Total bookings in our system: {len(self.hotel_data):,}",
                    "Check-in time: 3:00 PM, Check-out time: 11:00 AM",
                    "Cancellation policy: Free cancellation up to 24 hours before arrival",
                    "Amenities: Free WiFi throughout property, Fitness center (24/7), Business center, On-site restaurant and bar",
                    "Pet policy: Small pets (under 15kg) welcome with $50 additional fee per stay",
                    "Parking: Complimentary on-site parking available",
                    "Room service: Available 24/7 with varied menu options",
                    "Concierge services: Available for local recommendations, bookings, and tours",
                    "Accessibility: Wheelchair accessible rooms and facilities available",
                    "Security: 24/7 front desk and security personnel",
                    "Languages: Staff fluent in English, Spanish, French, and German"
                ]
                
                for item in knowledge_items:
                    documents.append(Document(page_content=item))
                
                print(f"✓ Knowledge base created with {len(documents)} documents")
                
            except Exception as e:
                print(f"⚠️  Error creating knowledge base: {e}")
                documents = [Document(page_content="Hotel information temporarily unavailable.")]
        
        return documents if documents else [Document(page_content="Hotel services available.")]
    
    def process_query(self, user_query: str, session_id: str = "default", 
                     ground_truth: Optional[str] = None, evaluate_rag: bool = False) -> str:
        """Process customer query with optional RAG evaluation"""
        query_start_time = time.time()
        
        # Intent and entity extraction
        intent, confidence = self.nlp_processor.extract_intent_ml(user_query)
        entities = self.nlp_processor.extract_entities_ner(user_query)
        
        # Add conversation context
        context = self.conversation_context.get(session_id, "")
        if context:
            enhanced_query = f"Previous context: {context}\nCurrent query: {user_query}"
        else:
            enhanced_query = user_query
        
        # Generate response using RAG
        contexts = []
        try:
            if self.rag_chain:
                result = self.rag_chain.invoke({"input": enhanced_query})
                response = result["answer"]
                
                # Extract contexts
                if "context" in result:
                    if isinstance(result["context"], list):
                        contexts = [doc.page_content for doc in result["context"]]
                    else:
                        contexts = [result["context"]]
            else:
                response = self._fallback_response(intent, entities)
            
            # Determine resolution
            unresolved_indicators = [
                'cannot help', 'need to escalate', 'speak to manager', 
                'contact human', 'technical difficulties', 'unable to assist'
            ]
            resolved = not any(indicator in response.lower() for indicator in unresolved_indicators)
            
            if confidence > 0.8:
                resolved = True
            elif confidence < 0.3:
                resolved = False
                
        except Exception as e:
            print(f"⚠️  Error generating response: {e}")
            response = "I apologize for the technical difficulty. Let me connect you with a human agent."
            resolved = False
            confidence = 0.0
        
        # Calculate processing time
        total_processing_time = time.time() - query_start_time
        
        # Create query record
        query_record = CustomerQuery(
            text=user_query,
            intent=intent,
            entities=entities,
            timestamp=datetime.datetime.now().isoformat(),
            response_time=total_processing_time,
            response=response,
            resolved=resolved,
            confidence_score=confidence
        )
        
        self.queries.append(query_record)
        self.performance_tracker.record_query(query_record, total_processing_time)
        
        # RAG Evaluation if requested
        if evaluate_rag and contexts:
            self._evaluate_rag_response(
                question=user_query,
                answer=response,
                contexts=contexts,
                ground_truth=ground_truth
            )
        
        # Update context
        self.conversation_context[session_id] = f"Intent: {intent.value}, Resolved: {resolved}"
        
        return response
    
    def _evaluate_rag_response(self, question: str, answer: str, 
                              contexts: List[str], ground_truth: Optional[str] = None):
        """Evaluate RAG response using RAGAS metrics"""
        try:
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            
            if ground_truth:
                data["ground_truth"] = [ground_truth]
            
            dataset = Dataset.from_dict(data)
            
            # Select metrics
            metrics_to_use = [
                faithfulness,
                answer_relevancy,
                context_precision,
            ]
            
            if ground_truth:
                metrics_to_use.extend([
                    context_recall,
                    answer_correctness,
                ])
            
            print("\n🔍 Evaluating RAG response...")
            result = evaluate(dataset, metrics=metrics_to_use)
            
            # Store evaluation
            eval_result = RAGEvaluation(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
                faithfulness_score=result.get('faithfulness', 0.0),
                answer_relevancy_score=result.get('answer_relevancy', 0.0),
                context_precision_score=result.get('context_precision', 0.0),
                context_recall_score=result.get('context_recall', 0.0) if ground_truth else 0.0,
                answer_correctness_score=result.get('answer_correctness', 0.0) if ground_truth else 0.0,
            )
            
            self.rag_evaluations.append(eval_result)
            print("✓ RAG evaluation complete")
            
        except Exception as e:
            print(f"⚠️  RAG evaluation failed: {e}")
    
    def batch_evaluate_rag(self, test_cases: List[Dict]):
        """Evaluate RAG on multiple test cases"""
        if not self.rag_chain:
            print("⚠️  LLM not available for evaluation")
            return
        
        print(f"\n{'='*60}")
        print("BATCH RAG EVALUATION")
        print(f"{'='*60}")
        print(f"Evaluating {len(test_cases)} test cases...\n")
        
        for i, test_case in enumerate(test_cases, 1):
            question = test_case.get("question", "")
            ground_truth = test_case.get("ground_truth", None)
            
            print(f"[{i}/{len(test_cases)}] Testing: {question[:60]}...")
            
            answer = self.process_query(question, ground_truth=ground_truth, evaluate_rag=True)
            time.sleep(1)  # Rate limiting
    
    def print_rag_evaluation_summary(self):
        """Print comprehensive RAG evaluation summary"""
        if not self.rag_evaluations:
            print("\n⚠️  No RAG evaluations available")
            return
        
        print(f"\n{'='*60}")
        print("RAG EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        # Calculate averages
        avg_faithfulness = sum(e.faithfulness_score for e in self.rag_evaluations) / len(self.rag_evaluations)
        avg_relevancy = sum(e.answer_relevancy_score for e in self.rag_evaluations) / len(self.rag_evaluations)
        avg_precision = sum(e.context_precision_score for e in self.rag_evaluations) / len(self.rag_evaluations)
        
        has_ground_truth = any(e.ground_truth is not None for e in self.rag_evaluations)
        
        print(f"\nOverall Metrics (n={len(self.rag_evaluations)}):")
        print(f"  • Faithfulness:       {avg_faithfulness:.3f} {'✓' if avg_faithfulness > 0.7 else '⚠️'}")
        print(f"  • Answer Relevancy:   {avg_relevancy:.3f} {'✓' if avg_relevancy > 0.7 else '⚠️'}")
        print(f"  • Context Precision:  {avg_precision:.3f} {'✓' if avg_precision > 0.7 else '⚠️'}")
        
        if has_ground_truth:
            evals_with_gt = [e for e in self.rag_evaluations if e.ground_truth is not None]
            avg_recall = sum(e.context_recall_score for e in evals_with_gt) / len(evals_with_gt)
            avg_correctness = sum(e.answer_correctness_score for e in evals_with_gt) / len(evals_with_gt)
            
            print(f"  • Context Recall:     {avg_recall:.3f} {'✓' if avg_recall > 0.7 else '⚠️'}")
            print(f"  • Answer Correctness: {avg_correctness:.3f} {'✓' if avg_correctness > 0.7 else '⚠️'}")
        
        # Individual results
        print(f"\nIndividual Evaluations:")
        for i, eval_result in enumerate(self.rag_evaluations, 1):
            print(f"\n  [{i}] Q: {eval_result.question[:50]}...")
            print(f"      Faithfulness: {eval_result.faithfulness_score:.3f}")
            print(f"      Relevancy:    {eval_result.answer_relevancy_score:.3f}")
            print(f"      Precision:    {eval_result.context_precision_score:.3f}")
            if eval_result.ground_truth:
                print(f"      Recall:       {eval_result.context_recall_score:.3f}")
                print(f"      Correctness:  {eval_result.answer_correctness_score:.3f}")
        
        # Recommendations
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        
        if avg_faithfulness < 0.7:
            print("⚠️  Low faithfulness - answers may not be grounded in context")
            print("   → Improve: Add more specific documents, refine prompts")
        
        if avg_relevancy < 0.7:
            print("⚠️  Low answer relevancy - answers may be off-topic")
            print("   → Improve: Better retrieval, more focused documents")
        
        if avg_precision < 0.7:
            print("⚠️  Low context precision - irrelevant contexts retrieved")
            print("   → Improve: Tune retrieval parameters, better embeddings")
        
        if all([avg_faithfulness > 0.7, avg_relevancy > 0.7, avg_precision > 0.7]):
            print("✓ RAG system performing well across all metrics!")
    
    def export_rag_evaluations(self, filename: str = "hotel_rag_evaluations.json"):
        """Export RAG evaluations to JSON"""
        if not self.rag_evaluations:
            print("⚠️  No evaluations to export")
            return
        
        data = {
            "evaluations": [
                {
                    "question": e.question,
                    "answer": e.answer,
                    "contexts": e.contexts,
                    "ground_truth": e.ground_truth,
                    "scores": {
                        "faithfulness": e.faithfulness_score,
                        "answer_relevancy": e.answer_relevancy_score,
                        "context_precision": e.context_precision_score,
                        "context_recall": e.context_recall_score,
                        "answer_correctness": e.answer_correctness_score,
                    },
                    "timestamp": e.timestamp
                }
                for e in self.rag_evaluations
            ],
            "summary": {
                "total_evaluations": len(self.rag_evaluations),
                "avg_faithfulness": sum(e.faithfulness_score for e in self.rag_evaluations) / len(self.rag_evaluations),
                "avg_answer_relevancy": sum(e.answer_relevancy_score for e in self.rag_evaluations) / len(self.rag_evaluations),
                "avg_context_precision": sum(e.context_precision_score for e in self.rag_evaluations) / len(self.rag_evaluations),
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ RAG evaluations exported to {filename}")
    
    def _fallback_response(self, intent: QueryIntent, entities: Dict[str, str]) -> str:
        """Enhanced fallback responses"""
        if self.hotel_data is not None and len(self.hotel_data) > 0:
            avg_price = self.hotel_data['adr'].mean() if 'adr' in self.hotel_data.columns else 100.0
            occupancy = (1 - self.hotel_data['is_canceled'].mean()) * 100 if 'is_canceled' in self.hotel_data.columns else 85.0
        else:
            avg_price = 100.0
            occupancy = 85.0
        
        responses = {
            QueryIntent.BOOKING_INQUIRY: f"I'd be happy to help with your booking! Our average rate is ${avg_price:.2f} per night. What dates are you considering?",
            QueryIntent.CANCELLATION: "I can assist with cancellation. Please provide your booking confirmation number. Free cancellation is available up to 24 hours before arrival.",
            QueryIntent.ROOM_AVAILABILITY: f"We currently have {occupancy:.0f}% occupancy. What dates would you like to stay?",
            QueryIntent.PRICING: f"Our rooms start at ${avg_price*0.7:.2f} and average ${avg_price:.2f} per night. Would you like information about a specific room type?",
            QueryIntent.AMENITIES: "We offer complimentary WiFi, fitness center (24/7), business center, and on-site dining. What specific amenities interest you?",
            QueryIntent.CHECK_IN_OUT: "Check-in: 3:00 PM, Check-out: 11:00 AM. We can accommodate early check-in or late check-out based on availability.",
            QueryIntent.GENERAL_INFO: "I'm here to help! We're a full-service hotel with modern amenities. What would you like to know?",
            QueryIntent.COMPLAINT: "I sincerely apologize for any inconvenience. Your satisfaction is our priority. Please describe the issue in detail.",
            QueryIntent.SPECIAL_REQUESTS: "We're happy to accommodate special requests! What do you need?",
            QueryIntent.UNKNOWN: "I'd be happy to help! Could you provide more details? I can assist with bookings, cancellations, amenities, and more."
        }
        
        return responses.get(intent, "I'm here to assist you. What information do you need about our hotel services?")
    
    def get_real_performance_metrics(self) -> Dict:
        """Get actual measured performance metrics"""
        return self.performance_tracker.get_real_metrics()
    
    def print_detailed_metrics(self):
        """Print comprehensive performance analysis"""
        metrics = self.get_real_performance_metrics()
        
        if "error" in metrics:
            print(f"\n⚠️  {metrics['error']}")
            return
        
        print("\n" + "="*60)
        print("CHATBOT PERFORMANCE METRICS")
        print("="*60)
        
        print(f"\n📊 Query Processing:")
        print(f"   Total Queries: {metrics.get('total_queries', 0)}")
        print(f"   Avg Response Time: {metrics.get('average_response_time', 0):.2f}s")
        print(f"   Baseline Time: {metrics.get('baseline_response_time', 15):.1f}s")
        
        print(f"\n🚀 Performance Improvements:")
        print(f"   Response Time Reduction: {metrics.get('response_time_reduction_actual', 0):.1f}%")
        print(f"   Resolution Accuracy: {metrics.get('query_resolution_accuracy_actual', 0):.1f}%")
        print(f"   Avg Confidence: {metrics.get('average_confidence', 0):.3f}")
        
        # Intent distribution
        intent_counts = {}
        for query in self.queries:
            intent = query.intent.value
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        print(f"\n🎯 Intent Distribution:")
        for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {intent.replace('_', ' ').title():25}: {count} queries")
        
        # Resolution by intent
        resolved_by_intent = {}
        total_by_intent = {}
        for query in self.queries:
            intent = query.intent.value
            total_by_intent[intent] = total_by_intent.get(intent, 0) + 1
            if query.resolved:
                resolved_by_intent[intent] = resolved_by_intent.get(intent, 0) + 1
        
        print(f"\n✅ Resolution Rate by Intent:")
        for intent in sorted(total_by_intent.keys()):
            resolved = resolved_by_intent.get(intent, 0)
            total = total_by_intent[intent]
            rate = (resolved / total) * 100 if total > 0 else 0
            print(f"   {intent.replace('_', ' ').title():25}: {rate:5.1f}% ({resolved}/{total})")
    
    def stress_test(self, num_queries: int = 100):
        """Perform stress testing"""
        print(f"\n{'='*60}")
        print(f"🔥 STRESS TEST: {num_queries} queries")
        print("="*60)
        
        test_queries = [
            "I want to book a room for tonight",
            "What are your room rates?",
            "Cancel my reservation please",
            "Do you have wifi?",
            "What time is check-in?",
            "I need a room for 3 people",
            "Are pets allowed?",
            "I have a complaint",
            "Pool hours?",
            "Parking available?",
        ] * 15
        
        start_time = time.time()
        
        for i in range(num_queries):
            query = test_queries[i % len(test_queries)]
            self.process_query(query, f"stress_{i}")
            
            if (i + 1) % 25 == 0:
                print(f"   Progress: {i + 1}/{num_queries}...")
        
        total_time = time.time() - start_time
        qps = num_queries / total_time
        
        print(f"\n⚡ Results:")
        print(f"   Total: {num_queries} queries in {total_time:.2f}s")
        print(f"   Throughput: {qps:.1f} queries/second")
        print(f"   Avg per query: {(total_time/num_queries)*1000:.1f}ms")
        
        self.print_detailed_metrics()
    
    def save_conversation_history(self, filename: str = "hotel_chat_history.json"):
        """Save conversation history"""
        data = {
            "queries": [
                {
                    "text": q.text,
                    "intent": q.intent.value,
                    "entities": q.entities,
                    "response": q.response,
                    "timestamp": q.timestamp,
                    "response_time": q.response_time,
                    "resolved": q.resolved,
                    "confidence": q.confidence_score
                }
                for q in self.queries
            ],
            "metrics": self.get_real_performance_metrics(),
            "saved_at": datetime.datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ History saved to {filename}")

def main():
    """Main function with RAG evaluation"""
    print("="*60)
    print("Hotel Chatbot with RAG EVALUATION")
    print("="*60)
    
    chatbot = EnhancedHotelChatbot(use_kaggle_data=True)
    
    # Demo queries
    print("\n" + "="*60)
    print("📋 DEMO QUERIES")
    print("="*60)
    
    sample_queries = [
        "I want to book a room for next Friday",
        "What are your room rates?",
        "Do you have wifi?",
        "What time is check-in?",
    ]
    
    for query in sample_queries:
        print(f"\nCustomer: {query}")
        response = chatbot.process_query(query)
        print(f"Bot: {response}")
    
    # RAG Evaluation
    print("\n" + "="*60)
    print("🔍 RAG EVALUATION")
    print("="*60)
    
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
        {
            "question": "What are your average room rates?",
            "ground_truth": "Room rates vary by type and season, with an average around $100 per night."
        }
    ]
    
    chatbot.batch_evaluate_rag(test_cases)
    chatbot.print_rag_evaluation_summary()
    chatbot.export_rag_evaluations()
    
    # Stress test
    print("\n" + "="*60)
    print("Running stress test...")
    print("="*60)
    chatbot.stress_test(30)
    
    chatbot.save_conversation_history()
    
    print("\n✅ Demo Complete!")

if __name__ == "__main__":
    main()
