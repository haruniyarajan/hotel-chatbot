import re
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional

class ConversationSession:
    """Manages individual conversation sessions"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.current_flow = None
        self.user_data = {}
        self.booking_data = {}
        self.conversation_history = []
        self.context = {}
    
    def add_to_history(self, user_message: str, bot_response: str, intent: str):
        """Add conversation to history"""
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'bot_response': bot_response,
            'intent': intent
        })

class HotelChatbot:
    """Main chatbot engine for hotel customer service"""
    
    def __init__(self, hotel_data):
        self.hotel_data = hotel_data
        self.sessions = {}  # Store conversation sessions
        
        # Intent patterns for message classification
        self.intent_patterns = {
            'booking': [
                r'\b(book|reserve|reservation|stay|room for)\b',
                r'\b(make.*reservation|want.*room|need.*room)\b'
            ],
            'availability': [
                r'\b(available|check availability|free rooms|vacant)\b',
                r'\b(any rooms|rooms available)\b'
            ],
            'amenities': [
                r'\b(amenities|facilities|services|what do you have)\b',
                r'\b(pool|spa|gym|restaurant|wifi)\b'
            ],
            'rooms': [
                r'\b(room types|rooms|suites|accommodation)\b',
                r'\b(what.*rooms|room.*options)\b'
            ],
            'location': [
                r'\b(where|location|address|how to get|directions)\b',
                r'\b(find.*hotel|hotel.*located)\b'
            ],
            'policies': [
                r'\b(policy|rules|check in|check out|cancel)\b',
                r'\b(pet.*policy|smoking|parking)\b'
            ],
            'local': [
                r'\b(restaurant|attraction|nearby|recommend|area)\b',
                r'\b(things to do|places to visit|dining)\b'
            ],
            'existing_booking': [
                r'\b(my booking|my reservation|confirmation|change|modify)\b',
                r'\b(existing.*booking|booked.*room)\b'
            ],
            'complaint': [
                r'\b(problem|issue|complaint|wrong|dissatisfied)\b',
                r'\b(not happy|poor service|disappointed)\b'
            ],
            'contact': [
                r'\b(contact|phone|email|speak to|human|agent)\b',
                r'\b(call.*hotel|talk.*someone)\b'
            ],
            'greeting': [
                r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
                r'\b(greetings|howdy)\b'
            ],
            'thanks': [
                r'\b(thank|thanks|appreciate|grateful)\b'
            ]
        }
    
    def get_session(self, session_id: str) -> ConversationSession:
        """Get or create conversation session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(session_id)
        return self.sessions[session_id]
    
    def detect_intent(self, message: str) -> str:
        """Detect user intent from message"""
        message_lower = message.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return intent
        
        return 'general'
    
    def process_message(self, message: str, session_id: str) -> Dict:
        """Main message processing function"""
        session = self.get_session(session_id)
        intent = self.detect_intent(message)
        
        # Route to appropriate handler
        response = self.route_intent(intent, message, session)
        
        # Add to conversation history
        session.add_to_history(message, response.get('message', ''), intent)
        
        return response
    
    def route_intent(self, intent: str, message: str, session: ConversationSession) -> Dict:
        """Route message to appropriate handler based on intent"""
        
        handlers = {
            'greeting': self.handle_greeting,
            'booking': self.handle_booking_intent,
            'availability': self.handle_availability,
            'amenities': self.handle_amenities,
            'rooms': self.handle_room_types,
            'location': self.handle_location,
            'policies': self.handle_policies,
            'local': self.handle_local_recommendations,
            'existing_booking': self.handle_existing_booking,
            'complaint': self.handle_complaint,
            'contact': self.handle_contact,
            'thanks': self.handle_thanks,
            'general': self.handle_general_query
        }
        
        handler = handlers.get(intent, self.handle_general_query)
        return handler(message, session)
    
    def handle_greeting(self, message: str, session: ConversationSession) -> Dict:
        """Handle greeting messages"""
        import random
        responses = [
            f"Hello! Welcome to {self.hotel_data.hotel_info['name']}! 🏨 How can I assist you today?",
            f"Hi there! I'm here to help with your stay at {self.hotel_data.hotel_info['name']}. What can I do for you?",
            f"Good day! Welcome to {self.hotel_data.hotel_info['name']}. How may I help you today?"
        ]
        
        message_text = random.choice(responses)
        
        return {
            'message': message_text,
            'quick_replies': [
                'Make a Reservation',
                'Check Availability',
                'Hotel Amenities',
                'Local Recommendations'
            ],
            'type': 'text'
        }
    
    def handle_booking_intent(self, message: str, session: ConversationSession) -> Dict:
        """Handle booking-related requests"""
        session.current_flow = 'booking'
        
        return {
            'message': "I'd be delighted to help you make a reservation! Let me show you our booking form.",
            'type': 'booking_form',
            'form_data': {
                'room_types': self.hotel_data.room_types,
                'min_date': datetime.now().strftime('%Y-%m-%d'),
                'max_date': (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')
            }
        }
    
    def handle_amenities(self, message: str, session: ConversationSession) -> Dict:
        """Handle amenities inquiry"""
        amenities = self.hotel_data.format_amenities_list()
        
        return {
            'message': f"Here are all the wonderful amenities available at {self.hotel_data.hotel_info['name']}:",
            'type': 'amenities_list',
            'amenities': amenities,
            'quick_replies': [
                'Room Types',
                'Make Reservation',
                'Local Area Info'
            ]
        }
    
    def handle_room_types(self, message: str, session: ConversationSession) -> Dict:
        """Handle room types inquiry"""
        return {
            'message': "Here are our available room types and rates:",
            'type': 'room_types',
            'rooms': self.hotel_data.room_types,
            'quick_replies': [
                'Make Reservation',
                'Check Availability',
                'View Amenities'
            ]
        }
    
    def handle_availability(self, message: str, session: ConversationSession) -> Dict:
        """Handle availability check"""
        return {
            'message': "I'd be happy to check availability for you! What dates are you considering for your stay?",
            'type': 'text',
            'quick_replies': [
                'This Weekend',
                'Next Week',
                'Specific Dates',
                'Make Reservation'
            ]
        }
    
    def handle_policies(self, message: str, session: ConversationSession) -> Dict:
        """Handle hotel policies inquiry"""
        return {
            'message': "Here are our hotel policies and important information:",
            'type': 'policies',
            'policies': self.hotel_data.policies,
            'quick_replies': [
                'Make Reservation',
                'Contact Front Desk',
                'Local Recommendations'
            ]
        }
    
    def handle_location(self, message: str, session: ConversationSession) -> Dict:
        """Handle location and directions inquiry"""
        address = self.hotel_data.hotel_info['address']
        
        location_info = {
            'address': f"{address['street']}, {address['city']} {address['zip']}",
            'details': {
                'From Airport': '25 minutes',
                'To Downtown': '15-minute walk',
                'Metro Access': 'Marina Station (5 min)',
                'Parking': 'Valet service available'
            },
            'description': "Situated directly on the marina with stunning water views and easy access to the city's best attractions."
        }
        
        return {
            'message': "We're perfectly located in the heart of the waterfront district:",
            'type': 'location_info',
            'location': location_info,
            'quick_replies': [
                'Transportation Options',
                'Local Attractions',
                'Make Reservation'
            ]
        }
    
    def handle_local_recommendations(self, message: str, session: ConversationSession) -> Dict:
        """Handle local recommendations request"""
        recommendations = self.hotel_data.get_local_recommendations_formatted()
        
        return {
            'message': "Here are some fantastic local recommendations near our hotel:",
            'type': 'local_recommendations',
            'recommendations': recommendations,
            'quick_replies': [
                'Make Reservation',
                'Arrange Transportation',
                'More Information'
            ]
        }
    
    def handle_existing_booking(self, message: str, session: ConversationSession) -> Dict:
        """Handle existing booking inquiries"""
        contact = self.hotel_data.format_contact_info()
        
        return {
            'message': "I can help you with your existing booking! For security, I'll need to transfer you to our front desk team who can access your reservation details.",
            'type': 'escalation',
            'escalation_info': {
                'reason': 'secure_booking_management',
                'contact': {
                    'phone': contact['front_desk'],
                    'email': contact['reservations_email'],
                    'availability': '24/7'
                }
            },
            'quick_replies': [
                'Call Front Desk',
                'Email Support',
                'Make New Booking'
            ]
        }
    
    def handle_complaint(self, message: str, session: ConversationSession) -> Dict:
        """Handle complaints and issues"""
        contact = self.hotel_data.format_contact_info()
        
        return {
            'message': "I'm sorry to hear you're experiencing an issue. Customer satisfaction is our top priority, and I want to make sure this gets proper attention.",
            'type': 'escalation',
            'escalation_info': {
                'reason': 'priority_support',
                'contact': {
                    'guest_services': contact['front_desk'],
                    'manager_duty': contact['manager_duty'],
                    'email': contact['service_email']
                }
            },
            'quick_replies': [
                'Speak to Manager',
                'File Formal Complaint',
                'Call Front Desk'
            ]
        }
    
    def handle_contact(self, message: str, session: ConversationSession) -> Dict:
        """Handle contact information requests"""
        contact = self.hotel_data.format_contact_info()
        hotel_info = self.hotel_data.hotel_info
        
        contact_info = {
            'hotel_name': hotel_info['name'],
            'address': hotel_info['address'],
            'phones': {
                'Front Desk': contact['front_desk'],
                'Reservations': contact['reservations'],
                'Concierge': contact['concierge']
            },
            'email': contact['email'],
            'availability': '24/7'
        }
        
        return {
            'message': "Here's how to reach us:",
            'type': 'contact_info',
            'contact': contact_info,
            'quick_replies': [
                'Make Reservation',
                'Back to Main Menu',
                'Local Recommendations'
            ]
        }
    
    def handle_thanks(self, message: str, session: ConversationSession) -> Dict:
        """Handle thank you messages"""
        import random
        responses = [
            "You're very welcome! Is there anything else I can help you with?",
            "My pleasure! I'm here if you need any other assistance.",
            "You're welcome! Feel free to ask if you have any other questions."
        ]
        
        return {
            'message': random.choice(responses),
            'type': 'text',
            'quick_replies': [
                'Make Reservation',
                'Hotel Information',
                'Local Area'
            ]
        }
    
    def handle_general_query(self, message: str, session: ConversationSession) -> Dict:
        """Handle general or unclear queries"""
        import random
        responses = [
            "I'd be happy to help you with that! Could you please provide more details about what you're looking for?",
            "I want to make sure I give you the best assistance. Could you tell me more about what you need?",
            "Let me help you with that. What specific information would be most helpful?"
        ]
        
        return {
            'message': random.choice(responses),
            'type': 'text',
            'quick_replies': [
                'Make Reservation',
                'Hotel Information',
                'Local Area',
                'Contact Front Desk'
            ]
        }
    
    def process_booking_form(self, form_data: Dict, session_id: str) -> Dict:
        """Process booking form submission"""
        session = self.get_session(session_id)
        
        try:
            # Parse dates
            checkin_date = datetime.strptime(form_data['checkin_date'], '%Y-%m-%d')
            checkout_date = datetime.strptime(form_data['checkout_date'], '%Y-%m-%d')
            
            # Validate dates
            if checkout_date <= checkin_date:
                return {
                    'success': False,
                    'message': 'Check-out date must be after check-in date.'
                }
            
            room_type = form_data['room_type']
            guests = int(form_data['guests'])
            
            # Check availability
            availability = self.hotel_data.check_availability(checkin_date, checkout_date, room_type)
            
            if not availability['available']:
                return {
                    'success': False,
                    'message': availability['message']
                }
            
            # Calculate pricing
            pricing = self.hotel_data.calculate_total(room_type, checkin_date, checkout_date, guests)
            room = self.hotel_data.get_room_by_type(room_type)
            
            # Store booking data in session
            session.booking_data = {
                'checkin_date': checkin_date.isoformat(),
                'checkout_date': checkout_date.isoformat(),
                'room_type': room_type,
                'guests': guests,
                'pricing': pricing,
                'room': room
            }
            
            session.current_flow = 'booking_confirmation'
            
            return {
                'success': True,
                'message': 'Perfect! Here are your booking details:',
                'type': 'booking_confirmation',
                'booking_data': session.booking_data,
                'quick_replies': [
                    'Confirm Booking',
                    'Modify Dates',
                    'Choose Different Room'
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error processing booking: {str(e)}'
            }
    
    def handle_quick_reply(self, reply_text: str, session_id: str) -> Dict:
        """Handle quick reply button clicks"""
        session = self.get_session(session_id)
        
        # Map quick replies to appropriate handlers
        if reply_text == 'Make a Reservation' or reply_text == 'Make Reservation':
            return self.handle_booking_intent(reply_text, session)
        elif reply_text == 'Hotel Amenities' or reply_text == 'View Amenities':
            return self.handle_amenities(reply_text, session)
        elif reply_text == 'Check Availability':
            return self.handle_availability(reply_text, session)
        elif reply_text == 'Local Recommendations' or reply_text == 'Local Area Info':
            return self.handle_local_recommendations(reply_text, session)
        elif reply_text == 'Room Types':
            return self.handle_room_types(reply_text, session)
        elif reply_text == 'Contact Front Desk':
            return self.handle_contact(reply_text, session)
        elif reply_text == 'Confirm Booking':
            return self.confirm_booking(session)
        else:
            # Treat as regular message
            return self.process_message(reply_text, session_id)
    
    def confirm_booking(self, session: ConversationSession) -> Dict:
        """Confirm booking and generate confirmation"""
        if not session.booking_data:
            return {
                'success': False,
                'message': 'No booking data found. Please start a new reservation.'
            }
        
        # Generate confirmation number
        import random
        confirmation_number = f"RM{random.randint(100000, 999999)}"
        
        session.booking_data['confirmation_number'] = confirmation_number
        session.booking_data['status'] = 'confirmed'
        session.booking_data['booking_date'] = datetime.now().isoformat()
        
        return {
            'success': True,
            'message': '🎉 Booking Confirmed! Thank you for choosing The Royal Marina Hotel.',
            'type': 'booking_confirmed',
            'booking_data': session.booking_data,
            'confirmation_number': confirmation_number,
            'quick_replies': [
                'Email Confirmation',
                'Add to Calendar',
                'Contact Concierge'
            ]
        }
