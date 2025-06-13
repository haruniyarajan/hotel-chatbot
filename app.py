from flask import Flask, render_template, request, jsonify, session
from datetime import datetime, timedelta
import uuid
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Import our chatbot components
from chatbot_engine import HotelChatbot
from hotel_data import HotelData

# Initialize components
hotel_data = HotelData()
chatbot = HotelChatbot(hotel_data)

@app.route('/')
def index():
    """Main chat interface"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('chat.html')

@app.route('/api/message', methods=['POST'])
def handle_message():
    """Process user messages and return bot responses"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        session_id = session.get('session_id')
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Process message through chatbot
        response = chatbot.process_message(user_message, session_id)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/booking', methods=['POST'])
def process_booking():
    """Handle booking form submissions"""
    try:
        data = request.get_json()
        session_id = session.get('session_id')
        
        response = chatbot.process_booking_form(data, session_id)
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/quick_reply', methods=['POST'])
def handle_quick_reply():
    """Handle quick reply button clicks"""
    try:
        data = request.get_json()
        reply_text = data.get('reply')
        session_id = session.get('session_id')
        
        response = chatbot.handle_quick_reply(reply_text, session_id)
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=int(os.environ.get('PORT', 5000))
    )
