from datetime import datetime, timedelta
import random

class HotelData:
    """Manages all hotel-related data and business logic"""
    
    def __init__(self):
        self.hotel_info = {
            'name': 'The Royal Marina Hotel',
            'tagline': 'Luxury Waterfront Experience • 24/7 Concierge Service',
            'location': 'Downtown Waterfront District',
            'rating': '5-star luxury',
            'address': {
                'street': '123 Waterfront Boulevard',
                'city': 'Marina District',
                'zip': '12345'
            }
        }
        
        self.room_types = {
            'standard': {
                'name': 'Standard Marina View',
                'price': 299,
                'description': 'Elegant room with partial marina view',
                'max_guests': 2,
                'size': '350 sq ft',
                'amenities': ['Marina view', 'King bed', 'Work desk', 'WiFi']
            },
            'deluxe': {
                'name': 'Deluxe Waterfront Suite',
                'price': 499,
                'description': 'Spacious suite with full waterfront view',
                'max_guests': 4,
                'size': '600 sq ft',
                'amenities': ['Full waterfront view', 'Separate living area', 'Mini bar', 'Balcony']
            },
            'presidential': {
                'name': 'Presidential Marina Suite',
                'price': 899,
                'description': 'Luxury suite with private balcony and premium amenities',
                'max_guests': 6,
                'size': '1200 sq ft',
                'amenities': ['Private balcony', 'Butler service', 'Premium bath', 'Dining area']
            }
        }
        
        self.amenities = [
            'Infinity Pool', 'Spa & Wellness Center', 'Fine Dining Restaurant',
            'Fitness Center', 'Marina Access', 'Concierge Service',
            'Business Center', 'Valet Parking', 'Room Service', 'WiFi',
            'Pet-Friendly', 'Conference Rooms'
        ]
        
        self.policies = {
            'checkin': '3:00 PM',
            'checkout': '11:00 AM',
            'cancellation': '24 hours before arrival',
            'pets': 'Pets welcome with $50 fee',
            'parking': 'Valet parking $25/night',
            'wifi': 'Complimentary high-speed WiFi',
            'smoking': 'Non-smoking property'
        }
        
        self.contact_info = {
            'front_desk': '(555) 123-4567',
            'reservations': '(555) 123-4570',
            'concierge': '(555) 123-4575',
            'manager_duty': '(555) 123-4580',
            'email': 'info@royalmarina.com',
            'reservations_email': 'reservations@royalmarina.com',
            'service_email': 'service@royalmarina.com'
        }
        
        self.local_recommendations = {
            'restaurants': [
                {
                    'name': 'Marina Grill',
                    'distance': '2 min walk',
                    'cuisine': 'Seafood',
                    'description': 'Fresh seafood with waterfront views'
                },
                {
                    'name': 'The Golden Spoon',
                    'distance': '5 min walk',
                    'cuisine': 'French',
                    'description': 'Award-winning French cuisine'
                }
            ],
            'attractions': [
                {
                    'name': 'Marina Park & Boardwalk',
                    'distance': 'adjacent',
                    'description': 'Perfect for evening strolls'
                },
                {
                    'name': 'Aquarium & Maritime Museum',
                    'distance': '10 min walk',
                    'description': 'Family-friendly marine exhibits'
                },
                {
                    'name': 'Historic Downtown District',
                    'distance': '15 min walk',
                    'description': 'Shopping, dining, and entertainment'
                }
            ],
            'transportation': [
                'Airport shuttle service available',
                'Metro station (5 min walk)',
                'Taxi and rideshare readily available',
                'Bike rental on-site'
            ]
        }
    
    def check_availability(self, checkin_date, checkout_date, room_type):
        """Simulate room availability check"""
        # Simple simulation - assume 80% availability
        available = random.random() > 0.2
        
        if available:
            return {
                'available': True,
                'room': self.room_types[room_type],
                'message': 'Great! Your selected room is available.'
            }
        else:
            # Suggest alternative dates
            alt_date = checkin_date + timedelta(days=1)
            return {
                'available': False,
                'message': f'Sorry, that room type is not available for your dates. How about {alt_date.strftime("%B %d")}?',
                'alternative_date': alt_date.isoformat()
            }
    
    def calculate_total(self, room_type, checkin_date, checkout_date, guests=1):
        """Calculate total booking cost"""
        room = self.room_types[room_type]
        nights = (checkout_date - checkin_date).days
        
        room_total = room['price'] * nights
        taxes = room_total * 0.15  # 15% taxes
        
        # Additional fees
        extra_guest_fee = max(0, (guests - 2)) * 25 * nights  # $25 per extra guest per night
        
        return {
            'room_total': room_total,
            'taxes': taxes,
            'extra_guest_fee': extra_guest_fee,
            'grand_total': room_total + taxes + extra_guest_fee,
            'nights': nights,
            'nightly_rate': room['price']
        }
    
    def get_room_by_type(self, room_type):
        """Get room details by type"""
        return self.room_types.get(room_type)
    
    def format_amenities_list(self):
        """Format amenities for display"""
        return self.amenities
    
    def format_contact_info(self):
        """Format contact information for display"""
        return self.contact_info
    
    def get_local_recommendations_formatted(self):
        """Get formatted local recommendations"""
        return self.local_recommendations
