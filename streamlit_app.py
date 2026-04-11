import streamlit as st
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.linalg import norm
import cv2
import os
from PIL import Image
import json
from datetime import datetime
import tempfile

# Page configuration
st.set_page_config(
    page_title="StyleSync",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #9f678f;
        color: white;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: bold;
    }
    .product-card {
        background: purple;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .cart-item {
        background: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #ff4b4b;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'product_database' not in st.session_state:
    st.session_state.product_database = {}
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'uploaded_image_path' not in st.session_state:
    st.session_state.uploaded_image_path = None

# Product database with realistic Indian fashion items
def initialize_product_database():
    """Initialize product database with details"""
    if not st.session_state.product_database:
        # Sample product data - you can expand this
        products = {
            'saree': {'sizes': ['Free Size'], 'price_range': (1500, 5000)},
            'kurta': {'sizes': ['S', 'M', 'L', 'XL', 'XXL'], 'price_range': (800, 2500)},
            'shirt': {'sizes': ['S', 'M', 'L', 'XL', 'XXL'], 'price_range': (600, 2000)},
            'dress': {'sizes': ['S', 'M', 'L', 'XL'], 'price_range': (1000, 3500)},
            'jeans': {'sizes': ['28', '30', '32', '34', '36', '38'], 'price_range': (1200, 3000)},
            'tshirt': {'sizes': ['S', 'M', 'L', 'XL', 'XXL'], 'price_range': (400, 1200)},
            'lehenga': {'sizes': ['S', 'M', 'L', 'XL'], 'price_range': (3000, 15000)},
            'suit': {'sizes': ['S', 'M', 'L', 'XL', 'XXL'], 'price_range': (2500, 8000)},
        }
        
        # Generate random prices for each product
        for key in products:
            min_price, max_price = products[key]['price_range']
            products[key]['price'] = np.random.randint(min_price, max_price)
        
        st.session_state.product_database = products

def get_product_details(filename):
    """Extract product details from filename or database"""
    initialize_product_database()
    
    # Extract product type from filename
    basename = os.path.basename(filename).lower()
    product_type = 'fashion item'
    
    for key in st.session_state.product_database.keys():
        if key in basename:
            product_type = key.title()
            break
    
    # Get details or use defaults
    if product_type.lower() in st.session_state.product_database:
        details = st.session_state.product_database[product_type.lower()]
        return {
            'name': product_type,
            'sizes': details['sizes'],
            'price': details['price'],
            'filename': filename
        }
    else:
        return {
            'name': 'Fashion Item',
            'sizes': ['S', 'M', 'L', 'XL'],
            'price': np.random.randint(800, 2500),
            'filename': filename
        }

# Load model and features
@st.cache_resource
def load_model():
    """Load ResNet50 model - same as main.py"""
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = tensorflow.keras.Sequential([
        model,
        GlobalMaxPooling2D(),
    ])
    return model

@st.cache_data
def load_features():
    """Load precomputed features and filenames"""
    try:
        feature_list = pickle.load(open('embeddings.pkl', 'rb'))
        filenames = pickle.load(open('filenames.pkl', 'rb'))
        return feature_list, filenames
    except FileNotFoundError:
        st.error("⚠️ Feature files not found. Please run main.py to generate embeddings first.")
        return None, None

def extract_hsv_histogram(img_path):
    """Extract HSV color histogram from image - same as main.py"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    except Exception as e:
        st.error(f"Error extracting HSV histogram from {img_path}: {e}")
        return None

def extract_pattern_features(img_path, model):
    """Extract ResNet50 pattern features - same as main.py"""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img = np.expand_dims(img_array, axis=0)
        processed_img = preprocess_input(expanded_img)
        result = model.predict(processed_img, verbose=0).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        st.error(f"Error extracting pattern features: {e}")
        return None

def extract_features(img_path, model):
    """Extract combined features: 60% pattern (ResNet50) + 40% color (HSV histogram) - same as main.py"""
    try:
        pattern_feats = extract_pattern_features(img_path, model)
        if pattern_feats is None:
            return None
        
        color_feats = extract_hsv_histogram(img_path)
        if color_feats is None:
            return None
        
        # Normalize color features
        color_feats = color_feats / (norm(color_feats) + 1e-8)
        
        # Combine: 60% pattern + 40% color
        pattern_feats_weighted = pattern_feats * 0.6
        color_feats_weighted = color_feats * 0.4
        
        # Concatenate and normalize
        combined_feats = np.concatenate([pattern_feats_weighted, color_feats_weighted])
        normalized_combined = combined_feats / (norm(combined_feats) + 1e-8)
        
        return normalized_combined
    except Exception as e:
        st.error(f"Error extracting features from '{img_path}': {e}")
        return None

def recommend(features, feature_list, filenames, top_n=5):
    """Find top 5 most similar products using KNN"""
    neighbors = NearestNeighbors(n_neighbors=top_n, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices[0]  # Returns indices sorted by closest match first

def add_to_cart(product, size):
    """Add product to cart"""
    cart_item = {
        'product': product,
        'size': size,
        'quantity': 1,
        'added_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.cart.append(cart_item)
    st.success(f"✅ Added {product['name']} (Size: {size}) to cart!")

def remove_from_cart(index):
    """Remove item from cart"""
    if 0 <= index < len(st.session_state.cart):
        removed_item = st.session_state.cart.pop(index)
        st.success(f"🗑️ Removed {removed_item['product']['name']} from cart")

def calculate_total():
    """Calculate cart total"""
    return sum(item['product']['price'] * item['quantity'] for item in st.session_state.cart)

# Sidebar navigation
with st.sidebar:
    st.title("🛍️ Explore")
    
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = 'home'
        st.rerun()
    
    if st.button(f"🛒 Cart ({len(st.session_state.cart)})", use_container_width=True):
        st.session_state.page = 'cart'
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.info("Upload your fashion image and get recommendations!")

# Main content
if st.session_state.page == 'home':
    st.title("👗 StyleSync - Fashion Recommendation System")
    st.markdown("### Upload your fashion image to get personalized recommendations")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'webp'])
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Your Uploaded Image")
            img = Image.open(uploaded_file)
            st.image(img, use_container_width=True)
        
        with col2:
            st.markdown("#### Get Recommendations")
            if st.button("🔍 Find Similar Products", use_container_width=True):
                with st.spinner("🤖 Analyzing your image..."):
                    # Load model and features
                    model = load_model()
                    feature_list, filenames = load_features()
                    
                    if feature_list is not None and filenames is not None:
                        # Save uploaded image temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            # Ensure image mode is compatible with JPEG (remove alpha channel)
                            if img.mode == 'RGBA':
                                background = Image.new('RGB', img.size, (255, 255, 255))
                                background.paste(img, mask=img.split()[3])
                                img_to_save = background
                            elif img.mode in ('LA', 'P'):
                                img_to_save = img.convert('RGB')
                            elif img.mode != 'RGB':
                                img_to_save = img.convert('RGB')
                            else:
                                img_to_save = img
                            img_to_save.save(tmp_file.name, format='JPEG')
                            tmp_path = tmp_file.name
                        
                        try:
                            # Extract features from uploaded image using your model
                            features = extract_features(tmp_path, model)
                            
                            if features is not None:
                                # Get recommendations
                                indices = recommend(features, feature_list, filenames)
                                st.session_state.recommendations = [filenames[i] for i in indices]
                                st.success("✨ Found similar products!")
                            else:
                                st.error("Failed to process image")
                        finally:
                            # Clean up temp file
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
        
        # Display recommendations
        if st.session_state.recommendations:
            st.markdown("---")
            st.markdown("## 🎯 Top 5 Recommended Products")
            
            # Create grid layout
            cols_per_row = 5
            recommendations = st.session_state.recommendations
            
            for i in range(0, len(recommendations), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(recommendations):
                        with col:
                            img_path = recommendations[i + j]
                            if os.path.exists(img_path):
                                product = get_product_details(img_path)
                                
                                # Product image
                                st.image(img_path, use_container_width=True)
                                
                                # Product details
                                st.markdown(f"**{product['name']}**")
                                st.markdown(f"💰 ₹{product['price']:,}")
                                
                                # View details button
                                if st.button("View Details", key=f"view_{i+j}", use_container_width=True):
                                    st.session_state.selected_product = product
                                    st.session_state.page = 'product_details'
                                    st.rerun()

elif st.session_state.page == 'product_details':
    if 'selected_product' in st.session_state:
        product = st.session_state.selected_product
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(product['filename'], use_container_width=True)
        
        with col2:
            st.title(product['name'])
            st.markdown(f"## 💰 ₹{product['price']:,}")
            
            st.markdown("### Product Details")
            st.markdown(f"""
            - **Category:** {product['name']}
            - **Price:** ₹{product['price']:,}
            - **Available Sizes:** {', '.join(product['sizes'])}
            - **Material:** Premium Quality
            - **Care:** Machine Wash
            """)
            
            # Size Chart
            st.markdown("### 📏 Size Chart")
            with st.expander("View Size Guide", expanded=False):
                # Different size charts based on product type
                product_type = product['name'].lower()
                
                if product_type in ['kurta', 'shirt', 'dress', 'tshirt']:
                    st.markdown("""
                    **Top Wear Size Guide (in inches)**
                    
                    | Size | Chest | Waist | Length | Shoulder |
                    |------|-------|-------|--------|----------|
                    | S    | 36-38 | 30-32 | 27     | 15.5     |
                    | M    | 38-40 | 32-34 | 28     | 16       |
                    | L    | 40-42 | 34-36 | 29     | 16.5     |
                    | XL   | 42-44 | 36-38 | 30     | 17       |
                    | XXL  | 44-46 | 38-40 | 31     | 17.5     |
                    
                    **How to Measure:**
                    - **Chest:** Measure around the fullest part of your chest
                    - **Waist:** Measure around your natural waistline
                    - **Length:** Measure from shoulder to hem
                    - **Shoulder:** Measure from shoulder tip to shoulder tip
                    """)
                
                elif product_type in ['jeans', 'trousers', 'pants']:
                    st.markdown("""
                    **Bottom Wear Size Guide (in inches)**
                    
                    | Size | Waist | Hip   | Length | Thigh |
                    |------|-------|-------|--------|-------|
                    | 28   | 28-29 | 36-37 | 40     | 20    |
                    | 30   | 30-31 | 38-39 | 40     | 21    |
                    | 32   | 32-33 | 40-41 | 40     | 22    |
                    | 34   | 34-35 | 42-43 | 40     | 23    |
                    | 36   | 36-37 | 44-45 | 40     | 24    |
                    | 38   | 38-39 | 46-47 | 40     | 25    |
                    
                    **How to Measure:**
                    - **Waist:** Measure around your natural waistline
                    - **Hip:** Measure around the fullest part of your hips
                    - **Length:** Measure from waist to ankle
                    - **Thigh:** Measure around the fullest part of your thigh
                    """)
                
                elif product_type in ['lehenga', 'saree']:
                    st.markdown("""
                    **Ethnic Wear Size Guide (in inches)**
                    
                    | Size | Bust  | Waist | Hip   | Length |
                    |------|-------|-------|-------|--------|
                    | S    | 32-34 | 26-28 | 36-38 | 40     |
                    | M    | 34-36 | 28-30 | 38-40 | 42     |
                    | L    | 36-38 | 30-32 | 40-42 | 42     |
                    | XL   | 38-40 | 32-34 | 42-44 | 44     |
                    
                    **How to Measure:**
                    - **Bust:** Measure around the fullest part of your bust
                    - **Waist:** Measure around your natural waistline
                    - **Hip:** Measure around the fullest part of your hips
                    - **Length:** Measure from shoulder to hem
                    
                    **Note:** Blouses and cholis can be customized to your measurements
                    """)
                
                elif product_type == 'suit':
                    st.markdown("""
                    **Suit Size Guide (in inches)**
                    
                    | Size | Chest | Waist | Length | Sleeve |
                    |------|-------|-------|--------|--------|
                    | S    | 36-38 | 30-32 | 28     | 32     |
                    | M    | 38-40 | 32-34 | 29     | 33     |
                    | L    | 40-42 | 34-36 | 30     | 34     |
                    | XL   | 42-44 | 36-38 | 31     | 35     |
                    | XXL  | 44-46 | 38-40 | 32     | 36     |
                    
                    **How to Measure:**
                    - **Chest:** Measure around the fullest part of your chest
                    - **Waist:** Measure around your natural waistline
                    - **Length:** Measure from shoulder to hem
                    - **Sleeve:** Measure from shoulder to wrist
                    """)
                
                else:
                    st.markdown("""
                    **General Size Guide (in inches)**
                    
                    | Size | Chest/Bust | Waist | Hip   |
                    |------|------------|-------|-------|
                    | S    | 34-36      | 28-30 | 36-38 |
                    | M    | 36-38      | 30-32 | 38-40 |
                    | L    | 38-40      | 32-34 | 40-42 |
                    | XL   | 40-42      | 34-36 | 42-44 |
                    
                    **Tip:** If you're between sizes, we recommend sizing up for a comfortable fit.
                    """)
                
                st.info("💡 **Tip:** Measure yourself with a measuring tape for the most accurate fit!")
            
            # Size selection
            st.markdown("### Select Size")
            selected_size = st.selectbox("Choose your size", product['sizes'], key="size_selector")
            
            # Add to cart button
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🛒 Add to Cart", use_container_width=True, type="primary"):
                    add_to_cart(product, selected_size)
            with col_b:
                if st.button("⬅️ Back to Recommendations", use_container_width=True):
                    st.session_state.page = 'home'
                    st.rerun()

elif st.session_state.page == 'cart':
    st.title("🛒 Shopping Cart")
    
    if not st.session_state.cart:
        st.info("Your cart is empty. Start shopping to add items!")
        if st.button("🏠 Go to Home"):
            st.session_state.page = 'home'
            st.rerun()
    else:
        # Display cart items
        for idx, item in enumerate(st.session_state.cart):
            product = item['product']
            
            col1, col2, col3, col4, col5 = st.columns([2, 3, 2, 2, 1])
            
            with col1:
                if os.path.exists(product['filename']):
                    st.image(product['filename'], width=100)
            
            with col2:
                st.markdown(f"**{product['name']}**")
                st.caption(f"Size: {item['size']}")
            
            with col3:
                st.markdown(f"₹{product['price']:,}")
            
            with col4:
                quantity = st.number_input("Qty", min_value=1, max_value=10, value=item['quantity'], key=f"qty_{idx}")
                st.session_state.cart[idx]['quantity'] = quantity
            
            with col5:
                if st.button("🗑️", key=f"remove_{idx}"):
                    remove_from_cart(idx)
                    st.rerun()
            
            st.markdown("---")
        
        # Cart summary
        st.markdown("### 📊 Order Summary")
        total = calculate_total()
        
        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown(f"**Subtotal:** ₹{total:,}")
            st.markdown(f"**Shipping:** ₹{100 if total < 1000 else 0:,}")
            st.markdown(f"### **Total:** ₹{total + (100 if total < 1000 else 0):,}")
        
        st.markdown("---")
        
        # Checkout button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("💳 Proceed to Payment", use_container_width=True, type="primary"):
                st.session_state.page = 'payment'
                st.rerun()

elif st.session_state.page == 'payment':
    st.title("💳 Payment")
    
    total = calculate_total()
    delivery_charge = 100 if total < 1000 else 0
    final_total = total + delivery_charge
    
    st.markdown(f"### Total Amount: ₹{final_total:,}")
    
    # Payment form
    with st.form("payment_form"):
        st.markdown("#### Customer Details")
        name = st.text_input("Full Name*")
        email = st.text_input("Email*")
        phone = st.text_input("Phone Number*")
        
        st.markdown("#### Shipping Address")
        address = st.text_area("Address*")
        city = st.text_input("City*")
        state = st.text_input("State*")
        pincode = st.text_input("Pincode*")
        
        st.markdown("#### Payment Method")
        payment_method = st.radio("Select Payment Method", 
                                  ["💳 Credit/Debit Card", "📱 UPI", "💵 Cash on Delivery"])
        
        if payment_method == "💳 Credit/Debit Card":
            card_number = st.text_input("Card Number")
            col1, col2 = st.columns(2)
            with col1:
                expiry = st.text_input("Expiry (MM/YY)")
            with col2:
                cvv = st.text_input("CVV", type="password")
        
        elif payment_method == "📱 UPI":
            upi_id = st.text_input("UPI ID")
        
        submitted = st.form_submit_button("✅ Place Order", use_container_width=True, type="primary")
    
    # Handle form submission outside the form
    if submitted:
        if name and email and phone and address and city and state and pincode:
            st.success("🎉 Order placed successfully!")
            st.balloons()
            
            # Order confirmation
            st.markdown("---")
            st.markdown("### 📧 Order Confirmation")
            order_id = f"ORD{np.random.randint(10000, 99999)}"
            st.info(f"""
            **Order ID:** {order_id}
            
            **Delivery Address:**
            {name}
            {address}
            {city}, {state} - {pincode}
            
            **Contact:** {phone}
            **Email:** {email}
            
            **Total Amount:** ₹{final_total:,}
            **Payment Method:** {payment_method}
            
            **Estimated Delivery:** 5-7 business days
            """)
            
            # Set flag to show continue shopping button
            st.session_state.order_placed = True
        else:
            st.error("Please fill all required fields")
    
    # Show continue shopping button after order is placed
    if 'order_placed' in st.session_state and st.session_state.order_placed:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🛍️ Continue Shopping", use_container_width=True, type="primary"):
                st.session_state.cart = []
                st.session_state.page = 'home'
                st.session_state.order_placed = False
                st.rerun()
    
    # Back to cart button (only show if order not placed)
    if 'order_placed' not in st.session_state or not st.session_state.order_placed:
        if st.button("⬅️ Back to Cart"):
            st.session_state.page = 'cart'
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Made with ❤️ </p>
    <p>© 2026 StyleSync. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
