from flask import render_template, request, redirect, url_for, flash, Blueprint
from flask_login import login_user, logout_user, login_required, current_user
from app.forms import RegisterForm, LoginForm
from app.extensions import db, mail
from app.models import User, CareerResult, ResumeFeedback, PersonalityResult
from app.ai_module import get_career_recommendation
from flask_mail import Message
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from app.groq_utils import groq_client 
import fitz  # For PDF handling
import os
from groq import Groq
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from PIL import Image, ImageStat, ImageFilter, ImageEnhance, ImageDraw
import math
import numpy as np
from collections import Counter
import cv2
import base64
from io import BytesIO
import pytesseract  # For OCR text extraction

# Blueprint setup
main = Blueprint('main', __name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx', 'png', 'jpg', 'jpeg'}

# Initialize Groq client with API key from environment or use a placeholder
# Check for Groq API key in environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
print(f"DEBUG - GROQ API KEY IS {'SET' if groq_api_key else 'NOT SET'}")

# Initialize Groq client if API key is available, otherwise set to None
if groq_api_key:
    try:
        groq_client = Groq(api_key=groq_api_key)
        print("DEBUG - GROQ CLIENT INITIALIZED SUCCESSFULLY")
    except Exception as e:
        print(f"DEBUG - FAILED TO INITIALIZE GROQ CLIENT: {str(e)}")
        groq_client = None
else:
    print("DEBUG - NO GROQ API KEY, USING FALLBACK RESPONSES")
    groq_client = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------- Home and Error Routes --------------------

@main.route('/')
def home():
    return render_template('index.html')

@main.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

# -------------------- File Upload Routes --------------------

@main.route('/classify-image', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('main.dashboard'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('main.dashboard'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        try:
            # Use AI for classification
            classification_result = classify_tool_or_skill(filepath)
            career_suggestions = get_career_suggestions(classification_result)

            return render_template('classification_result.html', classification=classification_result, suggestions=career_suggestions)
        except Exception as e:
            flash(f'Error processing image: {str(e)}', 'danger')
            return redirect(url_for('main.dashboard'))
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        flash('Invalid file type. Please upload an image.', 'danger')
        return redirect(url_for('main.dashboard'))

@main.route('/analyze-resume', methods=['POST', 'GET'])
def analyze_resume():
    # Handle direct access to the page (GET request)
    if request.method == 'GET':
        # Return the template with example data
        example_analysis = "This is an example resume analysis. Upload your resume to get personalized feedback."
        example_suggestions = ["Software Developer", "Data Analyst", "Project Manager"]
        
        # Create example job postings for direct access
        example_job_postings = [
            {
                'title': "Software Developer",
                'company': "CareerPath AI",
                'location': "Remote",
                'description': "We're looking for a talented developer to join our team. This role involves working with modern web technologies and helping professionals advance their careers.",
                'url': "https://www.linkedin.com/jobs/search/?keywords=Software%20Developer&location=Remote",
                'date_posted': datetime.now().strftime("%Y-%m-%d")
            },
            {
                'title': "Data Analyst",
                'company': "Analytics Inc",
                'location': "New York, NY",
                'description': "A top financial firm is looking for a Data Analyst to join their team. Experience with SQL and data visualization required.",
                'url': "https://www.indeed.com/jobs?q=Data+Analyst&l=New+York",
                'date_posted': datetime.now().strftime("%Y-%m-%d")
            }
        ]
        
        return render_template('cv_analysis_result.html', 
                              analysis=example_analysis, 
                              suggestions=example_suggestions,
                              job_postings=example_job_postings)
    
    # Handle POST request (resume upload)
    try:
        if 'resume_file' not in request.files:
            flash('No file part', 'danger')
            return redirect(url_for('main.dashboard'))
        
        file = request.files['resume_file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(url_for('main.dashboard'))
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            
            # Ensure uploads directory exists
            os.makedirs('uploads', exist_ok=True)
            file.save(filepath)
            
            try:
                text = ""
                # Check file type and extract text accordingly
                if filename.lower().endswith('.pdf'):
                    try:
                        with fitz.open(filepath) as pdf:
                            for page in pdf:
                                text += page.get_text()
                    except Exception as e:
                        flash(f'Error reading PDF: {str(e)}', 'danger')
                        return redirect(url_for('main.dashboard'))
                else:  # For text files
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                    except Exception as e:
                        flash(f'Error reading file: {str(e)}', 'danger')
                        return redirect(url_for('main.dashboard'))
                
                if not text or len(text) < 10:
                    flash('Could not extract meaningful text from the file', 'danger')
                    return redirect(url_for('main.dashboard'))
                
                # Use AI logic for CV analysis
                analysis_result = analyze_resume(text)
                
                # Debug the skills found
                print(f"DEBUG - Skills found: {analysis_result['skills']}")
                
                # Find job postings based on skills - ensure we're passing a list
                job_postings = find_job_postings(analysis_result["skills"])
                
                # Get career suggestions
                career_suggestions = get_career_recommendation(analysis_result["skills"])
                
                # Store feedback in database if user is logged in
                if current_user.is_authenticated:
                    try:
                        feedback = ResumeFeedback(
                            user_id=current_user.id,
                            feedback=analysis_result["recommendation"]
                        )
                        db.session.add(feedback)
                        db.session.commit()
                    except Exception as e:
                        # Don't fail if database storage fails
                        print(f"Error storing feedback: {str(e)}")
                
                # Clean up file after processing
                if os.path.exists(filepath):
                    os.remove(filepath)
                
                # Debug print statements
                print("DEBUG - Job Postings:", job_postings)
                print("DEBUG - Career Suggestions:", career_suggestions)
                
                # More detailed debug information
                print("DEBUG - Job Postings Type:", type(job_postings))
                print("DEBUG - Job Postings Length:", len(job_postings) if job_postings else 0)
                print("DEBUG - First Job Posting (if any):", job_postings[0] if job_postings and len(job_postings) > 0 else "None")
                print("DEBUG - Rendering template with job_postings:", bool(job_postings))
                
                # Now pass the job_postings to the template
                return render_template('cv_analysis_result.html', 
                                      analysis=analysis_result["recommendation"], 
                                      suggestions=career_suggestions,
                                      job_postings=job_postings)
            except Exception as e:
                print(f"Error in resume analysis: {str(e)}")
                flash(f'Error processing file: {str(e)}', 'danger')
                # Clean up file in case of error
                if os.path.exists(filepath):
                    os.remove(filepath)
                return redirect(url_for('main.dashboard'))
        else:
            flash('Invalid file type. Please upload a PDF or text file.', 'danger')
            return redirect(url_for('main.dashboard'))
    except Exception as e:
        print(f"Exception in analyze_resume route: {str(e)}")
        flash(f'Unexpected error: {str(e)}', 'danger')
        return redirect(url_for('main.dashboard'))

# -------------------- Helper Functions --------------------

@main.route('/ask', methods=['POST'])
def ask():
    """
    Handles career-related questions from users
    """
    if request.method == "POST":
        question = request.form.get("user_question")
        
        if not question:
            flash('Please enter a question', 'danger')
            return redirect(url_for('main.dashboard'))

        try:
            # Provide a fallback response if GROQ API key is not set or client initialization failed
            if groq_client is None:
                print("DEBUG - Using fallback response (no Groq client)")
                raw_answer = "I'm a career advisor AI that can help with job recommendations, skill assessments, resume feedback, and industry trends. What would you like to know? (Note: Advanced AI responses are currently unavailable, please try again later.)"
            else:
                # Try to use GROQ API
                try:
                    print("DEBUG - Attempting to call Groq API")
                    response = groq_client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[
                            {"role": "system", "content": "You are a helpful career advisor. Provide concise, practical career advice. Format your answers with proper paragraphs and bullet points for readability."},
                            {"role": "user", "content": question}
                        ]
                    )
                    raw_answer = response.choices[0].message.content
                    print("DEBUG - Groq API response received")
                except Exception as e:
                    # Fallback if API call fails
                    print(f"DEBUG - GROQ API error: {str(e)}")
                    raw_answer = f"Based on your question about '{question}', I recommend researching the field and connecting with professionals. Consider online courses to build relevant skills. (Note: Our AI service is experiencing issues, this is a fallback response.)"
            
            # Format the answer for better display
            answer = format_chatbot_response(raw_answer)
            print(f"DEBUG - Formatted answer: {answer[:100]}...")
            
            # Return the template with the response
            return render_template("dashboard.html", 
                                  chatbot_reply=answer,
                                  user_question=question,
                                  now=datetime.now())  # Add datetime for proper greeting
        except Exception as e:
            print(f"DEBUG - Error in /ask route: {str(e)}")
            flash(f'Error processing question: {str(e)}', 'danger')
            return redirect(url_for('main.dashboard'))
    
    return redirect(url_for('main.dashboard'))

# Helper function to format chatbot responses
def format_chatbot_response(text):
    """
    Formats the chatbot response for better readability
    """
    # Clean up excess whitespace
    text = text.strip()
    
    # Replace * bullet points with proper HTML bullet points
    text = text.replace('* ', '• ')
    
    # Add line breaks for readability if they don't exist
    if '\n' not in text and len(text) > 150:
        # Try to break at sentence boundaries
        sentences = text.split('. ')
        formatted_text = ''
        current_paragraph = ''
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            if len(current_paragraph) > 120:
                formatted_text += current_paragraph.strip() + ('' if current_paragraph.endswith('.') else '.') + '\n\n'
                current_paragraph = sentence + '. '
            else:
                current_paragraph += sentence + '. '
        
        if current_paragraph:
            formatted_text += current_paragraph
        
        text = formatted_text
    
    # Handle lists and format them nicely
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Check for numbered lists (1., 2., etc.)
        if line.strip() and line.strip()[0].isdigit() and '. ' in line[:5]:
            line = '<strong>' + line[:line.find('.')+1] + '</strong>' + line[line.find('.')+1:]
            
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

# -------------------- Groq AI Integration --------------------

def extract_text_from_image(image):
    """
    Extract text from an image using Tesseract OCR
    """
    try:
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to preprocess the image
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Apply dilation to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilation = cv2.dilate(threshold, kernel, iterations=1)
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(dilation)
        
        # Clean up the extracted text
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        
        return text
    except Exception as e:
        print(f"Error in text extraction: {str(e)}")
        return ""

def classify_tool_or_skill(filepath):
    """
    Enhanced logo classification function using computer vision, OCR, and Groq AI
    """
    try:
        # Import libraries for image processing
        import cv2
        import numpy as np
        from PIL import Image
        import math
        import base64
        from io import BytesIO
        
        # Read the image with OpenCV
        img_cv = cv2.imread(filepath)
        if img_cv is None:
            # Fallback to PIL if OpenCV fails
            img_pil = Image.open(filepath).convert('RGB')
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Extract text from the image
        extracted_text = extract_text_from_image(img_cv)
        print(f"DEBUG: Extracted text: {extracted_text}")
        
        # Get image dimensions
        height, width = img_cv.shape[:2]
        aspect_ratio = width / height
        is_square = 0.8 < aspect_ratio < 1.2
        
        # Convert to different color spaces for analysis
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Resize for consistent analysis
        max_size = 200
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        img_resized = cv2.resize(img_cv, (new_width, new_height))
        img_gray_resized = cv2.resize(img_gray, (new_width, new_height))
        
        # Extract color features
        avg_color_per_row = np.average(img_rgb, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        r_mean, g_mean, b_mean = avg_color
        
        # Calculate brightness and color ratios
        brightness = (r_mean + g_mean + b_mean) / 3
        r_ratio = r_mean / (brightness + 0.1)
        g_ratio = g_mean / (brightness + 0.1)
        b_ratio = b_mean / (brightness + 0.1)
        
        # Calculate color variance
        color_std = np.std(img_rgb.reshape(-1, 3), axis=0)
        r_std, g_std, b_std = color_std
        color_variance = (r_std**2 + g_std**2 + b_std**2) / 3
        
        # Edge detection for shape analysis
        edges = cv2.Canny(img_gray_resized, 100, 200)
        edge_intensity = np.mean(edges)
        
        # Detect circles using Hough transform
        circles = cv2.HoughCircles(
            img_gray_resized, 
            cv2.HOUGH_GRADIENT, 
            1, 
            minDist=max(new_width, new_height)//4,
            param1=100, 
            param2=30, 
            minRadius=max(new_width, new_height)//8,
            maxRadius=max(new_width, new_height)//2
        )
        has_circles = circles is not None
        
        # Detect text-like features
        has_text = bool(extracted_text.strip())  # Use OCR result instead of line detection
        
        # Enhanced feature detection
        def detect_dominant_color(img, threshold=0.3):
            """Detect dominant color in the image"""
            pixels = img.reshape(-1, 3)
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            total_pixels = len(pixels)
            dominant_colors = []
            
            for color, count in zip(unique_colors, counts):
                if count / total_pixels > threshold:
                    dominant_colors.append(color)
            
            return dominant_colors
        
        def detect_color_patterns(img):
            """Detect specific color patterns"""
            patterns = {
                'gradient': False,
                'solid': False,
                'multi_color': False
            }
            
            # Check for gradients
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h_std = np.std(hsv[:,:,0])
            s_std = np.std(hsv[:,:,1])
            v_std = np.std(hsv[:,:,2])
            
            patterns['gradient'] = h_std > 30 or s_std > 30 or v_std > 30
            
            # Check for solid colors
            color_std = np.std(img, axis=(0,1))
            patterns['solid'] = np.all(color_std < 20)
            
            # Check for multiple colors
            unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))
            patterns['multi_color'] = unique_colors > 10
            
            return patterns
        
        def detect_shape_features(img):
            """Detect shape features"""
            features = {
                'symmetrical': False,
                'angular': False,
                'curved': False
            }
            
            # Convert to binary
            _, binary = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                main_contour = max(contours, key=cv2.contourArea)
                
                # Check symmetry
                moments = cv2.moments(main_contour)
                if moments['m00'] != 0:
                    cx = int(moments['m10']/moments['m00'])
                    cy = int(moments['m01']/moments['m00'])
                    
                    # Check if the shape is symmetrical around center
                    left = main_contour[main_contour[:,:,0] < cx]
                    right = main_contour[main_contour[:,:,0] > cx]
                    features['symmetrical'] = abs(len(left) - len(right)) < 50
                
                # Check for angular features
                epsilon = 0.02 * cv2.arcLength(main_contour, True)
                approx = cv2.approxPolyDP(main_contour, epsilon, True)
                features['angular'] = len(approx) > 4
                
                # Check for curved features
                features['curved'] = cv2.arcLength(main_contour, True) > 1.5 * cv2.arcLength(approx, True)
            
            return features
        
        # Get enhanced features
        dominant_colors = detect_dominant_color(img_rgb)
        color_patterns = detect_color_patterns(img_rgb)
        shape_features = detect_shape_features(img_rgb)
        
        # Convert image to base64 for Groq AI analysis
        pil_image = Image.fromarray(img_rgb)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare feature description for Groq AI
        feature_description = {
            'colors': {
                'dominant_colors': [color.tolist() for color in dominant_colors],
                'patterns': color_patterns,
                'brightness': float(brightness),
                'color_variance': float(color_variance)
            },
            'shapes': {
                'is_square': bool(is_square),
                'aspect_ratio': float(aspect_ratio),
                'has_circles': bool(has_circles),
                'has_text': bool(has_text),
                'features': shape_features
            },
            'edges': {
                'intensity': float(edge_intensity)
            },
            'extracted_text': extracted_text
        }
        
        # Use Groq AI for classification if available
        if groq_client:
            try:
                # Create a prompt for Groq AI
                prompt = f"""
                Analyze this technology logo image and identify the technology it represents.
                Here are the detected features:
                - Colors: {feature_description['colors']}
                - Shapes: {feature_description['shapes']}
                - Edge Analysis: {feature_description['edges']}
                - Extracted Text: "{feature_description['extracted_text']}"
                
                Based on these features and your knowledge of technology logos, identify the most likely technology.
                Consider:
                1. Color schemes and patterns
                2. Shape characteristics
                3. Text presence and content
                4. Overall design elements
                
                Return only the name of the technology, nothing else.
                """
                
                # Call Groq AI
                response = groq_client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {"role": "system", "content": "You are an expert in technology logo recognition. Your task is to identify technology logos based on their visual features and any text they contain."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Get AI's classification
                ai_classification = response.choices[0].message.content.strip()
                
                # If AI provides a confident classification, use it
                if ai_classification and ai_classification != "Technology Logo":
                    return ai_classification
                
            except Exception as e:
                print(f"Error in Groq AI classification: {str(e)}")
        
        # Fallback to traditional classification if Groq AI fails or is unavailable
        scores = {
            # Programming Languages
            "JavaScript": 0,
            "Python": 0,
            "Java": 0,
            "TypeScript": 0,
            "C++": 0,
            "Ruby": 0,
            "Go": 0,
            "Rust": 0,
            "Swift": 0,
            "PHP": 0,
            
            # Frontend Frameworks
            "React": 0,
            "Angular": 0,
            "Vue.js": 0,
            "Svelte": 0,
            "Next.js": 0,
            
            # Backend Frameworks
            "Node.js": 0,
            "Django": 0,
            "Flask": 0,
            "Express.js": 0,
            "Spring": 0,
            
            # Databases
            "MongoDB": 0,
            "MySQL": 0,
            "PostgreSQL": 0,
            "Redis": 0,
            "Cassandra": 0,
            
            # Cloud & DevOps
            "Docker": 0,
            "Kubernetes": 0,
            "AWS": 0,
            "Azure": 0,
            "GCP": 0,
            
            # Version Control
            "Git": 0,
            "GitHub": 0,
            "GitLab": 0,
            "Bitbucket": 0,
            
            # IDEs & Editors
            "Visual Studio Code": 0,
            "IntelliJ IDEA": 0,
            "Eclipse": 0,
            "Sublime Text": 0,
            "Atom": 0,
            
            # Design Tools
            "Adobe Photoshop": 0,
            "Adobe Illustrator": 0,
            "Figma": 0,
            "Sketch": 0,
            "InVision": 0,
            
            # Social Media
            "LinkedIn": 0,
            "Twitter": 0,
            "Facebook": 0,
            "Instagram": 0,
            "YouTube": 0,
            
            # Productivity
            "Microsoft Excel": 0,
            "Microsoft Word": 0,
            "Microsoft PowerPoint": 0,
            "Google Docs": 0,
            "Slack": 0,
            
            # AI & ML
            "TensorFlow": 0,
            "PyTorch": 0,
            "Scikit-learn": 0,
            "OpenAI": 0,
            "Hugging Face": 0
        }
        
        # Score technologies based on enhanced features and extracted text
        # JavaScript (yellow square with JS)
        if any(np.all(color > [180, 180, 100]) for color in dominant_colors):
            scores["JavaScript"] += 5
            if is_square:
                scores["JavaScript"] += 3
            if "js" in extracted_text.lower() or "javascript" in extracted_text.lower():
                scores["JavaScript"] += 4
            if shape_features['angular']:
                scores["JavaScript"] += 2
        
        # React (blue circular atom)
        if any(np.all(color > [100, 100, 180]) for color in dominant_colors):
            scores["React"] += 5
            if has_circles:
                scores["React"] += 3
            if "react" in extracted_text.lower():
                scores["React"] += 4
            if shape_features['curved']:
                scores["React"] += 2
            if color_patterns['gradient']:
                scores["React"] += 2
        
        # Python (blue and yellow snake)
        if any(np.all(color > [100, 100, 180]) for color in dominant_colors) and \
           any(np.all(color > [180, 180, 100]) for color in dominant_colors):
            scores["Python"] += 5
            if shape_features['curved']:
                scores["Python"] += 3
            if "python" in extracted_text.lower():
                scores["Python"] += 4
            if color_patterns['multi_color']:
                scores["Python"] += 2
        
        # MongoDB (green leaf)
        if any(np.all(color > [100, 180, 100]) for color in dominant_colors):
            scores["MongoDB"] += 5
            if shape_features['curved']:
                scores["MongoDB"] += 3
            if "mongo" in extracted_text.lower():
                scores["MongoDB"] += 4
            if not is_square:
                scores["MongoDB"] += 2
        
        # LinkedIn (blue square with "in")
        if any(np.all(color > [100, 100, 180]) for color in dominant_colors):
            scores["LinkedIn"] += 5
            if is_square:
                scores["LinkedIn"] += 3
            if "in" in extracted_text.lower() or "linkedin" in extracted_text.lower():
                scores["LinkedIn"] += 4
            if shape_features['angular']:
                scores["LinkedIn"] += 2
        
        # YouTube (red play button)
        if any(np.all(color > [180, 100, 100]) for color in dominant_colors):
            scores["YouTube"] += 5
            if has_circles:
                scores["YouTube"] += 3
            if "you" in extracted_text.lower() or "tube" in extracted_text.lower():
                scores["YouTube"] += 4
            if shape_features['curved']:
                scores["YouTube"] += 2
        
        # Adobe Photoshop (blue square with Ps)
        if any(np.all(color > [100, 100, 180]) for color in dominant_colors):
            scores["Adobe Photoshop"] += 5
            if is_square:
                scores["Adobe Photoshop"] += 3
            if "ps" in extracted_text.lower() or "photoshop" in extracted_text.lower():
                scores["Adobe Photoshop"] += 4
            if color_patterns['gradient']:
                scores["Adobe Photoshop"] += 2
        
        # Get the technology with the highest score
        max_tech = max(scores.items(), key=lambda x: x[1])
        
        # Print top 3 scores for debugging
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"DEBUG: Top 3 technology scores: {top_scores}")
        
        # If we have a confident match (score >= 5)
        if max_tech[1] >= 5:
            return max_tech[0]
        
        # If we have a somewhat confident match (score >= 3)
        if max_tech[1] >= 3:
            # Check if second best is very close
            next_best = sorted(scores.items(), key=lambda x: x[1], reverse=True)[1]
            if abs(max_tech[1] - next_best[1]) < 1:
                # Too close to call, use advanced disambiguation
                return disambiguate_similar_techs(max_tech[0], next_best[0], img_rgb, has_circles, r_ratio, g_ratio, b_ratio, r_mean, g_mean, b_mean)
            return max_tech[0]
        
        # If no confident match, return generic response
        return "Technology Logo"
        
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return "Technology Logo"

def disambiguate_similar_techs(tech1, tech2, img, has_circles, r_ratio, g_ratio, b_ratio, r_mean, g_mean, b_mean):
    """
    Advanced disambiguation between similar technologies
    """
    # JavaScript vs React
    if tech1 in ["JavaScript", "React"] and tech2 in ["JavaScript", "React"]:
        js_score = 0
        if r_mean > 180 and g_mean > 180 and b_mean < 100:  # Strong yellow
            js_score += 3
        if not has_circles:
            js_score += 2
        if r_ratio > 0.4 and g_ratio > 0.4:
            js_score += 2
        
        react_score = 0
        if has_circles:
            react_score += 3
        if b_ratio > 0.4:
            react_score += 2
        if r_mean < 150 and g_mean < 150:
            react_score += 2
        
        return "JavaScript" if js_score > react_score else "React"
    
    # Angular vs Vue.js
    if tech1 in ["Angular", "Vue.js"] and tech2 in ["Angular", "Vue.js"]:
        angular_score = 0
        if r_ratio > 0.5 and r_mean > 150:
            angular_score += 3
        if is_square:
            angular_score += 2
        
        vue_score = 0
        if g_ratio > 0.4 and b_ratio > 0.3:
            vue_score += 3
        if color_variance > 500:
            vue_score += 2
        
        return "Angular" if angular_score > vue_score else "Vue.js"
    
    # MongoDB vs MySQL
    if tech1 in ["MongoDB", "MySQL"] and tech2 in ["MongoDB", "MySQL"]:
        mongodb_score = 0
        if g_ratio > 0.5 and brightness < 200:
            mongodb_score += 3
        if not is_square:
            mongodb_score += 2
        if color_variance > 400:
            mongodb_score += 2
        
        mysql_score = 0
        if b_ratio > 0.4 and color_variance > 800:
            mysql_score += 3
        if r_ratio > 0.25:
            mysql_score += 2
        if not is_square:
            mysql_score += 2
        
        return "MongoDB" if mongodb_score > mysql_score else "MySQL"
    
    # LinkedIn vs Twitter
    if tech1 in ["LinkedIn", "Twitter"] and tech2 in ["LinkedIn", "Twitter"]:
        linkedin_score = 0
        if b_ratio > 0.5 and is_square:
            linkedin_score += 3
        if has_text:
            linkedin_score += 2
        if brightness > 120:
            linkedin_score += 2
        
        twitter_score = 0
        if b_ratio > 0.5 and not is_square:
            twitter_score += 3
        if brightness > 120:
            twitter_score += 2
        if edge_intensity > 30:
            twitter_score += 2
        
        return "LinkedIn" if linkedin_score > twitter_score else "Twitter"
    
    # Adobe Photoshop vs Figma
    if tech1 in ["Adobe Photoshop", "Figma"] and tech2 in ["Adobe Photoshop", "Figma"]:
        photoshop_score = 0
        if b_ratio > 0.5 and brightness < 120:
            photoshop_score += 3
        if is_square and has_text:
            photoshop_score += 2
        if color_variance > 400:
            photoshop_score += 2
        
        figma_score = 0
        if b_ratio > 0.4 and r_ratio > 0.3:
            figma_score += 3
        if color_variance > 600:
            figma_score += 2
        if not has_circles:
            figma_score += 2
        
        return "Adobe Photoshop" if photoshop_score > figma_score else "Figma"
    
    # Default to the first technology if no specific disambiguation rules
    return tech1

# Functions removed: detect_circles, detect_gradient, detect_text_features, 
# combine_resolution_results, score_technologies, disambiguate_similar_techs, simplified_classification
# These were duplicates created during an edit operation

import fitz  # PyMuPDF

def analyze_cv_file(filepath):
    """
    Analyzes the uploaded CV using PyMuPDF to extract text and identify skills.
    :param filepath: Path to the uploaded CV file.
    :return: Extracted skills as a string.
    """
    # Extract text from the PDF
    text = ""
    with fitz.open(filepath) as pdf:
        for page in pdf:
            text += page.get_text()

    # Analyze the extracted text to identify skills
    skill_keywords = ["Python", "Machine Learning", "Data Analysis", "Excel", "SQL"]
    skills = [skill for skill in skill_keywords if skill in text]

    return ", ".join(skills)

def get_career_suggestions(result):
    """
    Provides career suggestions based on the identified technology
    """
    tech_career_map = {
        "Python": ["Data Scientist", "Machine Learning Engineer", "Backend Developer", "DevOps Engineer"],
        "JavaScript": ["Frontend Developer", "Full Stack Developer", "Web Application Developer"],
        "React": ["Frontend Developer", "UI Developer", "React Specialist"],
        "Angular": ["Frontend Developer", "Angular Specialist", "Enterprise Application Developer"],
        "Node.js": ["Backend Developer", "Full Stack Developer", "API Developer"],
        "Docker": ["DevOps Engineer", "Cloud Engineer", "System Administrator"],
        "MongoDB": ["Database Administrator", "Backend Developer", "NoSQL Specialist"],
        "MySQL": ["Database Administrator", "Backend Developer", "Data Analyst"],
        "Git": ["Software Developer", "DevOps Engineer", "Version Control Specialist"],
        "GitHub": ["Open Source Contributor", "Software Developer", "DevOps Engineer"],
        "Bootstrap": ["UI Developer", "Frontend Developer", "Web Designer"],
        "Adobe Photoshop": ["Graphic Designer", "UI Designer", "Digital Artist", "Marketing Specialist"],
        "Microsoft Excel": ["Data Analyst", "Financial Analyst", "Business Analyst", "Operations Manager"],
        "Excel": ["Data Analyst", "Financial Analyst", "Business Analyst", "Operations Manager"],
        "TensorFlow": ["Machine Learning Engineer", "AI Researcher", "Data Scientist"],
        "LinkedIn": ["Recruiter", "HR Specialist", "Social Media Manager"],
        "Twitter": ["Social Media Manager", "Digital Marketer", "Community Manager"],
        "YouTube": ["Content Creator", "Video Editor", "Digital Marketing Specialist"],
        "Spotify": ["Audio Engineer", "Music Producer", "Content Curator"],
        "Apple": ["iOS Developer", "macOS Developer", "UI/UX Designer"],
        "Linux": ["System Administrator", "DevOps Engineer", "Backend Developer"],
        "Android Studio": ["Android Developer", "Mobile App Developer", "UI Designer for Mobile"],
        "Vue.js": ["Frontend Developer", "UI Developer", "JavaScript Specialist"],
        "Heroku": ["Cloud Developer", "DevOps Engineer", "Full Stack Developer"],
        "Swift": ["iOS Developer", "Mobile App Developer", "Apple Ecosystem Specialist"],
        "npm": ["JavaScript Developer", "Frontend Developer", "Package Manager"],
        "Java": ["Enterprise Developer", "Android Developer", "Backend System Engineer"],
        "Cloud Computing": ["Cloud Architect", "DevOps Engineer", "Infrastructure Engineer"],
        "Unknown Technology": ["Technology Consultant", "IT Support Specialist", "Technical Trainer"]
    }
    
    # Get career suggestions for the identified technology
    if result in tech_career_map:
        return tech_career_map[result]
    else:
        # Default suggestions if no match
        return ["Software Developer", "IT Specialist", "Technology Consultant"]

# -------------------- Chatbot Routes --------------------
@main.route('/chat', methods=['GET', 'POST'])
def chat():
    reply = None
    if request.method == 'POST':
        user_input = request.form.get('message')
        if user_input:
            # Provide a fallback response if GROQ API key is not set
            if groq_client is None:
                print("DEBUG - Using fallback response (no Groq client)")
                raw_reply = "I'm a career advisor AI that can help with job recommendations, skill assessments, resume feedback, and industry trends. What would you like to know? (Note: Advanced AI responses are currently unavailable, please try again later.)"
            else:
                try:
                    print("DEBUG - Attempting to call Groq API")
                    response = groq_client.chat.completions.create(
                        model="mixtral-8x7b-32768",
                        messages=[
                            {"role": "system", "content": "You are a helpful career advisor assistant. Provide concise, practical career advice. Format your answers with proper paragraphs and bullet points for readability."},
                            {"role": "user", "content": user_input}
                        ]
                    )
                    raw_reply = response.choices[0].message.content
                    print("DEBUG - Groq API response received")
                except Exception as e:
                    print(f"DEBUG - GROQ API error: {str(e)}")
                    raw_reply = f"Based on your question about '{user_input}', I would suggest exploring career paths that align with your interests and skills. Consider obtaining relevant certifications and networking with professionals in your desired field. (Note: Our AI service is experiencing issues, this is a fallback response.)"
            
            # Format the response
            reply = format_chatbot_response(raw_reply)
            print(f"DEBUG - Chat reply prepared: {reply[:100]}...")
    
    # Pass current datetime and user info to the template
    now = datetime.now()
    return render_template('chat.html', reply=reply, now=now)

# -------------------- Quiz Routes --------------------

@main.route('/quiz', methods=['GET'])
def quiz():
    return render_template("quiz.html")

@main.route('/quiz-result', methods=['POST'])
def quiz_result():
    # Collect answers from the form
    answers = {
        'people': request.form.get('people'),
        'creative': request.form.get('creative'),
        'data': request.form.get('data'),
        'problem_solving': request.form.get('problem_solving'),
        'independent': request.form.get('independent'),
        'tech': request.form.get('tech')
    }

    # Match a role based on the answers
    role = match_role_based_on_personality(answers)
    
    # Store the result in database if user is logged in
    if current_user.is_authenticated:
        try:
            personality_result = PersonalityResult(
                user_id=current_user.id,
                result=role
            )
            db.session.add(personality_result)
            db.session.commit()
        except Exception as e:
            print(f"Error storing personality result: {str(e)}")

    # Render the result page with the matched role
    return render_template('quiz_result.html', role=role)

def match_role_based_on_personality(answers):
    """
    Matches a role based on the user's personality quiz answers.
    :param answers: A dictionary of answers from the quiz.
    :return: A string representing the matched role.
    """
    # Create a score-based approach for more nuanced matching
    tech_oriented = answers.get('tech') == 'yes'
    problem_solver = answers.get('problem_solving') == 'yes'
    independent_worker = answers.get('independent') == 'yes'
    people_oriented = answers.get('people') == 'yes'
    creative = answers.get('creative') == 'yes'
    data_oriented = answers.get('data') == 'yes'
    
    # Technical roles
    if tech_oriented and problem_solver:
        if data_oriented:
            if creative:
                return "Data Visualization Engineer"
            else:
                return "Data Scientist"
        elif creative:
            if people_oriented:
                return "UX/UI Designer"
            else:
                return "Frontend Developer"
        else:
            if independent_worker:
                return "Backend Developer"
            else:
                return "DevOps Engineer"
    
    # Business and management roles
    if people_oriented:
        if problem_solver:
            if data_oriented:
                return "Project Manager"
            else:
                return "Team Leader"
        elif creative:
            if tech_oriented:
                return "Digital Marketing Specialist"
            else:
                return "Marketing Manager"
        else:
            return "Human Resources Specialist"
    
    # Creative roles
    if creative:
        if tech_oriented:
            if independent_worker:
                return "Graphic Designer"
            else:
                return "Digital Content Creator"
        else:
            return "Creative Director"
    
    # Data-focused roles
    if data_oriented:
        if independent_worker:
            return "Data Analyst"
        else:
            return "Business Intelligence Analyst"
    
    # Default case: use original logic for backward compatibility
    if answers.get('people') == 'yes' and answers.get('creative') == 'yes':
        return "Marketing Specialist"
    elif answers.get('people') == 'yes' and answers.get('data') == 'yes':
        return "Project Manager"
    elif answers.get('creative') == 'yes' and answers.get('data') == 'yes':
        return "Data Visualization Expert"
    elif answers.get('people') == 'no' and answers.get('data') == 'yes':
        return "Data Analyst"
    elif answers.get('people') == 'no' and answers.get('creative') == 'yes':
        return "Graphic Designer"
    else:
        return "Versatile Professional"

# -------------------- Dashboard and Related Routes --------------------

@main.route('/dashboard')
@login_required
def dashboard():
    # Pass the current datetime to the template for personalized greeting
    now = datetime.now()
    return render_template('dashboard.html', now=now)

@main.route('/suggest', methods=['POST'])
def suggest():
    user_input = request.form['skills']
    suggestions = get_career_recommendation(user_input)
    # Pass the current datetime for personalized greeting
    now = datetime.now()
    return render_template("dashboard.html",
                           now=now,
                           suggestions=suggestions,
                           resume_feedback="You need more leadership roles listed.",
                           blog_post="<p>Your journey in AI begins here...</p>",
                           personality_match="UX Designer",
                           chatbot_reply="Yes! Switching to UX is great with your skills.",
                           image_suggestion="You might fit well into UI/UX Design.")

# -------------------- Authentication Routes --------------------

@main.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        # Check if the email already exists
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            flash('Email already registered. Please log in.', 'danger')
            return redirect(url_for('main.login'))

        # Hash the password and create a new user
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('main.login'))

    return render_template('register.html', form=form)

@main.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # Check if the user exists
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('main.dashboard'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html', form=form)

@main.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect('/')

# -------------------- Admin Routes --------------------

@main.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('Unauthorized access.', 'danger')
        return redirect('/')

    users = User.query.all()
    return render_template('admin.html', users=users)

@main.route('/make-premium/<int:user_id>', methods=['POST'])
@login_required
def make_premium(user_id):
    if not current_user.is_admin:
        flash('Unauthorized action.', 'danger')
        return redirect('/')

    user = User.query.get_or_404(user_id)
    user.is_premium = True
    db.session.commit()
    flash(f'{user.username} has been upgraded to Premium!', 'success')
    return redirect('/admin')

# -------------------- Profile Route --------------------

@main.route('/profile')
@login_required
def profile():
    career_results = CareerResult.query.filter_by(user_id=current_user.id).all()
    resume_feedbacks = ResumeFeedback.query.filter_by(user_id=current_user.id).all()
    personality_results = PersonalityResult.query.filter_by(user_id=current_user.id).all()

    return render_template('profile.html', 
                           user=current_user,
                           career_results=career_results,
                           resume_feedbacks=resume_feedbacks,
                           personality_results=personality_results)

# -------------------- Utility Functions --------------------

def premium_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_premium:
            flash('This feature is available only to premium users.', 'warning')
            return redirect(url_for('main.dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def analyze_resume(text):
    """
    Provides a more detailed and personalized resume analysis
    
    :param text: The extracted text from the resume
    :return: Dictionary with skills, recommendations, experience, and more
    """
    # Lists of skills to check for
    tech_skills = [
        "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "PHP", "Swift", "Go", "Kotlin",
        "SQL", "MySQL", "PostgreSQL", "MongoDB", "Oracle", "SQLite", "Redis", "Cassandra",
        "HTML", "CSS", "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring", "ASP.NET",
        "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Terraform", "Jenkins", "Git", "GitHub",
        "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-learn", "NLP",
        "Data Analysis", "Data Science", "Big Data", "Hadoop", "Spark", "Tableau", "Power BI",
        "Excel", "Word", "PowerPoint", "Photoshop", "Illustrator", "Figma", "Sketch", "UX/UI"
    ]
    
    soft_skills = [
        "Leadership", "Management", "Communication", "Teamwork", "Problem Solving",
        "Critical Thinking", "Creativity", "Time Management", "Adaptability", "Flexibility",
        "Organization", "Planning", "Decision Making", "Conflict Resolution", "Negotiation",
        "Presentation", "Customer Service", "Client Relations", "Mentoring", "Training",
        "Research", "Analysis", "Reporting", "Writing", "Editing", "Project Management",
        "Agile", "Scrum", "Kanban", "Lean", "Six Sigma", "Budget Management"
    ]
    
    # Find skills in the resume text
    found_tech_skills = [skill for skill in tech_skills if skill.lower() in text.lower() or re.search(r'\b' + re.escape(skill.lower()) + r'\b', text.lower())]
    found_soft_skills = [skill for skill in soft_skills if skill.lower() in text.lower() or re.search(r'\b' + re.escape(skill.lower()) + r'\b', text.lower())]
    
    # Look for education information
    education_pattern = re.compile(r'(?:degree|bachelor|master|phd|doctorate|bs|ba|ms|ma|mba|b\.s\.|m\.s\.|b\.a\.|m\.a\.|ph\.d\.)', re.IGNORECASE)
    has_education = bool(education_pattern.search(text))
    
    # Look for experience information
    exp_years_pattern = re.compile(r'(?:(\d+)(?:\+)?\s+years?(?:\s+of)?\s+experience)', re.IGNORECASE)
    exp_years_match = exp_years_pattern.search(text)
    experience_years = int(exp_years_match.group(1)) if exp_years_match else 0
    
    # Detect job titles
    job_titles = []
    common_titles = [
        "Software Engineer", "Developer", "Programmer", "Data Scientist", "Data Analyst",
        "Project Manager", "Product Manager", "UX Designer", "UI Designer", "DevOps Engineer",
        "System Administrator", "Network Engineer", "Database Administrator", "Business Analyst",
        "Marketing", "Sales", "Customer Service", "Support", "Specialist", "Coordinator", "Associate",
        "Manager", "Director", "VP", "CTO", "CEO", "CFO", "COO", "Architect", "Lead", "Senior"
    ]
    
    for title in common_titles:
        if re.search(r'\b' + re.escape(title.lower()) + r'\b', text.lower()):
            job_titles.append(title)
    
    # Generate personalized recommendations
    recommendations = []
    
    # Technical skills recommendations
    if len(found_tech_skills) < 5:
        recommendations.append("Consider adding more specific technical skills to your resume.")
    
    # Soft skills recommendations
    if len(found_soft_skills) < 3:
        recommendations.append("Adding soft skills like leadership, communication, or teamwork could strengthen your profile.")
    
    # Education recommendations
    if not has_education:
        recommendations.append("Including your education background would provide important context to your qualifications.")
    
    # Experience format recommendations
    if experience_years == 0:
        recommendations.append("Clearly state your years of experience to help employers understand your experience level.")
    
    # ATS recommendations
    recommendations.append("Ensure your resume is ATS-friendly by using standard section headings and avoiding complex formatting.")
    
    # Quantifiable achievements
    if not re.search(r'\d+%|\d+x', text):
        recommendations.append("Add quantifiable achievements (e.g., 'increased efficiency by 20%') to demonstrate your impact.")
    
    # Get seniority level based on skills and experience
    seniority = "Entry Level"
    if experience_years > 5 or len(found_tech_skills) > 10:
        seniority = "Senior Level"
    elif experience_years > 2 or len(found_tech_skills) > 5:
        seniority = "Mid Level"
    
    # Determine career path suggestions
    possible_paths = []
    if "Python" in found_tech_skills or "Data Analysis" in found_tech_skills or "Machine Learning" in found_tech_skills:
        possible_paths.append("Data Science")
    if "JavaScript" in found_tech_skills or "HTML" in found_tech_skills or "CSS" in found_tech_skills:
        possible_paths.append("Web Development")
    if "AWS" in found_tech_skills or "Docker" in found_tech_skills or "Kubernetes" in found_tech_skills:
        possible_paths.append("DevOps / Cloud Engineering")
    if "Project Management" in found_tech_skills or "Agile" in found_tech_skills or "Scrum" in found_tech_skills:
        possible_paths.append("Project Management")
    if "UX/UI" in found_tech_skills or "Figma" in found_tech_skills or "Design" in text.lower():
        possible_paths.append("UX/UI Design")
    
    # If no specific path is detected, suggest based on predominant skills
    if not possible_paths:
        if len(found_tech_skills) > len(found_soft_skills):
            possible_paths.append("Technical Role")
        else:
            possible_paths.append("Management / Leadership Role")
    
    return {
        "skills": found_tech_skills + found_soft_skills,
        "technical_skills": found_tech_skills,
        "soft_skills": found_soft_skills,
        "job_titles": job_titles,
        "experience_years": experience_years,
        "seniority": seniority,
        "career_paths": possible_paths,
        "has_education": has_education,
        "recommendation": " ".join(recommendations) if recommendations else "Your resume looks well-rounded!"
    }

# Function to find job postings based on skills
def find_job_postings(skills):
    """
    Searches for job postings based on skills extracted from the resume.
    
    :param skills: List of skills extracted from the resume
    :return: List of job posting dictionaries with title, company, location, and description
    """
    try:
        print(f"\n==== JOB POSTING FUNCTION START ====")
        print(f"DEBUG - find_job_postings() called with skills: {skills}")
        
        # Initialize job postings list
        job_postings = []
        
        # Always include at least one test job posting
        default_job = {
            'title': "Software Developer",
            'company': "CareerPath AI",
            'location': "Remote",
            'description': "We are looking for a talented developer to join our team. This role involves working with modern web technologies and helping professionals advance their careers.",
            'url': "https://www.linkedin.com/jobs/search/?keywords=Software%20Developer&location=Remote",
            'date_posted': datetime.now().strftime("%Y-%m-%d")
        }
        job_postings.append(default_job)
        
        # Ensure skills is a list
        if not isinstance(skills, list):
            print(f"DEBUG - Skills is not a list, converting: {type(skills)}")
            if isinstance(skills, str):
                skills = [skills]
            else:
                skills = ["Software Development", "Programming"]
                
        # Ensure we have at least one skill to work with
        if not skills or len(skills) == 0 or (len(skills) == 1 and skills[0].strip() == ""):
            print(f"DEBUG - No valid skills found, using default skills")
            skills = ["Software Development", "Programming", "Data Analysis"]
        
        print(f"DEBUG - Normalized skills: {skills}")
            
        # Company information with actual career page URLs
        company_data = {
            "Google": {
                "domains": ["Python", "Java", "C++", "Go", "Machine Learning", "AI", "Cloud", "DevOps"],
                "locations": ["Mountain View, CA", "New York, NY", "Seattle, WA", "Austin, TX"],
                "career_url": "https://careers.google.com/jobs/results/"
            },
            "Microsoft": {
                "domains": ["C#", ".NET", "Azure", "SQL", "Cloud", "DevOps", "AI"],
                "locations": ["Redmond, WA", "New York, NY", "Boston, MA"],
                "career_url": "https://careers.microsoft.com/us/en/search-results"
            },
            "Amazon": {
                "domains": ["Java", "Python", "AWS", "Cloud", "DevOps", "Logistics"],
                "locations": ["Seattle, WA", "New York, NY", "Arlington, VA"],
                "career_url": "https://www.amazon.jobs/en/teams/aws"
            },
            "Meta": {
                "domains": ["React", "JavaScript", "Python", "Data Science", "AR/VR"],
                "locations": ["Menlo Park, CA", "New York, NY", "Seattle, WA"],
                "career_url": "https://www.metacareers.com/"
            },
            "Apple": {
                "domains": ["Swift", "iOS", "macOS", "Hardware", "UX Design"],
                "locations": ["Cupertino, CA", "Austin, TX", "New York, NY"],
                "career_url": "https://jobs.apple.com/en-us/search"
            },
            "Netflix": {
                "domains": ["Java", "Python", "JavaScript", "Data Science", "Streaming"],
                "locations": ["Los Gatos, CA", "Los Angeles, CA", "New York, NY"],
                "career_url": "https://jobs.netflix.com/"
            },
            "IBM": {
                "domains": ["Java", "Python", "AI", "Cloud", "Quantum Computing"],
                "locations": ["Armonk, NY", "San Jose, CA", "Austin, TX"],
                "career_url": "https://www.ibm.com/careers/us-en/"
            },
            "Salesforce": {
                "domains": ["Apex", "JavaScript", "CRM", "Cloud", "Sales"],
                "locations": ["San Francisco, CA", "New York, NY", "Indianapolis, IN"],
                "career_url": "https://www.salesforce.com/company/careers/"
            }
        }
        
        # Job title formats
        title_prefixes = ["Senior ", "Lead ", "Staff ", "Principal ", ""]
        title_suffixes = ["Engineer", "Developer", "Architect", "Specialist"]
        
        # For each skill, generate a job posting based on matching companies
        for skill in skills[:3]:  # Limit to top 3 skills
            matched_companies = []
            
            # Find companies that match this skill
            for company, data in company_data.items():
                for domain in data["domains"]:
                    if skill.lower() in domain.lower() or domain.lower() in skill.lower():
                        matched_companies.append((company, data))
                        break
            
            # If no companies matched, use default companies
            if not matched_companies:
                # Try some common skill categories
                if any(tech in skill.lower() for tech in ["python", "java", "c++", "go", "ruby", "typescript"]):
                    matched_companies = [(c, company_data[c]) for c in ["Google", "Microsoft", "Amazon"] if c in company_data]
                elif any(tech in skill.lower() for tech in ["data", "analytics", "statistics", "sql"]):
                    matched_companies = [(c, company_data[c]) for c in ["Microsoft", "IBM", "Google"] if c in company_data]
                elif any(tech in skill.lower() for tech in ["cloud", "aws", "azure", "devops"]):
                    matched_companies = [(c, company_data[c]) for c in ["Amazon", "Microsoft", "Google"] if c in company_data]
                else:
                    # Default to random companies
                    company_names = list(company_data.keys())
                    if company_names:
                        random_company = company_names[hash(skill + str(datetime.now().minute)) % len(company_names)]
                        matched_companies = [(random_company, company_data[random_company])]
            
            # If we still have no matches, skip this skill
            if not matched_companies:
                continue
                
            # Select a company from matches
            company_name, company_info = matched_companies[hash(skill + str(datetime.now().second)) % len(matched_companies)]
            
            # Create job title
            title_prefix = title_prefixes[hash(skill + company_name) % len(title_prefixes)]
            title_suffix = title_suffixes[hash(skill + str(datetime.now().minute)) % len(title_suffixes)]
            
            # Get location
            location = company_info["locations"][hash(company_name + str(datetime.now().second)) % len(company_info["locations"])]
            
            # Create job posting
            job_posting = {
                'title': f"{title_prefix}{skill} {title_suffix}",
                'company': company_name,
                'location': location,
                'description': f"We're looking for an experienced {skill} {title_suffix.lower()} to join our team. You'll be working on exciting projects using cutting-edge technology. Requirements include {skill} experience and strong problem-solving skills.",
                'url': company_info["career_url"],
                'date_posted': datetime.now().strftime("%Y-%m-%d")
            }
            
            # Add to list
            job_postings.append(job_posting)
            print(f"DEBUG - Added job: {job_posting['title']} at {job_posting['company']}")
        
        # Add more fallback jobs if we didn't get enough
        if len(job_postings) < 3:
            print(f"DEBUG - Not enough jobs generated, adding fallbacks")
            fallback_jobs = [
                {
                    'title': "Data Analyst",
                    'company': "LinkedIn",
                    'location': "New York, NY",
                    'description': "Analyzing business data and creating reports for decision makers.",
                    'url': "https://www.linkedin.com/jobs/search/?keywords=Data%20Analyst&location=New%20York",
                    'date_posted': datetime.now().strftime("%Y-%m-%d")
                },
                {
                    'title': "Project Manager",
                    'company': "Indeed",
                    'location': "Chicago, IL",
                    'description': "Managing technical projects from initiation to completion.",
                    'url': "https://www.indeed.com/jobs?q=Project+Manager&l=Chicago",
                    'date_posted': datetime.now().strftime("%Y-%m-%d")
                }
            ]
            job_postings.extend(fallback_jobs[:4 - len(job_postings)])
        
        # Print summary
        print(f"DEBUG - Generated {len(job_postings)} job postings")
        print(f"==== JOB POSTING FUNCTION END ====\n")
        
        # Return the job postings (up to 4)
        return job_postings[:4]
        
    except Exception as e:
        print(f"ERROR: Exception in find_job_postings: {str(e)}")
        # Return fallback job listings if there's an error
        return [
            {
                "title": "Software Developer", 
                "company": "Tech Company", 
                "location": "Remote", 
                "description": "Looking for software developers with various skills and experience levels.", 
                "url": "https://www.linkedin.com/jobs/search/?keywords=Software%20Developer&location=Remote",
                "date_posted": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "title": "Data Analyst", 
                "company": "Analytics Firm", 
                "location": "New York, NY", 
                "description": "Analyzing business data and creating reports for decision makers.", 
                "url": "https://www.indeed.com/jobs?q=Data+Analyst&l=New+York",
                "date_posted": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "title": "Project Manager", 
                "company": "Solutions Inc.", 
                "location": "Chicago, IL", 
                "description": "Managing technical projects from initiation to completion.", 
                "url": "https://www.glassdoor.com/Job/project-manager-jobs-SRCH_KO0,15.htm",
                "date_posted": datetime.now().strftime("%Y-%m-%d")
            }
        ]

# -------------------- Test Routes --------------------

@main.route('/test-jobs')
def test_jobs():
    """A test route to verify job postings display properly"""
    # Generate some example skills
    test_skills = ["Python", "Java", "Machine Learning", "Cloud Computing"]
    # Get job postings based on these skills
    job_postings = find_job_postings(test_skills)
    return render_template('jobs_test.html', job_postings=job_postings)

@main.route('/standalone-test')
def standalone_test():
    """A test route with a standalone template"""
    # Generate some example skills
    test_skills = ["Data Science", "DevOps", "JavaScript", "React"]
    # Get job postings based on these skills
    job_postings = find_job_postings(test_skills)
    return render_template('standalone_jobs_test.html', job_postings=job_postings)

@main.route('/test-logo-classification', methods=['GET', 'POST'])
def test_logo_classification():
    """
    Test route for logo classification
    """
    result = None
    debug_info = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            
            # Ensure uploads directory exists
            os.makedirs('uploads', exist_ok=True)
            file.save(filepath)
            
            try:
                # Get classification result
                result = classify_tool_or_skill(filepath)
                
                # Get debug information
                debug_info = {
                    'filename': filename,
                    'file_size': os.path.getsize(filepath),
                    'image_dimensions': f"{width}x{height}",
                    'color_info': {
                        'r_mean': round(r_mean, 2),
                        'g_mean': round(g_mean, 2),
                        'b_mean': round(b_mean, 2),
                        'brightness': round(brightness, 2)
                    },
                    'shape_info': {
                        'is_square': is_square,
                        'aspect_ratio': round(aspect_ratio, 2),
                        'has_circles': has_circles,
                        'has_text': has_text
                    },
                    'feature_scores': {
                        'edge_intensity': round(edge_intensity, 2),
                        'color_variance': round(color_variance, 2)
                    }
                }
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'danger')
            finally:
                # Clean up uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            flash('Invalid file type. Please upload an image.', 'danger')
            return redirect(request.url)
    
    return render_template('logo_test.html', result=result, debug_info=debug_info)