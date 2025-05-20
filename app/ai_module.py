# ai_module.py

def get_career_recommendation(user_input):
    # More comprehensive career recommendations based on skills
    career_paths = []
    
    # Handle both string and list inputs
    skills = user_input
    if isinstance(user_input, str):
        skills = [user_input]
    
    skill_to_career_map = {
        "Python": ["Data Scientist", "Machine Learning Engineer", "Backend Developer", "AI Researcher"],
        "JavaScript": ["Frontend Developer", "Full Stack Developer", "Web Developer", "React Developer"],
        "HTML": ["Frontend Developer", "Web Designer", "UI Developer"],
        "CSS": ["Frontend Developer", "UI Designer", "Web Designer"],
        "SQL": ["Database Administrator", "Data Analyst", "Business Intelligence Analyst"],
        "Machine Learning": ["ML Engineer", "AI Researcher", "Data Scientist", "Computational Linguist"],
        "Data Analysis": ["Data Analyst", "Business Analyst", "Market Research Analyst", "Financial Analyst"],
        "AWS": ["Cloud Engineer", "DevOps Engineer", "Solutions Architect", "SRE"],
        "Docker": ["DevOps Engineer", "Cloud Engineer", "Platform Engineer"],
        "Kubernetes": ["Platform Engineer", "DevOps Engineer", "Site Reliability Engineer"],
        "Java": ["Backend Developer", "Android Developer", "Enterprise Architect"],
        "C#": [".NET Developer", "Game Developer", "Windows Application Developer"],
        "C++": ["Game Developer", "Systems Programmer", "Embedded Systems Engineer"],
        "React": ["Frontend Developer", "UI Developer", "React Developer"],
        "Angular": ["Frontend Developer", "UI Developer", "Angular Specialist"],
        "Node.js": ["Backend Developer", "Full Stack Developer", "API Developer"]
    }
    
    # Get career paths for each skill
    for skill in skills:
        if isinstance(skill, str) and skill in skill_to_career_map:
            career_paths.extend(skill_to_career_map[skill])
    
    # Deduplicate and limit to top 5
    unique_paths = list(set(career_paths))
    
    # If no specific paths found, return default suggestions
    if not unique_paths:
        return ["Data Analyst", "Software Developer", "Project Manager", "UX Designer", "Digital Marketer"]
    
    return unique_paths[:5]

def analyze_resume(resume_text):
    """
    Returns an analysis of the resume text with skills and recommendations
    """
    # Define common skills to look for
    all_skills = [
        "Python", "JavaScript", "Java", "C++", "C#", "SQL", "HTML", "CSS",
        "React", "Angular", "Vue.js", "Node.js", "Express", "Django", "Flask",
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git", "GitHub", 
        "Machine Learning", "Data Analysis", "Excel", "PowerPoint", "Word",
        "Communication", "Leadership", "Project Management", "Agile", "Scrum"
    ]
    
    # Find skills in the resume text
    found_skills = []
    for skill in all_skills:
        if skill.lower() in resume_text.lower():
            found_skills.append(skill)
    
    # Ensure we always have at least some skills for matching
    if not found_skills:
        found_skills = ["Python", "JavaScript", "Communication"]
    
    print(f"DEBUG: Found skills in resume: {found_skills}")
    
    # Generate a recommendation based on found skills
    recommendation = "Consider adding more specific technical skills to your resume."
    if len(found_skills) > 5:
        recommendation = "Your resume contains a good range of skills. Consider adding quantifiable achievements."
    elif len(found_skills) > 2:
        recommendation = "Including proficiency in common office tools could be beneficial"
    
    return {
        "skills": found_skills,
        "recommendation": recommendation
    }

def ai_chat_response(user_question):
    # Placeholder chatbot logic
    return "That's a great question! Let's explore some options together."

def summarize_text(text):
    return "This is a summary of your text."

def classify_uploaded_image(image_path):
    return "Suggested Career: Graphic Designer"

def generate_blog_post(goals):
    # Placeholder for Groq LPU / GPT-style API
    return f"This is a career blog based on your goals: {goals}"

def match_role_based_on_personality(answers):
    if answers["creative"] == "yes" and answers["people"] == "yes":
        return "UX Designer"
    elif answers["data"] == "yes" and answers["people"] == "no":
        return "Data Analyst"
    else:
        return "Product Manager"

def ai_chat_response(user_question):
    # Placeholder: Later integrate Groq LPU
    return f"I'm here to help! Based on your question '{user_question}', consider exploring data science roles."

def classify_uploaded_image(image_path):
    # Placeholder for AI model
    return "Suggested Career: Graphic Designer"

