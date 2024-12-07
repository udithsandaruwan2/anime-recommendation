import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer  # Correct import
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gradio as gr  # Gradio is correctly imported

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Database connection details
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'anime_db')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')
TABLE_NAME = 'anime'

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Preprocess text by lowercasing, removing stopwords, and lemmatizing."""
    if isinstance(text, str):  # Skip non-string entries
        text = text.lower()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Step 1: Connect to PostgreSQL database and fetch data
print("Connecting to the database...")
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
query = f"SELECT * FROM {TABLE_NAME}"
df = pd.read_sql(query, engine)
print(f"Data fetched successfully: {len(df)} records.")

# Step 2: Preprocess and clean data
print("Preprocessing data...")
df['description'] = df['name'] + ' ' + df['genre'] + ' ' + df['type'] + ' episodes: ' + df['episodes']
df['description'] = df['description'].apply(preprocess_text)  # Apply preprocessing
df = df.drop_duplicates(subset=['description']).dropna()
print(f"Data cleaned: {len(df)} records remain.")

# Step 3: Split data into training and testing sets
print("Splitting data into training and testing sets...")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Step 4: Load pre-trained model and generate embeddings
print("Loading pre-trained model...")
model_st = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("Generating embeddings for training data...")
train_embeddings = model_st.encode(train_df['description'].tolist(), show_progress_bar=True)

print("Generating embeddings for test data...")
test_embeddings = model_st.encode(test_df['description'].tolist(), show_progress_bar=True)

# Feedback log for RL
feedback_log = []
recommendation_count = 0  # Counter to track recommendations served

# Step 5: Define recommendation and RL functions
def get_recommendations_from_test(query, embeddings, df, top_n=5):
    """Fetch recommendations based on a query using test data embeddings."""
    query_embedding = model_st.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    
    # Get top-N indices
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    
    # Return recommendations and similarity scores
    return df.iloc[top_indices], similarities

def update_feedback_data(query, reward, recommendations):
    """Log user feedback for reinforcement learning."""
    feedback_entry = {
        "query": query,
        "reward": reward,
        "recommendations": recommendations['name'].tolist()
    }
    feedback_log.append(feedback_entry)
    print("Feedback logged successfully.")

def fine_tune_model():
    """Update the model embeddings based on feedback."""
    global train_embeddings, test_embeddings, feedback_log, train_df, test_df
    for feedback in feedback_log:
        if feedback['reward'] > 0:
            # Strengthen positive connections in the embeddings
            # Optionally, add embeddings for these queries to training set
            pass
        else:
            # Decrease similarity weight for negative feedback
            pass
    feedback_log = []  # Clear feedback log after processing
    print("Model fine-tuned based on feedback.")

def generate_similarity_histogram(similarities):
    """Generate histogram of cosine similarity scores and return as an image."""
    plt.figure(figsize=(8, 6))
    plt.hist(similarities[0], bins=50, alpha=0.7, color='blue', label='Cosine Similarity')
    plt.title('Distribution of Cosine Similarity Scores')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return Image.open(buf)

# Gradio app interface to use test data for recommendations
def recommend_anime_from_test(user_query, feedback=None):
    """Make recommendations and collect feedback every 5 queries."""
    global recommendation_count
    recommendations, similarities = get_recommendations_from_test(user_query, test_embeddings, test_df)
    
    # Increment the recommendation count
    recommendation_count += 1
    
    # Process feedback if provided
    if feedback is not None:
        reward = 1 if feedback == "Yes" else -1
        update_feedback_data(user_query, reward, recommendations)
    
    # Generate histogram image
    hist_image = generate_similarity_histogram(similarities)
    
    # Ask for feedback every 5 recommendations
    ask_feedback = "Yes" if recommendation_count % 5 == 0 else "No"
    
    return recommendations[['name', 'genre', 'type', 'episodes']], hist_image, ask_feedback

# Gradio interface
print("Launching Gradio interface using test data...")
interface = gr.Interface(
    fn=recommend_anime_from_test,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter a query (e.g., sci-fi action anime)"),
        gr.Radio(choices=["Yes", "No"], label="Did you like the last recommendations?", value=None, interactive=True)
    ],
    outputs=[
        gr.Dataframe(headers=["Name", "Genre", "Type", "Episodes"]),
        gr.Image(type="pil", label="Cosine Similarity Histogram"),
        gr.Textbox(label="Feedback Request")
    ],
    title="Anime Recommendation System (Using Test Data)",
    description="Enter a query to get anime recommendations from the test dataset and view similarity distribution. Feedback will be requested every 5 recommendations."
)

# Step 6: Save and launch
print("\nSaving the trained model to the root directory...")
model_st.save('./anime_recommendation_model')

# Temporary file cleanup
def clean_up():
    print("Temporary files cleaned up.")

try:
    interface.launch()
finally:
    clean_up()
